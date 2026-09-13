# shellcheck shell=bash  # sourced, so no shebang
# Clamp rank count to memory; reservations prevent concurrent overcommit.
# Source this, then call `ns_budget_ranks <requested> <MB per rank> <label>`.
# NS_RANK_BUDGET_DIR and NS_AVAILABLE_MB override state for checks.
# ponytail: a fixed MB-per-rank measured on one host, not a live measurement.
# Re-measure after a material change to the imaging stack's resident set.

# Rounded-up warm-worker RSS estimates from this repo's images.
NS_R2D2_MB_PER_RANK="${NS_R2D2_MB_PER_RANK:-3500}"
NS_WSCLEAN_MB_PER_RANK="${NS_WSCLEAN_MB_PER_RANK:-200}"

# Left free for the OS, the page cache and the non-worker parts of a run.
NS_RANK_BUDGET_HEADROOM_MB="${NS_RANK_BUDGET_HEADROOM_MB:-4096}"
# How long a reservation counts for. Long enough for containers to start,
# torch to import and the first evaluations to reach a steady resident set;
# short enough that a later run is sized from MemAvailable instead.
NS_RANK_BUDGET_RESERVE_SECONDS="${NS_RANK_BUDGET_RESERVE_SECONDS:-60}"

_ns_available_mb() {
  if [ -n "${NS_AVAILABLE_MB:-}" ]; then
    printf '%s\n' "${NS_AVAILABLE_MB}"
    return 0
  fi
  # Inside a Slurm job the cgroup limit is what the OOM killer enforces, and
  # MemAvailable on a shared node says nothing about it. Slurm spells the limit
  # per node (--mem) or per core (the partition default); `--mem 0` is the
  # whole node, which MemAvailable below then describes truthfully.
  if [ -n "${SLURM_MEM_PER_NODE:-}" ] && [ "${SLURM_MEM_PER_NODE}" != 0 ]; then
    printf '%s\n' "${SLURM_MEM_PER_NODE}"
    return 0
  fi
  if [ -n "${SLURM_MEM_PER_CPU:-}" ]; then
    printf '%s\n' "$((SLURM_MEM_PER_CPU * ${SLURM_CPUS_ON_NODE:-$(nproc)}))"
    return 0
  fi
  if [ -r /proc/meminfo ]; then
    awk '/^MemAvailable:/ { print int($2 / 1024); found = 1 } END { exit !found }' /proc/meminfo
    return
  fi
  # No memory source means no clamping.
  return 1
}

# The whole read-decide-reserve in ns_budget_ranks has to be atomic against
# another run doing the same thing. flock(1) isn't stock on macOS, so fall
# back to an mkdir spinlock - mkdir is atomic on any POSIX filesystem. A
# stale lock (holder SIGKILLed) is pruned by checking the PID it recorded,
# same as a stale reservation below.
_ns_lock() {
  local dir="$1" lockdir="${1}/.lock.d" holder
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${dir}/.lock"
    flock 9
    return
  fi
  while ! mkdir "${lockdir}" 2>/dev/null; do
    holder="$(cat "${lockdir}/pid" 2>/dev/null || true)"
    if [ -n "${holder}" ] && ! kill -0 "${holder}" 2>/dev/null; then
      rm -rf "${lockdir}"
      continue
    fi
    sleep 0.05
  done
  echo "$$" >"${lockdir}/pid"
}

_ns_unlock() {
  local dir="$1"
  if command -v flock >/dev/null 2>&1; then
    exec 9>&-
  else
    rm -rf "${dir}/.lock.d"
  fi
}

# Match host-visible ranks and `mpirun` processes by their
# `--output-dir`; anchor the path so run-name prefixes do not match. Sidecars
# use `--fifo-dir` and are excluded because they outlive killed runs.
ns_run_process_pattern() {
  printf 'polychord_[a-z0-9_]*\.py .*--output-dir %s( |$)' "$1"
}

# On a cluster the ranks are on a compute node whose processes the login node
# cannot see, so a run is also live while a Slurm job named after it (slurm.sh)
# is queued or running - other than the job asking, which is the run itself.
_ns_slurm_job_live() {
  command -v squeue >/dev/null 2>&1 || return 1
  squeue -h -u "${USER:-$(id -un)}" -n "${1##*/}" -o %i 2>/dev/null \
    | awk -v me="${SLURM_JOB_ID:-}" '$1 != me' | grep -q .
}

# Whether a job drives `$1`. Check both path spellings because callers may use
# a symlink; a not-yet-created output directory cannot be live.
ns_run_is_live() {
  local dir="$1" real
  _ns_slurm_job_live "${dir}" && return 0
  pgrep -f "$(ns_run_process_pattern "${dir}")" >/dev/null 2>&1 && return 0
  [ -d "${dir}" ] || return 1
  real="$(cd "${dir}" && pwd -P)"
  [ "${real}" = "${dir}" ] && return 1
  pgrep -f "$(ns_run_process_pattern "${real}")" >/dev/null 2>&1
}

# A run killed with SIGKILL leaves its worker pools (start-sidecars.sh)
# running, each R2D2 one holding ~3.4GB of warm worker per rank that nothing
# will ever free, so the next run is sized against - or refused for - memory
# a dead run is sitting on. Slurm's job cgroup makes this moot on the cluster;
# on a shared box it is the same debris as a stale reservation above.
#
# A pool is leaked when its run has no ranks *and* the shell that started it
# is gone. Neither alone is enough: the ranks are dead between retries while
# the launcher waits to start them again, and the launcher can be SIGKILLed
# while the ranks it started carry on (they are mpirun's children, not its).
# Pid reuse only ever makes this skip a pool, never take a live one, which is
# the direction to be wrong in.
#
# Reads `<pgid><TAB><launcher pid><TAB><run dir>` per pool so the rule can be
# checked without any pool running.
_ns_dead_pools() {
  local pgid launcher run_dir
  while IFS=$'\t' read -r pgid launcher run_dir; do
    case "${pgid}" in
      '' | *[!0-9]*) continue ;;
    esac
    if [ -n "${run_dir}" ] && ns_run_is_live "${run_dir}"; then
      continue
    fi
    case "${launcher}" in
      '' | *[!0-9]*) printf '%s\n' "${pgid}"; continue ;;
    esac
    kill -0 "${launcher}" 2>/dev/null || printf '%s\n' "${pgid}"
  done
}

# Every pool on the host: the pool's shell names its FIFO directory, which is
# `<run dir>/.<worker>-workers`, as its last argument, and start-sidecars.sh
# leaves the launcher's pid in `<run dir>/.launcher.pid`.
_ns_pool_table() {
  local pid pgid args fifo_dir launcher
  for pid in $(pgrep -f -- '/\.[a-z0-9]*-workers$' 2>/dev/null); do
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    args="$(ps -ww -o args= -p "${pid}" 2>/dev/null)"
    [ -n "${pgid}" ] && [ -n "${args}" ] || continue
    fifo_dir="${args##* }"
    launcher="$(cat "${fifo_dir%/*}/.launcher.pid" 2>/dev/null || true)"
    printf '%s\t%s\t%s\n' "${pgid}" "${launcher}" "${fifo_dir%/*}"
  done | sort -u
}

ns_reap_leaked_sidecars() {
  local dead pgid
  dead="$(_ns_pool_table | _ns_dead_pools | sort -u)"
  [ -n "${dead}" ] || return 0
  # Said out loud: this is another run's wreckage being removed.
  # shellcheck disable=SC2086  # pgids cannot contain whitespace
  echo "NOTE: killing worker pool(s) left behind by a run that is gone," \
    "which were holding memory against this run: process group(s)" ${dead} >&2
  for pgid in ${dead}; do
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  done
}

# Echoes the rank count to use. Never more than requested, never less than 1.
ns_budget_ranks() {
  local requested="$1" mb_per_rank="$2" label="$3"
  local dir="${NS_RANK_BUDGET_DIR:-${TMPDIR:-/tmp}/ri-ns-rank-budget-$(id -u)}"
  local available reserved=0 budget affordable now entry pid expiry mb

  # Before the read, so the memory a dead run is still holding is counted as
  # free rather than clamping this run down to fit around it.
  ns_reap_leaked_sidecars

  # No memory reading (neither /proc/meminfo nor vm_stat, i.e. a platform
  # this hasn't been taught) means no clamp: the guard is a safety net on
  # the hosts that can support it, not a hard dependency.
  if ! available="$(_ns_available_mb)"; then
    printf '%s\n' "${requested}"
    return 0
  fi

  mkdir -p "${dir}"
  _ns_lock "${dir}"

  now="$(date +%s)"
  for entry in "${dir}"/*; do
    [ -f "${entry}" ] || continue
    pid="${entry##*/}"
    if [ "${pid}" = ".lock" ]; then
      continue
    fi
    # Not our own: this run's reservation is replaced below, and counting it
    # here would have a second call shrink the run on the strength of what
    # its first call already set aside.
    if [ "${pid}" = "$$" ]; then
      continue
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${entry}"
      continue
    fi
    read -r expiry mb <"${entry}" || continue
    if [ "${now}" -ge "${expiry}" ]; then
      rm -f "${entry}"
      continue
    fi
    reserved=$((reserved + mb))
  done

  budget=$((available - reserved - NS_RANK_BUDGET_HEADROOM_MB))
  affordable=$((budget / mb_per_rank))

  if [ "${affordable}" -lt "${requested}" ]; then
    if [ "${affordable}" -lt 1 ]; then
      # One rank does not fit, so there is nothing to run that would not be
      # sampling the OOM killer instead of the algorithm.
      echo "FATAL: not enough free memory for a single ${label} rank:" \
        "${available}MB available, ${reserved}MB reserved by other runs," \
        "${NS_RANK_BUDGET_HEADROOM_MB}MB headroom, ${mb_per_rank}MB needed per rank." \
        "Sidecars left by a dead run were already removed, so this is memory" \
        "something live is holding: wait for the other runs to finish," \
        "or see ./ri health." >&2
      _ns_unlock "${dir}"
      return 1
    fi
    # Said out loud: a run that quietly used fewer cores than asked for would
    # be its own surprise, and this is the line that explains a slow run.
    echo "NOTE: ${label} ranks ${requested} -> ${affordable}" \
      "(${available}MB available, ${reserved}MB reserved by other runs," \
      "${mb_per_rank}MB per rank)" >&2
    requested="${affordable}"
  fi

  printf '%s %s\n' "$((now + NS_RANK_BUDGET_RESERVE_SECONDS))" \
    "$((requested * mb_per_rank))" >"${dir}/$$"
  _ns_unlock "${dir}"

  printf '%s\n' "${requested}"
}

# Warn, but obey: an explicit NS_MPI_PROCS is the caller saying they know
# better than the guard, and that is theirs to decide.
ns_budget_warn_if_over() {
  local requested="$1" mb_per_rank="$2" label="$3"
  local available
  ns_reap_leaked_sidecars
  _ns_available_mb >/dev/null 2>&1 || return 0
  available="$(_ns_available_mb)"
  if [ "$((requested * mb_per_rank))" -gt "$((available - NS_RANK_BUDGET_HEADROOM_MB))" ]; then
    echo "WARNING: ${label} was asked for ${requested} ranks," \
      "~$((requested * mb_per_rank))MB, with only ${available}MB available." \
      "An evaluation the OOM killer takes is never scored, so this costs the" \
      "run rather than the result: the attempt is retried against a fresh" \
      "worker and then the run stops, to be picked up with ./ri resume" \
      "- see docs/robustness.md." >&2
  fi
}

# `bash scripts/lib/rank-budget.sh --self-check` - the arithmetic, the
# clamp, the refusal, and that a reservation is both seen by the next caller
# and ignored once its owner is gone.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  NS_RANK_BUDGET_DIR="$(mktemp -d)"
  export NS_RANK_BUDGET_DIR
  NS_RANK_BUDGET_HEADROOM_MB=4096

  if _ns_self_check_mb="$(_ns_available_mb)"; then
    [ "${_ns_self_check_mb}" -gt 0 ]
  fi

  # Slurm's three spellings of the job's memory.
  [ "$(NS_AVAILABLE_MB='' SLURM_MEM_PER_NODE=8000 _ns_available_mb)" = 8000 ]
  [ "$(NS_AVAILABLE_MB='' SLURM_MEM_PER_NODE='' SLURM_MEM_PER_CPU=3420 SLURM_CPUS_ON_NODE=4 _ns_available_mb)" = 13680 ]
  _whole_node="$(NS_AVAILABLE_MB='' SLURM_MEM_PER_NODE=0 SLURM_MEM_PER_CPU='' _ns_available_mb || true)"
  [ "${_whole_node:-1}" != 0 ]

  export NS_AVAILABLE_MB=40960
  clear_reservations() { rm -f "${NS_RANK_BUDGET_DIR}"/[0-9]*; }

  ns_reap_leaked_sidecars() { :; }

  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  [ "$(awk '{print $2}' "${NS_RANK_BUDGET_DIR}/$$")" = 27200 ]
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  clear_reservations

  [ "$(ns_budget_ranks 20 3400 r2d2 2>/dev/null)" = 10 ]
  clear_reservations

  NS_AVAILABLE_MB=5000 ns_budget_ranks 8 3400 r2d2 >/dev/null 2>&1 && {
    echo "FAIL: 8 ranks of 3400MB granted against 5000MB free"; exit 1
  }
  clear_reservations

  printf '%s %s\n' "$(($(date +%s) + 60))" 27200 >"${NS_RANK_BUDGET_DIR}/${PPID}"
  [ "$(ns_budget_ranks 8 3400 r2d2 2>/dev/null)" = 2 ]
  clear_reservations

  printf '%s %s\n' "$(($(date +%s) + 60))" 27200 >"${NS_RANK_BUDGET_DIR}/999999"
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  [ ! -f "${NS_RANK_BUDGET_DIR}/999999" ]
  clear_reservations

  printf '%s %s\n' "$(($(date +%s) - 1))" 27200 >"${NS_RANK_BUDGET_DIR}/${PPID}"
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  [ ! -f "${NS_RANK_BUDGET_DIR}/${PPID}" ]
  clear_reservations

  [ "$(printf '4242\t999999\t\n4343\t%s\t\n' "$$" | _ns_dead_pools)" = 4242 ]
  [ "$(printf '4242\t\t\n' | _ns_dead_pools)" = 4242 ]
  [ -z "$(printf 'notapgid\t999999\t\nsomething-else\n' | _ns_dead_pools)" ]
  [ -z "$(printf '' | _ns_dead_pools)" ]

  # The orphaned run: launcher pid gone, run directory that still has ranks.
  # Reaping these is what would kill the search - so a live run wins over the
  # pid. A real process with the real command line, because this is what pgrep
  # has to see, spelled the way a rank spells it.
  _orphan_dir="$(mktemp -d)"
  _orphan_run="${_orphan_dir}/wsclean-vlaa-20260101T000001Z"
  mkdir -p "${_orphan_run}"
  [ "$(printf '4242\t999999\t%s\n' "${_orphan_run}" | _ns_dead_pools)" = 4242 ]
  printf 'import time\ntime.sleep(30)\n' >"${_orphan_dir}/polychord_wsclean.py"
  python3 "${_orphan_dir}/polychord_wsclean.py" --output-dir "${_orphan_run}" --nlive 50 &
  _orphan_pid=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ns_run_is_live "${_orphan_run}" && break
    sleep 0.2
  done
  ns_run_is_live "${_orphan_run}" \
    || { echo "FAIL: a live rank on this run must be seen"; kill "${_orphan_pid}"; exit 1; }
  [ -z "$(printf '4242\t999999\t%s\n' "${_orphan_run}" | _ns_dead_pools)" ] \
    || { echo "FAIL: a live run's pool was offered up for killing"
         kill "${_orphan_pid}"; exit 1; }
  [ "$(printf '4242\t999999\t%s\n' "${_orphan_run}-other" | _ns_dead_pools)" = 4242 ] \
    || { echo "FAIL: a dead run's pool must still be reaped"
         kill "${_orphan_pid}"; exit 1; }
  kill "${_orphan_pid}" 2>/dev/null || true
  wait "${_orphan_pid}" 2>/dev/null || true

  # A queued or running Slurm job named after the run makes it live from the
  # login node, where pgrep sees nothing; the job itself is not its own rival.
  mkdir -p "${_orphan_dir}/bin"
  printf '#!/bin/sh\ncase "$*" in *"-n wsclean-vlaa-20260101T000001Z "*) echo 4242 ;; esac\n' \
    >"${_orphan_dir}/bin/squeue"
  chmod +x "${_orphan_dir}/bin/squeue"
  PATH="${_orphan_dir}/bin:${PATH}" ns_run_is_live "${_orphan_run}" \
    || { echo "FAIL: a Slurm job named after the run must make it live"; exit 1; }
  PATH="${_orphan_dir}/bin:${PATH}" ns_run_is_live "${_orphan_run}-other" \
    && { echo "FAIL: another run's job must not make this one live"; exit 1; }
  PATH="${_orphan_dir}/bin:${PATH}" SLURM_JOB_ID=4242 ns_run_is_live "${_orphan_run}" \
    && { echo "FAIL: the run's own job must not count as a rival"; exit 1; }
  rm -rf "${_orphan_dir}"

  ns_budget_warn_if_over 8 3400 r2d2 2>/dev/null
  [ -n "$(ns_budget_warn_if_over 20 3400 r2d2 2>&1)" ]
  [ -z "$(ns_budget_warn_if_over 2 3400 r2d2 2>&1)" ]

  rm -rf "${NS_RANK_BUDGET_DIR}"
  echo "rank-budget self-check passed"
fi
