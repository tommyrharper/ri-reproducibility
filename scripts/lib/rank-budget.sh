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
  if [ -r /proc/meminfo ]; then
    awk '/^MemAvailable:/ { print int($2 / 1024); found = 1 } END { exit !found }' /proc/meminfo
    return
  fi
  # macOS runs containers in Docker's Linux VM, so use the daemon's memory
  # minus current container usage rather than host memory.
  if command -v docker >/dev/null 2>&1; then
    _ns_docker_available_mb
    return
  fi
  # No memory source means no clamping.
  return 1
}

# `docker stats --format '{{.MemUsage}}'` gives a human string per container
# ("3.6GiB / 46.95GiB"), not raw bytes, so summing usage across containers
# means converting each one - pulled out so the self-check can exercise the
# unit conversion without a live daemon.
_ns_mem_string_to_mb() {
  awk '
    function to_mb(v,   n) {
      n = v + 0
      if (v ~ /TiB$/) return n * 1024 * 1024
      if (v ~ /GiB$/) return n * 1024
      if (v ~ /MiB$/) return n
      if (v ~ /KiB$/) return n / 1024
      return n / 1024 / 1024  # bare bytes
    }
    { sum += to_mb($1) }
    END { printf "%d\n", sum }
  '
}

_ns_docker_available_mb() {
  local total_bytes used_mb
  total_bytes="$(docker info --format '{{.MemTotal}}' 2>/dev/null)"
  [ -n "${total_bytes}" ] && [ "${total_bytes}" -gt 0 ] 2>/dev/null || return 1
  used_mb="$(docker stats --no-stream --format '{{.MemUsage}}' 2>/dev/null \
    | awk -F' / ' '{ print $1 }' | _ns_mem_string_to_mb)"
  [ -n "${used_mb}" ] || used_mb=0
  printf '%d\n' "$(( total_bytes / 1024 / 1024 - used_mb ))"
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

# Match host-visible ranks, `mpirun` and `docker exec` processes by their
# `--output-dir`; anchor the path so run-name prefixes do not match. Sidecars
# use `--fifo-dir` and are excluded because they outlive killed runs.
ns_run_process_pattern() {
  printf 'polychord_[a-z0-9_]*\.py .*--output-dir %s( |$)' "$1"
}

# Whether a job drives `$1`. Check both path spellings because callers may use
# a symlink; a not-yet-created output directory cannot be live.
ns_run_is_live() {
  local dir="$1" real
  pgrep -f "$(ns_run_process_pattern "${dir}")" >/dev/null 2>&1 && return 0
  [ -d "${dir}" ] || return 1
  real="$(cd "${dir}" && pwd -P)"
  [ "${real}" = "${dir}" ] && return 1
  pgrep -f "$(ns_run_process_pattern "${real}")" >/dev/null 2>&1
}

# A run killed with SIGKILL leaves its `ri-ns-sidecar-*` containers running,
# each holding ~3.4GB of warm imaging worker that nothing will ever free. That
# is the same shape of debris as a stale reservation above, and the rule starts
# the same way: the launcher's pid is in the container name, so a name whose
# pid is gone is a candidate. Until now `./ri health` only named them and
# the FATAL below only suggested looking, which meant the next run was sized
# against - or refused for - memory a dead run was sitting on.
#
# The pid alone is not enough, which is what the `ri.run-dir` label is for. A
# run script killed with SIGKILL leaves the run itself going - the ranks are
# children of containerd-shim, not of the shell - so its containers have a dead
# launcher pid and a live search inside them. Reaping those kills the search,
# and `./ri health` was handing out the `docker rm -f` line for them. So a
# container whose labelled run still has processes is never dead, whatever its
# pid says; the pid rule is the fallback for a container started before the
# label existed, and for common.py's per-rank fallback containers, which have
# no run directory to name.
#
# The label exempts by run, so a container genuinely leaked by an earlier
# attempt at a run that is live again would be exempted too. It cannot survive
# to be: this function runs from ns_budget_ranks, before the new attempt's
# ranks exist, so the run is not live at the moment the question is asked.
#
# Reads `<name><TAB><run dir>` so the rule can be checked without a daemon.
# Names are `ri-ns-sidecar-<launcher pid>-<n>` (start-sidecars.sh) and
# `ri-ns-sidecar-<rank pid>-<uuid8>` (common.py's fallback); the pid is in the
# same position in both. Pid reuse only ever makes this skip a container, never
# take a live one, which is the direction to be wrong in.
_ns_dead_sidecar_names() {
  local line name run_dir pid
  while IFS= read -r line; do
    name="${line%%$'\t'*}"
    run_dir=""
    [ "${line}" = "${name}" ] || run_dir="${line#*$'\t'}"
    pid="${name#ri-ns-sidecar-}"
    pid="${pid%%-*}"
    case "${pid}" in
      '' | *[!0-9]*) continue ;;
    esac
    if [ -n "${run_dir}" ] && ns_run_is_live "${run_dir}"; then
      continue
    fi
    kill -0 "${pid}" 2>/dev/null || printf '%s\n' "${name}"
  done
}

ns_reap_leaked_sidecars() {
  local dead
  command -v docker >/dev/null 2>&1 || return 0
  dead="$(docker ps --filter name=ri-ns-sidecar \
      --format '{{.Names}}\t{{.Label "ri.run-dir"}}' 2>/dev/null \
    | _ns_dead_sidecar_names)"
  [ -n "${dead}" ] || return 0
  # Said out loud: this is another run's wreckage being removed, and a silent
  # `docker rm --force` is not something to do on someone else's host.
  # shellcheck disable=SC2086  # container names cannot contain whitespace
  echo "NOTE: removing sidecar container(s) left behind by a run that is gone," \
    "which were holding memory against this run:" ${dead} >&2
  # shellcheck disable=SC2086  # container names cannot contain whitespace
  docker rm --force ${dead} >/dev/null 2>&1 || true
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

  [ "$(printf '3.5GiB\n512MiB\n' | _ns_mem_string_to_mb)" = 4096 ]
  [ "$(printf '1024KiB\n' | _ns_mem_string_to_mb)" = 1 ]

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

  [ "$(printf 'ri-ns-sidecar-999999-0\nri-ns-sidecar-%s-1\n' "$$" | _ns_dead_sidecar_names)" \
    = "ri-ns-sidecar-999999-0" ]
  [ "$(printf 'ri-ns-sidecar-999999-a1b2c3d4\n' | _ns_dead_sidecar_names)" \
    = "ri-ns-sidecar-999999-a1b2c3d4" ]
  [ -z "$(printf 'ri-ns-sidecar-notapid-0\nsomething-else\n' | _ns_dead_sidecar_names)" ]
  [ -z "$(printf '' | _ns_dead_sidecar_names)" ]

  # The orphaned run: launcher pid gone, `ri.run-dir` label naming a run that
  # still has ranks. Reaping these is what killed the search - so the label
  # wins over the pid. A real process with the real command line, because this
  # is what pgrep has to see, spelled the way a rank spells it.
  _orphan_dir="$(mktemp -d)"
  _orphan_run="${_orphan_dir}/wsclean-vlaa-20260101T000001Z"
  mkdir -p "${_orphan_run}"
  [ "$(printf 'ri-ns-sidecar-999999-0\t%s\n' "${_orphan_run}" | _ns_dead_sidecar_names)" \
    = "ri-ns-sidecar-999999-0" ]
  printf 'import time\ntime.sleep(30)\n' >"${_orphan_dir}/polychord_wsclean.py"
  python3 "${_orphan_dir}/polychord_wsclean.py" --output-dir "${_orphan_run}" --nlive 50 &
  _orphan_pid=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ns_run_is_live "${_orphan_run}" && break
    sleep 0.2
  done
  ns_run_is_live "${_orphan_run}" \
    || { echo "FAIL: a live rank on this run must be seen"; kill "${_orphan_pid}"; exit 1; }
  [ -z "$(printf 'ri-ns-sidecar-999999-0\t%s\n' "${_orphan_run}" | _ns_dead_sidecar_names)" ] \
    || { echo "FAIL: a live run's sidecar was offered up for removal"
         kill "${_orphan_pid}"; exit 1; }
  [ "$(printf 'ri-ns-sidecar-999999-0\t%s\n' "${_orphan_run}-other" | _ns_dead_sidecar_names)" \
    = "ri-ns-sidecar-999999-0" ] \
    || { echo "FAIL: a dead run's labelled sidecar must still be reaped"
         kill "${_orphan_pid}"; exit 1; }
  kill "${_orphan_pid}" 2>/dev/null || true
  wait "${_orphan_pid}" 2>/dev/null || true
  rm -rf "${_orphan_dir}"

  ns_budget_warn_if_over 8 3400 r2d2 2>/dev/null
  [ -n "$(ns_budget_warn_if_over 20 3400 r2d2 2>&1)" ]
  [ -z "$(ns_budget_warn_if_over 2 3400 r2d2 2>&1)" ]

  rm -rf "${NS_RANK_BUDGET_DIR}"
  echo "rank-budget self-check passed"
fi
