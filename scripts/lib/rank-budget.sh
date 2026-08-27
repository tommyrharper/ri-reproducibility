# Clamp a run's rank count to the memory the host can actually give it.
#
# Rank count, not NS_NLIVE, is what costs memory: every rank keeps one warm
# worker holding its own copy of the imaging stack. Measured on a 20-CPU,
# 62GB host, holding NS_NLIVE fixed at 12 and varying only the rank count:
# 4 ranks 13.5GB, 12 ranks 40.6GB - 3.4GB per R2D2 rank, dead linear. So
# NS_NLIVE can be raised for search quality without touching memory, and
# NS_MPI_PROCS is the knob that has to fit in RAM.
#
# Nothing here used to check that. NS_MPI_PROCS defaulted to
# `min(NS_NLIVE, host CPUs)`, which on this host is 20 ranks - 68GB, more
# than the box has. The failure is silent rather than loud: the host OOM
# killer takes an imaging worker, the rank's `readline()` on the reply FIFO
# returns empty, and common.py records the evaluation with FAILURE_OBJECTIVE
# (100.0). PolyChord maximizes the objective, and a real `total_rms_jy` is
# ~0.008, so an OOM kill scores as the best point the search has ever seen
# and it concentrates live points there. A run that ran out of memory
# reports "R2D2 fails catastrophically in this corner of the parameter
# space", which is exactly the conclusion this repo exists to draw. Worse,
# the dead worker is dropped from the cache, so the next evaluation starts a
# fresh one - another ~3.4GB - while already out of memory.
#
# Several agent sessions share this host and each run sizes itself from
# `nproc` alone, so the common way to hit that wall is two sessions starting
# runs, not one oversized run. MemAvailable covers a run that starts while
# another is already warm. It does not cover two runs sizing themselves in
# the same second, before either has allocated anything, so a run also
# reserves what it is about to take.
#
# Reservations are files named by the reserving PID, holding "<expiry> <MB>".
# A reader skips entries whose PID is gone or whose expiry has passed, so
# there is no release path to get wrong - a run killed with SIGKILL leaves an
# entry that the next reader prunes. They expire because a reservation only
# has to cover the gap between deciding a rank count and the workers actually
# allocating; after that MemAvailable is the truth and still counting the
# reservation would double-count it.
#
# Source this, then:
#
#   ranks="$(ns_budget_ranks <requested> <MB per rank> <label>)"
#
# Everything is overridable for the self-check: NS_RANK_BUDGET_DIR moves the
# reservation directory, NS_AVAILABLE_MB replaces the /proc/meminfo read.
#
# ponytail: a fixed MB-per-rank measured on one host, not a live measurement.
# It only has to be right to within a rank or two; re-measure it if the
# imaging stack's resident set changes materially.

# Resident set of one rank's warm worker, measured on this repo's images by
# running the same NS_NLIVE at different rank counts and dividing out the
# difference: R2D2 3.39GB (13.5GB at 4 ranks, 40.6GB at 12), WSClean 0.17GB
# (1.3GB at 8 ranks, almost all of it the rank's simulate worker rather than
# `wsclean -j 1` itself, which peaks at 50MB). Rounded up, because being a
# rank short costs wall clock and being a rank over costs the run.
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
  # macOS has no MemAvailable. Free + inactive + speculative pages is the
  # same "reclaimable without swapping" idea vm_stat can give us; page size
  # comes from vm_stat's own header so this isn't hardcoding 4096/16384.
  if command -v vm_stat >/dev/null 2>&1; then
    vm_stat | awk -v pagesize="$(vm_stat | sed -n 's/.*page size of \([0-9]*\) bytes.*/\1/p')" '
      /^Pages free:/ { free = $3 }
      /^Pages inactive:/ { inactive = $3 }
      /^Pages speculative:/ { spec = $3 }
      END {
        if (pagesize == "") { exit 1 }
        gsub(/\./, "", free); gsub(/\./, "", inactive); gsub(/\./, "", spec)
        print int((free + inactive + spec) * pagesize / 1024 / 1024)
      }'
    return
  fi
  # Neither read is available: returning nothing here is what turns the
  # clamp off.
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

# Echoes the rank count to use. Never more than requested, never less than 1.
ns_budget_ranks() {
  local requested="$1" mb_per_rank="$2" label="$3"
  local dir="${NS_RANK_BUDGET_DIR:-${TMPDIR:-/tmp}/ri-ns-rank-budget-$(id -u)}"
  local available reserved=0 budget affordable now entry pid expiry mb

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
        "Wait for the other runs to finish, or free memory. If nothing is" \
        "running, check for sidecars a killed run left behind:" \
        "docker ps --filter name=ri-ns-sidecar" >&2
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
  _ns_available_mb >/dev/null 2>&1 || return 0
  available="$(_ns_available_mb)"
  if [ "$((requested * mb_per_rank))" -gt "$((available - NS_RANK_BUDGET_HEADROOM_MB))" ]; then
    echo "WARNING: ${label} was asked for ${requested} ranks," \
      "~$((requested * mb_per_rank))MB, with only ${available}MB available." \
      "Out-of-memory evaluations are recorded as failures and score as the" \
      "search's best points - see scripts/lib/rank-budget.sh." >&2
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

  # The real reader works on whatever platform runs this check - /proc/meminfo
  # on Linux, `vm_stat` on macOS - not just the NS_AVAILABLE_MB override below.
  [ "$(_ns_available_mb)" -gt 0 ]

  export NS_AVAILABLE_MB=40960
  # 40960 available - 4096 headroom = 36864, over 3400 per rank, is 10 ranks.
  # Each case starts from an empty directory: ns_budget_ranks runs in the
  # command substitution's subshell, where $$ is still this shell, so every
  # call leaves a reservation of its own behind.
  clear_reservations() { rm -f "${NS_RANK_BUDGET_DIR}"/[0-9]*; }

  # Asking for less than the budget affords is left alone.
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  # ...and reserves what it is about to take, for whoever reads next.
  [ "$(awk '{print $2}' "${NS_RANK_BUDGET_DIR}/$$")" = 27200 ]
  # Asking again gives the same answer: a run does not count its own
  # reservation against itself.
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  clear_reservations

  # Asking for more is clamped to what fits, not granted.
  [ "$(ns_budget_ranks 20 3400 r2d2 2>/dev/null)" = 10 ]
  clear_reservations

  # Not even one rank fits: refuses, rather than sampling the OOM killer.
  ! NS_AVAILABLE_MB=5000 ns_budget_ranks 8 3400 r2d2 >/dev/null 2>&1
  clear_reservations

  # Another run's live reservation comes out of this one's budget:
  # 36864 - 27200 leaves 9664, which affords 2 ranks of 3400.
  printf '%s %s\n' "$(($(date +%s) + 60))" 27200 >"${NS_RANK_BUDGET_DIR}/${PPID}"
  [ "$(ns_budget_ranks 8 3400 r2d2 2>/dev/null)" = 2 ]
  clear_reservations

  # A reservation whose owner is gone is pruned, not counted - which is what
  # makes a SIGKILLed run self-healing instead of poisoning the host.
  printf '%s %s\n' "$(($(date +%s) + 60))" 27200 >"${NS_RANK_BUDGET_DIR}/999999"
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  [ ! -f "${NS_RANK_BUDGET_DIR}/999999" ]
  clear_reservations

  # So is a live owner's expired one, so the warm-up window cannot leak into
  # a later run's budget.
  printf '%s %s\n' "$(($(date +%s) - 1))" 27200 >"${NS_RANK_BUDGET_DIR}/${PPID}"
  [ "$(ns_budget_ranks 8 3400 r2d2)" = 8 ]
  [ ! -f "${NS_RANK_BUDGET_DIR}/${PPID}" ]
  clear_reservations

  # An explicit rank count is obeyed, and warned about when it will not fit.
  ns_budget_warn_if_over 8 3400 r2d2 2>/dev/null
  [ -n "$(ns_budget_warn_if_over 20 3400 r2d2 2>&1)" ]
  [ -z "$(ns_budget_warn_if_over 2 3400 r2d2 2>&1)" ]

  rm -rf "${NS_RANK_BUDGET_DIR}"
  echo "rank-budget self-check passed"
fi
