# shellcheck shell=bash  # sourced, so no shebang
# Record what a run was actually started with, so it can be resumed exactly.
#
# Written at startup rather than derived afterwards, because the settings a
# resume needs are precisely the ones an interrupted run never got to write
# down: summary.json is written after PolyChord returns, so a run that stopped
# has none. Without this, resuming means remembering the flags by hand and
# getting a different search if you remember wrong.
#
# The values are the resolved ones, after defaults.toml and after the memory
# clamp in rank-budget.sh, so a run that was clamped to seven ranks resumes on
# seven and not on the eight it originally asked for. PolyChord's resume file
# describes a live-point set that a different rank count would not match.
#
# A KEY=VALUE file rather than JSON so that resuming is `set -a; . run.env`
# with no parser: `%q` quoting makes it safe to source even when NS_METRIC is
# an expression with spaces in it.
#
#   write_run_config <output-dir> <algorithm>

write_run_config() {
  local output_dir="$1" algorithm="$2"
  {
    printf 'NS_ALGORITHM=%q\n' "${algorithm}"
    printf 'NS_NLIVE=%q\n' "${NS_NLIVE}"
    printf 'NS_NUM_REPEATS=%q\n' "${NS_NUM_REPEATS}"
    printf 'NS_MAX_NDEAD=%q\n' "${NS_MAX_NDEAD}"
    printf 'NS_SEED=%q\n' "${NS_SEED}"
    printf 'NS_METRIC=%q\n' "${NS_METRIC}"
    printf 'NS_MPI_PROCS=%q\n' "${NS_MPI_PROCS}"
    printf 'NS_RETRIES=%q\n' "${NS_RETRIES}"
    if [ -n "${R2D2_OMP_THREADS:-}" ]; then
      printf 'R2D2_OMP_THREADS=%q\n' "${R2D2_OMP_THREADS}"
    fi
  } >"${output_dir}/run.env"
}

# Claim a run directory named for the moment the run started, and create it.
#
# `mkdir -p` on a name built from a whole-second UTC stamp is not a claim: two
# searches started in the same second - two agent sessions sharing this host,
# or one script launching a pair - land on the same directory and then write
# each other's evaluations, FIFOs and summary.json, and the first to finish
# deletes the FIFO directory the other is still reading. Seen for real: the
# second run died with `mkfifo: ... No such file or directory` while the first
# reported success.
#
# A bare `mkdir` is the claim, because it fails when the name is taken. The
# loser then waits for the next second rather than decorating its name, so
# every run directory keeps the `<algorithm>-vlaa-<stamp>` shape that
# `./ri runs`, the health report and its `started_at` ordering all rely on.
# Bounded, so an unwritable parent is an error rather than a spin.
#
#   ns_claim_run_dir <parent> <prefix>   - prints the directory it created

ns_claim_run_dir() {
  local parent="$1" prefix="$2" stamp="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}" dir
  # RUN_ID names the run when it is set, which is what lets the self-check
  # below force the collision this exists to survive.
  mkdir -p "${parent}"
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    dir="${parent}/${prefix}${stamp}"
    if mkdir "${dir}" 2>/dev/null; then
      printf '%s\n' "${dir}"
      return 0
    fi
    sleep 1
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  done
  echo "FATAL: could not create a run directory under ${parent}" >&2
  return 1
}

# The other way two jobs end up in one run directory: not a stamp collision but
# a `--output-dir` naming a run that is still going. `./ri resume` has always
# refused that, and `./ri search` has always allowed it - measured, the second
# search deleted the live run's FIFO directory, recreated it with its own rank
# count, and wrote its own chains/*.resume over the live checkpoint, while the
# first run was still imaging. `mkdir -p` cannot see any of this; only the
# host's process list can.
#
# Refuses rather than claiming a different name, because a caller who named the
# directory meant that directory - and the run they want to join is the one
# already in it.
#
#   ns_refuse_live_run <output-dir>

ns_refuse_live_run() {
  ns_run_is_live "$1" || return 0
  echo "FATAL: ${1##*/} is still running, so it is not a directory to start into." >&2
  echo "       A second job over the same checkpoint and FIFOs would corrupt both." >&2
  echo "       Watch it instead:  ./ri health ${1##*/}" >&2
  exit 1
}

# `bash scripts/lib/run-config.sh --self-check` - that what is written can be
# sourced back to the same values, including a metric that needs quoting.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  _dir="$(mktemp -d)"
  NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 NS_SEED=41 NS_RETRIES=2 \
    NS_METRIC='total_rms_jy - 0.5 * snr' NS_MPI_PROCS=7 R2D2_OMP_THREADS=2 \
    write_run_config "${_dir}" r2d2
  # A subshell, so the sourced values cannot leak into the checks below it.
  (
    set -a
    # shellcheck disable=SC1091
    . "${_dir}/run.env"
    set +a
    [ "${NS_ALGORITHM}" = r2d2 ]
    [ "${NS_MPI_PROCS}" = 7 ]
    [ "${NS_RETRIES}" = 2 ]
    [ "${NS_METRIC}" = 'total_rms_jy - 0.5 * snr' ]
    [ "${R2D2_OMP_THREADS}" = 2 ]
  )
  # WSClean has no thread setting, and must not write an empty one.
  NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 NS_SEED=41 NS_RETRIES=0 \
    NS_METRIC=total_rms_jy NS_MPI_PROCS=8 R2D2_OMP_THREADS='' \
    write_run_config "${_dir}" wsclean
  grep -q R2D2_OMP_THREADS "${_dir}/run.env" && {
    echo "FAIL: empty R2D2_OMP_THREADS written for wsclean"; exit 1
  }

  # A run directory is claimed, not shared: the second caller in the same
  # second must not be handed the first one's directory.
  _parent="${_dir}/runs"
  _first="$(RUN_ID=20260101T000000Z ns_claim_run_dir "${_parent}" wsclean-vlaa-)"
  [ "${_first}" = "${_parent}/wsclean-vlaa-20260101T000000Z" ]
  [ -d "${_first}" ]
  # Same stamp again: it waits for the next second rather than returning the
  # taken name, so the two runs never share a directory.
  _second="$(RUN_ID=20260101T000000Z ns_claim_run_dir "${_parent}" wsclean-vlaa-)"
  [ "${_second}" != "${_first}" ]
  [ -d "${_second}" ]
  # ...and it still ends in a stamp, which is what orders the report.
  case "${_second}" in
    "${_parent}/wsclean-vlaa-"[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) ;;
    *) echo "FAIL: claimed run directory is not stamp-named: ${_second}"; exit 1 ;;
  esac

  # A `--output-dir` naming a live run is refused, not started into. A real
  # process with the real command line, rather than a faked argv: this is what
  # pgrep has to see, spelled the way a rank spells it.
  # shellcheck source=scripts/lib/progress-bar.sh
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/progress-bar.sh"
  _live="${_dir}/wsclean-vlaa-20260101T000001Z"
  mkdir -p "${_live}"
  # Nothing is running yet, so the guard must let the directory through - and
  # `exit 1` inside it would take this check down with it.
  ( ns_refuse_live_run "${_live}" ) || { echo "FAIL: nothing live, so the guard must not fire"; exit 1; }
  printf 'import time\ntime.sleep(30)\n' > "${_dir}/polychord_wsclean.py"
  python3 "${_dir}/polychord_wsclean.py" --output-dir "${_live}" --nlive 50 &
  _fake=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ns_run_is_live "${_live}" && break
    sleep 0.2
  done
  ns_run_is_live "${_live}" || { echo "FAIL: a live rank on this run must be seen"; kill "${_fake}"; exit 1; }
  # A neighbouring run whose name this one is a prefix of must not be caught.
  ( ns_refuse_live_run "${_live%Z}" ) || { echo "FAIL: a prefix of the run directory must not match"; kill "${_fake}"; exit 1; }
  if _out="$( ns_refuse_live_run "${_live}" 2>&1 )"; then
    echo "FAIL: starting into a live run must be refused, got: ${_out}"; kill "${_fake}"; exit 1
  fi
  case "${_out}" in
    *"still running"*) ;;
    *) echo "FAIL: the refusal must say why, got: ${_out}"; kill "${_fake}"; exit 1 ;;
  esac
  kill "${_fake}" 2>/dev/null || true
  wait "${_fake}" 2>/dev/null || true

  rm -rf "${_dir}"
  echo "run-config self-check passed"
fi
