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
  rm -rf "${_dir}"
  echo "run-config self-check passed"
fi
