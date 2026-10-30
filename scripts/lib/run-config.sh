# shellcheck shell=bash  # sourced, so no shebang
# Save resolved run settings in a source-safe KEY=VALUE file for resume.

# The SIF's build-input label (ns_image_id, scripts/lib/defaults.sh); the
# self-check runs without defaults.sh and records unknown.
_run_image_id() {
  if declare -F ns_image_id >/dev/null; then
    ns_image_id "${1:-}"
  else
    printf '%s\n' unknown
  fi
}

write_run_config() {
  local output_dir="$1" algorithm="$2"
  local meqtrees_image_id polychord_image_id imager_image_id
  meqtrees_image_id="$(_run_image_id "${MEQTREES_SIF:-}")"
  polychord_image_id="$(_run_image_id "${POLYCHORD_SIF:-}")"
  if [ "${algorithm}" = wsclean ]; then
    imager_image_id="$(_run_image_id "${WSCLEAN_SIF:-}")"
  else
    imager_image_id="$(_run_image_id "${R2D2_SIF:-}")"
  fi
  {
    printf 'NS_ALGORITHM=%q\n' "${algorithm}"
    printf 'NS_NLIVE=%q\n' "${NS_NLIVE}"
    printf 'NS_NUM_REPEATS=%q\n' "${NS_NUM_REPEATS}"
    printf 'NS_MAX_NDEAD=%q\n' "${NS_MAX_NDEAD}"
    printf 'NS_SEED=%q\n' "${NS_SEED}"
    printf 'NS_METRIC=%q\n' "${NS_METRIC}"
    printf 'NS_MPI_PROCS=%q\n' "${NS_MPI_PROCS}"
    printf 'NS_IMAGE_DIM=%q\n' "${NS_IMAGE_DIM}"
    printf 'NS_RETRIES=%q\n' "${NS_RETRIES}"
    # Persist run settings, including watchdog behavior, for exact resume.
    printf 'NS_STALL_TIMEOUT=%q\n' "${NS_STALL_TIMEOUT}"
    printf 'NS_SYNCHRONOUS=%q\n' "${NS_SYNCHRONOUS}"
    printf 'NS_KEEP_MEASUREMENT_SETS=%q\n' "${NS_KEEP_MEASUREMENT_SETS}"
    # Opt-in contour adoption (round 312); a resume keeps the same sampler.
    if [ -n "${NS_ADOPT_CONTOUR:-}" ]; then
      printf 'NS_ADOPT_CONTOUR=%q\n' "${NS_ADOPT_CONTOUR}"
    fi
    if [ -n "${NS_EARLY_STOP:-}" ]; then
      printf 'NS_EARLY_STOP=%q\n' "${NS_EARLY_STOP}"
    fi
    printf 'NS_IMAGER_IMAGE_ID=%q\n' "${imager_image_id}"
    printf 'NS_MEQTREES_IMAGE_ID=%q\n' "${meqtrees_image_id}"
    printf 'NS_POLYCHORD_IMAGE_ID=%q\n' "${polychord_image_id}"
    if [ "${algorithm}" = wsclean ]; then
      printf 'NS_WSCLEAN_MGAIN=%q\n' "${NS_WSCLEAN_MGAIN}"
      printf 'NS_WSCLEAN_NITER=%q\n' "${NS_WSCLEAN_NITER}"
      printf 'NS_WSCLEAN_LOG_TIME=%q\n' "${NS_WSCLEAN_LOG_TIME:-1}"
      if [ -n "${WSCLEAN_TARGET_CPU:-}" ]; then
        printf 'WSCLEAN_TARGET_CPU=%q\n' "${WSCLEAN_TARGET_CPU}"
      fi
    fi
    # Without it `./ri resume` brought a GPU run back on the CPU.
    if [ "${algorithm}" = r2d2 ]; then
      printf 'R2D2_DEVICE=%q\n' "${R2D2_DEVICE:-cpu}"
      printf 'R2D2_RUNTIME=%q\n' "${R2D2_RUNTIME:-torch}"
      # A multi-node search (round 287) resumes on as many nodes.
      if [ "${NS_NODES:-1}" -gt 1 ]; then
        printf 'NS_NODES=%q\n' "${NS_NODES}"
      fi
      if [ -n "${R2D2_TRT_SHARED:-}" ]; then
        printf 'R2D2_TRT_SHARED=%q\n' "${R2D2_TRT_SHARED}"
      fi
      # Opt-in batching (round 224); a resume keeps its plans and server.
      if [ -n "${R2D2_TRT_BATCH:-}" ]; then
        printf 'R2D2_TRT_BATCH=%q\n' "${R2D2_TRT_BATCH}"
      fi
      if [ -n "${R2D2_TRT_LANES:-}" ]; then
        printf 'R2D2_TRT_LANES=%q\n' "${R2D2_TRT_LANES}"
      fi
    fi
    if [ -n "${R2D2_OMP_THREADS:-}" ]; then
      printf 'R2D2_OMP_THREADS=%q\n' "${R2D2_OMP_THREADS}"
    fi
    if [ -n "${R2D2_INTEROP_THREADS:-}" ]; then
      printf 'R2D2_INTEROP_THREADS=%q\n' "${R2D2_INTEROP_THREADS}"
    fi
    if [ -n "${R2D2_MAX_THREADS:-}" ]; then
      printf 'R2D2_MAX_THREADS=%q\n' "${R2D2_MAX_THREADS}"
    fi
    if [ -n "${R2D2_INTERLEAVE_WEIGHTS:-}" ]; then
      printf 'R2D2_INTERLEAVE_WEIGHTS=%q\n' "${R2D2_INTERLEAVE_WEIGHTS}"
    fi
    if [ -n "${R2D2_WEIGHTS_PER_NODE:-}" ]; then
      printf 'R2D2_WEIGHTS_PER_NODE=%q\n' "${R2D2_WEIGHTS_PER_NODE}"
    fi
    if [ -n "${R2D2_CUDA_BLOCKING_SYNC:-}" ]; then
      printf 'R2D2_CUDA_BLOCKING_SYNC=%q\n' "${R2D2_CUDA_BLOCKING_SYNC}"
    fi
  } >"${output_dir}/run.env"
}

# Claim a unique `<algorithm>-vlaa-<stamp>` directory. Bare `mkdir` prevents
# same-second searches from sharing evaluations, FIFOs or checkpoints; retry
# is bounded so an unwritable parent fails instead of spinning.
#
#   ns_claim_run_dir <parent> <prefix>   - prints created directory

ns_claim_run_dir() {
  local parent="$1" prefix="$2" stamp="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}" dir
  # RUN_ID lets the self-check force a same-stamp collision.
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

# A run directory the containers cannot see is not a run directory. Every
# container is started with one bind mount, `--bind ${REPO_ROOT}`
# (scripts/lib/start-sidecars.sh) - Apptainer adds $HOME and /tmp, but hpc-work
# is neither - so a `--output-dir` outside the repo exists on the host, which
# is where run.env and the FIFOs land, and separately, emptily, inside each
# container, which is where PolyChord's chains and the evaluation directories
# land. Measured on a real `--output-dir /tmp/...`
# search: the ranks lost their warm worker pool, fell back to rank-started
# workers, and evaluation 1 died with `FileNotFoundError: .../evaluations/
# eval-0001-*/simulate.stdout.log` - two minutes of container startup spent to
# fail on a path that could have been checked in front of it.
#
# The test is on the path as spelled, because the mount is the path as spelled:
# the container sees a path iff it is under REPO_ROOT, whatever either side
# resolves to. Callers pass an absolute path with `..` already collapsed.
#
#   ns_refuse_unmounted_run <output-dir>

# Refuse missing checkpoints before claiming a run directory: otherwise every
# evaluation scores FAILURE_OBJECTIVE, which PolyChord maximizes.
#
#   ns_refuse_missing_checkpoints <checkpoints-dir> <set-name>

ns_refuse_missing_checkpoints() {
  local dir="$1/$2" candidate
  if [ -d "${dir}" ]; then
    for candidate in "${dir}"/*.ckpt; do
      [ -e "${candidate}" ] && return 0
    done
  fi
  echo "FATAL: no R2D2 checkpoints in ${dir}" >&2
  echo "       Without them every evaluation fails, and a failed evaluation scores" >&2
  echo "       FAILURE_OBJECTIVE, which PolyChord maximizes - so the search would not" >&2
  echo "       stop, it would report the broken imager as its best discovery." >&2
  echo "       Get them with:  ./ri fetch-checkpoints" >&2
  echo "       Extract so that ${dir}/R2D2_UNet_N<k>.ckpt exists." >&2
  echo "       Set CHECKPOINTS_DIR to look somewhere else - a worktree does not" >&2
  echo "       share the checkpoints of the checkout it was made from." >&2
  exit 1
}

ns_refuse_unmounted_run() {
  case "$1/" in
    "${REPO_ROOT}/"*) return 0 ;;
  esac
  # `mkdir -p` created it a moment ago only so the path could be resolved, so
  # unmake it - `-p` for the parents that came with it. rmdir stops at the
  # first directory with anything in it, so a run directory that already
  # existed, and every parent that is not ours, survives.
  rmdir -p "$1" 2>/dev/null || true
  echo "FATAL: $1" >&2
  echo "       is outside the repository, which is the only directory the run's" >&2
  echo "       containers can see, so the chains, evaluations and imager logs" >&2
  echo "       written there would stay inside the containers and be lost." >&2
  echo "       Repository:  ${REPO_ROOT}" >&2
  echo "       Use a directory under it - the default is results/nested-sampling/." >&2
  exit 1
}

# Stripped as it goes, an evaluation leaves ~1.1 files (its eval-*.json
# record, and a dir for the few it keeps: 8,562 files for 7,821 evaluations,
# docs/csd3-speed.md round 92; ~2.1 before), and a converged run makes ~9 x
# nlive dead points at ~3.4 x num_repeats evaluations each (round 91). So ~4
# files per repeat per dead point; a run sized past the free file quota dies on EDQUOT
# part way through. Warns, never refuses: the estimate is rough, and only
# Lustre's project quota (CSD3's hpc-work) is read.
#   ns_warn_file_quota <run dir> <nlive> <num repeats> <max ndead>
ns_warn_file_quota() {
  local project ndead need free
  command -v "${LFS:-lfs}" >/dev/null 2>&1 || return 0
  # `|| true`: off Lustre (a /dev/shm tree) lfs fails, and under the run
  # scripts' `set -e -o pipefail` that failure ended the search (round 149).
  project="$("${LFS:-lfs}" project -d "$1" 2>/dev/null | awk '{ print $1 }' || true)"
  [ "${project:-0}" -gt 0 ] 2>/dev/null || return 0
  ndead="$4"
  [ "${ndead}" -gt 0 ] 2>/dev/null || ndead=$((9 * $2))
  need=$((4 * $3 * ndead))
  # Header, then: filesystem kbytes quota limit grace files quota limit grace.
  free="$("${LFS:-lfs}" quota -p "${project}" "$1" 2>/dev/null \
    | awk 'NR > 2 && $8 + 0 > 0 { print $8 - $6; exit }' || true)"
  [ -n "${free}" ] && [ "${need}" -gt "${free}" ] || return 0
  echo "WARNING: this run may make ~${need} files (nlive $2, num_repeats $3), and" \
    "its project quota has ${free} left: a full quota kills it with EDQUOT. Delete" \
    "finished runs' evaluations/ or lower --nlive (docs/cluster.md)." >&2
}

# `bash scripts/lib/run-config.sh --self-check` - that what is written can be
# sourced back to the same values, including a metric that needs quoting.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  _dir="$(mktemp -d)"
  NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 NS_SEED=41 NS_RETRIES=2 \
    NS_METRIC='total_rms_jy - 0.5 * snr' NS_MPI_PROCS=7 R2D2_OMP_THREADS=2 R2D2_INTEROP_THREADS=1 R2D2_MAX_THREADS=16 \
    NS_STALL_TIMEOUT=3600 NS_SYNCHRONOUS=1 NS_KEEP_MEASUREMENT_SETS=1 NS_IMAGE_DIM=96 NS_NODES=2 \
    write_run_config "${_dir}" r2d2
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
    [ "${R2D2_INTEROP_THREADS}" = 1 ]
    [ "${R2D2_MAX_THREADS}" = 16 ]
    [ "${NS_NODES}" = 2 ]
    [ "${NS_STALL_TIMEOUT}" = 3600 ]
    [ "${NS_SYNCHRONOUS}" = 1 ]
    [ "${NS_KEEP_MEASUREMENT_SETS}" = 1 ]
    [ "${NS_IMAGE_DIM}" = 96 ]
    [ "${NS_IMAGER_IMAGE_ID}" = unknown ]
    [ "${NS_MEQTREES_IMAGE_ID}" = unknown ]
    [ "${NS_POLYCHORD_IMAGE_ID}" = unknown ]
  )
  NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 NS_SEED=41 NS_RETRIES=0 \
    NS_METRIC=total_rms_jy NS_MPI_PROCS=8 R2D2_OMP_THREADS='' \
    NS_STALL_TIMEOUT=0 NS_SYNCHRONOUS=0 NS_KEEP_MEASUREMENT_SETS=0 \
    NS_WSCLEAN_MGAIN=0.9 NS_WSCLEAN_NITER=250 NS_IMAGE_DIM=64 \
    write_run_config "${_dir}" wsclean
  grep -q NS_NODES "${_dir}/run.env" && {
    echo "FAIL: a single-node run.env names NS_NODES"; exit 1
  }
  grep -qx 'NS_IMAGE_DIM=64' "${_dir}/run.env" || {
    echo "FAIL: NS_IMAGE_DIM not recorded in run.env"; exit 1
  }
  grep -qx 'NS_WSCLEAN_MGAIN=0.9' "${_dir}/run.env" || {
    echo "FAIL: --mgain not recorded in run.env"; exit 1
  }
  grep -qx 'NS_WSCLEAN_NITER=250' "${_dir}/run.env" || {
    echo "FAIL: --niter not recorded in run.env"; exit 1
  }
  grep -qx 'NS_STALL_TIMEOUT=0' "${_dir}/run.env" || {
    echo "FAIL: --stall-timeout 0 not recorded in run.env"; exit 1
  }
  grep -qx 'NS_KEEP_MEASUREMENT_SETS=0' "${_dir}/run.env" || {
    echo "FAIL: NS_KEEP_MEASUREMENT_SETS=0 not recorded in run.env"; exit 1
  }
  grep -q R2D2_OMP_THREADS "${_dir}/run.env" && {
    echo "FAIL: empty R2D2_OMP_THREADS written for wsclean"; exit 1
  }
  NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 NS_SEED=41 NS_RETRIES=0 \
    NS_METRIC=total_rms_jy NS_MPI_PROCS=8 NS_STALL_TIMEOUT=0 NS_SYNCHRONOUS=0 \
    NS_KEEP_MEASUREMENT_SETS=0 NS_WSCLEAN_MGAIN=0.9 NS_WSCLEAN_NITER=250 \
    NS_IMAGE_DIM=96 WSCLEAN_TARGET_CPU=native \
    write_run_config "${_dir}" wsclean
  grep -qx 'WSCLEAN_TARGET_CPU=native' "${_dir}/run.env" || {
    echo "FAIL: native WSClean target not recorded in run.env"; exit 1
  }

  _parent="${_dir}/runs"
  _first="$(RUN_ID=20260101T000000Z ns_claim_run_dir "${_parent}" wsclean-vlaa-)"
  [ "${_first}" = "${_parent}/wsclean-vlaa-20260101T000000Z" ]
  [ -d "${_first}" ]
  _second="$(RUN_ID=20260101T000000Z ns_claim_run_dir "${_parent}" wsclean-vlaa-)"
  [ "${_second}" != "${_first}" ]
  [ -d "${_second}" ]
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

  # A `--output-dir` the containers cannot see is refused before anything is
  # started, and one under REPO_ROOT is not.
  REPO_ROOT="${_dir}/repo"
  mkdir -p "${REPO_ROOT}/results/nested-sampling/keep-me"
  ( ns_refuse_unmounted_run "${REPO_ROOT}/results/nested-sampling/keep-me" ) ||
    { echo "FAIL: a run directory under REPO_ROOT must be allowed"; exit 1; }
  [ -d "${REPO_ROOT}/results/nested-sampling/keep-me" ] ||
    { echo "FAIL: an allowed run directory must not be removed"; exit 1; }
  # Not a prefix match on the repo's own name: /repo-elsewhere is outside /repo.
  _outside="${_dir}/repo-elsewhere/run"
  mkdir -p "${_outside}"
  if _out="$( ns_refuse_unmounted_run "${_outside}" 2>&1 )"; then
    echo "FAIL: a run directory outside REPO_ROOT must be refused, got: ${_out}"; exit 1
  fi
  case "${_out}" in
    *"outside the repository"*"${REPO_ROOT}"*) ;;
    *) echo "FAIL: the refusal must name the repository it is outside of, got: ${_out}"; exit 1 ;;
  esac
  # The directory the run script created a moment ago to resolve the path is
  # cleaned up, and only while it is empty.
  [ ! -d "${_outside}" ] || { echo "FAIL: the empty refused directory must be removed"; exit 1; }
  mkdir -p "${_outside}/evaluations"
  ( ns_refuse_unmounted_run "${_outside}" 2>/dev/null ) && :
  [ -d "${_outside}/evaluations" ] ||
    { echo "FAIL: a refused directory with contents must be left alone"; exit 1; }

  # A missing checkpoint set is refused before the run starts, because the
  # search cannot report its own absence: every evaluation would score
  # FAILURE_OBJECTIVE, which PolyChord maximizes.
  _ckpts="${_dir}/checkpoints"
  mkdir -p "${_ckpts}/R2D2_A1"
  if _out="$( ns_refuse_missing_checkpoints "${_ckpts}" R2D2_A1 2>&1 )"; then
    echo "FAIL: an empty checkpoint directory must be refused, got: ${_out}"; exit 1
  fi
  case "${_out}" in
    *"no R2D2 checkpoints"*"fetch-checkpoints"*"CHECKPOINTS_DIR"*) ;;
    *) echo "FAIL: the refusal must say how to fix it, got: ${_out}"; exit 1 ;;
  esac
  # A directory that is not there at all is the worktree case, and reads the same.
  if _out="$( ns_refuse_missing_checkpoints "${_ckpts}" R2D2_MISSING 2>&1 )"; then
    echo "FAIL: a missing checkpoint directory must be refused, got: ${_out}"; exit 1
  fi
  # ...and one checkpoint is enough: which realisations the run needs is the
  # imager's business, and it says so itself once it can start.
  touch "${_ckpts}/R2D2_A1/R2D2_UNet_N1.ckpt"
  ( ns_refuse_missing_checkpoints "${_ckpts}" R2D2_A1 ) ||
    { echo "FAIL: a populated checkpoint directory must be allowed"; exit 1; }
  # The name is a name, not a path: the same set under a different
  # CHECKPOINTS_DIR is found, which is what makes a worktree fixable.
  mkdir -p "${_dir}/elsewhere/R2D2_A1"
  touch "${_dir}/elsewhere/R2D2_A1/R2D2_UNet_N1.ckpt"
  ( ns_refuse_missing_checkpoints "${_dir}/elsewhere" R2D2_A1 ) ||
    { echo "FAIL: CHECKPOINTS_DIR must be what decides where to look"; exit 1; }

  # The file-quota warning, against a stand-in lfs with FREE files left on
  # project 110845 (FREE empty: no project, as off Lustre).
  # shellcheck disable=SC2016  # expanded by the stand-in, not here
  printf '%s\n' '#!/bin/sh' \
    'case "$1" in project) [ -n "${FREE}" ] && echo "110845 P $3" || echo "0 - $3" ;;' \
    'quota) echo "Disk quotas for prj 110845 (pid 110845):"' \
    '  echo "Filesystem kbytes quota limit grace files quota limit grace"' \
    '  echo ". 1 2 3 - $((1048576 - FREE)) 1048576 1048576 -" ;; esac' >"${_dir}/lfs"
  chmod +x "${_dir}/lfs"
  quota_warns() { LFS="${_dir}/lfs" FREE="$1" ns_warn_file_quota "${_dir}" "$2" 15 "$3" 2>&1 | grep -c EDQUOT; }
  # nlive 150 converged: 4 x 15 x 1350 = 81,000 files.
  [ "$(quota_warns 650000 150 -1)" = 0 ] || { echo "FAIL: 650k free must fit nlive 150"; exit 1; }
  [ "$(quota_warns 60000 150 -1)" = 1 ] || { echo "FAIL: 60k free must warn for nlive 150"; exit 1; }
  [ "$(quota_warns 60000 150 900)" = 0 ] || { echo "FAIL: max-ndead must bound the estimate"; exit 1; }
  [ "$(quota_warns '' 5000 -1)" = 0 ] || { echo "FAIL: no project quota must stay quiet"; exit 1; }
  [ "$(LFS="${_dir}/missing" ns_warn_file_quota "${_dir}" 5000 15 -1 2>&1 | grep -c EDQUOT)" = 0 ] \
    || { echo "FAIL: no lfs must stay quiet"; exit 1; }
  # Not inside `||` or `[ ]`, which would suspend `set -e`: lfs failing (off
  # Lustre) must not end the caller.
  printf '#!/bin/sh\nexit 25\n' >"${_dir}/lfs-fails"
  chmod +x "${_dir}/lfs-fails"
  _reached="$(set -e -o pipefail; LFS="${_dir}/lfs-fails" ns_warn_file_quota "${_dir}" 150 15 -1; echo reached)"
  [ "${_reached}" = reached ] || { echo "FAIL: lfs failing off Lustre must not end the run"; exit 1; }

  rm -rf "${_dir}"
  echo "run-config self-check passed"
fi
