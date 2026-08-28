#!/usr/bin/env bash
# Continue an interrupted nested-sampling run from where it stopped.
#
# The run directory carries everything needed: run.env holds the settings the
# run actually used (see scripts/lib/run-config.sh) and PolyChord's own
# .resume file holds the live points, so this only has to put the settings
# back and start the same script again with the same OUTPUT_DIR. The run
# script picks the resume file up, and polychord_*.py adopts the evaluations
# already on disk so none of them is paid for twice.
#
#   scripts/resume-nested-sampling-run.sh <run directory or name>
#   scripts/resume-nested-sampling-run.sh --self-check
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# For ns_run_is_live - the same command lines the stall watchdog kills are the
# ones that make a run too alive to resume, so there is one spelling. The guard
# belongs in a shared file rather than in each place that suggests a resume,
# because resuming a live run starts a second MPI job over the same checkpoint
# and the same FIFO directories - and `./ri runs` printed exactly that command
# for a live run, while the HTML report still can (it is a snapshot, and
# liveness in a static page would be stale by the time anyone read it).
# shellcheck source=scripts/lib/progress-bar.sh
. "${REPO_ROOT}/scripts/lib/progress-bar.sh"

if [ "${1:-}" = "--self-check" ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' EXIT
  FAKE_RUN="$(cd "${TMP}" && pwd)/r2d2-selfcheck"
  mkdir -p "${FAKE_RUN}/chains"
  echo "NS_ALGORITHM=r2d2" > "${FAKE_RUN}/run.env"
  # A fuse: with the guard broken this run reads as finished and the script
  # stops one line later, rather than the check launching a real search.
  echo '{}' > "${FAKE_RUN}/summary.json"

  ns_run_is_live "${FAKE_RUN}" && { echo "FAIL: nothing is running, so the guard must not fire"; exit 1; }

  # A real process with the real command line, rather than a faked argv: this
  # is what pgrep has to see during a run, spelled the way a rank spells it.
  printf 'import time\ntime.sleep(30)\n' > "${TMP}/polychord_r2d2.py"
  python3 "${TMP}/polychord_r2d2.py" --output-dir "${FAKE_RUN}" --nlive 50 &
  FAKE_PID=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ns_run_is_live "${FAKE_RUN}" && break
    sleep 0.2
  done
  ns_run_is_live "${FAKE_RUN}" || { echo "FAIL: a live rank on this run must be seen"; exit 1; }
  # A neighbouring run whose name this one is a prefix of must not be caught.
  ns_run_is_live "${FAKE_RUN%-selfcheck}" && { echo "FAIL: a prefix of the run directory must not match"; exit 1; }

  if OUT="$(bash "$0" "${FAKE_RUN}" 2>&1)"; then
    echo "FAIL: resuming a live run must refuse, got: ${OUT}"
    exit 1
  fi
  case "${OUT}" in
    *"still running"*) ;;
    *) echo "FAIL: the refusal must say why, got: ${OUT}"; exit 1 ;;
  esac

  kill "${FAKE_PID}" 2>/dev/null || true
  wait "${FAKE_PID}" 2>/dev/null || true
  echo "resume-nested-sampling-run self-check passed"
  exit 0
fi

RUN="${1:-}"
if [ -z "${RUN}" ]; then
  echo "usage: $0 <run directory or name>" >&2
  echo "       ./ri runs lists what there is to resume" >&2
  exit 2
fi

# A bare name is the common case, because that is what `./ri runs` prints.
if [ -d "${RUN}" ]; then
  RUN_DIR="$(cd "${RUN}" && pwd)"
elif [ -d "results/nested-sampling/${RUN}" ]; then
  RUN_DIR="$(cd "results/nested-sampling/${RUN}" && pwd)"
else
  echo "FATAL: no such run: ${RUN}" >&2
  echo "       ./ri runs lists what there is to resume" >&2
  exit 1
fi

if ns_run_is_live "${RUN_DIR}"; then
  echo "FATAL: ${RUN_DIR##*/} is still running, so there is nothing to resume." >&2
  echo "       A second job over the same checkpoint would corrupt both." >&2
  echo "       Watch it instead:  ./ri health ${RUN_DIR##*/}" >&2
  exit 1
fi

if [ -e "${RUN_DIR}/summary.json" ]; then
  echo "Nothing to do: ${RUN_DIR##*/} already finished (it has a summary.json)."
  echo "Start a new search instead, or delete the summary to force a rerun."
  exit 0
fi

if [ ! -e "${RUN_DIR}/run.env" ]; then
  # Runs started before run.env existed, or a directory that was never a run.
  echo "FATAL: ${RUN_DIR##*/} has no run.env, so its settings are unknown." >&2
  echo "       Resume it by hand with the flags it was started with:" >&2
  echo "       ./ri search <imager> --output-dir ${RUN_DIR}" >&2
  exit 1
fi

# `set -a` so the run script and defaults.sh see these as environment
# overrides, which is exactly how a flag would have reached them.
set -a
# shellcheck disable=SC1091
. "${RUN_DIR}/run.env"
set +a
export OUTPUT_DIR="${RUN_DIR}"

case "${NS_ALGORITHM}" in
  r2d2) RUN_SCRIPT="scripts/run-nested-sampling-r2d2.sh" ;;
  wsclean) RUN_SCRIPT="scripts/run-nested-sampling.sh" ;;
  *)
    echo "FATAL: ${RUN_DIR}/run.env has an unknown NS_ALGORITHM=${NS_ALGORITHM}" >&2
    exit 1
    ;;
esac

DONE="$(find "${RUN_DIR}/evaluations" -maxdepth 1 -name 'eval-*' 2>/dev/null | wc -l | tr -d ' ')"
echo "Resuming ${RUN_DIR##*/} (${NS_ALGORITHM}, ${DONE} evaluations already done)"
exec "${RUN_SCRIPT}"
