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
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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
