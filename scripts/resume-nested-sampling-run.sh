#!/usr/bin/env bash
# Resume an interrupted nested-sampling run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/progress-bar.sh
. "${REPO_ROOT}/scripts/lib/progress-bar.sh"
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"

if [ "${1:-}" = "--self-check" ]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' EXIT
  FAKE_RUN="$(cd "${TMP}" && pwd)/r2d2-selfcheck"
  mkdir -p "${FAKE_RUN}/chains"
  echo "NS_ALGORITHM=r2d2" > "${FAKE_RUN}/run.env"
  echo '{}' > "${FAKE_RUN}/summary.json"

  ns_run_is_live "${FAKE_RUN}" && { echo "FAIL: nothing is running, so the guard must not fire"; exit 1; }

  printf 'import time\ntime.sleep(30)\n' > "${TMP}/polychord_r2d2.py"
  python3 "${TMP}/polychord_r2d2.py" --output-dir "${FAKE_RUN}" --nlive 50 &
  FAKE_PID=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    ns_run_is_live "${FAKE_RUN}" && break
    sleep 0.2
  done
  ns_run_is_live "${FAKE_RUN}" || { echo "FAIL: a live rank on this run must be seen"; exit 1; }
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

  SHORT_RUN="${TMP}/wsclean-selfcheck"
  mkdir -p "${SHORT_RUN}"
  printf 'NS_ALGORITHM=wsclean\nNS_MPI_PROCS=8\n' >"${SHORT_RUN}/run.env"
  mkdir -p "${TMP}/bin"
  printf '#!/bin/sh\nexit 0\n' >"${TMP}/bin/docker"
  chmod +x "${TMP}/bin/docker"
  # shellcheck disable=SC2031  # the self-check block exits; there is no
  # enclosing shell for the export to be lost to
  export PATH="${TMP}/bin:${PATH}"
  export NS_RANK_BUDGET_DIR="${TMP}/budget"

  OUT="$(NS_AVAILABLE_MB=4900 bash "$0" "${SHORT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"Resuming wsclean-selfcheck (wsclean, 0 evaluations already done, 4 ranks)"*) ;;
    *) echo "FAIL: resume must re-clamp the recorded rank count, got: ${OUT}"; exit 1 ;;
  esac

  if ! grep -q '^20.*Z resumed at 0 evaluations$' "${SHORT_RUN}/restarts.log"; then
    echo "FAIL: a resume must record its downtime, got:" \
      "$(cat "${SHORT_RUN}/restarts.log" 2>/dev/null)"
    exit 1
  fi

  if OUT="$(NS_AVAILABLE_MB=1 bash "$0" "${SHORT_RUN}" 2>&1)"; then
    echo "FAIL: resuming with no memory for a rank must refuse, got: ${OUT}"
    exit 1
  fi
  case "${OUT}" in
    *"not enough free memory"*) ;;
    *) echo "FAIL: the refusal must say why, got: ${OUT}"; exit 1 ;;
  esac
  if [ "$(wc -l <"${SHORT_RUN}/restarts.log" | tr -d ' ')" != 1 ]; then
    echo "FAIL: a refused resume must record nothing, got:" \
      "$(cat "${SHORT_RUN}/restarts.log")"
    exit 1
  fi

  COUNT_RUN="${TMP}/wsclean-counted"
  mkdir -p "${COUNT_RUN}/evaluations/eval-0001-a" \
           "${COUNT_RUN}/evaluations/eval-0002-b" \
           "${COUNT_RUN}/evaluations/eval-0003-c"
  printf 'NS_ALGORITHM=wsclean\nNS_MPI_PROCS=1\n' >"${COUNT_RUN}/run.env"
  echo '{}' >"${COUNT_RUN}/evaluations/eval-0001-a/metrics.json"
  echo '{}' >"${COUNT_RUN}/evaluations/eval-0002-b/metrics.json"

  OUT="$(NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"(wsclean, 2 evaluations already done, 1 rank)"*) ;;
    *) echo "FAIL: resume must count scored evaluations, got: ${OUT}"; exit 1 ;;
  esac
  for want in wsclean meqtrees polychord; do
    case "${OUT}" in
      *"ri-reproducibility/${want}"*) ;;
      *) echo "FAIL: a resume must build ${want}, got: ${OUT}"; exit 1 ;;
    esac
  done

  OUT="$(NS_NO_BUILD=1 NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  for want in wsclean meqtrees polychord; do
    case "${OUT}" in
      *"ri-reproducibility/${want}"*)
        echo "FAIL: --no-build must skip the ${want} build, got: ${OUT}"; exit 1 ;;
    esac
  done

  echo '{ "evaluations": [] }' >"${COUNT_RUN}/summary.json"
  OUT="$(NS_NO_BUILD=1 NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"already finished"*) ;;
    *) echo "FAIL: a finished run must not be resumed, got: ${OUT}"; exit 1 ;;
  esac
  printf '{\n  "evaluations": [\n    {\n      "eval_id": 1,\n      "para' \
    >"${COUNT_RUN}/summary.json"
  OUT="$(NS_NO_BUILD=1 NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"already finished"*)
      echo "FAIL: a torn summary must not count as finished, got: ${OUT}"; exit 1 ;;
  esac
  case "${OUT}" in
    *"half-written summary.json"*"2 evaluations already done"*) ;;
    *) echo "FAIL: the resume must say it is rewriting the summary, got: ${OUT}"; exit 1 ;;
  esac
  rm -f "${COUNT_RUN}/summary.json"

  echo "resume-nested-sampling-run self-check passed"
  exit 0
fi

RUN="${1:-}"
if [ -z "${RUN}" ]; then
  echo "usage: $0 <run directory or name>" >&2
  echo "       ./ri runs lists what there is to resume" >&2
  exit 2
fi

# Resolve path or bare name printed by `./ri runs`.
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

# Complete summary means finished; partial summary is rebuilt from checkpoint
# and cached evaluations. Check tail because summaries can be tens of MB.
if [ -s "${RUN_DIR}/summary.json" ] \
  && [ "$(tail -c 64 "${RUN_DIR}/summary.json" | tr -d '[:space:]' | tail -c 1)" = "}" ]; then
  echo "Nothing to do: ${RUN_DIR##*/} already finished (it has a summary.json)."
  echo "Start a new search instead, or delete the summary to force a rerun."
  exit 0
fi
if [ -e "${RUN_DIR}/summary.json" ]; then
  echo "${RUN_DIR##*/} has a half-written summary.json, so it did not finish."
  echo "Resuming rewrites it from the evaluations already on disk."
fi

if [ ! -e "${RUN_DIR}/run.env" ]; then
  # Covers pre-run.env runs and directories that were never runs.
  echo "FATAL: ${RUN_DIR##*/} has no run.env, so its settings are unknown." >&2
  echo "       Resume it by hand with the flags it was started with:" >&2
  echo "       ./ri search <imager> --output-dir ${RUN_DIR}" >&2
  exit 1
fi

# Export run.env values as the same environment overrides flags produce.
set -a
# shellcheck disable=SC1091
. "${RUN_DIR}/run.env"
set +a
export OUTPUT_DIR="${RUN_DIR}"

case "${NS_ALGORITHM}" in
  r2d2) RUN_SCRIPT="scripts/run-nested-sampling-r2d2.sh"
        MB_PER_RANK="${NS_R2D2_MB_PER_RANK}"
        IMAGES="r2d2 meqtrees polychord" ;;
  wsclean) RUN_SCRIPT="scripts/run-nested-sampling.sh"
        MB_PER_RANK="${NS_WSCLEAN_MB_PER_RANK}"
        IMAGES="wsclean meqtrees polychord" ;;
  *)
    echo "FATAL: ${RUN_DIR}/run.env has an unknown NS_ALGORITHM=${NS_ALGORITHM}" >&2
    exit 1
    ;;
esac

# Rebuild images before rank clamping so resume uses current code and memory.
# NS_NO_BUILD=1 skips this for a deliberately frozen working tree.
if [ -z "${NS_NO_BUILD:-}" ]; then
  for image in ${IMAGES}; do
    scripts/build.sh "${image}"
  done
fi

# Re-clamp ranks against current memory; checkpoints resume safely with fewer
# ranks, but refuse before evaluation 1 if none fits.
if [ -n "${NS_MPI_PROCS:-}" ]; then
  # No `export`: `set -a` above already marked it, and reassigning keeps that.
  NS_MPI_PROCS="$(ns_budget_ranks "${NS_MPI_PROCS}" "${MB_PER_RANK}" "${NS_ALGORITHM}")"
fi

# Count scored evaluations; guard missing evaluations/ under `pipefail`.
DONE=0
if [ -d "${RUN_DIR}/evaluations" ]; then
  DONE="$(find "${RUN_DIR}/evaluations" -mindepth 2 -maxdepth 2 -name 'metrics.json' \
    | wc -l | tr -d ' ')"
fi
if [ "${NS_MPI_PROCS:-}" = 1 ]; then RANKS="1 rank"; else RANKS="${NS_MPI_PROCS:-?} ranks"; fi
# Record downtime beside automatic retries; health excludes it.
printf '%s resumed at %s evaluations\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DONE}" >>"${RUN_DIR}/restarts.log"
echo "Resuming ${RUN_DIR##*/} (${NS_ALGORITHM}, ${DONE} evaluations already done," \
  "${RANKS})"
exec "${RUN_SCRIPT}"
