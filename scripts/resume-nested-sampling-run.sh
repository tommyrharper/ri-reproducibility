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
# For ns_budget_ranks and the per-rank memory figures it is called with.
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"

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

  # The rank count run.env recorded goes through the memory clamp rather than
  # straight into mpirun. Both directions are checked end to end on a run
  # directory outside the repository: with the clamp taken out the run script
  # refuses that in ~0.1s (ns_refuse_unmounted_run) with a *different*
  # message, so a broken clamp fails these assertions instead of starting a
  # real search from the self-check.
  SHORT_RUN="${TMP}/wsclean-selfcheck"
  mkdir -p "${SHORT_RUN}"
  printf 'NS_ALGORITHM=wsclean\nNS_MPI_PROCS=8\n' >"${SHORT_RUN}/run.env"
  # A docker stub, because the budget reaps leaked sidecars and a check may
  # not `docker rm` anything on a host it shares; and a reservation directory
  # of its own, so the memory it sets aside cannot shrink a real run.
  mkdir -p "${TMP}/bin"
  printf '#!/bin/sh\nexit 0\n' >"${TMP}/bin/docker"
  chmod +x "${TMP}/bin/docker"
  # shellcheck disable=SC2031  # the self-check block exits; there is no
  # enclosing shell for the export to be lost to
  export PATH="${TMP}/bin:${PATH}"
  export NS_RANK_BUDGET_DIR="${TMP}/budget"

  # 4900MB less the 4096MB headroom is 804MB, which affords 4 WSClean ranks.
  OUT="$(NS_AVAILABLE_MB=4900 bash "$0" "${SHORT_RUN}" 2>&1 || true)"
  # The whole line, so that it also covers the evaluation count reaching it:
  # a run with no evaluations directory used to take the script out here.
  case "${OUT}" in
    *"Resuming wsclean-selfcheck (wsclean, 0 evaluations already done, 4 ranks)"*) ;;
    *) echo "FAIL: resume must re-clamp the recorded rank count, got: ${OUT}"; exit 1 ;;
  esac

  # The hours between the run stopping and someone typing this are downtime,
  # and `./ri health` measures every rate over the time a run was running - so
  # a resume has to leave the same record run_with_retries leaves, in the same
  # file and the same stamp format, or a resumed run's throughput and forecast
  # are divided by the night it spent stopped.
  if ! grep -q '^20.*Z resumed at 0 evaluations$' "${SHORT_RUN}/restarts.log"; then
    echo "FAIL: a resume must record its downtime, got:" \
      "$(cat "${SHORT_RUN}/restarts.log" 2>/dev/null)"
    exit 1
  fi

  # Not even one rank fits: stops here, rather than at evaluation 1 with the
  # OOM killer scoring the run's best points.
  if OUT="$(NS_AVAILABLE_MB=1 bash "$0" "${SHORT_RUN}" 2>&1)"; then
    echo "FAIL: resuming with no memory for a rank must refuse, got: ${OUT}"
    exit 1
  fi
  case "${OUT}" in
    *"not enough free memory"*) ;;
    *) echo "FAIL: the refusal must say why, got: ${OUT}"; exit 1 ;;
  esac
  # ...and a refused resume is not downtime, because nothing stopped and
  # nothing started. Only a resume that reaches the run script writes a line.
  if [ "$(wc -l <"${SHORT_RUN}/restarts.log" | tr -d ' ')" != 1 ]; then
    echo "FAIL: a refused resume must record nothing, got:" \
      "$(cat "${SHORT_RUN}/restarts.log")"
    exit 1
  fi

  # What a resume says it is continuing, and what it builds first. Its own
  # run directory, so the restarts.log assertions above are not disturbed.
  COUNT_RUN="${TMP}/wsclean-counted"
  mkdir -p "${COUNT_RUN}/evaluations/eval-0001-a" \
           "${COUNT_RUN}/evaluations/eval-0002-b" \
           "${COUNT_RUN}/evaluations/eval-0003-c"
  printf 'NS_ALGORITHM=wsclean\nNS_MPI_PROCS=1\n' >"${COUNT_RUN}/run.env"
  # Two scored, one killed before it wrote a record - which the resume throws
  # away (adopt_completed_evaluations), so it must not be counted as done.
  echo '{}' >"${COUNT_RUN}/evaluations/eval-0001-a/metrics.json"
  echo '{}' >"${COUNT_RUN}/evaluations/eval-0002-b/metrics.json"

  OUT="$(NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"(wsclean, 2 evaluations already done, 1 rank)"*) ;;
    *) echo "FAIL: resume must count scored evaluations, got: ${OUT}"; exit 1 ;;
  esac
  # The three images this algorithm's ranks execute, built before the run
  # script is reached - the docker stub above makes each build a no-op, so
  # what is checked is that build.sh was asked for the right set.
  for want in wsclean meqtrees polychord; do
    case "${OUT}" in
      *"ri-reproducibility/${want}"*) ;;
      *) echo "FAIL: a resume must build ${want}, got: ${OUT}"; exit 1 ;;
    esac
  done

  OUT="$(NS_NO_BUILD=1 NS_AVAILABLE_MB=4900 bash "$0" "${COUNT_RUN}" 2>&1 || true)"
  case "${OUT}" in
    *"ri-reproducibility/"*)
      echo "FAIL: --no-build must skip the builds, got: ${OUT}"; exit 1 ;;
  esac

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

# The same builds `./ri search` does, for the same reason and at the same
# cost. A run executes the code baked into its images - `polychord` copies
# the whole of scripts/lib/nested_sampling, `meqtrees` three of its scripts
# and defaults.toml - so without this a resume runs whatever was baked
# whenever those images were last built, silently. That is exactly backwards
# for the case a resume is most often typed in: something killed the run, the
# bug was fixed in the working tree, and `./ri health` said `./ri resume`.
# Reproduced by marking polychord_wsclean.py and resuming: the mark never ran.
#
# Free when nothing changed - build.sh compares an inputs hash held in a
# label and skips, ~0.05s per image - and the builds go before the rank clamp
# below so the memory that clamp reads is the memory left after them.
# NS_NO_BUILD=1 (`./ri resume --no-build`) is the way out for a working tree
# that has moved on and must not reach a run already in flight.
if [ -z "${NS_NO_BUILD:-}" ]; then
  for image in ${IMAGES}; do
    scripts/build.sh "${image}"
  done
fi

# The rank count in run.env is what the memory clamp handed the run when it
# started, not a number anyone chose, so replaying it verbatim is the one
# place a resume can walk into the failure rank-budget.sh exists to prevent:
# 16 R2D2 ranks is 53GB, and on a shared host the memory that was free
# yesterday is another session's run today. The run scripts only *warn* about
# an NS_MPI_PROCS they are given, because a caller who typed --mpi-procs
# means it - a resume did not type anything. Every rank that does not fit has
# its worker OOM-killed, and common.py scores that as FAILURE_OBJECTIVE,
# which PolyChord maximizes, so the run does not crash: it reports the corner
# of the parameter space where it ran out of memory as its best discovery.
#
# Clamping down is free because PolyChord's checkpoint carries live points,
# not ranks: measured by resuming a 3-rank wsclean run at 1 rank, which
# adopted its 29 evaluations and wrote its summary.json normally. Refusing
# outright (not even one rank fits) exits here rather than at evaluation 1.
if [ -n "${NS_MPI_PROCS:-}" ]; then
  # No `export`: `set -a` above already marked it, and reassigning keeps that.
  NS_MPI_PROCS="$(ns_budget_ranks "${NS_MPI_PROCS}" "${MB_PER_RANK}" "${NS_ALGORITHM}")"
fi

# Guarded rather than relying on `2>/dev/null`: under `set -o pipefail` a
# find over a directory that is not there fails the pipeline, which took the
# whole script down with exit 1 and not a word printed - and a run that died
# before its first evaluation is exactly the one someone resumes.
# Scored evaluations, not eval-* directories: a metrics.json is what an
# evaluation leaves behind when it succeeds, and adopt_completed_evaluations
# deletes the directories without one on the way in - so counting directories
# credited this resume with work it is about to throw away (three startup
# deaths here advertised 7, 7 and 15). Same rule as `./ri runs`' EVALS column
# and `./ri health`'s progress line, so the three now agree.
DONE=0
if [ -d "${RUN_DIR}/evaluations" ]; then
  DONE="$(find "${RUN_DIR}/evaluations" -mindepth 2 -maxdepth 2 -name 'metrics.json' \
    | wc -l | tr -d ' ')"
fi
if [ "${NS_MPI_PROCS:-}" = 1 ]; then RANKS="1 rank"; else RANKS="${NS_MPI_PROCS:-?} ranks"; fi
# Into the same file run_with_retries writes its restarts to, in the same
# format (UTC stamp first), because the gap between the last evaluation of the
# old attempt and the first of this one is downtime for the identical reason:
# the run was not running. `./ri health` measures every rate over time the run
# was actually going, and without this line a resume after a night stopped read
# as a night of near-zero throughput - and forecast the remainder off it.
# "resumed" rather than "exit" is what tells the report a human continued this
# run instead of it healing itself.
printf '%s resumed at %s evaluations\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DONE}" >>"${RUN_DIR}/restarts.log"
echo "Resuming ${RUN_DIR##*/} (${NS_ALGORITHM}, ${DONE} evaluations already done," \
  "${RANKS})"
exec "${RUN_SCRIPT}"
