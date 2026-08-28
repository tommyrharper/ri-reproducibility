#!/usr/bin/env bash
# End-to-end check that a search which is killed mid-flight heals itself.
#
# The one robustness property this repo depends on most: a multi-day R2D2 search
# that loses its ranks at hour three has to come back on its own, because
# nothing else notices for hours. `run_with_retries` in scripts/lib/progress-bar.sh
# implements it and its own --self-check covers the decision with fixtures, but
# nothing joined the fixtures to a real search - and the two bugs found there so
# far (the retry predicate reading a checkpoint-frozen counter, the stall
# accounting refusing to excuse a run's own restart) were both invisible to a
# fixture and obvious to a real kill. So: start a real search, SIGKILL it, and
# assert it finishes anyway.
#
# ~40 seconds and ~0.6GB, on a throwaway output directory that `./ri runs` and
# the report never see. WSClean rather than R2D2 because it reaches its first
# checkpoint in ~15s where R2D2 takes over an hour.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Under REPO_ROOT, not /tmp: the simulate workers are reached over FIFOs inside
# the run directory, and only the repo is bind-mounted into the sidecars. An
# output directory outside it silently drops the worker pool, which is a
# different run from the one this is meant to check. Under results/ but not
# results/nested-sampling/, so it is already gitignored and `./ri runs` - which
# globs results/nested-sampling/* - never sees it.
OUT="${REPO_ROOT}/results/.self-heal-check-$$"
# Deliberately fewer than `--nlive`, so the kill lands before PolyChord has
# written its first `.resume` - the regime that was broken. A run killed after
# one takes the same path with strictly more on disk to adopt.
KILL_AFTER_EVALS=8
RECOVER_TIMEOUT_SECONDS=300
SEARCH_PID=""
PASSED=""

# The run directory is kept when the check fails: run.log is the only artifact
# that says why a search stopped, and this is the check that produces one worth
# reading.
cleanup() {
  # `|| true` on both: this runs under `set -e`, and a pkill that matches
  # nothing exits 1, which killed the trap half-way through and left every run
  # of this check behind - including the ones that passed.
  [ -n "${SEARCH_PID}" ] && { kill -9 "${SEARCH_PID}" 2>/dev/null || true; }
  pkill -9 -f "polychord_wsclean.py --output-dir ${OUT}" 2>/dev/null || true
  if [ -n "${PASSED}" ]; then
    rm -rf "${OUT}" "${OUT}.log"
  else
    echo "self-heal: left ${OUT} and ${OUT}.log for inspection" >&2
  fi
  return 0
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

completed_evals() {
  find "${OUT}/evaluations" -maxdepth 2 -name metrics.json 2>/dev/null | wc -l | tr -d ' '
}

echo "self-heal: starting a wsclean search in ${OUT}"
# --retries 1 rather than the default 2, so the run finishing also proves the
# first retry was the one that did it.
"${REPO_ROOT}/ri" search wsclean \
  --nlive 20 --num-repeats 2 --mpi-procs 3 --retries 1 --no-build \
  --output-dir "${OUT}" >"${OUT}.log" 2>&1 &
SEARCH_PID=$!

# Killing before the first evaluation would be a different test - that case is
# the anti-spin guard, and progress-bar.sh's own self-check covers it.
for _ in $(seq 1 120); do
  [ "$(completed_evals)" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null || fail "search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${OUT}.log"
  sleep 1
done
before="$(completed_evals)"
[ "${before}" -ge "${KILL_AFTER_EVALS}" ] || fail "only ${before} evaluations after 120s; see ${OUT}.log"

# The point of killing this early: PolyChord writes `.resume` at its first
# checkpoint, so before that the restart has evaluations on disk and no
# checkpoint to resume from. Adopting them was once conditional on the resume
# file, and without it the restart began at eval id 1 on top of the first
# attempt's directories - one rank died on FileExistsError and the others hung
# in the collective forever. Asserted rather than assumed, because a fixture
# that quietly drifts past the first checkpoint stops testing that.
compgen -G "${OUT}/chains/*.resume" >/dev/null && {
  echo "FAIL: PolyChord already checkpointed at ${before} evaluations; lower KILL_AFTER_EVALS" >&2
  exit 1
}

# Matched on this run's own --output-dir so a search someone else is running on
# this host is not caught in it. SIGKILL because that is what the OOM killer
# sends, and it leaves no chance to shut down cleanly.
echo "self-heal: killing the sampler at ${before} evaluations, before any checkpoint"
pkill -9 -f "polychord_wsclean.py --output-dir ${OUT}" || fail "no sampler process to kill"

# Bounded rather than a bare `wait`, because the failure this is most likely to
# catch is a hang, not an exit: PolyChord calls the likelihood from Fortran, so
# an exception in one rank unwinds that rank and leaves the others in a
# collective that never completes. That is why the likelihood aborts the whole
# job on anything it does not expect - and why waiting forever here would turn
# a regression in that into a check that never returns. The run itself takes
# ~30s.
for _ in $(seq 1 "${RECOVER_TIMEOUT_SECONDS}"); do
  kill -0 "${SEARCH_PID}" 2>/dev/null || break
  sleep 1
done
kill -0 "${SEARCH_PID}" 2>/dev/null && fail "the restarted run neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s - a rank is stuck in an MPI collective; see ${OUT}.log"
wait "${SEARCH_PID}" && status=0 || status=$?
SEARCH_PID=""
[ "${status}" -eq 0 ] || fail "the killed search did not recover (exit ${status}); see ${OUT}.log"

[ -f "${OUT}/summary.json" ] || fail "run finished with no summary.json"
[ -f "${OUT}/restarts.log" ] || fail "nothing recorded in restarts.log"
restarts="$(wc -l <"${OUT}/restarts.log" | tr -d ' ')"
[ "${restarts}" -eq 1 ] || fail "expected 1 restart, restarts.log has ${restarts}"
grep -q "exit 137" "${OUT}/restarts.log" || fail "restarts.log did not record the SIGKILL: $(cat "${OUT}/restarts.log")"

# The evaluations the first attempt scored have to survive the restart, or the
# retry is redoing hours of imaging rather than resuming (adopt_completed_
# evaluations in common.py serves those points from cache).
after="$(completed_evals)"
[ "${after}" -ge "${before}" ] || fail "restart lost work: ${before} evaluations before the kill, ${after} after"

# The run healed itself, so the report must not call that a fault: a false
# warning here is a false alarm in anything scripted on ./ri health's exit code.
health="$("${REPO_ROOT}/ri" health "${OUT}" 2>&1)" && health_status=0 || health_status=$?
[ "${health_status}" -eq 0 ] || fail "./ri health warns about a run that healed itself (exit ${health_status}):
${health}"
grep -q "self-healed restart" <<<"${health}" || fail "./ri health does not report the restart:
${health}"

PASSED=1
echo "self-heal: killed at ${before} evaluations, recovered and finished at ${after}"
echo "self-heal check passed"
