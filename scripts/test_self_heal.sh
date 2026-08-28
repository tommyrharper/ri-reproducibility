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
# fixture and obvious to a real kill. So: start a real search, break it, and
# assert it finishes anyway.
#
# Two ways of breaking it, because they recover through different machinery. A
# SIGKILL is the crash `run_with_retries` was written for - the run exits and
# the loop sees a status. A frozen rank is the failure it cannot see: PolyChord
# calls the likelihood from Fortran, so one rank that stops answering leaves
# every other rank in a collective forever, with no exit for anything to act
# on. `_ns_stall_watchdog` in scripts/lib/progress-bar.sh is what turns the
# second into the first, and SIGSTOP on one rank reproduces it exactly.
#
# ~90 seconds and ~0.6GB, on throwaway output directories that `./ri runs` and
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
HUNG_OUT="${REPO_ROOT}/results/.self-heal-hang-check-$$"
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
  # -CONT before the -9 on the hang scenario: a stopped process does take a
  # SIGKILL, but anything of this check's own that is blocked waiting on it
  # will not move until it does.
  pkill -CONT -f "polychord_wsclean.py --output-dir ${HUNG_OUT}" 2>/dev/null || true
  pkill -9 -f "polychord_wsclean.py --output-dir ${OUT}" 2>/dev/null || true
  pkill -9 -f "polychord_wsclean.py --output-dir ${HUNG_OUT}" 2>/dev/null || true
  if [ -n "${PASSED}" ]; then
    rm -rf "${OUT}" "${OUT}.log" "${HUNG_OUT}" "${HUNG_OUT}.log"
  else
    echo "self-heal: left ${OUT}* and ${HUNG_OUT}* for inspection" >&2
  fi
  return 0
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

completed_evals() {
  find "${1}/evaluations" -maxdepth 2 -name metrics.json 2>/dev/null | wc -l | tr -d ' '
}

# Bounded rather than a bare `wait`, because the failure this whole script is
# most likely to catch is a hang, not an exit - waiting forever on one would
# turn a regression into a check that never returns.
wait_for_exit() {
  local pid="$1" seconds="$2"
  for _ in $(seq 1 "${seconds}"); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 1
  done
  return 1
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
  [ "$(completed_evals "${OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null || fail "search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${OUT}.log"
  sleep 1
done
before="$(completed_evals "${OUT}")"
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

# The restarted run itself takes ~30s. An exception in one rank unwinds that
# rank and leaves the others in a collective that never completes, which is why
# the likelihood aborts the whole job on anything it does not expect.
wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "the restarted run neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s - a rank is stuck in an MPI collective; see ${OUT}.log"
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
after="$(completed_evals "${OUT}")"
[ "${after}" -ge "${before}" ] || fail "restart lost work: ${before} evaluations before the kill, ${after} after"

# The run healed itself, so the report must not call that a fault: a false
# warning here is a false alarm in anything scripted on ./ri health's exit code.
health="$("${REPO_ROOT}/ri" health "${OUT}" 2>&1)" && health_status=0 || health_status=$?
[ "${health_status}" -eq 0 ] || fail "./ri health warns about a run that healed itself (exit ${health_status}):
${health}"
grep -q "self-healed restart" <<<"${health}" || fail "./ri health does not report the restart:
${health}"

echo "self-heal: killed at ${before} evaluations, recovered and finished at ${after}"

# Scenario two: the run does not die, it stops answering. SIGSTOP on one rank
# is the real thing - the rank is alive and PolyChord is mid-collective, so
# every other rank blocks in the Fortran caller forever. Nothing exits, so
# run_with_retries is never reached and `./ri health` sees a live run. Only
# _ns_stall_watchdog notices, and all it has to go on is that no evaluation is
# finishing.
#
# --stall-timeout 20 with a 2s poll rather than the shipped 7200s/60s, so the
# check costs a minute instead of two hours; the code path is the same one.
# Any value has to clear this search's own gaps, which are milliseconds.
echo "self-heal: starting a second wsclean search in ${HUNG_OUT}"
NS_STALL_POLL_SECONDS=2 "${REPO_ROOT}/ri" search wsclean \
  --nlive 20 --num-repeats 2 --mpi-procs 3 --retries 1 --no-build \
  --stall-timeout 20 --output-dir "${HUNG_OUT}" >"${HUNG_OUT}.log" 2>&1 &
SEARCH_PID=$!

for _ in $(seq 1 120); do
  [ "$(completed_evals "${HUNG_OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null \
    || fail "second search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${HUNG_OUT}.log"
  sleep 0.5
done
hung_before="$(completed_evals "${HUNG_OUT}")"
[ "${hung_before}" -ge "${KILL_AFTER_EVALS}" ] \
  || fail "only ${hung_before} evaluations after 60s; see ${HUNG_OUT}.log"

# One rank, not all of them: freezing every rank would also be recovered by a
# watchdog that only looked at whether the ranks were running. What has to be
# caught here is a job that is fully alive and simply not progressing.
frozen="$(pgrep -f "polychord_wsclean.py --output-dir ${HUNG_OUT}" | tail -1)"
[ -n "${frozen}" ] || fail "no rank of the second search to freeze"
echo "self-heal: freezing rank ${frozen} at ${hung_before} evaluations"
kill -STOP "${frozen}"

wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "the hung run was never noticed within ${RECOVER_TIMEOUT_SECONDS}s - the stall watchdog did not fire; see ${HUNG_OUT}.log"
wait "${SEARCH_PID}" && hung_status=0 || hung_status=$?
SEARCH_PID=""
[ "${hung_status}" -eq 0 ] || fail "the hung search did not recover (exit ${hung_status}); see ${HUNG_OUT}.log"

[ -f "${HUNG_OUT}/summary.json" ] || fail "hung run finished with no summary.json"
grep -q "no evaluation has finished" "${HUNG_OUT}/run.log" \
  || fail "run.log does not say the run was killed for landing nothing - see ${HUNG_OUT}/run.log"
grep -q "exit 137" "${HUNG_OUT}/restarts.log" 2>/dev/null \
  || fail "restarts.log did not record the stall kill: $(cat "${HUNG_OUT}/restarts.log" 2>/dev/null)"
hung_after="$(completed_evals "${HUNG_OUT}")"
[ "${hung_after}" -ge "${hung_before}" ] \
  || fail "stall restart lost work: ${hung_before} evaluations before, ${hung_after} after"
health="$("${REPO_ROOT}/ri" health "${HUNG_OUT}" 2>&1)" && health_status=0 || health_status=$?
[ "${health_status}" -eq 0 ] || fail "./ri health warns about a run that healed itself (exit ${health_status}):
${health}"

PASSED=1
echo "self-heal: hung at ${hung_before} evaluations, recovered and finished at ${hung_after}"
echo "self-heal check passed"
