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
# Three ways of breaking it, because they recover through different machinery.
# A SIGKILL is the crash `run_with_retries` was written for - the run exits and
# the loop sees a status. A frozen rank is the failure it cannot see: PolyChord
# calls the likelihood from Fortran, so one rank that stops answering leaves
# every other rank in a collective forever, with no exit for anything to act
# on. `_ns_stall_watchdog` in scripts/lib/progress-bar.sh is what turns the
# second into the first, and SIGSTOP on one rank reproduces it exactly. The
# third is the one that does not heal: a run killed with its retry budget
# already spent, which is where `./ri health` stops reporting and starts
# telling a human to type `./ri resume`. That advice is the last line of
# defence for a multi-day search - both of `./ri health`'s warnings about a run
# nothing is driving end on it, and so does every message `run_with_retries`
# gives up with - and nothing joined it to a real interrupted run either.
#
# ~3 minutes and ~0.6GB, on throwaway output directories that `./ri runs` and
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
RESUME_OUT="${REPO_ROOT}/results/.self-heal-resume-check-$$"
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
  pkill -9 -f "polychord_wsclean.py --output-dir ${RESUME_OUT}" 2>/dev/null || true
  if [ -n "${PASSED}" ]; then
    rm -rf "${OUT}" "${OUT}.log" "${HUNG_OUT}" "${HUNG_OUT}.log" \
      "${RESUME_OUT}" "${RESUME_OUT}.log"
  else
    echo "self-heal: left ${OUT}*, ${HUNG_OUT}* and ${RESUME_OUT}* for inspection" >&2
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

# The run healed itself, so the report must not call that a fault. The run's
# own headline, not the exit status: that is 1 for a host warning too, and this
# host is shared - the same race the hang scenario below documents caught this
# assertion on the search that had just finished one line above, whose three
# sidecars are still running for the ~0.4s `_sidecar_remove` takes to remove
# them in the background after their launcher pid is gone.
health="$("${REPO_ROOT}/ri" health "${OUT}" 2>&1)" || true
case "$(printf '%s\n' "${health}" | head -1)" in
  *WARNING*) fail "./ri health warns about a run that healed itself:
${health}" ;;
esac
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
# The run's own headline, not the exit status: that is 1 for a host warning
# too, and the host is shared. This check failed on "3 sidecar container(s)
# outlived the run that started them" naming the containers of the search that
# had just finished one line above - `_sidecar_remove` backgrounds its
# `docker rm --force`, so for ~0.4s after a run exits its containers are
# running with a launcher pid that is already gone. Another session's leaked
# sidecar, or a host low on memory, failed it just as easily. What this check
# is about is whether the *run* looks healthy after healing itself.
health="$("${REPO_ROOT}/ri" health "${HUNG_OUT}" 2>&1)" || true
case "$(printf '%s\n' "${health}" | head -1)" in
  *WARNING*) fail "./ri health warns about a run that healed itself:
${health}" ;;
esac

echo "self-heal: hung at ${hung_before} evaluations, recovered and finished at ${hung_after}"

# Scenario three: the run does not heal itself, and a human puts it back. Every
# route into `./ri resume` that `./ri health` prints - the STOPPED warning, the
# orphaned-launcher warning, all three "not retrying" messages
# `run_with_retries` gives up with - is a promise that a search interrupted hours in can be continued
# rather than restarted, and none of it was checked against a real one. The two
# scenarios above cannot reach it: they finish on their own, and a finished run
# is the one thing `./ri resume` refuses.
#
# --retries 0, so the SIGKILL is final. That is also exactly what a run looks
# like once it has spent its retry budget on the fault that keeps coming back.
echo "self-heal: starting a third wsclean search in ${RESUME_OUT}"
"${REPO_ROOT}/ri" search wsclean \
  --nlive 20 --num-repeats 2 --mpi-procs 3 --retries 0 --no-build \
  --output-dir "${RESUME_OUT}" >"${RESUME_OUT}.log" 2>&1 &
SEARCH_PID=$!

for _ in $(seq 1 120); do
  [ "$(completed_evals "${RESUME_OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null \
    || fail "third search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${RESUME_OUT}.log"
  sleep 0.5
done
resume_before="$(completed_evals "${RESUME_OUT}")"
[ "${resume_before}" -ge "${KILL_AFTER_EVALS}" ] \
  || fail "only ${resume_before} evaluations after 60s; see ${RESUME_OUT}.log"

echo "self-heal: killing the third sampler at ${resume_before} evaluations, with no retries left"
pkill -9 -f "polychord_wsclean.py --output-dir ${RESUME_OUT}" || fail "no third sampler to kill"
wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "the un-retried run neither exited nor was noticed within ${RECOVER_TIMEOUT_SECONDS}s; see ${RESUME_OUT}.log"
wait "${SEARCH_PID}" && resume_status=0 || resume_status=$?
SEARCH_PID=""
[ "${resume_status}" -eq 0 ] \
  && fail "a search killed with no retries left must not exit 0; see ${RESUME_OUT}.log"
[ -e "${RESUME_OUT}/summary.json" ] \
  && fail "the killed search wrote a summary.json, so there would be nothing to resume"

# The report is how anyone finds out, so it is asserted before the resume: a
# stopped run has to headline STOPPED and name the command that continues it.
# Not the exit status - that is 1 for a host warning too, on a host this check
# shares with other searches.
health="$("${REPO_ROOT}/ri" health "${RESUME_OUT}" 2>&1)" || true
case "$(printf '%s\n' "${health}" | head -1)" in
  *STOPPED*) ;;
  *) fail "./ri health does not headline the killed run STOPPED:
${health}" ;;
esac
grep -qF "./ri resume ${RESUME_OUT##*/}" <<<"${health}" \
  || fail "./ri health does not offer ./ri resume on a stopped run:
${health}"

# `./ri resume` reads run.env for every setting the run was started with and
# re-clamps only the rank count, so this also covers that file being written,
# sourceable, and complete enough to finish a search from.
echo "self-heal: resuming ${RESUME_OUT##*/} by hand"
"${REPO_ROOT}/ri" resume "${RESUME_OUT}" >>"${RESUME_OUT}.log" 2>&1 &
SEARCH_PID=$!
wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "./ri resume neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s - a rank is stuck in an MPI collective; see ${RESUME_OUT}.log"
wait "${SEARCH_PID}" && resumed_status=0 || resumed_status=$?
SEARCH_PID=""
[ "${resumed_status}" -eq 0 ] \
  || fail "./ri resume did not finish the run (exit ${resumed_status}); see ${RESUME_OUT}.log"
[ -f "${RESUME_OUT}/summary.json" ] || fail "the resumed run finished with no summary.json"

# Continued, not restarted. The count the resume script prints is evaluation
# *directories*, which is the completed ones plus any that were in flight when
# the kill landed, so it can only be >= what completed - a resume that ignored
# the previous attempt would print 0 and re-image everything.
adopted="$(grep -o '[0-9]\+ evaluations already done' "${RESUME_OUT}.log" | head -1 | cut -d' ' -f1)"
[ -n "${adopted}" ] && [ "${adopted}" -ge "${resume_before}" ] \
  || fail "./ri resume started from '${adopted:-nothing}' rather than the ${resume_before} evaluations already on disk; see ${RESUME_OUT}.log"
resume_after="$(completed_evals "${RESUME_OUT}")"
[ "${resume_after}" -ge "${resume_before}" ] \
  || fail "the resume lost work: ${resume_before} evaluations before the kill, ${resume_after} after"

# And now it is finished, so the advice stops: resuming a second time is a
# no-op rather than a second job over the same chains.
again="$("${REPO_ROOT}/ri" resume "${RESUME_OUT}" 2>&1)" \
  || fail "./ri resume on a finished run must succeed and do nothing, got: ${again}"
grep -qF "already finished" <<<"${again}" \
  || fail "./ri resume re-ran a finished run instead of saying so: ${again}"

echo "self-heal: killed unretried at ${resume_before} evaluations, resumed by hand and finished at ${resume_after}"

PASSED=1
echo "self-heal check passed"
