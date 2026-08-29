#!/usr/bin/env bash
# End-to-end check that a search which is killed mid-flight heals itself.
#
# The robustness property this repo depends on most: a multi-day R2D2 search
# that loses its ranks at hour three has to come back on its own, because
# nothing else notices for hours. `run_with_retries` implements it and its own
# --self-check covers the decision with fixtures - but every bug found in this
# machinery so far was invisible to a fixture and obvious to a real kill. So:
# start a real search, break it, and assert it finishes anyway.
#
# Six breaks, because they recover through different machinery. Each is
# explained where it is performed below; docs/robustness.md has the whole
# story. In short: a SIGKILL is the crash the retry loop was written for; a
# SIGSTOPped rank is the hang it cannot see, which the stall watchdog turns
# into a crash; a kill with the retry budget spent is the one that does *not*
# heal, and has to hand a human a `./ri resume` that works; killed workers are
# supposed to cost nothing at all; a removed sidecar container needs
# `sidecar_restore` before a retry has anywhere to land; and a truncated
# checkpoint has to be moved aside rather than read again forever.
#
# ~5 minutes and ~0.6GB, on throwaway output directories that `./ri runs` and
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
WORKER_OUT="${REPO_ROOT}/results/.self-heal-worker-check-$$"
SIDECAR_OUT="${REPO_ROOT}/results/.self-heal-sidecar-check-$$"
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
  pkill -9 -f "polychord_wsclean.py --output-dir ${WORKER_OUT}" 2>/dev/null || true
  pkill -9 -f "polychord_wsclean.py --output-dir ${SIDECAR_OUT}" 2>/dev/null || true
  if [ -n "${PASSED}" ]; then
    rm -rf "${OUT}" "${OUT}.log" "${HUNG_OUT}" "${HUNG_OUT}.log" \
      "${RESUME_OUT}" "${RESUME_OUT}.log" "${WORKER_OUT}" "${WORKER_OUT}.log" \
      "${SIDECAR_OUT}" "${SIDECAR_OUT}.log" "${SIDECAR_OUT}.torn.log" \
      "${SIDECAR_OUT}.summary.log"
  else
    echo "self-heal: left ${OUT}*, ${HUNG_OUT}*, ${RESUME_OUT}*, ${WORKER_OUT}* and ${SIDECAR_OUT}* for inspection" >&2
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
# Polled every 0.05s rather than every second: a 3-rank search runs at ~25
# evaluations a second, so a one-second tick overshot the 8 this wants and
# landed past PolyChord's first checkpoint at 20 - the drift the assertion
# below is there to catch. Throughput only goes up, so poll finer than the
# window is wide.
for _ in $(seq 1 2400); do
  [ "$(completed_evals "${OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null || fail "search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${OUT}.log"
  sleep 0.05
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

# One of its records is zeroed out first, which is what a rank killed in the
# middle of writing a metrics.json leaves behind (and what a full disk leaves).
# That one file used to end a search for good: json.loads raised at startup in
# the resume and in every restart, all before scoring anything, so nothing
# retried. The resume below has to skip it and finish anyway.
corrupt_record="$(find "${RESUME_OUT}/evaluations" -maxdepth 2 -name metrics.json | sort | head -1)"
[ -n "${corrupt_record}" ] || fail "the killed run scored nothing to corrupt"
: >"${corrupt_record}"

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
# Taken from the report and run verbatim, rather than compared against a
# spelling this check believes in. `./ri health` gives a run under
# results/nested-sampling/ its bare name and any other run - these throwaway
# directories included - its path, and asserting either one here only says
# which branch was taken. What matters is that the command a human is handed
# is one `./ri resume` accepts, and the only way to know that is to type it.
resume_advice="$(grep -o './ri resume [^ ]*' <<<"${health}" | head -1)"
[ -n "${resume_advice}" ] \
  || fail "./ri health does not offer ./ri resume on a stopped run:
${health}"

# `./ri resume` reads run.env for every setting the run was started with and
# re-clamps only the rank count, so this also covers that file being written,
# sourceable, and complete enough to finish a search from.
echo "self-heal: resuming by hand with what the report offered: ${resume_advice}"
"${REPO_ROOT}/ri" resume "${resume_advice##* }" >>"${RESUME_OUT}.log" 2>&1 &
SEARCH_PID=$!
wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "./ri resume neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s - a rank is stuck in an MPI collective; see ${RESUME_OUT}.log"
wait "${SEARCH_PID}" && resumed_status=0 || resumed_status=$?
SEARCH_PID=""
[ "${resumed_status}" -eq 0 ] \
  || fail "./ri resume did not finish the run (exit ${resumed_status}); see ${RESUME_OUT}.log"
[ -f "${RESUME_OUT}/summary.json" ] || fail "the resumed run finished with no summary.json"
grep -qF "WARNING: ignoring unreadable ${corrupt_record}" "${RESUME_OUT}.log" \
  || fail "the resume did not say it had skipped the half-written ${corrupt_record}; see ${RESUME_OUT}.log"
[ -e "${corrupt_record%/metrics.json}" ] \
  && fail "the half-written record's directory survived the resume, so the sampler collides with it when it proposes that point again"

# Continued, not restarted. The count the resume script prints is the scored
# evaluations - the same `eval-*/metrics.json` rule `completed_evals` uses -
# read after the kill, so it can only be >= what was counted before it. A
# resume that ignored the previous attempt would print 0 and re-image
# everything.
adopted="$(grep -o '[0-9]\+ evaluations already done' "${RESUME_OUT}.log" | head -1 | cut -d' ' -f1)"
if [ -z "${adopted}" ] || [ "${adopted}" -lt "${resume_before}" ]; then
  fail "./ri resume started from '${adopted:-nothing}' rather than the ${resume_before} evaluations already on disk; see ${RESUME_OUT}.log"
fi
resume_after="$(completed_evals "${RESUME_OUT}")"
[ "${resume_after}" -ge "${resume_before}" ] \
  || fail "the resume lost work: ${resume_before} evaluations before the kill, ${resume_after} after"

# The stop between the kill and the resume is downtime, and every rate `./ri
# health` prints is measured over the time the run was running - so the resume
# has to have recorded itself in restarts.log, and the report has to say a
# human continued this run rather than counting it as a self-healed restart.
grep -q 'Z resumed at [0-9]\+ evaluations$' "${RESUME_OUT}/restarts.log" 2>/dev/null \
  || fail "./ri resume left no record of the downtime: $(cat "${RESUME_OUT}/restarts.log" 2>/dev/null)"
resumed_report="$("${REPO_ROOT}/ri" health "${RESUME_OUT}" 2>&1 || true)"
grep -q '^  resumes   1 manual resume,' <<<"${resumed_report}" \
  || fail "./ri health does not report the manual resume:
${resumed_report}"
grep -q 'self-healed restart' <<<"${resumed_report}" \
  && fail "./ri health called a resume someone typed a self-healed restart:
${resumed_report}"

# And now it is finished, so the advice stops: resuming a second time is a
# no-op rather than a second job over the same chains.
again="$("${REPO_ROOT}/ri" resume "${RESUME_OUT}" 2>&1)" \
  || fail "./ri resume on a finished run must succeed and do nothing, got: ${again}"
grep -qF "already finished" <<<"${again}" \
  || fail "./ri resume re-ran a finished run instead of saying so: ${again}"

echo "self-heal: killed unretried at ${resume_before} evaluations, resumed by hand and finished at ${resume_after}"

# Scenario four: nothing about the run is killed - one of its workers is, while
# the rank that owns it is between evaluations. That is the host's OOM killer's
# usual victim (the biggest resident process on a shared box is always an
# imager worker, and an idle R2D2 one still holds ~3.4GB), and common.py's
# retry loops exist to absorb it inside the evaluation. So this is the one
# scenario where a *restart* is the failure: the run has to finish with
# restarts.log never written.
echo "self-heal: starting a fourth wsclean search in ${WORKER_OUT}"
"${REPO_ROOT}/ri" search wsclean \
  --nlive 20 --num-repeats 2 --mpi-procs 3 --retries 1 --no-build \
  --output-dir "${WORKER_OUT}" >"${WORKER_OUT}.log" 2>&1 &
SEARCH_PID=$!

for _ in $(seq 1 120); do
  [ "$(completed_evals "${WORKER_OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null || fail "fourth search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${WORKER_OUT}.log"
  sleep 1
done
worker_before="$(completed_evals "${WORKER_OUT}")"
[ "${worker_before}" -ge "${KILL_AFTER_EVALS}" ] \
  || fail "only ${worker_before} evaluations after 120s; see ${WORKER_OUT}.log"

# Found by this run's own `ri.run-dir` label, so no other search on this host
# can be caught in it - and killed from inside the container, because the
# workers are the `sh` processes on the far end of each rank's `docker exec`
# and there is nothing on the host to signal but the client. Every rank's
# worker at once: which of the three is idle at this instant is a race, and a
# rank killed mid-request already had a covered path (its reply never comes).
worker_sidecar="$(docker ps --filter "label=ri.run-dir=${WORKER_OUT}" \
  --format '{{.Names}}\t{{.Image}}' | grep -i wsclean | cut -f1)"
[ -n "${worker_sidecar}" ] || fail "no wsclean sidecar labelled for ${WORKER_OUT}"
echo "self-heal: killing every wsclean worker in ${worker_sidecar} at ${worker_before} evaluations"
# /proc rather than pkill: the wsclean image ships no procps. `$$` is this
# `sh`, which is itself named sh and would otherwise kill itself first.
# `wsclean-zygote` as well as `sh`: since the fork server landed
# (docs/nested-sampling-wsclean-zygote.md) the imaging worker is a zygote and
# not a shell, so matching only `sh` quietly made this check kill nothing.
docker exec "${worker_sidecar}" sh -c \
  'killed=0
   for d in /proc/[0-9]*; do
     p="${d#/proc/}"
     comm="$(cat "${d}/comm" 2>/dev/null)"
     case "${comm}" in sh|wsclean-zygote) ;; *) continue ;; esac
     [ "${p}" != "$$" ] || continue
     kill -9 "${p}" && killed=$((killed+1))
   done
   [ "${killed}" -gt 0 ] || { echo "no workers to kill" >&2; exit 1; }' \
  || fail "could not kill the workers in ${worker_sidecar}"

wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "the search neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s of losing its workers; see ${WORKER_OUT}.log"
wait "${SEARCH_PID}" && worker_status=0 || worker_status=$?
SEARCH_PID=""
[ "${worker_status}" -eq 0 ] \
  || fail "the search did not survive losing its workers (exit ${worker_status}); see ${WORKER_OUT}.log"
[ -f "${WORKER_OUT}/summary.json" ] || fail "run finished with no summary.json"
# The whole point. A restart here means the death escaped the evaluation and
# aborted the job, which is what it did before worker_send in common.py.
[ -e "${WORKER_OUT}/restarts.log" ] \
  && fail "losing a worker cost a whole restart instead of being retried in place: $(cat "${WORKER_OUT}/restarts.log")"
worker_after="$(completed_evals "${WORKER_OUT}")"
[ "${worker_after}" -gt "${worker_before}" ] \
  || fail "no evaluation completed after the workers were killed (${worker_before} then ${worker_after})"

echo "self-heal: workers killed at ${worker_before} evaluations, retried in place and finished at ${worker_after}"

# Scenario five: the container the workers live in is removed while the run is
# using it. Unlike scenario four this cannot be absorbed inside the evaluation -
# there is nowhere to start a replacement worker - so the run does die and does
# spend a restart. What is being checked is that the restart lands somewhere:
# before `sidecar_restore`, every attempt after the removal `docker exec`ed
# into a name that no longer existed and scored nothing, so the run stopped for
# good at exit 1 with no summary.json.
echo "self-heal: starting a fifth wsclean search in ${SIDECAR_OUT}"
"${REPO_ROOT}/ri" search wsclean \
  --nlive 20 --num-repeats 2 --mpi-procs 3 --retries 1 --no-build \
  --output-dir "${SIDECAR_OUT}" >"${SIDECAR_OUT}.log" 2>&1 &
SEARCH_PID=$!

for _ in $(seq 1 120); do
  [ "$(completed_evals "${SIDECAR_OUT}")" -ge "${KILL_AFTER_EVALS}" ] && break
  kill -0 "${SEARCH_PID}" 2>/dev/null || fail "fifth search exited before scoring ${KILL_AFTER_EVALS} evaluations; see ${SIDECAR_OUT}.log"
  sleep 1
done
sidecar_before="$(completed_evals "${SIDECAR_OUT}")"
[ "${sidecar_before}" -ge "${KILL_AFTER_EVALS}" ] \
  || fail "only ${sidecar_before} evaluations after 120s; see ${SIDECAR_OUT}.log"

# This run's own label again, so no other search on this host is touched.
sidecar_name="$(docker ps --filter "label=ri.run-dir=${SIDECAR_OUT}" \
  --format '{{.Names}}\t{{.Image}}' | grep -i wsclean | cut -f1)"
[ -n "${sidecar_name}" ] || fail "no wsclean sidecar labelled for ${SIDECAR_OUT}"
echo "self-heal: removing the sidecar ${sidecar_name} at ${sidecar_before} evaluations"
docker rm --force "${sidecar_name}" >/dev/null || fail "could not remove ${sidecar_name}"

wait_for_exit "${SEARCH_PID}" "${RECOVER_TIMEOUT_SECONDS}" \
  || fail "the search neither finished nor died within ${RECOVER_TIMEOUT_SECONDS}s of losing its sidecar; see ${SIDECAR_OUT}.log"
wait "${SEARCH_PID}" && sidecar_status=0 || sidecar_status=$?
SEARCH_PID=""
[ "${sidecar_status}" -eq 0 ] \
  || fail "the search did not survive losing its sidecar (exit ${sidecar_status}); see ${SIDECAR_OUT}.log"
[ -f "${SIDECAR_OUT}/summary.json" ] || fail "run finished with no summary.json"
# Asserted as well as the finish, because a run that happened to be told to
# stop by something else would also produce a summary: this is the line that
# says the missing container is what came back. In run.log rather than the
# terminal capture, because run.log is the artifact a stopped run is diagnosed
# from and the restore is part of why it restarted.
grep -q "sidecar_restore: .* starting it again" "${SIDECAR_OUT}/run.log" \
  || fail "nothing restarted the removed sidecar; see ${SIDECAR_OUT}/run.log"
sidecar_after="$(completed_evals "${SIDECAR_OUT}")"
[ "${sidecar_after}" -gt "${sidecar_before}" ] \
  || fail "no evaluation completed after the sidecar was removed (${sidecar_before} then ${sidecar_after})"

echo "self-heal: sidecar removed at ${sidecar_before} evaluations, started again and finished at ${sidecar_after}"

# Scenario six: the checkpoint itself is what is broken. A rank killed part-way
# through writing `chains/*.resume` leaves a truncated file, and PolyChord
# aborts reading it in Fortran before evaluation 1 - so the run scored nothing,
# `run_with_retries`' anti-spin guard refused to restart it, and every later
# `./ri resume` died in the identical place with every scored evaluation
# unreachable on disk. The recovery is to move the checkpoint aside and let the
# sampler start over, which replays those evaluations out of the point cache
# without imaging any of them.
#
# On the run scenario five just finished rather than a sixth search: the break
# is deterministic (truncate a file) and needs no kill timing, so the only
# thing a fresh run would add is another minute. Deleting summary.json is what
# the resume script itself documents as the way to make a finished run
# resumable.
resume_file="$(find "${SIDECAR_OUT}/chains" -maxdepth 1 -name '*.resume' | head -1)"
[ -n "${resume_file}" ] || fail "the finished run left no checkpoint to tear; see ${SIDECAR_OUT}/chains"
torn_before="$(completed_evals "${SIDECAR_OUT}")"
rm -f "${SIDECAR_OUT}/summary.json"
truncate -s "$(( $(wc -c <"${resume_file}") / 2 ))" "${resume_file}"
echo "self-heal: truncating ${resume_file##*/} and resuming at ${torn_before} evaluations"
"${REPO_ROOT}/ri" resume "${SIDECAR_OUT}" --no-build >"${SIDECAR_OUT}.torn.log" 2>&1 \
  || fail "an unreadable checkpoint stopped the run for good; see ${SIDECAR_OUT}.torn.log"
[ -f "${SIDECAR_OUT}/summary.json" ] || fail "the resume finished with no summary.json"
# The checkpoint was moved, not deleted: it is the only record of where the
# sampler had reached, and unreadable to PolyChord is not unreadable to a human.
[ -f "${resume_file}.unreadable" ] \
  || fail "the torn checkpoint was not kept as evidence; see ${SIDECAR_OUT}/chains"
# Started over rather than resumed, which is the whole point of moving the file.
grep -q 'no checkpoint to resume from, re-sampling from the cache' "${SIDECAR_OUT}/run.log" \
  || fail "the retry read the torn checkpoint again; see ${SIDECAR_OUT}/run.log"
# And the work already on disk survived it. Equal, not greater: every point the
# restart drew came back out of the cache, so nothing was imaged twice.
torn_after="$(completed_evals "${SIDECAR_OUT}")"
[ "${torn_after}" -ge "${torn_before}" ] \
  || fail "the checkpoint recovery lost work: ${torn_before} evaluations before, ${torn_after} after"

echo "self-heal: torn checkpoint at ${torn_before} evaluations, sampler restarted and finished at ${torn_after}"

# Scenario seven: the summary is what is broken. summary.json is written once,
# after PolyChord returns, and carries every evaluation of the run - seconds of
# writing on a long R2D2 search - so a rank killed there leaves half a file.
# Every reader called a run with a summary.json finished, which made that the
# worst case of all: `./ri health` said FINISHED, the report died on it, and
# `./ri resume` - the only command that can rewrite it - refused. Again on the
# run scenario six just finished, because tearing a file needs no kill timing.
summary_before="$(completed_evals "${SIDECAR_OUT}")"
head -c "$(( $(wc -c <"${SIDECAR_OUT}/summary.json") / 2 ))" \
  "${SIDECAR_OUT}/summary.json" >"${SIDECAR_OUT}/summary.half"
mv "${SIDECAR_OUT}/summary.half" "${SIDECAR_OUT}/summary.json"
echo "self-heal: tearing summary.json at ${summary_before} evaluations"
summary_health="$("${REPO_ROOT}/ri" health "${SIDECAR_OUT}" 2>&1 || true)"
case "${summary_health}" in
  *"summary.json is half written"*"./ri resume"*) ;;
  *) fail "./ri health called a torn summary a finished run: ${summary_health}" ;;
esac
"${REPO_ROOT}/ri" resume "${SIDECAR_OUT}" --no-build >"${SIDECAR_OUT}.summary.log" 2>&1 \
  || fail "a torn summary.json could not be repaired; see ${SIDECAR_OUT}.summary.log"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "${SIDECAR_OUT}/summary.json" \
  || fail "the resume left a summary.json that still does not parse"
# Rewritten from the evaluations on disk, so none of them was imaged again.
summary_after="$(completed_evals "${SIDECAR_OUT}")"
[ "${summary_after}" -eq "${summary_before}" ] \
  || fail "repairing the summary changed the evaluations: ${summary_before} then ${summary_after}"

echo "self-heal: torn summary.json rewritten at ${summary_after} evaluations"

PASSED=1
echo "self-heal check passed"
