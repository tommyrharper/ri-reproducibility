#!/usr/bin/env bash
# A pinned status line for a nested-sampling search: elapsed time, dead points
# with a percent and an ETA extrapolated from the rate so far, plus the raw
# evaluation count. The denominator is the run's --max-ndead cap when it has
# one and, when it does not, the total its own evidence implies - the same
# estimate `./ri health` reports, marked `~` so the two cannot be confused.
#
# Dead points come from PolyChord's own chains/*_dead-birth.txt (one line per
# dead point) - the same file scripts/lib/nested_sampling/anesthetic_io.py
# already reads for finished runs. evaluations/eval-* is *not* the same
# count: PolyChord's slice sampler makes several likelihood evaluations per
# accepted dead point (roughly num_repeats per dimension), so it always runs
# ahead - shown separately, not conflated with dead points.
#
# On a TTY the status is pinned to the terminal's last line via a scroll
# region (DECSTBM), so PolyChord's own feedback scrolling by above it doesn't
# bury the line - the fix for a WSClean search producing dead points faster
# than a human can read a scrolling counter.

# Usage: run_with_progress <output_dir> <max_ndead> <nlive> -- cmd args...
run_with_progress() {
  local output_dir="$1" max_ndead="$2" nlive="$3"
  shift 3
  [ "${1:-}" = "--" ] && shift

  # Keep the run's own output. Everything a stopped run leaves on disk says
  # *that* it broke - empty imager directories, chains stuck at the initial
  # live points - and nothing says *why*: the traceback existed only in the
  # terminal it was started from, and diagnosing one meant starting the search
  # again under a redirect and waiting for it to happen twice.
  #
  # Teed rather than redirected, because the status line below needs this
  # shell's own stdout to still be a terminal: `> run.log 2>&1` around the
  # whole script makes `[ -t 1 ]` false and silently takes the progress bar
  # away. Only the command's output is diverted, and nothing downstream
  # notices - the `docker exec` this wraps is issued without `-t`, so the
  # container's stdout was already a pipe and its buffering does not change.
  #
  # Through a named pipe rather than `> >(tee ...)` so that tee has a pid that
  # can be waited for. The last thing a dying run writes is the whole point of
  # having the file, and a process substitution offers no way to know it was
  # flushed before the script moves on.
  local log="${output_dir}/run.log" pipe="${output_dir}/.run.log.fifo"
  rm -f "${pipe}"
  mkfifo "${pipe}"
  # Appended, not truncated: `./ri resume` re-runs this against a directory
  # that already has a log, and the first failure is usually the one to read.
  tee -a "${log}" <"${pipe}" &
  local tee_pid=$!
  "$@" >"${pipe}" 2>&1 &
  local pid=$!
  local start
  start="$(date +%s)"

  if [ -t 1 ] && _ns_pin_setup; then
    _ns_add_trap '_ns_pin_teardown' EXIT
    _ns_add_trap '_ns_pin_teardown' INT
    _ns_add_trap '_ns_pin_teardown' TERM
    while kill -0 "${pid}" 2>/dev/null; do
      _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")"
      sleep 1
    done
    _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")"
    _ns_pin_teardown
  elif [ -t 1 ]; then
    # No usable terminal control (e.g. tput/TERM missing): fall back to a
    # plain redrawn line instead of a pinned one.
    while kill -0 "${pid}" 2>/dev/null; do
      printf '\r%s' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")")"
      sleep 1
    done
    printf '\r%s\n' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")")"
  fi

  local status=0
  wait "${pid}" || status=$?
  # The command's last writer is now closed, so tee sees EOF. Waited for so
  # the function leaves no child behind and the file is known to be complete
  # before the caller reads it. In practice tee always drains first and the
  # self-check below cannot force it not to, so this is a guard against a race
  # rather than a fix for an observed one - kept because the lines at risk are
  # exactly the ones the file exists for.
  wait "${tee_pid}" 2>/dev/null || true
  rm -f "${pipe}"
  return "${status}"
}

# Usage: run_with_retries <retries> <output_dir> <max_ndead> <nlive> -- cmd args...
#
# The recovery half of the above: PolyChord checkpoints continuously, so a run
# that dies at hour three already has everything needed to carry on from where
# it stopped - what it did not have was anything to start it again. A dead
# worker (WORKER_DIED in common.py), a wedged meqserver that escaped the
# in-worker watchdog, an OOM kill: each of those ends a multi-day search that
# is then simply gone until a human notices and types `./ri resume`.
#
# Retried only while the failed attempt made forward progress, measured in
# completed evaluations. That is the whole guard against spinning: a code bug
# every rank hits deterministically, a bad parameter space, a missing image -
# all fail before a single evaluation is scored, so they stop immediately
# rather than failing three times as slowly. Something that killed a run which
# was working is the only thing that gets another go.
#
# Evaluations rather than the dead points this used to count, because
# PolyChord writes chains/ only every `nlive` dead points: a crash inside that
# interval - up to seventy minutes of imaging on the 16-rank R2D2 search, and
# every crash before the first checkpoint of a fresh run - leaves the
# dead-point count at exactly the number the attempt started from, so the
# guard fired on runs that had been working for an hour and called them
# deterministic. Reproduced with a real search killed at 31 scored
# evaluations and 0 dead points: it refused to retry. Nor is that work lost
# by retrying - the retry adopts the finished evaluations and serves those
# points from its cache (adopt_completed_evaluations in common.py).
#
# The retry reuses the sidecar containers this run already started, but not
# their pooled workers: those exit on EOF when the dying ranks close their end
# of the FIFOs. Each rank then waits out `_connect_shell_started_worker`'s
# 10s deadline in common.py and starts its own worker inside the same sidecar -
# still one long-lived worker per rank, so the cost is that one-off wait and
# not a per-evaluation penalty. Measured on a real killed WSClean search: 216
# evaluations/min over the 53 before the kill against 219/min over the 34
# after, with a 12.1s gap across the restart. Deliberate at that price - the
# alternative is re-launching the pool into a container whose old workers may
# not all have exited yet, and two readers on one FIFO split the messages
# between them.
run_with_retries() {
  local retries="$1" output_dir="$2"
  shift
  local attempt=0 status=0 before after
  while :; do
    before="$(_ns_completed_evals "${output_dir}")"
    status=0
    run_with_progress "$@" || status=$?
    [ "${status}" -eq 0 ] && return 0
    after="$(_ns_completed_evals "${output_dir}")"
    if [ "${attempt}" -ge "${retries}" ]; then
      break
    fi
    if [ "${after}" -le "${before}" ]; then
      _ns_retry_say "${output_dir}" \
        "not retrying: the attempt that just failed (exit ${status}) scored no evaluations," \
        "so another one fails the same way. Why it stopped is above;" \
        "./ri resume ${output_dir##*/} tries again anyway."
      break
    fi
    attempt=$((attempt + 1))
    # An index of the restarts, next to run.log which holds the tracebacks
    # themselves. Its own file because `./ri health` wants the count of a run
    # that has been going for a day, and run.log by then is megabytes of
    # PolyChord feedback with the restart lines scattered through it.
    printf '%s exit %s after %s evaluations\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${after}" \
      >>"${output_dir}/restarts.log"
    _ns_retry_say "${output_dir}" \
      "attempt failed (exit ${status}) at ${after} evaluations; resuming from PolyChord's" \
      "checkpoint - retry ${attempt} of ${retries}. Why it stopped is above."
  done
  return "${status}"
}

# Both to the terminal and into run.log, because run.log is the only artifact
# that records why a run stopped and a restart is part of that story - `./ri
# health` reads these lines back to say a live run has healed itself.
_ns_retry_say() {
  local output_dir="$1"
  shift
  echo "run_with_retries: $*" | tee -a "${output_dir}/run.log" >&2
}

# Evaluations this run has finished, across every attempt. metrics.json
# rather than the directory holding it, because a directory without one is an
# evaluation that was in flight when the run died and the next attempt deletes
# those before it starts - so this counts work that survived, and cannot be
# inflated by the handful of directories a crash leaves half-written.
_ns_completed_evals() {
  find "$1/evaluations" -maxdepth 2 -name metrics.json 2>/dev/null | wc -l | tr -d ' '
}

# Appends a command to whatever trap is already registered for a signal
# instead of replacing it - start-sidecars.sh's cleanup trap on EXIT/INT/TERM
# must keep running (a leftover R2D2 sidecar holds ~33.7GB), so ours must not
# clobber it. New command runs first: INT/TERM's existing handler ends in
# `exit N`, which would skip anything appended after it.
_ns_add_trap() {
  local new="$1" sig="$2" line existing=""
  line="$(trap -p "${sig}")"
  if [ -n "${line}" ]; then
    # `trap -p` prints valid shell source for re-registering the trap
    # (`trap -- 'cmd' SIG`), quoted so embedded quotes in the existing
    # command survive - letting bash's own parser split it back into words
    # is what correctly reverses that quoting; substring-stripping the
    # 'trap -- ' prefix textually breaks the moment the command itself
    # contains a quote.
    eval "set -- ${line}"
    existing="$3"
  fi
  # shellcheck disable=SC2064  # expanding now is the point: the trap
  # must carry the commands as they are, not re-read these locals later
  if [ -n "${existing}" ]; then
    trap "${new}; ${existing}" "${sig}"
  else
    trap "${new}" "${sig}"
  fi
}

_ns_pin_active=0

# ponytail: scroll region is sized once at setup and not redone on SIGWINCH,
# so resizing the terminal mid-run can leave the reserved line mis-positioned
# until the next run. Fix by trapping WINCH if that ever bites someone.
_ns_pin_setup() {
  local rows
  rows="$(tput lines 2>/dev/null)" || return 1
  [ -n "${rows}" ] && [ "${rows}" -gt 1 ] || return 1
  printf '\n'
  printf '\e7\e[1;%dr\e8' "$((rows - 1))"
  _ns_pin_active=1
}

_ns_pin_teardown() {
  [ "${_ns_pin_active}" = "1" ] || return 0
  _ns_pin_active=0
  local rows
  rows="$(tput lines 2>/dev/null)" || return 0
  printf '\e[1;%dr' "${rows}"
  printf '\e[%d;1H\e[2K' "${rows}"
}

_ns_pin_draw() {
  local rows
  rows="$(tput lines 2>/dev/null)" || return 0
  printf '\e7\e[%d;1H\e[2K%s\e8' "${rows}" "$(_ns_truncate_pad "$1")"
}

# A status line wider than the terminal wraps onto the row above once
# printed - which sits inside the scroll region, not on the reserved line,
# and disappears the moment normal output scrolls past it. Confirmed by
# replaying a captured real run through a terminal emulator: the line was
# there in the byte stream but wrapped and then vanished on screen within a
# couple of seconds. Every draw is clamped to the terminal's actual width so
# it can never wrap, and padded so a shorter redraw doesn't leave the
# previous line's tail on screen.
_ns_truncate_pad() {
  local text="$1" cols width
  cols="$(tput cols 2>/dev/null)" || cols=80
  width=$((cols > 1 ? cols - 1 : cols))
  printf '%-*.*s' "${width}" "${width}" "${text}"
}

# "<evaluations> <of those, landed after `reference` was written>". `find`
# rather than the glob loop this replaced, which cost 283ms of every
# one-second redraw on a live 7,200-evaluation run - one find answering both
# questions costs ~100ms, so the second count is free and the bar got cheaper.
_ns_count_evals() {
  local dir="$1" reference="${2:-}" total since=0
  total="$(find "${dir}" -maxdepth 1 -name 'eval-*' 2>/dev/null | wc -l | tr -d ' ')"
  if [ -n "${reference}" ] && [ -e "${reference}" ]; then
    since="$(find "${dir}" -maxdepth 1 -name 'eval-*' -newer "${reference}" 2>/dev/null |
      wc -l | tr -d ' ')"
  fi
  echo "${total} ${since}"
}

# Dead points now, rather than at the last checkpoint. PolyChord writes
# chains/ every `nlive` dead points, so between writes the count is frozen by
# construction: on the 16-rank R2D2 search here that is two hours at a time,
# and a bar that sits still for two hours and then jumps fifty points is not a
# progress bar. Evaluation directories appear every few seconds instead, and
# the slice sampler spends a near-constant number of them per dead point
# (num_repeats times the dimension), so the evaluations banked since the
# checkpoint convert straight back into dead points. `banked` is the count as
# of the checkpoint, so the ratio is not contaminated by the evaluations being
# converted. Same estimate `./ri health` prints on its forecast line; carries
# a `~` wherever it is shown.
_ns_dead_now() {
  local dead="$1" since="$3" banked=$(($2 - $3))
  if [ "${dead}" -le 0 ] || [ "${banked}" -le 0 ] || [ "${since}" -le 0 ]; then
    echo "${dead}"
    return
  fi
  echo $((dead + (since * dead + banked / 2) / banked))
}

_ns_dead_birth_file() {
  local chains_dir="$1" f
  for f in "${chains_dir}"/*_dead-birth.txt; do
    [ -e "${f}" ] && { echo "${f}"; return; }
  done
}

_ns_count_lines() {
  local file="$1"
  [ -f "${file}" ] || { echo 0; return; }
  wc -l <"${file}" | tr -d ' '
}

_ns_format_hms() {
  local s="$1"
  printf '%d:%02d:%02d' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

_ns_render_bar() {
  local pct="$1" width="${2:-30}" filled
  filled=$((pct * width / 100))
  printf '%s%s' "$(printf '%*s' "${filled}" '' | tr ' ' '#')" "$(printf '%*s' $((width - filled)) '')"
}

_ns_status_line() {
  local output_dir="$1" max_ndead="$2" nlive="$3" start="$4"
  local now elapsed dead_file dead_count counts eval_count since dead_now

  now="$(date +%s)"
  elapsed=$((now - start))
  dead_file="$(_ns_dead_birth_file "${output_dir}/chains")"
  dead_count=0
  [ -n "${dead_file}" ] && dead_count="$(_ns_count_lines "${dead_file}")"
  counts="$(_ns_count_evals "${output_dir}/evaluations" "${dead_file}")"
  eval_count="${counts%% *}"
  since="${counts##* }"
  dead_now="$(_ns_dead_now "${dead_count}" "${eval_count}" "${since}")"

  # PolyChord treats max_ndead <= 0 as "no bound, stop on evidence tolerance
  # instead" - there's no dead-point budget to measure a percent or ETA
  # against, but the tolerance itself is a real number we can approximate
  # progress against instead (see _ns_evidence_total).
  if [ "${max_ndead}" -le 0 ]; then
    local total
    # The total is estimated from the checkpoint's own dead count, because the
    # log(Z) and live points it needs were written by that same checkpoint;
    # only the position within it is carried forward.
    if total="$(_ns_evidence_total "${output_dir}/chains" "${dead_count}" "${nlive}")"; then
      _ns_evidence_line "${total}" "${dead_now}" "${eval_count}" "${elapsed}"
    else
      _ns_unbounded_line "${dead_count}" "${eval_count}" "${elapsed}"
    fi
    return
  fi

  local pct eta bar carried=""
  [ "${dead_now}" -ne "${dead_count}" ] && carried="~"
  pct=$((dead_now * 100 / max_ndead))
  [ "${pct}" -gt 100 ] && pct=100

  eta="?"
  if [ "${dead_now}" -gt 0 ] && [ "$((max_ndead - dead_now))" -gt 0 ]; then
    eta="$(_ns_format_hms $(((max_ndead - dead_now) * elapsed / dead_now)))"
  elif [ "${dead_now}" -ge "${max_ndead}" ]; then
    eta="0:00:00"
  fi

  bar="$(_ns_render_bar "${pct}" 30)"

  # The cap is a number the operator asked for and stays plain; the position
  # within it is carried across the checkpoint interval, so it and everything
  # derived from it take the `~` whenever it is not the checkpoint's own count.
  printf '[%s] %s%3d%%  %s%d/%d dead points (%d evaluations)  elapsed %s  eta %s%s' \
    "${bar}" "${carried}" "${pct}" "${carried}" "${dead_now}" "${max_ndead}" \
    "${eval_count}" "$(_ns_format_hms "${elapsed}")" "${carried}" "${eta}"
}

# A block bouncing back and forth across the bar, one step per elapsed
# second, so a no-budget run still visibly ticks over instead of sitting
# frozen. Position, not percent: there is nothing to be a percent of. Used
# only until _ns_evidence_total has enough data to give a real number (before
# the first dead point, chains/*.stats and chains/*_phys_live.txt don't
# exist yet).
_ns_unbounded_line() {
  local dead_count="$1" eval_count="$2" elapsed="$3"
  local bar_width=30 period pos i
  period=$((2 * (bar_width - 1)))
  pos=$((elapsed % period))
  [ "${pos}" -ge "${bar_width}" ] && pos=$((period - pos))

  local bar=""
  for ((i = 0; i < bar_width; i++)); do
    bar+=$([ "${i}" -eq "${pos}" ] && echo "#" || echo " ")
  done

  local rate="-"
  [ "${elapsed}" -gt 0 ] && rate="$((dead_count * 60 / elapsed))/min"

  printf '[%s] %d dead points (%d evaluations)  rate %s  elapsed %s  (no --max-ndead cap, stopping on evidence tolerance)' \
    "${bar}" "${dead_count}" "${eval_count}" "${rate}" "$(_ns_format_hms "${elapsed}")"
}

# Every figure derived from the estimated total carries a `~`: the cap branch
# above divides by a number the operator asked for, this one divides by one
# the run's own evidence implies, and a bar that looks the same either way
# would hide that difference.
_ns_evidence_line() {
  local total="$1" dead_now="$2" eval_count="$3" elapsed="$4"
  local bar eta="?" pct
  pct=$((dead_now * 100 / total))
  [ "${pct}" -gt 100 ] && pct=100
  bar="$(_ns_render_bar "${pct}" 30)"
  if [ "$((total - dead_now))" -gt 0 ] && [ "${dead_now}" -gt 0 ]; then
    eta="$(_ns_format_hms $(((total - dead_now) * elapsed / dead_now)))"
  elif [ "${dead_now}" -ge "${total}" ]; then
    eta="0:00:00"
  fi
  printf '[%s] ~%3d%%  ~%d/~%d dead points (%d evaluations)  elapsed %s  eta ~%s  (evidence tolerance, no --max-ndead cap)' \
    "${bar}" "${pct}" "${dead_now}" "${total}" "${eval_count}" \
    "$(_ns_format_hms "${elapsed}")" "${eta}"
}

# The fraction of the banked evidence at which a run actually stops. Measured,
# not read out of PolyChord's documentation: its `precision_criterion` defaults
# to 1e-3 (source: PolyChordLite's nested_sampling.F90 - `live_logZ(...) <
# log(precision_criterion) + RTI%logZ`, not currently exposed as a flag here),
# but the two searches on this host that terminated naturally stopped at 1.3e-4
# and 9.6e-5, so 1e-3 predicted 350 dead points for runs that took 446 and 463.
#
# scripts/nested-sampling-health.py carries the same number as
# TERMINATION_EVIDENCE_RATIO and forecasts from the same model, so that the
# pinned status line and `./ri health` cannot report different progress for the
# same run. self_check below fails if the two copies drift apart.
_NS_TERMINATION_EVIDENCE_RATIO=1.2e-4

_ns_stats_file() {
  local chains_dir="$1" f
  for f in "${chains_dir}"/*.stats; do
    [ -e "${f}" ] && { echo "${f}"; return; }
  done
}

_ns_phys_live_file() {
  local chains_dir="$1" f
  for f in "${chains_dir}"/*_phys_live.txt; do
    [ -e "${f}" ] && { echo "${f}"; return; }
  done
}

# How many dead points an uncapped run (--max-ndead <= 0) will take in total,
# from the same output files ./ri report and
# scripts/lib/nested_sampling/anesthetic_io.py already read: chains/*.stats
# for the accumulated log(Z) and chains/*_phys_live.txt for the live points'
# current log-likelihoods (last column of each row). PolyChord's own
# live_logZ is logsumexp(live loglikes) - log(nlive) + logXp, where logXp
# (the remaining prior volume) is tracked per-cluster internally; this
# approximates it as a single global -ndead/nlive, the textbook single-
# cluster expectation. A run whose live points split across several clusters
# (chains/*.stats' `ncluster` line) will see this diverge further from
# PolyChord's own exact per-cluster figure than a single-cluster run does.
#
# A count of dead points, not the evidence ratio itself. The prior volume
# shrinks one e-fold per `nlive` dead points, so how much further the ratio
# has to fall converts to a count - and a count is what the rate so far can be
# extrapolated into a percent and an ETA against, the same way the --max-ndead
# branch does it. Dividing the criterion by the ratio instead measures progress
# in a quantity that falls exponentially, which read 3% on a live search that
# this model and `./ri health` both put at 38%.
_ns_evidence_total() {
  local chains_dir="$1" dead_count="$2" nlive="$3"
  local stats_file live_file logz
  [ "${dead_count}" -gt 0 ] && [ "${nlive}" -gt 0 ] || return 1
  # Inside the first e-fold the live set is still the prior, so the ratio is
  # ~1 and the estimate would be reporting nlive*ln(1/RATIO) - its own
  # constant, and nothing about this run. The bounce is more honest.
  [ "${dead_count}" -ge "${nlive}" ] || return 1
  stats_file="$(_ns_stats_file "${chains_dir}")"
  live_file="$(_ns_phys_live_file "${chains_dir}")"
  [ -n "${stats_file}" ] && [ -n "${live_file}" ] || return 1
  logz="$(grep -m1 '^log(Z)' "${stats_file}" 2>/dev/null | awk '{print $3}')"
  [ -n "${logz}" ] || return 1
  awk -v logz="${logz}" -v ndead="${dead_count}" -v nlive="${nlive}" \
      -v ratio="${_NS_TERMINATION_EVIDENCE_RATIO}" '
    { n++; ll[n] = $NF }
    END {
      if (n == 0) exit 1
      max = ll[1]
      for (i = 2; i <= n; i++) if (ll[i] > max) max = ll[i]
      s = 0
      for (i = 1; i <= n; i++) s += exp(ll[i] - max)
      live_logz = (max + log(s)) - log(n) + (-ndead / nlive)
      remaining = nlive * (live_logz - logz - log(ratio))
      if (remaining < 0) remaining = 0
      printf "%d", ndead + int(remaining + 0.5)
    }
  ' "${live_file}"
}

self_check() {
  local tmp health_report
  tmp="$(mktemp -d)"
  mkdir "${tmp}/chains" "${tmp}/evaluations"

  [ "$(_ns_dead_birth_file "${tmp}/chains")" = "" ] || { echo "FAIL: no dead-birth file yet"; exit 1; }
  [ "$(_ns_count_lines "${tmp}/chains/missing")" = "0" ] || { echo "FAIL: missing file counts as 0"; exit 1; }

  printf 'l1\nl2\nl3\n' >"${tmp}/chains/wsclean_vlaa_dead-birth.txt"
  [ "$(_ns_dead_birth_file "${tmp}/chains")" = "${tmp}/chains/wsclean_vlaa_dead-birth.txt" ] || {
    echo "FAIL: dead-birth file not found"; exit 1
  }
  [ "$(_ns_count_lines "${tmp}/chains/wsclean_vlaa_dead-birth.txt")" = "3" ] || { echo "FAIL: dead-birth line count"; exit 1; }

  mkdir "${tmp}/evaluations/eval-0001-a" "${tmp}/evaluations/eval-0002-b" \
    "${tmp}/evaluations/eval-0003-c" "${tmp}/evaluations/eval-0004-d"
  local dead_file="${tmp}/chains/wsclean_vlaa_dead-birth.txt"
  [ "$(_ns_count_evals "${tmp}/evaluations")" = "4 0" ] || {
    echo "FAIL: evaluations count: $(_ns_count_evals "${tmp}/evaluations")"; exit 1
  }
  # -t rather than -d: the timestamp form is POSIX and BSD touch has no -d.
  touch -t 202601010000 "${tmp}/evaluations"/eval-*
  touch -t 202601010100 "${dead_file}"
  touch -t 202601010200 "${tmp}/evaluations/eval-0003-c" "${tmp}/evaluations/eval-0004-d"
  [ "$(_ns_count_evals "${tmp}/evaluations" "${dead_file}")" = "4 2" ] || {
    echo "FAIL: split either side of the checkpoint: $(_ns_count_evals "${tmp}/evaluations" "${dead_file}")"
    exit 1
  }
  [ "$(_ns_count_evals "${tmp}/evaluations" "${tmp}/chains/absent")" = "4 0" ] || {
    echo "FAIL: missing reference must not split"; exit 1
  }

  # Carrying the frozen dead-point count across the checkpoint interval: 100
  # dead points banked over 900 evaluations is one per nine, so the 100
  # evaluations since the checkpoint are worth 11 more.
  [ "$(_ns_dead_now 100 1000 100)" = "111" ] || { echo "FAIL: dead_now: $(_ns_dead_now 100 1000 100)"; exit 1; }
  # Rounded, not truncated - truncation pins a slow run to its checkpoint.
  [ "$(_ns_dead_now 10 100 5)" = "11" ] || { echo "FAIL: dead_now rounds up"; exit 1; }
  [ "$(_ns_dead_now 10 100 4)" = "10" ] || { echo "FAIL: dead_now rounds down"; exit 1; }
  # Nothing to extrapolate from, or nothing to extrapolate: the checkpoint's
  # own count, never a division by zero.
  [ "$(_ns_dead_now 0 1000 100)" = "0" ] || { echo "FAIL: dead_now with no dead points"; exit 1; }
  [ "$(_ns_dead_now 100 1000 0)" = "100" ] || { echo "FAIL: dead_now on a fresh checkpoint"; exit 1; }
  [ "$(_ns_dead_now 100 100 100)" = "100" ] || { echo "FAIL: dead_now with nothing banked"; exit 1; }

  [ "$(_ns_format_hms 3661)" = "1:01:01" ] || { echo "FAIL: format_hms"; exit 1; }
  [ "$(_ns_format_hms 59)" = "0:00:59" ] || { echo "FAIL: format_hms short"; exit 1; }

  # A pinned draw is written to a fixed row and never re-checked, so a line
  # even one column wider than the terminal wraps onto the row above (inside
  # the scroll region) and gets scrolled away within seconds - confirmed by
  # replaying a real run through a terminal emulator: the bar was in the byte
  # stream but invisible on screen. `tput cols` fails outside a real
  # terminal, so this exercises the 80-column fallback.
  local long_line padded
  long_line="$(printf 'x%.0s' $(seq 1 200))"
  padded="$(_ns_truncate_pad "${long_line}")"
  [ "${#padded}" = "79" ] || { echo "FAIL: long line not clamped to 79 cols: ${#padded}"; exit 1; }
  padded="$(_ns_truncate_pad "short")"
  [ "${#padded}" = "79" ] || { echo "FAIL: short line not padded to 79 cols: ${#padded}"; exit 1; }
  case "${padded}" in
    "short"*) ;;
    *) echo "FAIL: padding altered the content: ${padded}"; exit 1 ;;
  esac

  # 3 dead points against a cap of 12 must drive the percent/ETA, not the 4
  # evaluations - that mismatch (dead points lag evaluations) is exactly the
  # R2D2 run that showed "29/12 dead points" before this used dead-birth.txt.
  #
  # Two of the four evaluations landed after the checkpoint and two before, so
  # the three checkpointed dead points carry forward to ~6 - and every figure
  # that came from the carry is marked `~`, while the cap the operator asked
  # for is not.
  local line
  line="$(_ns_status_line "${tmp}" 12 8 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"~ 50%  ~6/12 dead points (4 evaluations)"*) ;;
    *) echo "FAIL: status line dead/eval split: ${line}"; exit 1 ;;
  esac
  # A checkpoint newer than every evaluation has nothing to carry, and then the
  # count is exact and says so by dropping the `~`.
  touch "${dead_file}"
  line="$(_ns_status_line "${tmp}" 12 8 "$(($(date +%s) - 60))")"
  case "${line}" in
    *" 25%  3/12 dead points (4 evaluations)"*) ;;
    *) echo "FAIL: fresh checkpoint must not be marked estimated: ${line}"; exit 1 ;;
  esac

  # max_ndead <= 0 has no budget to divide by - must not claim an ETA or a
  # percent, but should still show a rate, before chains/*.stats and
  # chains/*_phys_live.txt exist (see the evidence-pct test below for once
  # they do).
  line="$(_ns_status_line "${tmp}" -1 8 "$(($(date +%s) - 90))")"
  case "${line}" in
    *eta*) echo "FAIL: unbounded max_ndead printed an eta: ${line}"; exit 1 ;;
  esac
  case "${line}" in
    *%*) echo "FAIL: unbounded max_ndead printed a percent: ${line}"; exit 1 ;;
  esac
  case "${line}" in
    *"2/min"*) ;;
    *) echo "FAIL: unbounded max_ndead did not show a rate: ${line}"; exit 1 ;;
  esac

  # The bounce must sweep from one end of the bar to the other and back
  # instead of running off it or freezing.
  _ns_bar_pos() {
    local bar="${1#*[}" i=0
    bar="${bar%%]*}"
    while [ "${i}" -lt "${#bar}" ]; do
      [ "${bar:${i}:1}" = "#" ] && { echo "${i}"; return; }
      i=$((i + 1))
    done
    echo "-1"
  }
  [ "$(_ns_bar_pos "$(_ns_unbounded_line 0 0 0)")" = "0" ] || { echo "FAIL: bounce at t=0"; exit 1; }
  [ "$(_ns_bar_pos "$(_ns_unbounded_line 0 0 29)")" = "29" ] || { echo "FAIL: bounce at t=29 (far end)"; exit 1; }
  [ "$(_ns_bar_pos "$(_ns_unbounded_line 0 0 30)")" = "28" ] || { echo "FAIL: bounce at t=30 (reversed)"; exit 1; }
  [ "$(_ns_bar_pos "$(_ns_unbounded_line 0 0 58)")" = "0" ] || { echo "FAIL: bounce at t=58 (full period)"; exit 1; }

  # Once chains/*.stats and chains/*_phys_live.txt exist, an uncapped run
  # (--max-ndead <= 0) should switch from the ambiguous bounce to a percent
  # and an estimated total. nlive=2, ndead=10, both live points at loglike 0,
  # log(Z)=0.0: live_logZ = logsumexp(0,0) - log(2) - 10/2 = -5, so the ratio
  # still has ln(1.2e-4) - (-5) = 4.03 nats to fall, which at one e-fold per
  # nlive dead points is 8 more dead points - 10/18 = 55%.
  printf 'log(Z)       =   0.0 +/-   0.1\n' >"${tmp}/chains/wsclean_vlaa.stats"
  printf '0.1 0.2 0.0\n0.3 0.4 0.0\n' >"${tmp}/chains/wsclean_vlaa_phys_live.txt"
  [ "$(_ns_evidence_total "${tmp}/chains" 10 2)" = "18" ] || {
    echo "FAIL: evidence total: $(_ns_evidence_total "${tmp}/chains" 10 2)"; exit 1
  }
  # _ns_status_line reads ndead from chains/*_dead-birth.txt itself, so match
  # the 10 dead points the hand-computed 18 above assumes, with the checkpoint
  # newer than every evaluation so there is nothing to carry: 10/18 is 55%, and
  # 8 dead points left at 10 per 60s is a 48s eta.
  seq 1 10 >"${dead_file}"
  line="$(_ns_status_line "${tmp}" -1 2 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"~ 55%"*"~10/~18 dead points"*"eta ~0:00:48"*"evidence tolerance"*) ;;
    *) echo "FAIL: status line did not switch to the evidence total: ${line}"; exit 1 ;;
  esac
  # ...and with two of the four evaluations landing after that checkpoint, the
  # 10 banked over 2 carry forward to 20 - which is past the estimated total,
  # so the percent clamps at 100 rather than printing "111%" on a run that is
  # about to stop anyway.
  touch -t 202601010100 "${dead_file}"
  touch -t 202601010200 "${tmp}/evaluations/eval-0003-c" "${tmp}/evaluations/eval-0004-d"
  line="$(_ns_status_line "${tmp}" -1 2 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"~100%"*"~20/~18 dead points"*"eta ~0:00:00"*) ;;
    *) echo "FAIL: carried count past the estimated total: ${line}"; exit 1 ;;
  esac
  touch "${dead_file}"
  # No dead points yet means no ndead/nlive to divide by - must fall back to
  # the bounce rather than divide by zero or report a bogus percent.
  _ns_evidence_total "${tmp}/chains" 0 2 >/dev/null && { echo "FAIL: evidence total with 0 dead points should fail"; exit 1; }
  # ...and neither does the first e-fold, where the live set is still the
  # prior: the answer there is nlive*ln(1/RATIO), the function's own constant.
  _ns_evidence_total "${tmp}/chains" 1 2 >/dev/null && { echo "FAIL: evidence total inside the first e-fold should fail"; exit 1; }
  # A run that is already past the stopping ratio (ndead=20, nlive=2 puts
  # live_logZ at -10, below ln(1.2e-4) = -9.03) has negative work left. The
  # total must stay at the dead points it has rather than going below them and
  # printing "20/~19 dead points" on a run about to stop.
  [ "$(_ns_evidence_total "${tmp}/chains" 20 2)" = "20" ] || {
    echo "FAIL: past the stopping ratio: $(_ns_evidence_total "${tmp}/chains" 20 2)"; exit 1
  }

  # The pinned status line and `./ri health` forecast an uncapped run from the
  # same calibrated stopping fraction. Two hardcoded copies of a measured
  # constant drift; this fails the moment they do, because a bar and a report
  # disagreeing about the same run is exactly what this replaced.
  health_report="${BASH_SOURCE[0]%/*}/../nested-sampling-health.py"
  if [ -f "${health_report}" ]; then
    grep -q "^TERMINATION_EVIDENCE_RATIO = ${_NS_TERMINATION_EVIDENCE_RATIO}\$" \
      "${health_report}" || {
      echo "FAIL: ${health_report} no longer carries TERMINATION_EVIDENCE_RATIO = ${_NS_TERMINATION_EVIDENCE_RATIO}"
      exit 1
    }
  fi

  # _ns_add_trap must append to an existing trap, not replace it - a naive
  # `trap ... EXIT` here would silently disable start-sidecars.sh's Docker
  # cleanup on exit/Ctrl-C.
  (
    log="${tmp}/trap.log"
    # shellcheck disable=SC2064  # ${log} must expand before the subshell exits
    trap "echo existing >>'${log}'" EXIT
    _ns_add_trap "echo new >>'${log}'" EXIT
  )
  [ "$(cat "${tmp}/trap.log")" = "$(printf 'new\nexisting')" ] || {
    echo "FAIL: trap chaining order/content: $(cat "${tmp}/trap.log")"; exit 1
  }

  # run_with_progress must keep the run's output rather than only showing it,
  # and must still hand back the command's exit status - the run scripts are
  # `set -e` and a swallowed failure would let one report OK after failing.
  local run_dir="${tmp}/run"
  mkdir -p "${run_dir}"
  local status=0
  run_with_progress "${run_dir}" -1 2 -- \
    sh -c 'echo to-stdout; echo to-stderr >&2; exit 7' >"${tmp}/terminal" 2>&1 || status=$?
  [ "${status}" = "7" ] || { echo "FAIL: exit status ${status}, want 7"; exit 1; }
  # Both streams, in the file and on the way through to the terminal.
  grep -q '^to-stdout$' "${run_dir}/run.log" || { echo "FAIL: stdout not logged"; exit 1; }
  grep -q '^to-stderr$' "${run_dir}/run.log" || { echo "FAIL: stderr not logged"; exit 1; }
  grep -q '^to-stdout$' "${tmp}/terminal" || { echo "FAIL: stdout not passed through"; exit 1; }
  # Appended across runs, because ./ri resume runs again into the same
  # directory and the first failure is usually the one worth reading.
  run_with_progress "${run_dir}" -1 2 -- sh -c 'echo second-run' >/dev/null 2>&1
  grep -q '^to-stdout$' "${run_dir}/run.log" || { echo "FAIL: log truncated on resume"; exit 1; }
  grep -q '^second-run$' "${run_dir}/run.log" || { echo "FAIL: second run not logged"; exit 1; }
  # The whole of a noisy run's output, not just the start of it: the last line
  # before a crash is the one worth having. Sized past the pipe buffer so this
  # is a real completeness check rather than one satisfied by a single write.
  # shellcheck disable=SC2016  # $i is sh's, not ours
  run_with_progress "${run_dir}" -1 2 -- \
    sh -c 'i=0; while [ $i -lt 20000 ]; do echo "line-$i"; i=$((i + 1)); done; exit 3' \
    >/dev/null 2>&1
  [ "$(tail -n 1 "${run_dir}/run.log")" = "line-19999" ] || {
    echo "FAIL: last line lost: $(tail -n 1 "${run_dir}/run.log")"; exit 1
  }
  # And the pipe it went through is not left behind in the run directory.
  [ ! -e "${run_dir}/.run.log.fifo" ] || { echo "FAIL: fifo left behind"; exit 1; }

  # run_with_retries: a failure that scored evaluations gets another go, a
  # failure that scored none does not, and the exit status still travels.
  #
  # The command counts its own attempts through a file and scores one more
  # evaluation each time, which is what a working run does - so "retried" and
  # "made progress" are the same fixture. Deliberately no dead points
  # anywhere in it: that is the case this used to get wrong, a run killed
  # inside PolyChord's checkpoint interval after real work.
  local retry_dir="${tmp}/retry"
  mkdir -p "${retry_dir}/chains"
  # Single quotes deliberately: $0 is the directory argument passed to the
  # child `sh` below, not this shell's own name.
  # shellcheck disable=SC2016
  local progressing='n=$(ls "$0"/evaluations 2>/dev/null | wc -l);
    mkdir -p "$0"/evaluations/eval-$n; echo {} >"$0"/evaluations/eval-$n/metrics.json;
    echo a >>"$0"/attempts; exit 5'
  status=0
  run_with_retries 2 "${retry_dir}" -1 2 -- sh -c "${progressing}" "${retry_dir}" \
    >/dev/null 2>&1 || status=$?
  [ "${status}" = "5" ] || { echo "FAIL: retried exit status ${status}, want 5"; exit 1; }
  [ "$(_ns_count_lines "${retry_dir}/attempts")" = "3" ] || {
    echo "FAIL: attempts $(_ns_count_lines "${retry_dir}/attempts"), want 1 + 2 retries"; exit 1
  }
  # ./ri health counts its restarts from this file, one line per retry.
  [ "$(_ns_count_lines "${retry_dir}/restarts.log")" = "2" ] || {
    echo "FAIL: restarts.log has $(_ns_count_lines "${retry_dir}/restarts.log") lines, want 2"
    exit 1
  }

  # Same command, scoring nothing: a deterministic failure - a missing image,
  # a bad parameter space - must stop after one attempt rather than fail three
  # times as slowly. It leaves behind the half-written evaluation directory a
  # crashed rank does, which must not read as work: no metrics.json, no score.
  local stuck_dir="${tmp}/stuck"
  mkdir -p "${stuck_dir}/chains"
  status=0
  # shellcheck disable=SC2016
  run_with_retries 2 "${stuck_dir}" -1 2 -- \
    sh -c 'mkdir -p "$0"/evaluations/eval-in-flight; echo a >>"$0"/attempts; exit 5' \
    "${stuck_dir}" >/dev/null 2>&1 || status=$?
  [ "${status}" = "5" ] || { echo "FAIL: stuck exit status ${status}, want 5"; exit 1; }
  [ "$(_ns_count_lines "${stuck_dir}/attempts")" = "1" ] || {
    echo "FAIL: no-progress failure retried $(_ns_count_lines "${stuck_dir}/attempts") times"; exit 1
  }
  [ ! -e "${stuck_dir}/restarts.log" ] || { echo "FAIL: restart logged without a retry"; exit 1; }

  # A run that banked dead points before it died is still retried, which is
  # the case that worked before evaluations became the measure - the two
  # counters must not have swapped places.
  local banked_dir="${tmp}/banked"
  mkdir -p "${banked_dir}/chains"
  echo x >"${banked_dir}/chains/r_dead-birth.txt"
  status=0
  run_with_retries 1 "${banked_dir}" -1 2 -- sh -c "${progressing}" "${banked_dir}" \
    >/dev/null 2>&1 || status=$?
  [ "$(_ns_count_lines "${banked_dir}/attempts")" = "2" ] || {
    echo "FAIL: run with dead points retried $(_ns_count_lines "${banked_dir}/attempts") times"
    exit 1
  }

  # And the counter itself: finished evaluations only, from the directory the
  # run writes them to.
  [ "$(_ns_completed_evals "${stuck_dir}")" = "0" ] || {
    echo "FAIL: in-flight evaluation counted as completed"; exit 1
  }
  [ "$(_ns_completed_evals "${tmp}/absent")" = "0" ] || {
    echo "FAIL: missing run directory did not count zero"; exit 1
  }
  [ "$(_ns_completed_evals "${retry_dir}")" = "3" ] || {
    echo "FAIL: completed evaluations $(_ns_completed_evals "${retry_dir}"), want 3"; exit 1
  }

  # 0 retries is the old behaviour exactly, and a command that succeeds runs
  # once and returns 0.
  local once_dir="${tmp}/once"
  mkdir -p "${once_dir}/chains"
  # shellcheck disable=SC2016
  run_with_retries 0 "${once_dir}" -1 2 -- sh -c 'echo a >>"$0"/attempts; exit 5' \
    "${once_dir}" >/dev/null 2>&1 || status=$?
  [ "$(_ns_count_lines "${once_dir}/attempts")" = "1" ] || { echo "FAIL: retries=0 retried"; exit 1; }
  # shellcheck disable=SC2016
  run_with_retries 2 "${once_dir}" -1 2 -- sh -c 'echo b >>"$0"/attempts' "${once_dir}" \
    >/dev/null 2>&1 || { echo "FAIL: success returned nonzero"; exit 1; }
  [ "$(_ns_count_lines "${once_dir}/attempts")" = "2" ] || { echo "FAIL: success re-ran"; exit 1; }

  rm -rf "${tmp}"
  echo "progress-bar self-check passed"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  self_check
fi
