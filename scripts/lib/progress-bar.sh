#!/usr/bin/env bash
# Nested-sampling status line and retry loop; restarts re-clamp ranks to free memory.
# shellcheck source=scripts/lib/rank-budget.sh
. "${BASH_SOURCE[0]%/*}/rank-budget.sh"

# Usage: run_with_progress <output_dir> <max_ndead> <nlive> -- cmd args...
run_with_progress() {
  local output_dir="$1" max_ndead="$2" nlive="$3"
  shift 3
  [ "${1:-}" = "--" ] && shift

  # FIFO keeps tee waitable, so run.log is complete while stdout stays a TTY.
  local log="${output_dir}/run.log" pipe="${output_dir}/.run.log.fifo"
  rm -f "${pipe}"
  mkfifo "${pipe}"
  # Resume appends, preserving the first failure in the log.
  tee -a "${log}" <"${pipe}" &
  local tee_pid=$!
  "$@" >"${pipe}" 2>&1 &
  local pid=$!
  local start
  start="$(date +%s)"

  # Keep watchdog coverage when started under nohup, where no status bar runs.
  local watchdog_pid=""
  if [ "${NS_STALL_TIMEOUT:-0}" -gt 0 ]; then
    _ns_stall_watchdog "${output_dir}" "${NS_STALL_TIMEOUT}" &
    watchdog_pid=$!
    # EXIT covers INT/TERM; SIGKILL is reported by `./ri health`.
    _NS_WATCHDOG_PID="${watchdog_pid}"
    _ns_add_trap '_ns_stop_watchdog' EXIT
  fi

  if [ -t 1 ] && _ns_pin_setup; then
    _ns_add_trap '_ns_pin_teardown' EXIT
    _ns_add_trap '_ns_pin_teardown' INT
    _ns_add_trap '_ns_pin_teardown' TERM
    local drawn
    while kill -0 "${pid}" 2>/dev/null; do
      drawn="$(_ns_now_us)"
      _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")"
      sleep "$(_ns_backoff_interval "${drawn}")"
    done
    _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")"
    _ns_pin_teardown
  elif [ -t 1 ]; then
    # Missing terminal control: redraw a plain line.
    local drawn
    while kill -0 "${pid}" 2>/dev/null; do
      drawn="$(_ns_now_us)"
      printf '\r%s' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")")"
      sleep "$(_ns_backoff_interval "${drawn}")"
    done
    printf '\r%s\n' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${nlive}" "${start}")")"
  fi

  local status=0
  wait "${pid}" || status=$?
  _ns_stop_watchdog
  if [ -n "${watchdog_pid}" ]; then
    wait "${watchdog_pid}" 2>/dev/null || true
  fi
  # Wait for tee after the command closes its FIFO, leaving complete run.log.
  wait "${tee_pid}" 2>/dev/null || true
  rm -f "${pipe}"
  return "${status}"
}

# Usage: run_with_retries <retries> <output_dir> <max_ndead> <nlive> -- cmd args...
# Recovery retries require progress, reuse sidecars without dead FIFO workers,
# and reset after long attempts. See docs/robustness.md for the contract.
run_with_retries() {
  local retries="$1" output_dir="$2"
  shift
  local reset_after="${NS_RETRY_RESET_SECONDS:-1800}"
  local attempt=0 status=0 before after started ranks arg prev resized
  local log_before=0 from_where
  # Keep arguments mutable so a restart can re-clamp its rank count.
  local -a args=("$@") rescaled
  while :; do
    before="$(_ns_completed_evals "${output_dir}")"
    log_before="$(_ns_log_size "${output_dir}")"
    status=0
    started="$(date +%s)"
    run_with_progress "${args[@]}" || status=$?
    [ "${status}" -eq 0 ] && return 0
    after="$(_ns_completed_evals "${output_dir}")"
    # Let the attempt that earned the reset use it.
    if [ "$(($(date +%s) - started))" -ge "${reset_after}" ]; then
      attempt=0
    fi
    if [ "${attempt}" -ge "${retries}" ]; then
      if [ "${retries}" -gt 0 ]; then
        _ns_retry_say "${output_dir}" \
          "not retrying: ${retries} of ${retries} restarts used and this attempt" \
          "(exit ${status}) died inside ${reset_after}s, so the fault is still there." \
          "Why it stopped is above; ./ri resume ${output_dir##*/} tries again anyway."
      fi
      break
    fi
    from_where="PolyChord's checkpoint"
    if [ "${after}" -le "${before}" ]; then
      # Zero progress normally means a deterministic failure. Quarantine a
      # checkpoint only when this attempt's output names its Fortran reader;
      # otherwise preserve the checkpoint and stop.
      if _ns_attempt_output "${output_dir}" "${log_before}" \
           | grep -q 'read_write\.F90' \
        && _ns_quarantine_checkpoint "${output_dir}"; then
        from_where="scratch"
        _ns_retry_say "${output_dir}" \
          "the attempt that just failed (exit ${status}) died inside PolyChord's checkpoint" \
          "I/O before scoring anything, so its checkpoint cannot be read: moved aside as" \
          "chains/*.resume.unreadable. Restarting the sampler from scratch, which replays" \
          "the ${after} evaluations already scored from the point cache without imaging."
      else
        _ns_retry_say "${output_dir}" \
          "not retrying: the attempt that just failed (exit ${status}) scored no evaluations," \
          "so another one fails the same way. Why it stopped is above;" \
          "./ri resume ${output_dir##*/} tries again anyway."
        break
      fi
    fi
    # Re-clamp against current memory: another run may have grown since the
    # failed attempt. Down costs time; up risks an OOM score. Restarts create
    # their own workers, so the original FIFO pool does not constrain them.
    ranks="$(_ns_retry_rank_count "${args[@]}")"
    if [ "${ranks}" = "0" ]; then
      _ns_retry_say "${output_dir}" \
        "not retrying: there is no longer memory for even one rank (the FATAL above" \
        "says what is holding it), and a restart that cannot fit scores OOM kills as" \
        "the search's best points. ./ri resume ${output_dir##*/} once there is room."
      break
    fi
    resized=''
    if [ -n "${ranks}" ]; then
      resized=", re-sized to ${ranks} ranks to fit the memory free now"
      rescaled=()
      prev=''
      for arg in "${args[@]}"; do
        case "${prev}" in -np) arg="${ranks}" ;; esac
        case "${arg}" in NS_MPI_PROCS=*) arg="NS_MPI_PROCS=${ranks}" ;; esac
        rescaled+=("${arg}")
        prev="${arg}"
      done
      args=("${rescaled[@]}")
    fi
    # Sidecar restore is optional for fixtures; log failures, then let the next
    # attempt's progress guard stop if the sidecar remains unavailable.
    if declare -F sidecar_restore >/dev/null; then
      sidecar_restore 2>&1 | tee -a "${output_dir}/run.log" >&2 || true
    fi
    attempt=$((attempt + 1))
    printf '%s exit %s after %s evaluations\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${after}" \
      >>"${output_dir}/restarts.log"
    _ns_retry_say "${output_dir}" \
      "attempt failed (exit ${status}) at ${after} evaluations; resuming from ${from_where}" \
      "- retry ${attempt} of ${retries}${resized}. Why it stopped is above."
  done
  return "${status}"
}

# Return a lower rank count, 0 when one rank does not fit, or nothing when the
# command is not an MPI search.
_ns_retry_rank_count() {
  local arg prev='' current='' mb='' label='' ranks
  for arg in "$@"; do
    case "${prev}" in -np) current="${arg}" ;; esac
    case "${arg}" in
      *polychord_r2d2.py) mb="${NS_R2D2_MB_PER_RANK}" label=r2d2 ;;
      *polychord_wsclean.py) mb="${NS_WSCLEAN_MB_PER_RANK}" label=wsclean ;;
    esac
    prev="${arg}"
  done
  # Not an mpirun command at all - the self-checks' fixtures, and any future
  # caller of run_with_retries that is not a search.
  [ -n "${current}" ] && [ -n "${mb}" ] || return 0
  # `ns_budget_ranks` fails, loudly, only when one rank no longer fits.
  ranks="$(ns_budget_ranks "${current}" "${mb}" "${label} restart")" || {
    printf '0\n'
    return 0
  }
  [ "${ranks}" -lt "${current}" ] && printf '%s\n' "${ranks}"
  return 0
}

# Print retry status to stderr and run.log; health reads run.log.
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

# Return run.log size so retry output can be isolated from earlier attempts.
_ns_log_size() {
  [ -f "$1/run.log" ] || { printf '0\n'; return 0; }
  wc -c <"$1/run.log" | tr -d ' '
}

# Usage: _ns_attempt_output <output_dir> <bytes written before the attempt>
_ns_attempt_output() {
  [ -f "$1/run.log" ] || return 0
  tail -c "+$(($2 + 1))" "$1/run.log"
}

# Quarantine the run's PolyChord checkpoint, if present.
_ns_quarantine_checkpoint() {
  local f moved=''
  for f in "$1"/chains/*.resume; do
    [ -f "${f}" ] || continue
    mv -f "${f}" "${f}.unreadable" && moved=1
  done
  [ -n "${moved}" ]
}

# Stop the watchdog from normal cleanup or its EXIT trap; clear its PID to avoid
# a delayed trap signaling a reused PID.
_ns_stop_watchdog() {
  [ -n "${_NS_WATCHDOG_PID:-}" ] || return 0
  kill "${_NS_WATCHDOG_PID}" 2>/dev/null || true
  _NS_WATCHDOG_PID=""
}

# Usage: _ns_stall_watchdog <output_dir> <timeout seconds>
#
# Detect hangs that worker timeouts cannot: a stuck rank leaves PolyChord
# waiting forever. Watch completed evaluations, then kill the run by command
# line so retry logic can resume it; the `docker exec` client PID is not enough.
# Default timeout is twice IMAGING_REPLY_TIMEOUT, well above the measured
# 23.5s maximum gap; `./ri health` provides the interactive equivalent.
_ns_stall_watchdog() {
  local output_dir="$1" timeout="$2" floor="${NS_STALL_POLL_SECONDS:-60}"
  local last quiet=0 now poll scanned
  poll="${floor}"
  last="$(_ns_completed_evals "${output_dir}")"
  while sleep "${poll}"; do
    scanned="$(_ns_now_us)"
    now="$(_ns_completed_evals "${output_dir}")"
    if [ "${now}" != "${last}" ]; then
      last="${now}"
      quiet=0
    else
      quiet=$((quiet + poll))
      if [ "${quiet}" -ge "${timeout}" ]; then
        _ns_retry_say "${output_dir}" \
          "no evaluation has finished in ${quiet}s (${now} scored, still counting)." \
          "A run that is alive but landing nothing is hung, not slow - killing it" \
          "so it can restart from PolyChord's checkpoint."
        pkill -9 -f "$(ns_run_process_pattern "${output_dir}")" || true
        return 0
      fi
    fi
    # `_ns_completed_evals` walks every evaluation *directory*, not just
    # `evaluations/`, so it costs several times the status line's pass and
    # grows the same way: 0.28s at the 34,682 evaluations of an --nlive 200
    # search here on an idle host, and ~2.4x that on the loaded one a watchdog
    # actually runs on, so of order 5s a poll at the 270,000 an
    # --nlive 500 --num-repeats 25 search reaches, and rising with the run.
    # Backing the poll off to nine times the scan's own cost bounds it at ~10%
    # of one core at any run size; below ~450,000 evaluations the 60s floor
    # still wins and the cadence is exactly what it was. The 600s cap keeps the
    # quiet time this reports inside a tenth of the 7200s default timeout, and
    # the floor is the caller's poll, so an NS_STALL_POLL_SECONDS set to go
    # with a short timeout still holds.
    poll="$(_ns_backoff_interval "${scanned}" "${floor}" 600)"
  done
}

# Append a trap without replacing sidecar cleanup; run the new command first.
_ns_add_trap() {
  local new="$1" sig="$2" line existing=""
  line="$(trap -p "${sig}")"
  if [ -n "${line}" ]; then
    # Let Bash parse trap's quoted command; textual prefix stripping breaks on quotes.
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

# Microseconds since the epoch, or nothing at all where the shell cannot say.
# EPOCHREALTIME is bash 5 and CI runs this file's self-check on macOS, whose
# /bin/bash is 3.2; the one caller treats "cannot say" as "do not adapt". The
# separator is matched as a class because it follows the locale.
_ns_now_us() {
  local t="${EPOCHREALTIME:-}"
  printf '%s' "${t/[.,]/}"
}

# Usage: _ns_backoff_interval <_ns_now_us when the poll started> [floor] [cap]
#
# How long to sleep before repeating a poll whose own cost is O(evaluations).
# Both loops in this file walk `evaluations/` - the status line for its counts,
# the stall watchdog for its "has anything landed" - so a fixed cadence gets
# more expensive the longer the run goes. The bar's two passes measure 0.37s
# and 0.35s at the 270,000 evaluations an --nlive 500 --num-repeats 25 search
# reaches on this host, so at a fixed `sleep 1` it spent 44% of a core - 2.2%
# of the machine - counting the run rather than advancing it, and rising.
#
# Sleeping nine times the poll's own cost holds the loop to ~10% of one core at
# any run size, which is the point: the cost stops scaling with the run instead
# of merely being smaller. The floor keeps the old cadence for every run whose
# poll is cheap (a draw under ~0.11s still redraws once a second, exactly as it
# did), and the cap stops a pathological poll from looking dead.
# See docs/nested-sampling-throughput.md.
_ns_backoff_interval() {
  local started="$1" floor="${2:-1}" cap="${3:-30}" now cost
  now="$(_ns_now_us)"
  if [ -z "${started}" ] || [ -z "${now}" ]; then
    echo "${floor}"
    return
  fi
  # Clamped in arithmetic rather than with `[ ] &&`, which is an AND-list that
  # returns non-zero whenever the bound does not bite - and this file is
  # sourced into a script running under `set -e`.
  cost=$(((now - started) * 9 / 1000000))
  cost=$((cost > cap ? cap : cost))
  echo "$((cost < floor ? floor : cost))"
}

# "<evaluations> <of those, landed after `reference` was written>". `find`
# rather than the glob loop this replaced, which cost 283ms of every
# one-second redraw on a live 7,200-evaluation run against ~100ms for one
# find pass here.
#
# Two passes, though, not the one the comment here used to claim: POSIX find
# has no way to mark which entries matched `-newer`, and `-printf '%T@'` -
# which would answer both questions in one walk - is GNU-only, while CI runs
# this file's self-check on macOS as well. Both passes are O(evaluations), so
# what bounds their cost on a long run is `_ns_backoff_interval` above, not
# this function.
_ns_count_evals() {
  local dir="$1" reference="${2:-}" total since=0
  # GNU find can classify both counts during one directory walk. Keep the
  # portable two-pass fallback for macOS and other BSD find implementations.
  if find "${dir}" -maxdepth 0 -printf '' >/dev/null 2>&1; then
    if [ -n "${reference}" ] && [ -e "${reference}" ]; then
      read -r total since < <(
        find "${dir}" -maxdepth 1 -name 'eval-*' \
          \( -newer "${reference}" -printf 's\n' -o -printf 't\n' \) 2>/dev/null |
          awk '{ total++; since += ($1 == "s") } END { print total + 0, since + 0 }'
      )
    else
      total="$(find "${dir}" -maxdepth 1 -name 'eval-*' -printf 'x\n' 2>/dev/null |
        wc -l | tr -d ' ')"
    fi
    echo "${total:-0} ${since:-0}"
    return
  fi
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

# Usage: _ns_run_bounded <seconds> <command...>  ->  124 if it did not finish
#
# GNU coreutils' `timeout` is not on macOS, and this is the only place that
# wants one - the self-check below has to bound a fixture that is *supposed* to
# hang, and CI runs that check on both runners. Same contract as `timeout` for
# the one case used here, including its exit 124. Like `timeout`, it signals
# only the command it started; the self-check pkills the fixture's own ranks.
_ns_run_bounded() {
  local seconds="$1" pid waited=0
  shift
  "$@" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${waited}" -ge "${seconds}" ]; then
      kill -9 "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 124
    fi
    sleep 1
    waited=$((waited + 1))
  done
  wait "${pid}"
}

_ns_count_lines() {
  local file="$1"
  [ -f "${file}" ] || { echo 0; return; }
  wc -l <"${file}" | tr -d ' '
}

# `24s`, `7m36s`, `10h24m`, `2d 6h` - a duration, not a clock time. The same
# rendering as `format_elapsed` in scripts/nested-sampling-health.py, so an
# interval reads the same on the pinned status line and in `./ri health`;
# `H:MM:SS` read as a timestamp beside the ISO stamps both surfaces also print.
_ns_format_elapsed() {
  local s="$1"
  if [ "${s}" -lt 60 ]; then
    printf '%ds' "${s}"
  elif [ "${s}" -lt 3600 ]; then
    printf '%dm%02ds' $((s / 60)) $((s % 60))
  elif [ "${s}" -lt 86400 ]; then
    printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60))
  else
    printf '%dd %dh' $((s / 86400)) $(((s % 86400) / 3600))
  fi
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
    eta="$(_ns_format_elapsed $(((max_ndead - dead_now) * elapsed / dead_now)))"
  elif [ "${dead_now}" -ge "${max_ndead}" ]; then
    eta="0s"
  fi

  bar="$(_ns_render_bar "${pct}" 30)"

  # The cap is a number the operator asked for and stays plain; the position
  # within it is carried across the checkpoint interval, so it and everything
  # derived from it take the `~` whenever it is not the checkpoint's own count.
  printf '[%s] %s%3d%%  %s%d/%d dead points (%d evaluations)  elapsed %s  eta %s%s' \
    "${bar}" "${carried}" "${pct}" "${carried}" "${dead_now}" "${max_ndead}" \
    "${eval_count}" "$(_ns_format_elapsed "${elapsed}")" "${carried}" "${eta}"
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
    "${bar}" "${dead_count}" "${eval_count}" "${rate}" "$(_ns_format_elapsed "${elapsed}")"
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
  local note="(evidence tolerance, no --max-ndead cap)"
  if [ "$((total - dead_now))" -gt 0 ] && [ "${dead_now}" -gt 0 ]; then
    eta="$(_ns_format_elapsed $(((total - dead_now) * elapsed / dead_now)))"
  elif [ "${dead_now}" -ge "${total}" ]; then
    # Not "eta 0s". The total came from the last checkpoint and is as
    # stale as it is, so a carried count reaching it means the estimate has
    # been overtaken, not that the run is done - the live R2D2 search here sat
    # past its own estimate for over three hours while still sampling. The
    # same distinction `./ri health` draws on its forecast line.
    note="(past the ~${total} estimate; the next checkpoint revises it)"
  fi
  printf '[%s] ~%3d%%  ~%d/~%d dead points (%d evaluations)  elapsed %s  eta ~%s  %s' \
    "${bar}" "${pct}" "${dead_now}" "${total}" "${eval_count}" \
    "$(_ns_format_elapsed "${elapsed}")" "${eta}" "${note}"
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

  # The redraw backs off with the cost of the draw, so the bar cannot grow
  # into a share of the machine on a long run. A shell that cannot time itself
  # keeps the one-second cadence.
  local now_us
  now_us="$(_ns_now_us)"
  if [ -n "${now_us}" ]; then
    [ "$(_ns_backoff_interval "$((now_us - 40000))")" = "1" ] || {
      echo "FAIL: a 40ms draw must keep the one-second cadence"; exit 1
    }
    [ "$(_ns_backoff_interval "$((now_us - 720000))")" = "6" ] || {
      echo "FAIL: a 720ms draw must back off: $(_ns_backoff_interval "$((now_us - 720000))")"; exit 1
    }
    [ "$(_ns_backoff_interval "$((now_us - 60000000))")" = "30" ] || {
      echo "FAIL: the back-off must be capped"; exit 1
    }
    # The stall watchdog's poll is the same back-off with the caller's own
    # floor and a cap sized to the timeout rather than to a human's patience.
    [ "$(_ns_backoff_interval "$((now_us - 200000))" 60 600)" = "60" ] || {
      echo "FAIL: a cheap scan must keep the watchdog's poll floor"; exit 1
    }
    [ "$(_ns_backoff_interval "$((now_us - 20000000))" 60 600)" = "180" ] || {
      echo "FAIL: an expensive scan must back the watchdog off: $(_ns_backoff_interval "$((now_us - 20000000))" 60 600)"
      exit 1
    }
    [ "$(_ns_backoff_interval "$((now_us - 600000000))" 60 600)" = "600" ] || {
      echo "FAIL: the watchdog's back-off must be capped"; exit 1
    }
  fi
  [ "$(_ns_backoff_interval "")" = "1" ] || { echo "FAIL: no clock means no back-off"; exit 1; }
  [ "$(_ns_backoff_interval "" 60 600)" = "60" ] || {
    echo "FAIL: no clock means the caller's floor"; exit 1
  }

  [ "$(_ns_dead_now 100 1000 100)" = "111" ] || { echo "FAIL: dead_now: $(_ns_dead_now 100 1000 100)"; exit 1; }
  [ "$(_ns_dead_now 10 100 5)" = "11" ] || { echo "FAIL: dead_now rounds up"; exit 1; }
  [ "$(_ns_dead_now 10 100 4)" = "10" ] || { echo "FAIL: dead_now rounds down"; exit 1; }
  [ "$(_ns_dead_now 0 1000 100)" = "0" ] || { echo "FAIL: dead_now with no dead points"; exit 1; }
  [ "$(_ns_dead_now 100 1000 0)" = "100" ] || { echo "FAIL: dead_now on a fresh checkpoint"; exit 1; }
  [ "$(_ns_dead_now 100 100 100)" = "100" ] || { echo "FAIL: dead_now with nothing banked"; exit 1; }

  [ "$(_ns_format_elapsed 3661)" = "1h01m" ] || { echo "FAIL: format_elapsed"; exit 1; }
  [ "$(_ns_format_elapsed 59)" = "59s" ] || { echo "FAIL: format_elapsed short"; exit 1; }
  [ "$(_ns_format_elapsed 65)" = "1m05s" ] || { echo "FAIL: format_elapsed minutes"; exit 1; }
  [ "$(_ns_format_elapsed 3599)" = "59m59s" ] || { echo "FAIL: format_elapsed hour boundary"; exit 1; }
  [ "$(_ns_format_elapsed 194400)" = "2d 6h" ] || { echo "FAIL: format_elapsed days"; exit 1; }

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
    *"~ 55%"*"~10/~18 dead points"*"eta ~48s"*"evidence tolerance"*) ;;
    *) echo "FAIL: status line did not switch to the evidence total: ${line}"; exit 1 ;;
  esac
  # ...and with two of the four evaluations landing after that checkpoint, the
  # 10 banked over 2 carry forward to 20 - which is past the estimated total,
  # so the percent clamps at 100 rather than printing "111%" on a run that is
  # about to stop anyway. The eta does *not* become 0s: the total came
  # from the checkpoint and is as stale as it is, so overtaking it means the
  # estimate is out of date rather than that the run has finished.
  touch -t 202601010100 "${dead_file}"
  touch -t 202601010200 "${tmp}/evaluations/eval-0003-c" "${tmp}/evaluations/eval-0004-d"
  line="$(_ns_status_line "${tmp}" -1 2 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"~100%"*"~20/~18 dead points"*"eta ~?"*"past the ~18 estimate; the next checkpoint revises it"*) ;;
    *) echo "FAIL: carried count past the estimated total: ${line}"; exit 1 ;;
  esac
  case "${line}" in
    *"eta ~0s"*|*"evidence tolerance"*)
      echo "FAIL: past the estimate still claims it is done: ${line}"; exit 1 ;;
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
  local progressing='n=$(ls "$0"/evaluations 2>/dev/null | wc -l | tr -d " ");
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

  # Every retry gives the sidecars a chance to come back first. A container
  # that died takes every later attempt with it - the `docker exec` fails
  # instantly and the no-progress guard above then calls the run
  # deterministic - so the hook has to fire once per retry, not once per run.
  # Called by name because start-sidecars.sh is sourced by the run scripts and
  # not by this file, which keeps the retry loop runnable without docker.
  local hook_dir="${tmp}/hook"
  mkdir -p "${hook_dir}/chains"
  sidecar_restore() { echo r >>"${hook_dir}/restores"; }
  run_with_retries 2 "${hook_dir}" -1 2 -- sh -c "${progressing}" "${hook_dir}" \
    >/dev/null 2>&1 || true
  unset -f sidecar_restore
  [ "$(_ns_count_lines "${hook_dir}/restores")" = "2" ] || {
    echo "FAIL: sidecar_restore called $(_ns_count_lines "${hook_dir}/restores") times, want 2"
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

  # The one no-progress failure that is *not* deterministic: a truncated
  # PolyChord checkpoint, which aborts in Fortran before evaluation 1 and then
  # kills every later `./ri resume` the same way. The checkpoint has to be
  # moved aside and the attempt retried, or the search is stuck for good with
  # its scored evaluations unreachable. The fixture fails the way a real one
  # does - the gfortran error naming read_write.F90 - and succeeds once the
  # checkpoint is gone, which is exactly what starting the sampler from
  # scratch does.
  local torn_dir="${tmp}/torn"
  mkdir -p "${torn_dir}/chains"
  echo truncated >"${torn_dir}/chains/r.resume"
  status=0
  # shellcheck disable=SC2016
  run_with_retries 2 "${torn_dir}" -1 2 -- sh -c '
    echo a >>"$0"/attempts
    [ -f "$0"/chains/r.resume ] || exit 0
    echo "At line 354 of file read_write.F90 (unit = 10, file = '\''$0/chains/r.resume'\'')"
    echo "Fortran runtime error: End of file"
    exit 2' "${torn_dir}" >/dev/null 2>&1 || status=$?
  [ "${status}" = "0" ] || { echo "FAIL: torn checkpoint exit ${status}, want 0"; exit 1; }
  [ "$(_ns_count_lines "${torn_dir}/attempts")" = "2" ] || {
    echo "FAIL: torn checkpoint attempts $(_ns_count_lines "${torn_dir}/attempts"), want 2"; exit 1
  }
  [ -f "${torn_dir}/chains/r.resume.unreadable" ] || {
    echo "FAIL: torn checkpoint not kept as evidence"; exit 1
  }
  [ ! -e "${torn_dir}/chains/r.resume" ] || {
    echo "FAIL: torn checkpoint left in place, so the restart reads it again"; exit 1
  }
  # ./ri health reads this file, and a run that healed itself this way healed
  # itself like any other.
  [ "$(_ns_count_lines "${torn_dir}/restarts.log")" = "1" ] || {
    echo "FAIL: checkpoint recovery left no restart line"; exit 1
  }

  # A fault that tears the checkpoint again every time it is written - a full
  # disk - is bounded by the retry budget like any other, not by capping the
  # recovery at one: a multi-day search that healed itself this way on day one
  # must still be able to on day three.
  local torn_again_dir="${tmp}/torn-again"
  mkdir -p "${torn_again_dir}/chains"
  echo truncated >"${torn_again_dir}/chains/r.resume"
  # shellcheck disable=SC2016
  run_with_retries 2 "${torn_again_dir}" -1 2 -- sh -c '
    echo a >>"$0"/attempts; echo truncated >"$0"/chains/r.resume
    echo "At line 354 of file read_write.F90"; exit 2' \
    "${torn_again_dir}" >/dev/null 2>&1 || true
  [ "$(_ns_count_lines "${torn_again_dir}/attempts")" = "3" ] || {
    echo "FAIL: repeated checkpoint tear ran $(_ns_count_lines "${torn_again_dir}/attempts")" \
      "attempts, want 1 + 2 retries"; exit 1
  }

  # And only on that evidence. A missing image or a bad parameter space also
  # scores nothing, and a good checkpoint thrown away for one of those costs a
  # long search its sampler position for nothing.
  local intact_dir="${tmp}/intact"
  mkdir -p "${intact_dir}/chains"
  echo fine >"${intact_dir}/chains/r.resume"
  # shellcheck disable=SC2016
  run_with_retries 2 "${intact_dir}" -1 2 -- \
    sh -c 'echo a >>"$0"/attempts; echo "docker: no such image" >&2; exit 5' \
    "${intact_dir}" >/dev/null 2>&1 || true
  [ "$(_ns_count_lines "${intact_dir}/attempts")" = "1" ] || {
    echo "FAIL: unrelated failure retried $(_ns_count_lines "${intact_dir}/attempts") times"; exit 1
  }
  [ "$(cat "${intact_dir}/chains/r.resume")" = "fine" ] || {
    echo "FAIL: good checkpoint moved aside for an unrelated failure"; exit 1
  }

  # The evidence has to come from the attempt that just failed, not from
  # anywhere in run.log: the file is appended to across attempts, so an
  # earlier checkpoint failure this loop already recovered from would
  # otherwise condemn a good checkpoint written since. Attempt 1 hits it and
  # scores an evaluation, attempt 2 fails for an unrelated reason.
  local stale_dir="${tmp}/stale"
  mkdir -p "${stale_dir}/chains"
  echo fine >"${stale_dir}/chains/r.resume"
  # shellcheck disable=SC2016
  run_with_retries 2 "${stale_dir}" -1 2 -- sh -c '
    if [ ! -f "$0"/attempts ]; then
      mkdir -p "$0"/evaluations/eval-0; echo {} >"$0"/evaluations/eval-0/metrics.json
      echo "At line 354 of file read_write.F90 (unit = 10, file = r.resume)"
    fi
    echo a >>"$0"/attempts; exit 5' "${stale_dir}" >/dev/null 2>&1 || true
  [ "$(cat "${stale_dir}/chains/r.resume")" = "fine" ] || {
    echo "FAIL: an earlier attempt's checkpoint error condemned a later good one"; exit 1
  }

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

  # A restart re-sizes itself against the memory free now. The rank count in
  # the command is `ns_budget_ranks`' answer from when the run started, and
  # the host it fitted then is several sessions' host now - so a restart that
  # replays it lands in the OOM killer, which PolyChord scores as the best
  # points of the search rather than as a failure.
  #
  # The fixture is the `progressing` command with an mpirun-shaped tail, and
  # it writes back the arguments it was actually called with: that is the only
  # place the rewrite is observable. `docker` is stubbed and the reservation
  # directory is private because `ns_budget_ranks` reaps leaked sidecars and
  # writes a real reservation - neither belongs in a self-check on a host
  # other sessions are running searches on.
  mkdir -p "${tmp}/bin"
  printf '#!/bin/sh\nexit 0\n' >"${tmp}/bin/docker"
  chmod +x "${tmp}/bin/docker"
  local resize_dir="${tmp}/resize"
  mkdir -p "${resize_dir}/chains"
  # shellcheck disable=SC2016  # $0 and $* are the child `sh`'s, not ours
  local recording='n=$(ls "$0"/evaluations 2>/dev/null | wc -l | tr -d " ");
    mkdir -p "$0"/evaluations/eval-$n; echo {} >"$0"/evaluations/eval-$n/metrics.json;
    echo "$*" >>"$0"/attempts; exit 5'
  status=0
  # 4096MB of headroom plus two R2D2 ranks, against a command asking for 8.
  # shellcheck disable=SC2030,SC2031  # the subshell is the point: each case
  # gets its own budget dir, PATH and free-memory reading and leaks neither
  (
    export NS_RANK_BUDGET_DIR="${tmp}/budget" PATH="${tmp}/bin:${PATH}"
    export NS_AVAILABLE_MB=$((4096 + 2 * 3500))
    run_with_retries 1 "${resize_dir}" -1 2 -- sh -c "${recording}" "${resize_dir}" \
      -e NS_MPI_PROCS=8 -np 8 python3 /opt/ri-nested-sampling/polychord_r2d2.py
  ) >/dev/null 2>&1 || status=$?
  [ "${status}" = "5" ] || { echo "FAIL: re-sized retry exit ${status}, want 5"; exit 1; }
  # The first attempt runs what the run script sized; only the restart moves.
  case "$(sed -n 1p "${resize_dir}/attempts")" in
    *"NS_MPI_PROCS=8"*"-np 8 "*) ;;
    *) echo "FAIL: first attempt rewritten: $(sed -n 1p "${resize_dir}/attempts")"; exit 1 ;;
  esac
  # Both spellings of the rank count, because the ranks read NS_MPI_PROCS for
  # summary.json while mpirun reads -np, and a run whose two disagree reports
  # a rank count it never ran at.
  case "$(sed -n 2p "${resize_dir}/attempts")" in
    *"NS_MPI_PROCS=2"*"-np 2 "*) ;;
    *) echo "FAIL: restart not re-sized: $(sed -n 2p "${resize_dir}/attempts")"; exit 1 ;;
  esac

  # Memory the run had and no longer has: a restart that cannot fit even one
  # rank is not a restart worth making, so it stops here rather than spending
  # the budget failing the same way.
  local starved_dir="${tmp}/starved"
  mkdir -p "${starved_dir}/chains"
  status=0
  # shellcheck disable=SC2030,SC2031  # the subshell is the point: each case
  # gets its own budget dir, PATH and free-memory reading and leaks neither
  (
    export NS_RANK_BUDGET_DIR="${tmp}/budget" PATH="${tmp}/bin:${PATH}"
    export NS_AVAILABLE_MB=$((4096 + 100))
    run_with_retries 1 "${starved_dir}" -1 2 -- sh -c "${recording}" "${starved_dir}" \
      -e NS_MPI_PROCS=8 -np 8 python3 /opt/ri-nested-sampling/polychord_r2d2.py
  ) >/dev/null 2>&1 || status=$?
  [ "${status}" = "5" ] || { echo "FAIL: starved retry exit ${status}, want 5"; exit 1; }
  [ "$(_ns_count_lines "${starved_dir}/attempts")" = "1" ] || {
    echo "FAIL: retried with no memory for a rank"; exit 1
  }
  [ ! -e "${starved_dir}/restarts.log" ] || {
    echo "FAIL: restart logged for a retry that never ran"; exit 1
  }
  grep -q "memory for even one rank" "${starved_dir}/run.log" || {
    echo "FAIL: run.log does not say why the restart was refused"; exit 1
  }

  # And a host with room leaves the command exactly as it was: the re-clamp
  # is a guard, not a rewrite that happens every time.
  local roomy_dir="${tmp}/roomy"
  mkdir -p "${roomy_dir}/chains"
  status=0
  # shellcheck disable=SC2030,SC2031  # the subshell is the point: each case
  # gets its own budget dir, PATH and free-memory reading and leaks neither
  (
    export NS_RANK_BUDGET_DIR="${tmp}/budget" PATH="${tmp}/bin:${PATH}"
    export NS_AVAILABLE_MB=$((4096 + 64 * 3500))
    run_with_retries 1 "${roomy_dir}" -1 2 -- sh -c "${recording}" "${roomy_dir}" \
      -e NS_MPI_PROCS=8 -np 8 python3 /opt/ri-nested-sampling/polychord_r2d2.py
  ) >/dev/null 2>&1 || status=$?
  [ "$(sed -n 1p "${roomy_dir}/attempts")" = "$(sed -n 2p "${roomy_dir}/attempts")" ] || {
    echo "FAIL: command changed with memory to spare: $(sed -n 2p "${roomy_dir}/attempts")"
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

  # The retry budget is for a crash loop, not for the run's lifetime: an
  # attempt that ran a long time before dying hands it back, so a multi-day
  # search is not out of restarts for the rest of the week because it healed
  # itself twice on day one.
  #
  # One fixture, run twice, so the only difference between the two is the
  # reset. It fails three times and then succeeds, against a budget of 1:
  # with the reset off that is 2 attempts and exit 5, with it on 4 attempts
  # and exit 0. The reset is forced by setting the window to 0 rather than by
  # making the fixture slow - these attempts take milliseconds, and a fixture
  # that had to outlive a real window would put minutes into the self-check.
  local reset_dir="${tmp}/reset"
  mkdir -p "${reset_dir}/chains"
  # shellcheck disable=SC2016
  local flaky='n=$(ls "$0"/evaluations 2>/dev/null | wc -l | tr -d " ");
    mkdir -p "$0"/evaluations/eval-$n; echo {} >"$0"/evaluations/eval-$n/metrics.json;
    echo a >>"$0"/attempts;
    [ "$(wc -l <"$0"/attempts)" -ge 4 ] && exit 0; exit 5'
  status=0
  run_with_retries 1 "${reset_dir}" -1 2 -- sh -c "${flaky}" "${reset_dir}" \
    >/dev/null 2>&1 || status=$?
  [ "${status}" = "5" ] || { echo "FAIL: quick crash loop exit ${status}, want 5"; exit 1; }
  [ "$(_ns_count_lines "${reset_dir}/attempts")" = "2" ] || {
    echo "FAIL: crash loop ran $(_ns_count_lines "${reset_dir}/attempts") attempts, want 1 + 1 retry"
    exit 1
  }
  # And it says why it gave up. run.log is the only artifact recording that,
  # and "the run just stopped" with nothing about the exhausted budget is the
  # report this whole path exists to avoid.
  grep -q "not retrying: 1 of 1 restarts used" "${reset_dir}/run.log" || {
    echo "FAIL: exhausted retry budget not explained in run.log"; exit 1
  }

  local long_dir="${tmp}/reset-long"
  mkdir -p "${long_dir}/chains"
  status=0
  # Set and unset rather than prefixed onto the call: an assignment prefixing
  # a shell function can outlive it, which would silently disable the reset
  # window for every case below.
  NS_RETRY_RESET_SECONDS=0
  run_with_retries 1 "${long_dir}" -1 2 -- sh -c "${flaky}" "${long_dir}" \
    >/dev/null 2>&1 || status=$?
  [ "${status}" = "0" ] || { echo "FAIL: long-lived attempts exit ${status}, want 0"; exit 1; }
  [ "$(_ns_count_lines "${long_dir}/attempts")" = "4" ] || {
    echo "FAIL: reset gave $(_ns_count_lines "${long_dir}/attempts") attempts, want 4"; exit 1
  }
  # ...and the reset never turns "no retries" into one: 0 is still 0.
  local never_dir="${tmp}/reset-never"
  mkdir -p "${never_dir}/chains"
  status=0
  run_with_retries 0 "${never_dir}" -1 2 -- sh -c "${flaky}" "${never_dir}" \
    >/dev/null 2>&1 || status=$?
  unset NS_RETRY_RESET_SECONDS
  [ "$(_ns_count_lines "${never_dir}/attempts")" = "1" ] || {
    echo "FAIL: retries=0 retried under the reset"; exit 1
  }

  # _ns_stall_watchdog: a run that is alive but landing nothing is killed so
  # the retry loop can act on it, and one that keeps scoring is left alone.
  #
  # Both fixtures are real processes spelling their run the way a rank does
  # (`polychord_*.py ... --output-dir <dir>`), because the kill is by command
  # line - a fixture with any other argv would be "killed" by a watchdog that
  # matches nothing, and pass.
  cat >"${tmp}/polychord_stall.py" <<'PY'
import os, sys, time
run = sys.argv[sys.argv.index("--output-dir") + 1]
evals = os.path.join(run, "evaluations")
os.makedirs(evals, exist_ok=True)
banked = os.path.join(evals, "eval-%d" % len(os.listdir(evals)))
os.makedirs(banked)
open(os.path.join(banked, "metrics.json"), "w").write("{}")
with open(os.path.join(run, "attempts"), "a") as fh:
    fh.write("a\n")
if os.environ.get("STALL_FIXTURE_HANGS"):
    # Bounded, so a watchdog that never fires ends this check with a failure
    # rather than with a hang - the same fault it exists to catch.
    time.sleep(25)
else:
    # Slower than the poll interval and faster than the timeout, for longer
    # than the timeout: each poll window that sees no change must be forgiven
    # by the next evaluation rather than added to the last one.
    for i in range(8):
        time.sleep(1.5)
        step = os.path.join(evals, "eval-live-%d" % i)
        os.makedirs(step)
        open(os.path.join(step, "metrics.json"), "w").write("{}")
PY
  # Exported, not just set: the NS_STALL_TIMEOUT=0 case below runs in its own
  # bash, and a plain assignment would leave it reading the unset default
  # there - which is the same "off" it is trying to prove, so the check would
  # pass whatever the guard did.
  export NS_STALL_TIMEOUT=2 NS_STALL_POLL_SECONDS=1

  # Scored one evaluation, then hung. Killed, and retried because it had
  # banked work - the whole point of turning a hang into an exit.
  local hung_dir="${tmp}/hung" hung_start hung_elapsed
  mkdir -p "${hung_dir}/chains"
  hung_start="$(date +%s)"
  status=0
  STALL_FIXTURE_HANGS=1 run_with_retries 1 "${hung_dir}" -1 2 -- \
    python3 "${tmp}/polychord_stall.py" --output-dir "${hung_dir}" >/dev/null 2>&1 || status=$?
  hung_elapsed=$(($(date +%s) - hung_start))
  [ "${status}" != "0" ] || { echo "FAIL: a hung run exited 0"; exit 1; }
  [ "${hung_elapsed}" -lt 60 ] || { echo "FAIL: hung run took ${hung_elapsed}s to be killed"; exit 1; }
  [ "$(_ns_count_lines "${hung_dir}/attempts")" = "2" ] || {
    echo "FAIL: hung run ran $(_ns_count_lines "${hung_dir}/attempts") attempts, want 1 + 1 retry"
    exit 1
  }
  grep -q 'no evaluation has finished' "${hung_dir}/run.log" || {
    echo "FAIL: run.log does not say why the run was killed"; exit 1
  }

  # The false positive that would be worse than the bug: a run whose gaps are
  # each shorter than the timeout but which, added together, run well past it.
  local live_dir="${tmp}/live"
  mkdir -p "${live_dir}/chains"
  run_with_progress "${live_dir}" -1 2 -- \
    python3 "${tmp}/polychord_stall.py" --output-dir "${live_dir}" >/dev/null 2>&1 || {
    echo "FAIL: a run that kept scoring evaluations was killed"; exit 1
  }
  # And the watchdog is reaped with it: a leaked one would kill the *next*
  # attempt into the same directory, turning one retry into an endless loop.
  sleep 3
  grep -q 'no evaluation has finished' "${live_dir}/run.log" && {
    echo "FAIL: watchdog outlived the run it was watching"; exit 1
  }

  # The other way a watchdog outlives its run: a signal rather than an exit.
  # It is a forked subshell, so without a trap Ctrl-C or a SIGTERM to the run
  # script left it polling for the rest of NS_STALL_TIMEOUT - two hours by
  # default, and two were found orphaned on this host against run directories
  # that had already been deleted.
  local term_dir="${tmp}/term" watchdog
  mkdir -p "${term_dir}/chains"
  export NS_STALL_TIMEOUT=600 NS_STALL_POLL_SECONDS=1
  # Marked through $0 so the watchdog, which inherits the whole command line
  # from the fork, is countable; $0 is deliberately not this file's path, for
  # the reason spelled out in the NS_STALL_TIMEOUT=0 case below.
  # shellcheck disable=SC2016  # $1..$3 are the sub-shell's, not ours
  STALL_FIXTURE_HANGS=1 bash -c '
    . "$1"; run_with_progress "$2" -1 2 -- python3 "$3" --output-dir "$2"' \
    ns-watchdog-signal "${BASH_SOURCE[0]}" "${term_dir}" "${tmp}/polychord_stall.py" \
    >/dev/null 2>&1 &
  local term_pid=$!
  sleep 3
  # The watchdog by pid, not by a count: without the traps the signal kills the
  # run shell outright and leaves the watchdog behind, so "one process left"
  # is what both the fix and the bug produce. Found as the run shell's own
  # child rather than by the marker alone, so that any other process on the
  # host carrying the same string - the shell that started this check, for one
  # - cannot be mistaken for it.
  watchdog="$(pgrep -P "${term_pid}" -f ns-watchdog-signal | head -1)"
  [ -n "${watchdog}" ] || {
    echo "FAIL: no stall watchdog was started to leak"
    kill -9 "${term_pid}" 2>/dev/null || true
    exit 1
  }
  kill -TERM "${term_pid}" 2>/dev/null || true
  # The fixture is left running on purpose: killing it would end the run
  # through the normal path, which stops the watchdog whatever the traps do.
  sleep 2
  kill -0 "${watchdog}" 2>/dev/null && {
    kill -9 "${watchdog}" "${term_pid}" 2>/dev/null || true
    pkill -9 -f "$(ns_run_process_pattern "${term_dir}")" 2>/dev/null || true
    echo "FAIL: SIGTERM left the stall watchdog running"; exit 1
  }
  # Only the fixture, so the run shell ends by itself: SIGKILLing a background
  # job makes the shell print its own "Killed" notification over the output.
  pkill -9 -f "$(ns_run_process_pattern "${term_dir}")" 2>/dev/null || true
  wait "${term_pid}" 2>/dev/null || true
  unset NS_STALL_TIMEOUT NS_STALL_POLL_SECONDS

  # 0 is off, which is what every caller that does not set it gets.
  local off_dir="${tmp}/off"
  mkdir -p "${off_dir}/chains"
  export NS_STALL_TIMEOUT=0
  # $0 is deliberately not this file's path: sourcing it with $0 set to it
  # makes the `BASH_SOURCE == $0` guard at the bottom true, and the sub-shell
  # re-runs this whole self-check instead of the one line it was asked for -
  # which then hits the 8s timeout and "passes" whatever the guard does.
  export STALL_FIXTURE_HANGS=1
  status=0
  # shellcheck disable=SC2016  # $1..$3 are the sub-shell's, not ours
  _ns_run_bounded 8 bash -c '
    . "$1"; run_with_progress "$2" -1 2 -- python3 "$3" --output-dir "$2"' \
    watchdog-off "${BASH_SOURCE[0]}" "${off_dir}" "${tmp}/polychord_stall.py" \
    >/dev/null 2>&1 || status=$?
  unset STALL_FIXTURE_HANGS
  [ "${status}" = "124" ] || {
    echo "FAIL: NS_STALL_TIMEOUT=0 still killed the run (exit ${status}, want 124)"; exit 1
  }
  pkill -9 -f "$(ns_run_process_pattern "${off_dir}")" 2>/dev/null || true
  unset NS_STALL_TIMEOUT NS_STALL_POLL_SECONDS

  # The pattern the kill and the resume guard share: a rank of this run, and
  # not a run whose directory name this one is a prefix of.
  echo "polychord_r2d2.py --output-dir ${hung_dir} --nlive 50" \
    | grep -Eq "$(ns_run_process_pattern "${hung_dir}")" || {
    echo "FAIL: a rank of the run is not matched"; exit 1
  }
  echo "polychord_r2d2.py --output-dir ${hung_dir}-other --nlive 50" \
    | grep -Eq "$(ns_run_process_pattern "${hung_dir}")" && {
    echo "FAIL: a neighbouring run is matched"; exit 1
  }

  rm -rf "${tmp}"
  echo "progress-bar self-check passed"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  self_check
fi
