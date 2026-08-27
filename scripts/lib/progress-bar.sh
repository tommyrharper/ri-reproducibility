#!/usr/bin/env bash
# A pinned status line for a nested-sampling search: elapsed time, dead
# points against the run's --max-ndead cap (when it has one) with a percent
# and ETA extrapolated from the rate so far, plus the raw evaluation count.
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

_ns_count_glob() {
  local dir="$1" pattern="$2" n=0 f
  for f in "${dir}"/${pattern}; do
    [ -e "${f}" ] || continue
    n=$((n + 1))
  done
  echo "${n}"
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
  local now elapsed dead_file dead_count eval_count

  now="$(date +%s)"
  elapsed=$((now - start))
  dead_file="$(_ns_dead_birth_file "${output_dir}/chains")"
  dead_count=0
  [ -n "${dead_file}" ] && dead_count="$(_ns_count_lines "${dead_file}")"
  eval_count="$(_ns_count_glob "${output_dir}/evaluations" 'eval-*')"

  # PolyChord treats max_ndead <= 0 as "no bound, stop on evidence tolerance
  # instead" - there's no dead-point budget to measure a percent or ETA
  # against, but the tolerance itself is a real number we can approximate
  # progress against instead (see _ns_evidence_pct).
  if [ "${max_ndead}" -le 0 ]; then
    local evidence_pct
    if evidence_pct="$(_ns_evidence_pct "${output_dir}/chains" "${dead_count}" "${nlive}")"; then
      _ns_evidence_line "${evidence_pct}" "${dead_count}" "${eval_count}" "${elapsed}"
    else
      _ns_unbounded_line "${dead_count}" "${eval_count}" "${elapsed}"
    fi
    return
  fi

  local pct eta bar
  pct=$((dead_count * 100 / max_ndead))
  [ "${pct}" -gt 100 ] && pct=100

  eta="?"
  if [ "${dead_count}" -gt 0 ] && [ "$((max_ndead - dead_count))" -gt 0 ]; then
    eta="$(_ns_format_hms $(((max_ndead - dead_count) * elapsed / dead_count)))"
  elif [ "${dead_count}" -ge "${max_ndead}" ]; then
    eta="0:00:00"
  fi

  bar="$(_ns_render_bar "${pct}" 30)"

  printf '[%s] %3d%%  %d/%d dead points (%d evaluations)  elapsed %s  eta %s' \
    "${bar}" "${pct}" "${dead_count}" "${max_ndead}" "${eval_count}" "$(_ns_format_hms "${elapsed}")" "${eta}"
}

# A block bouncing back and forth across the bar, one step per elapsed
# second, so a no-budget run still visibly ticks over instead of sitting
# frozen. Position, not percent: there is nothing to be a percent of. Used
# only until _ns_evidence_pct has enough data to give a real number (before
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

_ns_evidence_line() {
  local pct="$1" dead_count="$2" eval_count="$3" elapsed="$4"
  local bar
  bar="$(_ns_render_bar "${pct}" 30)"
  printf '[%s] %3d%%  %d dead points (%d evaluations)  elapsed %s  (evidence tolerance, no --max-ndead cap)' \
    "${bar}" "${pct}" "${dead_count}" "${eval_count}" "$(_ns_format_hms "${elapsed}")"
}

# PolyChord's stopping test (source: PolyChordLite's nested_sampling.F90 -
# `live_logZ(...) < log(precision_criterion) + RTI%logZ`) is not currently
# exposed as a --flag by this repo: neither polychord_wsclean.py nor
# polychord_r2d2.py override it, so pypolychord's own default applies.
# Update this if that ever changes.
_NS_PRECISION_CRITERION=0.001

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

# Approximates how close an uncapped run (--max-ndead <= 0) is to PolyChord's
# real stopping test, from the same output files ./ri report and
# scripts/lib/nested_sampling/anesthetic_io.py already read: chains/*.stats
# for the accumulated log(Z) and chains/*_phys_live.txt for the live points'
# current log-likelihoods (last column of each row). PolyChord's own
# live_logZ is logsumexp(live loglikes) - log(nlive) + logXp, where logXp
# (the remaining prior volume) is tracked per-cluster internally; this
# approximates it as a single global -ndead/nlive, the textbook single-
# cluster expectation. Good enough to watch a number climb toward 100 - a
# run whose live points split across several clusters (chains/*.stats'
# `ncluster` line) will see this diverge further from PolyChord's own exact
# per-cluster figure than a single-cluster run does.
_ns_evidence_pct() {
  local chains_dir="$1" dead_count="$2" nlive="$3"
  local stats_file live_file logz
  [ "${dead_count}" -gt 0 ] && [ "${nlive}" -gt 0 ] || return 1
  stats_file="$(_ns_stats_file "${chains_dir}")"
  live_file="$(_ns_phys_live_file "${chains_dir}")"
  [ -n "${stats_file}" ] && [ -n "${live_file}" ] || return 1
  logz="$(grep -m1 '^log(Z)' "${stats_file}" 2>/dev/null | awk '{print $3}')"
  [ -n "${logz}" ] || return 1
  awk -v logz="${logz}" -v ndead="${dead_count}" -v nlive="${nlive}" -v pc="${_NS_PRECISION_CRITERION}" '
    { n++; ll[n] = $NF }
    END {
      if (n == 0) exit 1
      max = ll[1]
      for (i = 2; i <= n; i++) if (ll[i] > max) max = ll[i]
      s = 0
      for (i = 1; i <= n; i++) s += exp(ll[i] - max)
      live_logz = (max + log(s)) - log(n) + (-ndead / nlive)
      ratio = exp(live_logz - logz)
      if (ratio <= 0) exit 1
      pct = 100 * pc / ratio
      if (pct > 100) pct = 100
      if (pct < 0) pct = 0
      printf "%d", pct
    }
  ' "${live_file}"
}

self_check() {
  local tmp
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
  [ "$(_ns_count_glob "${tmp}/evaluations" 'eval-*')" = "4" ] || { echo "FAIL: evaluations count"; exit 1; }

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
  local line
  line="$(_ns_status_line "${tmp}" 12 8 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"3/12 dead points (4 evaluations)"*) ;;
    *) echo "FAIL: status line dead/eval split: ${line}"; exit 1 ;;
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
  # (--max-ndead <= 0) should switch from the ambiguous bounce to a real
  # percent, approximating PolyChord's own stopping test (nlive=2,
  # ndead=10, both live points at loglike 0, log(Z)=0.0 -> 14%, checked by
  # hand against the same formula this function implements).
  printf 'log(Z)       =   0.0 +/-   0.1\n' >"${tmp}/chains/wsclean_vlaa.stats"
  printf '0.1 0.2 0.0\n0.3 0.4 0.0\n' >"${tmp}/chains/wsclean_vlaa_phys_live.txt"
  [ "$(_ns_evidence_pct "${tmp}/chains" 10 2)" = "14" ] || {
    echo "FAIL: evidence pct: $(_ns_evidence_pct "${tmp}/chains" 10 2)"; exit 1
  }
  # _ns_status_line reads ndead from chains/*_dead-birth.txt itself, so match
  # the 10 dead points the hand-computed 14% above assumes.
  seq 1 10 >"${tmp}/chains/wsclean_vlaa_dead-birth.txt"
  line="$(_ns_status_line "${tmp}" -1 2 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"14%"*"evidence tolerance"*) ;;
    *) echo "FAIL: status line did not switch to evidence pct: ${line}"; exit 1 ;;
  esac
  # No dead points yet means no ndead/nlive to divide by - must fall back to
  # the bounce rather than divide by zero or report a bogus percent.
  _ns_evidence_pct "${tmp}/chains" 0 2 >/dev/null && { echo "FAIL: evidence pct with 0 dead points should fail"; exit 1; }

  # _ns_add_trap must append to an existing trap, not replace it - a naive
  # `trap ... EXIT` here would silently disable start-sidecars.sh's Docker
  # cleanup on exit/Ctrl-C.
  (
    log="${tmp}/trap.log"
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
  run_with_progress "${run_dir}" -1 2 -- \
    sh -c 'i=0; while [ $i -lt 20000 ]; do echo "line-$i"; i=$((i + 1)); done; exit 3' \
    >/dev/null 2>&1
  [ "$(tail -n 1 "${run_dir}/run.log")" = "line-19999" ] || {
    echo "FAIL: last line lost: $(tail -n 1 "${run_dir}/run.log")"; exit 1
  }
  # And the pipe it went through is not left behind in the run directory.
  [ ! -e "${run_dir}/.run.log.fifo" ] || { echo "FAIL: fifo left behind"; exit 1; }

  rm -rf "${tmp}"
  echo "progress-bar self-check passed"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  self_check
fi
