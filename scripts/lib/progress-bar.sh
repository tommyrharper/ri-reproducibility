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

# Usage: run_with_progress <output_dir> <max_ndead> -- cmd args...
run_with_progress() {
  local output_dir="$1" max_ndead="$2"
  shift 2
  [ "${1:-}" = "--" ] && shift

  "$@" &
  local pid=$!
  local start
  start="$(date +%s)"

  if [ -t 1 ] && _ns_pin_setup; then
    _ns_add_trap '_ns_pin_teardown' EXIT
    _ns_add_trap '_ns_pin_teardown' INT
    _ns_add_trap '_ns_pin_teardown' TERM
    while kill -0 "${pid}" 2>/dev/null; do
      _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${start}")"
      sleep 1
    done
    _ns_pin_draw "$(_ns_status_line "${output_dir}" "${max_ndead}" "${start}")"
    _ns_pin_teardown
  elif [ -t 1 ]; then
    # No usable terminal control (e.g. tput/TERM missing): fall back to a
    # plain redrawn line instead of a pinned one.
    while kill -0 "${pid}" 2>/dev/null; do
      printf '\r%s' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${start}")")"
      sleep 1
    done
    printf '\r%s\n' "$(_ns_truncate_pad "$(_ns_status_line "${output_dir}" "${max_ndead}" "${start}")")"
  fi

  wait "${pid}"
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

_ns_status_line() {
  local output_dir="$1" max_ndead="$2" start="$3"
  local now elapsed dead_file dead_count eval_count

  now="$(date +%s)"
  elapsed=$((now - start))
  dead_file="$(_ns_dead_birth_file "${output_dir}/chains")"
  dead_count=0
  [ -n "${dead_file}" ] && dead_count="$(_ns_count_lines "${dead_file}")"
  eval_count="$(_ns_count_glob "${output_dir}/evaluations" 'eval-*')"

  # PolyChord treats max_ndead <= 0 as "no bound, stop on evidence tolerance
  # instead" - there's no budget to measure a percent or ETA against.
  if [ "${max_ndead}" -le 0 ]; then
    _ns_unbounded_line "${dead_count}" "${eval_count}" "${elapsed}"
    return
  fi

  local pct eta bar_width filled bar
  pct=$((dead_count * 100 / max_ndead))
  [ "${pct}" -gt 100 ] && pct=100

  eta="?"
  if [ "${dead_count}" -gt 0 ] && [ "$((max_ndead - dead_count))" -gt 0 ]; then
    eta="$(_ns_format_hms $(((max_ndead - dead_count) * elapsed / dead_count)))"
  elif [ "${dead_count}" -ge "${max_ndead}" ]; then
    eta="0:00:00"
  fi

  bar_width=30
  filled=$((pct * bar_width / 100))
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')$(printf '%*s' $((bar_width - filled)) '')"

  printf '[%s] %3d%%  %d/%d dead points (%d evaluations)  elapsed %s  eta %s' \
    "${bar}" "${pct}" "${dead_count}" "${max_ndead}" "${eval_count}" "$(_ns_format_hms "${elapsed}")" "${eta}"
}

# A block bouncing back and forth across the bar, one step per elapsed
# second, so a no-budget run still visibly ticks over instead of sitting
# frozen. Position, not percent: there is nothing to be a percent of.
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
  line="$(_ns_status_line "${tmp}" 12 "$(($(date +%s) - 60))")"
  case "${line}" in
    *"3/12 dead points (4 evaluations)"*) ;;
    *) echo "FAIL: status line dead/eval split: ${line}"; exit 1 ;;
  esac

  # max_ndead <= 0 has no budget to divide by - must not claim an ETA or a
  # percent, but should still show a rate.
  line="$(_ns_status_line "${tmp}" -1 "$(($(date +%s) - 90))")"
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

  rm -rf "${tmp}"
  echo "progress-bar self-check passed"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  self_check
fi
