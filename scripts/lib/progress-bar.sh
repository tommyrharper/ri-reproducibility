#!/usr/bin/env bash
# A single-line progress bar for a nested-sampling search: elapsed time, dead
# points done against the run's --max-ndead cap, and an ETA extrapolated from
# the rate so far. Dead points are counted via evaluations/eval-* instead of
# parsing PolyChord's own feedback, which is the same signal `./ri runs`
# already uses to report progress on a stopped run.

# Usage: run_with_progress <evaluations_dir> <max_ndead> -- cmd args...
run_with_progress() {
  local evaluations_dir="$1" max_ndead="$2"
  shift 2
  [ "${1:-}" = "--" ] && shift

  "$@" &
  local pid=$!
  local start
  start="$(date +%s)"

  # `\r`-redraws corrupt piped/logged output, so only draw one on a real TTY.
  if [ -t 1 ]; then
    while kill -0 "${pid}" 2>/dev/null; do
      _ns_print_progress "${evaluations_dir}" "${max_ndead}" "${start}"
      sleep 1
    done
    _ns_print_progress "${evaluations_dir}" "${max_ndead}" "${start}"
    echo
  fi

  wait "${pid}"
}

_ns_count_evaluations() {
  local dir="$1" n=0 f
  for f in "${dir}"/eval-*; do
    [ -e "${f}" ] || continue
    n=$((n + 1))
  done
  echo "${n}"
}

_ns_format_hms() {
  local s="$1"
  printf '%d:%02d:%02d' $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

_ns_print_progress() {
  local evaluations_dir="$1" max_ndead="$2" start="$3"
  local now elapsed done_count

  now="$(date +%s)"
  elapsed=$((now - start))
  done_count="$(_ns_count_evaluations "${evaluations_dir}")"

  # PolyChord treats max_ndead <= 0 as "no bound, stop on evidence tolerance
  # instead" - there's no budget to measure against, so just show the count.
  if [ "${max_ndead}" -le 0 ]; then
    printf '\r%d dead points  elapsed %s   ' \
      "${done_count}" "$(_ns_format_hms "${elapsed}")"
    return
  fi

  local pct eta bar_width filled
  pct=$((done_count * 100 / max_ndead))
  [ "${pct}" -gt 100 ] && pct=100

  eta="?"
  if [ "${done_count}" -gt 0 ] && [ "$((max_ndead - done_count))" -gt 0 ]; then
    eta="$(_ns_format_hms $(((max_ndead - done_count) * elapsed / done_count)))"
  elif [ "${done_count}" -ge "${max_ndead}" ]; then
    eta="0:00:00"
  fi

  bar_width=30
  filled=$((pct * bar_width / 100))
  local bar
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '#')$(printf '%*s' $((bar_width - filled)) '')"

  printf '\r[%s] %3d%%  %d/%d dead points  elapsed %s  eta %s   ' \
    "${bar}" "${pct}" "${done_count}" "${max_ndead}" "$(_ns_format_hms "${elapsed}")" "${eta}"
}

self_check() {
  local tmp
  tmp="$(mktemp -d)"

  [ "$(_ns_count_evaluations "${tmp}")" = "0" ] || { echo "FAIL: empty dir count"; exit 1; }

  mkdir "${tmp}/eval-0001-a" "${tmp}/eval-0002-b"
  [ "$(_ns_count_evaluations "${tmp}")" = "2" ] || { echo "FAIL: eval count"; exit 1; }

  [ "$(_ns_format_hms 3661)" = "1:01:01" ] || { echo "FAIL: format_hms"; exit 1; }
  [ "$(_ns_format_hms 59)" = "0:00:59" ] || { echo "FAIL: format_hms short"; exit 1; }

  # max_ndead <= 0 (PolyChord's "run until evidence tolerance" setting, e.g.
  # --max-ndead -1) has no budget to divide by - must not claim an ETA or
  # a percent, which the naive done_count >= max_ndead check used to do.
  local line
  line="$(_ns_print_progress "${tmp}" -1 "$(($(date +%s) - 90))")"
  case "${line}" in
    *eta*) echo "FAIL: unbounded max_ndead printed an eta: ${line}"; exit 1 ;;
  esac
  case "${line}" in
    *%*) echo "FAIL: unbounded max_ndead printed a percent: ${line}"; exit 1 ;;
  esac

  rm -rf "${tmp}"
  echo "progress-bar self-check passed"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  self_check
fi
