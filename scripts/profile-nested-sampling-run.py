#!/usr/bin/env python3
"""Print a per-stage timing breakdown for a nested-sampling run.

Post-processing only: reads the `profiling` block that `polychord_wsclean.py`
/ `polychord_r2d2.py` write into `summary.json` (summed from per-evaluation
`timing.*` fields recorded around each pipeline stage) and renders it as a
human-readable table, so you can see which stage actually dominates wall time
without guessing.

Every share is a fraction of the run's worker-time budget (wall clock x
workers, which is one less than the rank count - PolyChord's rank 0
administrates and never evaluates a likelihood), so the top-level rows add up
to 100% of what the whole process spent: the stages of an evaluation, then the
time no evaluation was running in - PolyChord itself, and workers waiting on
other workers. The same breakdown - and the same numbers - back the Profiling
section of the HTML run report.

Usage:

  uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC>
  uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC>/summary.json
  uv run scripts/profile-nested-sampling-run.py <run-dir> --json
  uv run scripts/profile-nested-sampling-run.py <run-dir> --phases
  uv run scripts/profile-nested-sampling-run.py <run-dir> --over-time
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib" / "nested_sampling"))

from common import (  # noqa: E402
    format_duration,
    format_share,
    profiling_breakdown,
)

LABEL_WIDTH = 42
COLUMN_WIDTH = 12
NESTED_SAMPLING_DIR = Path(__file__).resolve().parents[1] / "results" / "nested-sampling"


def resolve_run(raw: str) -> Path:
    """A path, or the bare run name `./ri runs` prints in its first column.

    The name is what a reader has in front of them, and `./ri health` and
    `./ri resume` both take one, so a path-only profiler was the odd door out.
    Only when nothing of that name exists as a path: a real `./wsclean-.../`
    in the working directory still wins, as it does for every other command.
    """
    target = Path(raw).expanduser()
    return NESTED_SAMPLING_DIR / raw if not target.exists() \
        and (NESTED_SAMPLING_DIR / raw).is_dir() else target.resolve()


def load_summary(target: Path) -> dict[str, Any]:
    summary_path = target / "summary.json" if target.is_dir() else target
    if not summary_path.is_file():
        if target.is_dir():
            # The common case now that `./ri tui` loops through this view:
            # a run that is still going has nothing to profile yet, and saying
            # which command does watch a live run is more use than a path.
            raise SystemExit(
                f"{target.name} has not written a summary.json yet - a run writes "
                f"one when it finishes.\n./ri health {target.name} is the view of a "
                "run still going."
            )
        raise SystemExit(f"no summary.json found at {summary_path}")
    try:
        return json.loads(summary_path.read_text())
    except ValueError:
        # Half a summary.json, from a rank killed writing it. Said plainly,
        # with the command that rewrites it, rather than as a JSONDecodeError.
        raise SystemExit(
            f"{summary_path} is only half written - the run was killed while "
            f"writing it.\n./ri resume {summary_path.parent.name} rewrites it "
            "from the evaluations already on disk."
        ) from None


def print_report(summary: dict[str, Any]) -> None:
    profiling = summary.get("profiling")
    if not profiling:
        raise SystemExit(
            "summary.json has no `profiling` block - it was written by a run "
            "predating profiler instrumentation; re-run it to get one."
        )

    breakdown = profiling_breakdown(profiling, summary.get("algorithm"))
    mpi_procs = breakdown["mpi_procs"]
    workers = breakdown["worker_procs"]
    budget = breakdown["worker_seconds_budget"]

    print(f"algorithm:        {summary.get('algorithm')}")
    print(f"evaluations:      {len(summary.get('evaluations', []))}")
    ranks = f"{mpi_procs}" if workers == mpi_procs else f"{mpi_procs} ({workers} workers + administrator)"
    print(f"mpi_procs:        {ranks}")
    print(f"wall clock:       {format_duration(breakdown['total_wall_seconds'])}")
    print(f"worker-time:      {format_duration(budget)}  ({workers} x wall clock)")
    print()

    header = (
        f"{'stage':<{LABEL_WIDTH}}"
        f"{'total':>{COLUMN_WIDTH}}"
        f"{'per eval':>{COLUMN_WIDTH}}"
        f"{'share':>{COLUMN_WIDTH}}"
        f"{'evals':>8}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)

    def line(label: str, seconds: float | None, share: float | None, per_eval: float | None = None, evals: str = "") -> None:
        print(
            (
                f"{label:<{LABEL_WIDTH}}"
                f"{format_duration(seconds):>{COLUMN_WIDTH}}"
                f"{(format_duration(per_eval) if per_eval is not None else ''):>{COLUMN_WIDTH}}"
                f"{format_share(share):>{COLUMN_WIDTH}}"
                f"{evals:>8}"
            ).rstrip()
        )

    for row in breakdown["rows"]:
        label = f"  {row['label']}" if row["is_sub"] else row["label"]
        line(label, row["seconds"], row["share"], row["per_eval_seconds"], str(row["evals"] or ""))

    print(rule)
    line(
        breakdown["subtotal_label"],
        breakdown["subtotal_seconds"],
        breakdown["subtotal_share"],
        breakdown["subtotal_per_eval_seconds"],
    )
    # Below the sum: the time no evaluation was running in, which is the half
    # of the run the stage rows above can say nothing about.
    for row in breakdown["remainder_rows"]:
        line(row["label"], row["seconds"], row["share"])
    print()
    # The same arithmetic the HTML report prints under its chart: worker-seconds
    # only reach the run's wall clock once they are spread across the workers.
    terms = list(breakdown["equation_terms"])
    if mpi_procs != 1:
        terms.append(f"= {format_duration(budget)} of worker-time")
        terms.append(f"/ {workers} workers")
    wall = (budget / workers) if budget else 0.0
    terms.append(f"= {format_duration(wall)} end-to-end wall clock")
    print(" ".join(terms))
    print()
    print(f"note: {breakdown['note']}")




# --- inside one wsclean process ------------------------------------------
#
# `wsclean -log-time` stamps every output line, so a run's own
# wsclean.stdout.log files are a phase timeline at production concurrency with
# no rig and no instrumentation (polychord_wsclean.py passes the flag; it costs
# nothing measurable). A line's stamp is written when the line *starts*, so the
# work a line announces sits in the gap between it and the next line - and the
# gaps are bucketed by (line, next line) rather than by line alone, because
# "Loading data in memory..." appears once per gridding pass and means something
# different each time. See docs/nested-sampling-phase-profile.md.
_MONTHS = {m: i + 1 for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
_STAMP = re.compile(
    r"^(\d{4})-(\w{3})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{6}) ?(.*)$")


def _phase_label(text: str) -> str:
    """Collapse the parts of a log line that differ between evaluations."""
    text = re.sub(r"/\S+", "<path>", text)
    return re.sub(r"[-+]?\d[\d.eE+-]*", "N", text)[:64]


def phase_gaps(log_paths: Any) -> tuple[int, float, dict[tuple[str, str], list[float]]]:
    """(logs read, mean logged milliseconds, gap milliseconds by phase)."""
    gaps: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    logged: list[float] = []
    for path in log_paths:
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            stamp = _STAMP.match(line)
            if stamp:
                *parts, text = stamp.groups()
                year, month, day, hour, minute, second, micro = parts
                rows.append((datetime.datetime(
                    int(year), _MONTHS[month], int(day), int(hour), int(minute),
                    int(second), int(micro)).timestamp(), text))
        if len(rows) < 2:
            continue
        logged.append((rows[-1][0] - rows[0][0]) * 1000.0)
        for (start, text), (end, following) in zip(rows, rows[1:]):
            gaps[(_phase_label(text), _phase_label(following))].append((end - start) * 1000.0)
    return len(logged), (statistics.mean(logged) if logged else 0.0), gaps


def print_phases(run_dir: Path, top: int) -> None:
    logs = sorted(run_dir.glob("evaluations/*/wsclean.stdout.log"))
    if not logs:
        raise SystemExit(f"no evaluations/*/wsclean.stdout.log under {run_dir}")
    count, logged_ms, gaps = phase_gaps(logs)
    if not count:
        raise SystemExit(
            f"{len(logs)} wsclean logs under {run_dir} carry no timestamps - the "
            "run predates `-log-time` being passed by default, so there is no "
            "timeline in them to read."
        )
    print(f"{count} wsclean logs, {logged_ms:.1f} ms of logged work per evaluation")
    print()
    print(f"{'ms/eval':>8} {'share':>7} {'n/eval':>7} {'ms each':>8}  phase (log line -> next log line)")
    print("-" * 110)
    for (before, after), values in sorted(gaps.items(), key=lambda kv: -sum(kv[1]))[:top]:
        total = sum(values) / count
        print(f"{total:8.2f} {total / logged_ms:6.1%} {len(values) / count:7.2f} "
              f"{statistics.mean(values):8.3f}  {before[:52]} -> {after[:40]}")


_VIS_COUNT = re.compile(r"^Gridded visibility count: (\d+)")


def evaluation_timeline(log_paths: Any) -> list[tuple[float, float, int]]:
    """(start epoch, logged milliseconds, gridded visibilities) per evaluation."""
    rows = []
    for path in log_paths:
        stamps: list[float] = []
        visibilities = 0
        for line in path.read_text(errors="replace").splitlines():
            stamp = _STAMP.match(line)
            if not stamp:
                continue
            *parts, text = stamp.groups()
            year, month, day, hour, minute, second, micro = parts
            stamps.append(datetime.datetime(
                int(year), _MONTHS[month], int(day), int(hour), int(minute),
                int(second), int(micro)).timestamp())
            found = _VIS_COUNT.match(text)
            if found:
                visibilities = int(found.group(1))
        if len(stamps) >= 2:
            rows.append((stamps[0], (stamps[-1] - stamps[0]) * 1000.0, visibilities))
    rows.sort()
    return rows


def print_over_time(run_dir: Path, buckets: int) -> None:
    """Throughput against wall clock, so a run's own slowdown can be attributed.

    An evaluation costs roughly a constant plus a rate times its visibility
    count, and nested sampling walks into the corner of the parameter space
    with the most visibilities - so a run's evaluations/second falls as it
    goes. Printing the visibility count beside the rate is what separates that
    from the machine (see docs/nested-sampling-cost-model.md).
    """
    logs = sorted(run_dir.glob("evaluations/*/wsclean.stdout.log"))
    if not logs:
        raise SystemExit(f"no evaluations/*/wsclean.stdout.log under {run_dir}")
    rows = evaluation_timeline(logs)
    if not rows:
        raise SystemExit(
            f"{len(logs)} wsclean logs under {run_dir} carry no timestamps - the "
            "run predates `-log-time` being passed by default, so there is no "
            "timeline in them to read."
        )
    start = rows[0][0]
    span = rows[-1][0] - start
    print(f"{len(rows)} evaluations over {span:.0f}s of wall clock, "
          f"in {buckets} equal-count buckets")
    print()
    print(f"{'t (s)':>7} {'evals/s':>9} {'ms/eval':>9} {'vis/eval':>9}")
    print("-" * 38)
    # Boundaries rather than a fixed stride, so the remainder joins the last
    # bucket: on its own it is a handful of evaluations draining after the
    # other ranks stopped, and it reads as a collapse that never happened.
    for index in range(buckets):
        chunk = rows[index * len(rows) // buckets:(index + 1) * len(rows) // buckets]
        if len(chunk) < 2:
            continue
        elapsed = chunk[-1][0] - chunk[0][0]
        rate = len(chunk) / elapsed if elapsed else float("nan")
        print(f"{chunk[0][0] - start:7.0f} {rate:9.1f} "
              f"{statistics.median(r[1] for r in chunk):9.1f} "
              f"{statistics.median(r[2] for r in chunk):9.0f}")


def self_check() -> None:
    """A two-line timeline is enough to pin the gap arithmetic and the labels."""
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        log = Path(raw) / "wsclean.stdout.log"
        log.write_text(
            "2026-Aug-29 10:19:02.100000 Gridding 1404 rows...\n"
            "2026-Aug-29 10:19:02.105500 Gridded visibility count: 4140\n"
            "not a timestamped line\n"
            "2026-Aug-29 10:19:02.107500 Opening reordered part 0 for /a/b/sim.ms\n"
        )
        count, logged, gaps = phase_gaps([log])
    assert count == 1, count
    assert abs(logged - 7.5) < 1e-3, logged
    assert abs(gaps[("Gridding N rows...", "Gridded visibility count: N")][0] - 5.5) < 1e-3, gaps
    assert ("Gridded visibility count: N", "Opening reordered part N for <path>") in gaps, gaps

    with tempfile.TemporaryDirectory() as raw:
        log = Path(raw) / "wsclean.stdout.log"
        log.write_text(
            "2026-Aug-29 10:19:02.100000 Gridding 1404 rows...\n"
            "2026-Aug-29 10:19:02.105500 Gridded visibility count: 4140\n"
            "2026-Aug-29 10:19:02.107500 Gridded visibility count: 4140\n"
        )
        timeline = evaluation_timeline([log])
    assert len(timeline) == 1, timeline
    assert abs(timeline[0][1] - 7.5) < 1e-3, timeline
    assert timeline[0][2] == 4140, timeline

    # A live run has no summary.json, and `./ri tui` now shows what this says.
    with tempfile.TemporaryDirectory() as raw:
        live = Path(raw) / "wsclean-vlaa-20260101T000000Z"
        live.mkdir()
        try:
            load_summary(live)
        except SystemExit as exit_message:
            assert "has not written a summary.json yet" in str(exit_message), exit_message
            assert "./ri health wsclean-vlaa-20260101T000000Z" in str(exit_message), exit_message
        else:
            raise AssertionError("a run directory with no summary.json must not load")
    print("OK: wsclean phase timeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", help="Run directory, run name, or summary.json path")
    parser.add_argument("--json", action="store_true", help="Print the raw profiling dict as JSON instead of a table")
    parser.add_argument("--phases", action="store_true",
                        help="Break the wsclean binary down by phase, from every evaluation's own -log-time timeline")
    parser.add_argument("--over-time", action="store_true",
                        help="Evaluations/second against wall clock, beside the visibility count that sets it")
    parser.add_argument("--top", type=int, default=25, help="Phases to print, largest first")
    parser.add_argument("--buckets", type=int, default=20, help="Buckets for --over-time")
    parser.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.run:
        parser.error("the following arguments are required: run")
    if args.phases or args.over_time:
        run_dir = resolve_run(args.run)
        run_dir = run_dir.parent if run_dir.is_file() else run_dir
        if args.phases:
            print_phases(run_dir, args.top)
        if args.over_time:
            print_over_time(run_dir, args.buckets)
        return

    summary = load_summary(resolve_run(args.run))
    if args.json:
        json.dump(summary.get("profiling"), sys.stdout, indent=2)
        print()
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
