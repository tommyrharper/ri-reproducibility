#!/usr/bin/env python3
"""Print a per-stage timing breakdown for a nested-sampling run.

Post-processing only: reads the `profiling` block that `polychord_wsclean.py`
/ `polychord_r2d2.py` write into `summary.json` (summed from per-evaluation
`timing.*` fields recorded around each pipeline stage) and renders it as a
human-readable table, so you can see which stage actually dominates wall time
without guessing.

Every share is a fraction of the run's worker-time budget (wall clock x
mpi_procs), so the top-level stages plus the unaccounted remainder add up to
100% of what the whole process spent. The same breakdown - and the same
numbers - back the Profiling section of the HTML run report.

Usage:

  uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC>
  uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC>/summary.json
  uv run scripts/profile-nested-sampling-run.py <run-dir> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib" / "nested_sampling"))

from common import (  # noqa: E402
    PROFILING_VIEW_NOTE,
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
    budget = breakdown["worker_seconds_budget"]

    print(f"algorithm:        {summary.get('algorithm')}")
    print(f"evaluations:      {len(summary.get('evaluations', []))}")
    print(f"mpi_procs:        {mpi_procs}")
    print(f"wall clock:       {format_duration(breakdown['total_wall_seconds'])}")
    print(f"worker-time:      {format_duration(budget)}  ({mpi_procs} x wall clock)")
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
    evals = breakdown["evals"]
    accounted = breakdown["accounted_seconds"]
    line(
        "accounted (sum of stages above)",
        accounted,
        breakdown["accounted_share"],
        accounted / evals if evals else None,
    )
    line(breakdown["unaccounted_label"], breakdown["unaccounted_seconds"], breakdown["unaccounted_share"])
    print()
    # The same arithmetic the HTML report prints under its chart: worker-seconds
    # only reach the run's wall clock once they are spread across the workers.
    terms = [
        f"{format_duration(accounted)} accounted",
        f"+ {format_duration(breakdown['unaccounted_seconds'])} unaccounted",
    ]
    if mpi_procs != 1:
        terms.append(f"= {format_duration(budget)} of worker-time")
        terms.append(f"/ {mpi_procs} workers")
    wall = (budget / mpi_procs) if budget else 0.0
    terms.append(f"= {format_duration(wall)} end-to-end wall clock")
    print(" ".join(terms))
    print()
    print(f"note: {PROFILING_VIEW_NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Run directory, run name, or summary.json path")
    parser.add_argument("--json", action="store_true", help="Print the raw profiling dict as JSON instead of a table")
    args = parser.parse_args()

    summary = load_summary(resolve_run(args.run))
    if args.json:
        json.dump(summary.get("profiling"), sys.stdout, indent=2)
        print()
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
