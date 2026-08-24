#!/usr/bin/env python3
"""Print a per-stage timing breakdown for a nested-sampling PoC run.

Post-processing only: reads the `profiling` block that `polychord_wsclean_poc.py`
/ `polychord_r2d2_poc.py` write into `poc-summary.json` (summed from per-
evaluation `timing.*` fields recorded around each pipeline stage) and renders
it as a human-readable table, so you can see which stage actually dominates
wall time without guessing.

Usage:

  uv run scripts/profile-nested-sampling-run.py results/nested-sampling-poc/wsclean-vlaa-<UTC>
  uv run scripts/profile-nested-sampling-run.py results/nested-sampling-poc/wsclean-vlaa-<UTC>/poc-summary.json
  uv run scripts/profile-nested-sampling-run.py <run-dir> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_summary(target: Path) -> dict[str, Any]:
    summary_path = target / "poc-summary.json" if target.is_dir() else target
    if not summary_path.is_file():
        raise SystemExit(f"no poc-summary.json found at {summary_path}")
    return json.loads(summary_path.read_text())


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:8.2f}s"


def format_pct(value: float | None, total: float) -> str:
    if value is None or total <= 0:
        return ""
    return f"({100.0 * value / total:5.1f}%)"


def print_report(summary: dict[str, Any]) -> None:
    profiling = summary.get("profiling")
    if not profiling:
        raise SystemExit(
            "poc-summary.json has no `profiling` block - it was written by a run "
            "predating profiler instrumentation; re-run the PoC to get one."
        )

    total = profiling["total_wall_seconds"]
    stages = profiling["stage_totals_seconds"]
    counts = profiling["stage_eval_counts"]
    n_evals = len(summary.get("evaluations", []))

    print(f"algorithm:        {summary.get('algorithm')}")
    print(f"evaluations:      {n_evals}")
    print(f"mpi_procs:        {profiling['mpi_procs']}")
    print(f"total_wall:       {format_seconds(total)}")
    print()
    print(f"{'stage':<28} {'total':>10} {'share':>8}  evals")
    print("-" * 58)

    def row(label: str, key: str, count_key: str) -> None:
        value = stages.get(key)
        print(f"{label:<28} {format_seconds(value):>10} {format_pct(value, total):>8}  {counts.get(count_key, 0)}")

    row("simulate (MeqTrees)", "simulate", "simulate_seconds")
    if stages.get("convert") is not None:
        row("convert (MS -> .mat)", "convert", "convert_seconds")
    row("image container (total)", "image_container", "image_container_seconds")
    if stages.get("image_binary") is not None:
        row("  of which: binary run", "image_binary", "image_binary_seconds")
        row("  of which: container overhead", "image_container_overhead", "image_container_overhead_seconds")
    row("metrics computation", "metrics", "metrics_seconds")
    print("-" * 58)
    accounted = profiling["accounted_seconds"]
    print(f"{'accounted (sum above)':<28} {format_seconds(accounted):>10} {format_pct(accounted, total):>8}")

    overhead = profiling["polychord_overhead_seconds"]
    print(f"{'PolyChord overhead (unaccounted)':<28} {format_seconds(overhead):>10} {format_pct(overhead, total):>8}")
    print()
    print(f"note: {profiling['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Run directory or poc-summary.json path")
    parser.add_argument("--json", action="store_true", help="Print the raw profiling dict as JSON instead of a table")
    args = parser.parse_args()

    summary = load_summary(Path(args.run).resolve())
    if args.json:
        json.dump(summary.get("profiling"), sys.stdout, indent=2)
        print()
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
