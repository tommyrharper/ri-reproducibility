#!/usr/bin/env python3
"""Give a run that is still going the summary its finished form would have.

Everything downstream of a run - the HTML report, `./ri merge`, the listing -
is built around `summary.json`, and a run in progress has none: that file is
the marker that says "finished", so writing one early would offer a half-run
for merging and hide it from `./ri resume`. This writes the same shape under
`summary.live.json` instead, from the evaluation records and `run.env` the run
has already put on disk, and only the live views look for it.

Written on demand (`./ri report --live`), not continuously: nothing here is a
daemon, and a live summary is only ever as fresh as the last write.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling"
LIVE_SUMMARY_NAME = "summary.live.json"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from common import (  # noqa: E402
    load_evaluations_from_dir,
    summarize_profiling,
    write_json_atomic,
)


def _load_runs_module():
    """`nested-sampling-runs.py` already knows how to read a run directory, and
    its name cannot be imported - dashes - so load it by path."""
    path = REPO_ROOT / "scripts" / "nested-sampling-runs.py"
    loader = importlib.machinery.SourceFileLoader("ns_runs", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


ns_runs = _load_runs_module()


def live_run_dirs(running: set[str] | None = None) -> list[Path]:
    """Run directories with a PolyChord rank alive in them, newest first."""
    if running is None:
        running = ns_runs.running_run_dirs()
    dirs = [Path(path) for path in running]
    return sorted((d for d in dirs if d.is_dir()),
                  key=ns_runs.started_at, reverse=True)


def latest_live_run() -> Path:
    runs = live_run_dirs()
    if not runs:
        raise SystemExit(
            f"No run in progress under {NESTED_SAMPLING_DIR}/. "
            "Start one with ./ri search, or name a finished run instead."
        )
    return runs[0]


def build_live_summary(run_dir: Path, now: float | None = None) -> dict[str, Any]:
    """The summary a report can render, from what the run has written so far.

    Key order matters: readers that only want the run-level fields stop at
    `evaluations`, which is the bulk of the file.
    """
    now = time.time() if now is None else now
    settings = ns_runs.read_run_env(run_dir)
    space = ns_runs.read_parameter_space(run_dir) or ns_runs.default_parameter_space()
    evaluations = load_evaluations_from_dir(run_dir / "evaluations")
    scored = [record for record in evaluations if "objective" in record]
    started = ns_runs.started_at(run_dir)
    # Wall clock since the run id's stamp, not summed evaluation time: it is
    # what the finished summary records, and the only clock a live run has.
    wall_seconds = max(0.0, now - started)
    mpi_procs = int(settings.get("NS_MPI_PROCS") or 1)

    def number(name: str) -> int | None:
        try:
            return int(settings[name])
        except (KeyError, ValueError):
            return None

    return {
        "algorithm": settings.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0],
        "vla_config": "VLA.A",
        "run_type": "nested-sampling run, still going",
        # The one field that separates this from a finished run's summary.
        "live": True,
        "live_generated_at": now,
        "metric": settings.get("NS_METRIC", ""),
        "polychord": {
            "nlive": number("NS_NLIVE"),
            "num_repeats": number("NS_NUM_REPEATS"),
            "max_ndead": number("NS_MAX_NDEAD"),
            "seed": number("NS_SEED"),
            "mpi_procs": mpi_procs,
            "synchronous": settings.get("NS_SYNCHRONOUS") == "1",
        },
        "parameter_space": space,
        "total_wall_seconds": wall_seconds,
        "profiling": summarize_profiling(scored, wall_seconds, mpi_procs, started),
        "worst_evaluation": (max(scored, key=lambda item: item["objective"])
                             if scored else None),
        "evaluations": evaluations,
    }


def write_live_summary(run_dir: Path, now: float | None = None) -> Path:
    path = run_dir / LIVE_SUMMARY_NAME
    # Atomic, because the report reads this while the next write replaces it.
    write_json_atomic(path, build_live_summary(run_dir, now))
    return path


def drop_stale_live_summaries(live: set[Path]) -> list[Path]:
    """Remove live summaries for runs that have since written a real one.

    A run that stopped without finishing keeps its last live summary: it is the
    only summary that run will ever have, and the report page built from it is
    the only place its evaluations are readable.
    """
    dropped = []
    if not NESTED_SAMPLING_DIR.is_dir():
        return dropped
    for path in sorted(NESTED_SAMPLING_DIR.glob(f"*/{LIVE_SUMMARY_NAME}")):
        if path.parent in live or not ns_runs.summary_is_complete(path.parent):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        dropped.append(path)
    return dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", metavar="RUN",
                        help="run directories (default: every run in progress)")
    args = parser.parse_args(argv)

    runs = [Path(run) for run in args.runs] if args.runs else live_run_dirs()
    if not runs:
        print(f"No run in progress under {NESTED_SAMPLING_DIR}/.")
    for run_dir in runs:
        print(f"wrote {write_live_summary(run_dir)}")
    for path in drop_stale_live_summaries(set(runs)):
        print(f"removed {path} (the run finished and wrote its own)")
    return 0


def self_check() -> None:
    import json
    import tempfile

    global NESTED_SAMPLING_DIR
    saved = NESTED_SAMPLING_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            NESTED_SAMPLING_DIR = Path(tmp)
            run_dir = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260101T000000Z"
            (run_dir / "evaluations" / "eval-0001-a").mkdir(parents=True)
            (run_dir / "evaluations" / "eval-0001-a" / "metrics.json").write_text(
                json.dumps({"eval_id": 1, "objective": 2.0, "params": {"channel_count": 4}})
            )
            # An evaluation still in flight: no record, and no reason to fail.
            (run_dir / "evaluations" / "eval-0002-b").mkdir()
            (run_dir / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_NLIVE=8\nNS_MPI_PROCS=4\n"
                "NS_METRIC='total_rms_jy - snr'\nNS_SYNCHRONOUS=1\n"
            )
            (run_dir / "parameter-space.json").write_text(
                '[{"name": "channel_count", "min": 1, "max": 8, "kind": "integer"}]'
            )

            path = write_live_summary(run_dir, now=ns_runs.started_at(run_dir) + 60.0)
            assert path.name == LIVE_SUMMARY_NAME, path
            summary = json.loads(path.read_text())
            assert summary["live"] is True, summary
            assert summary["algorithm"] == "wsclean", summary
            assert summary["metric"] == "total_rms_jy - snr", summary
            assert summary["polychord"]["nlive"] == 8, summary
            assert summary["polychord"]["synchronous"] is True, summary
            assert summary["parameter_space"][0]["name"] == "channel_count", summary
            assert [ev["eval_id"] for ev in summary["evaluations"]] == [1], summary
            assert summary["worst_evaluation"]["eval_id"] == 1, summary
            assert abs(summary["total_wall_seconds"] - 60.0) < 1.0, summary
            # The run-level fields sit above the bulk, so a reader that only
            # wants those never parses the evaluations array.
            text = path.read_text()
            assert text.index('"metric"') < text.index('"evaluations"'), text[:200]

            # A live summary is never named summary.json: that file is what
            # every other reader treats as "this run finished".
            assert not (run_dir / "summary.json").exists()

            # It survives the run finishing, and is cleared once it has.
            assert drop_stale_live_summaries(set()) == [], "no summary.json yet"
            assert path.is_file()
            (run_dir / "summary.json").write_text("{}")
            assert drop_stale_live_summaries({run_dir}) == [], "still live"
            assert drop_stale_live_summaries(set()) == [path], "finished"
            assert not path.exists()

            # A run with nothing on disk yet is still summarisable.
            bare = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260102T000000Z"
            bare.mkdir()
            bare_summary = json.loads(write_live_summary(bare).read_text())
            assert bare_summary["algorithm"] == "r2d2", bare_summary
            assert bare_summary["evaluations"] == [], bare_summary
            assert bare_summary["worst_evaluation"] is None, bare_summary

            # Newest first, and anything ps named that is not a directory is
            # not a run.
            ordered = live_run_dirs({str(bare), str(run_dir), f"{run_dir}-gone"})
            assert ordered == [bare, run_dir], ordered
    finally:
        NESTED_SAMPLING_DIR = saved
    print("live_runs self-check passed")


if __name__ == "__main__":
    if os.environ.get("LIVE_RUNS_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
