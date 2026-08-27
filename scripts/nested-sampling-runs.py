#!/usr/bin/env python3
"""List nested-sampling runs and say which ones stopped before they finished.

`summary.json` is written only after PolyChord returns, so a run directory
without one is a run that stopped - out of memory, Ctrl-C, a reboot. That is
the same definition merge-nested-sampling-runs.py already refuses on, and it
needs no extra bookkeeping to stay true.

It also needs surfacing, which is the point of this script. The HTML report
globs `results/nested-sampling/*/summary.json`, so an interrupted run does not
appear there at all - it is not shown as failed, it is simply absent, which is
the easiest kind of problem to not notice. Here it is listed, counted, and
paired with the command that continues it.

Usage:

  uv run scripts/nested-sampling-runs.py
  uv run scripts/nested-sampling-runs.py --incomplete
  uv run scripts/nested-sampling-runs.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NESTED_SAMPLING_DIR = Path("results/nested-sampling")


def read_run_env(run_dir: Path) -> dict[str, str]:
    """The settings a run recorded for itself, or {} for one started before that.

    Deliberately forgiving: this is a status listing, and a run directory that
    is half-written or hand-edited should still be listed rather than crash
    the only command that would have told you it was there.
    """
    path = run_dir / "run.env"
    values: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return values
    for line in text.splitlines():
        name, _, raw = line.partition("=")
        if not _:
            continue
        # `%q` quoting, which for these values is either bare or single-quoted.
        values[name.strip()] = raw.strip().strip("'").replace("'\\''", "'")
    return values


def describe(run_dir: Path) -> dict[str, object]:
    run_env = read_run_env(run_dir)
    algorithm = run_env.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0]
    evaluations = len(list((run_dir / "evaluations").glob("eval-*")))
    complete = (run_dir / "summary.json").exists()
    # PolyChord's checkpoint. A completed run keeps its resume file too, so
    # this only distinguishes "can be continued" among the incomplete ones.
    resumable = any((run_dir / "chains").glob("*.resume"))
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "algorithm": algorithm,
        "status": "complete" if complete else ("resumable" if resumable else "incomplete"),
        "evaluations": evaluations,
        "settings": run_env,
    }


def find_runs() -> list[dict[str, object]]:
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    runs = [d for d in NESTED_SAMPLING_DIR.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.name, reverse=True)
    return [describe(d) for d in runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incomplete", action="store_true",
                        help="only the runs that stopped before finishing")
    parser.add_argument("--json", action="store_true", help="raw JSON instead of a table")
    args = parser.parse_args(argv)

    runs = find_runs()
    unfinished = [r for r in runs if r["status"] != "complete"]
    if args.incomplete:
        runs = unfinished

    if args.json:
        print(json.dumps(runs, indent=2))
        return 0

    if not runs:
        where = "unfinished runs" if args.incomplete else "runs"
        print(f"No {where} under {NESTED_SAMPLING_DIR}/.")
        return 0

    width = max(len(str(r["name"])) for r in runs)
    print(f"{'RUN'.ljust(width)}  {'ALGORITHM':<9}  {'STATUS':<10}  EVALS")
    for run in runs:
        print(
            f"{str(run['name']).ljust(width)}  {str(run['algorithm']):<9}  "
            f"{str(run['status']):<10}  {run['evaluations']}"
        )

    if unfinished:
        count = len(unfinished)
        print()
        print(f"{count} run{'' if count == 1 else 's'} stopped before finishing.")
        print("Continue where it left off, keeping every evaluation already done:")
        for run in unfinished:
            print(f"  ./ri resume {run['name']}")
    return 0


def self_check() -> None:
    import tempfile

    global NESTED_SAMPLING_DIR
    saved = NESTED_SAMPLING_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            NESTED_SAMPLING_DIR = Path(tmp)

            done = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T000000Z"
            (done / "evaluations" / "eval-0001-a").mkdir(parents=True)
            (done / "summary.json").write_text("{}")

            stopped = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260102T000000Z"
            (stopped / "evaluations" / "eval-0001-b").mkdir(parents=True)
            (stopped / "evaluations" / "eval-0002-c").mkdir(parents=True)
            (stopped / "chains").mkdir()
            (stopped / "chains" / "wsclean_vlaa.resume").write_text("")
            (stopped / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_MPI_PROCS=7\nNS_METRIC='total_rms_jy - snr'\n"
            )

            runs = {r["name"]: r for r in find_runs()}
            assert runs[done.name]["status"] == "complete", runs[done.name]
            # A finished run keeps its resume file, so completeness must be
            # decided by summary.json and not by the checkpoint.
            assert runs[stopped.name]["status"] == "resumable", runs[stopped.name]
            assert runs[stopped.name]["evaluations"] == 2
            assert runs[stopped.name]["algorithm"] == "wsclean"
            # Quoted settings survive the round trip out of run.env.
            assert runs[stopped.name]["settings"]["NS_METRIC"] == "total_rms_jy - snr"
            assert runs[stopped.name]["settings"]["NS_MPI_PROCS"] == "7"

            # An interrupted run with no checkpoint is still reported, because
            # being told it is there matters more than being able to continue it.
            bare = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260103T000000Z"
            bare.mkdir()
            assert {r["name"]: r for r in find_runs()}[bare.name]["status"] == "incomplete"
    finally:
        NESTED_SAMPLING_DIR = saved
    print("nested-sampling-runs self-check passed")


if __name__ == "__main__":
    if os.environ.get("NESTED_SAMPLING_RUNS_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
