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

A run that is still going has no summary.json either, so on disk it is
indistinguishable from one that stopped - and this listing used to call the
live search `resumable` and print `./ri resume` for it, which would have
started a second MPI job over the same checkpoint. Liveness is not on disk, so
it is read from the process table, the same way `./ri health` decides which run
to report on.

Usage:

  uv run scripts/nested-sampling-runs.py
  uv run scripts/nested-sampling-runs.py --incomplete
  uv run scripts/nested-sampling-runs.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NESTED_SAMPLING_DIR = Path("results/nested-sampling")

# A run's output directory, off any command line still driving it: the MPI
# ranks, and the `mpirun` and `docker exec` wrapping them during startup before
# a rank exists. Deliberately broader than nested-sampling-health.py's
# RANK_COMMAND, which anchors at the interpreter because it has to *count*
# ranks; here the only question is whether anything at all is still running.
RUN_COMMAND = re.compile(r"polychord_\w+\.py\s.*?--output-dir\s+(\S+)")


def running_run_dirs(ps_output: str | None = None) -> set[str]:
    """Resolved output directories that still have a process driving them.

    Sidecar workers name their run by --fifo-dir rather than --output-dir, so a
    killed run's workers - which outlive it by ~3.4GB each until the next run
    reaps them - correctly do not make it look alive.
    """
    if ps_output is None:
        try:
            ps_output = subprocess.run(
                ["ps", "-eo", "args="], capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return set()
    found = set()
    for line in ps_output.splitlines():
        match = RUN_COMMAND.search(line)
        if match:
            found.add(match.group(1).rstrip("/"))
    return found


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


def describe(run_dir: Path, running: set[str]) -> dict[str, object]:
    run_env = read_run_env(run_dir)
    algorithm = run_env.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0]
    evaluations = len(list((run_dir / "evaluations").glob("eval-*")))
    complete = (run_dir / "summary.json").exists()
    # PolyChord's checkpoint. A completed run keeps its resume file too, so
    # this only distinguishes "can be continued" among the incomplete ones.
    resumable = any((run_dir / "chains").glob("*.resume"))
    if complete:
        status = "complete"
    elif str(run_dir.resolve()) in running:
        # Checked after summary.json so a run caught in the seconds between
        # PolyChord returning and its ranks exiting reads as complete.
        status = "running"
    else:
        status = "resumable" if resumable else "incomplete"
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "algorithm": algorithm,
        "status": status,
        "evaluations": evaluations,
        "settings": run_env,
    }


def find_runs(running: set[str] | None = None) -> list[dict[str, object]]:
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    runs = [d for d in NESTED_SAMPLING_DIR.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.name, reverse=True)
    if running is None:
        running = running_run_dirs()
    return [describe(d, running) for d in runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incomplete", action="store_true",
                        help="only the runs that stopped before finishing")
    parser.add_argument("--json", action="store_true", help="raw JSON instead of a table")
    args = parser.parse_args(argv)

    runs = find_runs()
    live = [r for r in runs if r["status"] == "running"]
    unfinished = [r for r in runs if r["status"] not in ("complete", "running")]
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

    if live:
        print()
        print(f"{len(live)} run{'' if len(live) == 1 else 's'} still going. Check on it with:")
        for run in live:
            print(f"  ./ri health {run['name']}")
    if unfinished:
        count = len(unfinished)
        print()
        print(f"{count} run{'' if count == 1 else 's'} stopped before finishing.")
        print("Continue where it left off, keeping every evaluation already done:")
        for run in unfinished:
            print(f"  ./ri resume {run['name']}")
    return 0


def self_check() -> None:
    import contextlib
    import io
    import tempfile

    global NESTED_SAMPLING_DIR, running_run_dirs
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

            runs = {r["name"]: r for r in find_runs(running=set())}
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
            assert {r["name"]: r for r in find_runs(running=set())}[bare.name]["status"] == "incomplete"

            # A run that is still going looks exactly like one that stopped,
            # right down to the checkpoint - so calling it `resumable` and
            # printing `./ri resume` for it was an instruction to start a
            # second MPI job over the live one's own checkpoint.
            live = {str(stopped.resolve())}
            by_name = {r["name"]: r for r in find_runs(running=live)}
            assert by_name[stopped.name]["status"] == "running", by_name[stopped.name]
            assert by_name[done.name]["status"] == "complete", by_name[done.name]
            assert by_name[bare.name]["status"] == "incomplete", by_name[bare.name]

            # The ranks of a run PolyChord has already returned from are still
            # exiting; it finished, and offering to resume it would be wrong.
            assert {r["name"]: r for r in find_runs(running={str(done.resolve())})
                    }[done.name]["status"] == "complete"

            # What `ps -eo args=` actually holds during a run: the rank, the
            # `mpirun` and the host-side `docker exec` all carry --output-dir,
            # while the imager sidecars name the same run by --fifo-dir only.
            rank = f"python3 /opt/ri-nested-sampling/polychord_r2d2.py --output-dir {stopped.resolve()} --nlive 50"
            ps_output = "\n".join([
                rank,
                f"mpirun --allow-run-as-root -np 16 {rank}",
                f"/usr/bin/docker exec -e NS_MPI_PROCS=16 c mpirun -np 16 {rank}",
                f"python3 /repo/scripts/lib/nested_sampling/r2d2_serve.py --fifo-dir {done.resolve()}/.r2d2-workers",
                "python3 /repo/scripts/nested-sampling-health.py",
            ])
            assert running_run_dirs(ps_output) == {str(stopped.resolve())}, running_run_dirs(ps_output)

            # A trailing slash on --output-dir is the same directory, and a run
            # whose name is a prefix of another is not that other run.
            neighbour = f"{stopped.resolve()}-2"
            assert running_run_dirs(
                f"python3 polychord_r2d2.py --output-dir {stopped.resolve()}/ --nlive 5\n"
                f"python3 polychord_r2d2.py --output-dir {neighbour} --nlive 5"
            ) == {str(stopped.resolve()), neighbour}

            # The whole point, at the level the operator reads: the live run is
            # offered to `./ri health`, never to `./ri resume`, and it is not
            # counted among the runs that stopped.
            saved_running = running_run_dirs
            running_run_dirs = lambda ps_output=None: live
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    assert main([]) == 0
            finally:
                running_run_dirs = saved_running
            text = out.getvalue()
            assert "1 run still going" in text, text
            assert f"./ri health {stopped.name}" in text, text
            assert f"./ri resume {stopped.name}" not in text, text
            assert "1 run stopped before finishing." in text, text
            assert f"./ri resume {bare.name}" in text, text
    finally:
        NESTED_SAMPLING_DIR = saved
    print("nested-sampling-runs self-check passed")


if __name__ == "__main__":
    if os.environ.get("NESTED_SAMPLING_RUNS_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
