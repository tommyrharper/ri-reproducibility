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
import time
from datetime import date, datetime, timedelta, timezone
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


RUN_ARTIFACTS = ("run.env", "run.log", "summary.json", "evaluations", "chains")


def summary_is_complete(run_dir: Path) -> bool:
    """Whether the run has a whole summary.json rather than half of one.

    A summary.json is what every reader here calls "finished", so half of one -
    a rank killed while writing it, or a full disk - used to be the worst of
    both: the run was called finished, while the HTML report, `./ri merge` and
    `./ri profile` could not read it and `./ri resume` refused to rewrite it.
    Not finished, then: `resumable`, which is the status that pairs the run
    with the command that repairs it.

    Tested by the last byte rather than by parsing, because this runs over
    every run in the results directory and a finished R2D2 search's summary
    carries all of its evaluations - tens of MB to parse for a question the
    tail answers in one seek. json.dumps ends every complete write with `}`.
    Runs written since write_json_atomic() cannot produce a torn one.
    """
    try:
        with open(run_dir / "summary.json", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64))
            return f.read().decode("utf-8", "replace").rstrip().endswith("}")
    except OSError:
        return False


RUN_ID_TS_RE = re.compile(r"(\d{8}T\d{6}Z)$")


def started_at(run_dir: Path) -> float:
    """When the run started, in epoch seconds.

    Off the UTC stamp every run directory is named for, falling back to the
    directory's own mtime for one named by hand. Sorting the names themselves
    put every `wsclean-*` run below every `r2d2-*` one, so "newest first" came
    out as the newest WSClean run followed by the newest R2D2 run - which on a
    host running both is not the newest run. nested-sampling-health.py's
    started_at() avoids the same trap for the same reason.
    """
    match = RUN_ID_TS_RE.search(run_dir.name)
    if match:
        try:
            return datetime.strptime(
                match.group(1), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0


def format_started(started: float, now: float | None = None) -> str:
    """`today 17:38 (2h ago)` - when the run started, as the reader thinks of it.

    Local time, because the reader is at a terminal on the host the run is on,
    the same reason `./ri health` prints local finish times. The age answers
    the question actually being asked ("is that the one I started before
    lunch?"); the clock time is what still separates two runs once neither is
    today's.
    """
    now = time.time() if now is None else now
    day = date.fromtimestamp(started)
    today = date.fromtimestamp(now)
    if day == today:
        label = "today"
    # Calendar arithmetic rather than now - 86400, which is off by an hour
    # across a DST change and would then call yesterday the day before.
    elif day == today - timedelta(days=1):
        label = "yesterday"
    else:
        label = time.strftime(
            "%a %d %b" if day.year == today.year else "%d %b %Y",
            time.localtime(started),
        )
    age = max(0.0, now - started)  # a stamp from the future is a skewed clock
    if age < 90:
        ago = "just now"
    elif age < 3600:
        ago = f"{int(age // 60)}m ago"
    elif age < 86400:
        ago = f"{int(age // 3600)}h ago"
    else:
        ago = f"{int(age // 86400)}d ago"
    return f"{label} {time.strftime('%H:%M', time.localtime(started))} ({ago})"


def describe(run_dir: Path, running: set[str]) -> dict[str, object]:
    run_env = read_run_env(run_dir)
    algorithm = run_env.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0]
    # Scored evaluations, not evaluation directories: a directory with no
    # metrics.json holds nothing (the run died between creating it and scoring
    # it) and `adopt_completed_evaluations` in scripts/lib/nested_sampling/
    # common.py deletes it on resume. Counting directories made three runs that
    # died during startup advertise 7, 7 and 15 evaluations under a footer
    # promising to keep every one, when the number that survives is zero - and
    # disagreed with `./ri health`, which has always counted the scored ones.
    # Costs 0.15s against 0.04s on the largest run here (17,760 evaluations).
    evaluations = len(list((run_dir / "evaluations").glob("eval-*/metrics.json")))
    complete = summary_is_complete(run_dir)
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
        "started": datetime.fromtimestamp(
            started_at(run_dir), timezone.utc).isoformat(),
        "algorithm": algorithm,
        "status": status,
        "evaluations": evaluations,
        "settings": run_env,
    }


def find_runs(running: set[str] | None = None) -> list[dict[str, object]]:
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    # A directory with none of these is not a run at all, and listing one as
    # `incomplete` pairs it with a `./ri resume` that refuses it for having no
    # run.env. `run.env` is written milliseconds after the run directory is
    # claimed; the rest are for runs from before it existed and for the
    # summary-only directories `./ri merge` writes. Same rule as
    # `is_run_directory` in scripts/nested-sampling-health.py.
    runs = [d for d in NESTED_SAMPLING_DIR.iterdir() if d.is_dir()
            and any((d / artifact).exists() for artifact in RUN_ARTIFACTS)]
    # Newest first, by when the run started rather than by what it is named.
    runs.sort(key=lambda d: (started_at(d), d.name), reverse=True)
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
    evals = max(len("EVALS"), *(len(str(r["evaluations"])) for r in runs))
    print(f"{'RUN'.ljust(width)}  {'ALGORITHM':<9}  {'STATUS':<10}  "
          f"{'EVALS':>{evals}}  STARTED")
    now = time.time()
    for run in runs:
        print(
            f"{str(run['name']).ljust(width)}  {str(run['algorithm']):<9}  "
            f"{str(run['status']):<10}  {run['evaluations']:>{evals}}  "
            f"{format_started(started_at(Path(str(run['path']))), now)}"
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
        # Split by whether PolyChord left a checkpoint. `./ri resume` is the
        # right command either way, but only a `resumable` run continues from
        # where it stopped; an `incomplete` one has no live points on disk, so
        # the sampler starts over and the single promise this footer used to
        # make for both was false for half of them.
        for status, lead in (
            ("resumable", "Continue where it left off, keeping every evaluation already done:"),
            ("incomplete", "No checkpoint, so the sampler starts over, "
                           "reusing the evaluations already scored:"),
        ):
            group = [r for r in unfinished if r["status"] == status]
            if group:
                print(lead)
                for run in group:
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

            def score(eval_dir: Path) -> None:
                """What a rank writes once it has an answer for that point."""
                eval_dir.mkdir(parents=True)
                (eval_dir / "metrics.json").write_text("{}")

            done = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T000000Z"
            score(done / "evaluations" / "eval-0001-a")
            (done / "summary.json").write_text("{}")

            stopped = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260102T000000Z"
            score(stopped / "evaluations" / "eval-0001-b")
            score(stopped / "evaluations" / "eval-0002-c")
            # In flight when the run stopped: no metrics.json, so it holds
            # nothing and the resumed run deletes it. Counting it would
            # overstate what resuming keeps, which is what the directory count
            # this replaced did.
            (stopped / "evaluations" / "eval-0003-d").mkdir()
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
            assert runs[stopped.name]["evaluations"] == 2, runs[stopped.name]
            assert runs[stopped.name]["algorithm"] == "wsclean"
            # Quoted settings survive the round trip out of run.env.
            assert runs[stopped.name]["settings"]["NS_METRIC"] == "total_rms_jy - snr"
            assert runs[stopped.name]["settings"]["NS_MPI_PROCS"] == "7"

            # An interrupted run with no checkpoint is still reported, because
            # being told it is there matters more than being able to continue
            # it. It is a run because of the run.env written milliseconds
            # after its directory was claimed, which is all a run that died
            # before its first evaluation ever has.
            bare = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260103T000000Z"
            bare.mkdir()
            (bare / "run.env").write_text("NS_ALGORITHM=r2d2\n")
            # The shape three real runs on this host have: every rank created
            # its first evaluation directory and the run died before any of
            # them was scored. It has nothing, and must not advertise 7.
            for rank in range(7):
                (bare / "evaluations" / f"eval-0001-{rank}").mkdir(parents=True)
            bare_run = {r["name"]: r for r in find_runs(running=set())}[bare.name]
            assert bare_run["status"] == "incomplete", bare_run
            assert bare_run["evaluations"] == 0, bare_run

            # Newest first across algorithms, not within them. Sorted by name,
            # this order was wsclean-0102, r2d2-0103, r2d2-0101: the newest run
            # on the host was not the one at the top.
            order = [r["name"] for r in find_runs(running=set())]
            assert order == [bare.name, stopped.name, done.name], order
            assert bare_run["started"] == "2026-01-03T00:00:00+00:00", bare_run
            # A directory not named for a time still sorts and dates by
            # something real, so one hand-named run cannot land it at the
            # bottom for ever.
            hand = NESTED_SAMPLING_DIR / "keep-this-one"
            hand.mkdir()
            (hand / "run.env").write_text("NS_ALGORITHM=r2d2\n")
            os.utime(hand, (0, datetime(2026, 1, 4, tzinfo=timezone.utc).timestamp()))
            assert [r["name"] for r in find_runs(running=set())][0] == hand.name

            # The label, against a fixed clock so it does not depend on today.
            noon = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc).timestamp()
            def label(offset_seconds: float) -> str:
                return format_started(noon - offset_seconds, noon)
            assert label(0).startswith("today "), label(0)
            assert label(0).endswith("(just now)"), label(0)
            assert label(2 * 3600).endswith("(2h ago)"), label(2 * 3600)
            assert label(20 * 60).endswith("(20m ago)"), label(20 * 60)
            # Local midnight, not 24 hours: a run at 23:00 last night is
            # yesterday's at 09:00 whatever the hour count says.
            midnight = datetime.fromtimestamp(noon).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            assert label(noon - midnight + 60).startswith("yesterday "), \
                label(noon - midnight + 60)
            assert label(30 * 86400).endswith("(30d ago)"), label(30 * 86400)
            # A run from a previous year names the year; one from this year
            # spends those characters on the weekday instead.
            assert "2026" not in label(30 * 86400), label(30 * 86400)
            assert "2025" in label(400 * 86400), label(400 * 86400)
            # A stamp from the future is a clock disagreeing with itself, not a
            # negative age.
            assert label(-3600).endswith("(just now)"), label(-3600)

            (hand / "run.env").unlink()
            hand.rmdir()

            # A directory with none of those is not a run, and listing it as
            # `incomplete` paired it with a `./ri resume` that refuses it for
            # having no run.env.
            stray = NESTED_SAMPLING_DIR / "notes-20260103T000001Z"
            stray.mkdir()
            (stray / "scratch.txt").write_text("not a run\n")
            assert stray.name not in {r["name"] for r in find_runs(running=set())}
            # Any one artifact makes it one again - a legacy run predating
            # run.env has evaluations, and `./ri merge` writes summary.json
            # alone. Spelled out rather than looped over RUN_ARTIFACTS, which
            # would make deleting an entry delete its own case.
            for artifact in ("run.env", "run.log", "summary.json",
                             "evaluations", "chains"):
                target = stray / artifact
                target.mkdir() if artifact in ("evaluations", "chains") \
                    else target.write_text("")
                assert stray.name in {r["name"] for r in find_runs(running=set())}, \
                    artifact
                target.rmdir() if target.is_dir() else target.unlink()
            (stray / "scratch.txt").unlink()
            stray.rmdir()

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
            # The only unfinished run here has no checkpoint, so the promise
            # to continue where it left off must not be the one printed.
            assert "starts over" in text, text
            assert "Continue where it left off" not in text, text

            # Nothing running: now one unfinished run has a checkpoint and one
            # does not, and each has to be listed under the sentence that is
            # true of it. `./ri resume` is right for both, which is exactly
            # why one shared line went unquestioned.
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert main([]) == 0
            text = out.getvalue()
            continue_at, over_at = (text.index("Continue where it left off"),
                                    text.index("starts over"))
            assert continue_at < text.index(f"./ri resume {stopped.name}") < over_at, text
            assert over_at < text.index(f"./ri resume {bare.name}"), text
            # And the run that died before scoring anything reports 0, not the
            # 7 directories its ranks left behind - beside the day it ran on,
            # which is the column a run directory's name spells in UTC and
            # nobody reads back as a time.
            assert f"{bare.name:<29}  r2d2       incomplete      0  " in text, text
            assert text.index("STARTED") < text.index(bare.name), text
            for run_dir in (bare, stopped, done):
                assert format_started(started_at(run_dir)) in text, run_dir
            # Top of the table is the newest run, whichever imager it is.
            assert text.index(bare.name) < text.index(stopped.name) < \
                text.index(done.name), text

            # Half a summary.json is not a finished run: it is one nothing can
            # report on, merge or profile, and the unfinished list is where it
            # gets paired with the command that rewrites it. The same run as
            # `done` above, with only the bytes of its summary changed, so
            # nothing but the completeness test can be what moves it.
            whole = (done / "summary.json").read_text()
            (done / "summary.json").write_text('{\n  "evaluations": [\n    {\n      "eval')
            torn_run = {r["name"]: r for r in find_runs(running=set())}[done.name]
            assert torn_run["status"] == "incomplete", torn_run
            (done / "summary.json").write_text(whole)
            assert {r["name"]: r for r in find_runs(running=set())
                    }[done.name]["status"] == "complete"
    finally:
        NESTED_SAMPLING_DIR = saved
    print("nested-sampling-runs self-check passed")


if __name__ == "__main__":
    if os.environ.get("NESTED_SAMPLING_RUNS_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
