#!/usr/bin/env python3
"""List nested-sampling runs and print resume commands."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from functools import cache
from pathlib import Path

NESTED_SAMPLING_DIR = Path("results/nested-sampling")

RUN_COMMAND = re.compile(r"polychord_\w+\.py\s.*?--output-dir\s+(\S+)")


def running_run_dirs(ps_output: str | None = None) -> set[str]:
    if ps_output is None:
        try:
            ps_output = subprocess.run(
                ["ps", "-eo", "args="], capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return set()
    return {match.group(1).rstrip("/") for line in ps_output.splitlines()
            if (match := RUN_COMMAND.search(line))}


def read_run_env(run_dir: Path) -> dict[str, str]:
    try:
        text = (run_dir / "run.env").read_text()
    except OSError:
        return {}
    return {name.strip(): raw.strip().strip("'").replace("'\\''", "'")
            for name, raw in (line.split("=", 1) for line in text.splitlines() if "=" in line)}


def read_parameter_space(run_dir: Path) -> list[dict[str, object]]:
    """The box the run actually searched, written by polychord_*.py at startup."""
    try:
        space = json.loads((run_dir / "parameter-space.json").read_text())
    except (OSError, ValueError):  # absent, or caught mid-write
        return []
    return space if isinstance(space, list) else []


def read_summary(run_dir: Path) -> dict[str, object]:
    """The finished run's own record of what it searched and scored. Runs from
    before run.env and parameter-space.json existed, and merged runs which
    write neither, still carry both here."""
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):  # absent, or caught mid-write
        return {}
    return summary if isinstance(summary, dict) else {}


@cache
def default_parameter_space() -> list[dict[str, object]]:
    """defaults.toml's box, for a run that died before recording its own.

    This is the repository's box now, not a record of the run's, so callers
    have to say so - defaults.toml is edited between runs. It is still the
    best answer available: nothing else on disk names the box, and the run
    was launched from this same file."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "lib" / "nested_sampling"))
        from common import load_parameter_space

        return load_parameter_space()
    except Exception:  # a listing must survive anything defaults.toml does
        return []


RUN_ARTIFACTS = ("run.env", "run.log", "summary.json", "evaluations", "chains")


def summary_is_complete(run_dir: Path) -> bool:
    try:
        with open(run_dir / "summary.json", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64))
            return f.read().decode("utf-8", "replace").rstrip().endswith("}")
    except OSError:
        return False


RUN_ID_TS_RE = re.compile(r"(\d{8}T\d{6}Z)$")


def started_at(run_dir: Path) -> float:
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
    now = time.time() if now is None else now
    day = date.fromtimestamp(started)
    today = date.fromtimestamp(now)
    if day == today:
        label = "today"
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
    space = read_parameter_space(run_dir)
    if not run_env.get("NS_METRIC") or not space:
        # Only read the summary when the cheap sources came up short: a
        # finished run's summary.json can be tens of megabytes.
        summary = read_summary(run_dir)
        if not run_env.get("NS_METRIC") and isinstance(summary.get("metric"), str):
            run_env = {**run_env, "NS_METRIC": summary["metric"]}
        if not space and isinstance(summary.get("parameter_space"), list):
            space = summary["parameter_space"]
    # A run that died warming its sidecars never wrote a box and never reached
    # a summary. defaults.toml is what it was launched from, so show that and
    # flag it rather than leave the column blank.
    from_defaults = not space
    if from_defaults:
        space = default_parameter_space()
    algorithm = run_env.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0]
    evaluations = len(list((run_dir / "evaluations").glob("eval-*/metrics.json")))
    complete = summary_is_complete(run_dir)
    resumable = any((run_dir / "chains").glob("*.resume"))
    if complete:
        status = "complete"
    elif str(run_dir.resolve()) in running:
        status = "running"
    else:
        status = "resumable" if resumable else "incomplete"
    started = started_at(run_dir)
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "started_label": format_started(started),
        "algorithm": algorithm,
        "status": status,
        "evaluations": evaluations,
        "settings": run_env,
        "parameter_space": space,
        # False unless the box above is defaults.toml standing in for a record
        # the run never wrote; the absent-field default is the honest one.
        "parameter_space_from_defaults": from_defaults and bool(space),
    }


def find_runs(running: set[str] | None = None) -> list[dict[str, object]]:
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    # One ps for the whole listing, not one per run: inside the comprehension
    # this cost a process spawn per directory, and dominated the listing.
    if running is None:
        running = running_run_dirs()
    runs = sorted([d for d in NESTED_SAMPLING_DIR.iterdir() if d.is_dir() and any((d / artifact).exists() for artifact in RUN_ARTIFACTS)], key=lambda d: (started_at(d), d.name), reverse=True)
    return [describe(d, running) for d in runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    for run in runs:
        print(
            f"{str(run['name']).ljust(width)}  {str(run['algorithm']):<9}  "
            f"{str(run['status']):<10}  {run['evaluations']:>{evals}}  "
            f"{run['started_label']}"
        )

    if live:
        print(f"\n{len(live)} run{'' if len(live) == 1 else 's'} still going. Check on it with:")
        for run in live:
            print(f"  ./ri health {run['name']}")
    if unfinished:
        count = len(unfinished)
        print(f"\n{count} run{'' if count == 1 else 's'} stopped before finishing.")
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
    import shutil
    import tempfile

    global NESTED_SAMPLING_DIR, running_run_dirs
    saved = NESTED_SAMPLING_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            NESTED_SAMPLING_DIR = Path(tmp)

            def score(eval_dir: Path) -> None:
                eval_dir.mkdir(parents=True)
                (eval_dir / "metrics.json").write_text("{}")

            done = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T000000Z"
            score(done / "evaluations" / "eval-0001-a")
            (done / "summary.json").write_text("{}")

            stopped = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260102T000000Z"
            score(stopped / "evaluations" / "eval-0001-b")
            score(stopped / "evaluations" / "eval-0002-c")
            (stopped / "evaluations" / "eval-0003-d").mkdir()
            (stopped / "chains").mkdir()
            (stopped / "chains" / "wsclean_vlaa.resume").write_text("")
            (stopped / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_MPI_PROCS=7\nNS_METRIC='total_rms_jy - snr'\n"
            )
            (stopped / "parameter-space.json").write_text(
                '[{"name": "channel_count", "min": 1, "max": 8, "kind": "integer"}]'
            )

            runs = {r["name"]: r for r in find_runs(running=set())}
            assert runs[done.name]["status"] == "complete", runs[done.name]
            assert runs[stopped.name]["status"] == "resumable", runs[stopped.name]
            assert runs[stopped.name]["evaluations"] == 2, runs[stopped.name]
            assert runs[stopped.name]["algorithm"] == "wsclean"
            assert runs[stopped.name]["settings"]["NS_METRIC"] == "total_rms_jy - snr"
            assert runs[stopped.name]["settings"]["NS_MPI_PROCS"] == "7"
            assert runs[stopped.name]["parameter_space"] == [
                {"name": "channel_count", "min": 1, "max": 8, "kind": "integer"}]
            # A recorded box is never relabelled as borrowed.
            assert runs[stopped.name]["parameter_space_from_defaults"] is False

            bare = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260103T000000Z"
            bare.mkdir()
            (bare / "run.env").write_text("NS_ALGORITHM=r2d2\n")
            for rank in range(7):
                (bare / "evaluations" / f"eval-0001-{rank}").mkdir(parents=True)
            bare_run = {r["name"]: r for r in find_runs(running=set())}[bare.name]
            assert bare_run["status"] == "incomplete", bare_run
            assert bare_run["evaluations"] == 0, bare_run
            assert bare_run["parameter_space"] == default_parameter_space(), bare_run
            assert bare_run["parameter_space_from_defaults"] is True, bare_run
            (bare / "parameter-space.json").write_text('[{"name": "torn"')
            torn_space = {r["name"]: r for r in find_runs(running=set())}[bare.name]
            assert torn_space["parameter_space"] == default_parameter_space(), (
                "a torn file must not break listing")
            assert torn_space["parameter_space_from_defaults"] is True, torn_space
            (bare / "parameter-space.json").unlink()

            # A legacy or merged run has neither run.env nor parameter-space.json,
            # but the summary it finished with names the metric and the box.
            legacy = NESTED_SAMPLING_DIR / "r2d2-vlaa-merged-20251231T000000Z"
            legacy.mkdir()
            (legacy / "summary.json").write_text(json.dumps({
                "metric": "total_rms_jy",
                "parameter_space": [{"name": "observation_minutes", "min": 4.0, "max": 10.0}],
            }))
            legacy_run = {r["name"]: r for r in find_runs(running=set())}[legacy.name]
            assert legacy_run["settings"]["NS_METRIC"] == "total_rms_jy", legacy_run
            assert legacy_run["parameter_space"] == [
                {"name": "observation_minutes", "min": 4.0, "max": 10.0}], legacy_run
            # run.env still wins where it has an answer of its own.
            (legacy / "run.env").write_text("NS_METRIC=snr\n")
            assert {r["name"]: r for r in find_runs(running=set())
                    }[legacy.name]["settings"]["NS_METRIC"] == "snr"
            # A torn summary must not break listing any more than a torn space does.
            (legacy / "run.env").unlink()
            (legacy / "summary.json").write_text('{"metric": "tot')
            torn = {r["name"]: r for r in find_runs(running=set())}[legacy.name]
            assert torn["settings"] == {}, torn
            # Nothing recorded at all falls back to defaults.toml, flagged.
            assert torn["parameter_space"] == default_parameter_space(), torn
            assert torn["parameter_space_from_defaults"] is True, torn
            shutil.rmtree(legacy)

            order = [r["name"] for r in find_runs(running=set())]
            assert order == [bare.name, stopped.name, done.name], order
            assert bare_run["started"] == "2026-01-03T00:00:00+00:00", bare_run
            assert bare_run["started_label"] == format_started(
                started_at(bare)), bare_run
            hand = NESTED_SAMPLING_DIR / "keep-this-one"
            hand.mkdir()
            (hand / "run.env").write_text("NS_ALGORITHM=r2d2\n")
            os.utime(hand, (0, datetime(2026, 1, 4, tzinfo=timezone.utc).timestamp()))
            assert [r["name"] for r in find_runs(running=set())][0] == hand.name

            noon = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc).timestamp()
            def label(offset_seconds: float) -> str:
                return format_started(noon - offset_seconds, noon)
            assert label(0).startswith("today "), label(0)
            assert label(0).endswith("(just now)"), label(0)
            assert label(2 * 3600).endswith("(2h ago)"), label(2 * 3600)
            assert label(20 * 60).endswith("(20m ago)"), label(20 * 60)
            midnight = datetime.fromtimestamp(noon).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp()
            assert label(noon - midnight + 60).startswith("yesterday "), \
                label(noon - midnight + 60)
            assert label(30 * 86400).endswith("(30d ago)"), label(30 * 86400)
            assert "2026" not in label(30 * 86400), label(30 * 86400)
            assert "2025" in label(400 * 86400), label(400 * 86400)
            assert label(-3600).endswith("(just now)"), label(-3600)

            (hand / "run.env").unlink()
            hand.rmdir()

            stray = NESTED_SAMPLING_DIR / "notes-20260103T000001Z"
            stray.mkdir()
            (stray / "scratch.txt").write_text("not a run\n")
            assert stray.name not in {r["name"] for r in find_runs(running=set())}
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

            live = {str(stopped.resolve())}
            by_name = {r["name"]: r for r in find_runs(running=live)}
            assert by_name[stopped.name]["status"] == "running", by_name[stopped.name]
            assert by_name[done.name]["status"] == "complete", by_name[done.name]
            assert by_name[bare.name]["status"] == "incomplete", by_name[bare.name]

            assert {r["name"]: r for r in find_runs(running={str(done.resolve())})
                    }[done.name]["status"] == "complete"

            # One ps for the whole listing: this used to run per run directory,
            # and that spawn cost was what made ./ri runs - and so the TUI's
            # first frame - slow once there were a few dozen runs.
            saved_running, calls = running_run_dirs, []
            running_run_dirs = lambda ps_output=None: (calls.append(1), live)[1]
            try:
                counted = find_runs()
            finally:
                running_run_dirs = saved_running
            assert len(counted) > 1, counted
            assert calls == [1], (len(calls), len(counted))

            rank = f"python3 /opt/ri-nested-sampling/polychord_r2d2.py --output-dir {stopped.resolve()} --nlive 50"
            ps_output = "\n".join([
                rank,
                f"mpirun --allow-run-as-root -np 16 {rank}",
                f"/usr/bin/docker exec -e NS_MPI_PROCS=16 c mpirun -np 16 {rank}",
                f"python3 /repo/scripts/lib/nested_sampling/r2d2_serve.py --fifo-dir {done.resolve()}/.r2d2-workers",
                "python3 /repo/scripts/nested-sampling-health.py",
            ])
            assert running_run_dirs(ps_output) == {str(stopped.resolve())}, running_run_dirs(ps_output)

            neighbour = f"{stopped.resolve()}-2"
            assert running_run_dirs(
                f"python3 polychord_r2d2.py --output-dir {stopped.resolve()}/ --nlive 5\n"
                f"python3 polychord_r2d2.py --output-dir {neighbour} --nlive 5"
            ) == {str(stopped.resolve()), neighbour}

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
            assert "starts over" in text, text
            assert "Continue where it left off" not in text, text

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                assert main([]) == 0
            text = out.getvalue()
            continue_at, over_at = (text.index("Continue where it left off"),
                                    text.index("starts over"))
            assert continue_at < text.index(f"./ri resume {stopped.name}") < over_at, text
            assert over_at < text.index(f"./ri resume {bare.name}"), text
            assert f"{bare.name:<29}  r2d2       incomplete      0  " in text, text
            assert text.index("STARTED") < text.index(bare.name), text
            for run_dir in (bare, stopped, done):
                assert format_started(started_at(run_dir)) in text, run_dir
            assert text.index(bare.name) < text.index(stopped.name) < \
                text.index(done.name), text

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
