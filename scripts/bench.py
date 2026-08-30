#!/usr/bin/env python3
"""Throughput per commit, machine and settings: append a row, print the table.

Rows live in `benchmarks.jsonl` at the repository root, one JSON object per
line, appended by every search that finishes. Reading groups them and reports
a median with a robust standard-error estimate, so one long-tail run cannot
dominate a group's result. See docs/nested-sampling-benchmarks.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import shutil
import statistics
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from common import backfill_busy_seconds, profiling_breakdown  # noqa: E402

LEDGER = REPO_ROOT / "benchmarks.jsonl"
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling"

# The run.env keys that change what a run costs, and so what a row can be
# compared against. NS_SEED is deliberately absent: it changes which points are
# drawn, not the configuration being measured, and it is randomised per run -
# in the key, every ad-hoc search would be a group of its own. NS_RETRIES and
# NS_STALL_TIMEOUT are absent because they cost nothing until something breaks.
WORKLOAD_KEYS = (
    "NS_NLIVE", "NS_NUM_REPEATS", "NS_MAX_NDEAD", "NS_METRIC", "NS_MPI_PROCS",
    "NS_SYNCHRONOUS", "NS_WSCLEAN_MGAIN", "NS_KEEP_MEASUREMENT_SETS",
    "R2D2_OMP_THREADS", "WSCLEAN_TARGET_CPU",
)

LABEL_WIDTH = 28  # the longest stage name, indented, plus a space
COLUMN_WIDTH = 13


# --- recording ---------------------------------------------------------------


def git(*args: str, default: str = "") -> str:
    try:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, check=False)
    except OSError:
        return default
    return done.stdout.strip() if done.returncode == 0 else default


def machine_id() -> str:
    """A stable id for this host, so a laptop's rows never pool with a server's.

    Survives reboots, reinstalls of this repository and a rename, which a
    hostname does not - the same machine must keep the same id for its history
    to stay one history.
    """
    raw = ""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw:
            break
    if not raw and sys.platform == "darwin":
        found = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, check=False).stdout
        for line in found.splitlines():
            if "IOPlatformUUID" in line and '"' in line:
                raw = line.rsplit('"', 2)[-2]
                break
    return hashlib.sha256((raw or platform.node()).encode()).hexdigest()[:8]


def read_run_env(run_dir: Path) -> dict[str, str]:
    # write_run_config quotes with printf %q; same unquoting as
    # scripts/nested-sampling-runs.py.
    try:
        text = (run_dir / "run.env").read_text()
    except OSError:
        return {}
    return {name.strip(): raw.strip().strip("'").replace("'\\''", "'")
            for name, raw in (line.split("=", 1)
                              for line in text.splitlines() if "=" in line)}


def presets() -> dict[str, dict[str, dict[str, Any]]]:
    with open(REPO_ROOT / "defaults.toml", "rb") as handle:
        return tomllib.load(handle).get("benchmark", {})


def preset_settings(name: str, imager: str) -> dict[str, str]:
    defined = presets()
    if name not in defined:
        raise SystemExit(f"no [benchmark.{name}] in defaults.toml; have: "
                         f"{', '.join(sorted(defined)) or 'none'}")
    settings = defined[name].get(imager)
    if not settings:
        raise SystemExit(f"no [benchmark.{name}.{imager}] in defaults.toml")
    return {key: str(value) for key, value in settings.items()}


def preset_for(run_env: dict[str, str], imager: str) -> str:
    """Which preset this run was, judged by its run.env rather than by a flag.

    So a search run by hand with the preset's settings counts as one, and the
    preset needs no plumbing through the shell scripts to say what it was.
    Matched against the whole run.env, not the grouping keys, so a preset can
    pin the seed - which is what makes its repeats measure the same work.
    """
    for name, per_imager in presets().items():
        want = per_imager.get(imager) or {}
        if want and all(str(value) == run_env.get(key)
                        for key, value in want.items()):
            return name
    return "custom"


def resolve_run(raw: str) -> Path:
    target = Path(raw).expanduser()
    if not target.exists() and (NESTED_SAMPLING_DIR / raw).is_dir():
        return NESTED_SAMPLING_DIR / raw
    return target.resolve()


def row_for(run_dir: Path) -> dict[str, Any] | None:
    """One ledger row for a finished run, or None with a reason on stderr.

    Every reason is a normal thing for a run to be, so recording is best
    effort: the run has already finished, and no measurement is worth failing
    it after the fact.
    """
    def skip(why: str) -> None:
        print(f"benchmark: not recording {run_dir.name}: {why}", file=sys.stderr)

    if not run_dir.is_dir():
        return skip("no such run directory")
    if NESTED_SAMPLING_DIR not in run_dir.parents:
        return skip(f"outside {NESTED_SAMPLING_DIR.relative_to(REPO_ROOT)}")
    if (run_dir / "restarts.log").exists():
        # Its wall clock covers one segment of the run and its evaluation count
        # covers all of them, so the throughput it implies is fiction.
        return skip("it restarted or was resumed")
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return skip("no whole summary.json - it did not finish")

    breakdown = profiling_breakdown(backfill_busy_seconds(summary),
                                    summary.get("algorithm"))
    wall, evals = breakdown["total_wall_seconds"], breakdown["evals"]
    if not wall or not evals:
        return skip("its summary.json carries no profiling block")

    imager = summary.get("algorithm") or "unknown"
    run_env = read_run_env(run_dir)
    settings = {key: value for key, value in run_env.items() if key in WORKLOAD_KEYS}
    per_eval = breakdown["subtotal_per_eval_seconds"]
    peak_memory = max(
        (float((record.get("metrics") or {}).get("peak_memory_bytes", 0.0))
         or float(record.get("peak_memory_bytes", 0.0))
         for record in summary.get("evaluations", [])
         if (record.get("metrics") or {}).get("peak_memory_bytes")
         or record.get("peak_memory_bytes")),
        default=0.0,
    )
    return {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git("rev-parse", "--short=7", "HEAD") or "unknown",
        # The ledger's own line makes the tree dirty the moment it is written,
        # so it does not count as a change to what was measured.
        "dirty": any(not line.endswith(LEDGER.name)
                     for line in git("status", "--porcelain").splitlines()),
        "machine": machine_id(),
        "host": platform.node(),
        "imager": imager,
        "preset": preset_for(run_env, imager),
        "run": run_dir.name,
        "evals": evals,
        "wall_s": round(wall, 3),
        "evals_per_s": round(evals / wall, 4),
        "ms_per_eval": round(per_eval * 1000.0, 4) if per_eval else None,
        "peak_memory_mb": round(peak_memory / (1024.0 ** 2), 1) if peak_memory else None,
        "stages_ms": {row["key"]: round(row["per_eval_seconds"] * 1000.0, 4)
                      for row in breakdown["rows"] if row["per_eval_seconds"]},
        "settings": settings,
    }


def do_record(args: argparse.Namespace) -> int:
    if os.environ.get("NS_BENCH_RECORD") == "0":
        print("benchmark: warm-up run, not recorded", file=sys.stderr)
        return 0
    row = row_for(resolve_run(args.run))
    if row is None:
        return 0
    # One short line, opened for append: concurrent runs finishing together
    # interleave whole lines rather than tearing one.
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"benchmark: {row['commit']}{'+' if row['dirty'] else ''} "
          f"{row['imager']} {row['preset']} {row['evals_per_s']:.1f} evals/s "
          f"-> {LEDGER.name}")
    return 0


# --- running -----------------------------------------------------------------


def do_run(args: argparse.Namespace) -> int:
    """Run the controlled search a preset defines, then let it record itself.

    Goes through ./ri search rather than the run script, so a benchmark run is
    the documented `NS_NLIVE=8 ./ri search wsclean` form and nothing about how
    a search starts is written down twice. ./ri bench run has already built.
    """
    env = {**os.environ, **preset_settings(args.preset, args.imager)}
    command = ["./ri", "search", args.imager, "--no-build"]
    # One unrecorded search first, to leave the host in the state every
    # recorded row is measured in. The first search after an idle spell
    # measured 8-16% faster than the ones behind it here: the package spends a
    # power budget it then has to pay back (docs/nested-sampling-power-limit.md),
    # so a cold first run is effectively a faster machine.
    for attempt in range(args.repeat + 1):
        warm_up = {"NS_BENCH_RECORD": "0"} if attempt == 0 else {}
        process = subprocess.Popen(command, cwd=REPO_ROOT,
                                   env={**env, **warm_up},
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"benchmark: timeout after {args.timeout}s; stopping run",
                  file=sys.stderr)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124
        if returncode:
            return returncode
    return 0


# --- reading -----------------------------------------------------------------


def load_rows() -> list[dict[str, Any]]:
    try:
        text = LEDGER.read_text()
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A row half written by a killed run costs that row, not the file.
                print(f"benchmark: skipping unreadable line in {LEDGER.name}",
                      file=sys.stderr)
    return rows


def commit_order() -> dict[str, int]:
    """Commit -> commit time, so re-measuring an old commit still sorts by age.

    Keyed on seven characters at both ends: a row records `--short=7`, but %h
    is as long as this repository needs a hash to be unambiguous.
    """
    order = {}
    for line in git("log", "--format=%h %ct", "--all").splitlines():
        short, _, when = line.partition(" ")
        order[short[:7]] = int(when)
    return order


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["imager"], row["preset"], row["machine"],
            json.dumps(row.get("settings", {}), sort_keys=True))


def plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def stat(values: list[float]) -> tuple[float, float | None]:
    """Median and IQR-based standard error, robust to long-tail timings."""
    if len(values) < 2:
        return statistics.median(values), None
    q1, q3 = statistics.quantiles(values, n=4, method="inclusive")[::2]
    return statistics.median(values), 0.93 * (q3 - q1) / len(values) ** 0.5


def number(value: float, digits: int = 4) -> str:
    # Plain digits past four of them: %g would turn R2D2's 14190 ms into
    # 1.419e+04, which no column of a table should have to hold.
    return f"{value:.0f}" if abs(value) >= 10 ** digits else f"{value:.{digits}g}"


def cell(values: list[float]) -> str:
    if not values:
        return ""
    median, error = stat(values)
    return number(median) if error is None else f"{number(median)} ±{number(error, 2)}"


def delta(new: list[float], old: list[float]) -> str:
    """Percentage change against the previous commit, starred when it is real.

    Starred means the two medians are more than two combined standard errors
    apart - one repeat each can never earn a star, which is the point.
    """
    if not new or not old:
        return ""
    new_mean, new_error = stat(new)
    old_mean, old_error = stat(old)
    if not old_mean:
        return ""
    change = (new_mean - old_mean) / old_mean * 100.0
    spread = ((new_error or 0.0) ** 2 + (old_error or 0.0) ** 2) ** 0.5
    star = "*" if spread and abs(new_mean - old_mean) > 2 * spread else ""
    return f"{change:+.1f}%{star}"


def series(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        value: Any = row
        for step in path:
            value = (value or {}).get(step) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def print_group(key: tuple[str, str, str, str], rows: list[dict[str, Any]],
                columns: int, order: dict[str, int]) -> None:
    imager, preset, machine, settings_json = key
    settings = json.loads(settings_json)
    host = rows[-1].get("host") or "?"
    print(f"{imager} · {preset} · machine {machine} ({host})")
    print("  " + "  ".join(f"{name.removeprefix('NS_').lower()}={value}"
                           for name, value in sorted(settings.items())))
    print()

    by_commit: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row["commit"] + ("+" if row.get("dirty") else "")
        by_commit.setdefault(label, []).append(row)
    # Newest first, by commit time where this checkout knows it - a row
    # measured on a commit that is gone falls back to when it was measured.
    labels = sorted(by_commit,
                    key=lambda label: (order.get(label.rstrip("+")[:7], 0),
                                       max(r["at"] for r in by_commit[label])),
                    reverse=True)[:columns]

    def line(label: str, cells: list[str]) -> None:
        print((f"{label:<{LABEL_WIDTH}}"
               + "".join(f"{text:<{COLUMN_WIDTH}}" for text in cells)).rstrip())

    line("", [f"{label}" for label in labels])
    line("", [max(by_commit[label], key=lambda r: r["at"])["at"][5:10]
              for label in labels])
    line("", [f"n={len(by_commit[label])}" for label in labels])

    def row_for_path(label: str, path: tuple[str, ...]) -> None:
        line(label, [cell(series(by_commit[c], path)) for c in labels])

    row_for_path("evals/s", ("evals_per_s",))
    line("Δ evals/s", [delta(series(by_commit[new], ("evals_per_s",)),
                            series(by_commit[old], ("evals_per_s",)))
                       for new, old in zip(labels, labels[1:])])
    row_for_path("ms/eval", ("ms_per_eval",))
    row_for_path("peak memory MB", ("peak_memory_mb",))
    stages: list[str] = []
    for row in rows:
        stages += [key for key in row.get("stages_ms", {}) if key not in stages]
    for stage in stages:
        row_for_path("  " + stage, ("stages_ms", stage))
    row_for_path("evals", ("evals",))
    print()


def do_table(_args: argparse.Namespace | None = None) -> int:
    rows = load_rows()
    if not rows:
        print(f"No rows in {LEDGER.name} yet.\n"
              "Every search that finishes adds one; ./ri bench run wsclean "
              "runs the controlled one.")
        return 0
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    width = shutil.get_terminal_size((100, 24)).columns
    columns = max(2, (width - LABEL_WIDTH) // COLUMN_WIDTH)
    order = commit_order()
    print(f"{LEDGER.name}: {plural(len(rows), 'row')}, "
          f"{plural(len(grouped), 'group')}\n")
    # Presets first, then the widest history: the controlled measurement is
    # what a regression is judged on, and ad-hoc searches pile up behind it.
    for key in sorted(grouped, key=lambda k: (k[1] == "custom", -len(grouped[k]))):
        print_group(key, grouped[key], columns, order)
    return 0


# --- self-check --------------------------------------------------------------


def self_check() -> None:
    global LEDGER

    assert machine_id() == machine_id(), "machine id must be stable"
    assert len(machine_id()) == 8, machine_id()

    median, error = stat([10.0, 12.0])
    assert median == 11.0 and abs(error - 0.657) < 1e-3, (median, error)
    assert stat([10.0]) == (10.0, None)
    # The error bar shrinks as repeats accumulate; that is the whole point.
    four = stat([10.0, 11.0, 11.0, 12.0])[1]
    assert four is not None and four < error, (four, error)
    assert stat([100.0, 100.0, 100.0, 1000.0])[0] == 100.0

    assert delta([12.0], [10.0]) == "+20.0%", delta([12.0], [10.0])
    assert delta([10.0], []) == "", "a missing side has no delta"
    # One repeat each cannot be significant; many tight ones can.
    assert not delta([12.0], [10.0]).endswith("*")
    assert delta([12.0, 12.1], [10.0, 10.1]).endswith("*")
    assert not delta([10.0, 1000.0], [1.0, 100.0]).endswith("*")

    assert series([{"stages_ms": {"simulate": 3.0}}], ("stages_ms", "simulate")) == [3.0]
    assert series([{"stages_ms": {}}], ("stages_ms", "simulate")) == []
    assert series([{"ms_per_eval": None}], ("ms_per_eval",)) == []

    defined = presets()
    assert "default" in defined, defined
    for imager in ("wsclean", "r2d2"):
        settings = preset_settings("default", imager)
        assert preset_for(settings, imager) == "default", settings
        assert preset_for({**settings, "NS_NLIVE": "999"}, imager) == "custom"
        throughput = preset_settings("throughput", imager)
        assert throughput["NS_SYNCHRONOUS"] == "0", throughput
        assert preset_for(throughput, imager) == "throughput", throughput
        production = preset_settings("production", imager)
        assert production["NS_NLIVE"] == "150", production
        assert production["NS_NUM_REPEATS"] == "15", production
        assert production["NS_MAX_NDEAD"] == "-1", production
        assert production["NS_SYNCHRONOUS"] == "0", production
        assert preset_for(production, imager) == "production", production
    # A preset only claims a run that matches every key it pins, but a run may
    # carry keys the preset says nothing about - NS_MPI_PROCS is host-derived.
    assert preset_for({**preset_settings("default", "wsclean"),
                       "NS_MPI_PROCS": "3"}, "wsclean") == "default"
    assert preset_for({}, "wsclean") == "custom"

    timeout = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        try:
            timeout.wait(timeout=0.01)
            raise AssertionError("timeout self-check child finished unexpectedly")
        except subprocess.TimeoutExpired:
            os.killpg(timeout.pid, signal.SIGTERM)
            timeout.wait(timeout=5)
    finally:
        if timeout.poll() is None:
            os.killpg(timeout.pid, signal.SIGKILL)
            timeout.wait()

    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        run = Path(raw) / "wsclean-vlaa-20260101T000000Z"
        run.mkdir()
        assert row_for(run) is None, "a run outside results/nested-sampling"

        ledger, LEDGER = LEDGER, Path(raw) / "benchmarks.jsonl"
        try:
            LEDGER.write_text(json.dumps({
                "at": "2026-01-01T00:00:00Z", "commit": "aaaaaaa", "dirty": False,
                "machine": "1234abcd", "host": "h", "imager": "wsclean",
                "preset": "default", "run": "r", "evals": 100, "wall_s": 1.0,
                "evals_per_s": 100.0, "ms_per_eval": 140.0,
                "peak_memory_mb": 256.0,
                "stages_ms": {"simulate": 40.0}, "settings": {"NS_NLIVE": "8"},
            }) + "\nnot json\n")
            rows = load_rows()
            assert len(rows) == 1, rows
            do_table()
        finally:
            LEDGER = ledger

    print("OK: benchmark ledger")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    subs = parser.add_subparsers(metavar="<command>")

    record = subs.add_parser("record", help="append one row for a finished run")
    record.add_argument("run", help="run directory or its name")
    record.set_defaults(handler=do_record)

    run = subs.add_parser("run", help="run a preset's controlled search")
    run.add_argument("imager", choices=("wsclean", "r2d2"))
    run.add_argument("--preset", default="default")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--timeout", type=float, metavar="SECONDS",
                     help="stop the whole benchmark, including its process tree, after this time")
    run.set_defaults(handler=do_run)

    args = parser.parse_args()
    if getattr(args, "handler", None) is do_run and args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if getattr(args, "handler", None) is do_run and args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    if args.self_check:
        self_check()
        return 0
    return getattr(args, "handler", do_table)(args)


if __name__ == "__main__":
    sys.exit(main())
