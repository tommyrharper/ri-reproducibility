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
import shlex
import shutil
import signal
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
# The three image ids are here because the images bake the code: two rows at
# one commit measured different binaries if a rebuild landed between them.
# NS_MPI_PROCS is per node, so NS_NODES (in run.env only when above 1) keeps a
# 2-node search out of its 1-node cell (docs/csd3-speed.md, round 306).
WORKLOAD_KEYS = (
    "NS_NLIVE", "NS_NUM_REPEATS", "NS_MAX_NDEAD", "NS_METRIC", "NS_MPI_PROCS",
    "NS_NODES", "NS_IMAGE_DIM",
    "NS_SYNCHRONOUS", "NS_ADOPT_CONTOUR", "NS_WSCLEAN_MGAIN", "NS_WSCLEAN_NITER", "NS_WSCLEAN_LOG_TIME",
    "NS_KEEP_MEASUREMENT_SETS", "WSCLEAN_TARGET_CPU",
    "R2D2_OMP_THREADS", "R2D2_INTEROP_THREADS", "R2D2_MAX_THREADS", "R2D2_INTERLEAVE_WEIGHTS",
    "R2D2_WEIGHTS_PER_NODE", "R2D2_CUDA_BLOCKING_SYNC", "R2D2_RUNTIME", "R2D2_DEVICE",
    "R2D2_TRT_SHARED", "R2D2_TRT_BATCH", "R2D2_TRT_LANES",
    "NS_IMAGER_IMAGE_ID", "NS_MEQTREES_IMAGE_ID", "NS_POLYCHORD_IMAGE_ID",
)

LABEL_WIDTH = 28  # the longest stage name, indented, plus a space
COLUMN_WIDTH = 15  # "0.4656 ±0.0027" is the widest cell the metrics stage makes


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
    # Cluster nodes boot one image and share its machine-id, so there the
    # machine is the node class: CSD3's icelake and ampere rows must not pool.
    if raw and os.environ.get("SLURM_JOB_ID"):
        raw += "|" + cpu_model()
    return hashlib.sha256((raw or platform.node()).encode()).hexdigest()[:8]


def cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor()
    return next((line.split(":", 1)[1].strip() for line in text.splitlines()
                 if line.startswith("model name")), "")


def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def read_run_env(run_dir: Path) -> dict[str, str]:
    # write_run_config quotes with printf %q; same unquoting as
    # scripts/nested-sampling-runs.py.
    try:
        text = (run_dir / "run.env").read_text()
    except OSError:
        return {}
    values = {}
    for name, raw in (line.split("=", 1) for line in text.splitlines() if "=" in line):
        try:
            parsed = shlex.split(raw, comments=False, posix=True)
        except ValueError:
            parsed = [raw.strip()]
        values[name.strip()] = parsed[0] if parsed else ""
    return values


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


def start_gaps(records: Any) -> tuple[float, float, float] | None:
    """(startup, stall, span) seconds from the evaluations' own start epochs.

    A gap is a stretch in which no rank started an evaluation for longer than
    max(2s, 10x the median evaluation). The first is start-up (the first
    evaluations load the imager: 23-88s on a GPU node); later ones are the whole
    search waiting, e.g. on PolyChord's root blocked 20-45s creating its resume
    file on Lustre (round 225; tmpfs chains since 227). Both swamp a 3-minute arm, so evals over the span
    minus the gaps is the rate a long run settles at (docs/csd3-speed.md, round 211).
    """
    timed = sorted((float(t["started_epoch"]), float(t["ended_epoch"]))
                   for t in ((record.get("timing") or {}) for record in records)
                   if t.get("started_epoch") is not None and t.get("ended_epoch") is not None)
    if len(timed) < 2:
        return None
    bound = max(2.0, 10.0 * statistics.median(end - start for start, end in timed))
    first_end = min(end for _, end in timed)
    startup = stall = 0.0
    for (earlier, _), (later, _) in zip(timed, timed[1:]):
        if later - earlier > bound:
            # Every rank starts together, so start-up opens before any ends.
            if earlier < first_end:
                startup += later - earlier
            else:
                stall += later - earlier
    return startup, stall, max(end for _, end in timed) - timed[0][0]


def first_request(records: Any) -> float | None:
    """Median seconds of the evaluations started within 1s of the first.

    When every rank starts together and a cold first request is shorter than
    start_gaps()'s bound (an async whole node: 112 first requests of 5-8s
    against 0.7s warm), no start-to-start gap opens and startup_s reads 0.0,
    so this is the start-up figure there (docs/csd3-speed.md, round 259).
    """
    starts = [(float(t["started_epoch"]), float(t["ended_epoch"]))
              for t in ((record.get("timing") or {}) for record in records)
              if t.get("started_epoch") is not None and t.get("ended_epoch") is not None]
    if not starts:
        return None
    first = min(start for start, _ in starts)
    return statistics.median(end - start for start, end in starts if start - first < 1.0)


def mean_r2d2_iterations(records: Any) -> float | None:
    """Mean U-Net forwards per R2D2 request. R2D2 stops its series early on
    the residual, so a request's cost goes as ~170 + 33 ms per iteration and
    the mean rises ~10-20% as a search climbs (round 317): arms at different
    phases differ in this, not in speed."""
    counts = [record["timing"]["r2d2_iterations"] for record in records or ()
              if isinstance(record, dict) and (record.get("timing") or {}).get("r2d2_iterations")]
    return sum(counts) / len(counts) if counts else None


def pre_run_phases(run_dir: Path, records: Any) -> dict[str, float] | None:
    """Seconds each phase took, from the launcher's start in the job to the
    first evaluation, out of the pre-run.tsv the R2D2 launcher and its ranks
    write; a rank phase ends at its slowest rank (round 341). `searched` is
    the last evaluation's end to the search's exit: rank 0's post-run."""
    try:
        rows = [line.split("\t") for line in (run_dir / "pre-run.tsv").read_text().splitlines()]
    except OSError:
        return None
    marks = []
    for row in rows:
        if len(row) == 3:
            try:
                marks = [] if row[1] == "job" else marks
                marks.append((row[1], float(row[0])))
            except ValueError:
                continue
    timed = [t for t in ((record.get("timing") or {}) for record in records or ())
             if t.get("started_epoch") is not None and t.get("ended_epoch") is not None]
    if not marks or marks[0][0] != "job" or not timed:
        return None
    searched = dict(marks).get("searched")
    # After the search: its own record is written between these two.
    marks = [mark for mark in marks if mark[0] not in ("searched", "recorded", "finished")]
    marks.append(("first_eval", min(float(t["started_epoch"]) for t in timed)))
    phases = {name: round(at - before, 2) for (_, before), (name, at) in zip(marks, marks[1:])}
    phases["total"] = round(marks[-1][1] - marks[0][1], 2)
    if searched is not None:
        phases["searched"] = round(searched - max(float(t["ended_epoch"]) for t in timed), 2)
    return phases


def trt_mean_batch(run_dir: Path) -> float | None:
    """The gathering TensorRT server's mean batch, from the r2d2-trt-batches.txt it keeps (round 330)."""
    try:
        words = (run_dir / "r2d2-trt-batches.txt").read_text().split()
        return float(words[words.index("mean") + 1])
    except (OSError, ValueError, IndexError):
        return None


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
    busy_wall = breakdown.get("busy_wall_seconds")

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
    gaps = start_gaps(summary.get("evaluations", []))
    steady = gaps and gaps[2] - gaps[0] - gaps[1]
    first = first_request(summary.get("evaluations", []))
    iterations = mean_r2d2_iterations(summary.get("evaluations", []))
    batch = trt_mean_batch(run_dir)
    pre_run = pre_run_phases(run_dir, summary.get("evaluations", []))
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
        "busy_wall_s": round(busy_wall, 3) if busy_wall else None,
        "idle_fraction": round(max(0.0, 1.0 - busy_wall / wall), 4)
        if busy_wall else None,
        "ms_per_eval": round(per_eval * 1000.0, 4) if per_eval else None,
        "startup_s": round(gaps[0], 1) if gaps else None,
        "stall_s": round(gaps[1], 1) if gaps else None,
        "first_request_s": round(first, 2) if first is not None else None,
        "steady_evals_per_s": round(evals / steady, 4) if steady and steady > 0 else None,
        "peak_memory_mb": round(peak_memory / (1024.0 ** 2), 1) if peak_memory else None,
        "r2d2_iterations": round(iterations, 3) if iterations else None,
        **({"trt_mean_batch": batch} if batch else {}),
        **({"pre_run_s": pre_run} if pre_run else {}),
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


def arm_for(attempt: int, arms: list[str]) -> str:
    """Which arm run `attempt` uses; attempt 0 is the unrecorded warm-up.

    The warm-up repeats the first arm rather than consuming one, so `--repeat N`
    is N recorded runs *per arm* and the arms stay balanced.
    """
    return arms[max(0, attempt - 1) % len(arms)]


def do_run(args: argparse.Namespace) -> int:
    """Run the controlled search a preset defines, then let it record itself.

    Goes through ./ri search rather than the run script, so a benchmark run is
    the documented `NS_NLIVE=8 ./ri search wsclean` form and nothing about how
    a search starts is written down twice.

    On a cluster login node the whole benchmark is one Slurm job instead: the
    warm-up and every repeat have to meet the same node, and an interleave is
    only an interleave if its arms share a machine. Inside the job (SLURM_JOB_ID
    is set) the searches run in place, one after the other.
    """
    if (not os.environ.get("SLURM_JOB_ID") and os.environ.get("NS_SBATCH", "1") == "1"
            and shutil.which("sbatch")):
        return submit(args)
    env = {**os.environ, **preset_settings(args.preset, args.imager)}
    if args.allow_oversubscription:
        env["NS_MPI_OVERSUBSCRIBE"] = "1"
    if args.mpi_procs is not None:
        env["NS_MPI_PROCS"] = str(args.mpi_procs)
    if args.omp_threads is not None:
        env["R2D2_OMP_THREADS"] = str(args.omp_threads)
    # An A/B arm has to meet the same host the other one did, and this host
    # drifts by more than most effects are worth, so the two settings alternate
    # run by run rather than running as two blocks.
    setting, arms = (args.interleave[0], args.interleave[1:]) if args.interleave else (None, [None])
    command = ["./ri", "search", args.imager]
    # One unrecorded search first, to leave the host in the state every
    # recorded row is measured in. The first search after an idle spell
    # measured 8-16% faster than the ones behind it here: the package spends a
    # power budget it then has to pay back (docs/nested-sampling-power-limit.md),
    # so a cold first run is effectively a faster machine.
    for attempt in range(args.repeat * len(arms) + 1):
        if setting:
            env[setting] = arm_for(attempt, arms)
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


def submit(args: argparse.Namespace) -> int:
    """Hand this same command to sbatch through slurm.sh, sized like a search.

    The job is named bench-<imager>-<stamp> and keeps its Slurm log under
    results/bench/; the searches it runs claim their own directories under
    results/nested-sampling as usual. `./ri bench run` sets NS_MPI_PROCS
    only per search, so the job takes a whole node unless --mpi-procs asks for
    less, the same rule `./ri search` follows.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_dir = REPO_ROOT / "results" / "bench" / f"bench-{args.imager}-{stamp}"
    job_dir.mkdir(parents=True)
    env = {**os.environ, "REPO_ROOT": str(REPO_ROOT)}
    if args.mpi_procs is not None:
        env["NS_MPI_PROCS"] = str(args.mpi_procs)
    script = (". scripts/lib/defaults.sh; . scripts/lib/rank-budget.sh; . scripts/lib/slurm.sh; "
              'mb="${NS_WSCLEAN_MB_PER_RANK}"; [ "$1" = r2d2 ] && mb="${NS_R2D2_MB}"; '
              'ns_submit_run "$2" "${mb}" "${@:3}"')
    returncode = subprocess.run(
        ["bash", "-c", script, "bench", args.imager, str(job_dir),
         "uv", "run", "scripts/bench.py", *sys.argv[1:]],
        cwd=REPO_ROOT, env=env,
    ).returncode
    if returncode:  # a refused submission (a bad account, say) leaves nothing behind
        job_dir.rmdir()
    return returncode


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
    # Hand-logged stage timings ("kind": "stage", rounds 201-203) are not runs,
    # and since round 245 some carry an "imager" too; some are marked
    # "preset": "stage" instead and carry no machine (round 341).
    return [row for row in rows if "imager" in row and "machine" in row
            and row.get("kind") != "stage" and row.get("preset") != "stage"]


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


# A converged async search needs ~0.41 x workers live points' worth of extra
# evaluations whatever nlive is (docs/csd3-speed.md, round 307), so raw evals/s
# flatters more ranks. Scaled by nlive/(nlive + 0.41 x workers) it is the rate
# a converged search makes progress at, and that compares across rank counts
# and nodes. Workers leave out PolyChord's root; synchronous runs get no line.
ASYNC_OVERHEAD_PER_WORKER = 0.41


def useful_evals_per_s(row: dict[str, Any]) -> float | None:
    settings = row.get("settings") or {}
    rate = row.get("steady_evals_per_s") or row.get("evals_per_s")
    try:
        nlive = int(settings["NS_NLIVE"])
        workers = int(settings["NS_MPI_PROCS"]) * int(settings.get("NS_NODES", "1")) - 1
    except (KeyError, ValueError):
        return None
    if settings.get("NS_SYNCHRONOUS", "0") != "0" or not rate or nlive <= 0 or workers < 1:
        return None
    return round(rate * nlive / (nlive + ASYNC_OVERHEAD_PER_WORKER * workers), 4)


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
        # Joined with a space rather than padded to the full width, so a cell
        # that fills its column is still separated from the next one.
        print((f"{label:<{LABEL_WIDTH}}"
               + " ".join(f"{text:<{COLUMN_WIDTH - 1}}" for text in cells)).rstrip())

    line("", [f"{label}" for label in labels])
    line("", [max(by_commit[label], key=lambda r: r["at"])["at"][5:10]
              for label in labels])
    line("", [f"n={len(by_commit[label])}" for label in labels])

    def row_for_path(label: str, path: tuple[str, ...]) -> None:
        # A field no row in this group carries - busy wall and idle fraction
        # predate most of the ledger - is left out rather than printed blank.
        cells = [cell(series(by_commit[c], path)) for c in labels]
        if any(cells):
            line(label, cells)

    row_for_path("evals/s", ("evals_per_s",))
    line("Δ evals/s", [delta(series(by_commit[new], ("evals_per_s",)),
                            series(by_commit[old], ("evals_per_s",)))
                       for new, old in zip(labels, labels[1:])])
    row_for_path("steady evals/s", ("steady_evals_per_s",))
    useful = [cell([v for v in map(useful_evals_per_s, by_commit[c]) if v is not None])
              for c in labels]
    if any(useful):
        line("converging evals/s", useful)
    row_for_path("stall s", ("stall_s",))
    row_for_path("first request s", ("first_request_s",))
    row_for_path("ms/eval", ("ms_per_eval",))
    row_for_path("busy wall s", ("busy_wall_s",))
    row_for_path("idle fraction", ("idle_fraction",))
    row_for_path("peak memory MB", ("peak_memory_mb",))
    row_for_path("R2D2 iterations", ("r2d2_iterations",))
    row_for_path("TensorRT mean batch", ("trt_mean_batch",))
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
    import tempfile

    global LEDGER

    assert machine_id() == machine_id(), "machine id must be stable"
    assert len(machine_id()) == 8, machine_id()
    saved = os.environ.pop("SLURM_JOB_ID", None)
    try:
        outside = machine_id()
        os.environ["SLURM_JOB_ID"] = "1"
        assert machine_id() != outside or not cpu_model(), "a Slurm node's id must carry its CPU model"
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
        if saved is not None:
            os.environ["SLURM_JOB_ID"] = saved

    median, error = stat([10.0, 12.0])
    assert median == 11.0 and abs(error - 0.657) < 1e-3, (median, error)
    assert stat([10.0]) == (10.0, None)
    # The error bar shrinks as repeats accumulate; that is the whole point.
    four = stat([10.0, 11.0, 11.0, 12.0])[1]
    assert four is not None and four < error, (four, error)
    assert stat([100.0, 100.0, 100.0, 1000.0])[0] == 100.0
    # A cell that exactly fills its column must not run into the next one.
    assert len(cell([0.4656, 0.4600])) < COLUMN_WIDTH, cell([0.4656, 0.4600])

    # The warm-up repeats arm A, then the recorded runs alternate: two repeats
    # of two arms is A A B A B, so each arm is measured --repeat times.
    assert [arm_for(n, ["A", "B"]) for n in range(5)] == ["A", "A", "B", "A", "B"]
    assert [arm_for(n, [None]) for n in range(3)] == [None, None, None]

    assert delta([12.0], [10.0]) == "+20.0%", delta([12.0], [10.0])
    assert delta([10.0], []) == "", "a missing side has no delta"
    # One repeat each cannot be significant; many tight ones can.
    assert not delta([12.0], [10.0]).endswith("*")
    assert delta([12.0, 12.1], [10.0, 10.1]).endswith("*")
    assert not delta([10.0, 1000.0], [1.0, 100.0]).endswith("*")

    assert series([{"stages_ms": {"simulate": 3.0}}], ("stages_ms", "simulate")) == [3.0]
    assert series([{"stages_ms": {}}], ("stages_ms", "simulate")) == []
    assert mean_r2d2_iterations([{"timing": {"r2d2_iterations": 25}}, {"timing": {"r2d2_iterations": 9}},
                                 {"timing": {}}, {"objective": 0.0}]) == 17.0
    assert mean_r2d2_iterations([]) is None
    with tempfile.TemporaryDirectory() as tmp:
        assert trt_mean_batch(Path(tmp)) is None
        (Path(tmp) / "r2d2-trt-batches.txt").write_text("launches 3 forwards 6 mean 2.000 sizes 1:2 4:1\n")
        assert trt_mean_batch(Path(tmp)) == 2.0
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
    # Two nodes at the same per-node NS_MPI_PROCS is twice the work rate.
    one, two = ({key: value for key, value in env.items() if key in WORKLOAD_KEYS}
                for env in ({"NS_MPI_PROCS": "39"}, {"NS_MPI_PROCS": "39", "NS_NODES": "2"}))
    assert group_key({"imager": "r2d2", "preset": "custom", "machine": "m", "settings": one}) \
        != group_key({"imager": "r2d2", "preset": "custom", "machine": "m", "settings": two})

    # Round 288: 2 x 39 ranks at nlive 300, 99.1 evals/s, is 77 workers.
    two_nodes = {"steady_evals_per_s": 99.1, "settings": {
        "NS_NLIVE": "300", "NS_MPI_PROCS": "39", "NS_NODES": "2", "NS_SYNCHRONOUS": "0"}}
    assert useful_evals_per_s(two_nodes) == round(99.1 * 300 / (300 + 0.41 * 77), 4)
    one_node = {"evals_per_s": 50.0, "settings": {"NS_NLIVE": "300", "NS_MPI_PROCS": "39"}}
    assert useful_evals_per_s(one_node) == round(50.0 * 300 / (300 + 0.41 * 38), 4)
    assert useful_evals_per_s({**one_node, "settings": {**one_node["settings"],
                                                        "NS_SYNCHRONOUS": "1"}}) is None
    assert useful_evals_per_s({"evals_per_s": 1.0, "settings": {}}) is None

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

    # An interleaved setting the ledger does not group by would file both arms
    # in one cell, so the probe would read as noise rather than as a comparison.
    guard = argparse.ArgumentParser()
    for probe, expected in ((["NS_WSCLEAN_MGAIN", "0.8", "0.9"], True),
                            (["NS_SEED", "1", "2"], False)):
        try:
            validate_run(guard, argparse.Namespace(
                repeat=1, timeout=None, interleave=probe, mpi_procs=None,
                allow_oversubscription=False))
            accepted = True
        except SystemExit:
            accepted = False
        assert accepted is expected, probe

    # Start-up and a mid-run stall are cut; ordinary spacing is not.
    timed = [{"timing": {"started_epoch": t, "ended_epoch": t + 0.3}}
             for t in [0.0, 0.0] + [30.0 + 0.1 * n for n in range(100)] + [50.0 + 0.1 * n for n in range(100)]]
    startup, stall, span = start_gaps(timed)
    assert (startup, round(stall, 1), round(span, 1)) == (30.0, 10.1, 60.2), (startup, stall, span)
    assert start_gaps([{"timing": {"started_epoch": 0.0, "ended_epoch": 1.0}},
                       {"timing": {"started_epoch": 0.5, "ended_epoch": 1.5}}]) == (0.0, 0.0, 1.5)
    assert start_gaps([{"timing": {}}]) is None
    # All ranks' cold first requests together: no gap, but the first wave is slow.
    wave = [{"timing": {"started_epoch": 0.1 * n, "ended_epoch": 6.0 + 0.1 * n}} for n in range(5)]
    wave += [{"timing": {"started_epoch": 6.0 + 0.1 * n, "ended_epoch": 6.7 + 0.1 * n}} for n in range(20)]
    assert start_gaps(wave)[0] == 0.0 and first_request(wave) == 6.0, first_request(wave)
    assert first_request([{"timing": {}}]) is None

    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        env_dir = Path(raw) / "env"
        env_dir.mkdir()
        (env_dir / "run.env").write_text("NS_METRIC=total_rms_jy\\ -\\ 0.5\\ *\\ snr\n")
        assert read_run_env(env_dir)["NS_METRIC"] == "total_rms_jy - 0.5 * snr"

        # The last launch's marks only (a resume appends), torn lines skipped.
        (env_dir / "pre-run.tsv").write_text(
            "1.0\tjob\t1.0\n5.0\tlaunch\t5.0\n100.0\tjob\t100.0\n101.5\tsized\t101.5\n"
            "103.0\tlaunch\t103.0\n109.0\twarm\t106.0\n130.0\tsearched\t130.0\n133.0\tfinished\t133.0\n10")
        assert pre_run_phases(env_dir, [{"timing": {"started_epoch": 110.0, "ended_epoch": 125.0}},
                                        {"timing": {}}]) == {
            "sized": 1.5, "launch": 1.5, "warm": 6.0, "first_eval": 1.0, "total": 10.0, "searched": 5.0}
        assert pre_run_phases(env_dir, []) is None
        assert pre_run_phases(Path(raw), [{"timing": {"started_epoch": 1.0}}]) is None

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
                "busy_wall_s": 0.9, "idle_fraction": 0.1,
            }) + "\nnot json\n" + json.dumps({"kind": "stage", "stage": "s"}) + "\n"
                + json.dumps({"kind": "stage", "imager": "r2d2", "run": "s"}) + "\n"
                + json.dumps({"preset": "stage", "imager": "r2d2", "run": "s"}) + "\n")
            rows = load_rows()
            assert len(rows) == 1, rows
            do_table()
        finally:
            LEDGER = ledger

    print("OK: benchmark ledger")


def validate_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject a probe that would silently measure nothing, or oversubscribe.

    Arms only mean anything if the ledger groups them apart, which it does by
    WORKLOAD_KEYS - so an interleaved setting outside that set would file both
    arms in one cell and read as noise.
    """
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.interleave and args.interleave[0] not in WORKLOAD_KEYS:
        # The image ids are derived from what is built, not set, so they are
        # in the group key but not offered as something to interleave.
        settable = [key for key in WORKLOAD_KEYS if not key.endswith("_IMAGE_ID")]
        parser.error(f"--interleave {args.interleave[0]} is not one of the "
                     f"settings the ledger groups by: {', '.join(settable)}")
    ranks = [args.mpi_procs] if args.mpi_procs is not None else []
    if args.interleave and args.interleave[0] == "NS_MPI_PROCS":
        try:
            ranks += [int(value) for value in args.interleave[1:]]
        except ValueError:
            parser.error("interleaved NS_MPI_PROCS values must be whole numbers")
    if any(rank < 1 for rank in ranks):
        parser.error("MPI worker counts must be at least 1")
    # Open MPI reports this as a slot error minutes into a probe otherwise.
    available = available_cpu_count()
    if ranks and not args.allow_oversubscription and max(ranks) > available:
        parser.error(f"MPI worker count cannot exceed {available} available "
                     "CPU slots without --allow-oversubscription")


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
    run.add_argument("--mpi-procs", type=int, metavar="N",
                     help="override MPI worker count for rank-scaling probes")
    run.add_argument("--omp-threads", type=int, metavar="N",
                     help="override per-rank R2D2 OpenMP/BLAS threads")
    run.add_argument("--interleave", nargs=3, metavar=("SETTING", "A", "B"),
                     help="alternate one workload setting between two values; "
                          "--repeat is per arm")
    run.add_argument("--allow-oversubscription", action="store_true",
                     help="allow MPI counts above CPU affinity for explicit probes")
    run.set_defaults(handler=do_run)

    args = parser.parse_args()
    if getattr(args, "handler", None) is do_run:
        validate_run(parser, args)

    if args.self_check:
        self_check()
        return 0
    return getattr(args, "handler", do_table)(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `./ri bench | head` is normal report consumption, not a failure.
        # Python flushes stdout on the way out and would raise this again, so
        # the real file descriptor goes to /dev/null first.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
