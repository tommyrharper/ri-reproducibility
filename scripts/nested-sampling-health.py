#!/usr/bin/env python3
"""Say whether a nested-sampling run is healthy, stalled, stopped - or worthless.

`./ri runs` answers "did this run finish?". This answers the question you have
while one is still going: is it actually making progress, and is the progress
worth anything? Both are needed, and neither is visible from the run directory
without knowing what to look at.

Four things it checks, each one a way a run has actually gone wrong here:

* **Progress.** `evaluations/eval-*/metrics.json` is written only when an
  evaluation succeeds, so its count is the progress, its newest mtime is the
  last sign of life, and the directories without one are the evaluations in
  flight (which should sit near `NS_MPI_PROCS`).
* **Liveness.** A run that stopped and a run that finished both stop writing,
  so a stale mtime on its own means nothing. The order matters: `summary.json`
  present is finished; no rank processes is stopped; only a live run with a
  stale mtime is stalled.
* **Poisoning.** A failed evaluation scores `FAILURE_OBJECTIVE` (100.0), which
  PolyChord maximizes, while a real `total_rms_jy` is ~0.008. A run whose
  imager is broken - a missing checkpoint mount, an OOM-killed worker - is
  therefore a run happily concentrating its live points on its own failures. It
  looks perfectly healthy by every other measure and is worth nothing.
* **Cost.** Gaps between evaluations, and `meqserver-wedged.log`, put a number
  on the MeqTrees deadlock the watchdogs in `simulate_point_source_ms.py`
  absorb (docs/nested-sampling.md, "When MeqTrees stops answering").

Plus the host: memory, and sidecar containers whose run is gone. A killed run
leaves its `ri-ns-sidecar-*` containers holding ~3.4GB per R2D2 rank, which
then counts against every later run's memory budget until someone notices.

Filesystem reads, one `ps` and one `docker ps`, plus a one second CPU sample
when a run has live ranks; nothing started, nothing imaged, so it costs a busy
host nothing to ask.

Usage:

  uv run scripts/nested-sampling-health.py            # the newest run
  uv run scripts/nested-sampling-health.py <run>
  uv run scripts/nested-sampling-health.py --all
  uv run scripts/nested-sampling-health.py --json

Exit status is 0 when nothing needs attention and 1 when something does, so it
can gate a script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

NESTED_SAMPLING_DIR = Path("results/nested-sampling")

# A run's own MPI ranks, as `ps` shows them.
RANK_COMMAND = re.compile(r"\S*python[\d.]*\s+\S*polychord_\w+\.py\b")

# What counts as the line worth quoting out of run.log. Deliberately broad: the
# useful line is whichever one names the failure, and this only has to beat
# "whatever happened to be printed last".
ERROR_LINE = re.compile(r"Traceback|Error|Exception|FATAL", re.IGNORECASE)

# common.py's FAILURE_OBJECTIVE, as write_evaluation_record's json.dumps writes
# it. Matched as text because the alternative is parsing thousands of files to
# read one number out of each.
FAILURE_OBJECTIVE_MARKER = '"objective": 100.0'

# Long enough that no legitimate evaluation reaches it: the slowest measured
# here is ~33s of R2D2 imaging, and common.py's own ceilings (10s for a
# simulate reply, 3600s for an imaging one) bound anything the run itself would
# wait on. Overridable because a search over a slower parameter space moves it.
DEFAULT_STALE_SECONDS = 600.0

# How long nothing may complete before every rank burning CPU is read as a
# deadlock rather than as ranks waiting on whichever peer is still imaging.
# Much shorter than the stale threshold above because the two signals together
# are unambiguous where either alone is not: a healthy run at any pace lands
# something well inside a minute.
SPIN_IDLE_SECONDS = 60.0

# The "how is it going now" window - a tenth of the run, never fewer than this
# many evaluations - and how far it has to diverge from the run's overall pace
# before both numbers are worth printing rather than one.
#
# Doubled rather than half again, because a healthy run's own pace is noisier
# than it looks: five minute bins over 107 minutes of one ranged 91-165 with no
# fault present. Walking that run's history at 269 sampled moments, the display
# fires in 15% of them at 1.5x and 5% at 2.0x, and the knee is where the
# ordinary swing stops and the real events start.
RATE_WINDOW = 50
RATE_WINDOW_DIVISOR = 10
RATE_DIVERGENCE_FACTOR = 2.0

# A gap between consecutive evaluations counts as a stall when it is this many
# times the run's own median gap. Relative because the two imagers are two
# orders of magnitude apart - WSClean lands 30-50 evaluations a second, R2D2
# roughly one every two seconds - so any fixed number is either blind on one or
# crying wolf on the other. Floored at MIN_STALL_GAP_SECONDS so a fast run's
# ordinary jitter does not register either.
STALL_GAP_FACTOR = 10.0
MIN_STALL_GAP_SECONDS = 2.0

# rank-budget.sh's NS_RANK_BUDGET_HEADROOM_MB. Reported, not enforced: it is
# the line under which the next run will refuse to size itself.
HEADROOM_MB = 4096


# --- host state: one `ps`, one `docker ps` ----------------------------------


def _clock_seconds(field: str) -> float | None:
    """`[[dd-]hh:]mm:ss` - what both BSD and GNU `ps` print for etime and time.

    `etimes` would give seconds directly but is GNU-only, and this repo's
    checks run on macOS too.
    """
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", field.strip())
    if not match:
        return None
    days, hours, minutes, seconds = match.groups()
    return (
        float(days or 0) * 86400
        + float(hours or 0) * 3600
        + float(minutes) * 60
        + float(seconds)
    )


def process_table() -> list[dict[str, object]]:
    """Every process, as (pid, state, elapsed, cpu, args).

    One call, because everything here needs it: the ranks of a run, whether a
    sidecar's launcher is still alive, and how much of its life a rank has
    spent on CPU.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,state=,etime=,time=,args="],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, object]] = []
    for line in out.splitlines():
        fields = line.split(None, 4)
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        pid, state, etime, cpu, args = fields
        rows.append({
            "pid": int(pid),
            # A process killed while its parent is not wait()ing stays as a
            # zombie, and `kill -0` succeeds on one - so state, not existence,
            # is what says a process is alive.
            "alive": state[:1] != "Z",
            "elapsed_seconds": _clock_seconds(etime),
            "cpu_seconds": _clock_seconds(cpu),
            "args": args,
        })
    return rows


def sidecar_containers() -> list[dict[str, object]] | None:
    """The run's long-lived containers, or None if docker cannot be asked.

    Names are `ri-ns-sidecar-<launcher pid>-<n>` from start-sidecars.sh and
    `ri-ns-sidecar-<rank pid>-<uuid8>` from common.py's own fallback path; both
    carry the pid of whatever started them in the same position, which is what
    makes a leak detectable.
    """
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=ri-ns-sidecar", "--format", "{{.Names}}\t{{.Image}}"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    containers = []
    for line in out.splitlines():
        name, _, image = line.partition("\t")
        if not name:
            continue
        parts = name.split("-")
        owner = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
        containers.append({"name": name, "image": image, "owner_pid": owner})
    return containers


def available_mb() -> int | None:
    """Free memory as rank-budget.sh's guard reads it, or None off Linux.

    scripts/lib/rank-budget.sh is the authority on what a run may take; this
    only reports the number its decision will be made from.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


# --- one run ----------------------------------------------------------------


def read_run_env(run_dir: Path) -> dict[str, str]:
    """The settings the run recorded for itself; {} if it recorded none."""
    values: dict[str, str] = {}
    try:
        text = (run_dir / "run.env").read_text()
    except OSError:
        return values
    for line in text.splitlines():
        name, sep, raw = line.partition("=")
        if sep:
            values[name.strip()] = raw.strip().strip("'").replace("'\\''", "'")
    return values


def stage(run_dir: Path) -> str:
    """How far into PolyChord the run got.

    Worth its own line because a run that dies before the main loop dies
    without a single dead point, and looks from every count like a run that
    simply has not got going yet.
    """
    if (run_dir / "summary.json").exists():
        return "finished"
    chains = run_dir / "chains"
    if any(chains.glob("*.resume")):
        return "sampling"
    if any(chains.glob("*_phys_live.txt")):
        return "generating initial live points"
    return "starting up"


def evaluation_scan(run_dir: Path) -> dict[str, object]:
    """Counts, timings and failures, in one pass over evaluations/.

    The stat and the read are the same pass because the interesting things -
    when an evaluation landed, and whether it landed on a failure - are one per
    file and there is no cheaper place to get either.
    """
    evaluations = run_dir / "evaluations"
    directories = 0
    times: list[float] = []
    failed = 0
    wedged_lines = 0
    for entry in evaluations.glob("eval-*"):
        if not entry.is_dir():
            continue
        directories += 1
        metrics = entry / "metrics.json"
        try:
            times.append(metrics.stat().st_mtime)
            if FAILURE_OBJECTIVE_MARKER in metrics.read_text():
                failed += 1
        except OSError:
            pass  # in flight, or a leftover the next run will sweep
        try:
            wedged_lines += len((entry / "meqserver-wedged.log").read_text().splitlines())
        except OSError:
            pass
    times.sort()
    gaps = [b - a for a, b in zip(times, times[1:])]
    threshold = MIN_STALL_GAP_SECONDS
    if gaps:
        threshold = max(threshold, STALL_GAP_FACTOR * statistics.median(gaps))
    stalls = [g for g in gaps if g > threshold]
    span = times[-1] - times[0] if len(times) > 1 else 0.0

    # How the run is going now against how it has gone, as a ratio of median
    # gaps. Medians, so one long gap cannot manufacture a collapse and a run
    # that is genuinely half its old speed cannot hide behind a fast tail.
    #
    # Gaps rather than "evaluations in the last N minutes", which is the
    # obvious simplification and is wrong: the most recent window is always
    # partial, so it reads low by exactly the fraction of it that has not
    # elapsed yet, and at the moment of sampling that is indistinguishable from
    # a real slowdown. Measured on a live run mid-window - 23.8/min from the
    # partial bucket against a 91-165 per five minutes baseline, which looks
    # like a collapse starting, where the gaps said 52.5/min and the bucket
    # went on to finish at 164, the highest of the run. An inter-arrival gap
    # cannot be measured until both ends exist, so there is no partial window
    # to misread.
    #
    # The one thing gaps cannot see is a stall that began after the last
    # completed evaluation, because that open-ended interval has no far end
    # yet. last_activity_seconds is exactly that interval, and the idle clauses
    # in describe() are what cover the hole. The two look redundant and are
    # complementary; do not drop either.
    # Both rates the same way: evaluations divided by the time they took. The
    # recent one used to be 60 / median(recent gaps), which is a different
    # quantity from the overall count-over-span it sits next to - a median
    # ignores the long tail, so during a phased run it reads roughly double.
    # Measured at one instant on a live search: 33.3/min from the median
    # against 18.2/min from the count, where an independent thirty-minute
    # window said 17.0. Printing those two side by side invited exactly the
    # comparison they could not support.
    #
    # A share of the run rather than a fixed count, because a fixed one covers
    # a wildly different span depending on pace - fifty evaluations is two
    # minutes at 25/min and ten at 5/min, so the window grows precisely when
    # the run slows and the reading whipsaws. The last fifty here swung
    # 4.9 -> 31.5 -> 37.6 over forty minutes; a tenth of the run gave
    # 28.1 -> 37.6 -> 33.4 across the same samples. Both ends of the window are
    # completed evaluations, so this stays immune to the partial-window problem
    # above.
    recent_rate, slowdown = None, None
    window = max(RATE_WINDOW, len(times) // RATE_WINDOW_DIVISOR)
    if len(times) >= 2 * window:
        recent_times = times[-window:]
        recent_span = recent_times[-1] - recent_times[0]
        overall_rate = len(times) * 60 / span if span > 0 else None
        if recent_span > 0 and overall_rate:
            recent_rate = len(recent_times) * 60 / recent_span
            slowdown = overall_rate / recent_rate if recent_rate > 0 else None
    else:
        recent_span = None

    return {
        "completed": len(times),
        "in_flight": directories - len(times),
        "failed": failed,
        "last_activity_seconds": time.time() - times[-1] if times else None,
        "span_seconds": span,
        "evals_per_minute": len(times) * 60 / span if span > 0 else None,
        "recent_evals_per_minute": recent_rate,
        "recent_span_seconds": recent_span,
        "slowdown_factor": slowdown,
        "stall_threshold_seconds": threshold,
        "stall_count": len(stalls),
        "stall_seconds": sum(stalls),
        "stall_fraction": sum(stalls) / span if span > 0 else 0.0,
        "meqserver_wedges": wedged_lines,
    }


def log_tail(run_dir: Path) -> dict[str, object] | None:
    """What the run last said, and how many ranks said it.

    `run.log` is the only place a traceback survives - every other artifact a
    stopped run leaves says that it broke, never why.

    The count is the diagnosis, not decoration. An MPI crash produces one
    traceback per rank, so a real failure here is the same stack fifteen or
    twenty times over and the plain last line of the file is the right answer
    only by luck of where the output stopped. All ranks reporting the same
    error is a code bug every rank hits deterministically; one rank alone is a
    flaky worker, an OOM kill, or bad luck on one evaluation. Those want
    opposite responses, and the multiplicity is the only thing in the file that
    tells them apart.

    Falls back to the last non-empty line, which for a run that stopped without
    a traceback is PolyChord's own last word on where it got to.
    """
    path = run_dir / "run.log"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            # Enough to hold every rank's copy: the one real crash captured
            # here was 15 tracebacks in 18KB.
            handle.seek(max(0, handle.tell() - 65536))
            lines = [line.strip() for line in
                     handle.read().decode("utf-8", "replace").splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    errors = [line for line in lines if ERROR_LINE.search(line)]
    line = errors[-1] if errors else lines[-1]
    return {"line": line, "occurrences": lines.count(line), "is_error": bool(errors)}


def dead_points(run_dir: Path) -> tuple[int, float | None]:
    """PolyChord's progress and how old that number is.

    The age is not decoration. PolyChord writes its checkpoint roughly every
    `nlive` dead points, so between writes the count cannot move by
    construction - and a count that has not moved for fifty minutes looks
    exactly like a run that has stopped making progress. That misreading has
    already cost an hour of investigation here, and it survived being checked
    against the terminal, because PolyChord's own feedback box and these files
    are written by the same event: two displays of one signal, not two
    witnesses. Nothing here decides anything from this count; it is reported
    with its age so that nobody else does either.
    """
    for path in (run_dir / "chains").glob("*_dead-birth.txt"):
        try:
            return sum(1 for _ in path.open()), time.time() - path.stat().st_mtime
        except OSError:
            return 0, None
    return 0, None


def rank_processes(run_dir: Path, processes: list[dict[str, object]]) -> list[dict[str, object]]:
    """This run's MPI ranks, found by the --output-dir they were launched with.

    Which is also how the run is known to be alive at all: no ranks, no run.

    Anchored at the interpreter because `mpirun` and the host-side `docker
    exec` both carry the whole rank command line in their own arguments, and
    counting those as ranks puts the count two over `NS_MPI_PROCS`.
    """
    marker = str(run_dir.resolve())
    return [
        p for p in processes
        if marker in str(p["args"]) and p["alive"] and RANK_COMMAND.match(str(p["args"]))
    ]


def _cpu_seconds(pid: int) -> float | None:
    """CPU time for one process, in ticks rather than whole seconds.

    `ps` rounds its `time` column to the second, which is too coarse to
    difference over a short interval; /proc counts in clock ticks, so a one
    second sample resolves to a percent. Linux only, and cpu_busy_fractions
    falls back to `ps` and a longer interval where it is not there.
    """
    try:
        # The comm field can itself contain spaces and brackets, so everything
        # before the last ") " is skipped rather than parsed. What is left
        # starts at `state`, which makes utime and stime fields 11 and 12.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _cpu_seconds_from_ps(pids: list[int]) -> dict[int, float]:
    """The same, from `ps`, for hosts without /proc."""
    if not pids:
        return {}
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,time=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    sampled = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit():
            seconds = _clock_seconds(fields[1])
            if seconds is not None:
                sampled[int(fields[0])] = seconds
    return sampled


def cpu_busy_fractions(pids: list[int]) -> dict[int, float]:
    """What share of a short interval each process spent on CPU, right now.

    Open MPI's ob1 busy-waits, so a rank blocked in a collective burns a core
    and is indistinguishable from a working one on instantaneous CPU percent.
    What separates them is how much of the interval went to CPU: a working rank
    spends most of it waiting on its imager, a spinning one spends all of it.

    Sampled twice rather than taken as cumulative CPU time over the process's
    whole life. The lifetime ratio only reveals a rank that wedged early - a
    run that works for an hour at 40% and then deadlocks still reads ~0.45 five
    minutes later, and goes on reading under the threshold for as long as it
    takes the spinning to outweigh the real work already done. Which is most of
    an hour, on the runs this is for.
    """
    if not pids:
        return {}
    fine = {pid: _cpu_seconds(pid) for pid in pids}
    if all(v is not None for v in fine.values()):
        # /proc resolves to a percent, so a second is a long enough sample.
        before, interval = fine, 1.0
    else:
        # `ps` counts whole seconds, so the interval has to be long enough that
        # the rounding does not swamp the difference.
        before, interval = _cpu_seconds_from_ps(pids), 5.0
    time.sleep(interval)
    after = ({pid: _cpu_seconds(pid) for pid in pids} if interval == 1.0
             else _cpu_seconds_from_ps(pids))
    fractions = {}
    for pid in pids:
        start, end = before.get(pid), after.get(pid)
        if start is not None and end is not None:
            fractions[pid] = (end - start) / interval
    return fractions


def spinning_ranks(ranks: list[dict[str, object]], busy: dict[int, float]) -> int:
    """Ranks that spent the whole sample on CPU.

    Rank 0 is PolyChord's administrator and legitimately spins, so one is
    normal and all of them is the signal.
    """
    return sum(1 for rank in ranks if busy.get(int(rank["pid"]), 0.0) > 0.95)


def describe(run_dir: Path, processes: list[dict[str, object]],
             stale_seconds: float, busy: dict[int, float] | None = None) -> dict[str, object]:
    run_env = read_run_env(run_dir)
    scan = evaluation_scan(run_dir)
    ranks = rank_processes(run_dir, processes)
    spinning = spinning_ranks(ranks, busy or {})
    dead, checkpoint_age = dead_points(run_dir)
    tail = log_tail(run_dir)
    idle = scan["last_activity_seconds"]
    complete = (run_dir / "summary.json").exists()

    # The order is the whole point. A finished run and a dead one both stop
    # writing, and a run that has only just started has not written yet, so
    # neither a stale mtime nor a missing summary means anything on its own.
    if complete:
        status = "finished"
    elif ranks:
        status = "stalled" if idle is not None and idle > stale_seconds else "healthy"
    elif idle is not None and idle <= stale_seconds:
        status = "starting"
    else:
        status = "stopped"

    warnings: list[str] = []
    completed = int(scan["completed"])
    failed = int(scan["failed"])
    if completed and failed == completed:
        warnings.append(
            "every evaluation scored FAILURE_OBJECTIVE - the search is exploring a "
            "broken imager, not the parameter space"
        )
    elif failed and failed * 2 >= completed:
        warnings.append(
            f"{failed} of {completed} evaluations scored FAILURE_OBJECTIVE, which "
            "PolyChord maximizes - the run is chasing its own failures"
        )
    if status == "stalled":
        warnings.append(
            f"no evaluation has landed in {idle:.0f}s while {len(ranks)} ranks are still running"
        )
    if status == "stopped":
        # What the log said, because "why" is the question a stopped run raises
        # and this is the only artifact that answers it. Runs from before
        # run.log was captured have none, hence the plain form.
        ending = ""
        if tail:
            copies = int(tail["occurrences"])
            ending = (f'; run.log ends "{tail["line"]}"'
                      + (f" (x{copies} ranks)" if copies > 1 else ""))
        warnings.append(
            f"stopped before finishing{ending}; continue it with ./ri resume {run_dir.name}"
        )
    # All but one, because rank 0 is PolyChord's administrator and does nothing
    # else - and only once evaluations have stopped landing, which is the
    # clause doing the real work here. The spin count on its own says nothing:
    # four independent measurements of one healthy 16-rank R2D2 run, across an
    # hour, gave 1, 2, 7 and 15, as the sampler alternated between imaging in
    # parallel and synchronising. Each was reproducible for as long as its
    # phase lasted - a count sampled for a minute lands there just as
    # confidently as one sampled once - so all-but-one spinning is a coin flip
    # on timing rather than a fault. What no phase of a working run produces is
    # a minute with nothing completed.
    if ranks and spinning >= max(1, len(ranks) - 1) and idle is not None \
            and idle > SPIN_IDLE_SECONDS:
        warnings.append(
            f"{spinning} of {len(ranks)} ranks are burning CPU and nothing has "
            f"completed in {idle:.0f}s - the signature of every rank spinning in "
            "one MPI collective"
        )
    # Throughput is measured and shown but deliberately not warned on. A run
    # can collapse to a fraction of its own rate without ever going quiet long
    # enough to look stalled - a live 16-rank R2D2 search fell from ~25/min to
    # ~5/min for ten minutes, passing every check here - so silence would hide
    # it. But the same run then recovered to ~37/min with nothing done to it:
    # five minute bins of 104, 23, 26, 93 against a 104-165 baseline. One
    # observed dip, one observed recovery, and no established trigger, is not
    # enough to tell a human that something needs doing. Showing the number and
    # letting them judge is what the evidence supports; warning on a phase that
    # heals itself would teach them to ignore the warnings that do not.
    #
    # A tenth of the run, not a twentieth: a few percent is the ordinary spread
    # of evaluation cost across the parameter space, and the deadlock this
    # number exists to catch cost 23-27% before the watchdogs absorbed it.
    if float(scan["stall_fraction"]) > 0.10:
        warnings.append(
            f"{scan['stall_fraction']:.0%} of wall clock lost to gaps over "
            f"{scan['stall_threshold_seconds']:.0f}s"
        )

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "algorithm": run_env.get("NS_ALGORITHM") or run_dir.name.split("-", 1)[0],
        "status": status,
        "stage": stage(run_dir),
        "settings": run_env,
        "ranks": len(ranks),
        "ranks_spinning": spinning,
        "dead_points": dead,
        "checkpoint_age_seconds": checkpoint_age,
        "log_tail": tail,
        "warnings": warnings,
        **scan,
    }


def host_report(processes: list[dict[str, object]]) -> dict[str, object]:
    alive = {int(p["pid"]) for p in processes if p["alive"]}
    containers = sidecar_containers()
    leaked = []
    if containers is not None:
        leaked = [c for c in containers
                  if c["owner_pid"] is not None and c["owner_pid"] not in alive]
    warnings = []
    if leaked:
        warnings.append(
            f"{len(leaked)} sidecar container(s) outlived the run that started them, "
            "holding ~3.4GB per R2D2 rank against every later run's memory budget: "
            "docker rm -f " + " ".join(str(c["name"]) for c in leaked)
        )
    memory = available_mb()
    if memory is not None and memory < HEADROOM_MB:
        warnings.append(
            f"{memory}MB available is below the {HEADROOM_MB}MB headroom rank-budget.sh "
            "keeps free; a new run will refuse to size itself"
        )
    return {
        "available_mb": memory,
        "sidecars": containers,
        "leaked_sidecars": leaked,
        "warnings": warnings,
    }


# --- rendering ---------------------------------------------------------------


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render(run: dict[str, object]) -> None:
    print(f"{run['name']}  {run['algorithm']}  {str(run['status']).upper()}")
    settings = run["settings"]
    assert isinstance(settings, dict)
    ranks = f"{run['ranks']} ranks"
    if settings.get("NS_MPI_PROCS"):
        ranks += f" of {settings['NS_MPI_PROCS']}"
    if run["ranks_spinning"]:
        ranks += f", {run['ranks_spinning']} busy-waiting"

    idle = run["last_activity_seconds"]
    rate = run["evals_per_minute"]
    slowdown = run["slowdown_factor"]
    # Where the count will next move to, so a reader who comes back has
    # something to compare against rather than a number that looks stuck.
    #
    # "past", because nlive is the cadence PolyChord checkpoints on and not the
    # size of the jump: three consecutive writes on one nlive=50 run moved the
    # count by 57, 56 and 60. Stated as a floor rather than fitted to those
    # three, which would be the same over-reading of a short series that this
    # whole line exists to prevent.
    next_update = ""
    if settings.get("NS_NLIVE", "").isdigit() and run["checkpoint_age_seconds"] is not None:
        next_update = f", next past ~{int(run['dead_points']) + int(settings['NS_NLIVE'])}"
    lines = [
        # The dead-point count never appears without how old it is. PolyChord
        # writes it every ~nlive points, so it is stale by design between
        # writes and a frozen-looking count means nothing on its own.
        ("stage", f"{run['stage']}, {run['dead_points']} dead points"
                  + (f" as of {format_hms(float(run['checkpoint_age_seconds']))} ago"
                     f"{next_update}"
                     if run["checkpoint_age_seconds"] is not None else "")),
        ("progress", f"{run['completed']} evaluations, {run['in_flight']} in flight"),
        ("activity", "nothing yet" if idle is None else
                     f"last evaluation {format_hms(float(idle))} ago"
                     + (f", {rate:.1f}/min over {format_hms(float(run['span_seconds']))}"
                        if rate else "")
                     # Only when the run has changed pace materially, in either
                     # direction: the two numbers agreeing says nothing.
                     # In its own units of time, not "the last N evaluations":
                     # the window is a share of the run, so how long it covers
                     # is the thing the reader needs and the count is not.
                     + (f" ({run['recent_evals_per_minute']:.1f}/min over the last "
                        f"{format_hms(float(run['recent_span_seconds']))})"
                        if slowdown is not None and (
                            float(slowdown) > RATE_DIVERGENCE_FACTOR
                            or float(slowdown) < 1 / RATE_DIVERGENCE_FACTOR) else "")),
        ("ranks", ranks),
        ("failures", f"{run['failed']} scored FAILURE_OBJECTIVE, "
                     f"{run['meqserver_wedges']} meqserver wedges recovered"),
        ("stalls", f"{run['stall_count']} gaps over {run['stall_threshold_seconds']:.0f}s, "
                   f"{run['stall_seconds']:.0f}s = {run['stall_fraction']:.1%} of wall clock"),
    ]
    for label, value in lines:
        print(f"  {label:<9} {value}")
    for warning in run["warnings"]:
        print(f"  WARNING   {warning}")


def render_host(host: dict[str, object]) -> None:
    print("host")
    memory = host["available_mb"]
    print(f"  {'memory':<9} " + ("unknown (not Linux)" if memory is None
                                 else f"{int(memory) / 1024:.1f}GB available, "
                                      f"{HEADROOM_MB / 1024:.0f}GB reserved as headroom"))
    sidecars = host["sidecars"]
    if sidecars is None:
        print(f"  {'sidecars':<9} unknown (docker did not answer)")
    else:
        print(f"  {'sidecars':<9} {len(sidecars)} running, {len(host['leaked_sidecars'])} leaked")
    for warning in host["warnings"]:
        print(f"  WARNING   {warning}")


# --- entry point -------------------------------------------------------------


def started_at(run_dir: Path) -> str:
    """The UTC stamp every run directory is named for, for ordering by age.

    Sorting the names themselves puts every `wsclean-*` after every `r2d2-*`,
    so "the newest run" would mean "the newest WSClean run" - which on a host
    running both is the wrong run to be shown by default.
    """
    match = re.search(r"(\d{8}T\d{6}Z)$", run_dir.name)
    return match.group(1) if match else run_dir.name


def run_directories() -> list[Path]:
    """Every run on disk, newest first."""
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    return sorted((d for d in NESTED_SAMPLING_DIR.iterdir() if d.is_dir()),
                  key=started_at, reverse=True)


def resolve(name: str | None) -> Path:
    """A run directory from a path, a bare run name, or nothing at all."""
    if name:
        candidate = Path(name)
        if not candidate.is_dir():
            candidate = NESTED_SAMPLING_DIR / name
        if not candidate.is_dir():
            raise SystemExit(f"No such run: {name}")
        return candidate
    runs = run_directories()
    if not runs:
        raise SystemExit(f"No runs under {NESTED_SAMPLING_DIR}/.")
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", metavar="RUN",
                        help="run directory or its name (default: the newest run)")
    parser.add_argument("--all", action="store_true", help="every run on disk")
    parser.add_argument("--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS,
                        help="a live run silent for this long is stalled "
                             "(default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="raw JSON instead of a report")
    args = parser.parse_args(argv)

    if args.all and args.run:
        parser.error("--all takes no run argument")

    processes = process_table()
    directories = run_directories() if args.all else [resolve(args.run)]
    # One sample interval for every run being reported, not one each: only a
    # run with live ranks is sampled at all, and there is rarely more than one.
    # A report over finished runs costs nothing.
    busy = cpu_busy_fractions(
        [int(p["pid"]) for d in directories for p in rank_processes(d, processes)])
    runs = [describe(d, processes, args.stale_seconds, busy) for d in directories]
    host = host_report(processes)

    if args.json:
        print(json.dumps({"runs": runs, "host": host}, indent=2))
    else:
        for run in runs:
            render(run)
            print()
        render_host(host)

    unhealthy = any(r["warnings"] for r in runs) or bool(host["warnings"])
    return 1 if unhealthy else 0


def self_check() -> None:
    import contextlib
    import io
    import tempfile

    def io_capture(report: dict[str, object]) -> str:
        """What render() prints for a run, so the report lines can be asserted on."""
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            render(report)
        return sink.getvalue()

    global NESTED_SAMPLING_DIR
    saved = NESTED_SAMPLING_DIR
    now = time.time()

    def write_eval(run: Path, index: int, mtime: float, objective: float = 0.008,
                   wedges: int = 0) -> None:
        eval_dir = run / "evaluations" / f"eval-{index:04d}-abc"
        eval_dir.mkdir(parents=True)
        metrics = eval_dir / "metrics.json"
        metrics.write_text(json.dumps({"eval_id": index, "objective": objective}, indent=2))
        os.utime(metrics, (mtime, mtime))
        if wedges:
            (eval_dir / "meqserver-wedged.log").write_text(
                "".join(f"attempt {i + 1}: no reply to the predict in 3.0s\n"
                        for i in range(wedges)))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            NESTED_SAMPLING_DIR = Path(tmp)

            # `[[dd-]hh:]mm:ss` in every form ps prints it.
            assert _clock_seconds("00:12") == 12
            assert _clock_seconds("01:02:03") == 3723
            assert _clock_seconds("2-01:00:00") == 176400
            assert _clock_seconds("12:34.56") == 754.56
            assert _clock_seconds("wat") is None

            # A live run: ranks running, evaluations landing.
            live = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T000000Z"
            (live / "chains").mkdir(parents=True)
            (live / "chains" / "r2d2_vlaa.resume").write_text("")
            (live / "chains" / "r2d2_vlaa_dead-birth.txt").write_text("a\nb\nc\n")
            (live / "run.env").write_text("NS_ALGORITHM=r2d2\nNS_MPI_PROCS=4\nNS_NLIVE=4\n")
            for i in range(4):
                write_eval(live, i + 1, now - 40 + i * 0.5)
            (live / "evaluations" / "eval-0005-inflight").mkdir()
            ranks = [{"pid": 100 + i, "alive": True, "elapsed_seconds": 100.0,
                      "cpu_seconds": 20.0,
                      "args": f"python3 polychord_r2d2.py --output-dir {live.resolve()}"}
                     for i in range(4)]

            # mpirun and the host-side `docker exec` both carry the whole rank
            # command line in their own arguments; neither is a rank.
            not_ranks = [
                {"pid": 90, "alive": True, "elapsed_seconds": 100.0, "cpu_seconds": 1.0,
                 "args": f"mpirun -np 4 python3 polychord_r2d2.py --output-dir {live.resolve()}"},
                {"pid": 91, "alive": True, "elapsed_seconds": 100.0, "cpu_seconds": 1.0,
                 "args": f"/usr/bin/docker exec c mpirun python3 polychord_r2d2.py "
                         f"--output-dir {live.resolve()}"},
            ]
            # PolyChord writes the dead-point count every ~nlive points, so it
            # is stale by design between writes. Aged here at ten minutes while
            # evaluations land every second: a run in exactly the state that
            # has already been misread as "progress has stopped", so the count
            # must never be shown without how old it is.
            os.utime(live / "chains" / "r2d2_vlaa_dead-birth.txt", (now - 600, now - 600))
            report = describe(live, not_ranks + ranks, DEFAULT_STALE_SECONDS)
            assert report["status"] == "healthy", report
            assert 590 < float(report["checkpoint_age_seconds"]) < 620, report
            aged = io_capture(report)
            assert "3 dead points as of 0:10:0" in aged, aged
            # ...and where it will next move past, so a reader coming back
            # has something to compare against. nlive is 4 in this fixture, and
            # the bound is a floor: PolyChord checkpoints every nlive points
            # but the count overshoots, by 57, 56 and 60 on one nlive=50 run.
            assert "next past ~7" in aged, aged
            assert report["completed"] == 4 and report["in_flight"] == 1, report
            assert report["dead_points"] == 3, report
            assert report["ranks"] == 4 and report["failed"] == 0, report
            assert report["warnings"] == [], report

            # A stopped run says why, when the run script captured a log for it
            # - which is the whole reason run.log exists.
            assert log_tail(live) is None
            # The shape a real MPI crash leaves: every rank raises the same
            # thing, so the last line is the right answer only by luck of where
            # the output happened to stop, and the count is the diagnosis.
            crash = "TypeError: _connect_shell_started_worker() missing 1 required argument"
            (live / "run.log").write_text(
                "ndead: 40\n"
                # Past the read window, so the tail is genuinely sought to.
                + "chatter\n" * 9000
                + "".join(f"Traceback (most recent call last):\n  frame\n{crash}\n"
                          for _ in range(15))
                + "mpirun detected that one process exited\n"
            )
            found = log_tail(live)
            assert found["line"] == crash, found
            assert found["occurrences"] == 15, found
            stopped_report = describe(live, [], 5.0)
            assert stopped_report["status"] == "stopped"
            assert any("TypeError" in w and "(x15 ranks)" in w and "./ri resume" in w
                       for w in stopped_report["warnings"]), stopped_report

            # One rank alone is a different fault and must not be reported as
            # if every rank hit it.
            (live / "run.log").write_text(f"chatter\n{crash}\nmpirun: exiting\n")
            assert log_tail(live)["occurrences"] == 1
            assert not any("ranks)" in w for w in describe(live, [], 5.0)["warnings"])

            # A run that stopped without a traceback still reports where it got
            # to, rather than nothing.
            (live / "run.log").write_text("ndead: 113  logZ = -1.2\n\n")
            quiet_exit = log_tail(live)
            assert quiet_exit["line"] == "ndead: 113  logZ = -1.2", quiet_exit
            assert quiet_exit["is_error"] is False, quiet_exit

            # A run from before run.log existed still gets the resume line.
            (live / "run.log").unlink()
            assert any("./ri resume" in w for w in describe(live, [], 5.0)["warnings"])

            # Same run, same files, no ranks: a stale mtime is only a stall
            # while something is still running.
            assert describe(live, [], 5.0)["status"] == "stopped"
            assert describe(live, ranks, 5.0)["status"] == "stalled"
            # ...and once it finishes, neither is true however old it gets.
            (live / "summary.json").write_text("{}")
            assert describe(live, [], 5.0)["status"] == "finished"
            assert describe(live, [], 5.0)["warnings"] == []
            (live / "summary.json").unlink()

            # Ranks that spend a whole sample on CPU are the deadlock
            # signature; ranks that wait on their imager are the healthy one,
            # at the same rank count and with the same lifetimes.
            working = {int(r["pid"]): 0.4 for r in ranks}
            spinning = {int(r["pid"]): 0.999 for r in ranks}
            assert spinning_ranks(ranks, spinning) == 4
            assert spinning_ranks(ranks, working) == 0
            # Unsampled is not spinning: a rank that exited between the two
            # samples must not read as wedged.
            assert spinning_ranks(ranks, {}) == 0
            # ...but busy-waiting alone is not a fault. Every rank spinning is
            # only a deadlock once nothing is completing either: on a healthy
            # run most ranks sit at 1.0 waiting for whichever peer is still
            # imaging. These evaluations landed seconds ago, so no warning.
            assert describe(live, ranks, DEFAULT_STALE_SECONDS, spinning)["warnings"] == []
            assert describe(live, ranks, DEFAULT_STALE_SECONDS, working)["warnings"] == []
            # The same ranks against a run that has gone quiet: a deadlock, and
            # said so well before the much longer stale threshold is reached.
            quiet = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T010000Z"
            (quiet / "chains").mkdir(parents=True)
            for i in range(4):
                write_eval(quiet, i + 1, now - 300 + i * 0.5)
            quiet_ranks = [dict(r, args=str(r["args"]).replace(str(live.resolve()),
                                                               str(quiet.resolve())))
                           for r in ranks]
            report = describe(quiet, quiet_ranks, DEFAULT_STALE_SECONDS, spinning)
            assert report["status"] == "healthy", report  # not yet stale...
            assert any("burning CPU" in w for w in report["warnings"]), report
            # One rank still working is PolyChord's usual shape, not a deadlock.
            all_but_one = {**working, int(ranks[0]["pid"]): 0.999}
            assert describe(quiet, quiet_ranks, DEFAULT_STALE_SECONDS,
                            all_but_one)["warnings"] == []

            # The sampler itself, against a process that really is spinning and
            # one that really is not - both in the same interval, because a
            # discriminator that always returned zero would pass a check made
            # only of idle processes. The bar is 0.5 rather than 0.95 so a
            # loaded host cannot fail it; the two are an order apart.
            assert cpu_busy_fractions([]) == {}
            spinner = subprocess.Popen([sys.executable, "-c", "while True: pass"])
            try:
                sampled = cpu_busy_fractions([spinner.pid, os.getpid()])
            finally:
                spinner.kill()
                spinner.wait()
            assert sampled.get(spinner.pid, 0) > 0.5, sampled
            assert sampled.get(os.getpid(), 1) < 0.5, sampled

            # A run that has gone serial: still landing evaluations, so never
            # idle long enough to look stalled, but at a fraction of its own
            # throughput. Every other check here passes it.
            collapsed = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T020000Z"
            (collapsed / "chains").mkdir(parents=True)
            (collapsed / "chains" / "r2d2_vlaa.resume").write_text("")
            # 200s of healthy phase plus 720s of collapse, landed so that the
            # last evaluation is a few seconds old: still well inside every
            # timeout the run itself has, which is the point.
            stamp = now - (200 + 60 * 12) - 5
            for i in range(200):          # the healthy phase, one a second
                stamp += 1
                write_eval(collapsed, i + 1, stamp)
            for i in range(60):           # ...and the collapse, one per 12s
                stamp += 12
                write_eval(collapsed, 201 + i, stamp)
            collapsed_ranks = [dict(r, args=str(r["args"]).replace(str(live.resolve()),
                                                                   str(collapsed.resolve())))
                               for r in ranks]
            report = describe(collapsed, collapsed_ranks, DEFAULT_STALE_SECONDS, spinning)
            # Not stalled: something landed seconds ago, well inside every
            # timeout - which is exactly why the rate had to be measured.
            assert report["status"] == "healthy", report
            assert float(report["last_activity_seconds"]) < SPIN_IDLE_SECONDS, report
            # Both rates are evaluations over the time they took, so they can
            # be divided: 60 a minute over the fast phase, 5 over the collapse,
            # and 17 overall because the fast phase is most of the run. A
            # median-of-gaps recent rate would read 5 against an overall of 60
            # and call this a twelvefold collapse.
            assert abs(float(report["recent_evals_per_minute"]) - 5.1) < 0.2, report
            assert abs(float(report["evals_per_minute"]) - 17.0) < 0.2, report
            assert abs(float(report["slowdown_factor"]) - 3.3) < 0.2, report
            # The window reports the time it covers, not a count: 50 twelve
            # second evaluations is very nearly ten minutes.
            assert abs(float(report["recent_span_seconds"]) - 588) < 2, report
            # Measured and shown, never warned on: a run that halves its pace
            # and recovers is a phase, and warning on it would teach the reader
            # to ignore the warnings that mean something.
            assert not any("throughput" in w for w in report["warnings"]), report
            rendered = io_capture(report)
            assert "5.1/min over the last 0:09:48" in rendered, rendered
            # A run holding its pace prints one rate, not two.
            steady = describe(live, ranks, DEFAULT_STALE_SECONDS, working)
            assert "over the last" not in io_capture(steady), io_capture(steady)
            # Too few evaluations to compare a window against a history: no
            # ratio at all rather than one drawn from a handful of gaps.
            assert describe(live, ranks, DEFAULT_STALE_SECONDS)["slowdown_factor"] is None

            # A run whose imager is broken: every count healthy, every point
            # worthless. This is the one the other checks cannot see.
            poisoned = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260102T000000Z"
            (poisoned / "chains").mkdir(parents=True)
            for i in range(3):
                write_eval(poisoned, i + 1, now - 3 + i, objective=100.0)
            report = describe(poisoned, [], DEFAULT_STALE_SECONDS)
            assert report["failed"] == 3, report
            assert any("broken imager" in w for w in report["warnings"]), report
            # It died before PolyChord's main loop, which no count shows.
            assert report["stage"] == "starting up", report

            # Stall accounting: an outlier gap against the run's own steady
            # state. Ten evaluations a second apart and then one 20s hole -
            # median 1s, so the threshold is 10s and only the hole clears it.
            stalled = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260103T000000Z"
            (stalled / "chains").mkdir(parents=True)
            for i in range(10):
                write_eval(stalled, i + 1, now - 40 + i)
            write_eval(stalled, 11, now - 11, wedges=1)
            report = describe(stalled, [], DEFAULT_STALE_SECONDS)
            assert report["stall_threshold_seconds"] == 10.0, report
            assert report["stall_count"] == 1, report
            assert abs(float(report["stall_seconds"]) - 20) < 0.1, report
            assert abs(float(report["stall_fraction"]) - 20 / 29) < 0.01, report
            assert report["meqserver_wedges"] == 1, report
            assert any("of wall clock lost" in w for w in report["warnings"]), report

            # The same shape at WSClean's own pace - 30 evaluations a second -
            # must not turn ordinary jitter into stalls, which is what the
            # floor under the relative threshold is for.
            fast = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260103T010000Z"
            (fast / "chains").mkdir(parents=True)
            for i in range(20):
                write_eval(fast, i + 1, now - 10 + i * 0.033)
            fast_report = describe(fast, [], DEFAULT_STALE_SECONDS)
            assert fast_report["stall_threshold_seconds"] == MIN_STALL_GAP_SECONDS
            assert fast_report["stall_count"] == 0, fast_report

            # A sidecar whose launcher is gone is memory nobody will free.
            processes = [{"pid": 4242, "alive": True, "elapsed_seconds": 1.0,
                          "cpu_seconds": 0.0, "args": "sh"}]
            live_container = {"name": "ri-ns-sidecar-4242-0", "image": "i", "owner_pid": 4242}
            leaked = {"name": "ri-ns-sidecar-9999-0", "image": "i", "owner_pid": 9999}
            global sidecar_containers
            original = sidecar_containers
            try:
                sidecar_containers = lambda: [live_container, leaked]  # noqa: E731
                host = host_report(processes)
                assert host["leaked_sidecars"] == [leaked], host
                assert any("docker rm -f ri-ns-sidecar-9999-0" in w for w in host["warnings"])
                # A zombie launcher is a dead one, however well `kill -0` does.
                zombie = [dict(processes[0], pid=9999, alive=False)]
                assert host_report(zombie)["leaked_sidecars"] == [live_container, leaked]
                sidecar_containers = lambda: None  # noqa: E731
                assert host_report(processes)["leaked_sidecars"] == []
            finally:
                sidecar_containers = original

            # Resolution: a bare name, a path, and the newest by default.
            assert resolve(live.name) == live
            assert resolve(str(stalled)) == stalled
            # Newest means newest, not last alphabetically: `wsclean-*` sorts
            # after `r2d2-*` whatever their stamps say.
            assert resolve(None).name == fast.name
            newest = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260104T000000Z"
            newest.mkdir()
            assert resolve(None).name == newest.name
            assert [d.name for d in run_directories()][0] == newest.name
            newest.rmdir()

            # And the whole thing renders and scores. Into a sink, because what
            # is checked is that both forms run and reach the right exit status,
            # not what they print. The host is stubbed healthy: whether the
            # machine running the check happens to be short of memory or to be
            # holding a leaked container is not what is under test.
            import contextlib
            import io

            global available_mb
            original_memory = available_mb
            try:
                sidecar_containers = lambda: []  # noqa: E731
                available_mb = lambda: HEADROOM_MB * 2  # noqa: E731
                with contextlib.redirect_stdout(io.StringIO()):
                    assert main(["--all", "--json"]) == 1
                    assert main([live.name]) == 0
            finally:
                sidecar_containers = original
                available_mb = original_memory
    finally:
        NESTED_SAMPLING_DIR = saved
    print("nested-sampling-health self-check passed")


if __name__ == "__main__":
    if os.environ.get("NESTED_SAMPLING_HEALTH_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
