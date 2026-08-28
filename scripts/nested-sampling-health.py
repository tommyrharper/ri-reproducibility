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
  looks perfectly healthy by every other measure and is worth nothing. Asked of
  the last `RATE_WINDOW` evaluations as well as of the whole run, because an
  imager that breaks part-way through a long one stays under any whole-run
  ratio for hours.
* **Cost.** Gaps between evaluations, and `meqserver-wedged.log`, put a number
  on the MeqTrees deadlock the watchdogs in `simulate_point_source_ms.py`
  absorb (docs/nested-sampling.md, "When MeqTrees stops answering").

* **Shape.** Every number above is one moment. `history` is the run's
  throughput binned over its own life, which is what separates a dip that
  recovered from a step down that did not - the two the medians report
  identically on the way past each other.

* **Where the time goes.** A falling evaluation rate is either a slower imager
  or idle ranks, and the rate alone cannot tell them apart. `imaging` reports
  the imager's own wall clock per evaluation and, against the wall clock the
  run has spent, how much of its hardware that cost is actually keeping busy -
  the only place here where memory a run holds but is not using is visible.
  `occupancy` is that same duty cycle binned over the run's life, which is what
  separates hardware that has been earning its keep all along from hardware
  that happens to be busy at the moment it was asked.

* **Cost, again, in memory and cores.** Memory is what caps a run here, so
  what the run holds is reported next to what the host has left - over every
  process carrying the run directory, because a rank is ~10MB and the imager
  worker behind it is ~3.3GB.

* **Cost, again, on disk.** The one resource nothing here reserves, checks or
  frees, and the only one that only ever grows: an evaluation directory keeps
  its measurement set and the imager's output, ~1.7MB, and nothing deletes it.
  A live R2D2 run writes ~2.6GB/hour, so the run's own rate against the free
  space is the only warning available before it ends on ENOSPC.

* **How far through.** `chains/*.stats` carries the evidence the search has
  actually accumulated and what each dead point cost in likelihood calls, and
  with the live points beside it that gives the one thing a `--max-ndead -1`
  run has nowhere else: a denominator. `forecast` turns the prior volume still
  to be compressed into dead points left and hours left.

Plus the host: memory, free disk, and sidecar containers whose run is gone. A killed run
leaves its `ri-ns-sidecar-*` containers holding ~3.4GB per R2D2 rank. The next
run frees those itself before it sizes itself, so this is here to explain where
the host's memory went, not as a chore.

Filesystem reads, one `ps` and one `docker ps`, plus a one second CPU sample
when a run has live processes; nothing started, nothing imaged, so it costs a
busy host nothing to ask.

Usage:

  uv run scripts/nested-sampling-health.py            # the run that is going
  uv run scripts/nested-sampling-health.py <run>
  uv run scripts/nested-sampling-health.py --all
  uv run scripts/nested-sampling-health.py --json

Exit status is 0 when nothing needs attention and 1 when something does, so it
can gate a script.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
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

# The imager's own wall clock for one evaluation, out of the metrics.json this
# scan already reads in full - so it costs a regex over a string in memory and
# no extra I/O. Pulled out by pattern rather than json.loads for the same
# reason the failure marker is: 5,000 files a run, and only one number wanted.
WALL_SECONDS_PATTERN = re.compile(r'"wall_seconds":\s*([0-9.eE+-]+)')

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

# Evaluations in the "how is it going now" window, and how far it has to
# diverge from the run's own median before both numbers are worth printing
# rather than one. Doubled rather than half again, because a healthy run's own
# pace is noisier than it looks: five minute bins over 107 minutes of one
# ranged 91-165 with no fault present. Walking that run's history, this fires
# at 269 sampled moments in 15% of them at 1.5x and 5% at 2.0x, and the knee is
# where the ordinary swing stops and the one real event starts.
RATE_WINDOW = 50
RATE_DIVERGENCE_FACTOR = 2.0

# Below this much elapsed time, dividing a count by it measures mtime
# granularity rather than throughput. Parallel ranks land their first batch
# together - so the opening evaluations of any run share a timestamp to within
# milliseconds - and a run killed during that batch is left holding nothing
# else. Real runs on this host printed "6176.5/min over 0:00:00" from 14
# evaluations 0.14s apart and "8700112/min" from 42 of them 0.3ms apart, both
# of which are the arithmetic working exactly as written on an input that
# means nothing. One second, because that is the resolution the span is
# displayed at: under it there is no honest way to print the denominator.
MIN_RATE_SPAN_SECONDS = 1.0

# The same window, read for failures rather than pace, and half of it is the
# bar. The overall ratio below cannot see an imager that broke part-way
# through: a run three hours healthy and twenty minutes broken is still ~2%
# failures overall and silent, while every point it is now adding is a
# FAILURE_OBJECTIVE that PolyChord will happily maximize. Half is not a tuned
# number - across 37,000 evaluations of the six real runs on this host the
# failure count is zero, so any sustained burst is a fault and the threshold
# only has to sit above the noise, which is nothing.
RECENT_FAILURE_FRACTION = 0.5

# The run's life as one line of text. Twenty slices so the whole thing fits
# beside a label at any terminal width, and a zero slice gets its own mark so a
# gap where nothing landed is visible rather than rounding to "slow".
HISTORY_BUCKETS = 20
HISTORY_LEVELS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
HISTORY_EMPTY = "\u00b7"

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

# Disk is the one resource nothing here reserves, checks or frees, and the only
# one that only ever grows: an evaluation directory keeps its measurement set,
# its .mat and the imager's output, ~1.7MB on this host, and nothing deletes
# it. A live R2D2 run writes ~2.6GB/hour and a WSClean run 18GB over a few, so
# a multi-day search on a 233GB filesystem is a plausible ENOSPC and there is
# no other place that would say so first.
#
# Measured from a sample rather than a walk: `du -s` on one live run directory
# cost 3-5s of I/O against the disk the run is using, which is not what a
# read-only health check should do to it. Twenty of the newest evaluations,
# which cost milliseconds, since what varies between them is imager output size
# and not the shape of the directory.
DISK_SAMPLE = 20
DISK_WARN_HOURS = 12.0

# Where PolyChord stops, as a fraction of the evidence already collected still
# sitting in the live points. This is the one number the forecast rests on, and
# it is measured rather than taken from the documentation: `precision_criterion`
# defaults to 1e-3, but the two searches on this host that ran to natural
# termination (wsclean, nlive=50, seeds 123 and 372) stopped at 446 and 463
# dead points where 1e-3 predicts 350 for both. The ratio they actually reached
# was 1.3e-4 and 9.6e-5; their mean forecasts 451, which is 1% and 3% out
# instead of 25% short twice. Recalibrate here if a PolyChord upgrade or a
# non-default precision_criterion moves it.
TERMINATION_EVIDENCE_RATIO = 1.2e-4


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
    """Every process, as (pid, state, elapsed, cpu, rss, args).

    One call, because everything here needs it: the ranks of a run, whether a
    sidecar's launcher is still alive, how much of its life a rank has spent on
    CPU, and how much memory the run is holding.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,state=,etime=,time=,rss=,args="],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, object]] = []
    for line in out.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        pid, state, etime, cpu, rss, args = fields
        rows.append({
            "pid": int(pid),
            # A process killed while its parent is not wait()ing stays as a
            # zombie, and `kill -0` succeeds on one - so state, not existence,
            # is what says a process is alive.
            "alive": state[:1] != "Z",
            "elapsed_seconds": _clock_seconds(etime),
            "cpu_seconds": _clock_seconds(cpu),
            # KB on both BSD and GNU ps.
            "rss_mb": int(rss) // 1024 if rss.isdigit() else 0,
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


def meminfo_mb(key: str) -> int | None:
    """One /proc/meminfo field in MB, or None off Linux.

    scripts/lib/rank-budget.sh is the authority on what a run may take;
    `MemAvailable` here only reports the number its decision will be made from.
    `MemTotal` is what turns a run's footprint into a share of the host.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(f"{key}:"):
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


def _bucket(times: list[float],
            values: list[float]) -> tuple[list[float], float] | None:
    """Sum `values` into HISTORY_BUCKETS even slices of [first, last].

    Binned over [first evaluation, last evaluation] rather than up to now, so
    every slice is a full one. A slice ending at the present moment is partial
    by definition and reads low by exactly the fraction of it that has not
    happened yet, which is indistinguishable from a real collapse - the same
    trap the gap-based rate elsewhere exists to avoid.

    None below two evaluations a slice, where the counts are too small to be a
    shape rather than noise, and below a second a slice, where the slice rates
    are mtime granularity rather than throughput (MIN_RATE_SPAN_SECONDS).
    """
    if len(times) < 2 * HISTORY_BUCKETS:
        return None
    span = times[-1] - times[0]
    if span < HISTORY_BUCKETS * MIN_RATE_SPAN_SECONDS:
        return None
    width = span / HISTORY_BUCKETS
    sums = [0.0] * HISTORY_BUCKETS
    for when, value in zip(times, values):
        sums[min(HISTORY_BUCKETS - 1, int((when - times[0]) / width))] += value
    return sums, width


def history(times: list[float]) -> dict[str, object] | None:
    """The shape of the run's throughput over its own life, as one line of text.

    The two medians already reported say how fast the run is going now against
    how fast it has gone; neither can show the shape, and the shape is the
    thing a reader actually wants. A dip that recovered, a step down that did
    not, and a steady run all produce the same pair of numbers on the way past
    each other - the observed collapse-and-recovery here (104, 23, 26, 93
    against a 104-165 baseline) reads as an ordinary slowdown from the medians
    alone and as an obvious V from twenty slices.

    Slicing, and the reasons for it, are in `_bucket`. How long ago the last
    evaluation landed is the activity line's job, not this one's.
    """
    binned = _bucket(times, [1.0] * len(times))
    if binned is None:
        return None
    sums, width = binned
    counts = [int(c) for c in sums]  # _bucket sums floats; these are tallies
    peak = max(counts)
    return {
        # Scaled to the run's own peak: this answers "what changed", and an
        # absolute scale shared with nothing else would only flatten it.
        "bar": "".join(HISTORY_EMPTY if c == 0
                       else HISTORY_LEVELS[(c * len(HISTORY_LEVELS) - 1) // peak]
                       for c in counts),
        "low_per_minute": min(counts) * 60 / width,
        "high_per_minute": peak * 60 / width,
        "bucket_seconds": width,
    }


def occupancy(times: list[float], costs: list[float | None],
              procs: int) -> dict[str, object] | None:
    """The shape of the run's rank utilisation over its own life.

    `imaging` says what fraction of the ranks the cost of an evaluation is
    keeping busy right now; `history` says how the arrival rate has moved.
    Neither answers the question a shared host actually raises, which is
    whether the memory this run is holding has been earning its keep all along
    - and the two cannot be read off each other, because the imager's own cost
    drifts as the search concentrates. The live R2D2 search here got twice as
    fast per evaluation while its arrival rate fell fivefold, so `history`
    showed a collapse over a period the ranks were merely idle for.

    Per slice: imaging seconds landed, over the rank-seconds the slice had to
    spend. A duty cycle, so unlike `history` the scale is absolute - a full bar
    is every rank imaging, and a run whose bar never leaves the floor is one
    that should have been given fewer ranks or a larger `--nlive`.

    Summed rather than taken from medians, because a duty cycle is a total over
    an interval; evaluations with no recorded cost are simply not counted, so a
    run whose imager never wrote `wall_seconds` gets no line rather than a
    misleadingly empty one.
    """
    if procs <= 0 or not any(c is not None for c in costs):
        return None
    binned = _bucket(times, [c or 0.0 for c in costs])
    if binned is None:
        return None
    seconds, width = binned
    # Clamped for the same reason the `imaging` line clamps: an evaluation is
    # attributed to the slice it *finished* in while its cost was spent partly
    # in the one before, so a slice can bank more imaging seconds than it had.
    fractions = [min(s / (width * procs), 1.0) for s in seconds]
    return {
        "bar": "".join(HISTORY_EMPTY if f == 0 else
                       HISTORY_LEVELS[min(int(f * len(HISTORY_LEVELS)),
                                          len(HISTORY_LEVELS) - 1)]
                       for f in fractions),
        "low_fraction": min(fractions),
        "high_fraction": max(fractions),
        "ranks": procs,
        "bucket_seconds": width,
    }


def _dir_bytes(path: Path) -> int:
    """A directory tree's disk usage, counted as `du` counts it.

    Allocated blocks rather than st_size: an evaluation holds a measurement
    set, which is a directory of many small files, so apparent size understates
    what the filesystem actually gave it.
    """
    total = 0
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue  # swept, or being written as this walks
        for entry in entries:
            try:
                total += entry.stat(follow_symlinks=False).st_blocks * 512
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
            except OSError:
                continue
    return total


def free_bytes(path: Path) -> tuple[int, int] | None:
    """(free, total) on the filesystem holding `path`, or None if it cannot say."""
    try:
        fs = os.statvfs(path)
    except (OSError, AttributeError):
        return None
    return fs.f_bavail * fs.f_frsize, fs.f_blocks * fs.f_frsize


def evaluation_scan(run_dir: Path, procs: int = 0) -> dict[str, object]:
    """Counts, timings and failures, in one pass over evaluations/.

    The stat and the read are the same pass because the interesting things -
    when an evaluation landed, and whether it landed on a failure - are one per
    file and there is no cheaper place to get either.
    """
    evaluations = run_dir / "evaluations"
    directories = 0
    # (when it landed, whether it failed, where it is), kept together so that
    # "how the run is going now" can be asked of failures as well as of pace,
    # and so the newest few can be measured for size without a second glob.
    records: list[tuple[float, bool, Path, float | None]] = []
    wedged_lines = 0
    for entry in evaluations.glob("eval-*"):
        if not entry.is_dir():
            continue
        directories += 1
        metrics = entry / "metrics.json"
        try:
            when, text = metrics.stat().st_mtime, metrics.read_text()
        except OSError:
            pass  # in flight, or a leftover the next run will sweep
        else:
            cost = WALL_SECONDS_PATTERN.search(text)
            records.append((when, FAILURE_OBJECTIVE_MARKER in text, entry,
                            float(cost.group(1)) if cost else None))
        try:
            wedged_lines += len((entry / "meqserver-wedged.log").read_text().splitlines())
        except OSError:
            pass
    records.sort()
    times = [when for when, _, _, _ in records]
    failed = sum(1 for _, bad, _, _ in records if bad)
    recent_failed = (sum(1 for _, bad, _, _ in records[-RATE_WINDOW:] if bad)
                     if len(records) >= RATE_WINDOW else None)
    gaps = [b - a for a, b in zip(times, times[1:])]
    threshold = MIN_STALL_GAP_SECONDS
    if gaps:
        threshold = max(threshold, STALL_GAP_FACTOR * statistics.median(gaps))
    # A restart's downtime is not a stall. The run was not running, the reason
    # is known, and it is already on the `restarts` line - counting it here
    # warned twice about one event and pointed the second warning at the wrong
    # cause. Measured on a self-healed wsclean run whose only gap over the
    # threshold was its own 12s restart: "13% of wall clock lost to gaps over
    # 2s", which reads as the MeqTrees deadlock this number exists to size.
    downtime = restart_times(run_dir)
    stalls = [gap for start, gap in zip(times, gaps)
              if gap > threshold and not any(start <= t <= start + gap for t in downtime)]
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
    recent_rate, slowdown = None, None
    if len(gaps) >= 2 * RATE_WINDOW:
        recent = statistics.median(gaps[-RATE_WINDOW:])
        overall = statistics.median(gaps)
        recent_rate = 60 / recent if recent > 0 else None
        slowdown = recent / overall if overall > 0 else None

    # What one evaluation costs the imager, against how fast evaluations are
    # arriving. The arrival rate alone cannot separate "the imager got slower"
    # from "the ranks are idle": both read as a smaller rate. The imager's own
    # wall clock separates them, and the imaging seconds banked per second of
    # wall clock is how many ranks the run is actually keeping busy, which is
    # the number that says whether the memory it is holding is being used.
    #
    # The cost is a median, because evaluation cost genuinely varies with the
    # parameters drawn and a nested-sampling run concentrates: the live R2D2
    # search here ran at a 25.4s median over its life and 12.2s over its last
    # 50, with no fault.
    def _cost(rows: list) -> float | None:
        seen = [w for _, _, _, w in rows if w is not None]
        return statistics.median(seen) if seen else None

    # The occupancy is a total, not a ratio of the two medians. A duty cycle is
    # by definition seconds worked over seconds elapsed, and the median gap is
    # shorter than the mean whenever a run stalls at all, so the ratio of
    # medians reads systematically high - the live R2D2 search here printed a
    # clamped "100% busy" over a life its own slices put at 6-92%. Two figures
    # in one report disagreeing about the same thing is worse than either.
    #
    # Gated on the same span floor as the rate, and for the same reason: a run
    # killed inside its opening parallel batch has every evaluation landing in
    # the same millisecond, so the elapsed time is mtime granularity and the
    # ratio is a division by noise rather than an occupancy.
    def _duty(rows: list) -> float | None:
        if len(rows) < 2:
            return None
        elapsed = rows[-1][0] - rows[0][0]
        if elapsed < MIN_RATE_SPAN_SECONDS:
            return None
        return sum(w for _, _, _, w in rows if w is not None) / elapsed

    cost, recent_cost = _cost(records), None
    busy_ranks, recent_busy_ranks = _duty(records), None
    if span >= MIN_RATE_SPAN_SECONDS and len(gaps) >= 2 * RATE_WINDOW:
        recent_cost = _cost(records[-RATE_WINDOW:])
        recent_busy_ranks = _duty(records[-RATE_WINDOW:])

    # Spread over the run's life rather than taken from its tail. An
    # evaluation's size follows its parameters, and a nested-sampling run
    # concentrates on a shrinking region, so the newest evaluations drift away
    # from the run's own average: newest-20 read 1.45MB against a true 1.68MB
    # on the live R2D2 run here, while an even stride of 20 read 1.70MB.
    # Strided rather than random so the number does not move between two
    # readings of an unchanged run.
    stride = max(1, len(records) // DISK_SAMPLE)
    sample = [where for _, _, where, _ in records[::stride]]
    per_evaluation = (statistics.mean(_dir_bytes(where) for where in sample)
                      if sample else None)
    rate = (len(times) * 60 / span if span >= MIN_RATE_SPAN_SECONDS else None)

    return {
        "completed": len(times),
        "in_flight": directories - len(times),
        "failed": failed,
        "last_activity_seconds": time.time() - times[-1] if times else None,
        "span_seconds": span,
        "recent_failed": recent_failed,
        "evals_per_minute": rate,
        "recent_evals_per_minute": recent_rate,
        "seconds_per_evaluation": cost,
        "recent_seconds_per_evaluation": recent_cost,
        "busy_ranks": busy_ranks,
        "recent_busy_ranks": recent_busy_ranks,
        "bytes_per_evaluation": per_evaluation,
        "disk_bytes": per_evaluation * len(times) if per_evaluation else None,
        "disk_bytes_per_hour": (per_evaluation * rate * 60
                                if per_evaluation and rate else None),
        "history": history(times),
        "occupancy": occupancy(times, [w for _, _, _, w in records], procs),
        "slowdown_factor": slowdown,
        "stall_threshold_seconds": threshold,
        "stall_count": len(stalls),
        "stall_seconds": sum(stalls),
        "stall_fraction": sum(stalls) / span if span > 0 else 0.0,
        "meqserver_wedges": wedged_lines,
    }


def restarts(run_dir: Path) -> list[str]:
    """The times this run died and started itself again from its checkpoint.

    Written by run_with_retries in scripts/lib/progress-bar.sh, one line per
    restart. Its own file rather than a grep of run.log because that file is
    megabytes of PolyChord feedback after a day, and this is a handful of
    lines - a self-healed run is still worth saying out loud, since the thing
    that killed it once will do it again.
    """
    try:
        return [line for line in
                (run_dir / "restarts.log").read_text(errors="replace").splitlines()
                if line.strip()]
    except OSError:
        return []


def restart_times(run_dir: Path) -> list[float]:
    """Epoch seconds of each restart, for the gap accounting to skip over.

    progress-bar.sh writes the line with `date -u`, so the stamp is UTC and
    has to be read as such - read as local time it would land hours away from
    the evaluation mtimes it is compared against and match no gap at all.
    """
    stamps = []
    for line in restarts(run_dir):
        try:
            when = time.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, IndexError):
            continue
        stamps.append(calendar.timegm(when))
    return stamps


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


def _setting(run_env: dict[str, str], key: str) -> int | None:
    """A numeric setting out of run.env, or None if it is absent or not one."""
    raw = run_env.get(key, "")
    return int(raw) if raw.lstrip("-").isdigit() else None


def sampler_stats(run_dir: Path) -> dict[str, object] | None:
    """What PolyChord itself says it has found, out of `chains/*.stats`.

    Rewritten at every checkpoint, and the only artifact that carries the
    number the search exists to produce. Nothing here read it before, so a run
    could be reported healthy on every operational line while saying nothing
    about its own result - and `nlike` per dead point, the sampler's own
    efficiency, is the one cost that a rate in evaluations per minute cannot
    show, because it is what that rate is being spent on.

    A checkpoint rewrite can be read half-written; every field is therefore
    optional and a torn read simply reports less.
    """
    for path in (run_dir / "chains").glob("*.stats"):
        try:
            text = path.read_text()
        except OSError:
            return None
        found: dict[str, object] = {}
        # The global evidence, which is the first `log(Z)` in the file - the
        # per-cluster ones below it are `log(Z_1)` and so on.
        match = re.search(r"^log\(Z\)\s*=\s*(\S+)\s*\+/-\s*(\S+)", text, re.MULTILINE)
        if match:
            try:
                found["log_z"] = float(match.group(1))
                found["log_z_error"] = float(match.group(2))
            except ValueError:
                pass
        for key in ("ndead", "nlive", "nlike"):
            hit = re.search(rf"^\s*{key}:\s*(\d+)", text, re.MULTILINE)
            if hit:
                found[key] = int(hit.group(1))
        return found or None
    return None


def live_loglikelihoods(run_dir: Path) -> list[float]:
    """The likelihood of every point still alive, from `chains/*_phys_live.txt`.

    Last column, the same file PolyChord rewrites with the stats. Empty for a
    finished run, which is how a finished run gets no forecast.
    """
    for path in (run_dir / "chains").glob("*_phys_live.txt"):
        try:
            return [float(line.split()[-1]) for line in path.read_text().splitlines()
                    if line.strip()]
        except (OSError, ValueError, IndexError):
            return []
    return []


def evidence_forecast(run_dir: Path, stats: dict[str, object] | None,
                      nlive: int | None, max_ndead: int | None) -> dict[str, object] | None:
    """How far through the search is, and how many dead points are left.

    The gap this closes is that with `--max-ndead -1`, the default, a run has
    no denominator anywhere: `./ri health` could say a search was healthy and
    fast for three days without ever saying whether it was a tenth of the way
    through or nearly done.

    Nested sampling supplies one. Each dead point shrinks the prior volume by
    the same factor, so what is left of it is exp(-ndead/nlive); the evidence
    still to come is that volume times the mean likelihood of the points now
    sitting in it, and the run ends when that falls to
    TERMINATION_EVIDENCE_RATIO of the evidence already banked. The volume
    shrinks one e-fold per `nlive` dead points, which turns "how much further
    that ratio has to fall" into a count.

    An explicit `--max-ndead` is a hard stop the sampler will hit first, so it
    is used directly and the answer is not an estimate at all.
    """
    if not stats or "ndead" not in stats or not nlive:
        return None
    ndead = int(stats["ndead"])
    if max_ndead is not None and max_ndead > 0:
        total, estimated = max_ndead, False
    else:
        live = live_loglikelihoods(run_dir)
        # Before the first e-fold the live set is still the prior and the
        # ratio is ~1, so the forecast would be reporting the constant
        # nlive*ln(1/ratio) and nothing about this run.
        if "log_z" not in stats or not live or ndead < nlive:
            return None
        # Shifted by the largest live likelihood before exponentiating, so a
        # metric with real dynamic range cannot overflow the mean. The metrics
        # used here span ~0.006 nats and would be safe either way.
        peak = max(live)
        log_z_live = (-ndead / nlive + peak
                      + math.log(sum(math.exp(x - peak) for x in live) / len(live)))
        remaining = nlive * (log_z_live - float(stats["log_z"])
                             - math.log(TERMINATION_EVIDENCE_RATIO))
        total, estimated = ndead + max(0, round(remaining)), True
    return {
        "total_dead_points": total,
        "fraction": min(1.0, ndead / total) if total > 0 else None,
        "estimated": estimated,
    }


def run_processes(run_dir: Path, processes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Everything alive that carries this run's directory in its arguments.

    Wider than the ranks on purpose: the memory a run actually holds is almost
    all in its imager workers, which name the run by their --fifo-dir and are
    visible on the host even though they live inside the sidecar containers. A
    rank itself is ~10MB against an R2D2 worker's ~3.3GB.
    """
    marker = str(run_dir.resolve())
    return [p for p in processes if p["alive"] and marker in str(p["args"])]


def rank_processes(run_dir: Path, processes: list[dict[str, object]]) -> list[dict[str, object]]:
    """This run's MPI ranks, found by the --output-dir they were launched with.

    Which is also how the run is known to be alive at all: no ranks, no run.

    Anchored at the interpreter because `mpirun` and the host-side `docker
    exec` both carry the whole rank command line in their own arguments, and
    counting those as ranks puts the count two over `NS_MPI_PROCS`.
    """
    return [p for p in run_processes(run_dir, processes)
            if RANK_COMMAND.match(str(p["args"]))]


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
    scan = evaluation_scan(run_dir, _setting(run_env, "NS_MPI_PROCS") or 0)
    owned = run_processes(run_dir, processes)
    ranks = [p for p in owned if RANK_COMMAND.match(str(p["args"]))]
    spinning = spinning_ranks(ranks, busy or {})
    # RSS, so shared pages are counted once per process holding them. The
    # imager workers are separate containers with separate model copies, so on
    # the run this exists for the sum lands within a percent of what the host
    # reports as used - and overcounting is the safe direction for a number
    # read to answer "can another run fit".
    resident_mb = sum(int(p["rss_mb"]) for p in owned)
    cores_busy = sum((busy or {}).get(int(p["pid"]), 0.0) for p in owned)
    dead, checkpoint_age = dead_points(run_dir)
    tail = log_tail(run_dir)
    restarted = restarts(run_dir)
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

    # Only for a run that is still going: a stopped run's remaining dead
    # points are not remaining, they are lost, and its hours-left would be
    # counted off a rate that stopped.
    stats = sampler_stats(run_dir)
    forecast = None
    if status in ("healthy", "stalled", "starting"):
        forecast = evidence_forecast(run_dir, stats, _setting(run_env, "NS_NLIVE"),
                                     _setting(run_env, "NS_MAX_NDEAD"))
    if forecast and stats and float(scan["span_seconds"] or 0) >= MIN_RATE_SPAN_SECONDS:
        # Dead points per second over the run's own life. Not the evaluation
        # rate: the two are related by the sampler efficiency this line exists
        # to make visible, and that efficiency changes as the search moves.
        left = int(forecast["total_dead_points"]) - int(stats["ndead"])
        rate = int(stats["ndead"]) / float(scan["span_seconds"])
        forecast["hours_remaining"] = left / rate / 3600 if rate > 0 else None

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
    # Only reached when the overall ratio is fine, which is the whole point: an
    # imager that breaks part-way through a long run stays under that ratio for
    # hours while every point it adds from now on is a failure.
    elif scan["recent_failed"] is not None \
            and int(scan["recent_failed"]) >= RECENT_FAILURE_FRACTION * RATE_WINDOW:
        warnings.append(
            f"{scan['recent_failed']} of the last {RATE_WINDOW} evaluations scored "
            f"FAILURE_OBJECTIVE against {failed} of {completed} over the whole run "
            "- the imager broke part-way through, and the search is now feeding on it"
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
    # Nothing reserves disk, nothing frees it, and no evaluation directory is
    # ever deleted, so the only warning available is the run's own write rate
    # against what the filesystem has left. Asked only of a run that is still
    # writing: a finished run's GB are already spent and its rate is history.
    space = free_bytes(run_dir)
    per_hour = scan["disk_bytes_per_hour"]
    disk_hours = None
    if space is not None and per_hour and status in ("healthy", "starting", "stalled"):
        disk_hours = space[0] / float(per_hour)
        if disk_hours < DISK_WARN_HOURS:
            warnings.append(
                f"{space[0] / 1024 ** 3:.0f}GB free is ~{disk_hours:.0f}h at this run's "
                f"{float(per_hour) / 1024 ** 3:.1f}GB/hour - nothing here prunes "
                "evaluations, so the run ends on ENOSPC unless space is made"
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
        "processes": len(owned),
        "resident_mb": resident_mb,
        "cores_busy": round(cores_busy, 1),
        "host_cores": os.cpu_count(),
        "disk_free_bytes": space[0] if space is not None else None,
        "disk_hours_remaining": disk_hours,
        "dead_points": dead,
        "checkpoint_age_seconds": checkpoint_age,
        "sampler": stats,
        "forecast": forecast,
        "log_tail": tail,
        "restarts": restarted,
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
            "holding ~3.4GB per R2D2 rank. The next run frees them before it sizes "
            "itself (scripts/lib/rank-budget.sh); to have the memory now: "
            "docker rm -f " + " ".join(str(c["name"]) for c in leaked)
        )
    memory = meminfo_mb("MemAvailable")
    if memory is not None and memory < HEADROOM_MB:
        warnings.append(
            f"{memory}MB available is below the {HEADROOM_MB}MB headroom rank-budget.sh "
            "keeps free; a new run will refuse to size itself"
        )
    space = free_bytes(NESTED_SAMPLING_DIR if NESTED_SAMPLING_DIR.exists() else Path("."))
    return {
        "available_mb": memory,
        "total_mb": meminfo_mb("MemTotal"),
        "disk_free_bytes": space[0] if space is not None else None,
        "disk_total_bytes": space[1] if space is not None else None,
        "cores": os.cpu_count(),
        "sidecars": containers,
        "leaked_sidecars": leaked,
        "warnings": warnings,
    }


# --- rendering ---------------------------------------------------------------


def format_gb(value: float) -> str:
    """GB at one decimal, MB below that, so a young run does not read as 0.0GB."""
    return (f"{value / 1024 ** 3:.1f}GB" if value >= 0.1 * 1024 ** 3
            else f"{value / 1024 ** 2:.0f}MB")


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def format_hours(hours: float) -> str:
    """`4h40m` under a day, `2d 6h` over one - a wait, not a duration to add up."""
    if hours >= 24:
        return f"{int(hours // 24)}d {int(hours % 24)}h"
    return f"{int(hours)}h{int(hours % 1 * 60):02d}m"


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
    # something to compare against rather than a number that looks stuck. Only
    # for a run that is still going: a finished or dead run's count will never
    # move again, and promising a next value is the one thing that would make
    # its stale-by-design mtime read as work still to come.
    next_update = ""
    if run["status"] in ("healthy", "stalled", "starting") \
            and settings.get("NS_NLIVE", "").isdigit() \
            and run["checkpoint_age_seconds"] is not None:
        next_update = f", next at ~{int(run['dead_points']) + int(settings['NS_NLIVE'])}"
    lines = [
        # The dead-point count never appears without how old it is. PolyChord
        # writes it every ~nlive points, so it is stale by design between
        # writes and a frozen-looking count means nothing on its own.
        ("stage", f"{run['stage']}, {run['dead_points']} dead points"
                  + (f" as of {format_hms(float(run['checkpoint_age_seconds']))} ago"
                     f"{next_update}"
                     if run["checkpoint_age_seconds"] is not None else "")),
        # Directories without a metrics.json are evaluations in flight only
        # while something is still flying them. On a run with no ranks left
        # they are what the ranks were holding when it died, and calling those
        # "in flight" reads as work still happening on a run that ended hours
        # ago.
        ("progress", f"{run['completed']} evaluations, {run['in_flight']} "
                     + ("in flight" if run["ranks"] else "abandoned")),
        ("activity", "nothing yet" if idle is None else
                     f"last evaluation {format_hms(float(idle))} ago"
                     + (f", {rate:.1f}/min over {format_hms(float(run['span_seconds']))}"
                        if rate else "")
                     # Only when the run has changed pace materially, in either
                     # direction: the two numbers agreeing says nothing.
                     + (f" ({run['recent_evals_per_minute']:.1f}/min over the last "
                        f"{RATE_WINDOW})"
                        if slowdown is not None and (
                            float(slowdown) > RATE_DIVERGENCE_FACTOR
                            or float(slowdown) < 1 / RATE_DIVERGENCE_FACTOR) else "")),
    ]
    # ...and the same throughput as a shape rather than as two numbers, which
    # is the only line here that shows a dip that recovered as different from
    # a step down that did not.
    past = run["history"]
    if past:
        assert isinstance(past, dict)
        lines.append(("history",
                      (f"{past['bar']}  {past['low_per_minute']:.0f}-"
                       f"{past['high_per_minute']:.0f}/min per "
                       f"{format_hms(float(past['bucket_seconds']))} slice")))
    # What an evaluation costs and how much of the run's hardware that cost is
    # spread over. `activity` reports arrival rate, which confounds a slower
    # imager with idle ranks; this line separates them, and the occupancy is
    # the only place in the report where memory the run is holding but not
    # using shows up as such.
    cost = run["seconds_per_evaluation"]
    if cost is not None:
        # As a percentage of the ranks the run was given, clamped: an
        # evaluation is banked at the moment it finished while its cost was
        # spent before that, so a window can hold more imaging seconds than it
        # had rank-seconds to spend and the raw ratio would print "23 of 16".
        procs = settings.get("NS_MPI_PROCS") or (str(run["ranks"]) or "")
        def _busy(value: object) -> str:
            if value is None or not str(procs).isdigit() or int(procs) <= 0:
                return ""
            return f", ranks {min(float(value) / int(procs), 1.0):.0%} busy"
        recent_cost = run["recent_seconds_per_evaluation"]
        changed = ""
        if recent_cost is not None and (
                not (1 / RATE_DIVERGENCE_FACTOR < float(recent_cost) / cost
                     < RATE_DIVERGENCE_FACTOR)
                or _busy(run["recent_busy_ranks"]) != _busy(run["busy_ranks"])):
            changed = (f"  (last {RATE_WINDOW}: {float(recent_cost):.1f}s"
                       f"{_busy(run['recent_busy_ranks']).replace(' ranks', '')})")
        lines.append(("imaging", f"{float(cost):.1f}s per evaluation"
                                 + _busy(run["busy_ranks"]) + changed))
    # ...and that occupancy as a shape, which is the one line here that says
    # whether the hardware the run is holding has been earning its keep all
    # along or only at the moment it was asked. Absolute scale, unlike
    # `history` above: a full bar is every rank imaging.
    used = run["occupancy"]
    if used:
        assert isinstance(used, dict)
        lines.append(("occupancy",
                      (f"{used['bar']}  {float(used['low_fraction']):.0%}-"
                       f"{float(used['high_fraction']):.0%} of "
                       f"{used['ranks']} ranks busy per "
                       f"{format_hms(float(used['bucket_seconds']))} slice")))
    # What the search has actually found, and what each dead point cost it.
    # Every other line here is operational; this one is the result, and the
    # calls-per-dead-point is the sampler's own efficiency - the thing an
    # evaluation rate is being spent on, and the only place a search that is
    # working hard for nothing shows up as such.
    stats = run["sampler"]
    if stats and "log_z" in stats:
        per_dead = ""
        if stats.get("ndead") and stats.get("nlike"):
            calls = round(int(stats["nlike"]) / int(stats["ndead"]))
            per_dead = (f", {calls} likelihood call{'' if calls == 1 else 's'} "
                        "per dead point")
        lines.append((
            "sampler",
            (f"logZ = {float(stats['log_z']):.3f} "
             f"+/- {float(stats['log_z_error']):.3f}{per_dead}")))
    # The denominator a `--max-ndead -1` search otherwise has nowhere: without
    # it, "healthy and fast" is all this report can say about a run that might
    # be a tenth done or nearly finished.
    ahead = run["forecast"]
    if ahead:
        assert isinstance(ahead, dict)
        about = "~" if ahead["estimated"] else ""
        left = ahead.get("hours_remaining")
        lines.append(("forecast",
                      f"{about}{float(ahead['fraction']):.0%} done, "
                      f"{about}{ahead['total_dead_points']} dead points total"
                      + (f", {about}{format_hours(float(left))} left" if left else "")))
    lines.append(("ranks", ranks))
    # Only for a run that still holds something. "0.0GB over 0 processes" is
    # what every finished run on disk would print, and none of them is the
    # question this line answers: memory, not CPU, is what caps a run, and what
    # a live one is holding is what the next one has to fit around.
    if run["processes"]:
        cores = run["host_cores"]
        lines.append(("resources",
                      f"{int(run['resident_mb']) / 1024:.1f}GB resident over "
                      f"{run['processes']} processes"
                      + (f", {run['cores_busy']:.1f}"
                         + (f" of {cores}" if cores else "")
                         + " cores busy" if run["cores_busy"] else "")))
    # Memory and cores are held and given back; disk is only ever taken, and
    # by the time it runs out the run is over. So the run's own share is shown
    # as a rate and, while it is still writing, as the time that rate has left.
    if run["disk_bytes"]:
        per_hour = run["disk_bytes_per_hour"]
        remaining = run["disk_hours_remaining"]
        lines.append(("disk",
                      f"{format_gb(float(run['disk_bytes']))} written"
                      + (f", +{format_gb(float(per_hour))}/hour" if per_hour else "")
                      + (f", {float(remaining):.0f}h of space left at that rate"
                         if remaining is not None else "")))
    # Only when there were any: a run that has never crashed should not carry a
    # line saying so. Reported, not warned on - the crash was survived, and a
    # warning here would make `./ri health` exit nonzero for a run that is
    # currently fine.
    if run["restarts"]:
        restarted = list(run["restarts"])
        plural = "" if len(restarted) == 1 else "s"
        lines.append(("restarts",
                      f"{len(restarted)} self-healed restart{plural}, "
                      f"last {restarted[-1]}"))
    lines += [
        ("failures", f"{run['failed']} scored FAILURE_OBJECTIVE"
                     + (f" ({run['recent_failed']} of the last {RATE_WINDOW})"
                        if run["recent_failed"] else "")
                     + f", {run['meqserver_wedges']} meqserver wedges recovered"),
        ("stalls", (f"{run['stall_count']} gaps over "
                    f"{run['stall_threshold_seconds']:.0f}s, {run['stall_seconds']:.0f}s = "
                    f"{run['stall_fraction']:.1%} of wall clock")),
    ]
    for label, value in lines:
        print(f"  {label:<9} {value}")
    for warning in run["warnings"]:
        print(f"  WARNING   {warning}")


def render_host(host: dict[str, object]) -> None:
    print("host")
    memory = host["available_mb"]
    total = host["total_mb"]
    print(f"  {'memory':<9} " + ("unknown (not Linux)" if memory is None
                                 else f"{int(memory) / 1024:.1f}GB available"
                                      + (f" of {int(total) / 1024:.1f}GB" if total else "")
                                      + f", {HEADROOM_MB / 1024:.0f}GB reserved as headroom"))
    space = host["disk_free_bytes"]
    print(f"  {'disk':<9} " + ("unknown" if space is None
                               else f"{float(space) / 1024 ** 3:.0f}GB free of "
                                    f"{float(host['disk_total_bytes']) / 1024 ** 3:.0f}GB"))
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


def resolve(name: str | None,
            processes: list[dict[str, object]] | None = None) -> Path:
    """A run directory from a path, a bare run name, or nothing at all.

    With nothing at all, the newest run that still has ranks - not simply the
    newest run. This report's question is about a search that is going, and the
    newest run stops being that one the moment a short test lands after a
    multi-hour search: the test finishes in minutes, the search does not, and
    "the newest run" then names the only one of the two nobody is asking about.
    Falls back to the newest on a host with nothing running, which is every
    host most of the time.

    Ranks rather than every process carrying the run directory: a killed run's
    sidecar workers outlive it until the next run reaps them, and they are not
    a run still going.
    """
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
    for run in runs:
        if rank_processes(run, processes or []):
            return run
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", metavar="RUN",
                        help="run directory or its name (default: the newest "
                             "run that is still running, else the newest run)")
    parser.add_argument("--all", action="store_true", help="every run on disk")
    parser.add_argument("--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS,
                        help="a live run silent for this long is stalled "
                             "(default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="raw JSON instead of a report")
    args = parser.parse_args(argv)

    if args.all and args.run:
        parser.error("--all takes no run argument")

    processes = process_table()
    directories = run_directories() if args.all else [resolve(args.run, processes)]
    # One sample interval for every run being reported, not one each: only a
    # run with live processes is sampled at all, and there is rarely more than
    # one. A report over finished runs costs nothing. Everything the run owns
    # rather than only its ranks, because the ranks wait while the imager
    # workers do the work - sampling only the ranks measures the waiting.
    busy = cpu_busy_fractions(
        [int(p["pid"]) for d in directories for p in run_processes(d, processes)])
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

    global NESTED_SAMPLING_DIR, process_table, cpu_busy_fractions
    saved = NESTED_SAMPLING_DIR
    now = time.time()

    def write_eval(run: Path, index: int, mtime: float, objective: float = 0.008,
                   wedges: int = 0, wall_seconds: float | None = None) -> None:
        eval_dir = run / "evaluations" / f"eval-{index:04d}-abc"
        eval_dir.mkdir(parents=True)
        metrics = eval_dir / "metrics.json"
        body: dict[str, object] = {"eval_id": index, "objective": objective}
        if wall_seconds is not None:
            body["metrics"] = {"wall_seconds": wall_seconds}
        metrics.write_text(json.dumps(body, indent=2))
        os.utime(metrics, (mtime, mtime))
        if wedges:
            (eval_dir / "meqserver-wedged.log").write_text(
                "".join(f"attempt {i + 1}: no reply to the predict in 3.0s\n"
                        for i in range(wedges)))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            NESTED_SAMPLING_DIR = Path(tmp)

            # The real `ps`, because its output format is what this parses
            # and BSD and GNU differ. This interpreter is in its own table,
            # alive, and holding more than a megabyte - so a column that moved
            # or stopped being a number fails here rather than silently
            # reporting a run as holding nothing.
            table = process_table()
            mine = [p for p in table if p["pid"] == os.getpid()]
            assert len(mine) == 1, f"{len(mine)} rows for pid {os.getpid()}"
            assert mine[0]["alive"] and int(mine[0]["rss_mb"]) > 0, mine

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
                      "cpu_seconds": 20.0, "rss_mb": 10,
                      "args": f"python3 polychord_r2d2.py --output-dir {live.resolve()}"}
                     for i in range(4)]

            # mpirun and the host-side `docker exec` both carry the whole rank
            # command line in their own arguments; neither is a rank.
            not_ranks = [
                {"pid": 90, "alive": True, "elapsed_seconds": 100.0, "cpu_seconds": 1.0,
                 "rss_mb": 5,
                 "args": f"mpirun -np 4 python3 polychord_r2d2.py --output-dir {live.resolve()}"},
                {"pid": 91, "alive": True, "elapsed_seconds": 100.0, "cpu_seconds": 1.0,
                 "rss_mb": 5,
                 "args": f"/usr/bin/docker exec c mpirun python3 polychord_r2d2.py "
                         f"--output-dir {live.resolve()}"},
                # The sidecar's imager worker: not a rank, and where all of the
                # run's memory is. Named by --fifo-dir rather than
                # --output-dir, which is why the footprint is taken over
                # everything carrying the run directory rather than the ranks.
                {"pid": 92, "alive": True, "elapsed_seconds": 100.0, "cpu_seconds": 90.0,
                 "rss_mb": 3300,
                 "args": f"python3 r2d2_serve.py --fifo-dir {live.resolve()}/.r2d2-workers"},
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
            # ...and where it will next move to, so a reader coming back has
            # something to compare against. nlive is 4 in this fixture.
            assert "next at ~7" in aged, aged
            assert report["completed"] == 4 and report["in_flight"] == 1, report
            assert report["dead_points"] == 3, report
            assert report["ranks"] == 4 and report["failed"] == 0, report
            assert report["warnings"] == [], report
            # The imager worker is 3.3GB of the run and none of its ranks:
            # 4 ranks at 10MB, mpirun and docker exec at 5MB each, worker 3300.
            assert report["processes"] == 7, report
            assert report["resident_mb"] == 3350, report
            assert "3.3GB resident over 7 processes" in aged, aged
            # CPU is sampled over the same processes, so a busy worker shows as
            # a busy core even while every rank sits in a collective.
            with_cpu = io_capture(describe(live, not_ranks + ranks,
                                           DEFAULT_STALE_SECONDS, {92: 0.9}))
            assert "0.9 of " in with_cpu and "cores busy" in with_cpu, with_cpu

            # The sampler's own view: what PolyChord has found, and how much
            # of the search is left. nlive 50 and 200 dead points is four
            # e-folds of prior volume gone, so a sixteen-thousandth of it is
            # left; a flat likelihood puts the same fraction of the evidence
            # in the live points, and the run stops when that reaches
            # TERMINATION_EVIDENCE_RATIO. ln(1/1.2e-4) = 9.03 e-folds in all,
            # so 451 dead points and 200 of them done. Replayed against the
            # two searches that ran to natural termination here, the same
            # arithmetic forecast 452-459 from ndead=100 onward against
            # observed 446 and 463.
            fc = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260101T000100Z"
            (fc / "chains").mkdir(parents=True)
            (fc / "chains" / "w.resume").write_text("")
            (fc / "chains" / "w_dead-birth.txt").write_text("x\n" * 200)
            (fc / "chains" / "w.stats").write_text(
                "Global evidence:\n"
                "log(Z)       =   0.000000000000000E+000 +/-   0.201075706705112E-002\n"
                "log(Z_1)     =  -0.900000000000000E+001 +/-   0.1E-002 (Still Active)\n"
                " ndead:           200\n nlive:            50\n nlike:          4800\n")
            (fc / "chains" / "w_phys_live.txt").write_text(
                "  0.1  0.000000000000000E+000\n" * 50)
            (fc / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_MPI_PROCS=4\nNS_NLIVE=50\nNS_MAX_NDEAD=-1\n")
            for i in range(60):
                write_eval(fc, i + 1, now - 12000 + i * (12000 / 59))
            fc_ranks = [{"pid": 200 + i, "alive": True, "elapsed_seconds": 12000.0,
                         "cpu_seconds": 20.0, "rss_mb": 10,
                         "args": f"python3 polychord_wsclean.py --output-dir {fc.resolve()}"}
                        for i in range(4)]
            # The local `log(Z_1)` line sits below the global one and must not
            # be read instead of it.
            stats = sampler_stats(fc)
            assert stats == {"log_z": 0.0, "log_z_error": 0.00201075706705112,
                             "ndead": 200, "nlive": 50, "nlike": 4800}, stats
            report = describe(fc, fc_ranks, DEFAULT_STALE_SECONDS)
            ahead = report["forecast"]
            assert ahead["total_dead_points"] == 451, ahead
            assert ahead["estimated"] is True, ahead
            # 251 dead points left at the 200-per-12000s this run has managed.
            assert 4.1 < float(ahead["hours_remaining"]) < 4.3, ahead
            shown = io_capture(report)
            # 4800 calls over 200 dead points.
            assert "logZ = 0.000 +/- 0.002, 24 likelihood calls per dead point" in shown, shown
            assert "~44% done, ~451 dead points total, ~4h11m left" in shown, shown

            # An explicit --max-ndead is a hard stop, so the count is known
            # rather than estimated and prints without the tildes.
            (fc / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_MPI_PROCS=4\nNS_NLIVE=50\nNS_MAX_NDEAD=300\n")
            capped = describe(fc, fc_ranks, DEFAULT_STALE_SECONDS)
            assert capped["forecast"]["total_dead_points"] == 300, capped["forecast"]
            assert capped["forecast"]["estimated"] is False, capped["forecast"]
            assert "67% done, 300 dead points total, 1h40m left" in io_capture(capped)

            # Inside the first e-fold the live set is still the prior, so the
            # estimate would be reporting its own constant and not this run.
            (fc / "chains" / "w.stats").write_text(
                "log(Z)       =   0.000000000000000E+000 +/-   0.2E-002\n"
                " ndead:            20\n nlive:            50\n nlike:           480\n")
            (fc / "run.env").write_text(
                "NS_ALGORITHM=wsclean\nNS_MPI_PROCS=4\nNS_NLIVE=50\nNS_MAX_NDEAD=-1\n")
            early = describe(fc, fc_ranks, DEFAULT_STALE_SECONDS)
            assert early["forecast"] is None, early["forecast"]
            # ...but the evidence it has is still worth showing.
            assert "logZ = 0.000" in io_capture(early)

            # A run with no chains at all says nothing rather than guessing.
            assert sampler_stats(live) is None, sampler_stats(live)
            assert live_loglikelihoods(live) == []
            assert describe(live, ranks, DEFAULT_STALE_SECONDS)["forecast"] is None

            # Disk: the one resource nothing here reserves, frees or prunes,
            # and the only one whose exhaustion ends a run outright.
            bulky = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T000200Z"
            (bulky / "evaluations").mkdir(parents=True)
            (bulky / "run.env").write_text("NS_ALGORITHM=r2d2\n")
            # Sizes that shrink over the run, the way a nested-sampling run's
            # do as it concentrates: sampling the tail would read 8KB where the
            # run averages 68KB, which is why the sample is strided.
            for i in range(40):
                eval_dir = bulky / "evaluations" / f"eval-{i:04d}-abc"
                # A measurement set is a directory, so a walk that does not
                # recurse reports an evaluation as costing a few hundred bytes.
                (eval_dir / "point.ms").mkdir(parents=True)
                (eval_dir / "point.ms" / "table.dat").write_bytes(
                    b"\0" * ((128 if i < 20 else 8) * 1024))
                metrics = eval_dir / "metrics.json"
                metrics.write_text(json.dumps({"eval_id": i, "objective": 0.008}))
                landed = now - 3600 + i * 60
                os.utime(metrics, (landed, landed))
            bulky_ranks = [{"pid": 200, "alive": True, "elapsed_seconds": 100.0,
                            "cpu_seconds": 20.0, "rss_mb": 10,
                            "args": f"python3 polychord_r2d2.py "
                                    f"--output-dir {bulky.resolve()}"}]
            report = describe(bulky, bulky_ranks, DEFAULT_STALE_SECONDS)
            truth = _dir_bytes(bulky / "evaluations")
            assert truth > 2 * 1024 ** 2, truth  # the payload, or the walk missed it
            estimate = float(report["disk_bytes"])
            assert abs(estimate - truth) < 0.1 * truth, (estimate, truth)

            # Free space is real and enormous on this host, so the projection
            # is exercised against a stub: the arithmetic is what is under
            # test, not statvfs. Four hours left, which is under the warning.
            per_hour = float(report["disk_bytes_per_hour"])
            real_free = globals()["free_bytes"]
            try:
                globals()["free_bytes"] = lambda _p: (int(per_hour * 4), 400 * 1024 ** 3)
                short = describe(bulky, bulky_ranks, DEFAULT_STALE_SECONDS)
                globals()["free_bytes"] = lambda _p: (500 * 1024 ** 3, 900 * 1024 ** 3)
                roomy = describe(bulky, bulky_ranks, DEFAULT_STALE_SECONDS)
                # No ranks and a stale mtime: a stopped run's rate is history,
                # so it reports what it wrote and projects nothing.
                globals()["free_bytes"] = lambda _p: (int(per_hour * 4), 400 * 1024 ** 3)
                done = describe(bulky, [], DEFAULT_STALE_SECONDS)
            finally:
                globals()["free_bytes"] = real_free
            assert 3.9 < float(short["disk_hours_remaining"]) < 4.1, short
            assert any("ENOSPC" in w for w in short["warnings"]), short["warnings"]
            assert "of space left at that rate" in io_capture(short)
            assert not any("ENOSPC" in w for w in roomy["warnings"]), roomy["warnings"]
            assert done["status"] == "stopped" and done["disk_hours_remaining"] is None, done
            ended = io_capture(done)
            assert "of space left" not in ended, ended
            # Megabytes, not "0.0GB": the fixture is 2.7MB, and so is a run ten
            # minutes old.
            assert "MB written" in ended, ended

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

            # A run that crashed and restarted itself is still healthy, but
            # the restarts must be visible: whatever killed it once will do it
            # again, and nothing else on disk records that it happened.
            assert "restarts" not in io_capture(describe(live, ranks, 5.0))
            (live / "restarts.log").write_text(
                "2026-08-28T09:00:00Z exit 1 after 40 dead points\n"
                "2026-08-28T11:00:00Z exit 1 after 91 dead points\n"
            )
            healed = describe(live, ranks, 5.0)
            assert len(healed["restarts"]) == 2, healed["restarts"]
            # Reported, never warned on - a self-healed run is fine right now,
            # and warning would make `./ri health` exit nonzero for one.
            assert not any("restart" in w for w in healed["warnings"]), healed["warnings"]
            shown = io_capture(healed)
            assert ("restarts  2 self-healed restarts, last 2026-08-28T11:00:00Z "
                    "exit 1 after 91 dead points") in shown, shown
            (live / "restarts.log").unlink()

            # Same run, same files, no ranks: a stale mtime is only a stall
            # while something is still running.
            assert describe(live, [], 5.0)["status"] == "stopped"
            assert describe(live, ranks, 5.0)["status"] == "stalled"
            # A stalled run's count can still move; a stopped one's cannot, so
            # only the first may say where it will move to.
            assert "next at ~7" in io_capture(describe(live, ranks, 5.0))
            assert "next at" not in io_capture(describe(live, [], 5.0))
            # ...and once it finishes, neither is true however old it gets.
            (live / "summary.json").write_text("{}")
            assert describe(live, [], 5.0)["status"] == "finished"
            assert describe(live, [], 5.0)["warnings"] == []
            assert "next at" not in io_capture(describe(live, [], 5.0))
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
            assert abs(float(report["slowdown_factor"]) - 12) < 1, report
            assert abs(float(report["recent_evals_per_minute"]) - 5) < 0.5, report
            # Measured and shown, never warned on: a run that halves its pace
            # and recovers is a phase, and warning on it would teach the reader
            # to ignore the warnings that mean something.
            assert not any("throughput" in w for w in report["warnings"]), report
            # The imager never changed - 12s an evaluation from first to last -
            # so the 12-fold drop in arrivals is idle ranks, and the occupancy
            # is what says so: 12 of 12 ranks kept busy over the run's life,
            # one of them over its last 50. Reported, not warned on, because
            # every wsclean run on this host ends its last 50 near 23% simply
            # by shutting down.

            # The same collapse seen from the imager's side, which is where the
            # two explanations for a falling arrival rate come apart. Eight
            # ranks; 150 evaluations 2s apart costing the imager 20s each, then
            # 60 evaluations 10s apart costing 5s each - so the imager got four
            # times *faster* while arrivals got five times slower, and the only
            # reading that fits is ranks going idle. This is the live R2D2
            # search's real shape (25.4s at full occupancy over its life, 12.3s
            # at 6% over its last 50), not an invented one.
            idle = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T025900Z"
            (idle / "chains").mkdir(parents=True)
            (idle / "chains" / "r2d2_vlaa.resume").write_text("")
            (idle / "run.env").write_text("NS_MPI_PROCS=8\n")
            stamp = now - (150 * 2 + 60 * 10) - 5
            for i in range(150):
                stamp += 2
                # One 1s outlier, so the cost is a median and not whatever the
                # first or the cheapest evaluation happened to be.
                write_eval(idle, i + 1, stamp, wall_seconds=1.0 if i == 0 else 20.0)
            for i in range(60):
                stamp += 10
                write_eval(idle, 151 + i, stamp, wall_seconds=5.0)
            serial = describe(idle, [], DEFAULT_STALE_SECONDS)
            assert float(serial["seconds_per_evaluation"]) == 20.0, serial
            assert float(serial["recent_seconds_per_evaluation"]) == 5.0, serial
            # 3281 imaging seconds banked over 898 of wall clock is 3.65 of the
            # 8 ranks kept busy across a life that was full for its first third
            # and idle after; the last 50 evaluations are all in the idle phase
            # at 250s over 490, half a rank. A ratio of the two medians would
            # have said 125% and 50% - see _duty.
            assert abs(float(serial["busy_ranks"]) - 3.654) < 0.01, serial
            assert abs(float(serial["recent_busy_ranks"]) - 0.510) < 0.01, serial
            rendered_idle = io_capture(serial)
            assert "20.0s per evaluation, ranks 46% busy" in rendered_idle, rendered_idle
            assert "(last 50: 5.0s, 6% busy)" in rendered_idle, rendered_idle
            # The same run as a shape. Its slices are 44.9s, so an early one
            # banks 22 evaluations at 20s - more imaging seconds than 8 ranks
            # had to spend, hence the clamp - and a late one banks 4 at 5s.
            used = serial["occupancy"]
            assert used is not None, serial
            assert used["bar"][0] == HISTORY_LEVELS[-1], used
            assert used["bar"][-1] == HISTORY_LEVELS[0], used
            assert abs(float(used["high_fraction"]) - 1.0) < 0.01, used
            assert abs(float(used["low_fraction"]) - 0.0625) < 0.01, used
            assert "6%-100% of 8 ranks busy per 0:00:44 slice" in rendered_idle, rendered_idle
            # Absolute scale, unlike `history` above: a run that spent its
            # whole life at half occupancy must draw half-height bars, not the
            # full ones a peak-relative scale would give it.
            half = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T025700Z"
            (half / "chains").mkdir(parents=True)
            (half / "run.env").write_text("NS_MPI_PROCS=8\n")
            for i in range(200):
                write_eval(half, i + 1, now - 3600 + i * 2, wall_seconds=8.0)
            steady = describe(half, [], DEFAULT_STALE_SECONDS)["occupancy"]
            assert set(steady["bar"]) == {HISTORY_LEVELS[4]}, steady
            assert abs(float(steady["high_fraction"]) - 0.5) < 0.01, steady
            # A run whose rank count is not recorded has no denominator, so it
            # gets no shape rather than one drawn against a guess.
            (half / "run.env").write_text("")
            assert describe(half, [], DEFAULT_STALE_SECONDS)["occupancy"] is None

            # A run whose evaluations all landed in the same millisecond has no
            # occupancy to report, for the same reason it has no rate: the gap
            # median is zero and the ratio would be a division by noise.
            burst = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T025800Z"
            (burst / "chains").mkdir(parents=True)
            for i in range(100):
                write_eval(burst, i + 1, now - 3600 + i * 0.005, wall_seconds=13.0)
            flat = describe(burst, [], DEFAULT_STALE_SECONDS)
            assert flat["busy_ranks"] is None, flat
            assert "% busy" not in io_capture(flat), io_capture(flat)
            rendered = io_capture(report)
            assert "5.0/min over the last 50" in rendered, rendered
            # The same collapse as a shape: the healthy phase fills the early
            # slices to the peak and the collapse empties the late ones, which
            # is the difference the two medians cannot show.
            past = report["history"]
            assert past is not None, report
            assert past["bar"][0] == HISTORY_LEVELS[-1], past
            assert past["bar"][-1] in HISTORY_LEVELS[:2] + HISTORY_EMPTY, past
            assert len(past["bar"]) == HISTORY_BUCKETS, past
            assert past["high_per_minute"] > 10 * past["low_per_minute"], past
            assert "/min per" in rendered, rendered
            # A steady run draws a flat line rather than a false trend: equal
            # counts must all land on the same level, whatever the peak is.
            flat = history([now + i for i in range(4 * HISTORY_BUCKETS)])
            assert set(flat["bar"]) == {HISTORY_LEVELS[-1]}, flat
            # A slice where nothing landed is marked, not rounded down to
            # "slow" - a hole in the run is exactly what a reader is scanning
            # this line for.
            holed = history([now + i * 0.1 for i in range(3 * HISTORY_BUCKETS)]
                            + [now + 1000])
            assert HISTORY_EMPTY in holed["bar"], holed
            # Too few evaluations to be a shape rather than noise.
            assert history([now + i for i in range(HISTORY_BUCKETS)]) is None
            assert history([now] * 100) is None  # no span, no bins

            # A run killed inside its opening batch: the ranks all landed
            # together, so the span is mtime granularity and every rate derived
            # from it is arithmetic on noise. Real runs here printed
            # "6176.5/min over 0:00:00" and a "0-8700112/min" sparkline off
            # exactly this shape, which discredits the honest numbers beside
            # them. Neither line is printed rather than printed wrong.
            batch = NESTED_SAMPLING_DIR / "wsclean-vlaa-20260101T040000Z"
            (batch / "chains").mkdir(parents=True)
            for i in range(4 * HISTORY_BUCKETS):
                write_eval(batch, i + 1, now - 3600 + i * 0.005)
            burst = describe(batch, [], DEFAULT_STALE_SECONDS)
            assert burst["completed"] == 4 * HISTORY_BUCKETS, burst
            assert burst["evals_per_minute"] is None, burst
            assert burst["history"] is None, burst
            assert "/min" not in io_capture(burst), io_capture(burst)
            # ...and one second later the same run does report both, so the
            # floor is a floor and not a way to lose a fast run's numbers.
            for i in range(4 * HISTORY_BUCKETS):
                write_eval(batch, 1000 + i, now - 3600 + i * 1.0)
            fast = describe(batch, [], DEFAULT_STALE_SECONDS)
            assert fast["evals_per_minute"] is not None, fast
            assert fast["history"] is not None, fast

            # Directories with no metrics.json are "in flight" only while
            # something is still flying them; on a dead run they are what its
            # ranks were holding when it died.
            assert "0 abandoned" in io_capture(burst), io_capture(burst)
            assert "in flight" in io_capture(describe(live, ranks, 5.0))

            # An imager that broke part-way through a long healthy run: 400
            # good evaluations then 50 failures is 11% overall, well under the
            # whole-run bar, while every point it is adding now is a failure.
            broke = NESTED_SAMPLING_DIR / "r2d2-vlaa-20260101T030000Z"
            (broke / "chains").mkdir(parents=True)
            for i in range(400):
                write_eval(broke, i + 1, now - 500 + i)
            for i in range(RATE_WINDOW):
                write_eval(broke, 401 + i, now - 100 + i, objective=100.0)
            report = describe(broke, [], DEFAULT_STALE_SECONDS)
            assert report["failed"] == 50 and report["completed"] == 450, report
            assert report["recent_failed"] == RATE_WINDOW, report
            assert any("broke part-way through" in w for w in report["warnings"]), report
            assert "(50 of the last 50)" in io_capture(report), io_capture(report)
            # ...and the run before it broke says nothing, so the warning is
            # the break rather than the presence of any failure at all.
            healthy_window = describe(collapsed, [], DEFAULT_STALE_SECONDS)
            assert healthy_window["recent_failed"] == 0, healthy_window
            assert not any("broke part-way" in w for w in healthy_window["warnings"])
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
            # Three evaluations is not a window: no recent ratio at all, rather
            # than one drawn from a handful.
            assert report["recent_failed"] is None, report
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

            # The same hole, now explained: a restart landing inside it means
            # the run was down, not stalled, and the `restarts` line already
            # says so. Written in UTC as progress-bar.sh writes it, so reading
            # it as local time here would miss the gap on any host east or
            # west of Greenwich.
            def _restart_at(when: float) -> None:
                stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))
                (stalled / "restarts.log").write_text(
                    f"{stamp} exit 137 after 25 dead points\n")

            _restart_at(now - 20)
            healed = describe(stalled, [], DEFAULT_STALE_SECONDS)
            assert healed["stall_count"] == 0, healed
            assert healed["stall_seconds"] == 0, healed
            assert not any("of wall clock lost" in w for w in healed["warnings"]), healed
            # And a restart outside the gap excuses nothing - otherwise any
            # restarts.log at all would silence the check for the rest of the run.
            _restart_at(now - 5)
            assert describe(stalled, [], DEFAULT_STALE_SECONDS)["stall_count"] == 1
            (stalled / "restarts.log").unlink()

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
                          "cpu_seconds": 0.0, "rss_mb": 0, "args": "sh"}]
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
            # But a run with ranks outranks every newer one that has none:
            # `live` is the oldest directory here and the only one running, and
            # it is the run the report exists to be about. `ranks` carry its
            # --output-dir; `processes` (a bare `sh`) carry nobody's.
            assert resolve(None, ranks).name == live.name
            assert resolve(None, processes).name == newest.name
            # Ranks, not every process carrying the run directory: a killed
            # run's sidecar workers outlive it holding ~3.4GB each, and
            # defaulting to a dead run because it leaked containers would show
            # the wrong run for exactly as long as nobody reaped them.
            leftover = [p for p in not_ranks if "r2d2_serve.py" in str(p["args"])]
            assert leftover and resolve(None, leftover).name == newest.name
            # Two runs going at once - the shared host this is written for -
            # and the newer of them wins, so "newest" still means newest.
            also_live = [dict(ranks[0], pid=777,
                              args=f"python3 polychord_r2d2.py --output-dir {newest.resolve()}")]
            assert resolve(None, ranks + also_live).name == newest.name
            # An explicit run still wins over both.
            assert resolve(fast.name, ranks).name == fast.name
            newest.rmdir()

            # And the whole thing renders and scores. Into a sink, because what
            # is checked is that both forms run and reach the right exit status,
            # not what they print. The host is stubbed healthy: whether the
            # machine running the check happens to be short of memory or to be
            # holding a leaked container is not what is under test.
            import contextlib
            import io

            global meminfo_mb
            original_memory, original_table = meminfo_mb, process_table
            original_cpu = cpu_busy_fractions
            try:
                sidecar_containers = lambda: []  # noqa: E731
                meminfo_mb = lambda key: HEADROOM_MB * 2  # noqa: E731
                with contextlib.redirect_stdout(io.StringIO()):
                    assert main(["--all", "--json"]) == 1
                    assert main([live.name]) == 0
                # ...and no-argument `./ri health` reaches the running run,
                # not the newest directory: `fast` is newer and finished.
                process_table = lambda: ranks
                # ...and no CPU sample for four pids that do not exist: the
                # real one sleeps a second to take it.
                cpu_busy_fractions = lambda pids: {}
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    main([])
                assert sink.getvalue().startswith(live.name), sink.getvalue()
            finally:
                sidecar_containers = original
                meminfo_mb = original_memory
                process_table = original_table
                cpu_busy_fractions = original_cpu
    finally:
        NESTED_SAMPLING_DIR = saved
    print("nested-sampling-health self-check passed")


if __name__ == "__main__":
    if os.environ.get("NESTED_SAMPLING_HEALTH_SELF_CHECK") == "1":
        self_check()
    else:
        sys.exit(main())
