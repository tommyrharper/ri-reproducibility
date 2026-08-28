# Run health

`./ri runs` answers "did it finish?". `./ri health` answers the question you
have while one is still going. See [nested-sampling.md](nested-sampling.md) for
how to start a run, and [robustness.md](robustness.md) for what happens when
one breaks.

```console
$ ./ri health
r2d2-vlaa-20260827T205418Z  r2d2  HEALTHY
  stage     sampling, 113 dead points as of 8m18s ago, next at 163+
  at risk   184 evaluations scored since that checkpoint, 8m18s of imaging a restart would redo
  progress  1287 evaluations, 15 in flight
  activity  last evaluation 2s ago, 27.3/min over 47m06s
  history   █▆▆▇▇▇█▆▂▆█▆█▄█▆█▅█▇  5-34/min per 7m29s slice
  imaging   25.4s per evaluation, ranks 66% busy  (last 50: 12.3s, 6% busy)
  occupancy ▇▆▆▇▇▂▆▆▇▆▆▆▇▆█▆▇▂▁▆  6%-88% of 16 ranks busy per 7m29s slice
  sampler   logZ = -0.044 +/- 0.012, 24 likelihood calls per dead point
  forecast  ~26% done, ~120 of ~454 dead points, ~2h11m left (~14:37)
  ranks     16 ranks of 16, 7 busy-waiting
  resources 48.9GB resident (+4.5GB swapped out) over 51 processes, 16.0 of 20 cores busy
  memory    3.3GB peak imager memory, 53.2GB across 16 ranks
  disk      7.5GB written, +2.6GB/hour, 93h of space left at that rate
  failures  0 scored FAILURE_OBJECTIVE, 0 meqserver wedges recovered
  stalls    8 gaps over 13s, 154s = 5.5% of running time

host
  memory    8.9GB available of 62.6GB, 4GB reserved as headroom
  swap      5.1GB of 32.0GB used
  load      16.7 / 17.0 / 17.3 against 20 cores (1m / 5m / 15m)
  pressure  memory 0.0% / 0.0%, io 0.0% / 0.0% of wall clock stalled (1m / 5m)
  disk      233GB free of 436GB
  sidecars  3 running, 0 leaked
```

## Which runs it reports on

With no argument, every run being driven anywhere on this host - ranks running,
or a `docker exec` client still starting them - falling back to the newest run
when nothing is going. All of them, not the newest, because memory caps a run
here and the host is shared: a second search is the usual reason the first is
slow.

Live runs are found in the host process list rather than by globbing this
checkout, so a run from another worktree is reported too, named by a `path`
line. `./ri health <name>` takes any name the report prints: a bare name is
looked for in this checkout first, then among what is running on the host.
Commands it *suggests* run the other way - `./ri resume` takes a bare name only
under this checkout, so warnings about a foreign run give it a path.

`--all` covers every run on disk under this checkout; `--json` is the
machine-readable form.

It reads files and runs one `ps` and one `docker ps`, plus a one second CPU
sample when a run has live ranks. Nothing is started and nothing is imaged, so
a live run does not notice it. Exit status is 1 when something needs attention.

## Status

Decided in this order, because a run that finished and a run that died both
stop writing:

| | |
|---|---|
| `FINISHED` | A whole `summary.json` is there. |
| `STALLED` | Ranks running, but no evaluation in `--stale-seconds` (default 600). The warning says how long until the stall watchdog restarts the run, or that `--stall-timeout 0` turned it off, or that no watchdog is left. |
| `STARTING` | No ranks yet, but a `docker exec` client is alive. An R2D2 search spends minutes here loading models. |
| `STOPPED` | No ranks and no client, however recently it wrote. `./ri resume <run>` continues it from its checkpoint, or starts the sampler over if there is none; the warning says which. A half-written `summary.json` lands here too, with a warning saying so. |
| `HEALTHY` | Ranks running, evaluations landing, nothing warned about. |

A directory with none of `run.env`, `run.log`, `summary.json`, `evaluations/`
or `chains/` is `Not a nested-sampling run` rather than a status - every status
above is decided by an absence, which is also what an unrelated directory looks
like. `./ri runs` skips the same directories.

## Warnings

The headline carries the warning count, because under `--all` it is the only
line worth scanning:

```console
$ ./ri health --all
wsclean-vlaa-20260828T022337Z  wsclean  FINISHED
wsclean-vlaa-20260828T022151Z  wsclean  STOPPED - 1 WARNING
r2d2-vlaa-20260827T205418Z  r2d2  RUNNING - 1 WARNING
```

`HEALTHY` is the only status word that is a claim rather than a lifecycle
point, so it stands down to `RUNNING` when there is something to say. No suffix
and no host warning is exactly exit 0.

On a terminal the headline and `WARNING` labels are coloured - green for
`HEALTHY`/`FINISHED`, cyan for `STARTING`, amber for warnings, red for
`STALLED`/`STOPPED` - and nothing else is. Piped, redirected or under
[`NO_COLOR`](https://no-color.org) the output is byte-identical.

## Fields

**stage** - how far into PolyChord the run got, from `chains/`: `*.resume`
means it reached the main loop, `*_phys_live.txt` alone means it is still
drawing initial live points.

The dead-point count never appears without its age, because PolyChord writes
the checkpoint only every ~`nlive` points and the count cannot move between
writes. A count that has not changed for an hour is ordinary, and the interval
grows as a run goes on (one 16-rank R2D2 search: 31 minutes to its first
checkpoint, 72 more to its second). `next at 163+` rather than `~163` because
`nlive` is a floor on the interval, not an estimate: measured gaps ran 22-36
dead points at `--nlive 20` and 51-92 at `--nlive 50`, never under. `next at`
appears only while the run is going.

**at risk** - what standing behind that checkpoint would cost: evaluations
scored since it, and their imaging time. A restart or `./ri resume` picks up at
the checkpoint's dead points and images its way back over different proposals,
so none of that work is reused. (Measured: a WSClean search killed after its
checkpoint and resumed shared the 122 evaluations scored before the kill with
an uninterrupted control, and none of the 146 after.) Reported, never warned
on - it only grows until the next checkpoint and there is nothing to do about
it. A finished run does not print it.

**progress** - `eval-*/metrics.json` is written only when an evaluation
succeeds, so its count is the progress and directories without one are in
flight. That number should sit near `NS_MPI_PROCS`; pinned there while the
count does not move is every rank stuck at once. On a run with no ranks left
they are *abandoned* instead.

**activity** - the overall rate, and the rate over the last 50 evaluations when
the two diverge. Both are evaluations over elapsed time, which is the only way
to read one against the other. A run can collapse to a fraction of its
throughput without ever going quiet enough to look stalled: on a live 16-rank
R2D2 search, 25/min fell to 5/min for ten minutes while evaluations kept
landing every 20-30s. Neither rate is warned on - that same run recovered to
37/min with nothing done to it.

Both rates, and the occupancy in **imaging**, are measured over the time the
run was *running*, not its age. Downtime from `restarts.log` comes out of the
span and is named: `213.6/min over 17s + 4h16m stopped`. **history** and
**occupancy** stay on wall clock, because showing the stop is their job.

Neither rate is shown when the run's first and last evaluation are under a
second apart: parallel ranks land their opening batch together, so a run killed
inside it measures mtime granularity rather than throughput. The same floor
silences **history**.

A falling rate does not mean the evaluations got harder - **imaging** answers
that. Whether it means stragglers is a third question, answered by the *spread*
of `metrics.json` `timing.image_container_seconds`: a fat tail is one slow
evaluation gating a batch, a tight distribution means the wall clock is going
into sampler overhead or synchronisation.

**history** - the same throughput binned into twenty equal slices of the run's
life, scaled to its own peak. Numbers cannot show *shape*: a dip that recovered
and a step down that did not read identically. A slice where nothing landed is
`·` rather than merely slow. Binned first-to-last evaluation, never up to now,
so no slice is partial.

**imaging** - what one evaluation costs the imager (median
`timing.image_container_seconds`, already in memory, so no extra I/O), and how
much of the run's hardware that cost keeps busy. The arrival rate cannot tell a
slower imager from idle ranks; both read as fewer evaluations a minute.

Occupancy is imaging seconds banked per rank-second of wall clock - a duty
cycle, not a ratio of two medians, which read systematically high (a clamped
"100% busy" over a life its own slices put at 6-88%). Clamped at 100%, because
an evaluation is banked when it finished while its cost was spent before that.
Both figures are shown over the last 50 evaluations when either moved
materially: that is how the live R2D2 search's 5-fold slowdown was diagnosed -
25.4s at 66% over its life against 12.3s at 6% over its last 50, so the imager
had got *faster* while fifteen of sixteen ranks went idle. Reported, never
warned on: every WSClean run ends near 23% simply by shutting down.

**occupancy** - that duty cycle binned into **history**'s twenty slices, which
is the only line saying whether the hardware has been earning its keep all
along. The scale is absolute rather than peak-relative, because a duty cycle
has a natural full: a solid bar is every rank imaging, a bar that never leaves
the floor is a run that should have had fewer ranks or a larger `--nlive`.
Withheld when `run.env` has no `NS_MPI_PROCS`.

**sampler** - PolyChord's own running total from `chains/*.stats`. Every other
line is operational; this is the number the search exists to produce, and
`logZ` moving is the only direct evidence the sampler is integrating rather
than merely running. Likelihood calls per dead point is what the evaluation
rate is being *spent* on: a run whose rate holds while this climbs is working
just as hard for less.

**forecast** - how far through, and how long left. With `--max-ndead -1` there
is otherwise no denominator anywhere. Each dead point shrinks the prior volume
by the same factor, so `exp(-ndead/nlive)` is what is left; the evidence still
to come is that volume times the mean likelihood of the live points in it, and
PolyChord stops when that falls to a fixed fraction of the evidence banked.

The wait is printed as a duration and as the clock time it ends at (`~2h11m
left (~14:37)`, gaining a day name once it is not today).

The position does not wait for the next checkpoint. `ndead` is frozen between
writes, so everything derived from it sits still and then jumps. The evaluation
directories are not frozen and the sampler spends a near-constant number per
dead point, so evaluations banked since the checkpoint convert back into dead
points. Only the position is carried; the total still comes from the
checkpoint. A carried count can walk past an estimated total, and the run says
so rather than claiming 100%:

```
forecast  past its ~452 dead-point estimate, set by the checkpoint 3h37m ago and revised by the next one
```

An explicit `--max-ndead` is a hard stop the sampler honours, so it is used
directly and printed without the `~`.

The stopping fraction is measured, not taken from the docs.
`precision_criterion` defaults to 1e-3, but the two searches here that ran to
natural termination stopped at 446 and 463 dead points where 1e-3 predicts 350.
`TERMINATION_EVIDENCE_RATIO` is calibrated to their mean and forecasts 452-459
from `ndead=100` onward - within 3% of both, and stable rather than drifting.
It holds at a very different `nlive` too (a `--nlive 5` search forecast 45-47
from 12% onward and terminated at 47). Recalibrate there, and in
`_NS_TERMINATION_EVIDENCE_RATIO` in `scripts/lib/progress-bar.sh`, whose pinned
status line forecasts from the same model; a self-check fails if the two drift.

Withheld inside the first e-fold, where the live set is still the prior, and
from a run that is not going, whose remaining dead points are not remaining.

**ranks** - found by the `--output-dir` they were launched with, so no ranks
means no run. `busy-waiting` counts ranks that spent a whole one-second sample
on CPU: Open MPI's `ob1` busy-waits, so a rank blocked in a collective burns a
core and looks identical to a working one on `%CPU`. Cumulative CPU cannot
separate them - a run that works for an hour then wedges reads under any
threshold for most of another hour.

On its own the count means nothing: four measurements of one healthy 16-rank
run gave 1, 2, 7 and 15 as the sampler alternated between imaging and
synchronising, and each was reproducible for as long as its phase lasted. So a
deadlock is reported only when all but one rank are burning CPU **and** nothing
has completed for a minute - the second clause does the work.

**resources** - what the run actually holds, so the host's free memory has an
owner. Taken over every live process carrying the run directory, not just the
ranks: a rank is ~10MB and the imager worker it talks to is ~3.3GB, and workers
name the run by `--fifo-dir`. This tool's own process tree is excluded, or a
run named by path reports the report measuring itself. RSS, so a shared page is
counted once per holder - overcounting, the safe direction for "will another
run fit".

Swap is shown beside it because RSS excludes it, so a squeezed run reads as
holding *less* than it does. A process is warned about when more of it is in
swap than in memory **and** at least 200MB is out there (`PAGED_OUT_MB`, a
whole WSClean rank). Both clauses are needed: healthy R2D2 workers keep ~70MB
of cold startup pages swapped against a 3.2GB footprint, while ranks at 10MB
resident against 14MB swapped are "mostly on disk" by ratio alone. The one
process worth naming was an imager worker at 52MB resident against 2.9GB
swapped - parked, and invisible in every other number on the page.

That warning is held back unless **pressure** says something is actually
waiting on those pages. Being on disk and being read back off it are different
things, and only the second costs anything.

**pressure** - the kernel's Pressure Stall Information (`some avgN` from
`/proc/pressure/memory` and `/proc/pressure/io`): the percentage of the last
minute and last five during which at least one task was stalled on that
resource. Every other resource number says how large a shortage is; this is the
only one saying what it *costs*, which is why the paged-out warning and the
host memory warning are both decided by it. At or above `MEMORY_STALL_PERCENT`
(5%) the host is short of RAM and the answer is fewer ranks, not a faster
imager. Omitted on a host with no PSI (macOS, or a kernel before 4.20), and
every decision resting on it falls back to its unconditional form.

The host block reports swap in use but never warns on it: swap in use may have
been paged out days ago and cost nothing since.

**load** - `/proc/loadavg` over one, five and fifteen minutes against the core
count. The only CPU reading covering work this project did not start, which is
what answers "my run got slower and every other number looks fine". Three
windows because the trend is the readable part. Never warned on: this host is
deliberately run at every core busy, so any load-against-cores rule would fire
on exactly the runs it is for. Same reason cpu PSI is not read.

**memory** - the same cost measured by the run itself: median
`peak_memory_bytes` over its evaluations, multiplied out over `NS_MPI_PROCS`,
because the host has to hold every rank's worker at once. Two things
**resources** cannot do. It survives the run - a finished or OOM-killed run has
no processes left to sample. And it is the continuous re-measurement behind
`scripts/lib/rank-budget.sh`, which sizes every run from a fixed 3500MB per
R2D2 rank and 200MB per WSClean rank: the live search reads 3.3GB against 3.42GB
budgeted, so the estimate is 3% conservative.

Not "per evaluation": WSClean's figure is `time -v` on one imaging run, R2D2's
is the warm worker's high-water RSS over the rank's whole life. Both answer
"what does one rank have to be budgeted"; neither is an average. The last-50
figure prints only on a move of more than 2x, because footprint is flat here
(3.45-3.57GB across 6,600 R2D2 evaluations) so a second agreeing number is
noise while a doubling is a leak the budgeter cannot see coming.

**disk** - the resource nothing reserves, checks or frees, and the only one
that only grows. An evaluation directory keeps ~1.7MB and nothing deletes it
(`./ri clean` leaves `results/` alone), so a live R2D2 run writes ~2.6GB/hour.
The projection is that rate against what the filesystem has left; nothing else
would say so before the run ends on ENOSPC.

It warns when the projection is shorter than **how much longer the run needs**,
not under a fixed number of hours - space running out after the search is over
is not a problem the run has. That is **forecast**'s hours left when there is
one, and otherwise the run's own age (a run is assumed to have at least as long
ahead as behind). The warning says which it used.

Estimated from a strided sample of 20 evaluations, not a walk: `du -s` cost
3-5s of I/O against the disk the run is using, while 20 stats cost milliseconds
and landed within 1%. Strided over the run's life rather than its tail, because
a nested-sampling run concentrates and the newest 20 read 1.45MB where the run
averaged 1.68MB.

**restarts** - how many times the run died and restarted itself from its
checkpoint, and when, from `restarts.log`. Shown only when there were any and
never warned on - the crash was survived - but it is the line to read first on
a run that looks slower than it should. See [robustness.md](robustness.md).

For a run still going it also says what is left of the budget - `; 1 of 2
left`, or `; 2 of 2 used, so the next crash stops the run until someone resumes
it`. Reconstructed from `restarts.log` by replaying the retry loop's two rules:
an attempt that ran `NS_RETRY_RESET_SECONDS` (1800) before dying hands the
budget back, and a `./ri resume` starts a fresh loop at zero. Silent when
`run.env` has no `NS_RETRIES`, and withheld from a finished or stopped run
where "0 left" would read as the reason it stopped.

**resumes** - the other half of the same file: a `./ri resume` someone typed,
written as `<stamp> resumed at N evaluations` where a self-healed restart
writes `<stamp> exit N after M evaluations`. Its own line, because a run that
healed itself and a run a human continued did different things and only the
first says "this will happen again". Both are downtime - see **activity**.

**supervision** - a warning when the shell that started the run is gone.
SIGKILLing a run script does not stop the run: the ranks are children of
`containerd-shim`, so they keep imaging and every other line stays healthy.
What dies with the shell is `run_with_retries`, so the run has quietly lost
restart-from-checkpoint and will end at the first crash it would have survived.
Nothing else on disk or in the process table shows this. Found through the
run's `docker exec` client, whose parent is checked for *being* a run script
rather than for being pid 1, because a reparented orphan lands on whatever
subreaper the session has.

**failures** - evaluations that scored `FAILURE_OBJECTIVE` (100.0), and
`meqserver-wedged.log` lines. **The one a run can pass every other check and
still fail.** PolyChord maximizes and a real `total_rms_jy` is ~0.008, so
failed evaluations are the best points the search has ever seen and it
concentrates its live points on them. A missing checkpoint mount or an
OOM-killed worker reports "the imager fails catastrophically here", which is
exactly the conclusion this repo exists to draw.

Asked of the last 50 evaluations as well as the whole run, because a whole-run
ratio cannot see an imager that broke part-way through: three hours healthy and
twenty minutes broken is ~2% overall and silent. Half the window is the bar,
which needs no tuning - across 37,000 evaluations of six real runs here the
failure count is zero, so any sustained burst is a fault.

**stalls** - gaps between evaluations more than 10x the run's own median, never
less than 2s. Relative, because WSClean lands 30-50 evaluations a second and
R2D2 one every two. Before the watchdogs the MeqTrees deadlock cost 23-27% of
wall clock here; after them, 0. A gap containing a restart is skipped - the run
was not running and the reason is already on the `restarts` line. That window
opens a second before the gap, because `restarts.log` stamps whole seconds
while evaluation mtimes are fractional. The percentage is of running time, not
wall clock.

**host** - free memory against the headroom `rank-budget.sh` keeps, free disk
on the filesystem holding `results/`, and `ri-ns-sidecar-*` containers whose
launching process is gone. A killed run leaves those holding ~3.4GB per R2D2
rank. Reported to know about, not to act on: `ns_reap_leaked_sidecars` removes
them before the next run reads free memory.

The launcher pid is not the whole rule. A run script killed with SIGKILL leaves
the search going, so on the pid alone a live 16-rank search's containers read
as leaked and `docker rm -f` would have killed it. Each container carries a
`ri.run-dir` label naming its run (`sidecar_launch` in
`scripts/lib/start-sidecars.sh`), and a container whose labelled run still has
processes is never leaked whatever its pid says. Containers from before the
label fall back to the pid, as do the per-rank fallback containers `common.py`
starts.

**why it stopped** - a stopped run's warning quotes its `run.log`, the only
place that says *why* rather than *that* it broke. It quotes the last line
naming an error rather than the last line outright, and how many ranks said it:

```
run.log ends "TypeError: _connect_shell_started_worker() ..." (x15 ranks)
```

The count is the diagnosis. An MPI crash leaves one traceback per rank, so a
plain tail lands on the real failure only by luck. Every rank reporting the
same error is a deterministic code bug; one rank alone is a flaky worker, an
OOM kill, or bad luck, and those want opposite responses. Without a traceback
it falls back to the last non-empty line, PolyChord's own last word.
