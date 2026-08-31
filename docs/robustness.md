# Robustness

What a nested-sampling run does when something breaks: what it scores, what it
retries, what it restarts, and what it leaves for a human. See
[nested-sampling.md](nested-sampling.md) for how to start a run and
[run-health.md](run-health.md) for reading one that is going.

## Infrastructure failures are not failure modes

A failed evaluation scores `FAILURE_OBJECTIVE` (`100.0`), which PolyChord
maximizes against a real `total_rms_jy` of ~0.008 - so a failure becomes the
most interesting point in the search. That is deliberate: failure modes are
what these runs look for.

It is only correct when the *algorithm* failed. A worker the host's OOM killer
took says nothing about R2D2, and scoring it would report a parameter region as
catastrophic when what failed was the machine.

| What happened | What the run does |
|---|---|
| The tool ran and exited non-zero | `FAILURE_OBJECTIVE` - a failure mode, scored |
| The tool was killed by SIGKILL | Retried, then the run stops - never scored |
| A worker died mid-request | Retried, then the run stops - never scored |
| A worker stopped answering | Its meqserver is replaced, or it is killed - never scored |

The SIGKILL row is the OOM killer, and it is a separate row because it is the
one failure that arrives looking like the algorithm's own. The zygote reports a
signal-killed child as `128 + signal` (`docker/wsclean/src/zygote.cpp`), so an
OOM-killed WSClean used to return 137, which is non-zero, which is scored -
`FAILURE_OBJECTIVE` on the machine's memory limit, in a search that maximises.
`is_infrastructure_failure()` in `common.py` classifies it with `WORKER_DIED`
instead. Nothing in this pipeline sends an imager SIGKILL, so there is no
legitimate case to lose; a crash the imager chose - SIGSEGV, SIGABRT, a
non-zero exit - is still scored, because that *is* a failure mode.

R2D2 never had this hole and does not need the rule: `r2d2_serve.py` runs
`imager.py` in-process, so the imaging memory is the worker's own and the OOM
killer takes the worker, which is already the "worker died" row. What it cannot
distinguish is an allocation failure the worker catches (`torch` raising rather
than the kernel killing), which still arrives as exit 1 and is scored. The
defence there is not to run at the edge of memory: see `NS_R2D2_MAX_RANKS`.

A dead worker is retried against a freshly started one, waiting longer each
time (`WORKER_RETRY_DELAYS` in `common.py`, ~51s in total). That is usually
enough, because the memory the attempt died for is released by its own death.
If it still cannot run, the run stops rather than inventing a likelihood: there
is no honest value: scoring it high makes the sampler chase the OOM killer, and
scoring it low carves a hole out of exactly the corner where the real failure
modes live.

## A missing R2D2 checkpoint set

The one input whose absence the search cannot report. R2D2 exits non-zero
without its pretrained checkpoints, which is scored `FAILURE_OBJECTIVE` - so
the run does not stop, it reports the broken imager as its best discovery.
Measured: an 8-rank search in a fresh worktree scored 55 of 55 that way and
terminated at `logZ = 99.93`, a triumphant-looking number for a search that
imaged nothing.

`ns_refuse_missing_checkpoints` checks for them before the run directory is
claimed, so this now costs 0.4s and an explanation instead of a run:

```console
$ ./ri search r2d2
FATAL: no R2D2 checkpoints in /.../checkpoints/R2D2_A1
       Without them every evaluation fails, and a failed evaluation scores
       FAILURE_OBJECTIVE, which PolyChord maximizes - so the search would not
       stop, it would report the broken imager as its best discovery.
       Get them with:  ./ri fetch-checkpoints
       Extract so that /.../checkpoints/R2D2_A1/R2D2_UNet_N<k>.ckpt exists.
       Set CHECKPOINTS_DIR to look somewhere else - a worktree does not
       share the checkpoints of the checkout it was made from.
```

`checkpoints/*` is gitignored, so a worktree starts without them and this is
the case it is most likely to catch. `CHECKPOINTS_DIR` is where to look and
`R2D2_CKPT_NAME` is which set - both from `defaults.toml`, both overridable,
and the same two the imager itself is given, so the check and the run cannot
look in different places. Only that *some* checkpoint is there: which
realisations a run needs is the imager's business, and it says so itself once
it can start. `./ri health` still names the fault after the fact for a run that
started before the checkpoints went away.

## When MeqTrees stops answering

MeqTrees deadlocks with its `meqserver` roughly once every 2,000 to 5,000
evaluations. The worker stays alive, the predict never completes, and no reply
is written - so this is not a worker that died, and nothing watching for a
death sees it.

A source at the phase centre no longer runs a predict at all - its visibility
is a constant that `phase_centre_visibility()` writes directly (see
[nested-sampling-throughput.md](nested-sampling-throughput.md)) - so a default
run cannot reach any of this. Everything below still stands, and is what an
`--enable-param source_offset_fraction` run depends on.

It used to stop the whole run. Timba's `wait=True` means wait *indefinitely*,
so the rank blocked forever and, because PolyChord keeps every rank in the same
collective, the other 19 burned a core each behind it.

Three bounds now stand in the way, each shorter than the one outside it:

| Bound | Where | What it does when it expires |
|---|---|---|
| `PREDICT_WAIT_SECONDS` (3s) | `simulate_point_source_ms.py` | The worker kills its own meqserver, starts a fresh one (~0.2s) and retries. The rank never learns anything happened. |
| `SIMULATE_REPLY_TIMEOUT` (10s) | `common.py` | The rank kills the worker, drops its pooled FIFO slot and retries against a rank-started one. |
| `WORKER_RETRY_DELAYS` (5 attempts) | `common.py` | `WORKER_DIED`: the run stops rather than scoring a host fault. |

The ordering is the design. If the worker's own bound ever exceeds the rank's,
the rank kills the worker before it can fix itself and every deadlock costs a
killed worker again - so `scripts/test_watchdogs.py` asserts the ladder holds,
and CI runs it.

The first layer absorbs nearly all of it: two full 20-rank runs after it was
added recovered 8 deadlocks between them with no gap above 2s anywhere. The
same shape before it lost 23-27% of wall clock to the layer below.

An evaluation that hit one leaves a `meqserver-wedged.log` and nothing else
marks it. One line means the worker fixed itself; two for the same evaluation
means it could not and the worker exited rather than replying - deliberately,
because an exit status would come back as a failed evaluation and the search
would start chasing a wedged meqserver instead of the algorithm.

```bash
R=$(ls -1dt results/nested-sampling/wsclean-* | head -1)
cat "$R"/evaluations/*/meqserver-wedged.log 2>/dev/null | wc -l
```

Burning ranks are not the symptom: a rank blocked in a collective spins whether
its peer is wedged or imaging perfectly normally, so the count on its own says
nothing - [run-health.md](run-health.md) has the measurements.

If a wedge ever gets past all three bounds, one of them has a bug, and
`./ri health` reports it as ranks burning CPU with nothing completing. To
localise it by hand: the stuck evaluation is the old `evaluations/eval-*`
directory with no `metrics.json` and two zero-byte simulate logs, and the rank
waiting on it is the one holding that worker's FIFO pair open, so
`ls -l /proc/<pid>/fd` matched against `.simulate-workers/<n>` names the
worker. The deadlock itself is that worker and its `meqserver` child both
parked in `futex_wait_queue_me` at 0% CPU.

## A run that dies restarts itself

PolyChord checkpoints continuously, so a run that dies at hour three already
holds what it needs to carry on. What it lacked was anything to start it again:
a worker that stopped answering, a wedged meqserver, an OOM kill - each ended a
multi-day search that then sat dead until someone noticed.

`run_with_retries` in `scripts/lib/progress-bar.sh` wraps the run and restarts
it in place, up to `--retries` times (default 2). The restart is an ordinary
resume: same `OUTPUT_DIR`, PolyChord reads its own `.resume` file and the
evaluations already on disk are adopted.

**It only retries an attempt that made forward progress**, measured in
evaluations that attempt actually scored. That guard is what stops it spinning:
a deterministic code bug, a missing image, a bad parameter space all fail
before a single evaluation is scored, so they stop immediately instead of
failing three times as slowly.

The measure used to be dead points added, which silently disabled the retry for
most of a real run - PolyChord writes `chains/` only every `nlive` dead points,
so inside that interval the count is frozen however much imaging happened. A
search SIGKILLed at 31 scored evaluations and 0 dead points printed `not
retrying: ... added no dead points` and stayed dead; the same kill now logs
`attempt failed (exit 137) at 31 evaluations` and the run finishes.

An evaluation directory with no `metrics.json` was in flight when the run died
and does not count; the next attempt deletes it. So is one whose `metrics.json`
does not parse, which is what a rank killed mid-write leaves behind:
`read_evaluation_record` skips it with a `WARNING: ignoring unreadable` line.
That one file used to end a search for good, because `json.loads` raised at
startup in every restart and every `./ri resume`. Records are now written
`metrics.json.partial` and renamed into place, so the window is closed for new
runs and the tolerance covers the runs already on disk.

### Half a `summary.json` is not a finished run

The same window one level out. `summary.json` is written once, after PolyChord
returns, and carries every evaluation - hundreds of MB for a long R2D2 search -
so a rank killed there leaves a truncated file. Every reader called a run with
a `summary.json` finished, which made that the worst of both: `./ri runs` and
`./ri health` said `complete`, the HTML report died on `json.load` and took
*every other run's* page down with it, `./ri merge` and `./ri profile` refused,
and `./ri resume` - the one command that could rewrite it - declined because
the run had "already finished".

The write now goes through `write_json_atomic` in `common.py`, so a new run
leaves either no summary or a whole one; and "finished" means a *whole*
summary, tested by the last byte rather than by parsing tens of MB. A torn one
reads as a run that stopped, paired with the `./ri resume` that repairs it.
Verified on a copy of a real finished 54-evaluation search with its 285KB
summary cut in half: the report built again, and the resume rewrote a whole
summary with the same 54 evaluations in 0.014s, imaging nothing.

### One no-progress failure is retried anyway: an unreadable checkpoint

A rank killed part-way through writing `chains/*.resume` leaves a truncated
file, and PolyChord aborts reading it in Fortran before evaluation 1 - so the
forward-progress guard stopped the run and every later `./ri resume` died in
the same place, leaving every scored evaluation unreachable on disk.

The checkpoint is the one input a restart can change, so `run_with_retries`
moves it to `chains/*.resume.unreadable` and retries. `polychord_*.py` sets
`read_resume` off that file's existence, so the next attempt starts the sampler
from scratch and `adopt_completed_evaluations` replays every scored evaluation
out of the point cache without imaging any of them. Renamed rather than
deleted, because a checkpoint the run could not read is still the only record
of where the sampler had reached.

Only on evidence that the checkpoint is what broke - a gfortran runtime error
naming `read_write.F90` in the output of the attempt that just failed, not
anywhere in `run.log`, which accumulates across attempts. Not capped at one
recovery per run: a full disk tears every checkpoint it writes, and the retry
budget already bounds a fault that keeps coming back.

### Two things to know about a restarted run

- **It reuses the sidecar containers but not their pooled workers**, which
  exited on EOF when the dying ranks closed the FIFOs. Each rank waits out
  `_connect_shell_started_worker`'s 10s deadline and starts its own worker in
  the same sidecar - still one long-lived worker per rank, so the price is that
  one-off wait, not a per-evaluation penalty. A real killed WSClean search
  scored 216 evaluations/min over the 53 before the kill and 219/min over the
  34 after, with a 12.1s gap across the restart. Re-launching the pool would
  mean a second reader on a FIFO whose old worker may not have exited, and two
  readers split the messages.
- **It re-sizes itself.** The rank count in the command is what
  `ns_budget_ranks` could afford when the run *started*, and on a shared host
  that is not a fact about now - the common way a long search dies is another
  session's run growing into it, and replaying the number puts the restart
  straight back into the OOM killer, which does not fail the run: it scores
  `FAILURE_OBJECTIVE`, which PolyChord maximizes. So the count goes back
  through the memory guard before each restart and the run says `retry 1 of 2,
  re-sized to 4 ranks to fit the memory free now`. Only ever downwards, and
  clamping down is free: the checkpoint carries live points, not ranks. If not
  even one rank fits the run stops there rather than spending a restart to fail
  the same way.

Each restart appends to `restarts.log`, and `./ri health` shows the count and
the latest. Reported, not warned on - the run is fine right now - but whatever
killed it once will do it again.

### The budget is for a crash loop, not for the run's lifetime

An attempt that ran for `NS_RETRY_RESET_SECONDS` (1800) before dying hands the
retry budget back, so the count is of failures that keep coming straight back
rather than of restarts ever made. Without that the counter only climbed: a
multi-day R2D2 search that healed itself twice on day one was out of retries
for the rest of the week, and the third unrelated OOM kill ended it exactly the
way `--retries 0` would have. Half an hour is ~70x a single R2D2 evaluation and
~150x the 12.1s a restart costs, so an attempt that clears it plainly got past
whatever killed the last one. Resetting too eagerly is the safe direction,
because a retry still has to have scored evaluations.

When the budget is gone the run says so in `run.log` rather than just stopping,
and `./ri health` reads that tail back. `--retries 0` restores the old
behaviour, where the first failure ends the run.

## A run that hangs instead of dying

`run_with_retries` can only act on a run that *exits*, and the worst failure
here does not. PolyChord calls the likelihood from Fortran, so a single rank
that stops answering leaves every other rank blocked in a collective that never
completes: every core busy, nothing landing, no exit status, and `./ri health`
correctly reporting a live run. The in-worker timeouts only cover a worker that
was *asked* for a reply; a deadlock between ranks is asked nothing.

`_ns_stall_watchdog` in `scripts/lib/progress-bar.sh` is the backstop. It
watches the one thing true of every healthy run and false of every hung one -
evaluations finishing - and after `--stall-timeout` seconds with none, writes
the reason into `run.log` and kills the run, turning the hang into the crash
`run_with_retries` already handles. It runs with or without a terminal, because
a multi-day search is exactly what somebody starts under `nohup`.

The kill is by command line rather than by the pid the run script holds: that
pid is the `docker exec` client, and the ranks are children of
`containerd-shim`. `ns_run_process_pattern` builds the pattern, anchored on the
run's own `--output-dir` so another search on the host is untouched.

**The default is 7200s, deliberately far above anything legitimate.**
`IMAGING_REPLY_TIMEOUT` already lets a single evaluation take an hour, so
anything shorter would kill runs that machinery is still working on. Against
that, the widest gap measured over 6.3 hours of a live 16-rank R2D2 search was
23.5s. `--stall-timeout 0` disables it. The timeout is recorded in `run.env`,
so `./ri resume` replays it instead of reverting to the default, and it is what
lets `./ri health` say when a stalled run is due to be killed.

A restart adopts what the previous attempt evaluated **whether or not PolyChord
left a checkpoint**, and that is load-bearing. Adoption used to be conditional
on the resume file, so a restart before the first checkpoint began at eval id 1
on top of the previous attempt's directories, which `simulate_measurement_set`
creates with `exist_ok=False`. One rank died on `FileExistsError` and left
every other one waiting forever in a collective - a 16-core restart that burned
every core, landed nothing, and never exited for `run_with_retries` to give up
on. Re-sampling from scratch costs nothing: the same seed redraws the same
points and the adopted cache answers them without imaging.

The same hang is closed at its source too: the likelihood aborts the whole job
(`MPI_Abort`) on **any** unexpected exception, not only `WorkerDied`.

## A worker that died while nobody was talking to it

Each rank keeps one long-lived worker per sidecar, and each request path
retries a worker that dies mid-request. What none covered was the worker that
died *between* requests, which is the common one: the biggest resident process
on this host is always an imager worker, an idle R2D2 one still holds ~3.4GB,
and the OOM killer takes it while its rank waits in a collective. The pipe is
then already broken when the next request is written, and the write raised
before any retry machinery was reached - and because the likelihood is called
from Fortran, that `BrokenPipeError` aborted the whole job.

`worker_send` in `common.py` is the counterpart to `worker_reply`: it reports a
failed write as "this worker is gone" instead of raising, and all three request
paths drop the worker and retry. Measured on a real search whose workers were
killed at 10 evaluations: before, the job aborted and spent a restart to reach
the same 54 evaluations; after, it finished with no restart at all.

## A sidecar container that went away

The same reproduction with the *container* removed (`docker rm -f`, an OOM kill
of its main process, a daemon restart) ends cleanly as `WORKER_DIED`, but there
is nowhere to start a replacement worker, so the run dies. That part is
correct; what was not is that it could not be restarted either. The containers
were started once, in front of `run_with_retries`, so every attempt after the
removal `docker exec`ed into a name that no longer existed, scored nothing, and
the anti-spin guard stopped the run for good at exit 1 with no `summary.json`.

`sidecar_launch` in `scripts/lib/start-sidecars.sh` now keeps each container's
own `docker run` arguments, and `sidecar_restore` - called before each retry -
starts any container that is gone again under the same name. Only missing
containers are touched, so an ordinary restart costs one `docker inspect` each,
and the line naming a container it had to start again is teed into `run.log`.
Re-launching is safe because the containers hold no run state.

## Finding and resuming a run that stopped

A run writes `summary.json` only once PolyChord returns, so a run directory
without one stopped early. `./ri runs` is the list:

```console
$ ./ri runs
RUN                        ALGORITHM  STATUS      EVALS  STARTED
r2d2-vlaa-20260828T2054Z   r2d2       running      6882  today 21:54 (2h ago)
r2d2-vlaa-20260827T1015Z   r2d2       resumable     659  yesterday 11:15 (1d ago)
wsclean-vlaa-20260827T09Z  wsclean    complete     1706  yesterday 10:04 (1d ago)
r2d2-vlaa-20260827T0932Z   r2d2       incomplete      0  yesterday 09:32 (1d ago)

1 run still going. Check on it with:
  ./ri health r2d2-vlaa-20260828T2054Z

2 runs stopped before finishing.
Continue where it left off, keeping every evaluation already done:
  ./ri resume r2d2-vlaa-20260827T1015Z
No checkpoint, so the sampler starts over, reusing the evaluations already scored:
  ./ri resume r2d2-vlaa-20260827T0932Z
```

`STATUS` is `complete` when a whole `summary.json` is there, `running` when a
process is driving it, `resumable` when neither but a `.resume` file is, and
`incomplete` when the run stopped before checkpointing anything.
`./ri runs --incomplete` lists only the ones needing attention; `--json` is the
machine-readable form.

`STARTED` is that run's own UTC name read back as local time and an age, with
the newest run at the top whichever imager it belongs to - sorting by name put
every `wsclean-*` run below every `r2d2-*` one.

`EVALS` counts evaluations that were *scored*, the same number `./ri health`
calls `progress`. A directory with no `metrics.json` holds nothing and
`adopt_completed_evaluations` deletes it on resume. Counting directories
instead made three runs here that died during startup advertise 7, 7 and 15
evaluations under a footer promising to keep every one, when what survives is
zero.

`./ri resume <run>` continues it in place, with no flags: each run records what
it was started with in `run.env`, so a resume cannot silently become a
different search.

It **builds the images first**, exactly as `./ri search` does. A run executes
the code baked into them, so a resume against a stale image silently continues
on whatever was baked last time - backwards for the case a resume is most often
typed in: something killed the run, the fix went into the working tree, and
`./ri health` printed `./ri resume`. Reproduced by marking
`polychord_wsclean.py` and resuming a killed search; the mark never ran. The
builds cost ~0.05s each when nothing changed, and go in front of the rank clamp
so it reads the memory left after them. `--no-build` skips them, for a working
tree that has moved on and must not reach a run already in flight.

The one setting a resume does not replay verbatim is the rank count, for the
same reason a restart re-sizes:

```console
$ ./ri resume wsclean-vlaa-20260828T0221Z
NOTE: wsclean ranks 3 -> 1 (4400MB available, 0MB reserved by other runs, 200MB per rank)
Resuming wsclean-vlaa-20260828T0221Z (wsclean, 31 evaluations already done, 1 rank)
```

`--mpi-procs` on `./ri search` is still obeyed as typed and only warned about;
the difference is that a resume did not type anything.

A live run is indistinguishable on disk from one that stopped, so liveness is
read from the process table. Before that, `./ri runs` called the live search
`resumable` and printed `./ri resume` for it, which is an instruction to start
a second MPI job over the live one's own checkpoint. `./ri resume` refuses:

```console
$ ./ri resume r2d2-vlaa-20260828T2054Z
FATAL: r2d2-vlaa-20260828T2054Z is still running, so there is nothing to resume.
       A second job over the same checkpoint would corrupt both.
       Watch it instead:  ./ri health r2d2-vlaa-20260828T2054Z
```

The refusal lives in `./ri resume` rather than in each place that suggests one,
because the HTML report can still offer it: that report is a snapshot, and a
liveness check baked into a static page would be stale by the time anyone read
it. Guarding the action covers every route to it.

This covers every way a long run stops - the memory guard giving up, a Ctrl-C,
a reboot - not only the ones above. A fresh run has no resume file and starts
clean, so leaving checkpointing on costs nothing.

## `./ri self-check self-heal`

The end-to-end check of all of the above, and part of `./ri self-check`. It
starts a real WSClean search on a throwaway directory and breaks it six times,
in the six ways that recover through different machinery:

| Break | What has to happen |
|---|---|
| `SIGKILL` after 8 evaluations (fewer than `--nlive`, so before any checkpoint) | Restarts itself, records the kill, keeps the evaluations, writes `summary.json`, and `./ri health` says nothing on its headline |
| `SIGSTOP` on one rank | Only the stall watchdog can notice; run with `--stall-timeout 20` and a 2s poll so it costs a minute rather than two hours |
| `SIGKILL` on a search started with `--retries 0` | Does *not* recover on its own: `./ri health` must headline it `STOPPED` and name `./ri resume <run>`, and that command - taken from the report and typed verbatim - must continue the search rather than begin one |
| `SIGKILL` on every worker, from inside the sidecar | Costs nothing: absorbed inside the evaluation, so `restarts.log` is never written |
| `docker rm --force` on a sidecar | Costs a restart: the run dies and `sidecar_restore` is what lets the retry fix it |
| A truncated `chains/*.resume`, then a truncated `summary.json`, on the finished run | `./ri resume` finishes both, keeps the torn checkpoint as `*.resume.unreadable`, and leaves the evaluation count unchanged - nothing imaged twice |

~5 minutes and ~0.6GB, so it is safe to run beside another search.

Fixtures cannot stand in for it. Every bug found in this machinery so far - the
retry reading a checkpoint-frozen counter, the stall accounting refusing to
excuse a run's own restart, the restart colliding with its own evaluation
directories, `./ri health` calling a run killed a second ago `STARTING`, and
the broken pipe above - passed the fixtures and failed a real kill.
