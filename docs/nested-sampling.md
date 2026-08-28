# Nested sampling

This repo uses PolyChord as a targeted search tool, not as a Bayesian posterior
fit. PolyChord maximizes a configurable objective metric (default
`total_rms_jy`).

Ground truth for every run is one unpolarized 1 Jy point source at phase centre.
Dynamic range is controlled by complex Gaussian thermal noise in the simulated
visibilities.

## Images

```bash
./ri build wsclean
./ri build r2d2
./ri build meqtrees
./ri build polychord
```

`./ri build` also builds the MeqTrees and PolyChord images.

The MeqTrees image uses KERN 10 packages on Ubuntu 24.04. The VLA.A antenna table
is unpacked from makems' bundled `VLAA_ANT` example inside the image, so antenna
positions are not hand-rolled in this repo. Visibilities for that skeleton are
predicted by an actual MeqTrees/Meow point-source RIME run
(`scripts/lib/nested_sampling/point_source_forest.py`, driven through
`meqtree-pipeliner.py`), not a hand-rolled formula; thermal noise is added on
top of that clean MeqTrees prediction.

## Run it

Both algorithms share the same `NS_*` and `OUTPUT_DIR` overrides (see "Environment
overrides" below). Each target builds its required images first and starts one
long-lived sidecar container per image; the PolyChord container mounts the
Docker socket and drives those sidecars. It is started the same way as the
others and the run is `docker exec`ed into it (see "The PolyChord container is
a sidecar too" and "Long-lived sidecar containers, one per image" in
[nested-sampling-profiling.md](nested-sampling-profiling.md)).

Once the containers are up, a status line tracks the search against its
`--max-ndead` budget: elapsed time, dead points done (from PolyChord's own
`chains/*_dead-birth.txt`, one line per dead point - not the raw evaluation
count, which is always higher since PolyChord's slice sampler makes several
evaluations per accepted dead point), a percent, and an ETA extrapolated from
the rate so far (`scripts/lib/progress-bar.sh`). With `--max-ndead <= 0` (run
until the evidence tolerance is met, no fixed cap) there is no dead-point cap
to measure a percent against, so `_ns_evidence_total` estimates one from the
run's own evidence, and every figure derived from it is marked `~`:
`~ 53%  ~241/~452 dead points ... eta ~4:35:00`.

PolyChord rewrites `chains/` only every `nlive` dead points, so the dead-point
count is frozen between writes by construction - two hours at a time on a
16-rank R2D2 search, which is a bar that sits still for two hours and then
jumps fifty. `_ns_dead_now` carries it across that interval: the evaluation
directories appear every few seconds, the slice sampler spends a near-constant
number of them per dead point, so the evaluations that landed after the
checkpoint convert back into dead points at the run's own measured ratio. The
carried count takes a `~` of its own even when the denominator is an exact
`--max-ndead`, and `./ri health`'s **forecast** line carries the count the same
way from the same ratio.

That estimate is the same model `./ri health`'s **forecast** line uses, from
the same two files (`chains/*.stats` for the accumulated `log(Z)`,
`chains/*_phys_live.txt` for the live points' current log-likelihoods) and the
same measured stopping fraction - see the **forecast** field below for the
calibration and its evidence. The bar's copy of that constant is
`_NS_TERMINATION_EVIDENCE_RATIO`, and `progress-bar.sh --self-check` fails if
it and the health script's `TERMINATION_EVIDENCE_RATIO` drift apart: a status
line and a report disagreeing about the same run is what this replaced. The
earlier bar divided PolyChord's documented `precision_criterion` by the
current evidence ratio, which is both uncalibrated and exponential in the
quantity being reported - it read 3% on a live 16-rank R2D2 search that this
model puts at 38%.

Like the health forecast, it approximates the remaining prior volume as a
single global `-ndead/nlive` rather than PolyChord's own per-cluster tracking,
so a run whose live points split across several clusters (`chains/*.stats`'
`ncluster` line) will diverge further from PolyChord's exact figure than a
single-cluster run does. Before the first e-fold (`ndead < nlive`, where the
live set is still the prior and the estimate would report its own constant),
and before `.stats` or `phys_live.txt` exist at all, the line falls back to a
bouncing bar and the raw dead-point rate. On a real terminal it is pinned to the bottom via
a scroll region, so PolyChord's own feedback scrolling past above it doesn't
bury it; only drawn
on a TTY, so piped or logged runs are unaffected.

### WSClean

```bash
./ri search wsclean
```

Outputs:

```text
results/nested-sampling/wsclean-vlaa-<UTC timestamp>/
```

The timestamp is claimed with a bare `mkdir`, not assumed: two searches started
in the same second - two sessions sharing this host, or one script launching a
pair - would otherwise resolve to the same directory and write each other's
evaluations, FIFOs and `summary.json`, with the first to finish deleting the
FIFO directory the other was reading. The loser of the race waits for the next
second rather than decorating its name, so a run directory always ends in a
stamp. An `--output-dir` you name yourself is yours and may already exist -
unless a job is still in it, which is refused for the same reason
`./ri resume` refuses a live run: measured, a second search into a live run
directory deleted its FIFOs, recreated them with its own rank count, and
wrote its own `chains/*.resume` over the live checkpoint while the first run
was still imaging. Liveness is the host's process list (`ns_run_is_live` in
`scripts/lib/progress-bar.sh`), which is the only thing `mkdir -p` cannot
see.

It also has to be inside the repository. Every container the run starts is
given one bind mount, `-v $REPO_ROOT:$REPO_ROOT`, so a run directory outside
it exists twice over: on the host, holding `run.env` and the FIFOs, and
emptily inside each container, where PolyChord's chains and the evaluation
directories are actually written. Measured on a real `--output-dir /tmp/...`
search, that cost two minutes of container startup and then died on
evaluation 1 with `FileNotFoundError: .../eval-0001-*/simulate.stdout.log`;
`ns_refuse_unmounted_run` in `scripts/lib/run-config.sh` now refuses it in
0.1s instead. The path is resolved to an absolute one first, so a relative
`--output-dir` still works and the run names itself the same way everywhere.

Useful overrides:

```bash
./ri search wsclean --nlive 8 --num-repeats 2 --max-ndead 12
./ri search wsclean --mpi-procs 4
./ri search wsclean --metric badness
./ri search wsclean --metric snr
./ri search r2d2 --metric off_source_rms_jy
./ri search r2d2 --metric sigma_res
./ri search wsclean --output-dir results/nested-sampling/manual
```

PolyChord likelihood evaluations run in parallel across MPI ranks inside the
PolyChord container. `NS_MPI_PROCS` sets the rank count (default
`min(NS_NLIVE, host CPUs)`). Set `NS_MPI_PROCS=1` to disable parallel
evaluations for debugging.

The target builds any missing WSClean, MeqTrees, and PolyChord images first.
Each likelihood evaluation runs one MeqTrees simulate and one WSClean imaging
step in this rank's already-running sidecar containers.

### R2D2

```bash
./ri search r2d2
```

Outputs:

```text
results/nested-sampling/r2d2-vlaa-<UTC timestamp>/
```

The target builds R2D2, MeqTrees, and PolyChord images first. Each likelihood
evaluation runs one MeqTrees simulate, one MeqTrees-hosted MS-to-`.mat`
conversion and one R2D2 imaging job, each a request to a long-lived process
inside one of the run's two sidecar containers.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `./ri fetch-checkpoints` and `./ri smoke r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
./ri smoke ms-to-mat                 # or: scripts/check-ms-to-r2d2-mat.sh
```

`run-nested-sampling-r2d2.sh` runs `NS_MPI_PROCS` PolyChord ranks
concurrently, each with its own `r2d2_serve.py` imaging worker inside the
shared R2D2 sidecar and its own simulate worker inside the MeqTrees one. Both
pools are started by their container's own command, over one FIFO pair per rank,
before the PolyChord container exists - the imaging pool as one `--fifo-dir`
process that imports torch once, forks a worker per pair, and opens every pair
before it starts importing so the ranks do not wait for it (see "R2D2 imaging runs in a long-lived
worker", "The workers are started by the container, not by the ranks" and "The
ranks attach to the pool before the warm-up" in
[nested-sampling-profiling.md](nested-sampling-profiling.md)). That process also
patches R2D2's `MeasOp.get_op_norm` to solve the operator norm with Lanczos
rather than upstream's power iteration - the same quantity, ~3.5x fewer NUFFT
pairs and ~3e-6 relative accuracy instead of ~1e-4, and no longer a different
answer on every run (see "The operator norm is solved with Lanczos" there) - and
it gives each measurement operator one FINUFFT plan per transform type instead
of the one-plan-per-transform `pytorch_finufft` builds, worth ~30% of a warm
imaging request (see "Each measurement operator keeps its FINUFFT plans"). Its
warm-up runs `imager.py`'s own import block - the file under a run name that is
not `__main__` - plus the NUFFT backend `create_meas_op` imports lazily, and
makes `utils` resolve its submodules on demand so the imaging path never pays
for `lightning` or `scipy.optimize` (see "The imaging worker warms what
`imager.py` imports, and no more"). The workers get
OpenMP/BLAS thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`) set from the host's available CPU count, overridable via
`R2D2_OMP_THREADS`. The previous image default of
`OMP_NUM_THREADS=4` capped finufft/OpenMP work when the Docker VM exposed more
CPUs than four. To avoid CPU oversubscription, the script defaults
`R2D2_OMP_THREADS` to `host CPUs / NS_MPI_PROCS` (minimum `1`) when not set
explicitly, so each rank's imaging worker gets a fair share of the host's cores
instead of all of them. Set `R2D2_OMP_THREADS` explicitly to override this
per-rank default. The same count is written into every per-evaluation
`r2d2_config.yaml` as `ncpus`, because those env vars alone do not reach
torch - see "R2D2 sizes its own torch thread pool" in
[nested-sampling-profiling.md](nested-sampling-profiling.md).

### Environment overrides

Both run scripts read the same variables and forward the sampler ones to
`polychord_wsclean.py` / `polychord_r2d2.py` as command-line flags.
Sampler defaults live in `defaults.toml` at the repository root, loaded by
`scripts/lib/defaults.sh`. Setting a variable yourself still wins over
`defaults.toml`; a flag wins over both.

#### Tweak these

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--nlive` | `NS_NLIVE` | Number of PolyChord live points | `8` |
| `--num-repeats` | `NS_NUM_REPEATS` | How much PolyChord explores inside the likelihood constraint before generating a replacement live point | `2` |
| `--max-ndead` | `NS_MAX_NDEAD` | Dead-point budget that terminates the run | `12` |
| `--seed` | `NS_SEED` | PolyChord random seed | `41` |
| `--metric` | `NS_METRIC` | Objective: `badness`, a bare metric name, or an expression over metric names - see "Choosing the objective" below | `total_rms_jy` |
| `--retries` | `NS_RETRIES` | Times a run that dies after scoring evaluations restarts itself from its checkpoint, counting only failures that come straight back (`NS_RETRY_RESET_SECONDS`, 1800, hands the budget back); `0` disables - see "A run that dies restarts itself" | `2` |
| `--stall-timeout` | `NS_STALL_TIMEOUT` | Seconds with no evaluation finishing before a run is killed as hung, so `--retries` can restart it; `0` disables - see "A run that hangs instead of dying" | `7200` |

#### Leave these alone

Flags exist, but the defaults are derived. Leave them unset unless you want
serial debugging (`--mpi-procs 1`), a different rank/thread split, or a
pinned run directory.

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--mpi-procs` | `NS_MPI_PROCS` | PolyChord rank count (`mpirun -np`); `1` is serial | `min(NS_NLIVE, host CPUs)`, host CPUs from `nproc` (`sysctl -n hw.ncpu` on macOS, which has no `nproc`), then clamped to what free memory holds - see "Rank count is the memory budget" |
| `--omp-threads` | `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS/torch threads | `host CPUs / NS_MPI_PROCS`, min 1, from the rank count before the memory clamp |
| `--output-dir` | `OUTPUT_DIR` | Run directory; must be inside the repository, and not one a job is still in | `results/nested-sampling/<algo>-vlaa-<UTC>` |

### Rank count is the memory budget

Rank count, not `NS_NLIVE`, is what costs memory: each rank keeps one warm
worker holding its own copy of the imaging stack. Measured on a 20-CPU, 62GB
host with `NS_NLIVE` held at 12 and only the rank count varied:

| Ranks | R2D2 peak memory |
|---:|---:|
| 4 | 13.5GB |
| 8 | 27.0GB |
| 12 | 40.6GB |

That is 3.4GB per R2D2 rank, linear, against ~0.2GB per WSClean rank. Two
consequences:

- **`NS_NLIVE` is free.** It sets search quality, not memory. `--nlive 40
  --mpi-procs 12` evaluates 40 live points 12 at a time and costs 12 ranks of
  memory. Raise `--nlive` for a better search without paying for it in RAM.
- **`--mpi-procs` is what has to fit.** On a 62GB host R2D2 tops out around 16
  ranks; the CPU would allow 20.

Both run scripts clamp an auto-derived rank count to what free memory can
hold, and reserve it so that runs started at the same moment size themselves
around each other rather than both assuming an empty host
(`scripts/lib/rank-budget.sh`). A clamp prints a `NOTE:`; a host with no room
for even one rank fails before starting any container. An explicit
`--mpi-procs` is honoured, with a `WARNING:` if it will not fit.

### Infrastructure failures are not failure modes

A failed evaluation scores `FAILURE_OBJECTIVE` (`100.0`), which PolyChord
maximizes against a real `total_rms_jy` of ~0.008 - so a failure becomes the
most interesting point in the search. That is deliberate: failure modes are
what these runs look for.

It is only correct when the *algorithm* failed. A worker the host's OOM killer
took says nothing about R2D2, and scoring it would report a parameter region
as catastrophic when what actually failed was the machine. The two used to be
indistinguishable, because a dead worker and a genuine non-zero exit both
arrived as `returncode=1`. They are now separate:

| What happened | What the run does |
|---|---|
| The tool ran and exited non-zero | `FAILURE_OBJECTIVE` - a failure mode, scored |
| A worker died mid-request | Retried, then the run stops - never scored |
| A worker stopped answering | Its meqserver is replaced, or it is killed - never scored |

A dead worker is retried against a freshly started one, waiting longer each
time (`WORKER_RETRY_DELAYS` in `common.py`, ~51s in total). That is usually
enough, because the memory the attempt died for is released by its own death
and the run holding the rest eventually finishes.

If it still cannot run, the run stops rather than inventing a likelihood.
There is no honest value to return: scoring it high makes the sampler chase
the OOM killer, and scoring it low carves a hole out of exactly the expensive
corner where the real failure modes live.

### When MeqTrees stops answering

MeqTrees deadlocks with its `meqserver` roughly once every 2,000 to 5,000
evaluations. The worker stays alive, the predict never completes, and no reply
is ever written - so this is not a worker that died, and nothing that watches
for a death sees it.

It used to stop the whole run. Timba's `wait=True` means wait *indefinitely*,
so the rank that asked for that simulate blocked forever, and because PolyChord
keeps every rank in the same collective, the other 19 burned a core each behind
it. A 20-rank run left overnight came back stopped rather than finished.

Three bounds now stand in the way, each one shorter than the one outside it:

| Bound | Where | What it does when it expires |
|---|---|---|
| `PREDICT_WAIT_SECONDS` (3s) | `simulate_point_source_ms.py` | The worker kills its own meqserver, starts a fresh one (~0.2s) and retries the predict. The rank never learns anything happened. |
| `SIMULATE_REPLY_TIMEOUT` (10s) | `common.py` | The rank kills the worker, drops its pooled FIFO slot and retries against a rank-started one. |
| `WORKER_RETRY_DELAYS` (5 attempts) | `common.py` | `WORKER_DIED`: the run stops rather than scoring a host fault. |

The ordering is the design, not a coincidence. If the worker's own bound ever
exceeds the rank's, the rank kills the worker before it can fix itself and
every deadlock silently costs a killed worker again - so
`scripts/test_watchdogs.py` asserts the ladder holds, and CI runs it.

In practice the first layer absorbs nearly all of it. Two full 20-rank runs
after it was added recovered 8 deadlocks between them without a single one
reaching the rank, and neither run had a gap above 2s anywhere. The same run
shape before it lost 23-27% of its wall clock to the layer below.

An evaluation that hit one leaves a `meqserver-wedged.log` next to its other
logs, and nothing else marks it:

```
$ cat results/nested-sampling/<run>/evaluations/*/meqserver-wedged.log
attempt 1: no reply to the predict in 3.0s
```

One line means the worker fixed itself. Two, for the same evaluation, means it
could not, and the worker exited rather than replying - deliberately, because
an exit status would come back as a failed evaluation and the search would
start chasing a wedged meqserver instead of the algorithm.

Counting those files is the honest way to ask how much a run is paying:

```bash
R=$(ls -1dt results/nested-sampling/wsclean-* | head -1)
cat "$R"/evaluations/*/meqserver-wedged.log 2>/dev/null | wc -l
```

### Is the run healthy?

`./ri runs` answers "did it finish?". `./ri health` answers the question you
have while one is still going:

```console
$ ./ri health
r2d2-vlaa-20260827T205418Z  r2d2  HEALTHY
  stage     sampling, 113 dead points as of 0:08:18 ago, next at ~163
  progress  1287 evaluations, 15 in flight
  activity  last evaluation 0:00:02 ago, 27.3/min over 0:47:06
  history   █▆▆▇▇▇█▆▂▆█▆█▄█▆█▅█▇  5-34/min per 0:07:29 slice
  imaging   25.4s per evaluation, ranks 66% busy  (last 50: 12.3s, 6% busy)
  occupancy ▇▆▆▇▇▂▆▆▇▆▆▆▇▆█▆▇▂▁▆  6%-88% of 16 ranks busy per 0:07:29 slice
  sampler   logZ = -0.044 +/- 0.012, 24 likelihood calls per dead point
  forecast  ~26% done, ~120 of ~454 dead points, ~2h11m left
  ranks     16 ranks of 16, 7 busy-waiting
  resources 48.9GB resident (+4.5GB swapped out) over 51 processes, 16.0 of 20 cores busy
  memory    3.3GB peak imager memory, 53.2GB across 16 ranks
  disk      7.5GB written, +2.6GB/hour, 93h of space left at that rate
  failures  0 scored FAILURE_OBJECTIVE, 0 meqserver wedges recovered
  stalls    8 gaps over 13s, 154s = 5.5% of wall clock

host
  memory    8.9GB available of 62.6GB, 4GB reserved as headroom
  swap      5.1GB of 32.0GB used
  pressure  memory 0.0% / 0.0%, io 0.0% / 0.0% of wall clock stalled (1m / 5m)
  disk      233GB free of 436GB
  sidecars  3 running, 0 leaked
```

With no argument it reports on every run that still has ranks, falling back to
the newest run when nothing is going. Runs rather than the newest run because a
five-minute test started after a ten-hour search would otherwise be "the newest
run" and hide the only one of the two worth asking about; all of them rather
than the newest live one because memory is what caps a run here and this host is
shared - a second search is the usual reason the first is slow, and reporting
one of the two explains a squeezed run with its cause off the page. Live runs
are found in the host process list rather than by globbing this checkout's
`results/nested-sampling/`, so a run started from another worktree is reported
too, named by a `path` line - from a worktree the report used to answer "why is
there no memory left" with that worktree's newest finished smoke run while the
48GB search that owned the machine stayed off the page. `--all` covers every run
on disk under this checkout and `--json` is the machine-readable form. It reads files and runs one `ps` and one `docker
ps`, plus a one second CPU sample when a run has live ranks - nothing is
started and nothing is imaged, so a run in progress does not notice it. Exit
status is 1 when something needs attention, so it can gate a script.

**The status is decided in this order, and the order is the point.** A run
that finished and a run that died both stop writing, so a stale mtime alone
says nothing:

| | |
|---|---|
| `FINISHED` | `summary.json` is there. |
| `STALLED` | Ranks are still running, but no evaluation has landed in `--stale-seconds` (default 600). |
| `STARTING` | No ranks yet, but something was written recently. |
| `STOPPED` | No ranks and nothing recent. `./ri resume <run>` continues it. |
| `HEALTHY` | Ranks running and evaluations landing, and nothing warned about. |

**The headline carries the warning count**, because it is the whole report for
a reader who does not get to the bottom of it - and because under `--all` it is
the only line worth scanning:

```console
$ ./ri health --all
wsclean-vlaa-20260828T022337Z  wsclean  FINISHED
wsclean-vlaa-20260828T022151Z  wsclean  STOPPED - 1 WARNING
r2d2-vlaa-20260827T205418Z  r2d2  RUNNING - 1 WARNING
```

`HEALTHY` is the only status word that is a claim about the run rather than a
point in its lifecycle, so it is the one that steps aside when there is
something to say: a run holding a worker that is 98% paged out headlined
`HEALTHY` one line above the warning naming it, while exiting 1. The other
words already say trouble and only gain the count. No suffix on any run and no
host warning is exactly exit 0.

What each line is reading, and why it is worth a line:

- **stage** - how far into PolyChord the run got, from `chains/`: `*.resume`
  means it reached the main loop, `*_phys_live.txt` alone means it is still
  drawing the initial live points. A run that dies before the main loop dies
  with no dead points, which otherwise looks the same as one that has barely
  started.

  The dead-point count never appears without **how old it is**, because
  PolyChord writes its checkpoint only every ~`nlive` points and the count
  cannot move between writes. One reading of 57 that had not changed in fifty
  minutes was taken here as evidence that progress had stopped, and cost an
  hour of investigation; the next write landed at 113. It survived being
  checked against the terminal, too - PolyChord's feedback box and these files
  come from the *same* update event, so agreement between them is one signal
  displayed twice, not two witnesses. Nothing in this report decides anything
  from the count, and printing its age plus where it will next land is what
  stops a reader doing so either. An age of an hour or more is ordinary rather
  than alarming, and it grows as a run goes on: one 16-rank R2D2 search took
  31 minutes to its first checkpoint and 72 more to its second, because each
  batch of `nlive` dead points costs more likelihood evaluations than the last
  (its `<nlike>` went 14.10 to 32.50 over the same two). So a later reading
  longer than an earlier one is the expected shape, not a slowdown. `next at`
  appears only while the run is going: a finished or dead run's count will
  never move again, and promising it a next value is the one thing that would
  make its stale-by-design checkpoint read as work still to come.
- **progress** - `evaluations/eval-*/metrics.json` is written only when an
  evaluation succeeds, so its count is the progress and the directories
  without one are the evaluations in flight. That number should sit near
  `NS_MPI_PROCS`; pinned there while the count does not move is every rank
  stuck at once. On a run with no ranks left they are counted as *abandoned*
  instead - what the ranks were holding when it died, not work still going.
- **activity** - the overall rate, and the rate over the last 50 evaluations
  when the two have diverged. Both are evaluations over elapsed time, which is
  the only way to read one against the other: the recent one used to be taken
  from the *median* gap of that window while the overall one is the count over
  the span, and the median gap is shorter than the mean whenever a run stalls
  at all. On the live 16-rank R2D2 search that gap-median rate read 26.8/min
  against an overall 22.5/min - a speedup - at a moment the divergence gate had
  fired on a slowdown, and against a recent occupancy (25% of 16 ranks at 17.8s
  an evaluation) that independently says 14/min. Measured the same way as the
  overall, it reads 14.2/min. Nothing was lost by dropping the median's
  robustness: walking both gates over every moment of that run, the mean ratio
  crosses the 2x threshold in 9.1% of 10,720 sampled moments against the median
  ratio's 10.2%, and over a 1,051-moment WSClean run neither fires at all.
  A run can collapse to a fraction of its own throughput without ever going
  quiet long enough to look stalled, and that
  state passes every other check here: on a live 16-rank R2D2 search, 25/min
  fell to 5/min for ten minutes while evaluations kept landing every 20-30s.
  Both numbers are shown and **neither is warned on**, because the same run
  then recovered to 37/min with nothing done to it - five minute bins of 104,
  23, 26, 93 against a 104-165 baseline. One dip and one recovery is not
  grounds for telling anyone to act.
  Neither rate is shown at all when the run's first and last evaluation are
  under a second apart: parallel ranks land their opening batch together, so a
  run killed inside it has a span that measures mtime granularity rather than
  throughput, and dividing by it printed things like `6176.5/min over
  0:00:00`. The same floor silences **history**, twenty times over.

  Both rates are medians of the gaps between evaluations, not counts in a
  window, and that is deliberate: the most recent window is always partial, so
  it reads low by whatever fraction of it has not elapsed. On this run,
  mid-window, the partial bucket said 23.8/min against a 91-165 per five
  minutes baseline - a collapse, apparently - while the gaps said 52.5/min and
  the bucket finished at 164, the highest of the run. A gap cannot be measured
  until both of its ends exist, so there is no partial window to misread. The
  one thing gaps cannot see is a stall that began *after* the last completed
  evaluation, which is what the idle thresholds cover; the two look redundant
  and are complementary.

  Do not conclude from a falling rate that the evaluations got harder -
  **imaging** below is what answers that, and on this run it answered *no*.
  Whether a falling rate means stragglers is a third question, answered by the
  *spread* of `metrics.json` `timing.image_container_seconds` rather than its
  median: a fat tail is one slow evaluation gating a batch, while a tight
  distribution (that run: min 11.6s, p50 21.2s, p90 30.4s, max 33.9s) means
  the missing wall clock is going into sampler overhead, contention or
  synchronisation. The three want different responses.
- **history** - the same throughput binned into twenty equal slices of the
  run's own life, scaled to its own peak. The two rates above are the only
  numbers here that change over time, and as numbers they cannot show the
  *shape*: a dip that recovered and a step down that did not read identically
  on the way past each other. The collapse-and-recovery described above -
  bins of 104, 23, 26, 93 against a 104-165 baseline - is an obvious V here
  and an ordinary-looking slowdown from the two rates alone. A slice where
  nothing landed is marked `·` rather than drawn as merely slow. Binned over
  first-to-last evaluation, never up to now, for the partial-window reason
  above; how long ago the last one landed is **activity**'s job.
- **imaging** - what one evaluation costs the imager, and how much of the
  run's hardware that cost is being spread over. The arrival rate in
  **activity** cannot tell a slower imager from idle ranks: both read as fewer
  evaluations a minute. `metrics.json` carries the imager's own wall clock, and
  this scan already reads every one of those files in full, so the median costs
  a regex over a string already in memory and no extra I/O.
  Beside it, the imaging seconds the run has banked per second of wall clock,
  as a percentage of the ranks it was given: 3,281 seconds of imaging over 898
  of wall clock is 3.7 of 8 ranks kept busy. It is the only place in the report
  where memory a run is *holding but not using* shows up as such.
  A total over the window rather than the ratio of the two medians, which is
  what this was first written as and read systematically high - a duty cycle is
  seconds worked over seconds elapsed, and the median gap is shorter than the
  mean the moment a run stalls at all, so the live R2D2 search printed a
  clamped "100% busy" over a life its own slices put at 6-88%. Two figures in
  one report disagreeing about the same thing is worse than either.
  Both are shown over the last 50 evaluations too when either has moved
  materially - which is how the live R2D2 search's 5-fold slowdown was
  diagnosed: 25.4s at 66% over its life against 12.3s at 6% over its last 50,
  so the imager had got twice as *fast* while fifteen of its sixteen ranks -
  and the ~44GB they hold - went idle waiting on the sampler.
  Clamped at 100%: an evaluation is banked at the moment it finished while its
  cost was spent before that, so a window can hold more imaging seconds than it
  had rank-seconds to spend and the raw ratio would print "23 of 16".
  Reported, never warned on. Every WSClean run on this host ends its last 50
  evaluations near 23% simply by shutting down, and the live R2D2 run's own
  twenty slices ranged 4.4 to 23.1 effective ranks with no fault - there is no
  threshold here that would not mostly fire on ordinary phases. Withheld
  entirely below the same one-second span floor as the rate, where the elapsed
  time is mtime granularity and the ratio is a division by noise.
- **occupancy** - that same duty cycle binned into the twenty slices
  **history** uses, which is the only line here that says whether the hardware
  a run is holding has been earning its keep *all along* or only at the moment
  it was asked. **history** cannot answer that: the imager's own cost drifts as
  the search concentrates, so the live R2D2 run got twice as fast per
  evaluation while its arrival rate fell fivefold, and **history** drew a
  collapse over a stretch the ranks were merely idle for.
  Per slice: imaging seconds landed, over the rank-seconds the slice had to
  spend. The scale is absolute rather than **history**'s peak-relative one,
  because a duty cycle has a natural full - a solid bar is every rank imaging,
  and a bar that never leaves the floor is a run that should have been given
  fewer ranks or a larger `--nlive`. Free: the costs and the slicing are both
  already computed for the two lines above. Withheld when `run.env` does not
  record `NS_MPI_PROCS`, since without a rank count there is no denominator.
- **sampler** - PolyChord's own running total, out of `chains/*.stats`, which
  it rewrites at every checkpoint. Every other line here is operational; this
  is the number the search exists to produce, and `logZ` moving is the only
  direct evidence that the sampler is integrating rather than merely running.
  The likelihood calls per dead point beside it is the sampler's efficiency -
  what the evaluation rate in **activity** is being *spent* on. A run whose
  rate holds while this climbs is working just as hard for less, which no
  other line here can show.
- **forecast** - how far through the search is, and how long is left. With
  `--max-ndead -1`, the default for a real search, there is otherwise no
  denominator anywhere: a run could be reported healthy and fast for three
  days without this report ever saying whether it was a tenth done or nearly
  finished. Nested sampling supplies one. Each dead point shrinks the prior
  volume by the same factor, so `exp(-ndead/nlive)` is what is left of it; the
  evidence still to come is that volume times the mean likelihood of the live
  points now in it (`chains/*_phys_live.txt`), and PolyChord stops when that
  falls to a fixed fraction of the evidence already banked. The volume shrinks
  one e-fold per `nlive` dead points, which turns "how much further that ratio
  has to fall" into a count of dead points, and the run's own dead-point rate
  turns that into hours.

  The position within that total does not wait for the next checkpoint.
  PolyChord rewrites `chains/` every `nlive` dead points, so `ndead` is frozen
  between writes and everything derived from it sits still and then jumps by
  fifty: the live 16-rank R2D2 search read `~38% done, ~8h12m left` two hours
  into an interval that ended with it past half way. The evaluation
  directories are not frozen, and the sampler spends a near-constant number of
  them per dead point, so the evaluations banked *since* the checkpoint convert
  straight back into dead points (`dead_points_now`). The total still comes
  from the checkpoint's own `ndead`, because the `log(Z)` and live points it
  needs were written by the same checkpoint; only the position is carried, and
  it is printed with its own `~` whenever it differs from the checkpointed
  count. **stage** still reports the raw count with its age, which is the
  honest statement of what PolyChord last wrote down.

  Measured against a live wsclean search (`--nlive 5 --num-repeats 2
  --mpi-procs 3`, 48 dead points over six checkpoint writes): sampled once a
  second, the carried count immediately before each write was 0, +7, 0, +1,
  -1 and -5 out of what that write then revealed, where the raw count was
  short by the whole write every time (-7, -7, -6, -7, -6, -9).

  The stopping fraction is measured rather than taken from the documentation.
  `precision_criterion` defaults to 1e-3, but the two searches on this host
  that ran to natural termination (wsclean, nlive=50, seeds 123 and 372)
  stopped at 446 and 463 dead points where 1e-3 predicts 350 for both; the
  ratio they actually reached was 1.3e-4 and 9.6e-5. Calibrated to their mean
  (`TERMINATION_EVIDENCE_RATIO`), replaying those two runs through the shipped
  code forecasts 452-459 from `ndead=100` onward - within 3% of both, and
  stable across the whole run rather than drifting. It also holds at a very
  different `nlive`: a wsclean search at `--nlive 5 --max-ndead -1` was watched
  live from 12% onward forecasting 45-47 dead points, and terminated naturally
  at 47. Recalibrate there if a PolyChord upgrade or a non-default
  `precision_criterion` moves it - and in `_NS_TERMINATION_EVIDENCE_RATIO` in
  `scripts/lib/progress-bar.sh`, whose pinned status line forecasts from the
  same model so that the bar and this line cannot disagree; its self-check
  fails if the two copies drift.

  Withheld inside the first e-fold, where the live set is still the prior and
  the estimate would be reporting its own constant rather than this run, and
  withheld from a run that is not going: a stopped run's remaining dead points
  are not remaining, they are lost. An explicit `--max-ndead` is a hard stop
  the sampler hits first, so it is used directly and printed without the `~`.
- **ranks** - found by the `--output-dir` they were launched with, so no ranks
  means no run. `busy-waiting` counts the ranks that spent a whole one-second
  sample on CPU. Open MPI's `ob1` busy-waits, so a rank blocked in a collective
  burns a core and looks identical to a working one on `%CPU`; sampling twice
  and taking the increment is what separates them, and it does so whenever the
  rank got stuck. (Cumulative CPU over the process's whole life does not: a run
  that works for an hour and then wedges still reads under any threshold for
  most of another hour, because the real work already done outweighs the
  spinning.) On its own the count means nothing, and not just because some
  spinning is normal: four independent measurements of one healthy 16-rank
  R2D2 run, across an hour, gave 1, 2, 7 and 15, as the sampler alternated
  between imaging in parallel and synchronising. Each was reproducible for as
  long as its phase lasted - **a count sampled for a minute lands there just as
  confidently as one sampled once**, which is what made every one of those
  readings persuasive to whoever took it. So it is reported as a deadlock only
  when all but one are burning CPU **and** nothing has completed for a minute,
  and it is that second clause that does the work.
- **resources** - what the run is actually holding, so the host's free memory
  below has an owner. Taken over every live process carrying the run
  directory, not just its ranks: a rank is ~10MB and the imager worker it
  talks to is ~3.3GB, and the workers name the run by their `--fifo-dir`
  rather than by `--output-dir`. Everything except this tool's own process
  tree, which carries the run directory too whenever the run is named as an
  argument: a finished run reported by path used to print `0.1GB resident over
  3 processes` - the report measuring itself. RSS, so a page shared between
  processes is
  counted once per holder - overcounting, which is the safe direction for a
  number read to answer "will another run fit". Cores busy is the same one
  second CPU sample as the spin check, summed over those processes, so it
  measures the imaging rather than the ranks waiting on it.

  Swap is shown beside it because RSS excludes it, so a run the host has
  squeezed reads as holding *less* memory than it does - and the pages it is
  missing cost a disk read the next time it touches them, which surfaces as
  slow evaluations and never as a failure. A process is warned about when
  more of it is in swap than in memory **and** what is out there is at least
  200MB (`PAGED_OUT_MB`, the smallest per-rank footprint
  `scripts/lib/rank-budget.sh` budgets - a whole WSClean rank). Both clauses
  are needed: on the live 16-rank R2D2 search here every healthy imager worker
  kept ~70MB of cold startup pages swapped against a 3.2GB footprint and cost
  nothing, while nineteen ranks and shims sat at 10MB resident against 14MB
  swapped and were "mostly on disk" by ratio alone. The one process worth
  naming was a single imager worker at 52MB resident against 2.9GB swapped -
  parked, with 2.9GB to read back before its next evaluation, and invisible in
  every other number on the page.

  That warning is then held back unless the kernel says something is actually
  waiting on those pages - **pressure** below. Being on disk and being read
  back off it are different things, and only the second costs the run anything:
  the parked worker above sat there for hours while the host reported 0.02% of
  each five minutes stalled on memory, i.e. ~0.06s of waiting across five
  minutes in which the run scored ~110 evaluations. `./ri health` warned and
  exited 1 on every call for a cost of nothing. The swapped total stays on the
  **resources** line either way; what is withheld is the claim that it is
  slowing the run down.
- **pressure** - the kernel's own Pressure Stall Information (`some avgN` from
  `/proc/pressure/memory` and `/proc/pressure/io`): the percentage of the last
  minute, and of the last five, during which at least one task was stalled
  waiting on that resource. Every other resource number here says how large a
  shortage is; this is the only one that says what it is *costing*, which is
  why the paged-out warning above and the host's own memory warning are both
  decided by it - at or above `MEMORY_STALL_PERCENT` (5%, ~250x this host's
  idle baseline and ~1.3s per evaluation on the live R2D2 search, which is
  where it stops hiding inside the ordinary spread of evaluation cost) the
  host is short of RAM and the answer is fewer ranks or fewer concurrent runs,
  not a faster imager. Two averages because they answer different questions: a
  minute for "is this happening now", five for "has it been happening long
  enough to explain the run's numbers". The line is omitted, and every
  decision that rests on it falls back to its old unconditional form, on a
  host with no PSI - macOS, or a kernel before 4.20.

  The host block reports swap in use but never warns on it: swap that is in
  use may have been paged out days ago and cost nothing since. Whose pages
  those are is the actionable question, and that is the per-run warning.
- **memory** - the same cost measured by the run itself instead of sampled off
  the host: the median `peak_memory_bytes` its evaluations recorded, and that
  multiplied out over `NS_MPI_PROCS`, because what the host has to hold is
  every rank's worker at once. Two things **resources** above cannot do. It
  survives the run - a finished or OOM-killed run has no processes left to
  sample, and this is the only place its footprint is still on record. And it
  is the measurement behind `scripts/lib/rank-budget.sh`, which sizes every
  run on this host from a fixed 3500MB per R2D2 rank and 200MB per WSClean
  rank, hand-measured once against one set of images and carrying its own
  `ponytail:` note asking to be re-measured if the imaging stack changes.
  This line is that re-measurement, running continuously: the live 16-rank
  R2D2 search here reads 3.3GB against the 3.42GB budgeted, so the estimate
  is 3% conservative and the run fits.

  Not "per evaluation", because the two imagers measure different things
  under the same key: WSClean's is GNU `time -v` on that one imaging run,
  R2D2's is the warm worker's own high-water RSS and therefore a running
  maximum over the rank's whole life. Both answer "what does one rank have to
  be budgeted", which is the question; neither is an average, and R2D2's can
  only ever rise.

  The last-50 figure is printed only when it has moved by more than 2x.
  Footprint is flat on this host - 3.45-3.57GB across 6,600 R2D2 evaluations,
  0.05GB dead flat across 1,800 WSClean ones - so a second number that agrees
  is noise, while a doubling is a parameter region or a leak that the budgeter
  cannot see coming. Reported and not warned on: the host block below already
  warns on the only threshold that is a fact rather than an inference, free
  memory under the headroom `rank-budget.sh` reserves.
- **disk** - the resource nothing here reserves, checks or frees, and the only
  one that only ever grows. An evaluation directory keeps its measurement set,
  its `.mat` and the imager's output - ~1.7MB on this host - and nothing
  deletes it (`./ri clean` deliberately leaves `results/` alone), so a live
  R2D2 run writes ~2.6GB/hour and one WSClean run left 18GB behind. The
  projection is that rate against what the filesystem has left: there is no
  other place that would say so before the run ends on ENOSPC, hours of
  imaging from its last checkpoint.

  It warns when that projection is shorter than **how much longer the run
  needs**, not under a fixed number of hours, because space running out after
  the search is over is not a problem the run has. A WSClean smoke run 35
  seconds old, writing 29.6GB/hour against 218GB free, projected `~7h` and
  warned under the old 12-hour floor - `RUNNING - 1 WARNING`, exit 1 - while
  finishing in minutes; a multi-day R2D2 search with 20h of space never
  tripped that floor at all.

  "How much longer" is **forecast**'s hours left when there is one. There is
  none before PolyChord's first checkpoint writes `chains/*.stats`, which is
  the whole of a short run and was still true seven hours into the 16-rank
  R2D2 search on this host, so the fallback is the run's own age: a run is
  assumed to have at least as long ahead of it as behind, and the warning
  fires once the space left is shorter than the run so far. The warning says
  which of the two it used (`against ~4h20m still to run`, `against a run
  already 6h11m old`). The `disk` line itself always reports the projection,
  warned on or not.

  Estimated from a **strided sample of 20 evaluations**, not a walk. `du -s`
  on one live run cost 3-5s of I/O against the disk that run is using, which
  is not what a read-only check should do to it; 20 stats cost milliseconds
  and landed within 1% (7.5GB against a true 7.57GB). Strided over the run's
  life rather than taken from its tail because evaluation size follows the
  parameters and a nested-sampling run concentrates: the newest 20 read
  1.45MB where the same run averaged 1.68MB.
- **restarts** - how many times the run died and started itself again from its
  checkpoint, and when the last one was, read from `restarts.log`. Shown only
  when there were any, and never warned on: the crash was survived and the run
  is healthy now, so warning would make `./ri health` exit nonzero for a run
  that is fine. It is still the line to read first on a run that looks slower
  than it should. See "A run that dies restarts itself".
- **failures** - evaluations that scored `FAILURE_OBJECTIVE` (100.0), and
  `meqserver-wedged.log` lines. **This is the one that a run can pass every
  other check and still fail.** PolyChord maximizes, and a real
  `total_rms_jy` is ~0.008, so failed evaluations are the best points the
  search has ever seen and it concentrates its live points on them. A run with
  a missing checkpoint mount or an OOM-killed worker reports "the imager fails
  catastrophically here", which is exactly the conclusion this repo exists to
  draw.

  Asked of **the last 50 evaluations as well as of the whole run**, because a
  whole-run ratio cannot see an imager that broke part-way through: three
  hours healthy and twenty minutes broken is ~2% failures overall and silent,
  while every point the search adds from that moment on is a failure. Half the
  window is the bar, which is not a tuned number - across the 37,000
  evaluations of the six real runs on this host the failure count is zero, so
  any sustained burst is a fault and the threshold only has to clear noise
  there is none of.
- **stalls** - gaps between evaluations more than 10x the run's own median,
  and never less than 2s. Relative because WSClean lands 30-50 evaluations a
  second and R2D2 roughly one every two, so no fixed threshold suits both.
  Before the watchdogs above, the MeqTrees deadlock cost 23-27% of wall clock
  here; after them, 0. A gap containing one of the restarts below is skipped -
  the run was not running, the reason is known, and it is already on the
  `restarts` line. A self-healed WSClean run whose only gap over the threshold
  was its own 12s restart used to report "13% of wall clock lost", which reads
  as the deadlock this number exists to size. That window opens a second before
  the gap, because `restarts.log` stamps whole seconds while evaluation mtimes
  are fractional and the crash lands in the same second as the last evaluation
  that survived it - so the stamp reads just *before* the gap it explains, and
  a self-healed run was warned about for having healed itself (gap start
  `...45.09` against a `...45` stamp, missed by 90ms).
- **host** - free memory against the headroom `scripts/lib/rank-budget.sh`
  keeps, free disk on the filesystem holding `results/` (nothing reserves it,
  so this is the denominator the per-run projection above divides), and
  `ri-ns-sidecar-*` containers whose launching process is gone. A
  killed run leaves those holding ~3.4GB per R2D2 rank, which would count
  against every later run's memory budget. Reported here because it is
  something to know about the host; not something to act on, because the next
  run removes them itself - `ns_reap_leaked_sidecars` in
  `scripts/lib/rank-budget.sh` runs before the run reads free memory, so the
  memory a dead run is sitting on is freed rather than sized around. The
  launcher's pid is in the container name, and pid reuse can only make it skip
  a container, never take a live one. The pid is not the whole rule: a run
  script killed with SIGKILL leaves the *search* going - the ranks are children
  of `containerd-shim`, not of the shell - so on the pid alone a live 16-rank
  search's three containers read as leaked, and both this warning's `docker rm
  -f` line and `ns_reap_leaked_sidecars` would have killed it. Each container
  therefore carries a `ri.run-dir` label naming the run that started it
  (`sidecar_launch` in `scripts/lib/start-sidecars.sh`), and a container whose
  labelled run still has processes - the same `ns_run_is_live` check `./ri
  resume` and `./ri search --output-dir` refuse on - is never leaked whatever
  its pid says. Containers started before the label existed have none and fall
  back to the pid, as do the per-rank fallback containers `common.py` starts,
  which belong to a rank rather than to a run.

- **why it stopped** - a stopped run's warning quotes its `run.log`, which is
  where the run's own output is kept. Everything else on disk says *that* a
  run broke; only this says why. It quotes the last line naming an error
  rather than the last line outright, **and how many ranks said it**:

  ```
  run.log ends "TypeError: _connect_shell_started_worker() ..." (x15 ranks)
  ```

  The count is the diagnosis. An MPI crash leaves one traceback per rank, so
  the real failure here was the same stack fifteen times over and a plain tail
  of the file lands on it only by luck. Every rank reporting the same error is
  a code bug that every rank hits deterministically - that was the `run.log`
  captured from PR #66. One rank alone is a flaky worker, an OOM kill, or bad
  luck on one evaluation, and those want opposite responses. For a run that
  stopped without a traceback it falls back to the last non-empty line, which
  is PolyChord's own last word on where it got to.

### A run that dies restarts itself

PolyChord checkpoints continuously, so a run that dies at hour three already
holds everything it needs to carry on. What it did not have was anything to
start it again: a worker that stopped answering (`WORKER_DIED`), a meqserver
that wedged past the in-worker watchdog, an OOM kill - each one ended a
multi-day search that then sat dead until someone noticed.

`run_with_retries` in `scripts/lib/progress-bar.sh` wraps the run and restarts
it in place, up to `--retries` times (default 2). The restart is an ordinary
resume: same `OUTPUT_DIR`, PolyChord reads its own `.resume` file and the
evaluations already on disk are adopted, exactly as `./ri resume` does.

**It only retries an attempt that made forward progress**, measured in
evaluations the attempt actually scored. That guard is what stops it spinning:
a code bug every rank hits deterministically, a missing image, a bad parameter
space - all of those fail before a single evaluation is scored, so they stop
immediately instead of failing three times as slowly. Only something that
killed a run which was working gets another go.

The measure used to be dead points added, which silently disabled the retry
for most of a real run. PolyChord writes `chains/` only every `nlive` dead
points, so inside that interval - up to seventy minutes on the 16-rank R2D2
search, and the whole of a fresh run before its first checkpoint - the
dead-point count is frozen at the number the attempt started from, however
much imaging happened. A real search SIGKILLed at 31 scored evaluations and 0
dead points printed `not retrying: ... added no dead points` and stayed dead;
the same kill now logs `attempt failed (exit 137) at 31 evaluations` and the
run finishes. Retrying does not repeat that work either: the restart adopts
the finished evaluations and serves those points from its cache. An
evaluation directory with no `metrics.json` is one that was in flight when
the run died, and does not count - the next attempt deletes it.

Two things to know about a restarted run:

- It reuses the sidecar containers but not their pooled workers, which exited
  on EOF when the dying ranks closed the FIFOs. Each rank waits out
  `_connect_shell_started_worker`'s 10s deadline in `common.py` and then starts
  its own worker inside the same sidecar - still one long-lived worker per
  rank, so the price is that one-off wait and not a per-evaluation penalty. A
  real killed WSClean search scored 216 evaluations/min over the 53 before the
  kill and 219/min over the 34 after, with a 12.1s gap across the restart.
  Deliberate at that price: re-launching the pool would mean a second reader on
  a FIFO whose old worker may not have exited yet, and two readers split the
  messages between them.
- Each restart appends a line to `restarts.log` in the run directory, and
  `./ri health` shows the count and the latest one. It is reported, not warned
  on - the run is fine right now - but whatever killed it once will do it
  again, so the line is worth reading.
- **It re-sizes itself.** The rank count in the command is what
  `ns_budget_ranks` could afford when the run *started*, and on a host several
  sessions share that is not a fact about now - the common way a long search
  dies is another session's run growing into it, so replaying the number puts
  the restart straight back into the OOM killer, which does not fail the run:
  it scores `FAILURE_OBJECTIVE`, which PolyChord maximizes. So the count goes
  back through the memory guard before each restart, exactly as `./ri resume`
  does with the `NS_MPI_PROCS` in `run.env`, and the run says
  `retry 1 of 2, re-sized to 4 ranks to fit the memory free now` when it
  moved. Only ever downwards, because `ns_budget_ranks` never returns more
  than it is asked for, and clamping down is free here: PolyChord's checkpoint
  carries live points rather than ranks, and the FIFO pool laid out for the
  original count is unused by a restart anyway (see above), so the spare FIFOs
  simply go unread. If not even one rank fits any more the run stops there -
  `not retrying: there is no longer memory for even one rank` - rather than
  spending a restart to fail the same way.

**The budget is for a crash loop, not for the run's lifetime.** An attempt
that ran for `NS_RETRY_RESET_SECONDS` (1800) before dying hands the retry
budget back, so the count is of failures that keep coming straight back rather
than of restarts ever made. Without that the counter only climbed: a
multi-day R2D2 search that healed itself twice on day one was out of retries
for the rest of the week, and the third unrelated OOM kill - hours of imaging
later - ended it exactly the way `--retries 0` would have. Half an hour is
~70x a single R2D2 evaluation and ~150x the 12.1s a restart itself costs, so
an attempt that clears it plainly got past whatever killed the last one.
Resetting too eagerly is the safe direction, because a retry still has to have
scored evaluations: the worst case is a run that grinds forward slowly, not
one that spins.

When the budget is gone the run says so in `run.log` - `not retrying: 2 of 2
restarts used and this attempt (exit 137) died inside 1800s` - rather than
just stopping. `./ri health` reads that log tail back for a stopped run.

Set `--retries 0` to get the old behaviour, where the first failure ends the
run.

#### A run that hangs instead of dying

`run_with_retries` can only act on a run that *exits*, and the worst failure
here does not. PolyChord calls the likelihood from Fortran, so a single rank
that stops answering leaves every other rank blocked in a collective that never
completes: every core busy, nothing landing, no exit status, and `./ri health`
correctly reporting a live run. The in-worker timeouts in `common.py` only
cover a worker that was *asked* for a reply; a deadlock between the ranks
themselves is asked nothing.

`_ns_stall_watchdog` in `scripts/lib/progress-bar.sh` is the backstop. It
watches the one thing true of every healthy run and false of every hung one -
evaluations finishing - and after `--stall-timeout` seconds with none, writes
the reason into `run.log` and kills the run, which turns the hang into the
crash `run_with_retries` already handles. It runs whether or not there is a
terminal, because a multi-day search is exactly the thing somebody starts
under `nohup`.

The kill is by command line rather than by the pid the run script holds: that
pid is the `docker exec` client, and the ranks are children of
`containerd-shim`, not of it, so killing the client leaves the run running.
`ns_run_process_pattern` builds the pattern, anchored on the run's own
`--output-dir` so a search somebody else is running on the host is untouched -
the same pattern `ns_run_is_live` builds on, which is what `./ri resume` and
`./ri search --output-dir` both use to refuse a run that is still going.

**The default is 7200s, and it is deliberately far above anything legitimate.**
`IMAGING_REPLY_TIMEOUT` in `common.py` already lets a single evaluation take an
hour before its worker is declared dead, so anything shorter would kill runs
that machinery is still working on. Against that, the widest gap between
evaluations measured over 6.3 hours of a live 16-rank R2D2 search was 23.5s.
This is the backstop for when nobody is watching; `./ri health` answers the
same question in seconds for somebody who is. `--stall-timeout 0` disables it.

A restart adopts what the previous attempt evaluated **whether or not
PolyChord left a checkpoint behind**, and that distinction is load-bearing.
PolyChord writes `<file_root>.resume` at its first checkpoint, so an attempt
killed before that leaves evaluations on disk and no resume file. Adoption used
to be conditional on the resume file, so such a restart began at eval id 1 on
top of the previous attempt's directories, which `simulate_measurement_set`
creates with `exist_ok=False`. One rank died on `FileExistsError`; PolyChord
calls the likelihood from Fortran, so that traceback unwound one rank and left
every other one waiting forever in a collective that never completed - a
16-core R2D2 restart that burned every core, landed nothing, and never exited
for `run_with_retries` to give up on. Re-sampling from scratch costs nothing
either way: PolyChord redraws the same points from the same seed and the
adopted cache answers them without imaging.

The same hang is now closed at its source as well: the likelihood aborts the
whole job (`abort_run`, i.e. `MPI_Abort`) on **any** exception it does not
expect, not only on `WorkerDied`. A bug in a rank ends the run with the reason
on stderr and every finished evaluation still on disk, which `run_with_retries`
can act on, instead of hanging it.

`./ri self-check self-heal` is the end-to-end check of all of the above, and
it is part of `./ri self-check`. It starts a real WSClean search on a throwaway
directory and breaks it twice, in the two ways that recover through different
machinery:

- **`SIGKILL` once it has scored 8 evaluations** - fewer than `--nlive`, so the
  kill lands before any checkpoint, which is the regime that was broken.
- **`SIGSTOP` on one rank of a second search**, which is the hang above: the
  job is fully alive and simply not progressing, so only the stall watchdog can
  notice. Run with `--stall-timeout 20` and a 2s poll so it costs a minute
  rather than two hours; the code path is the shipped one.

Both then assert that the run restarts itself, records the kill in
`restarts.log`, keeps the evaluations the first attempt scored, writes
`summary.json`, and comes out of `./ri health` with nothing on its own
headline. Its headline rather than the exit status, which is 1 for a host
warning too: this host is shared, and the check failed on the just-finished
search's own containers, still running for the ~0.4s `_sidecar_remove` takes to
remove them in the background after their launcher pid is gone. The
wait for recovery is bounded, because the failure it is most likely to catch is
a hang rather than an exit. ~90 seconds and ~0.6GB, so it is safe to run beside
another search. Fixtures cannot stand in for it - every bug found in this
machinery so far (the retry reading a checkpoint-frozen counter, the stall
accounting refusing to excuse a run's own restart, the restart colliding with
its own evaluation directories) passed the fixtures and failed a real kill.

### Finding and resuming a run that stopped

A run writes `summary.json` only once PolyChord returns, so a run directory
without one stopped early. `./ri runs` is the list:

```console
$ ./ri runs
RUN                        ALGORITHM  STATUS      EVALS
r2d2-vlaa-20260828T2054Z   r2d2       running     6882
r2d2-vlaa-20260827T1015Z   r2d2       resumable   659
wsclean-vlaa-20260827T09Z  wsclean    complete    1706

1 run still going. Check on it with:
  ./ri health r2d2-vlaa-20260828T2054Z

1 run stopped before finishing.
Continue where it left off, keeping every evaluation already done:
  ./ri resume r2d2-vlaa-20260827T1015Z
```

`./ri resume <run>` continues it in place. No flags: each run records what it
was started with (`run.env`, written at startup, holding the values actually
used), so a resume cannot silently become a different search. PolyChord's own
checkpointing supplies the live points and the evaluations already on disk are
adopted, so their ids carry on and no point is paid for twice.

The one setting a resume does not replay verbatim is the rank count. What
`run.env` holds there is the memory guard's own output from when the run
started, not a number anyone chose, and on a shared host the memory that was
free yesterday is another session's run today - so `./ri resume` puts it back
through `ns_budget_ranks` (scripts/lib/rank-budget.sh) and says what it did:

```console
$ ./ri resume wsclean-vlaa-20260828T0221Z
NOTE: wsclean ranks 3 -> 1 (4400MB available, 0MB reserved by other runs, 200MB per rank)
Resuming wsclean-vlaa-20260828T0221Z (wsclean, 31 evaluations already done, 1 rank)
```

Clamping down is free: the checkpoint carries live points, not ranks, and the
run above finished normally after dropping from 3 ranks to 1. Without it a
16-rank R2D2 run - 53GB - resumed onto a busy host has every rank that does
not fit OOM-killed, and an OOM-killed worker is scored `FAILURE_OBJECTIVE`,
which PolyChord maximizes: the run does not fail, it reports the corner of the
parameter space where it ran out of memory as its best discovery. A resume
that cannot afford even one rank stops here rather than at evaluation 1.
`--mpi-procs` on `./ri search` is still obeyed as typed and only warned about
- the difference is that a resume did not type anything.

`STATUS` is `complete` when `summary.json` is there, `running` when the run
still has a process driving it, `resumable` when neither but a PolyChord
`.resume` file is there, and `incomplete` when a run stopped before it
checkpointed anything. `./ri runs --incomplete` lists only the ones needing
attention, and `--json` is the machine-readable form.

A live run is indistinguishable on disk from one that stopped - same missing
`summary.json`, same checkpoint - so liveness is read from the process table,
the way `./ri health` picks the run to report on. Before that, `./ri runs`
called the live search `resumable` and printed `./ri resume` for it, which is
an instruction to start a second MPI job over the live one's own checkpoint and
FIFO directories. `./ri resume` refuses that outright:

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

The HTML report lists unfinished runs at the top of its index for the same
reason, because they have no page of their own: a run that stops does not
appear as failed there, it simply is not there, which is the easiest kind of
problem to miss.

This covers every way a long run stops - the memory guard giving up, a Ctrl-C,
a reboot - not only the one the previous section is about. A fresh run has no
resume file and starts clean, so leaving checkpointing on costs nothing.

### Running WSClean and R2D2 at the same time

Fine, and barely worth it. A default WSClean search is ~3s and ~1.3GB against
R2D2's ~100s and ~27GB, so running WSClean first and R2D2 afterwards costs
about three seconds against perfect overlap. Measured over three interleaved
pairs, a concurrent WSClean run changed R2D2's wall clock by under 1%.

What does need care is two *R2D2* runs, or any run started while another is
warming up. Give both an explicit `--mpi-procs` whose R2D2 total stays under
16 ranks, and prefer giving cores to R2D2: it is ~100x more expensive per
evaluation (~14s against ~0.15s), so a core spent on WSClean buys far less
progress.

The run scripts also export these for the containers they start. They have
no `./ri` flags. Unset, the Python ranks fall back to starting their own
workers, which is slower.

| Variable | Meaning | Set by the run scripts to |
|---|---|---|
| `NS_SIDECARS` | JSON map from image name to that image's long-lived sidecar container | Exported by `scripts/lib/start-sidecars.sh` |
| `NS_SIMULATE_FIFO_DIR` | Directory holding the per-rank `<rank>.in` / `<rank>.out` FIFOs of the pre-warmed simulate workers | `${OUTPUT_DIR}/.simulate-workers` |
| `NS_R2D2_FIFO_DIR` | Same, for the pre-warmed `r2d2_serve.py` imaging workers | `${OUTPUT_DIR}/.r2d2-workers` (R2D2 run only) |

## Parameter space

VLA configuration is an outer-loop dimension. The runs here only use `VLA.A`.

PolyChord dimensions for both algorithms:

| Dimension | Range | Meaning |
|---|---:|---|
| `dynamic_range` | `1e2` to `1e3` | One-Jy source divided by thermal-noise sigma |
| `observation_minutes` | `4` to `10` | Total requested observing time |
| `channel_count` | `2` to `6` | Number of frequency channels |
| `start_frequency_hz` | a receiver band (see below) | First channel frequency |
| `channel_width_hz` | `0.5e6` to `2.0e6` | Uniform spacing between channels |
| `source_offset_fraction` | `0.0` to `0.35` | Source offset from the phase centre, as a fraction of the image half-width |

Channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

The current box for every dimension is in `defaults.toml`, which is the one
authoritative copy - this table names them, not their exact ranges.

### Toggling dimensions on and off

Every `[[parameter_space]]` entry in `defaults.toml` takes `enabled` (default
true). Setting `enabled = false` pins that dimension out of the search
instead of deleting it: `cube_to_params()` fixes it at its `default` (falling
back to `min` when no `default` is given) rather than drawing it from the
cube. `source_offset_fraction`, for example, disables back to the old
hard-coded centred source, because its `min` already is `0.0`.

Two ways to see and change this without editing the file:

```
./ri params                                     # what is searched, what is pinned
./ri search wsclean --disable-param source_offset_fraction --enable-param channel_count
```

`--enable-param` / `--disable-param` are repeatable and set `NS_ENABLE_PARAMS`
/ `NS_DISABLE_PARAMS` (comma-separated names), which override `enabled` in
defaults.toml for that one invocation - an env-var edit, not a file edit, for
a one-off search. `--enable-param` wins if a name is passed to both.

Toggling a dimension changes PolyChord's dimension count, so - like
reordering `[[parameter_space]]` - it invalidates existing chains, and
`merge-nested-sampling-runs.py` refuses to merge runs whose `parameter_space`
differs.

### Receiver bands

A telescope only receives inside its bands, so `start_frequency_hz` is not
scaled onto a plain box: the `[[receiver_band]]` list in `defaults.toml` names
the bands, each gets an equal share of that unit-cube dimension, and the start
frequency is uniform inside the band its share picks. Equal share per band
rather than uniform across the union of them, or the 32 MHz-wide 4-band would
come up about once in 1500 draws and never actually be searched.

The committed list is the VLA's:

| Band | Range | Band | Range |
|---|---:|---|---:|
| 4 | 54-86 MHz | X | 8-12 GHz |
| P | 224-480 MHz | Ku | 12-18 GHz |
| L | 1-2 GHz | K | 18-26.5 GHz |
| S | 2-4 GHz | Ka | 26.5-40 GHz |
| C | 4-8 GHz | Q | 40-50 GHz |

Nothing in the code is VLA-specific: another telescope is a matter of
replacing the list (any number of bands, in any order, gaps allowed).

### Fitting the window into the band

`channel_count` and `channel_width_hz` are drawn from their own boxes, knowing
nothing about frequency, so a window can easily be wider than the room left
above the start frequency that came up. `fit_spectral_window()` in
`common.py` fits it to that room, giving up as little as possible at each
step:

1. the window fits - keep the draw;
2. it does not - narrow the channels until it does, if that stays at or above
   `channel_width_hz`'s min;
3. it would go below that - hold the width at the min and drop channels
   instead, if that stays at or above `channel_count`'s min;
4. even the smallest window does not fit, so the start frequency is too close
   to the top of its band to hold anything - draw another start frequency and
   start over.

The redraw steps a fixed distance (the golden ratio conjugate) around the unit
interval rather than drawing from an RNG, which spreads successive tries
across every band and keeps the prior transform a pure function of the cube -
what PolyChord requires, and what makes the `theta -> cube -> params` round
trip a fixed point: a fitted window is one that already fits, so re-deriving it
changes nothing.

Two consequences worth knowing. The fitting only ever gives ground, so the
mins are hard floors and the maxes are never exceeded - but a run does measure
narrower channels than it drew, and the prior on `channel_width_hz` is no
longer flat: it is deformed towards the narrow end near the top of each band.
And a box whose *smallest* window - `channel_count`'s min at
`channel_width_hz`'s min - fits no band at all cannot be fitted from any start
frequency, so `check_channel_box_against_bands()` refuses it at load.

### What the fitting costs

Every draw that is fitted rather than kept is a draw whose parameters are not
the ones the sampler asked for, and every redraw is work thrown away, so both
are counted. `WINDOW_FIT_STATS` tallies them and each run's `summary.json`
carries the result under `spectral_window_fitting`:

```json
"spectral_window_fitting": {
  "draws": 2000, "as_sampled": 1972, "width_reduced": 25, "count_reduced": 3,
  "redrawn_draws": 5, "redraws": 5, "seconds": 0.0017,
  "seconds_per_draw": 8.6e-07
}
```

The run also prints that line when it finishes. With the committed box the
fitting is cheap and rare - the window tops out at 12 MHz against a 32 MHz
narrowest band, so it only bites near the top of a band - and the cost is
microseconds a draw against seconds an evaluation. Widen `channel_count` or
`channel_width_hz` and the reduced counts climb, which is the number to watch:
it is how much of the box the run is not really searching.

`self_check_spectral_window()` is the guard - it samples the cube, asserts
every fitted window lands inside a band and inside the configured channel
boxes, forces each rung of the ladder above, and asserts the round trip comes
back to the same parameters.

The band guarantee is the sampler's. `simulate_point_source_ms.py` takes
`--start-frequency-hz` and `--channel-width-hz` as given, so a hand-run
simulate (or a smoke test) can still ask for whatever it likes.

### Cell size

The cell size is derived per evaluation, not fixed. R2D2 sizes its own pixels
from the sampling pattern it is handed - upstream `src/utils/io.py` takes the
longest projected baseline in wavelengths and sets

    image_pixel_size = 206265 arcsec / (super_resolution * 2 * max_proj_baseline)

so the WSClean runner applies the same formula (`image_pixel_size_arcsec()` in
`common.py`) to the `observation.max_proj_baseline_lambda` the simulator
records, and passes the result as `-scale`. Both imagers then reconstruct the
same sky at the same resolution, and each WSClean evaluation records the value
it used as `image_pixel_size_arcsec`.

A fixed `-scale` cannot work here, because `start_frequency_hz` spans 54 MHz to
50 GHz while the VLA-A maximum baseline does not move: against the 1 arcsec
cell this used to pass, the synthesized beam is ~31 arcsec at the bottom of
that range and ~0.04 arcsec at the top. The search would have been measuring
how badly WSClean's grid was mismatched to the sampled frequency - and only
WSClean's, since R2D2 rescaled either way.

`super_resolution` is 1.5, R2D2's own default, now written into the R2D2
config explicitly rather than left implicit, because WSClean's `-scale` is
derived from it.

### Source offset

`source_offset_fraction` moves the source off the phase centre, at a fixed 30
degree position angle (non-axis-aligned, to avoid the symmetries a purely
horizontal or vertical offset would have). At `0.0` it reproduces the old
hard-coded behaviour exactly: no bandwidth smearing, no time smearing, no
w-term, no pixel-interpolation error, and `point_source_forest.py` skips
K-Jones outright.

`source_offset_to_lm()` in `common.py` converts the fraction to an (l, m)
offset in arcsec using a *nominal* image half-width - `image_pixel_size_arcsec()`
against VLA-A's ~36 km maximum baseline and the sampled frequency, not the
`max_proj_baseline_lambda` the simulator will actually record - because the
source position has to reach `simulate_point_source_ms.py` before the MS (and
its real baselines) exist. `compute_image_metrics()` places the truth pixel at
that same offset (`source_pixel()`), so an off-centre evaluation is not scored
against a source that is not there.

Caveat: `ms_to_r2d2_mat.py` writes only `u` and `v` (see the bridge table
below) - `w` is dropped, so R2D2 sees a coplanar 2-D array while WSClean does
not. `source_offset_fraction`'s box tops out at 0.35 to stay inside the
small-field regime that keeps this an acceptable approximation rather than
comparing the two imagers on different physics.

Fixed hyperparameters (not searched) on every evaluation:

**WSClean:** `-niter 100`, `-auto-threshold 3.0`, and a `128x128` image,
recorded in `summary.json` under `wsclean_fixed_hyperparameters`.

**R2D2:** `128x128` image size (the same footprint as the WSClean run),
`num_iter 25`, `architecture unet`, `num_chans 64`, `ckpt_path
/checkpoints/R2D2_A1`, and `ckpt_realisations 1`, recorded in `summary.json`
under `r2d2_fixed_hyperparameters`.

## MS to R2D2 `.mat` bridge

R2D2-RI reads visibilities from a MATLAB `.mat` file via `load_data_to_tensor()`
in the upstream `src/utils.py`. The nested-sampling simulator produces a CASA
Measurement Set (`sim.ms`) that WSClean consumes directly. The R2D2 run adds
`scripts/lib/nested_sampling/ms_to_r2d2_mat.py`, which the rank's simulate
worker runs in-process inside the MeqTrees sidecar (python3-casacore plus
scipy), and which writes the minimal field set R2D2 loads without flag
metadata:

| Field | Meaning |
|---|---|
| `u`, `v` | UV coordinates in wavelengths, flattened across rows and channels |
| `y` | Complex visibilities for correlation index 0 (parallel-hand Stokes I) |
| `nW` | `sqrt(WEIGHT)` from the MS (sqrt of inverse variance) |

Imaging weights are generated inside R2D2 when `data_weighting: True` in the
per-evaluation YAML config. The converter does not replicate the bundled
`data_3c353.mat` pruning or tau-compressed weight fields.

## Metrics and objective

For each sample, the pipeline records:

| Metric | Source |
|---|---|
| `snr` | Reconstructed image peak divided by off-source RMS |
| `log_snr` | `log10(snr)` |
| `off_source_rms_jy` | Off-source RMS in Jy/beam |
| `total_rms_jy` | RMS of (reconstructed image − one-pixel truth) over all pixels |
| `peak_jy_per_beam` | Peak absolute flux in the reconstructed image |
| `relative_l2_error` | Image residual versus the one-pixel point-source truth |
| `peak_flux_abs_error_jy` | Absolute centre-pixel flux error |
| `sigma_res` | Paper data-fidelity \(\overline{\sigma}_{\textrm{res.}}=\|\widehat{\mathbf{r}}\|_2/\|\mathbf{x}_{\textrm{d}}\|_2\) (final residual dirty over dirty) |
| `wall_seconds` | Imaging container runtime |
| `peak_memory_bytes` | Peak imaging memory: GNU `time -v` for WSClean; for R2D2 the imaging worker's own high-water RSS, which is a running maximum across that rank's evaluations |

PolyChord maximizes whatever value the run returns as its log-likelihood. The
default objective is `total_rms_jy` (RMS of the reconstructed image minus
the one-pixel truth, over all pixels).

An optional composite `badness` score is also available (higher means worse
reconstruction or a more expensive run):

```text
max(0, 3 - log_snr)
+ min(relative_l2_error, 10)
+ 0.05 * min(wall_seconds / 60, 5)
+ 0.02 * min(peak_memory_bytes / 2 GiB, 5)
```

### Choosing the objective (`--metric` / `NS_METRIC`)

Both `polychord_wsclean.py` and `polychord_r2d2.py` accept
`--metric <value>` (default `total_rms_jy`). The shell wrappers forward
`NS_METRIC`, whose default lives in `defaults.toml`, with the same value.
Resolution order:

1. `badness` - the composite formula above.
2. Any bare metric name from the table - use that raw value directly as the
   objective (including the default `total_rms_jy`).
3. Any other string - treat it as an arithmetic expression over the same metric
   names (for example `log_snr + 0.1 * wall_seconds`, or the composite formula
   rewritten by hand).

Expressions are compiled once at startup (before any Docker evaluations) and
evaluated in a restricted namespace: no Python builtins, metric names as locals,
and `math` module functions available by name. A typo or unsafe expression fails
immediately at startup.

PolyChord always maximizes the returned value with no automatic sign flip. The
`badness` composite is oriented so higher is worse. Raw metrics keep their
natural orientation: the default `total_rms_jy` search prefers higher
whole-image RMS error, `--metric snr` searches for the highest-SNR corner, and
a worst-SNR search must negate explicitly (`--metric "-snr"` or
`--metric "1/snr"`). `off_source_rms_jy` and `sigma_res` are also
higher-is-worse (noisier reconstruction / worse data fidelity); search for the
best corner with `--metric "-total_rms_jy"`, `--metric "-off_source_rms_jy"`
or `--metric "-sigma_res"`. Failed simulations or imaging runs still receive
objective `100.0`.

Each evaluation record and `summary.json` store the chosen value in an
`objective` field. `summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

## Profiling

Every run times each stage of every likelihood evaluation automatically -
there is no separate flag. To read the breakdown of a completed run:

```bash
./ri profile results/nested-sampling/wsclean-vlaa-<UTC timestamp>
# or directly:
uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC timestamp> [--json]
```

The numbers come from the `timing` block in each evaluation's `metrics.json`
and the run-level `profiling` block in that run's `summary.json`.

The same breakdown is available without the CLI: a run's HTML report page has a
collapsible "Profiling (where the run's time went)" section, shown whenever that
run's `summary.json` carries a `profiling` block (runs predating the profiler
instrumentation simply omit the section). It leads with a stacked bar of where
the worker-time went and then the same rows as the CLI table, so the report page
and `profile-nested-sampling-run.py` always agree - both call
`profiling_breakdown()` in `scripts/lib/nested_sampling/common.py`.

Both views show, per stage: the total, the mean per evaluation, the share of the
run's worker-time budget, and the evaluation count. Durations are rendered in
whatever unit carries their digits (`33ms`, `1.44s`, `39m 15s`, `1h 00m 45s`).

See [nested-sampling-profiling.md](nested-sampling-profiling.md) for what each
field means and for every measured (and rejected) optimisation behind the
current run scripts and images.

## Output files

### WSClean

Each likelihood evaluation:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/wsclean/recon-image.fits
evaluations/eval-*/wsclean/recon-dirty.fits
evaluations/eval-*/wsclean/recon-residual.fits
evaluations/eval-*/metrics.json
```

### R2D2

Each likelihood evaluation:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/r2d2_data.mat
evaluations/eval-*/r2d2_config.yaml
evaluations/eval-*/r2d2/r2d2_data/R2D2_model_image.fits
evaluations/eval-*/r2d2/r2d2_data/dirty_normalised.fits
evaluations/eval-*/r2d2/r2d2_data/R2D2_residual_dirty_image.fits
evaluations/eval-*/metrics.json
```

### Run summary and reports

Run-level summary, written only once PolyChord returns:

```text
summary.json
```

Everything the run printed, written as it goes and appended to by `./ri
resume`, so it survives a run that never reaches `summary.json`:

```text
run.log
```

This is the only artifact that records *why* a run stopped - a traceback out
of the PolyChord container reaches nowhere else. `./ri health` quotes its last
line for a stopped run.

One line per time the run died and restarted itself from its checkpoint, when
that happened at all (see "A run that dies restarts itself"):

```text
restarts.log
```

View completed runs (settings, evidence, per-evaluation metrics and
reconstructions) in the nested-sampling HTML report:

```bash
./ri report
# open reports/nested-sampling-report/index.html

./ri report --last 1
./ri report --run results/nested-sampling/r2d2-vlaa-merged-20260818T125604Z
./ri report --upgrade
./ri report --force
```

Each run gets its own page, `reports/nested-sampling-report/<run>.html`,
plus an `index.html` that lists every run on disk and links into them; each
run page links back to the index. Rendering a run means reading its FITS
output, so **run pages that are already up to date are skipped** - a re-run
only builds pages for new runs.

Each page built prints its own `wrote <path>` line, immediately followed by
elapsed time and an ETA - reading FITS output and drawing plots is most of
the report's wall clock (see below), so a rebuild with several outdated or
missing runs has something to show for the wait.

The index has a toolbar above the run cards: filter by algorithm (R2D2 /
WSClean) or by merged/unmerged, and sort newest/oldest or by eval count. It is
plain client-side JavaScript over the cards already on the page - no rebuild
or server needed, and it works the same off a `file://` open as it does
through `./ri serve`.

The plots themselves are PNG files under
`reports/nested-sampling-report/images/`, named after a hash of what they
were drawn from, and the pages link to them rather than inlining them. That
is where almost all of the report's time goes, so rebuilding a page (below)
redraws nothing that its inputs still match - only deleting the report
directory forces a full redraw. The hash covers a plot's inputs, not how it
was drawn, so a change to the drawing code has to bump `IMAGE_RENDER_VERSION`
in `scripts/lib/generate_report.py` to retire the PNGs already on disk.

Evaluation rasters are colour-mapped straight into a PNG at the FITS data's
own resolution and scaled up by the browser, rather than drawn through a
matplotlib figure - roughly 16x cheaper per image and 5x smaller on disk, and
indistinguishable at the size the pages display them. What is left of a full
redraw's cost is the anesthetic corner plot. A rebuild that redraws nothing does not even
import astropy or matplotlib - they are loaded on the first missing PNG - which
is most of what is left of a page-only rebuild. numpy goes the same way:
`common.py` binds `np` to a shim that imports numpy on first attribute
access, because the report reaches into that module only for its formatting
helpers and a page-only rebuild never touches an array (0.19s -> 0.16s on a five-run page-only rebuild, and
`self_check_lazy_numpy()` there fails if a module-level `np.` use creeps back
in). The two halves load separately:
the corner plot needs only matplotlib, the eval rasters astropy and PIL on top,
so a cold build keeps astropy out of the parent's serial prologue and off the
corner plot's critical path (the longest task in the build). The index is always rebuilt, so
it picks up new runs immediately.

Run pages are built in parallel, and each run splits into two concurrent
tasks - the anesthetic corner plot and the rest of the page - so the pool has
twice as many pages to overlap. The two kinds of task go into two pools, forked
either side of the astropy import: the corner plots - the critical path - start
first, and the eval-raster workers forked afterwards inherit astropy rather than
each importing it again while the plots want the CPU. anesthetic is imported in
the parent for the same reason, just before the first fork, so every corner-plot
worker inherits it instead of repeating the same 0.34s. Together that is ~25%
less CPU on a five-run cold build for the same output, and once there are more
runs to draw than cores it is wall-clock too (20 runs: 5.25s -> 4.75s).
The container is given a single BLAS thread
(the work is matplotlib rasterisation, not linear algebra, and multi-threaded
BLAS only oversubscribes the CPU). Override with `R2D2_OMP_THREADS=`. It also
runs with `--network none`: the report only reads the repo and writes
`reports/`, and setting up the container's network is ~0.3s of every
invocation - most of the cost of a build that draws nothing. The container is
removed from an `EXIT` trap rather than by `docker run --rm`, which blocks the
CLI for another ~0.12s tearing the rootfs down after the report has already
been written to the bind mount: five-run cold build 1.75s -> 1.63s on the host
and a rebuild that draws nothing 0.55s -> 0.48s, with byte-identical output.
The trap also fires on the failure paths, so a container is removed whether the
report succeeded or not.

The corner plot itself is the build's longest task, and roughly a fifth of it
was pandas repeating work. anesthetic draws the grid as 15 separate pandas plot
calls, and pandas re-runs its shared-axis tick housekeeping over every axis in
the figure after each one. `dedupe_pandas_tick_housekeeping()` in
`scripts/lib/generate_report.py` makes that run once per axis. It is not a plain
skip: reading an axis' tick labels un-stales matplotlib's shared view limits,
and that side effect is what keeps each diagonal panel's CDF twin on its
parent's x range, so the repeats still touch `viewLim` - drop that and the plot
visibly changes. Corner plot 1.01s -> 0.81s, five-run cold build 2.01s -> 1.80s
in-container and 15% less CPU, with byte-identical PNGs. The patch is
best-effort: if pandas moves the private helper it is a no-op and the plot is
just slower again.

With the labels themselves de-duplicated, what was left of a repeat call was the
scan that decides which axes to hand to the helper: for every axis, and for both
of its axes, walk the whole shared-axis group comparing positions - 420 of those
per corner plot. The same function records the axes the first scan selects and
replays only their `viewLim` touch afterwards, keyed on the figure's axes, their
visibility and the grid shape, so anything that changes the answer misses. That
takes the scan to 28 calls; corner plot 0.552s -> 0.538s, five-run cold build
1.449s -> 1.435s in-container and 1.5% less CPU, with byte-identical PNGs.
Note that pandas' plotting core binds `handle_shared_axes` by `from ... import`,
so the patch has to rebind the name in every module that holds it, not just in
`pandas.plotting._matplotlib.tools`.

Matplotlib repeats work of its own on the same figure. Every layout pass -
`tight_layout()`, the tight-bbox draw, the render draw, and each spine and axis
label placed in between - recomputes every axis' tick positions and label text
from scratch, ~930 times for one corner plot and about a third of its cost.
`memoize_matplotlib_tick_updates()` caches `Axis._update_ticks` per axis against
the axis' view interval, data interval and locator/formatter pair, which is
everything the result depends on; building that key reads `get_view_interval()`,
so a cache hit still performs the same `viewLim` un-staling the de-duplication
above relies on. Corner plot 0.86s -> 0.69s, five-run cold build 1.65s -> 1.43s
in-container and 18% less CPU, with byte-identical PNGs. Best-effort in the same
way: a matplotlib that renames the private method just draws slower.

Laying out a `Text` - splitting it into lines, measuring each against the font,
rotating the box - is repeated the same way. Those three passes ask for it 441
times per corner plot, for 70 distinct texts: the tick labels and axis titles
are re-measured in full on every pass, and the diagonal panels share their
labels with the panels below them. The result is position-free, so
`memoize_matplotlib_text_layout()` caches `Text._get_layout` in one process-wide
dict keyed on the string, the renderer class and figure dpi, the font
properties, the usetex/parse-math flags, the line spacing, the rotation and
rotation mode, and the three alignments - every input the layout reads. Texts
with wrapping enabled bypass the cache entirely, because their line breaks also
depend on the figure width. Corner plot 0.523s -> 0.496s (5% of every plot, so
5% of the CPU once there are more runs than cores), five-run cold build 1.443s
-> 1.423s in-container and 1.6% less CPU, with byte-identical PNGs. Best-effort
like the rest.

anesthetic repeats work of its own. Its labelled frames resolve every `df[key]`
by attempting the lookup against each of four label-stripped views of the frame
and keeping the best answer, and each attempt rebuilds that axis' paramname ->
label mapping from the index. One corner plot does that ~420 times.
`memoize_anesthetic_labels_map()` caches `_LabelledObject.get_labels_map` on the
pandas `Index` it is read off. An `Index` is immutable and anything that adds or
drops a column swaps in a new one, so a mutated frame misses the cache rather
than seeing a stale mapping; the `fill=False` variant, which `set_label()`
mutates in place, is deliberately never cached. Corner plot 0.66s -> 0.62s,
five-run cold build 1.43s -> 1.37s in-container and 5% less CPU, with
byte-identical PNGs. Best-effort like the two above.

Each of those four attempts also deep-copies the whole frame first, so one
corner plot makes ~380 copies of the same handful of frames.
`memoize_anesthetic_drop_labels()` caches `_LabelledObject.drop_labels` against
the frame's identity, both of its pandas `Index` objects, and which
`(axis, level)` pairs actually get dropped - so the specs that drop the same
levels share one copy and the rest stay apart. Pinning the indexes is what makes
it safe: an `Index` is immutable, so adding, dropping or relabelling a column
swaps in a new one and misses the cache instead of handing back a stale copy of
the frame. Rewriting an existing column's values in place would still slip past
that, but nothing on the plotting path does - the frame is read-only from load
to `savefig`. Corner plot 0.57s -> 0.53s, five-run cold build 1.55s -> 1.51s
in-container and 4% less CPU (7% at twenty runs), with byte-identical PNGs.
Best-effort like the rest.

Caching both halves still leaves anesthetic *running* all four attempts and
throwing three away, which is the expensive part: `ac` evaluates every
candidate, sorts them by dimensionality (fewest first, then most index levels,
ties going to the earliest candidate) and returns the first. For the case the
corner plot hits ~65 times per plot - a plain string column name on a frame
whose columns carry the labels level - the winner is decided in advance, so
`shortcut_anesthetic_labelled_column()` runs only that one. It is the candidate
that strips the labels off the columns: the two that keep them index a
two-or-more-level `MultiIndex` and so return a 2-D frame, which loses to any 1-D
result outright, and of the two that strip them this one leaves the *index*
alone, so it never has fewer index levels than the other and it wins the tie by
position. A `DataFrame` lookup cannot return a 0-D result, so nothing can
undercut a 1-D one. The guards are premises of that argument rather than safety
nets - a string key is what forces the label-keeping candidates to be 2-D - but
anything outside them, any candidate that raises, and any result that turns out
not to be 1-D all fall back to anesthetic's own search, so an uncovered shape is
slow rather than wrong. Corner plot 0.449s -> 0.413s (-8.2% CPU), five-run cold
build 1.168s -> 1.120s wall in-container and 4.2% less CPU, with byte-identical
PNGs. Best-effort like the rest. `_self_check_labelled_column_shortcut` builds
each frame twice before comparing the two resolutions, because pandas caches a
frame's column `Series` and resolving both ways on one frame would compare an
object with itself.

Saving the finished figure repeated a pass too. `savefig(bbox_inches="tight")`
cannot trust the artist positions it is handed - a layout engine may still be
pending - so it walks the whole figure once in a draw-disabled pass before
measuring the box. After `fig.tight_layout()` both reasons are already spent:
the layout has run, and all it leaves behind is a do-nothing placeholder engine.
`tight_bbox()` in `scripts/lib/generate_report.py` clears the engine, measures
the box itself and hands `savefig` a `Bbox` instead of the string, which drops
that pass. Save 0.108s -> 0.083s, corner plot 0.572s -> 0.545s, five-run cold
build 1.51s -> 1.47s in-container and 3% less CPU, with byte-identical PNGs.
Unlike the caches above this only ever asks matplotlib to do less, so there is
nothing to degrade gracefully; `_self_check_tight_bbox` asserts the two save
paths still produce the same bytes.

Two more repeats are structural rather than per-pass. Every read of an axis'
`viewLim` asks matplotlib whether anything sharing a limit with it still needs
autoscaling, and that question walks the whole shared-axis group through a
`WeakSet`, once per axis name - ~3200 times over 20 axes for one corner plot,
almost always to answer "nothing". Staleness is only ever created in one place,
`_request_autoscale_view`, so `skip_settled_matplotlib_viewlims()` counts calls
to it: an axis whose group was found settled at epoch *N* is still settled while
the counter reads *N*, and the scan is skipped outright. The epoch is re-read
after the wrapped call, so an autoscale that re-stales the group on its way out
is not recorded as settled. Nothing else in matplotlib or `mpl_toolkits` writes
`_stale_viewlims` to `True`, so the counter sees every transition.

And anesthetic gives each panel its limit-linking behaviour by defining a fresh
`Axes` subclass *inside* the per-axis helper and rebinding `__class__` to it, so
a 5x5 corner plot builds 15 one-instance classes. Each one pays matplotlib's
`Artist.__init_subclass__`, which regenerates `set()`'s signature and docstring
by parsing the docstring of all ~265 setters, and each leaves its panel with a
class no other panel shares, so no type-level cache is reused either. The class
bodies close over nothing but their base, so
`share_anesthetic_axes_subclasses()` lets the first panel build the class as
usual and rebinds the rest onto it, one class per (helper, base type).

Together: corner plot 0.499s -> 0.456s (-8.5%), five-run cold build 1.207s ->
1.171s in-container and 3.6% less CPU, with byte-identical PNGs. Best-effort
like the rest - if matplotlib or anesthetic renames either private hook, the
plot is just slower.

Two of matplotlib's own lookup helpers rebuild the same answer on every call,
which only shows up at this call volume. `_axis_map` - how an `Axes` finds its
`XAxis`/`YAxis`, read ~4500 times per corner plot - is a property that builds a
fresh dict from two f-strings and two `getattr`s each time;
`cache_matplotlib_axis_map()` keeps the dict on the instance and validates it by
identity against `self.xaxis`/`self.yaxis`, so the one place matplotlib swaps
those objects (`_init_axis`) misses the cache instead of getting a stale map.
Projections whose `_axis_names` is not `("x", "y")` fall through to the original
property. And `cbook.normalize_kwargs`, which every artist constructor and every
`.set()` call routes its kwargs through (~3400 times per corner plot), flattens
the artist class' alias map (`{'linewidth': ['lw'], ...}`) into an
alias -> canonical dict from scratch on each call; that flattening is a pure
function of the class, so `memoize_matplotlib_alias_maps()` memoises it per
class and keeps matplotlib's duplicate-alias `TypeError`. Callers that pass a
plain dict keep the original path. Like the shared-axis scan in iteration 16,
the patch has to rebind `normalize_kwargs` in every module that did
`from ... import normalize_kwargs`, because the hot call sites are not in the
defining module.

Together: corner plot 2.330s -> 2.262s CPU over the five runs (-2.6%), five-run
cold build 1.167s -> 1.157s wall and 1.0% less worker CPU in-container, with
byte-identical PNGs. The build's wall clock moves less than its CPU because
half of it is the parent's serial import prologue, which no plot-side change
touches.

`main()` disables the cyclic garbage collector. The report is a batch process -
it draws its figures, writes the pages and exits - so nothing it allocates has
to be reclaimed while it runs, and matplotlib figures are dense enough in
reference cycles that the collector wakes constantly and finds almost nothing
refcounting would not have freed anyway. Turning it off is 12% of a corner
plot's render (0.41s -> 0.37s in-process) and 9% of the whole build (five-run
cold build 1.060s -> 0.962s in-container), with byte-identical PNGs; the cycles
it would have collected are held to exit instead, which measured as +3MB on a
worker's 128MB peak. It is disabled in `main()` rather than at import, so
importing the module - the self-checks, host-side use - leaves the caller's gc
alone; the pool workers inherit the setting across fork. This is the one lever
that helps the serial import prologue as well as the plots.

`load_plot_libs()` skips matplotlib's 3d projection. `matplotlib.projections`
imports `mpl_toolkits.mplot3d` solely to register the `'3d'` projection, inside
a `try`/`except` that already degrades to "3d unavailable" when the import
fails. Nothing here draws in 3d - a corner plot is a grid of 2d axes - so a
`None` is parked in `sys.modules` for the duration of the `import
matplotlib.pyplot`, which makes that import raise `ImportError` and take the
branch matplotlib already handles. The entry is removed again immediately, so a
later `import mpl_toolkits.mplot3d` still works; only the registry entry stays
gone, which turns an accidental `projection='3d'` into a loud `ValueError`
rather than a silently wrong plot. Worth 17ms of the 258ms `import
matplotlib.pyplot` (-6.7%), median of 15 interleaved runs with the two
distributions not overlapping at all.

Alongside it, `multiprocessing` (3.4ms) is imported only once `main()` has runs
to fan out, and `tempfile`/`shutil` (2.0ms) only inside the self-checks that
use them, so a rebuild with nothing to do stops paying for either: its CPU
drops from 0.093s to 0.081s (-13%). Together with the 3d skip the parent's
serial import prologue - `import generate_report` plus matplotlib plus
anesthetic, which is over half a cold build's critical path - goes from 600.6ms
to 577.0ms (-3.9%), on every build that draws anything.

Two neighbouring import trims are measured dead and should not be retried:
`pyparsing` (29ms) cannot be dropped with the `_fontconfig_pattern` it first
appears under, because `matplotlib._mathtext` imports it anyway for the math
labels; and `importlib.metadata` (23ms) cannot be stubbed for anesthetic's
`__version__`, because pandas' plotting-backend loader needs its real
`entry_points()` on the same path.

The `r2d2` image bakes matplotlib's font list into `/opt/matplotlib`
(`MPLCONFIGDIR`). Containers run with `--rm`, so without it the first
`import matplotlib.pyplot` in every one of them rebuilds that list from the
installed fonts - ~0.07s, on the report's serial prologue, on every cold
build. Rebuild with `scripts/build.sh r2d2` to pick it up; an image without
it still works, just that much slower.

Every page carries the version of the report generator that wrote it (the
hash of `scripts/lib/generate_report.py`, in a
`<meta name="report-version">` tag), so changing the card design, the CSS or
anything else in that file makes existing pages **outdated** rather than
silently stale. Outdated pages are still skipped by a plain run - it says how
many it saw - and the index flags them with an `outdated page` badge.
`UPGRADE=1` rebuilds exactly those, bringing every page up to the current
design.

`LAST=N` only considers the newest N runs (timestamp sort). `RUN=` targets
one named run (directory, repo-relative path, or directory name) and always
rebuilds its page. `UPGRADE=1` rebuilds the pages an older report version
wrote. `FORCE=1` rebuilds every page in scope, up to date or not. `LAST=` and
`RUN=` cannot be combined. Make cannot take `--last`; use `LAST=1`.

The report globs `results/nested-sampling/*/summary.json` directly
(no manifest join), so a merged run directory (see **Merge runs** below) shows
up as its own card automatically. Evidence prefers a `log_z` /
`log_z_err` pair already in the summary (written for merged runs); otherwise
it parses PolyChord `chains/*.stats` for log(Z). It shows
each run's total wall-clock duration (from `total_wall_seconds`, when present)
top-right in the card header. Per-run images - the shared synthesized
ground-truth image and a per-evaluation card gallery (reconstruction,
objective, and searched parameters) - sit in an Images tab, and the
best-effort `anesthetic` KDE contour corner plot sits in a Likelihood tab, both inside
one collapsed-by-default details block, separate from the collapsed raw
metrics table. Corner plots are weighted by the raw log-likelihood (the
failure score), not by nested-sampling posterior mass. Runs are ordered newest-first by the UTC timestamp in the
run directory name.

### Read the report from another machine

Searches normally run on a headless remote host, where there is no browser to
open `index.html` with. `./ri serve` puts the report behind an HTTP server:

```bash
./ri serve                  # loopback:8000
./ri serve --port 9000      # [REPORT_PORT]
./ri serve --bind 0.0.0.0   # [REPORT_BIND]
```

It binds to `127.0.0.1` and prints the tunnel command to run on your own
machine:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<host>
# then open http://localhost:8000/
```

Nothing goes out to the network: the loopback bind means the only route in
from another machine is an SSH session the host already accepts, and the tunnel
runs at your end. It is not an access control, though - the report is
unauthenticated, so anyone with an account on that host, and any container
sharing its network namespace, can read it over loopback without SSH. On a
shared login node that is worth knowing before starting one. `--bind 0.0.0.0`
drops even that and serves the report to anything that can reach the host.

The `ssh -L` line forwards to whatever `--bind` actually bound, and guesses the
host address from `hostname -I` - right on a cloud box and wrong behind NAT, so
`REPORT_SSH_HOST` overrides it.

The server is `python3 -m http.server` from the host's standard library -
no Docker, no dependencies - and runs in the foreground until Ctrl-C. It is
also why copying a single page off the host does not work: the pages link to
sibling PNGs under `images/`, which is most of the report's bytes, so the whole
directory has to travel together (`rsync -a`) or be served in place.

### Replay a run in anesthetic's GUI

For an interactive nested-sampling replay (live points vs \(\ln X\), \(\beta\)
tempering) with human-readable parameter labels, run on the **host** (needs a
display; not inside Docker/Colima):

```bash
./ri plot gui
./ri plot gui results/nested-sampling/wsclean-vlaa-<UTC timestamp>
uv run scripts/anesthetic-gui.py results/nested-sampling/wsclean-vlaa-<UTC timestamp>
```

With no `RUN=`, the latest *completed* run under `results/nested-sampling/*/`
is used - either a plain run (`summary.json` and `chains/`) or a merged
run (`summary.json` with `merged_from`, no local `chains/`). For a plain
run the script writes/refreshes `chains/<root>.paramnames` from that run's
`summary.json` / `parameter-space.json`; either way it passes only the
searched Fourier parameter names into `samples.gui(params=...)` (not `logL` /
`logL_birth` / `nlive`). Close the GUI window to return to the shell.
Requires the host `uv` project dependency `anesthetic`
(`uv add anesthetic` if missing).

The run also writes a standard environment manifest through
`scripts/record-environment.sh`.

## Merge runs

Independent PolyChord runs of the **same likelihood and prior** can be fused
after the fact into one run directory, without re-running PolyChord. This is
post-processing only: it concatenates nested-sampling dead points with
`anesthetic.samples.merge_nested_samples` and recomputes live-point weights.
Evaluation directories, FITS images, and PolyChord chain files are never
copied; the merged summary just points back at the absolute evaluation paths
and source run directories already on disk.

Sampler effort may differ between sources; the search itself must not:

| May differ | Must match |
|---|---|
| `NS_NLIVE` / `polychord.nlive` | `algorithm` |
| `NS_NUM_REPEATS` / `polychord.num_repeats` | `vla_config` |
| `NS_MAX_NDEAD` / `polychord.max_ndead` | `metric` |
| `seed`, `mpi_procs` | `parameter_space` (name/min/max/kind) |
| | `r2d2_fixed_hyperparameters` **or** `wsclean_fixed_hyperparameters` |

WSClean and R2D2 runs never merge with each other, nor do runs with a
different `--metric` / `NS_METRIC` or a different prior box (the prior box is
`[[parameter_space]]` in `defaults.toml`, copied into
every `summary.json` as `parameter_space`).

With no directories, every completed source run under
`results/nested-sampling/` is grouped by the must-match fields above
and one merged directory is written per group of 2+. Incomplete dirs,
previous merges (`merged_from`), and singleton groups are skipped.
Zero groups of 2+ exits non-zero. `--out` is only valid with an explicit
directory list.

```bash
uv run scripts/merge-nested-sampling-runs.py
./ri merge

uv run scripts/merge-nested-sampling-runs.py \
  results/nested-sampling/r2d2-vlaa-AAA \
  results/nested-sampling/r2d2-vlaa-BBB

./ri merge results/nested-sampling/r2d2-vlaa-AAA results/nested-sampling/r2d2-vlaa-BBB
```

Writes `results/nested-sampling/<algorithm>-vlaa-merged-<UTC>/summary.json`
(pass `--out DIR` on the explicit form to pick a different output directory).
The explicit form refuses with a non-zero exit on fewer than two runs, a run
missing `summary.json` or `chains/`, or any must-match field above differing.
`polychord.nlive` in the merged summary is the sum of source nlives;
`num_repeats` / `max_ndead` / `seed` stay a single value when all sources
agree, else become a list. Pooled `evaluations` keep source argument order,
are renumbered `eval_id` `1..N` (originals kept as `source_eval_id` /
`source_run`), and keep their original absolute `paths`.

`./ri report` and `./ri plot gui <merged-dir>` both
treat the merged directory as a completed run - see the sections above.

## Deferred

What to search next, ranked, with the plumbing each one needs:
[`docs/parameter-space-proposal.md`](parameter-space-proposal.md).

Deferred deliberately:

- VLA.B and VLA.D.
- Full parameter-space exploration.
- VLA.C production exploration, although VLA.A and VLA.C are the prioritized
  follow-up configurations.
- A general multi-algorithm or multi-VLA orchestrator.

To add the next run family, reuse the simulator and metrics modules, keep VLA
configuration as the outer loop, and add only the missing algorithm-specific
runner.
