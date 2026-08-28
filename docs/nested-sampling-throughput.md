# Nested sampling: throughput

[nested-sampling-profiling.md](nested-sampling-profiling.md) is about what one
likelihood evaluation costs. This one is about the other half of the wall
clock: how much of the time the ranks are running an evaluation at all.

The two are separate problems with separate fixes. Making an evaluation 10%
cheaper and making the ranks 10% busier both buy 10% of the run, but nothing
in the per-stage timings tells you which one you are short of - a run whose
every stage is at its floor can still spend half its cores idle.

## The one number that matters

Every run's `summary.json` carries a `profiling` block. Its
`accounted_worker_seconds` is the time the ranks spent inside a likelihood
evaluation; `total_wall_seconds x (mpi_procs - 1)` is the worker-time budget
the run had. The ratio is worker utilisation, and `./ri profile <run>` prints its
complement as "unaccounted (PolyChord sampling + idle)":

```bash
python3 -c 'import json,sys; j=json.load(open(sys.argv[1]+"/summary.json")); p=j["profiling"]
n=p["mpi_procs"]; w=n-1 if n>1 else 1
print("%.0f%% utilised" % (100*p["accounted_worker_seconds"]/(p["total_wall_seconds"]*w)))' \
  results/nested-sampling/<run>
```

The denominator is the *worker* count, one less than the rank count: rank 0 is
PolyChord's administrator and never evaluates a likelihood. See
[rank 0 is not a worker](#rank-0-is-not-a-worker) below, which is also why the
figures on this page are not the ones it was first written with.

PolyChord's own sampling is not in that remainder in any meaningful quantity -
it is a Cholesky decomposition and a live-point insertion per iteration
against seconds to tens of seconds of imaging. What the remainder is, almost
entirely, is ranks with nothing to do.

## The finding: the ranks were idle a third to a half of the run

Measured across the runs on disk before any of this changed:

| Run | ranks | nlive | num_repeats | wall | utilisation |
|---|---:|---:|---:|---:|---:|
| `r2d2-vlaa-20260827T205418Z` | 16 | 50 | 10 | 14.0h | 58% |
| `wsclean-vlaa-20260827T190109Z` | 20 | 50 | 10 | 501s | 47% |
| `wsclean-vlaa-20260827T201926Z` | 20 | 50 | 10 | 290s | 71% |

Two things to read off that. Half the machine was idle. And the same settings
gave 47% and 71% on two consecutive runs - the throughput was not just low, it
was not repeatable, which is what makes a run's remaining time impossible to
predict.

Reconstructing the 14-hour R2D2 run's timeline shows the shape of it. Each
evaluation directory's `metrics.json` mtime is when that evaluation finished,
and its `timing` block is how long it took, so the two give an interval per
evaluation; counting overlapping intervals gives how many ranks were busy at
any moment. Over that run:

```
concurrency:  1 rank busy 17% of the time,  2 busy 11%,  15 busy 37%
mean 8.7 of the 15 workers
```

Bimodal, not a uniform 58%: the ranks were either all working or nearly all
stopped. Plotted against time it is a sawtooth - every rank starts together,
they finish staggered, the last one finishes minutes after the first, and only
then does the next round start.

That is a barrier, and PolyChord names it in its own run header:

```
Synchronous parallelisation
```

## Cause: pypolychord defaults to synchronous MPI

`PolyChordSettings.synchronous` defaults to `True`
([`pypolychord/settings.py`](https://github.com/PolyChord/PolyChordLite/blob/master/pypolychord/settings.py)).
In that mode the administrator rank throws a slice-sampling chain to every
worker, then blocks in a loop of `catch_babies` until *all* of them have come
back, before it hands out any more work
(`src/polychord/nested_sampling.F90`). A round therefore costs the slowest
chain in it, and every other rank idles out the difference. The asynchronous
mode reseeds each worker the instant it reports back, so no rank ever waits
for another.

Upstream's own docstring is explicit that synchronous "parallelisation is less
effective than asynchronous by a factor ~O(1) for large parallelisation", and
that asynchronous is the right choice when likelihood speed is roughly
constant across the parameter space.

Two properties of this search make the barrier unusually expensive:

- **Chain length is stochastic.** A chain is `num_repeats` slice-sampling
  steps, and each step needs however many likelihood calls its bracket
  expansion and contraction take. The 14-hour run averaged ~38 calls per
  chain, but the spread across 15 simultaneous chains is what sets the round.
- **A call is expensive and variable.** R2D2 evaluations in that run ranged
  3.0s to 36.9s (median 24.8s). The barrier pays the maximum of 15 sums of
  ~38 such draws, and only the mean is useful work.

Both get *worse* with the bigger runs this repo is aiming at: the maximum of
N draws grows with N, so more ranks means a longer straggler tail, and a
larger `num_repeats` means a longer chain to be a straggler in.

## Fix: run asynchronously

`NS_SYNCHRONOUS` (default `0`, in `defaults.toml`; `./ri search
--synchronous` to put it back) sets `settings.synchronous` in both drivers.

A/B on WSClean, same seed within each pair, 15 ranks, `--nlive 50
--num-repeats 10`, run to completion:

| seed | mode | wall | rank utilisation | dead points | s / dead point | log(Z) |
|---:|---|---:|---:|---:|---:|---|
| 4242 | synchronous | 293.1s | 69% | 435 | 0.674 | 0.024636 ± 0.000024 |
| 4242 | **asynchronous** | **238.8s** | **98%** | 497 | **0.480** | 0.024620 ± 0.000023 |
| 7 | synchronous | 316.6s | 68% | 438 | 0.723 | 0.024625 ± 0.000024 |
| 7 | **asynchronous** | **243.7s** | **97%** | 502 | **0.486** | 0.024625 ± 0.000024 |
| 99 | synchronous | 293.2s | 72% | 440 | 0.666 | 0.024697 ± 0.000030 |
| 99 | **asynchronous** | **245.7s** | **98%** | 502 | **0.489** | 0.024709 ± 0.000031 |

Three pairs, no exceptions: **27-33% less wall clock per dead point**, rank
utilisation 68-72% -> 97-98%. The asynchronous runs went *further* as well as
faster - each ran to the same termination criterion and got ~60 more dead
points out of it, so the raw wall-clock column understates the gain.

Two things beyond the headline are worth noting.

The likelihood calls per dead point barely moved (19.0 synchronous, 19.1
asynchronous), so the extra points are not bought by sampling more cheaply -
asynchronous mode is not doing less work per point, it is doing the same work
on more cores.

And the utilisation is *steady*: 97, 98, 98 against 68, 69, 72, and against
47% and 71% on two historical 20-rank runs at the same settings. The
straggler tail is where the run-to-run variance in throughput was coming from,
so removing it makes a run's remaining time predictable as well as shorter.
The evidences agree to within 0.5 sigma on every pair.


## What asynchronous mode gives up

It is not free, and the reason it is not upstream's default is worth stating
rather than burying:

- **The order points are incorporated is completion order, not issue order.**
  A chain seeded from an older likelihood contour can come back after the
  contour has moved on. PolyChord tags each chain with the administrator's
  epoch and discards ones whose epoch has since changed, so nothing invalid
  enters the live set, but a slow chain is more likely to be thrown away than
  a fast one.
- **So a systematic correlation between how long an evaluation takes and what
  it scores would bias the result**, by under-representing the slow corner of
  the parameter space. Synchronous mode has no such coupling.

For this repo that is an acceptable trade and it is worth being clear why.
The point of these runs is to find where the imagers fail, and the run is
scored by where the sampler concentrates, not by an evidence integral quoted
to its error bar. The two modes' evidences agreed within their own
uncertainties on every pair measured (above). A run whose result has to be
defensible as a *measurement of Z* should be started with `--synchronous`, and
`summary.json`'s `polychord.synchronous` records which mode any run used.

## What was ruled out

- **PolyChord's own arithmetic.** Not the remainder: a run at
  `NS_MPI_PROCS=1` accounts for it exactly, and `polychord_overhead_seconds`
  there is 2.5% of the wall clock.
- **PolyChord's built-in `wait_time/slice_time` report.** It prints a per-worker
  ratio at the end of a run (`Worker N: efficient MPI parallelisation...`) and
  it reports near-zero waiting even on runs that were plainly stalling. Do not
  use it; measure the ranks from the evaluation directories instead.
- **Host oversubscription.** The 14-hour run's idle time is bimodal and
  synchronised across ranks. Contention would have shown up as every
  evaluation being uniformly slower, not as all ranks stopping together.


<a id="rank-0-is-not-a-worker"></a>

## Rank 0 is not a worker, and the utilisation numbers above were wrong

PolyChord's rank 0 is the *administrator*: it hands slice-sampling chains out,
collects them, keeps the live set and writes the files. It never calls the
likelihood. `nested_sampling.F90` says so in its own array shapes -
`worker_cluster(nprocs-1)`, `worker_epochs(nprocs-1)` - and `generate.F90`
sets `active_workers = nprocs-1` when it makes the initial live points.

So a job of N ranks has N-1 workers, and its worker-time budget is
`wall x (N-1)`, not `wall x N`. Everything above this section, and
`./ri profile`, and the report's profiling table, used the rank count. That
understated utilisation by (N-1)/N - 7% at 15 ranks - and the shortfall reads
as idle time, which is the one thing on this page anybody is trying to remove.
`worker_procs()` in `common.py` is the fix, and every figure above has been
restated through it: the asynchronous pairs move from 91-92% to 97-98%.

The conclusion of the section above does not change - the barrier was real and
removing it was worth 27-33%. What changes is what is left: **the workers are
97-98% busy, not 91-92%.** There is no meaningful idle time left to recover at
these settings, and any further speed has to come from a cheaper evaluation,
fewer evaluations per dead point, or more workers.

To read utilisation off a run:

```bash
python3 -c 'import json,sys; j=json.load(open(sys.argv[1]+"/summary.json")); p=j["profiling"]
n=p["mpi_procs"]; w=n-1 if n>1 else 1
print("%.0f%% utilised" % (100*p["accounted_worker_seconds"]/(p["total_wall_seconds"]*w)))' \
  results/nested-sampling/<run>
```

## The administrator burns a hardware thread, and nothing tried gets it back

Rank 0 waits for chains inside a blocking `MPI_Recv` (`catch_babies` in
`mpi_utils.F90`), and Open MPI's progress engine spins on it in userspace
rather than sleeping. Measured on a 15-rank WSClean search, sampling
`ps -o pcpu` every 10s for the length of the run:

```
rank:    0     1    2    3    4    5   ...  14
%CPU:  99.9  1.2  1.1  1.3  1.4  1.6   ...  1.2
```

`/proc/<rank 0>/syscall` reads `running` on every sample - it is not in a
syscall, it is spinning. The working ranks sit at 1-2% because their imaging
happens in the sidecars; the number that matters is the 99.9%. One hardware
thread of a 20-thread host, for the length of every run.

Three ways to get it back were measured. **None of them changed the
throughput**, so none of them shipped:

| change | evaluations/s (3 interleaved pairs, WSClean, `--nlive 50 --num-repeats 10`) |
|---|---|
| baseline, 20 ranks | 44.4, 42.9, 41.3 |
| `-np 21 --oversubscribe` (20 workers instead of 19) | 43.6, 41.7, 42.0 |
| baseline, 20 ranks | 45.5, 44.8, 43.4 |
| administrator at `nice 19` | 44.5, 44.5, 43.2 |

- **`OMPI_MCA_mpi_yield_when_idle=1`** does nothing. It makes the progress loop
  call `sched_yield()`, which is close to a no-op for a CFS task; rank 0 still
  measures 99% CPU with it set, confirmed from `/proc/<pid>/environ`.
- **One more rank**, so the *worker* count matches the thread count rather than
  being one short, is 2% *slower* if anything. The box is already saturated at
  19 workers.
- **`nice 19` on the administrator**, which does work where `sched_yield` does
  not (its `ni` really is 19 and it really is descheduled under load), is also
  within noise. The thread it gives up is not one the workers can use.

The last two are the same result seen twice, and the section below says why:
this host is already past the point where another worker is worth much, so
freeing a thread for one buys nothing. That may not hold on a bigger box or a
cheaper likelihood, and the thing to carry forward is the measurement rather
than the verdict - `ps -o pcpu,ni` across the ranks plus `evals/s` from
`summary.json` is enough to re-run any of it in ten minutes.

## What is left: the evaluation gets more expensive the more workers there are

With the idle time gone, scaling a run is no longer about keeping the workers
busy - they are 96-99% busy at every rank count. It is about what a worker
gets done while it is busy, and that falls off sharply. One WSClean search per
row, same seed (4242), same settings (`--nlive 50 --num-repeats 10`), only
`--mpi-procs` changed:

| workers | evals/s | evals/s per worker | simulate | `wsclean` | per evaluation | utilisation |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.7 | 5.43 | 40ms | 136ms | 183ms | 99% |
| 8 | 31.3 | 3.92 | 61ms | 180ms | 248ms | 97% |
| 12 | 37.7 | 3.14 | 70ms | 228ms | 306ms | 96% |
| 19 | 45.5 | 2.39 | 105ms | 290ms | 403ms | 97% |
| 20 | 43.6 | 2.18 | 122ms | 309ms | 440ms | 96% |

Five times the workers buys 2.1x the throughput. The workers are not waiting -
**an evaluation costs 2.4x more at 20 workers than at 4** (183ms -> 440ms),
and both halves of it inflate: simulate 3.1x, `wsclean` 2.3x. The container
round-trip overhead does not (5.9ms -> 7.0ms), so this is not the daemon.

Two things make that shape unsurprising on this host, and they pull apart when
extrapolating:

- **The cores are not equal.** This is an i5-13500: 6 P-cores with two threads
  each, plus 8 single-threaded E-cores, for the 20 that `nproc` reports. The
  first few workers get a P-core to themselves; the twentieth gets whatever is
  left. A homogeneous server should fall off less steeply.
- **An evaluation is not one core.** Each one runs a MeqTrees predict and a
  `wsclean` in the sidecars, and they share memory bandwidth and last-level
  cache with every other worker's.

What to take from it when sizing a bigger run:

- **More ranks still helps** - 45.5 evals/s against 21.7 - so `NS_MPI_PROCS =
  min(NS_NLIVE, host threads)` stays the default. One rank *more* than that,
  so the worker count rather than the rank count matches the threads, was
  measured and is not worth it (the section above).
- **But the marginal worker is worth about 40% of the first one**, which
  changes the arithmetic on the memory wall. An R2D2 search capped at ~14
  workers by RAM (`scripts/lib/rank-budget.sh`) is not losing what the rank
  count suggests, and spending the same memory on a larger `--nlive` is a
  better trade than it looks.
- **Anything that makes one evaluation cheaper is now worth more than anything
  that adds a worker**, and `wsclean` itself is 69% of the worker-time budget
  at the default rank count (`./ri profile <run>`).

## Where the evaluation's time actually goes

The section above ends on "anything that makes one evaluation cheaper is now
worth more than anything that adds a worker". This one takes that apart.

The measurements below replay a real run's `wsclean` commands instead of
starting searches: `results/nested-sampling/<run>/summary.json` records the
exact argv of every evaluation's `wsclean`, so 200 of them can be fed to *N*
`sh` processes inside one sidecar container. That reproduces the run's
concurrency and its per-rank `-j 1` without PolyChord, an MPI layer or a
stochastic evaluation count in the way, and it is repeatable to about 1%,
which the end-to-end evals/s is not. Keep a run's `evaluations/` directory
alive to do it - `./ri search` writes one `sim.ms` per evaluation and they are
the input.

`wsclean` prints its own accounting. Over 400 evaluations of one run:

| | per evaluation |
|---|---:|
| inversion | 54.4ms |
| prediction | 35.8ms |
| deconvolution | 9.3ms |
| **imaging total** | **99.6ms (77%)** |
| everything else | 29.7ms (23%) |
| `wsclean` binary, wall | 129.3ms |

The 23% that is not imaging is process start, opening and reordering the
Measurement Set, and writing five FITS files. `wsclean --version` alone - fork,
exec, 73 shared libraries, C++ static initialisers - is 15ms of it, and the
dynamic loader is only 1.4ms of that 15ms, so it is constructor work inside
casacore and friends and there is no flag for it.

The imaging 77% is a major-cycle loop: `-mgain 0.8` with `-niter 100` runs a
mean of **6.76 major iterations**, each one a full prediction and a full
inversion. That is the shape of the cost, and it is set by the deconvolution
settings, not by anything the harness controls.

## The host saturates at ~56 evaluations/s, and it is memory, not scheduling

Replaying the same 200 evaluations at increasing concurrency, with host CPU
sampled from `/proc/stat` across each arm:

| workers | evals/s | per evaluation | host CPU |
|---:|---:|---:|---:|
| 1 | 7.1 | 140ms | 5% |
| 2 | 14.8 | 135ms | 10% |
| 4 | 27.7 | 145ms | 20% |
| 6 | 37.8 | 159ms | 30% |
| 8 | 39.2 | 204ms | 38% |
| 12 | 45.3 | 265ms | 54% |
| 16 | 53.0 | 302ms | 74% |
| 20 | 56.4 | 354ms | 92% |
| 24 | 55.2 | 435ms | 91% |

Scaling is linear to 4 workers and then bends; past 20 it goes backwards. Host
CPU tracks the worker count almost exactly the whole way (1 worker = 5% = one
of twenty threads), so nobody is blocking on anything - the workers are all
running, each just gets less done.

Pinning with `taskset` inside the container says what they are contending on.
This host is an i5-13500: CPUs 0-11 are the six P-cores' hyperthread pairs,
CPUs 12-19 are eight single-threaded E-cores.

| placement | workers | evals/s |
|---|---:|---:|
| 6 P-cores, one thread each (0,2,4,6,8,10) | 6 | 39.7 |
| 8 E-cores (12-19) | 8 | 26.8 |
| 12 P-core threads (0-11) | 12 | 52.7 |
| 6 P-cores + 8 E-cores | 14 | 43.3 |
| all 20 threads | 20 | 54.8 |
| 20, unpinned | 20 | 56.6 |
| 12, unpinned | 12 | 45.3 |

Two things fall out of that table:

- **The parts do not add up.** 6 P-cores alone do 39.7 evals/s and 8 E-cores
  alone do 26.8, but the same 14 workers together do 43.3, not 66.5. Nothing
  is idle and nothing is scheduled badly; they are competing for last-level
  cache and memory bandwidth. That is the wall, and it is why every CPU-side
  win below shrinks by roughly two thirds between one worker and twenty.
- **The E-cores are nearly free of charge and nearly worthless.** Twelve
  workers on the P-core threads do 52.7 evals/s; adding all eight E-cores on
  top gets 54.8. The last eight ranks of a 20-rank run are buying ~4%. For
  WSClean that costs nothing worth counting, but an R2D2 rank is ~3.4GB
  (`scripts/lib/rank-budget.sh`), so on a memory-capped R2D2 search the same
  RAM is better spent on `--nlive`.

Pinning is not worth wiring in: at the rank count runs actually use, letting
the kernel place the workers (56.6) beat every fixed placement tried (54.8).
It only helps at rank counts nobody runs - 12 pinned to the P-core threads is
52.7 against 45.3 unpinned.

## Six ways of making the evaluation cheaper that do not work

All measured by replay, 200 evaluations, serial and at 20 concurrent workers,
each arm run at least twice interleaved with the baseline. None of them is in
the working tree.

| change | serial evals/s | 20-worker evals/s | verdict |
|---|---:|---:|---|
| baseline | 7.3 | 70.3 | - |
| evaluation directories on `/dev/shm` | 7.4 | 66.6 | noise |
| `-no-reorder` | 3.7 | 36.1 | 2x slower |
| `-gridder wstacking` | 7.4 | 67.3 | noise |
| `-nwlayers 1` | 7.3 | 57.6 | slower |
| `-no-small-inversion` | 7.2 | 54.9 | slower |
| `-abs-mem 0.05` | 7.3 | 56.1 | slower |

Two are worth spelling out.

**Disk is not the bottleneck.** An evaluation directory is ~1.9MB and a
20-worker run writes ~100MB/s of them, which sounds like something until you
copy them all to `/dev/shm` and measure no difference. The host's NVMe and the
page cache absorb it. This also rules out the reorder temporaries, which live
in the same directory (`-temp-dir`), and it means the `evaluations/` bloat
noted below is a disk-space problem only, never a speed one.

**`-no-reorder` is a trap.** WSClean's reordering pass looks like pure
overhead for a single 2808-row Measurement Set - it copies the data out of the
MS into temporary files before imaging - and removing it halves throughput,
because the ~7 major cycles then re-read the casacore table instead of a flat
file.

### `-wgridder-accuracy`, which works and is still the wrong trade

`-wgridder-accuracy 1e-2` against the 1e-4 default is 9.9 evals/s serial
(+35%) and 74 at 20 workers (+5%), the largest single-flag effect found. It is
rejected on the science, not the speed: 1e-2 of a 1 Jy peak is 10mJy of
gridding error, and the objective this search maximises *is* the image RMS,
which these runs measure between 0.1 and 25mJy. The knob would be moving the
thing being searched for.

Worth noticing while passing: the *default* 1e-4 is 100µJy against a 1 Jy
peak, and some evaluations report an off-source RMS below that. Those numbers
are the gridder's own error floor, not the imager's noise. That is a
parameter-space concern rather than a throughput one, but it is the reason
this flag is not a free 5%.

## What does work: build WSClean for the CPU it runs on

`docker/wsclean/Dockerfile` takes `WSCLEAN_PORTABLE`, which is WSClean's own
`-DPORTABLE` CMake option, and defaults it to `ON` - a binary that runs on any
x86-64, so no AVX2, no FMA. WSClean's gridder is the one place that costs the
most.

Replaying 200 evaluations, three interleaved repeats each:

| | portable | native | |
|---|---:|---:|---:|
| 1 worker | 7.29 evals/s | 8.74 evals/s | **+19.8%** |
| 20 workers | 70.3 evals/s | 74.8 evals/s | **+6.5%** |

That gap between the two rows is the memory wall from the section above doing
its work: a fifth off the CPU time is worth a fifth only when there is a core
free to spend it on.

End to end - `./ri search wsclean --nlive 25 --num-repeats 10`, default 20
ranks, three seeds, portable and native alternating:

| seed | evals/s portable | evals/s native | | `wsclean` binary |
|---:|---:|---:|---:|---:|
| 4242 | 43.2 | 47.6 | +10.3% | -9.5% |
| 7 | 44.3 | 46.2 | +4.4% | -5.8% |
| 99 | 41.2 | 48.4 | +17.6% | -15.6% |

Native wins all three, by a median of 10%. The spread is the run-to-run
throughput noise this repo has seen throughout - the replay's +6.5% at matched
concurrency is the number to plan with, and the `wsclean` binary column, which
is a mean over ~6000 evaluations rather than one wall clock, is the one to
check a rebuild against.

**It does not move the science.** Recomputing the objective from both builds'
FITS output over 200 evaluations, the relative difference is a median of
9.7e-8 and a maximum of 3.7e-7 - `-march=native`'s FMA contraction, four
orders of magnitude below the gridder's own 1e-4 accuracy setting and further
still below the PolyChord noise that makes two same-seed runs disagree on
their evaluation count by 10%.

**How to use it.** `./ri search wsclean --native`. The flag has to be on the
*search*, not only on `./ri build wsclean --native`: both write the same
`ri-reproducibility/wsclean:v3.7` tag, and a search builds its images first,
so a plain `./ri search` puts the portable binary back under the tag before it
runs. It stays opt-in because the binary will die with SIGILL on any other
CPU, and the tag is shared with every other worktree on the host.
