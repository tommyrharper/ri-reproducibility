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
which the end-to-end evals/s is not. The input is each evaluation's `sim.ms`,
which a search deletes as it scores the evaluation, so the run being replayed
has to have been started with `./ri search --keep-measurement-sets` (see
[the disk footprint section](#the-disk-footprint-is-what-caps-a-big-run-not-the-clock))
and to have kept its `evaluations/` directory.

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

## The host saturates at ~56 evaluations/s, and it is not scheduling

> The reading below - that the shortfall is last-level cache and memory
> bandwidth - was measured again with the clock the workers actually ran at,
> and it is mostly wrong. See "The wall is the clock, not the memory bus"
> further down. The throughput numbers here all reproduce; the *cause*
> attributed to them does not.

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
  is idle and nothing is scheduled badly. This was read as last-level cache
  and memory-bandwidth contention; measuring the clock says otherwise - a
  six-core arm runs its cores near 4.5GHz and a fourteen-core arm near 3.2GHz,
  which accounts for most of the gap on its own. See "The wall is the clock,
  not the memory bus".
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

> **Superseded.** The default is no longer the plain x86-64 baseline this
> section measured against: it is `x86-64-v3`, which is AVX2 and FMA on a
> named target rather than the build machine's. See
> [the AVX2 build stopped being opt-in](#the-avx2-build-stopped-being-opt-in).
> The `--native` figures below are still what `--native` is worth *over the
> baseline*; over today's default it is a few percent.

`docker/wsclean/Dockerfile` took `WSCLEAN_PORTABLE`, which is WSClean's own
`-DPORTABLE` CMake option, and defaulted it to `ON` - a binary that runs on any
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

## The other half: the simulate stage spends 13-22ms predicting a constant

`wsclean` is 69-70% of the worker-time budget, and the sections above are all
about it. The other 28% is the MeqTrees simulate. Replaying a run's recorded
`simulate` argv the same way - `summary.json` records those too, and the
meqtrees sidecar answers them over the same per-rank FIFOs a real run uses -
breaks a serial simulate down as:

| | per evaluation |
|---|---:|
| MeqTrees predict | 17.9ms |
| skeleton copy + `SPECTRAL_WINDOW` patch | 2.3ms |
| noise fill, `DATA`/`FLAG`/`WEIGHT` writes, baseline maximum | 7.4ms |
| makems config, correlation probe, moving the MS out of `/dev/shm` | 4.0ms |
| **simulate, wall** | **31.6ms** |

The predict is over half of it, and it does not scale with the data. Binning
the same 400 evaluations by the shape they asked for:

| timeslots | fewest channels in the bin | 8 channels |
|---:|---:|---:|
| 1 | 13.4ms (3 channels) | 13.8ms |
| 5 | 16.0ms (1 channel) | 16.4ms |
| 10 | 21.0ms (2 channels) | 21.8ms |

Eight times the visibilities costs nothing; ten times the timeslots costs 8ms.
So the predict is ~13ms of fixed meqserver round trip and MS open plus ~0.85ms
per timeslot of `VisDataMux` tile handling, and approximately none of it is the
RIME arithmetic.

That would be worth attacking on its own. It is worth more than that, because
of what the arithmetic actually produces. `point_source_forest.py` hands Meow
the phase centre `Direction` object itself when the source offset is zero, so
no K-Jones phase shift is applied and the brightness matrix reaches the sinks
unchanged. Read the `DATA` column back after a predict and every row, timeslot
and channel holds the same number:

```
{'chan': 1, 'mins': 0.3,  'f0': 5.4e10} unique rows: 1 [[1. +0.j 0.+0.j 0.+0.j 1. +0.j]]
{'chan': 8, 'mins': 20.0, 'f0': 5.4e7 } unique rows: 1 [[1. +0.j 0.+0.j 0.+0.j 1. +0.j]]
{'chan': 5, 'mins': 7.3,  'f0': 1.4e9 } unique rows: 1 [[2.5+0.j 0.+0.j 0.+0.j 2.5+0.j]]   # 2.5 Jy
```

`source_offset_fraction` is `enabled = false` in `defaults.toml`, so *every*
evaluation of a default run pays a meqserver round trip to be told the source
flux back.

**Fix.** `phase_centre_visibility()` in `simulate_point_source_ms.py` writes
that constant directly, and MeqTrees is asked only for a source that is
actually off the phase centre - which is the case the predict exists for, and
which `--enable-param source_offset_fraction` still runs unchanged.
`self_check_phase_centre_predict()` (in `./ri self-check simulate`) is the
guard: it runs the real predict at three shapes and three fluxes and asserts
the column matches the constant exactly.

Replaying 40 evaluations per worker:

| | before | after | |
|---|---:|---:|---:|
| 1 worker | 32.9ms | 13.2ms | **-60%** |
| 19 workers | 74.0ms | 28.7ms | **-61%** |

That is 45ms off a ~400ms evaluation at the default rank count.

End to end - `./ri search wsclean --nlive 25 --num-repeats 10`, default 20
ranks, three seeds, the two meqtrees images alternating:

| seed | evals/s before | evals/s after | | simulate | `wsclean` |
|---:|---:|---:|---:|---:|---:|
| 4242 | 44.10 | 51.06 | **+15.8%** | 108.7ms -> 39.6ms | +3.2% |
| 7 | 43.97 | 50.92 | **+15.8%** | 106.2ms -> 38.1ms | +3.8% |
| 99 | 43.30 | 52.06 | **+20.2%** | 106.1ms -> 38.4ms | +1.7% |

Tighter than any A/B on this page so far - three seeds inside five points of
each other, against the 10% run-to-run swing everything else here has had to
fight. The `wsclean` column is the memory wall taking a cut back: freeing 45ms
of one worker gives the other 18 more bandwidth to compete for, so the imaging
gets 2-4% slower and the run still comes out 16-20% ahead.

**It does not move the science.** Running the same argv and seed through the
old image and the new one gives Measurement Sets whose `DATA`, `UVW`,
`WEIGHT`, `SIGMA` and `FLAG` columns are equal element for element, and
identical `simulation.json`. Across the three A/B pairs above, 343 evaluations
landed on a parameter vector both arms happened to visit, and all 343 scored
bit-identical objectives.

Two things fall out of it beyond the time:

- A default run now does **no predicts at all**, so it cannot hit the MeqTrees
  deadlock that `docs/robustness.md` documents at roughly one evaluation in
  2,000-5,000. The watchdogs, the wedge counter and `restart_meqserver_session()`
  all stay - they are what the offset-source case needs - but the default path
  no longer reaches them.
- The `meqserver` per rank is still started and the forest still compiled
  during container start-up, which is free (it happens while the other two
  containers, the manifest and mpirun still have to happen) and is what keeps
  the offset case fast when it is enabled.

## Throughput falls through a run, and it is the sampler doing it

A run's evaluations get steadily more expensive as it goes. Across all 27
WSClean runs on disk, the last tenth of a run's evaluations cost a **median of
1.17x** the first tenth (range 1.11-1.34), and not one run bucks it.

It is not thermal throttling, disk fill or a leak. It is what nested sampling
is for. `channel_count` averages 4.6 in a run's first tenth and 7.4 in its
last, while `observation_minutes` barely moves (12.5 -> 11.2): the sampler
contracts onto the corner of the parameter space that maximises
`total_rms_jy`, that corner has the most channels, and an evaluation's cost is
roughly linear in its visibility count. A run gets slower because it is
succeeding.

Two consequences for planning a bigger run:

- **Extrapolating from a short prefix under-predicts.** A run stopped at 10%
  has not yet reached the expensive region it will spend most of its life in.
- **The cost of `--nlive` and `--num-repeats` is worse than linear in wall
  clock**, because both buy more of the *late* evaluations rather than a
  uniform sample of them. Sizing a run off `evals/s` measured over its first
  few minutes will be optimistic by 15-30%.

This is separate from the run-to-run swing (two same-seed runs differ by ~10%
in evaluation count under asynchronous mode) and from the straggler tail that
synchronous MPI used to add. It is a property of the search, not of the
harness, and there is nothing to fix in it - it is here so a future
extrapolation accounts for it.

## The wall is the clock, not the memory bus

The "parts do not add up" reading above - 6 P-cores at 39.7 evals/s plus 8
E-cores at 26.8 giving 43.3 rather than 66.5 - was taken as evidence of a
memory-bandwidth ceiling. It is not. Sampling `cpu MHz` from `/proc/cpuinfo`
across a replay arm, alongside the throughput, closes the gap without invoking
memory at all:

| workers | evals/s | per evaluation | busiest core | Mcycles per evaluation |
|---:|---:|---:|---:|---:|
| 1 | 6.87 | 145.6ms | 4714 MHz | 686 |
| 6 | 38.70 | 155.0ms | 4479 MHz | 694 |
| 14 | 51.04 | 274.3ms | 3180 MHz | 872 |
| 19 | 58.96 | 322.3ms | 2958 MHz | 953 |

The last column is the one that matters: multiply the wall time by the clock
the worker actually ran at and an evaluation costs **the same 690 Mcycles at 6
concurrent workers as it does alone**. Six `wsclean` processes, each with a
P-core to itself, contend for nothing measurable - which a memory-bandwidth
ceiling would not allow. What is left, 872 and 953 Mcycles, arrives only once
the extra workers are landing on E-cores (lower IPC at any clock) and on
hyperthread siblings, and both of those are the same core getting less done per
cycle, not a shared bus running out.

> **Correction.** The all-core clock below is not a property of the silicon:
> it is what a 65W package power limit buys at that worker count, and the
> part's own rated turbo power is 117W. Of the 1.59x, ~1.30x is the power
> limit and is recoverable with one sysfs write. See
> [docs/nested-sampling-power-limit.md](nested-sampling-power-limit.md).

So of the 2.21x an evaluation inflates between 1 and 19 workers, **1.59x is
the all-core clock** (4714 MHz down to 2958 MHz - stock behaviour for an
i5-13500, whose single-core turbo is 4.8GHz and whose all-core sustained clock
is around 3GHz) and 1.39x is core heterogeneity. `cpuinfo_max_freq` puts the
P-cores at 4.8GHz and the E-cores at 3.5GHz, so a lightly loaded run gets a
much better core than a full one does, in clock as well as in IPC.

Two things follow for the bigger runs:

- **A CPU-side win is not discounted by a bandwidth ceiling**, so the "shrinks
  by roughly two thirds" rule above is the wrong model. What a serial
  measurement over-states is narrower: the win is realised at the all-core
  clock, on cores of two different qualities, and (for a SIMD win such as the
  native build) on E-cores with narrower vector units.
- **A homogeneous machine will scale much better than this one.** The 6-worker
  point says the work itself parallelises perfectly; the fall-off is this
  particular CPU's hybrid topology and power budget.

## The first four seconds of any measurement lie by 20%

> **Correction.** The effect and the rule below both hold, but the cause is
> not the clock "settling": it is the package dropping from its 117W short-term
> power limit to its 65W long-term one once the 8-second averaging window
> fills. See [docs/nested-sampling-power-limit.md](nested-sampling-power-limit.md).

The all-core clock does not settle instantly. Sampling `/proc/cpuinfo` across
a 19-worker replay:

```
loaded t=2s   mean 3851 MHz
loaded t=4s   mean 3847 MHz
loaded t=6s   mean 2951 MHz
loaded t=8s   mean 2962 MHz
loaded t=10s  mean 2996 MHz
loaded t=12s  mean 2961 MHz
```

and the throughput follows it exactly: the first arm of any back-to-back
replay sweep reports 70-71 evals/s where every later arm reports 58.2-58.8,
a 20% overstatement that has nothing to do with the arm's contents. A real
search shows the same thing in its own numbers - the first 5-second bucket of
a fresh run does 72.8 evals/s against 52-58 for the rest of it.

Two rules come out of it:

- **Discard the first arm of a replay sweep**, or run a throwaway warm-up arm.
  Every table in this document that was measured after this was found does.
- **This is not the intra-run drift** in the last section. That one is the
  sampler contracting onto expensive parameters over minutes; this one is the
  package clock settling over seconds, and it is done before a search has
  finished its first hundred evaluations.

## The whole per-evaluation budget, measured

4770 evaluations of a default WSClean search (`--nlive 25 --num-repeats 10`,
20 ranks = 19 workers), reading each evaluation's `metrics.json` for the stage
timings and its `wsclean.stdout.log` for `wsclean`'s own accounting:

| | per evaluation | share |
|---|---:|---:|
| `wsclean` inversion (~7.5 of them) | 125.2ms | 37% |
| `wsclean` prediction (~6.5 of them) | 77.4ms | 23% |
| `wsclean` deconvolution | 19.7ms | 6% |
| `wsclean` everything else | 67.7ms | 20% |
| **`wsclean` binary total** | **290.0ms** | **86%** |
| simulate | 37.6ms | 11% |
| sidecar `docker exec` round trip | 7.3ms | 2% |
| metrics (read three FITS, score) | 1.5ms | 0.5% |
| **worker-seconds per evaluation** | **336.4ms** | |

The major-cycle loop - inversion plus prediction, 60% of the budget - is set
by `-mgain 0.8` and `-niter 100`, which are the fixed hyperparameters the
search is defined against (`wsclean_fixed_hyperparameters` in `summary.json`).
It is not harness overhead and it is not available to optimise.

Everything that *is* overhead comes to about a third: 67.7ms of non-imaging
`wsclean`, 37.6ms of simulate, 7.3ms of `docker exec`.

### Per major cycle: mostly fixed cost, not gridding

Bucketing the same 4770 evaluations by row count separates the two:

| rows | n | inversion per cycle | prediction per cycle | `wsclean` binary |
|---:|---:|---:|---:|---:|
| <500 | 197 | 10.6ms | 7.7ms | 212.8ms |
| 1000-1500 | 841 | 14.3ms | 10.1ms | 258.1ms |
| 2000-2500 | 1195 | 17.0ms | 12.2ms | 295.8ms |
| 3500+ | 591 | 20.4ms | 14.6ms | 339.5ms |

which fits `inversion = 9ms + 3.2ms per 1000 rows` and `prediction = 7ms +
2.1ms per 1000 rows` at 19 workers. **Roughly half the major-cycle cost is
fixed per call**, and the image is only 128x128: serially, `-niter 0` (which
still does the PSF and the dirty inversion) costs 38.4ms at `-size 16` and
40.6ms at `-size 128` - and 55.5ms at `-size 512` - so the gridding arithmetic
at this image size is
effectively free and what is being paid is per-call setup inside `wsclean`.
There is no flag for it.

### `wsclean` startup is casacore's static initialisers

`wsclean --version` - fork, exec, 73 shared libraries, constructors - is
13.8ms of the 148.6ms an evaluation's `wsclean` costs serially, i.e. ~9%, and
about 27ms of the 290ms at 19 workers. `LD_DEBUG=statistics` puts the dynamic
loader at 2.9M cycles (~0.9ms, 45k relocations), so it is constructor work.
`LD_PRELOAD`ing one library into `/bin/true` says which:

| | per exec |
|---|---:|
| `/bin/true` | 0.6ms |
| + `libcasa_casa.so.9` | 6.3ms |
| + `libcasa_ms.so.9` (pulls tables, measures, scimath) | 9.4ms |
| `wsclean --version` | 13.8ms |

`libmpi`, `libpython3.11` and `libhdf5` - all linked, none used by this
workload - cost 0.1ms, 0.2ms and 1.7ms respectively, so trimming the link line
is not the win it looks like. It is casacore's own static initialisation, it
is upstream, and there is no knob.

## Nine more ways of making the evaluation cheaper that do not work

Measured with the replay harness at 19 concurrent workers, each arm run at
least twice after a throwaway warm-up arm (see "The first four seconds"),
repeatable to about 0.5%. Baseline 58.3-58.8 evals/s.

| change | evals/s | verdict |
|---|---:|---|
| `OMP_NUM_THREADS=1` in the sidecar | 58.6 | no change |
| `-padding 1.0` | 58.6 | no change |
| `-parallel-gridding 1` | 58.4 | no change |
| `-temp-dir` on `/dev/shm` | 58.2 | marginally worse |
| `-gridder wstacking` | 56.6 | worse |
| `-no-small-inversion` | 56.6 | worse |
| `-gridder tuned-wgridder` | *crashes* | ducc0 `no appropriate kernel found` |
| `-wgridder-accuracy 1e-3` | 63.1 | +8%, wrong trade (below) |

Notes worth keeping:

- **`wsclean` runs 20 threads under `-j 1`.** They are a pool, they are
  blocked rather than spinning, and `OMP_NUM_THREADS=1` changes neither the
  count nor the throughput. It looks alarming at 19 concurrent workers and is
  not.
- **`-gridder tuned-wgridder` appears to be 4.8x faster** and is not: it aborts
  in ducc0's `gridding_kernel.h` before writing anything, and the "throughput"
  is the rate of failing. Any replay arm that beats the baseline by more than
  a few percent should have its exit status checked first.
- **`-wgridder-accuracy 1e-3`** is the same wrong trade as the `1e-2` already
  rejected above, one order of magnitude in: 1e-3 of a 1 Jy peak is 1mJy of
  gridding error against an objective (`total_rms_jy`) measured between 0.1mJy
  and 25mJy.

## The simulate stage: two thirds of what is left is two constant columns

Timing the stage's own phases inside the meqtrees image, serially, against the
skeleton cache:

| | serial |
|---|---:|
| write makems config | 0.4ms |
| skeleton copy + `SPECTRAL_WINDOW` patch | 2.0ms |
| fill visibilities | 8.3ms |
| move out of `/dev/shm` into the evaluation directory | 2.7ms |
| **total** | **13.4ms** |

and inside "fill visibilities", **4.7ms of the 8.3ms is the TaQL `UPDATE ...
SET WEIGHT, SIGMA`** that writes two constants to every row. Everything else
in the stage - opening the MS, reading `DATA` and `UVW`, generating the noise,
writing `DATA` and `FLAG` - comes to 2.6ms combined.

The comment in `fill_point_source_visibilities()` already records that TaQL is
the fastest of the obvious options, and re-measuring agrees: `putcol` on the
pair costs 62.3ms (it is quadratic in rows on an `IncrementalStMan`
variable-shaped column), a `putcell` row loop 11.1ms, TaQL 4.7ms. Two further
attempts failed:

- **Writing row 0 only** and letting `IncrementalStMan` propagate it forward.
  It does not: makems leaves an explicit entry on many rows (the group holds
  `TIME`, `INTERVAL` and 15 other columns that do change per row), so row 0
  alone leaves every other row at makems' 1.0.
- **Collapsing the column first** with one full TaQL update at skeleton-publish
  time, then writing row 0 per evaluation. Same result - the ISM does not
  collapse to a single entry.

> **Corrected below.** This section closed with "dropping the write entirely
> is not available either ... anything that fixes this has to change what
> makems writes, not what the simulator writes afterwards". That was wrong:
> the write is gone, and makems was not touched. See "The two constant columns
> are gone" below. Every measurement above still reproduces.

Dropping the write looked unavailable: `ms_to_r2d2_mat.py` reads `WEIGHT` to
build R2D2's `nW = sqrt(1/sigma^2)`, so the column is load-bearing for the R2D2
search even though WSClean, weighting naturally with a constant weight, is
indifferent to its value.

## The two constant columns are gone

The [simulate-stage section](#the-simulate-stage-two-thirds-of-what-is-left-is-two-constant-columns)
above measured the TaQL `UPDATE $ms SET WEIGHT, SIGMA` at a third of the stage
and concluded that dropping it "has to change what makems writes". It does
not, because **makems already writes exactly the value the column needs to
hold**: a fresh skeleton comes out with `WEIGHT = SIGMA = 1.0` on every row.
The simulator was overwriting a uniform 1.0 with a uniform `1/sigma^2` - one
number, replicated across the whole Measurement Set, at 1us per row per column
because the pair sits in a variable-shaped `IncrementalStMan` group.

That number is already recorded, per evaluation, in `simulation.json` as
`noise.complex_sigma_jy`. So the write is gone and the sigma travels as data:

- `fill_point_source_visibilities()` leaves `WEIGHT` and `SIGMA` at makems' 1.0.
- `ms_to_r2d2_mat.py` takes `--noise-sigma-jy` and computes
  `nW = sqrt(WEIGHT) / sigma`, which is the same number the old
  `nW = sqrt(1/sigma^2)` produced while `WEIGHT` is 1.0. It defaults to 1.0, so
  the tool is still correct on a Measurement Set that carries real weights.
- `polychord_r2d2.py` reads `simulation.json` before the convert instead of
  after it, and passes the sigma.

`simulation.json` was already the channel simulator metadata reaches the
imagers through - both runners read `observation.max_proj_baseline_lambda` from
it for the cell size rather than recomputing it from the MS - so this is the
existing route, not a new one.

`self_check_noise_weighting()` (`./ri self-check simulate`) is what keeps the
equivalence honest. It asserts both halves at two dynamic ranges 10^5 apart:
that `WEIGHT` really is still all 1.0 after a simulate, and that `nW` comes out
elementwise equal to `sqrt(1/sigma^2)`. If a future makems, or a change to the
skeleton cache, ever leaves a different constant there, the first assertion
fails rather than the science quietly moving.

### What it is worth

The stage, timed inside the meqtrees image against the baked skeleton cache,
20 evaluations per worker, median:

| | before | after |
|---|---:|---:|
| serial | 10.97ms | 7.58ms |
| 19 concurrent workers | 22.80ms | 15.51ms |

and the phases at 19 workers:

| | before | after |
|---|---:|---:|
| skeleton copy + `SPECTRAL_WINDOW` patch | 4.06ms | 4.24ms |
| fill visibilities | 11.62ms | 4.60ms |
| &nbsp;&nbsp;of which the TaQL `UPDATE` | 7.14ms | - |
| move out of `/dev/shm` into the evaluation directory | 5.32ms | 5.48ms |
| **total** | **22.80ms** | **15.51ms** |

End to end, three interleaved seed pairs at `--nlive 25 --num-repeats 10
--max-ndead -1 --mpi-procs 20`, behind a discarded warm-up arm for the
[burst-clock trap](#the-first-four-seconds-of-any-measurement-lie-by-20), the
two arms differing only in which `MEQTREES_IMAGE` the search was handed:

| seed | evaluations/s before | after | | simulate ms/evaluation before | after |
|---|---:|---:|---:|---:|---:|
| 4242 | 51.74 | 54.34 | +5.0% | 36.14 | 24.74 |
| 7 | 53.08 | 53.12 | +0.1% | 36.79 | 24.81 |
| 99 | 50.60 | 53.82 | +6.4% | 37.93 | 24.86 |
| **mean** | **51.80** | **53.76** | **+3.8%** | **36.95** | **24.80** |

Read the two halves differently. The simulate column is the real measurement -
**-12.1ms per evaluation, -33%, in all three pairs and within 0.5% of itself
across them**. It is *larger* than the 7.3ms the isolated 19-worker timing
predicted, because in a real search the stage contends with 19 concurrent
`wsclean` processes rather than 19 other simulates. The evaluations/s column
carries the whole run's variance on top of a ~3% effect, which is why one pair
lands on +0.1%; take the +3.8% mean as the estimate and the -12.1ms as the
fact.

### What it costs: the objective moves in its seventh decimal

WSClean weights naturally, so a uniform weight images the same sky whatever
the constant is - but not to the last bit, because summing `w * V` in float32
rounds differently when `w` is 3.7e3 or 7.8e9 rather than 1.0. Re-imaging five
recorded evaluations spanning the run's cost distribution, once with the
`WEIGHT` the old simulator wrote and once with 1.0:

| | max relative change |
|---|---:|
| `total_rms_jy` (the default objective) | 4.5e-7 |
| `sigma_res` | 2.9e-4 |
| dirty, restored and PSF pixels | 6.6e-7 |

WSClean is bit-reproducible on the same input (the same MS twice gives
identical FITS), so these are real differences, not measurement noise. They
are the same size as the ones [the native build](#what-does-work-build-wsclean-for-the-cpu-it-runs-on)
already introduced for +6-10% - median 9.7e-8, max 3.7e-7 on the objective -
and four orders of magnitude below the gridder's own `1e-4` accuracy setting.
`sigma_res` amplifies it because it is a ratio of a near-zero residual to the
dirty image; the absolute move is 4.4e-8 on 1.5e-4.

R2D2's input does not move at all: `nW` is computed to the same float64 value
the column used to hold, which is what the self-check asserts.


## The sampler itself has no slack

Three completed WSClean searches (`--nlive 25 --num-repeats 10`, five sampled
dimensions) cost **20.9 likelihood evaluations per dead point** (6527/312,
6830/324, 6350/313). That is `num_repeats` slice-sampling steps at ~2.1
likelihood calls each, which is about as cheap as PolyChord's slice sampler
gets - there is no wasted bracket expansion to reclaim.

The consequence for the bigger runs is the plain one: evaluations scale
linearly in `num_repeats`, and `num_repeats = 10` at 5 dimensions is 2 per
dimension, well under PolyChord's own `5 x ndims` guidance. A statistically
respectable `--num-repeats 25` is 2.5x the evaluations at the same `--nlive`,
and nothing in the harness will absorb that.

## The disk footprint is what caps a big run, not the clock

Every section above is about evaluations per second. This one is about the
other way a big run fails to finish.

A WSClean evaluation directory held a mean **1.44MB** (109 evaluations of a
default search), and the Measurement Set was almost all of it:

| | one evaluation |
|---|---:|
| `sim.ms` | 1.50MB |
| five `wsclean/recon-*.fits` | 0.38MB |
| `VLAA_ANT`, logs, `metrics.json`, `simulation.json` | 0.06MB |
| **mean over 109 evaluations** | **1.44MB** |

(The mean is under the sum because `sim.ms` shrinks with `channel_count` and
`observation_minutes`.) At the ~45-52 evaluations a second the rest of this
document is about, that is **~65MB/s**, and it is never freed: 202GB of free
space is **~140k evaluations**. The runs this repo is aiming at are larger than
that on their own - `--nlive 500 --num-repeats 25` is 500/50 x 25/10 = 25 times
a default `--nlive 50` run's ~11k evaluations, so ~270k evaluations and ~390GB.
The disk fills about half way through and the run ends there, having spent
every hour of imaging it did on a result nobody can read.

Nothing outside an evaluation reads its Measurement Set. `./ri profile`,
`./ri health`, `./ri report`, `merge` and the resume path all read
`metrics.json`; the report renders the FITS images; `summary.json` records
every argv. So a search now deletes `sim.ms` (and, on the R2D2 side, the
`r2d2_data.mat` derived from it) as it writes the evaluation's
`metrics.json` - `prune_evaluation_artefacts()` in `common.py`, which is the
one funnel both imagers' `evaluate()` return through. Measured on the same
default search: **1.44MB -> 0.43MB per evaluation, 3.4x**.

Two things it deliberately does not do:

- **A failed evaluation keeps everything.** A failure is what this project
  exists to find, and its inputs are the first thing anyone will want.
- **The images stay.** They are the evidence, they are 0.38MB against the MS's
  1.50MB, and the report renders them.

The MS is reproducible from the record either way: `params` carries the
`noise_seed`, so re-running the recorded `simulate` argv rebuilds it byte for
byte. `./ri search --keep-measurement-sets` (`NS_KEEP_MEASUREMENT_SETS=1`)
keeps them for every evaluation, which is what the replay benchmarks earlier
in this document need. Both that flag and `--synchronous` are now written to
the run's `run.env`, so `./ri resume` no longer silently changes either one
part-way through a run.

### It costs no throughput

Three interleaved A/B pairs, `--nlive 25 --num-repeats 10 --mpi-procs 20`,
same images, a throwaway warm-up arm first (see the burst-clock section):

| seed | keeping the MS | pruning it |
|---|---:|---:|
| 4242 | 50.94 evals/s | 52.28 evals/s |
| 7 | 52.62 | 51.71 |
| 99 | 52.13 | 50.96 |
| **mean** | **51.90** | **51.65** |

-0.5%, against a within-arm spread of +/-1.7% - noise, and in both directions.
That is what the budget predicts: `shutil.rmtree` of one `sim.ms` (81 files) is
a median **0.52ms** against a ~336ms evaluation, or 0.15%, and iteration 3
already established that this host is not I/O-bound (the whole `evaluations/`
directory on `/dev/shm` moved replay throughput by under 1%).

So this is not a speed change and is not offered as one. It is the change that
lets a run long enough to need the speed actually reach its end.

## The next wall after disk: a resume of a big run wanted 62GB of RAM

With `sim.ms` pruned, `--nlive 500 --num-repeats 25` fits on the disk (~270k
evaluations x 0.43MB = ~116GB against 202GB free). The next thing it hits is
memory, and it hits it on a path nobody profiles because it runs for ten
seconds at startup: `adopt_completed_evaluations()`.

A run that long *will* be resumed - `run_with_retries` restarts the job after a
worker dies, the stall watchdog restarts it after a hang, and `./ri resume`
picks it up after a reboot. Every restart re-reads every `metrics.json` written
so far, and **every rank does it**, because the cache and the eval-id counter it
rebuilds are per-rank state. Measured against a synthetic `evaluations/`
directory of 120,000 real records:

| adopted evaluations | wall | peak RSS, *per rank* |
|---|---:|---:|
| 120,000, keeping the records | 5.6s | 1371MB |
| 120,000, keeping only the objective | 4.5s | 119MB |
| 270,000 (extrapolated), keeping the records | 12.6s | 3084MB |
| 270,000 (extrapolated), keeping only the objective | 10.2s | 268MB |

At 20 ranks that is **62GB on a 62GB host** against 5.4GB - a guaranteed OOM
kill at the target run size, and already 27GB (uncomfortable next to the
workers) at half of it.

Nothing ever read the records back. The likelihood does
`return float(cache[key]["objective"])`, and rank 0's `summary.json` re-reads
every record from disk with `load_evaluations_from_dir()` at the end of the run.
The list the records were appended to was used for exactly one thing:
`len(evaluations) + 1`, the next eval id. So the cache is now `dict[str, float]`
and the list is an `int`, in both `polychord_wsclean.py` and
`polychord_r2d2.py`. Behaviour is unchanged - same ids, same cache hits, same
scores - and `./ri self-check self-heal`, which kills and resumes five real
searches, covers it.

### And the other end of the same run: `json.dumps` was the peak

`write_json_atomic()` built the whole document as a string before writing it.
For `metrics.json` that is 4KB and irrelevant; for a 270k-evaluation
`summary.json` it is a **1.3GB string held alongside the 3.1GB of records it
was built from**. Streaming it with `json.dump(payload, handle)` instead
produces a byte-identical file and keeps the rename-into-place atomicity:

| 270k-evaluation summary.json | wall | peak RSS above the records |
|---|---:|---:|
| `dumps()` to a string, then write | 5.9s | 2633MB |
| `json.dump()` streamed to the file | 13.2s | 0MB |

7 seconds more, once, at the end of a multi-hour run, for 2.6GB of headroom on
the rank that is already holding the most. On a 4KB `metrics.json` the streamed
form costs 22us against a 336ms evaluation - 0.007%, and it is written once per
evaluation, so it does not reach the throughput either.

### What is still ahead on this path

`summary.json` itself. At 270k evaluations it is a **1.3GB single JSON
document** that every reader (`./ri profile`, `./ri report`, `merge`) loads
whole - ~3.1GB of Python objects to answer a question about one field. The
record is 3.9KB and 3.0KB of that is `paths` and `commands`, which are the same
two absolute-path prefixes and the same argv repeated on every line. Fixing it
means changing the on-disk format and every reader of it, which is why it is
recorded here rather than done.

## The progress bar cost more the longer the run got, and now it does not

The bar `run_with_progress` pins to the terminal redrew once a second, and
drawing it is O(evaluations): `_ns_count_evals()` walks `evaluations/` twice,
once for the total and once with `-newer` for the evaluations banked since the
last checkpoint. On a warm 270,000-entry directory here those two passes
measure 0.37s and 0.35s (~1.35us per entry each), so the *redraw* took 0.72s
and the *cycle* took 1.76s: the bar was busy for most of every cycle, for the
whole of the run, and it got worse the longer the run went on.

The fix is not a cheaper count, it is a slower one. `_ns_redraw_interval()`
sleeps nine times the draw's own cost instead of a fixed second, so the bar is
held to roughly a tenth of one core at any run size. Measured against a
synthetic 270,000-evaluation run directory, 60s per arm, host otherwise idle:

| redraw cadence | draws in 60s | CPU (user+sys) | share of one core |
|---|---:|---:|---:|
| fixed `sleep 1` | 34 | 26.6s | 44% |
| back off to 9x the draw | 9 | 7.0s | 12% |

Each arm was run twice, interleaved, and reproduced to within 0.1s of CPU.

2.2% of this 20-thread machine down to 0.6%, and the important part is the
second column: the fixed cadence scales with the run (a 1M-evaluation run
would have had the bar at ~8% of the machine) while the back-off does not.
Below ~100k evaluations a draw costs under 0.11s, the interval stays at one
second and nothing changes; at 270k the bar updates every ~7s instead, which
is well inside the interval a human reads it over. The back-off is capped at
30s so a pathological draw cannot leave the line looking dead.

Three things worth knowing about this one:

- It only ever cost anything on a TTY. The pinned-bar loop is behind
  `[ -t 1 ]`, so a search started under `nohup` - which is how a multi-day run
  should be started - never paid it.
- The one-pass fix is not available. POSIX `find` has no way to mark which
  entries matched `-newer`, and `-printf '%T@'` is GNU-only while CI runs this
  file's self-check on macOS; and it would only have halved a cost that still
  grows with the run.
- The timing uses `EPOCHREALTIME`, which is bash 5. On bash 3.2 (macOS)
  `_ns_now_us` returns nothing and the interval falls back to one second,
  which is what a run small enough to be started on a laptop wants anyway.

The stall watchdog's `find -maxdepth 2 -name metrics.json` is the same shape
and was left alone at the time on the strength of a synthetic measurement. It
is worse than that measurement said and it now shares this back-off - see
[the stall watchdog](#the-stall-watchdog-was-the-last-poll-that-grew-with-the-run)
at the end of this page.

## PolyChord itself costs microseconds a call, at every `nlive`

[The one number that matters](#the-one-number-that-matters) asserts that
PolyChord's own sampling is not a meaningful part of the unaccounted
remainder. This is the number behind the assertion, and it is worth having
because the obvious worry about a much larger `--nlive` is that the sampler's
own bookkeeping - the live-point insertion, the Cholesky, the cluster search -
grows with the number of live points.

Running `pypolychord` serially against a free likelihood (a 5-dimensional
Gaussian, ~1us a call) and dividing the whole run's wall clock by its
likelihood-call count charges the sampler for everything it does per call:

| `nlive` | `num_repeats` | likelihood calls | wall | per call |
|---:|---:|---:|---:|---:|
| 25 | 10 | 16,182 | 0.085s | 5.3us |
| 50 | 10 | 30,381 | 0.211s | 6.9us |
| 100 | 10 | 59,783 | 0.380s | 6.4us |
| 200 | 10 | 124,085 | 0.771s | 6.2us |
| 500 | 10 | 311,289 | 1.713s | 5.5us |
| 1000 | 10 | 615,496 | 3.387s | 5.5us |
| 500 | 25 | 776,932 | 2.686s | 3.5us |
| 1000 | 25 | 1,545,778 | 5.213s | 3.4us |

Against this repo's ~350ms evaluation that is **0.002% of the run**, and it
does not grow with `nlive`: forty times the live points costs the same per
call, and `--num-repeats 25` costs *less* per call than 10, because the fixed
per-iteration work is spread over more slice steps.

So there is no "running PolyChord" half of the wall clock to go after at any
run size this repo will reach. Whatever is unaccounted is idle, and the next
section is where the idle goes.

Two notes on reproducing the table: it is preceded by a throwaway
`run_polychord`, because the first one in a process measures ~22us a call and
that is import and first-allocation cost; and `max_ndead` is left unset,
because setting it does not stop a run where it says it will (a run asked to
stop at 2000 dead points terminated on the evidence criterion at 388).

## A bigger run is a *more* efficient run

Every throughput number above this line was taken on a two-minute `--nlive 25`
or `--nlive 50` search, and the scaling to the runs this repo actually wants
was assumed. Measured, on the current working tree, one search per row, same
seed (4242), `--num-repeats 10 --mpi-procs 20`, one warm-up search discarded
first for the [burst-clock trap](#the-first-four-seconds-of-any-measurement-lie-by-20):

| `nlive` | dead points | evaluations | evals/dead | wall | evals/s | per evaluation | utilisation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 310 | 5,925 | 19.1 | 113s | 52.6 | 339ms | 93.9% |
| 50 | 527 | 10,757 | 20.4 | 203s | 53.0 | 346ms | 96.4% |
| 100 | 936 | 18,844 | 20.1 | 353s | 53.4 | 347ms | 97.6% |
| 200 | 1,728 | 34,682 | 20.1 | 649s | 53.5 | 352ms | 98.3% |

**Utilisation rises monotonically with the run and throughput holds** - which
it should not do, because the evaluation itself gets 4% more expensive over
this range (the sampler contracting onto the high-`channel_count` corner, as
in [Throughput falls through a run](#throughput-falls-through-a-run-and-it-is-the-sampler-doing-it)).
The two cancel.

Turning the utilisation column back into wall clock says why. The
non-evaluation wall time of each run is `(1 - utilisation) x wall`:

| `nlive` | evaluations | non-evaluation wall |
|---:|---:|---:|
| 25 | 5,925 | 6.9s |
| 50 | 10,757 | 7.3s |
| 100 | 18,844 | 8.5s |
| 200 | 34,682 | 11.0s |

That is a **fixed ~6s per run plus ~0.14ms per evaluation**, not a rate. The
constant is startup - MPI init, `warm()`, and the ramp while the first
`nlive` prior points are drawn - and it is the whole of the 6% that an
`--nlive 25` search appears to lose. The per-evaluation term is 2.7ms of
worker time against a 352ms evaluation, i.e. **0.8%**, which is the MPI
hand-off of a point to a worker and its answer back, and is where utilisation
asymptotes: ~99%.

The practical consequences:

- **The idle time this document opened on is gone at the sizes this project is
  aiming at.** It was 35-45% before asynchronous MPI, 6% at `--nlive 25`
  today, and 1.7% at `--nlive 200`. Do not spend any more effort on it.
- **Do not size a big run from a small one's `evals/s`,** and do not read a
  small run's utilisation as a defect. A two-minute search is paying a 6s
  startup over its whole length.
- **Dead points are sub-linear in `nlive`.** The `ndead/nlive` ratio falls
  12.4 -> 10.5 -> 9.4 -> 8.6 across the table, so a big run is cheaper than a
  linear extrapolation from `--nlive 25` predicts, not dearer.

Which makes the target run concretely affordable. At `--nlive 500` the ratio
extrapolates to ~7.8, so ~3,900 dead points; `--num-repeats 25` is 2.5x the
20.1 evaluations per dead point, so ~50; that is **~196,000 evaluations**, or
about **65-90 minutes** at 50-53 evals/s and **~78GB** at the 0.40MB an
evaluation directory now costs. Both fit this host. The earlier estimate on
this page of 270,000 evaluations and 390GB predates the sub-linear `ndead`
measurement and the [`sim.ms` pruning](#the-disk-footprint-is-what-caps-a-big-run-not-the-clock).

### The harness does not decay as the evaluations pile up

The same `--nlive 200` run, split into twelve equal wall-clock buckets, with
utilisation computed per bucket from the evaluations that landed in it:

| bucket | evals/s | per evaluation | utilisation | mean `channel_count` |
|---:|---:|---:|---:|---:|
| 1 | 58.6 | 321ms | 0.989 | 4.36 |
| 2 | 55.2 | 341ms | 0.991 | 4.67 |
| 4 | 55.6 | 339ms | 0.991 | 5.36 |
| 6 | 54.4 | 346ms | 0.991 | 5.98 |
| 8 | 50.7 | 372ms | 0.992 | 6.45 |
| 10 | 52.0 | 362ms | 0.991 | 7.08 |
| 12 | 45.9 | 377ms | 0.911 | 7.35 |

**Utilisation is flat at 0.99 for eleven twelfths of the run** while
throughput falls 11%, so the fall is entirely the evaluation getting dearer
(321ms -> 377ms, a 1.18x first-to-last-decile drift that matches the 1.17x
median already recorded here) as `channel_count` climbs 4.36 -> 7.35. Nothing
in the harness gets slower as `evaluations/` fills: the flat-directory cost
was checked directly too, and creating an evaluation directory with ten files
in it costs 0.094ms whether the parent holds 0 entries or 280,000.

The last bucket is the run ending - the final chains finishing while the other
workers have nothing left to start - and it is a fixed cost, not a rate, in
the same way the startup is.

## The stall watchdog was the last poll that grew with the run

The [progress bar section](#the-progress-bar-cost-more-the-longer-the-run-got-and-now-it-does-not)
above ends by saying the stall watchdog's `find -maxdepth 2 -name metrics.json`
is the same shape and was left alone at "1.24s at 270k evaluations, 2% of one
core". Measured properly on a real run rather than a synthetic tree, it is
worse than that, because it walks every evaluation *directory* rather than
just `evaluations/`:

| scan | at 34,682 evaluations | per evaluation |
|---|---:|---:|
| `find -maxdepth 1 -name 'eval-*'` (the bar) | 0.05s | 1.4us |
| `find -maxdepth 2 -name metrics.json` (the watchdog) | 0.28s | 8.1us |

and 2.4x that on a loaded host, which is the only kind a watchdog runs on
(0.20s at 10,284 evaluations while the search was live). At the 270,000
evaluations of a `--nlive 500 --num-repeats 25` search that is of order 5s of
every 60s poll, and it keeps rising: a million-evaluation run would spend a
third of a core on it.

`_ns_redraw_interval` is now `_ns_backoff_interval <started> [floor] [cap]` and
the watchdog uses it too, with the caller's `NS_STALL_POLL_SECONDS` as the
floor and a 600s cap. Same rule as the bar - sleep nine times the poll's own
measured cost - so the loop is bounded at ~10% of one core at any run size.

Two honest caveats:

- **At today's target run size this changes nothing.** Nine times a 5s scan is
  45s, under the 60s floor, so the cadence only moves above ~450,000
  evaluations. What the change buys is the bound, not a saving today.
- **The signal is unchanged.** Counting `metrics.json` rather than `eval-*`
  directories is deliberate - a directory without a record is an evaluation
  that was in flight when the run died - and cheaper stall signals were
  rejected for that reason. `evaluations/` mtime does not move when a record is
  written, and `run.log` growing does not mean an evaluation landed, so a rank
  retrying forever without dying - the exact failure this watchdog is the
  backstop for - would look alive to both.

The 600s cap is what keeps the message it prints honest: `quiet` accumulates in
whole polls, so a poll longer than a tenth of `NS_STALL_TIMEOUT` would start
reporting a stall length noticeably past the timeout it fired on.

## The AVX2 build stopped being opt-in

[What does work: build WSClean for the CPU it runs on](#what-does-work-build-wsclean-for-the-cpu-it-runs-on)
above measured `-DPORTABLE=OFF` - WSClean compiled with `-march=native` - at
+19.8% serial and +6.5% at matched concurrency, then left it behind
`./ri search wsclean --native` because such a binary dies with SIGILL on any
other CPU and the image tag is shared with every other worktree on the host.
So the default build shipped without AVX2 or FMA, and every search this repo
has ever run paid for that.

That was a false choice. WSClean's `SetTargetCPU.cmake` takes `-DTARGET_CPU=`
alongside `-DPORTABLE=OFF`, documented for exactly this - "building containers
for a different platform" - and `x86-64-v3` is a *named* target, not the build
machine's: AVX2, FMA, BMI1/2, on every x86-64 CPU since Haswell (2013) and
Zen (2017). One fixed binary, so two hosts still image bit-identically, which
is more than `native` could say.

`WSCLEAN_TARGET_CPU` is now the one knob over the gridder's instruction set
and it defaults to `x86-64-v3`. `--native` sets it to `native`; setting it
empty gets the old plain-x86-64 baseline back.

### What it is worth

Replaying 378 recorded evaluations of one `--nlive 25 --num-repeats 10` search
(the harness from
[where the evaluation's time actually goes](#where-the-evaluations-time-actually-goes),
eight passes over the corpus per arm so each arm is ~40s and clear of
[the first four seconds](#the-first-four-seconds-of-any-measurement-lie-by-20)),
interleaved, behind a throwaway warm-up arm:

| | baseline | `x86-64-v3` | |
|---|---:|---:|---:|
| 1 worker | 8.65 / 8.55 evals/s | 9.96 / 10.08 evals/s | **+15.1% / +18.0%** |
| 19 workers | 72.8 / 71.6 / 71.6 evals/s | 77.9 / 78.1 / 77.8 evals/s | **+7.1% / +9.1% / +8.6%** |

End to end - `./ri search wsclean --nlive 25 --num-repeats 10 --max-ndead -1`,
default 20 ranks, three seeds, the two images alternating with `--no-build`:

| seed | evals/s baseline | evals/s `x86-64-v3` | | `wsclean` binary |
|---:|---:|---:|---:|---:|
| 4242 | 54.8 | 59.3 | +8.2% | 290.3ms -> 267.4ms (-7.9%) |
| 7 | 52.3 | 58.0 | +10.8% | 304.3ms -> 275.7ms (-9.4%) |
| 99 | 56.3 | 57.4 | +2.0% | 303.3ms -> 275.5ms (-9.2%) |

The `wsclean` binary column is the one to trust - it is a mean over ~6500
evaluations rather than one wall clock - and it is -9.2% at all three seeds
where the end-to-end column spans +2% to +11%, which is the same run-to-run
throughput noise
[the sampler's own variance](#throughput-falls-through-a-run-and-it-is-the-sampler-doing-it)
has produced throughout. Plan with the replay's +8.6% at matched concurrency.

That is nearly all of what `--native` was worth (+19.8% serial, +6.5% at 20
workers on its own corpus). `--native` still exists and is still a little
faster, but it is now a few percent on top of a good default rather than the
only way to get AVX2.

### What it costs: eleven float32 ULPs at the peak pixel

Recomputing every metric from both builds' FITS over the same 378 evaluations
(same Measurement Sets, so any difference is the arithmetic):

| metric | max relative difference |
|---|---:|
| `total_rms_jy` (the default objective) | 4.9e-7 |
| `relative_l2_error` | 4.9e-7 |
| `peak_jy_per_beam` | 3.6e-7 |
| `snr` | 1.8e-5 |
| `off_source_rms_jy` | 1.8e-5 |
| `sigma_res` | 6.9e-3 |
| `peak_flux_abs_error_jy` | 9.1e-2 |

The objective moves in its seventh decimal, four orders of magnitude below the
gridder's own 1e-4 accuracy setting. The two large-looking rows are both
ratios of near-zero quantities and neither is a surprise:
`peak_flux_abs_error_jy` is `|peak - 1.0|` at 1.31e-6 against 1.19e-6 Jy, i.e.
about eleven float32 ULPs of a 1.0 Jy peak, and `sigma_res` amplifies rounding
~600x for the reason
[the constant-columns section](#what-it-costs-the-objective-moves-in-its-seventh-decimal)
already records. FMA contraction changes the order of a float32 summation;
that is the whole of it.

### Two things this changed on the way

- **The CPU target no longer rebuilds casacore.** `ARG WSCLEAN_TARGET_CPU` is
  declared immediately above the WSClean step rather than with the other args
  at the top of the build stage, because an `ARG` instruction is itself a cache
  key for every layer below it - so flipping the flag used to recompile
  casacore, which is not built with it.
- **`make -j` follows `nproc`.** The Dockerfile's `BUILD_JOBS` default of 4 put
  a from-scratch WSClean image at ~40 minutes; `scripts/build.sh` now passes
  `nproc`, which brings it to ~9 minutes on this 20-thread host. It is
  deliberately outside `inputs_hash`: it changes how long the build takes, not
  what it produces.

## The Measurement Set never reaches the disk any more

The simulator builds its Measurement Set in `/dev/shm` and then moved the
finished tables onto the bind mount, because each container's `/dev/shm` is
private and the imager runs in a different one. For a WSClean search that copy
existed only so a second container could open a file that the next few hundred
milliseconds delete again.

`NS_SCRATCH_DIR` removes it. `scripts/lib/start-sidecars.sh` creates one host
tmpfs directory per run (`/dev/shm/ri-ns-scratch-<pid>`) and bind-mounts it into
all three containers at its own path, exactly as `$REPO_ROOT` already is; the
rank puts the evaluation's MS in `<scratch>/<eval dir name>/` rather than in the
evaluation directory (`evaluation_scratch_dir()` in `common.py`), the imager
reads it there, and `prune_evaluation_artefacts()` deletes it as `metrics.json`
is written. Only the evaluations in flight are ever present, so the directory
holds ~1.5MB per rank, not per evaluation, and the run's `EXIT`/`INT`/`TERM`
trap removes it. An evaluation that *failed*, and every evaluation under
`--keep-measurement-sets`, has its scratch contents moved back beside its
record first - the scratch is RAM and goes away with the run, and a failure's
inputs are the whole point of this search. With `NS_SCRATCH_DIR` unset (a
self-check, a host with no writable `/dev/shm`) everything behaves exactly as
before.

### The isolated rig said 5.1ms; the search says 1.2ms

Nineteen concurrent simulate processes, 60 evaluations each, median over the 19
workers, three interleaved pairs, first arm discarded for the burst clock:

| destination | simulate total | `move sim.ms` | skeleton | fill |
|---|---|---|---|---|
| bind mount (ext4) | 14.3 / 14.2 / 14.3 ms | 4.66 / 4.67 / 4.72 ms | 4.26 / 4.17 / 4.20 ms | 3.35 / 3.60 / 3.43 ms |
| shared tmpfs | 9.10 / 9.16 / 11.3 ms | 0.01 ms | 4.15 / 4.19 / 5.04 ms | 3.49 / 3.58 / 4.23 ms |

Same-filesystem, the move is a `rename()`; cross-device it is a 66-file
copytree. So the rig puts the copy at 5.1ms, 36% of the simulate stage.

Three interleaved end-to-end pairs (`--nlive 25 --num-repeats 10 --mpi-procs
20`, seed 4242, `sim`/`bin` are the summary's per-evaluation stage means):

| pair | scratch on ext4 | scratch on tmpfs |
|---|---|---|
| 1 | sim 24.97 ms, bin 269.1 ms, 58.00 evals/s | sim 23.35 ms, bin 260.4 ms, 60.02 evals/s |
| 2 | sim 24.73 ms, bin 264.0 ms, 59.29 evals/s | sim 23.65 ms, bin 262.9 ms, 59.02 evals/s |
| 3 | sim 24.82 ms, bin 263.4 ms, 60.47 evals/s | sim 24.04 ms, bin 269.5 ms, 60.49 evals/s |

The simulate column moves by -1.6/-1.1/-0.8 ms, consistent in sign and size;
the evals/s column does not move at all, because 1.2ms is 0.4% of a 327ms
evaluation and the run-to-run swing in evaluation count is 10%.

**The rig over-stated the copy by 4x, and the reason generalises.** The rig ran
the copy on all 19 workers at once; a real search has the simulate stage at 25ms
of a ~330ms evaluation, so only one or two ranks are ever copying, and the copy
runs uncontended at its ~2.3ms serial cost instead of its 5.1ms all-workers
cost. An isolated rig that saturates every worker with one stage measures that
stage's *contended* cost, and is only the right number for a stage whose duty
cycle in the real run is near 1. (This is the opposite correction to the one
[the constant-columns section](#the-two-constant-columns-are-gone) records,
where the rig *under*-stated a win because the real run's contention is with
`wsclean` rather than with 18 copies of the stage under test. Both say the same
thing: match the rig's mix of work to the run's, or expect the number to be
wrong in whichever direction the mismatch points.)

What is worth having anyway is what the change deletes rather than what it
saves: a run at 60 evaluations/s no longer creates 66 files and ~840KB of
Measurement Set per evaluation on the disk (~50MB/s) only to unlink them, and
`makems.cfg`/`makems.log` leave the evaluation directory with them.

## `image_container_overhead` is mostly GNU `time`'s clock resolution

`image_binary_seconds` comes from GNU `time -v`'s own "Elapsed (wall clock)
time", which prints centiseconds. On a ~270ms `wsclean` call that is a 10ms
quantum, and `image_container_overhead` - the profiler's
`image_container_seconds - image_binary_seconds` - is the residue. Over the
6133 evaluations of a default search its distribution is

| min | p10 | p25 | median | p75 | p90 | p99 |
|---|---|---|---|---|---|---|
| 1.07 ms | 3.06 | 5.12 | 8.00 | 10.63 | 13.13 | 20.02 ms |

- a 10ms-wide spread sitting on a ~1ms floor, which is the real cost: the
`sh` fork and the `/usr/bin/time` exec inside the already-running sidecar.
There is no `docker exec` in it (one long-lived `sh` per rank serves the whole
run, see `sidecar_shell()`), and there is nothing here to reclaim. Averaged
over thousands of evaluations the quantisation is a constant bias, so an A/B on
the `image_binary` column is sound - which is what
[the AVX2 section](#the-avx2-build-stopped-being-opt-in) relied on - but the
~7ms "overhead" line in the per-stage table is a rounding artefact, and the
`wsclean` binary is correspondingly ~5ms dearer than that table says.
