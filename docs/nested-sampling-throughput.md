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
evaluation; `total_wall_seconds x mpi_procs` is the worker-time budget the run
had. The ratio is rank utilisation, and `./ri profile <run>` prints its
complement as "unaccounted (PolyChord sampling + idle)":

```bash
python3 -c 'import json,sys; j=json.load(open(sys.argv[1]+"/summary.json")); p=j["profiling"]
print("%.0f%% utilised" % (100*p["accounted_worker_seconds"]/(p["total_wall_seconds"]*p["mpi_procs"])))' \
  results/nested-sampling/<run>
```

PolyChord's own sampling is not in that remainder in any meaningful quantity -
it is a Cholesky decomposition and a live-point insertion per iteration
against seconds to tens of seconds of imaging. What the remainder is, almost
entirely, is ranks with nothing to do.

## The finding: the ranks were idle a third to a half of the run

Measured across the runs on disk before any of this changed:

| Run | ranks | nlive | num_repeats | wall | utilisation |
|---|---:|---:|---:|---:|---:|
| `r2d2-vlaa-20260827T205418Z` | 16 | 50 | 10 | 14.0h | 54% |
| `wsclean-vlaa-20260827T190109Z` | 20 | 50 | 10 | 501s | 45% |
| `wsclean-vlaa-20260827T201926Z` | 20 | 50 | 10 | 290s | 67% |

Two things to read off that. Half the machine was idle. And the same settings
gave 45% and 67% on two consecutive runs - the throughput was not just low, it
was not repeatable, which is what makes a run's remaining time impossible to
predict.

Reconstructing the 14-hour R2D2 run's timeline shows the shape of it. Each
evaluation directory's `metrics.json` mtime is when that evaluation finished,
and its `timing` block is how long it took, so the two give an interval per
evaluation; counting overlapping intervals gives how many ranks were busy at
any moment. Over that run:

```
concurrency:  1 rank busy 17% of the time,  2 busy 11%,  15 busy 37%
mean 8.7 of 16 ranks
```

Bimodal, not a uniform 54%: the ranks were either all working or nearly all
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
| 4242 | synchronous | 293.1s | 64% | 435 | 0.674 | 0.024636 ± 0.000024 |
| 4242 | **asynchronous** | **238.8s** | **92%** | 497 | **0.480** | 0.024620 ± 0.000023 |
| 7 | synchronous | 316.6s | 64% | 438 | 0.723 | 0.024625 ± 0.000024 |
| 7 | **asynchronous** | **243.7s** | **91%** | 502 | **0.486** | 0.024625 ± 0.000024 |
| 99 | synchronous | 293.2s | 67% | 440 | 0.666 | 0.024697 ± 0.000030 |
| 99 | **asynchronous** | **245.7s** | **91%** | 502 | **0.489** | 0.024709 ± 0.000031 |

Three pairs, no exceptions: **27-33% less wall clock per dead point**, rank
utilisation 64-67% -> 91-92%. The asynchronous runs went *further* as well as
faster - each ran to the same termination criterion and got ~60 more dead
points out of it, so the raw wall-clock column understates the gain.

Two things beyond the headline are worth noting.

The likelihood calls per dead point barely moved (19.0 synchronous, 19.1
asynchronous), so the extra points are not bought by sampling more cheaply -
asynchronous mode is not doing less work per point, it is doing the same work
on more cores.

And the utilisation is *steady*: 91, 91, 92 against 64, 64, 67, and against
45% and 67% on two historical 20-rank runs at the same settings. The
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
