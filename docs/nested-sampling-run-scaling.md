# What a bigger run costs

**A ten-times-bigger `--nlive` costs nothing per evaluation: this tree runs at
126 evaluations/second and accounts for 95.4% of worker time at `--nlive 600`.
The floor is `70.7 ms + 4.58 us x visibilities`; the restoring-beam fit is its
largest fixed item at 8.7% of the `wsclean` binary, and dropping w-gridding is
closed by measurements across the parameter space.**

Rig: 20-thread i5-13500, 65 W package limit; [details](nested-sampling-power-limit.md). 29 August 2026, on `4b4698f`.

## Throughput does not care about `nlive`

Two searches, same seed, same `--max-ndead`, run back to back with the whole
host to themselves:

```
./ri search wsclean --nlive 60  --num-repeats 10 --max-ndead 400 --seed 4242
./ri search wsclean --nlive 600 --num-repeats 10 --max-ndead 400 --seed 4242
```

| | `--nlive 60` | `--nlive 600` |
| --- | ---: | ---: |
| evaluations | 8575 | 5962 |
| wall clock | 73.4 s | 47.3 s |
| **evaluations/second** | **116.8** | **126.0** |
| accounted worker-time | 94.4% | 95.4% |
| unaccounted (PolyChord + idle) | 5.6% | 4.6% |
| simulate | 12 ms/eval, 7.8% | 12 ms/eval, 8.2% |
| `wsclean` binary | 136 ms/eval, 84.4% | 128 ms/eval, 84.8% |
| metrics | 2 ms/eval, 0.9% | 2 ms/eval, 1.0% |

The bigger run is the *faster* one, and for the reason the
[cost model](nested-sampling-cost-model.md) gives: at a fixed `--max-ndead` a
small `nlive` compresses much further (400 dead points is 6.7 e-folds at
`nlive 60` and 0.67 at `nlive 600`), so it spends more of its life in the
long-observation, many-channel corner. `./ri profile --over-time` shows it as
`vis/eval`, which the `ms/eval` column tracks line for line in both runs.

So the objective's opening premise - "almost half of the time seems spent
idling or running PolyChord" - does not survive at any run size this host can
reach. It was true of the synchronous-MPI runs iteration 1 replaced, and it has
been 5% since. What "inconsistent throughput" is, is compression depth: a run's
evaluations/second is a function of how far into the prior it has got, not of
its `nlive`, not of the clock, and not of anything reclaimable.

## The refreshed cost model

Least squares over the per-visibility-count medians of every evaluation's own
`-log-time` timeline, on the current tree:

```
logged_ms = 70.7 + 4.58 us x visibilities        (n = 5962, --nlive 600)
logged_ms = 69.8 + 4.80 us x visibilities        (n = 8575, --nlive 60)
```

against `100.4 + 5.64 us` when
[the cost-model page](nested-sampling-cost-model.md) fitted it: patches 0003,
0004 and 0005, the zygote's cfitsio and casacore warm-up and `-data-column DATA`
have taken 30% off the constant and 19% off the rate since. At the median
visibility count (8775) that is 111 ms, of which **64% is independent of how
much data the evaluation has**.

Both fits agree to 5%, from runs whose visibility distributions barely overlap,
which is the check that the model is a model and not a summary of one run.

## The largest item left in the fixed half: the restoring-beam fit

`wsclean` has a `-no-fit-beam` that substitutes the theoretical beam for the
fitted one. Replaying 76 Measurement Sets kept by
`./ri search --keep-measurement-sets`, with the two arms **adjacent on each
set** so they share their slot on the host:

| | median total | paired ratio |
| --- | ---: | ---: |
| baseline | 52.50 ms | 1.0000 |
| `-no-fit-beam` | 48.12 ms | **0.9134** (q1 0.890, q3 0.940, 74/76 faster) |

So the fit and the smaller convolution that follows it are **8.7% of the
`wsclean` binary**, matching the 10.6 ms/eval the `./ri profile --phases` table
gives for `Fitting beam... -> Writing psf image... DONE` at production
concurrency (4.5 ms x the documented 2.2x concurrency inflation). It is the
largest single non-ducc0 item, and it is not takeable: the restored image is the
residual plus the model convolved with *that* beam, and every metric
`compute_image_metrics()` produces is read off the restored image.

The fitting box is not the lever either. `-beam-fitting-size 1` and `4` in the
same replay pass:

| | paired ratio vs `-beam-fitting-size 1` | fitted beam |
| --- | ---: | --- |
| `-beam-fitting-size 4` | 1.0089 | identical to the default on 68/76 |

A four-times-wider box costs 0.9%, which says the cost is not the residual
count. That is consistent with what
[the process warm-up page](nested-sampling-process-warm-up.md) found by
instrumenting the fitter directly (it is GSL's linear algebra and the divisions
in the Jacobian, not the `exp` calls), and with
[the cost model page](nested-sampling-cost-model.md)'s note that the retry
triggers on the fitted/theoretical ratio independently of the box.

**Read the `-beam-fitting-size` rows only against each other.** They were
replayed in a second pass, and both come out 4% faster than the baseline of the
first - the exact sequential-arm false positive
[the I/O placement page](nested-sampling-io-placement.md) warns about, appearing
here in a rig that had no A/B intent at all.

## Dropping w-gridding is taken, on the whole parameter space

[The gridder floor](nested-sampling-gridder-floor.md) prices
`do_wgridding=false` at **-29% on the `wsclean` binary** - by a wide margin the
biggest number this project has measured - on the strength of one corpus's
`w=[6.17:61289.5] lambdas` and a 2.3e-5 median image change. It is taken, as
`docker/wsclean/patches/0006`, and it is taken for every evaluation because
there is no lossless subset of it. The honest per-evaluation criterion is the
gridder's own: ignoring `w` costs at most a phase error of

```
2 pi x wmax x |n-1|max        |n-1|max = 1 - sqrt(1 - 2 lmax^2)
```

over the image, where `lmax` is the half-width of the *inversion* grid
(`Minimal inversion size: ... using optimal: N x N` in every log) at the
evaluation's own `-scale`. Below the gridder's `1e-4` epsilon, turning
w-gridding off is free by ducc0's own accuracy standard; above it, it is not.

Every one of the 5962 evaluations of the `--nlive 600` run, from its own logs:

| percentile | p0 | p10 | p25 | p50 | p75 | p90 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max ignored-`w` phase (rad) | 1.5e-4 | 2.1e-4 | 3.8e-4 | 1.2e-3 | 6.2e-3 | 8.9e-2 | 1.4e-1 |

**Zero of 5962 fall below 1e-4**, and the floor sits at 1.5e-4 rather than
anywhere near it. The relation is structural, not a coincidence of this seed:
`-scale` is set from `max_proj_baseline_lambda` at `DEFAULT_SUPER_RESOLUTION`,
so the phase reduces to `~2 pi x 36 x (wmax/maxuvw) / maxuvw` and only the
highest-frequency, longest-baseline corner of the space gets near the epsilon -
where `wmax/maxuvw` is largest too.

There is therefore no lossless per-evaluation rule that would let a patch skip
the six-plane w-cube on part of a run: `patches/0006` skips it on all of them,
which makes the -29% a science decision about the 2.3e-5 image change rather
than an optimisation, and it stays documented in the gridder-floor page as one.
Runs either side of that patch are not comparable.

## What actually limits a bigger run

Not the clock and, since [the disk footprint page](nested-sampling-disk-footprint.md),
not the disk. It is rank 0 at the end of the run. `load_evaluations_from_dir()`
reads every record back to build `summary.json`, and a record is 3.36 KB on
disk and **9.34 KB resident** (measured by holding 12600 of them).

At `~17 x nlive x num_repeats` evaluations, 126 evaluations/second, 89.5 KB an
evaluation on disk and 184 GB free here:

| `--nlive` | `--num-repeats` | evaluations | wall clock | disk | rank-0 RSS | `summary.json` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 10 | 34k | 4.5 min | 3.0 GB | 0.3 GB | 0.12 GB |
| 500 | 25 | 212k | 28 min | 19 GB | 2.0 GB | 0.75 GB |
| 1000 | 25 | 425k | 56 min | 38 GB | 4.0 GB | 1.5 GB |
| 1000 | 50 | 850k | 1 h 52 min | 76 GB | 7.9 GB | 3.0 GB |
| 2000 | 50 | 1.7M | 3 h 45 min | 152 GB | 15.9 GB | 6.0 GB |

`write_json_atomic()` already streams, so the peak is the record list itself and
not a serialised copy of it - see its docstring. The costs are one-off and
small in time (17.8 s to read 850k records back at the measured 20.9 us each,
14 s to write them), so what the table caps is *memory*, and only in the last
minute of the run. **The practical ceiling on this host is ~850k evaluations**,
`--nlive 1000 --num-repeats 50`: past that rank 0 wants more than 16 GB while
the sidecars are still up, and `./ri profile` needs the same again to read the
result back.

Two smaller notes for a run that size:

- `adopt_completed_evaluations()` runs on **every** rank, so a resume of an
  850k-evaluation run parses 850k records twenty times over - 18 s of CPU per
  rank, all twenty at once. Fine once; it is a reason not to resume a big run
  repeatedly.
- The last eighth of a bounded run reads ~25% slow in `--over-time` at the same
  `vis/eval` (89.5 evals/s against 120.7 at `--nlive 60`). That is the ranks
  draining as PolyChord terminates, a fixed cost, and it is why a short
  benchmark search under-reports its own throughput.

## Reproducing all of it

The two scaling runs are archived as `results/nested-sampling/i32-nl60` and
`i32-nl600`.

```bash
# the two scaling runs (about two minutes together)
./ri search wsclean --nlive 60  --num-repeats 10 --max-ndead 400 --seed 4242 \
  --output-dir results/nested-sampling/scale-nl60
./ri search wsclean --nlive 600 --num-repeats 10 --max-ndead 400 --seed 4242 \
  --output-dir results/nested-sampling/scale-nl600
./ri profile results/nested-sampling/scale-nl600
./ri profile results/nested-sampling/scale-nl600 --phases --top 12
./ri profile results/nested-sampling/scale-nl600 --over-time --buckets 8

# Measurement Sets to replay against
./ri search wsclean --nlive 20 --num-repeats 2 --max-ndead 20 --mpi-procs 4 \
  --keep-measurement-sets --output-dir results/nested-sampling/scale-ms
```

The replay loop is the one
[the shared-MS-open page](nested-sampling-shared-ms-open.md) documents, with the
two arms adjacent on each Measurement Set rather than one whole pass each; the
`-scale` for each set comes out of its own `metrics.json`
(`commands.wsclean`). Both the `w` table and the cost-model fit are computed
from `wsclean.stdout.log` alone - `w=[min:max] lambdas`, `using optimal: N x N`,
`Gridded visibility count: N` and the first and last timestamps - so any
archived run made since `-log-time` went in can be re-read the same way.
