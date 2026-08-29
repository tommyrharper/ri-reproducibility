# What one evaluation costs, and why a run slows down

**An evaluation costs a constant plus a rate times its visibility count, and
nested sampling walks into the corner of the parameter space with the most
visibilities - so a run's evaluations/second falls monotonically as it goes.
That is what "inconsistent throughput" is. It is not the machine, it is not
degradation, and `./ri profile <run> --over-time` shows it on any run.**

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65 W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## The measurement

One search, `--mpi-procs 20 --nlive 200 --num-repeats 10 --max-ndead 2000`:
34368 evaluations in 5m 59s, 95.7 evaluations/second, and `./ri profile` accounts
for 98.4% of the worker-time budget. `./ri profile <run> --over-time` reads each
evaluation's own `-log-time` timeline and buckets it by wall clock:

| t (s) | evals/s | ms/eval | vis/eval |
| ---: | ---: | ---: | ---: |
| 0 | 118.5 | 128.8 | 8424 |
| 31 | 103.5 | 152.3 | 9477 |
| 65 | 101.4 | 155.2 | 9828 |
| 99 | 99.2 | 158.4 | 10530 |
| 133 | 99.8 | 158.5 | 10530 |
| 168 | 94.7 | 167.6 | 12285 |
| 205 | 95.7 | 164.9 | 12285 |
| 241 | 90.6 | 176.8 | 13338 |
| 279 | 87.8 | 181.4 | 14742 |
| 318 | 89.7 | 179.0 | 14742 |

Throughput falls ~14% over six minutes. The visibility count rises 75% over the
same six minutes, and `ms/eval` tracks it line for line. (The first bucket's
118.5 is the burst-clock trap the
[throughput doc](nested-sampling-throughput.md) documents - the first ~4 s of
any burst runs ~20% fast. Discard it.)

The parameter space is what drives it: `observation_minutes` (0.3-20) and
`channel_count` (1-8) set the number of visibilities, and the default
`total_rms_jy` objective is maximised at the long, wide end, so the sampler
compresses towards it. A run that goes deeper - a bigger `--nlive`, a bigger
`--max-ndead`, or one run to convergence - spends proportionally more of its
life in the expensive corner.

## The cost model

Least squares over every evaluation in that run, on the milliseconds between
the `wsclean` binary's first and last log line:

```
logged_ms = 100.4 + 5.64 us x visibilities        (n = 34368, 19 workers)
```

Refitted on the current tree it reads `70.7 + 4.58 us x visibilities` - patches
0003-0005, the zygote's cfitsio and casacore warm-up and `-data-column DATA`
have taken 30% off the constant and 19% off the rate since. The shape below is
unchanged; the numbers in it are the ones this run had. See
[the run-scaling page](nested-sampling-run-scaling.md).

The deciles it comes from, median logged ms at median visibility count:

| vis | 2457 | 4914 | 7020 | 8424 | 10530 | 12285 | 14742 | 17199 | 19656 | 24570 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 108 | 123 | 135 | 145 | 155 | 167 | 177 | 191 | 204 | 225 |

So at the run's median (~12000 visibilities) **60% of an evaluation is
independent of how much data it has** and 40% is proportional to it - and the
proportional share grows as the run goes deeper (32% at the 8424 visibilities of
the first bucket, 45% at the 14742 of the last).

Serially, on a 7-worker run of the same tree, the same fit reads
`50.9 + 1.83 us x visibilities`. Both parts inflate at production concurrency
but the rate inflates more (3.1x against 2.0x), which is what the
[throughput doc](nested-sampling-throughput.md) predicts: the arithmetic is
clock-bound and the all-core clock is what the 65 W limit takes away. Do not
read the serial intercept as a production one.

Per gridding pass, from the same serial run bucketed by visibility count:

| | fixed | per visibility |
| --- | ---: | ---: |
| a gridding pass (8.5 per evaluation) | 0.72 ms | 115 ns |
| a degridding pass (6.5 per evaluation) | ~0.6 ms | ~95 ns |

115 ns is not slack: ducc0's w-gridder at `-wgridder-accuracy 1e-4` touches
~8x8x8 grid points per visibility, so that is ~35 GFLOP/s on one thread. The
per-visibility half of an evaluation is arithmetic running at roughly the width
of the machine.

## What that means for where to look next

Five iterations of profiling have been spent on the *fixed* half - the parent
Measurement Set opens, the process start, the subtables, the reorder. Those all
still pay, but their share shrinks as a run goes deeper, and a much bigger run
is exactly the case this repo is aiming at. The levers that scale with the run
are the ones that change **how many passes** an evaluation makes over its
visibilities, because every pass pays the full per-visibility rate:

- `-mgain` is the only one of those with a measured number
  ([the clean-loop doc](nested-sampling-clean-loop.md)): 0.9 removes ~1.8 major
  cycles, i.e. ~3.6 of the ~15 passes, and is worth +20% evaluations/second. It
  is deliberately left at 0.8 for the reasons that page gives.
- Anything that lowers the per-visibility rate is inside ducc0 and changes the
  image. `-wgridder-accuracy 1e-2` is +13.8% and is refused because
  `log10_dynamic_range` reaches 1e6
  ([the evaluation-budget doc](nested-sampling-evaluation-budget.md)).

## What shipped: `-data-column DATA`

`Settings::determineDataColumn` opens the whole parent Measurement Set - the
first `casacore::MeasurementSet` the process constructs, so it also pays
casacore's process-global lazy initialisation - purely to ask whether there is a
`CORRECTED_DATA` column. There never is: the simulator writes `DATA` and nothing
else, and `self_check_dropped_subtables()` now asserts both halves of that.
Naming the column skips the open.

Interleaved replay, 200 Measurement Sets on tmpfs, four passes, three arms
(baseline, a duplicate of the baseline as the null, `-data-column DATA`), all
2400 commands shuffled into one `xargs -P 19` list:

| arm | mean ms | median ms | median ratio | paired median ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline | 164.56 | 160.55 | 1.0000 | 1.0000 |
| null (same command) | 166.94 | 160.46 | 0.9994 | 0.9989 |
| `-data-column DATA` | 162.09 | 158.14 | 0.9850 | 0.9898 |

The null pair puts this rig at 0.1% on the median, so this is **-1.0% to -1.5%
on the `wsclean` binary**, ~2.4 ms of a 160 ms call. All 1000 output FITS data
blocks (image, dirty, residual, psf, model over 200 Measurement Sets) are
byte-identical between the two arms.

Note the mean column: the null pair reads +1.5% on the mean and -0.1% on the
median over the same 800 pairs. An evaluation's cost distribution has a long
tail, so take the median - or the paired median ratio - and never the mean.

## Three avenues closed

### `-gridder tuned-wgridder` does not run

ducc0's tuning entry points (`ms2dirty_tuning` / `dirty2ms_tuning`) pick kernel
parameters for the problem shape rather than taking WSClean's fixed
`sigma_min`/`sigma_max`. On this parameter space they abort on all 120 of 120
Measurement Sets tried:

```
external/schaapcommon/external/ducc0/math/gridding_kernel.h: 448 ... Assertion failure
no appropriate kernel found
```

### `-gridder wtowers` is not built

`WSClean not built with w-towers support`, 120 of 120, exit 255. It would need a
`WSCLEAN_GIT_TAG` build with the w-towers dependency, and it is a different
gridder, so it could not be result-preserving anyway.

### The beam fit's thrown-away first fit is structural

[The phase-profile doc](nested-sampling-phase-profile.md) attributes 6.7% of an
evaluation to fitting the Gaussian beam and says every evaluation pays a
thrown-away first fit. The exact mechanism, from
`schaapcommon::fitters::Fit2DGaussianCentred`:

```cpp
preferred_size = ceil(beam_estimate * box_scale_factor);   // even, -beam-fitting-size = 10
result = Fit2DGaussianCentredInBox(..., beam_estimate, box, box);
box_was_large_enough = result.major * box_scale_factor * 0.8 < box;
if (!box_was_large_enough) {
  preferred_size = ceil(result.major * box_scale_factor);
  beam_estimate  = max(result.major, beam_estimate);       // fit 2 is seeded by fit 1
}
```

Substituting `box = beam_estimate * box_scale_factor`, the retry condition is
`fitted_major > 1.25 x beam_estimate` - **independent of
`-beam-fitting-size`**. In this parameter space the fitted beam is consistently
~1.75x WSClean's theoretical estimate, so no value of `-beam-fitting-size`
avoids the second fit; it only scales both boxes. And the second fit is seeded
with the first fit's major axis, so the first cannot be skipped even if the box
were known. The avenue is closed at the flag level; it is only movable upstream.

## Reproducing it

```sh
./ri search wsclean --mpi-procs 20 --nlive 200 --num-repeats 10 --max-ndead 2000
./ri profile <run>                  # the stage budget
./ri profile <run> --over-time      # throughput against wall clock
./ri profile <run> --phases         # inside the binary
```

`--over-time` needs a run made since `-log-time` became a default; on an older
run it says so rather than printing an empty table. `--buckets N` changes the
resolution. The visibility count it prints is WSClean's own
`Gridded visibility count` line, so it is what the gridder actually saw.

The cost-model fit is a least squares of each log's (first stamp, last stamp)
span against that count, over `evaluation_timeline()` in
`scripts/profile-nested-sampling-run.py`. The replay rig - one line per
(Measurement Set, arm, pass), shuffled, 19 at a time, each timed by its own
`date +%s%N`, with a duplicated arm as the null pair - is the shape
[the clean-loop doc](nested-sampling-clean-loop.md) describes, against a corpus
kept by `./ri search wsclean --keep-measurement-sets` and copied onto
`/dev/shm`.

One thing that made the whole timeline readable: `wsclean -verbose` adds
`Logger::Debug` lines, and `-log-time` stamps those too, so a single verbose run
is a finer-grained free timeline than the default one `--phases` reads - it is
what shows that the pre-imaging path is three separate parent-Measurement-Set
opens rather than one block of "metadata".
