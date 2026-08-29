# The simulate stage, and what a median phase table says to do next

**The MeqTrees stage was 8.4% of a run's worker time and a fifth of it was work
nobody wanted: a Measurement Set column read back only for its shape, a second
open of the same table only for its correlation count, six subtables copied out
of the skeleton cache only to be deleted again, and three whole copies of DATA
allocated to add noise to it. Removing all four is -20% on the stage at
production concurrency (15.1 ms -> 12.1 ms an evaluation), with a bit-identical
Measurement Set.** The same page records the median-based phase table that
picked the target and the two avenues it closes.

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65 W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## Read the phase table on medians, not means

`./ri profile <run> --phases` reports the mean of each phase bucket, which is
the right thing for "where did the run's seconds go". It is the wrong thing for
"what should I work on", because several buckets here are heavy-tailed and their
means are set by the tail. The same 3168-evaluation search
(`--nlive 50 --num-repeats 5 --max-ndead 200 --mpi-procs 20`), bucketed the way
`--phases` buckets it but summarised on the per-evaluation median:

| median ms | mean ms | share | phase |
| ---: | ---: | ---: | --- |
| 40.25 | 43.46 | 29.1% | `Gridding N rows...` -> `Gridded visibility count` (8.45 passes) |
| 25.13 | 27.72 | 18.2% | `Predicting N rows...` -> `Writing...` (6.45 passes) |
| 11.06 | 11.05 | 8.0% | deconvolution, minor-loop threshold |
| **8.22** | 8.64 | 6.0% | *process start* -> `=== IMAGING TABLE ===` |
| 6.85 | 7.32 | 5.0% | the reorder |
| **6.25** | **10.56** | 4.5% | `Fitting beam...` -> `Writing psf image... DONE` |
| 4.80 | 4.79 | 3.5% | `Opening reordered part N` -> `Loading metadata` (predict side) |
| 4.36 | 4.79 | 3.2% | rendering and writing the restored image |
| **3.89** | 4.12 | 2.8% | end of the imaging table -> `WARNING: ... WEIGHT_SPECTRUM` |
| 3.49 | 3.61 | 2.5% | `Loading data in memory...` -> `Gridding N rows...` |
| 2.69 | 2.63 | 1.9% | `Gridded visibility count` -> `== Deconvolving (N) ==` |
| 2.51 | 2.57 | 1.8% | `Gridded visibility count` -> `Fitting beam...` (PSF normalisation) |
| 3.69 + 1.87 | | 4.0% | deconvolution, other stop reasons |

The beam fit is the row that changes: 10.56 ms on the mean, **6.25 ms on the
median**. [The phase-profile doc](nested-sampling-phase-profile.md) ranked it
the largest non-imaging item in the binary on the mean; on the median it is
smaller than the metadata block that runs before the first visibility is read
(8.22 + 3.89 = 12.1 ms, 8.8%), which is where the next wsclean-side iteration
should go - see "What is left" below.

## What the simulate stage was doing

Serial, warm, in the `meqtrees` image, 50 timed calls per row, on the shape a
median evaluation asks for (2808 rows x 8 channels x 4 correlations):

| step | ms | |
| --- | ---: | --- |
| `write_makems_config` | 0.40 | |
| `make_ms_skeleton` | 2.49 | of which `copytree` 1.56, `patch_spectral_window` 0.33 |
| `fill_point_source_visibilities` | 4.39 | see below |
| **total** | **7.30** | |

and inside `fill_point_source_visibilities`:

| call | ms | |
| --- | ---: | --- |
| noise draw and add | 2.23 | **removable in part** |
| `determine_corr_selection` | 0.43 | **removable** (its own open of the parent table) |
| `getcol("DATA")` | 0.38 | **removable** (read back, then overwritten) |
| open the parent table | 0.33 | |
| `SPECTRAL_WINDOW` open + `CHAN_FREQ` | 0.22 | |
| `rmtree` of six subtables | 0.16 | **removable** (never copy them) |
| close the parent table | 0.15 | |
| `putcol("DATA")` | 0.13 | |
| `data[:] = model`, `FLAG`, keywords, UVW, max baseline | 0.17 | |

Four items on that list are work with no consumer.

### The DATA column is read back only for its shape

Every evaluation in this parameter space has its source at the phase centre, so
`phase_centre_visibility()` writes the constant directly and MeqTrees is never
asked ([the throughput doc](nested-sampling-throughput.md) covers why). The old
code still read the whole DATA column out of the skeleton - 719 KB of zeros -
purely to get `n_rows, n_chan, n_corr` off `data.shape` before overwriting every
element of it. One row of the same column answers the same question for 0.05 ms,
and `nrows()` is free. The offset path still reads the column, because there the
MeqTrees predict has just written it.

### `determine_corr_selection()` opens the same table again

It exists to map the correlation count onto the `2x2`/`2`/`1` string
`meqtree-pipeliner.py` wants, and it opens the parent table on its own to read
one row of DATA. On the phase-centre path there is no pipeliner to hand the
string to, and the correlation count now comes off the shape probe above, so the
whole call goes.

### Six subtables were copied out of the cache only to be deleted

The simulator drops `FEED`, `FLAG_CMD`, `HISTORY`, `POINTING`, `PROCESSOR` and
`STATE` once the visibilities are written, because casacore attaches every
subtable on every open of the parent and WSClean opens it repeatedly
([the MS-open doc](nested-sampling-ms-open.md)). They were dropped *after* the
fill rather than left out of the cached skeleton because casacore refuses to
open an MS that is missing a required subtable, and the MeqTrees predict opens
it exactly that way.

That constraint only binds when there is a predict. `make_ms_skeleton()` now
takes `prune_unused`, and `simulate()` passes it whenever the source is at the
phase centre: those six directories are 27 of the skeleton's 66 files and 150 KB
of its 1.5 MB, and not copying them is 0.50 ms of the 1.56 ms `copytree`. The
delete afterwards stays unconditional - it already tolerated them being absent -
and it is what still covers the offset path.

`prune_unused` is an explicit argument rather than something read off `args`,
because two callers predict a source that is *at* the phase centre and would be
broken by inferring it: `warm_forest()`, which compiles the forest before the
first evaluation, and `self_check_phase_centre_predict()`, which is the guard
that the constant and the predict agree.

### The noise built three more copies of DATA

```python
noise = rng.normal(0.0, s, data.shape) + 1j * rng.normal(0.0, s, data.shape)
data = data + noise.astype(np.complex64)
```

is a complex128 array the size of DATA, a complex64 copy of it, and a third
array for the out-of-place add. Adding to the two float32 halves in place:

```python
data.real += rng.normal(0.0, s, data.shape).astype(np.float32)
data.imag += rng.normal(0.0, s, data.shape).astype(np.float32)
```

is **bit-identical** - `.astype(np.complex64)` rounded each component to float32
before the add, and so does this - and 2.38 ms -> 1.69 ms on the shape above.
The two draws have to stay two calls in that order: that, not the arithmetic, is
what fixes the stream for a given seed.

## What it is worth

Serial, three interleaved rounds of 100 timed `simulate()` calls each, alternating
between a `git archive HEAD` build of the `meqtrees` image and the new one:

| round | before | after |
| --- | ---: | ---: |
| 1 | 7.299 ms | 5.665 ms |
| 2 | 7.291 ms | 5.765 ms |
| 3 | 7.297 ms | 5.685 ms |

**-22%**, with no overlap between the two sets of three.

End to end, two swapped pairs of *simultaneous* searches - 10 ranks each on this
20-thread host, the only honest way to A/B here
([the wsclean-patches doc](nested-sampling-wsclean-patches.md) says why) - with
one arm on each image. The comparison is on the median of the per-evaluation
`timing.simulate_seconds` each run records, which is thousands of paired samples
of exactly the stage that changed; `image_binary_seconds` is untouched by this
change and is carried as the null:

| pair | simulate, base | simulate, new | ratio | wsclean binary (null) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15.09 ms | 12.12 ms | **0.803** | 1.031 |
| 2 (swapped) | 15.68 ms | 12.48 ms | **0.796** | 1.003 |

**-20% on the stage, 3.0 ms an evaluation.** The stage was 8.4% of a run's
worker time, so that is ~1.7% end to end - too small to read off
evaluations/second directly, which is why the paired stage median is the
measurement.

### Bit-identical Measurement Sets

Three cases - the two phase-centre shapes the self-checks use plus a
`(5", 3")` offset source that does run MeqTrees - built by both trees and
compared on the SHA-256 of every column (`DATA`, `UVW`, `FLAG`, `WEIGHT`,
`SIGMA`, `TIME`, `ANTENNA1`, `ANTENNA2`), the surviving keyword set, the
surviving subtable directories and the row count. All identical. The only
difference anywhere was the temporary path inside each run's `simulation.json`.

`self_check_dropped_subtables()` is the standing guard: it runs `simulate()` on
both predict paths and now also asserts DATA's shape against `nrows()` and that
the phase-centre case averages the 1 Jy source, which is what would fail if the
uninitialised array were ever the wrong shape or left unfilled.

## Two avenues closed on the way

- **There is no gridder warm-up to inherit from the zygote parent.** A single
  evaluation's log showed its first gridding pass at 9.2 ms against 5.1 ms for
  its last, which would have made ducc0's per-process lazy state worth
  pre-paying in the fork server the way
  [the zygote](nested-sampling-wsclean-zygote.md) pre-pays casacore's static
  initialisers. Over all 3168 evaluations the medians are 5.24, 4.65, 4.54,
  4.52, 4.50, 4.48 ms for passes 0-5: a 0.7 ms first-pass premium, 0.5% of an
  evaluation, and the single log was an outlier. Means and single samples both
  lie here; the per-pass median does not.
- **The simulate stage has no harness overhead left to find.** Its 6.8 ms of
  measured internals, times this host's ~2.2x concurrency inflation, is the
  whole 14.6 ms an evaluation records for it - so the FIFO round trip to the
  warm worker, the two log files `redirect_fds` opens, the scratch temporary
  directory and the closing move are together worth ~0. Anything further has to
  come out of the four items above, and now has.

## What is left

The stage is now 12.1 ms of a ~178 ms evaluation. Inside it, `copytree` (1.06 ms),
the noise draws themselves (1.69 ms serial, irreducible without changing the
stream) and four casacore opens are what is left, and none of them is worth an
iteration on its own.

The next target is in the `wsclean` binary, and the median table above names it:
**8.22 ms from process start to `=== IMAGING TABLE ===`, plus 3.89 ms from there
to the reorder - 8.8% of an evaluation before a single visibility is read.**
`wsclean -verbose` puts 3.1 ms of the first block (serial, on a small
Measurement Set) between `Using image size of ...` and `Total nr of channels
found in measurement sets`, and the source says why: `WSClean::RunClean()`
constructs a `casacore::MeasurementSet` over the parent in `getObservationInfo()`
(`main/wsclean.cpp:625`) and immediately constructs a second one over the same
path in `makeImagingTable()` (`main/wsclean.cpp:2003`). Each attaches all seven
surviving subtables; the first one in a process also pays casacore's
process-global lazy init. `-data-column DATA`
([the cost-model doc](nested-sampling-cost-model.md)) removed a third such open
at the same site and was worth -1.0% on the binary on its own, so a
`docker/wsclean/patches/` change that opens the parent once and shares it is the
natural next patch, in the same shape as the one that cached the antenna names.

## Reproducing

```sh
# the phase table on medians rather than means
./ri profile <run> --phases          # means, no rig
# the A/B: build the baseline image beside the working-tree one
git archive HEAD | tar -x -C /tmp/base && (cd /tmp/base && ./ri build meqtrees)
docker tag ri-reproducibility/meqtrees:kern-10 ri-reproducibility/meqtrees:base
./ri build meqtrees                  # retag the working tree's build as kern-10
# two simultaneous 10-rank searches, one image each, then swap and repeat
MEQTREES_IMAGE=ri-reproducibility/meqtrees:base ./ri search wsclean \
  --nlive 50 --num-repeats 5 --max-ndead 300 --mpi-procs 10 --output-dir results/nested-sampling/a &
./ri search wsclean \
  --nlive 50 --num-repeats 5 --max-ndead 300 --mpi-procs 10 --output-dir results/nested-sampling/b &
wait
# compare the median of timing.simulate_seconds; timing.image_binary_seconds is the null
```

`./ri build meqtrees` always tags `ri-reproducibility/meqtrees:kern-10`, so
build the baseline **first** and retag it, or the second build silently replaces
the arm you meant to keep.
