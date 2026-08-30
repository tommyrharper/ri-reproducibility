# The gridder's floor, and the last of the FFTW planner

**Two thirds of a WSClean evaluation is now ducc0's gridding and degridding
passes, and ducc0 will tell you where that time goes if you ask it: 57%
gridding proper, 23% FFT, 11% corrections, 3% index. There is no slack in it -
every pass builds a six-plane w-cube for a w-range that spans 0.7% of one
plane, because `nplanes = (wmax-wmin)/dw + supp` has a kernel-support floor,
and taking it away is worth -29% on the `wsclean` binary but changes the
restored image by 2.3e-5 of its peak.** What *was* left is FFTW's planner:
`docker/wsclean/patches/0004` caches the transform plans `schaapcommon`
rebuilds on every call, taking a run's 64 plan builds per evaluation to 12,
worth -2.7% on the binary in an interleaved tmpfs replay and -3.2% over four
simultaneous swapped searches, with 520 of 520 output FITS data blocks
byte-identical.

Rig: 20-thread i5-13500, 65 W package limit; [details](nested-sampling-power-limit.md).

For the current phase breakdown and cost model, see the
[run-scaling guide](nested-sampling-run-scaling.md).

## Asking ducc0 where its own time goes

It is a one-line answer. `WGriddingMSGridder::MakeGridder` in
`wgridder/wgriddingmsgridder.cpp` passes a hard-coded `0` as ducc0's
`verbosity`; ducc0 carries a `TimerHierarchy` through the whole gridding call
and dumps it at `verbosity > 0`. Build an image with that `0` changed to a `2`
and every pass prints its own tree - no rig, no profiler, no rebuild of
anything but WSClean:

```sh
# from a git-archive of HEAD, with a one-hunk patch dropped in
sed -i 's/accuracy_, 0, use_tuned_wgridder_/accuracy_, 2, use_tuned_wgridder_/' \
    wgridder/wgriddingmsgridder.cpp
```

Over the sixteen passes of one median evaluation (702 rows x 7 channels =
4914 visibilities, `dirty=(92x92)`, `grid=(154x154x6)`, `supp=6`, `eps=1e-4`),
serial and warm:

| gridding, 1.21 ms | | degridding, 1.03 ms | |
| --- | ---: | --- | ---: |
| gridding proper | 57.2% | degridding proper | 51.9% |
| FFT | 22.5% | FFT | 25.3% |
| wscreen + grid correction | 6.9% | wscreen + grid correction | 7.0% |
| global corrections | 5.5% | global corrections | 6.4% |
| building index | 2.7% | building index | 3.1% |
| `<unaccounted>` | 1.8% | `<unaccounted>` | 2.2% |
| Initial scan | 1.0% | Initial scan | 1.2% |
| zeroing grid | 1.0% | zeroing grid | 1.3% |
| allocating grid, grid regions, parameter calculation | 1.0% | the same | 1.1% |

Four fifths of a pass is the convolution and the transform. The bookkeeping
this repo has spent five iterations removing elsewhere - index building,
allocation, parameter calculation - is 4% of it. **The 58% is arithmetic.**

## Why every pass is a six-plane cube

The one line in that report that does not look like arithmetic is
`grid=(154x154x6)`: six w-planes, so six FFTs and a convolution kernel that is
`supp` times more expensive than a 2D one. But the same log says

```
w=[19.1688; 24884.2], min(n-1)=-1.64759e-07, dw=3.62591e+06, (wmax-wmin)/dw=0.00685758
```

- the whole w-range of the observation spans **0.7% of one w-plane**. The six
planes are not the data; they are `ducc0`'s floor:

```cpp
// external/schaapcommon/external/ducc0/wgridder/wgridder.h
double dw = 0.5/ofactor/max(abs(nm1max+nshift), abs(nm1min+nshift));
size_t nplanes = size_t((wmax_d-wmin_d)/dw+supp);
fftcost *= nplanes;
gridcost *= supp;
```

`supp` is the gridding kernel's support (6 at `eps=1e-4`), and the w-kernel
needs that many planes to exist at all. This is not one evaluation being
unusual. Over all 104 Measurement Sets of a corpus taken from a real search,
**every one** reports `grid=(154x154x6), supp=6`, and `(wmax-wmin)/dw` runs
from 6.6e-5 to 5.4e-2 - never as much as a twentieth of a single plane. The
parameter space images a 128 x 128 field at a scale derived from each
observation's own maximum baseline, so the field of view is always tiny and
the w-term always negligible; the cost of carrying it is not.

### What turning it off is worth, and what it costs

`WGridder<T>` passes `do_wgridding=true` to both `ms2dirty` and `dirty2ms`.
Flipping both to `false`, over 312 interleaved tmpfs replay pairs on the same
104 Measurement Sets:

| | median ms |
| --- | ---: |
| HEAD | 70.30 |
| `do_wgridding=false` | 48.82 |
| **ratio** | **0.694** (paired median 0.714) |

**-29% on the `wsclean` binary, i.e. about a quarter off every evaluation.**
That is by a wide margin the largest single number this project has measured.

It is not takeable as a default. The restored images are not bit-identical:
over the same 104 evaluations the largest pixel difference is 2.3e-5 of the
image peak on the median and 1.2e-4 at worst, and this search scores
`log10_dynamic_range` out to 1e6. The physics agrees - the maximum w phase the
approximation drops is `2*pi*wmax*max|n-1|` = 0.026 rad, which is 260x the
`eps=1e-4` the gridder is otherwise held to, so the two arms are simply two
different accuracies rather than two implementations of one.

It is recorded here because it is a *decision*, not a dead end: anyone willing
to image this parameter space at ~1e-4 relative accuracy instead of ~1e-6 buys
a quarter of the run back, and nobody could have made that trade before
without this number. Same family as `-wgridder-accuracy 1e-2`
([the evaluation-budget doc](nested-sampling-evaluation-budget.md), +13.8%),
but a much better rate: 2x the accuracy loss for 2x the speedup.

### It cannot be taken on part of a run either

The obvious way to have both is a per-evaluation rule: skip w-gridding on the
evaluations where ducc0's own accuracy standard says it is free, and pay for it
on the rest. The criterion is the phase above, `2*pi*wmax*max|n-1|`, against the
`eps=1e-4` the gridder is held to; everything it needs is already in every
evaluation's log (`w=[min:max] lambdas`, `using optimal: N x N`) and its
`-scale`.

Over all 5962 evaluations of a `--nlive 600` search, that phase runs from
1.5e-4 at the smallest to 1.4e-1 at the largest, with a median of 1.2e-3.
**Not one is below 1e-4**, and the floor is not close to it. The relation is
structural rather than an accident of the seed - `-scale` is set from each
observation's own `max_proj_baseline_lambda` at `DEFAULT_SUPER_RESOLUTION`, so
the phase reduces to roughly `2*pi*36*(wmax/maxuvw)/maxuvw` and only the
longest-baseline corner approaches the epsilon, which is also where
`wmax/maxuvw` is largest. See
[the run-scaling page](nested-sampling-run-scaling.md) for the percentile table
and how to recompute it from any archived run's logs.

So the -29% stays exactly what it is above: a decision about accuracy, with no
lossless subset of it available.

## 0004: the plans schaapcommon keeps rebuilding

[The FFTW-planner doc](nested-sampling-fftw-planner.md) closed with 1.81 ms an
evaluation of repeat plan builds and named the fix. This is it.

`schaapcommon::math::Convolve()` builds four 1-D `fftwf` plans and destroys
them again at the bottom of every call, and `Resampler` builds two 2-D ones in
its constructor and destroys them in its destructor. Counted with an
`LD_PRELOAD` shim over the five `fftwf_plan_*` entry points and
`fftwf_destroy_plan`, one evaluation:

| | plan builds | destroys |
| --- | ---: | ---: |
| HEAD before 0004 | 64 | 64 |
| with 0004 | **12** | **0** |

The 52 that go away are FFTW being asked for a plan it has already built, at
~0.035 ms each of solver-table walking under the planner lock. The zygote's
`WarmFftwPlanner()` already pays the *first* build of each size in the fork
server's parent; 0004 removes the repeats inside each child.

The patch adds `CachedPlan1D` and `CachedPlan2D` to
`external/schaapcommon/src/math/convolution.cc`, each a `std::map` from
transform shape to plan behind a mutex, and points both call sites at them.
Two details are load-bearing, and they are the same two `0003` documents
([the shared-MS-open doc](nested-sampling-shared-ms-open.md)):

- **The plans are shareable because they are pure.** Both call sites build
  with `FFTW_ESTIMATE` over null (that is, default-aligned) pointers and
  execute with FFTW's new-array interface, so a plan carries nothing from the
  call that made it. `Resampler`'s two are built over `fftwf_malloc`ed buffers
  to keep upstream's alignment assumption, and the buffers are freed
  immediately.
- **The cache is deliberately leaked** (`static map<...>& cache = *new
  map<...>()`). FFTW's planner state is a static too, and destroying a plan
  after that has gone is undefined.

### What it is worth

**Interleaved replay**, 104 Measurement Sets on tmpfs, each with its own
`-scale`, both binaries `docker cp`ed out of their images and driven from one
container over `LD_LIBRARY_PATH`, 8 passes alternating which arm goes first:

| | pairs | median A | median B | median ratio | paired median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0004 against HEAD | 832 | 72.70 ms | 70.55 ms | **0.9704** | 0.9723 |
| null (HEAD against a byte copy of itself) | 832 | 72.98 ms | 72.77 ms | 0.9972 | 0.9993 |

**Four simultaneous swapped searches**, 10 ranks each,
`--nlive 100 --num-repeats 2`, ~3500-4000 evaluations an arm, comparing the
median `image_binary_seconds` in twelve equal-count buckets of
`observation_minutes x channel_count` (the size normalisation
[the shared-MS-open doc](nested-sampling-shared-ms-open.md) explains) and
carrying the untouched `simulate_seconds` as an in-run null:

| pair | slot A | slot B | binary B/A | simulate B/A | corrected | as 0004/HEAD |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | HEAD | 0004 | 0.9580 | 0.9975 | 0.9604 | 0.9604 |
| 2 | 0004 | HEAD | 1.0168 | 1.0080 | 1.0087 | 0.9914 |
| 3 | HEAD | 0004 | 0.9566 | 0.9993 | 0.9573 | 0.9573 |
| 4 | 0004 | HEAD | 1.0193 | 0.9937 | 1.0257 | 0.9749 |

Median **0.968**, i.e. **-3.2% on the `wsclean` binary**; averaging within each
swap direction first gives 0.959 and 0.983, mean 0.971. The binary is 84.9% of
an evaluation, so that is ~+2.7% evaluations per second. (End-to-end
evaluations/second is not readable at this run length - the runs are 70-80 s
and the ~6 s startup constant plus the region each arm happened to sample
swamp a 3% effect. The stage ratio is the measurement.)

**Bit-identity**: 520 of 520 output FITS data blocks byte-identical across the
whole 104-Measurement-Set corpus (5 images each).

## What is left

Nothing on this avenue. The 12 remaining plan builds are one per distinct
transform shape per process, which is the floor.

The ranking that comes out of the median table above, for whoever is next:

- **Gridding and degridding, 58%.** Arithmetic, priced above. Only two levers
  and both are accuracy or pass-count trades: `do_wgridding` (-29%, 2.3e-5),
  `-wgridder-accuracy` (+13.8% at 1e-2), `-mgain`
  ([the clean-loop doc](nested-sampling-clean-loop.md), +20% at 0.9).
- **The reorder, 5.4%.** Taken - the row-at-a-time casacore column reads are
  now blocked, worth -48% of the phase and -4.1% on the `wsclean` binary. See
  [the row-block doc](nested-sampling-row-blocks.md).
- **Process start to the imaging table, 4.9%,** and the **beam fit, 4.4%,**
  are both closed - see
  [the shared-MS-open doc](nested-sampling-shared-ms-open.md) and
  [the cost-model doc](nested-sampling-cost-model.md).

## Reproducing

```sh
# the median phase table
./ri search wsclean --nlive 60 --num-repeats 2 --mpi-procs 20 --max-ndead 100000 \
    --output-dir results/nested-sampling/phases
./ri profile results/nested-sampling/phases --phases   # means; medians need the
                                                       # phase_gaps() helper directly

# a corpus of real Measurement Sets to replay against
./ri search wsclean --nlive 25 --num-repeats 2 --mpi-procs 14 --max-ndead 30 \
    --keep-measurement-sets --output-dir results/nested-sampling/corpus
# each replay must use that evaluation's own image_pixel_size_arcsec - a fixed
# -scale sends the beam fit into its box-growing retry on 14% of the corpus

# ducc0's own timers
git archive HEAD | tar -x -C /tmp/tree && cd /tmp/tree
sed -i 's/accuracy_, 0, use_tuned/accuracy_, 2, use_tuned/' \
    /tmp/wsclean-clone/wgridder/wgriddingmsgridder.cpp   # then git diff into
docker build -f docker/wsclean/Dockerfile -t ri-reproducibility/wsclean:probe .
```

Comparing output FITS files between two arms must compare the **data blocks**,
not the files: the header records the `-name` path the run was given, so two
arms writing into different directories differ in every byte-compare even when
every pixel matches.
