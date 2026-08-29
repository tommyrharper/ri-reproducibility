# FFTW spends 6.3 ms an evaluation planning transforms it already knows

**A `wsclean` process asks FFTW for 63 transform plans before it finishes an
evaluation and keeps none of them. Building a plan for a size FFTW has not seen
in this process costs 0.1-0.8 ms; rebuilding one it has costs 0.03-0.045 ms;
together that is 6.3 ms of a ~56 ms serial process. The once-per-size half is
process-global state, so `wsclean-zygote`'s parent now builds it at startup and
every forked child inherits it: -6.9% on the `wsclean` binary over 960
interleaved tmpfs replay pairs against a -0.2% null, with 400 output FITS data
blocks byte-identical.**

Host: the same 20-thread i5-13500 as every other measurement in `docs/`, at the
65 W package limit [the power-limit doc](nested-sampling-power-limit.md)
describes. 29 August 2026.

## Where it was hiding

[The phase profile](nested-sampling-phase-profile.md) buckets an evaluation by
the gap between one `-log-time` line and the next, and the deconvolution rows
have always looked too expensive for what they do. This parameter space's source
is a delta function at the phase centre, so a minor loop is 16 subtractions of
10% of one pixel - and the gap between radler's `Performed N iterations in
total` and its `Stopped on peak` reads 0.85-1.5 ms, once per major cycle.

Reading the source explains the shape but not the size. That gap is
`SubMinorLoop::CorrectResidualDirty` (radler
`cpp/algorithms/subminor_loop.cc:195`), which convolves the model with the PSF
by FFT so the residual is exact rather than Clark-approximate. The convolution
is 142 x 142 - `even(ceil(1.1 x 128))` from `GenericClean`'s
`convolution_padding_` - and `schaapcommon::math::Convolve` creates **four
`fftwf` plans and destroys them again on every call**
(`src/math/convolution.cc:127`). `Resampler` does the same with two 2-D plans in
its constructor (`src/math/resampler.cc:25`), and WSClean constructs one per
gridding and per degridding pass (`wgridder/wgriddingmsgridder.cpp:553,593`).

## Counting them

The repo's usual trick - an `LD_PRELOAD` shim over the entry points, no rebuild
and no profiler - prices the whole class. Wrap `fftwf_plan_dft_r2c_1d`,
`fftwf_plan_dft_c2r_1d`, `fftwf_plan_dft_1d`, `fftwf_plan_dft_r2c_2d` and
`fftwf_plan_dft_c2r_2d`, time each call, and key the totals on (kind, size).
One evaluation, serial, on a tmpfs Measurement Set:

| plan | size | count | total ms | first ms | mean of the rest ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| r2c 2-D | 108 | 9 | 0.959 | 0.625 | 0.042 |
| c2r 2-D | 108 | 7 | 0.579 | 0.363 | 0.036 |
| r2c 2-D | 156 | 7 | 0.685 | 0.427 | 0.043 |
| c2r 2-D | 156 | 9 | 0.768 | 0.434 | 0.042 |
| r2c 1-D | 142 | 7 | 1.016 | 0.837 | 0.030 |
| c2c 1-D forward | 142 | 7 | 0.713 | 0.521 | 0.032 |
| c2c 1-D backward | 142 | 7 | 0.292 | 0.112 | 0.030 |
| c2r 1-D | 142 | 7 | 0.281 | 0.142 | 0.023 |
| the four 1-D at 128 | 128 | 1 each | 0.987 | - | - |
| **total** | | **63** | **6.26** | **4.45** | **1.81** |

Two things fall out. Planning is **6.3 ms of a ~56 ms process, 11%**, and it is
all overhead: FFTW is being asked for the same twelve plans over and over.
And **4.45 ms of it is the first build of each (kind, size)** - the pattern
tables FFTW keeps in process-global memory for a size it has met before.

The sizes are fixed for this search:

| size | what it is |
| ---: | --- |
| 108 | the gridder's chosen inversion size - `using optimal: 108 x 108` in **all 6641** logs of a `--nlive 25` run, because `-scale` is derived from each observation's own maximum baseline |
| 128 | the image, `-size 128 128` (`DEFAULT_IMAGE_DIM`) |
| 142 | radler's deconvolution convolution, `even(ceil(1.1 x 128))` |
| 156 | the padded image |

## The fix is one loop in the zygote parent

FFTW's per-size state lives in the process, and
[the fork server](nested-sampling-wsclean-zygote.md) already exists to pay
process-global costs once per rank. `WarmFftwPlanner()` in
`docker/wsclean/src/zygote.cpp` builds and immediately destroys the twelve plans
before the parent accepts its first request; destroying a plan does not discard
what FFTW learned making it, so every forked child starts warm. It costs the
rank **9.6 ms, once**, against the minutes-to-hours a rank lives.

Re-running the counting shim with the same warm-up done in the process's own
constructor:

| | total planning |
| --- | ---: |
| cold | 6.26 ms |
| warmed | 2.33 ms |

**-3.9 ms an evaluation**, and what is left is the 1.81 ms of repeat builds
(see "What is left" below).

Nothing about this can change a result: FFTW plans with `FFTW_ESTIMATE` are
chosen by a fixed heuristic rather than by measurement, so a warm planner hands
back the same plan a cold one would have. A size that stops being used costs
this warm-up and nothing else; a size that starts being used is simply not
warmed. That is why the list is hard-coded with a pointer beside
`DEFAULT_IMAGE_DIM` rather than derived - getting it wrong loses the speedup and
cannot lose an image.

## Measured: the binary

Both zygote binaries `docker cp`ed out of their images and run from one
container against a shared `casacore` (the technique
[the shared-MS-open doc](nested-sampling-shared-ms-open.md) introduced), over 80
Measurement Sets from a real search on `/dev/shm`, each replayed with **its own**
`-scale` (a fixed one is the trap [that doc](nested-sampling-shared-ms-open.md)
records), alternating which arm goes first:

| arm | n | median | mean |
| --- | ---: | ---: | ---: |
| base | 960 | 52.02 ms | 56.63 ms |
| warmed | 960 | 48.14 ms | 52.81 ms |
| **paired median ratio** | | **0.9292** | |

and the same rig with the base binary in both slots as the null:

| arm | n | median | mean |
| --- | ---: | ---: | ---: |
| base (slot A) | 960 | 52.19 ms | 56.72 ms |
| base (slot B) | 960 | 52.11 ms | 56.58 ms |
| **paired median ratio** | | **0.9976** | |

**-6.9% on the `wsclean` binary against a -0.2% null.**

## Measured: two simultaneous searches

Ratios out of two searches run at the same instant, 10 ranks each, one arm on
each image, `--nlive 40 --num-repeats 5`, killed together after 240 s. The arms
do not sample the same points, so each run's `image_binary_seconds` is bucketed
into twelve equal-count bins of `observation_minutes x channel_count` and the
statistic is the median of the per-bin median ratios - the normalisation
[the shared-MS-open doc](nested-sampling-shared-ms-open.md) works through.
`simulate_seconds` is untouched by this change and is carried as the null:

| pair | seeds | evaluations base / warmed | wsclean binary | simulate (null) | corrected |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 101 / 202 | 4049 / 3912 | 0.9661 | 0.9990 | **0.9671** |
| 2 (swapped) | 101 / 202 | 3769 / 3930 | 0.9316 | 0.9909 | **0.9401** |
| 3 | 707 / 808 | 4168 / 4821 | 0.9380 | 0.9993 | **0.9387** |
| 4 (swapped) | 707 / 808 | 3908 / 3968 | 0.9543 | 0.9975 | **0.9567** |

**Median -5.0% on the `wsclean` binary** (range -3.3% to -6.1%), with every
null inside 0.9% of 1. The binary is 84-85% of an evaluation
([the phase profile](nested-sampling-phase-profile.md)), so that is ~-4.2% an
evaluation, ~+4.4% evaluations per second.

Two things about that number. It is smaller than the replay's -6.9%, which is
the shape [the shared-MS-open doc](nested-sampling-shared-ms-open.md) warned
about running the other way: plan building is branchy, allocating,
lock-taking work, and this host's ~2.2x concurrency inflation does not fall on
it as heavily as it falls on the FFT arithmetic around it. And four pairs were
needed rather than two: the per-pair ratios run 0.937-0.967 and either half
alone would have reported a different figure.

## Bit-identical images

One replay pass over the same 80 Measurement Sets keeping all five FITS outputs
per evaluation, compared on the SHA-256 of the data block after the header (the
header carries a `DATE` stamp): **400 of 400 identical.**

## What is left

Nothing - the 1.81 ms of repeat plan builds named here has since been taken.
`docker/wsclean/patches/0004` is the `schaapcommon` plan cache this section
called for, in exactly the shape `0003` uses for casacore tables and with the
same static-destruction-order care: 64 plan builds an evaluation down to 12,
worth -2.7% on the binary in an interleaved replay and -3.2% over four
simultaneous swapped searches, with bit-identical images. See
[the gridder-floor doc](nested-sampling-gridder-floor.md).

Not worth pursuing on this page:

- **Making the deconvolution convolution a nicer FFT size.** 142 is `2 x 71`, and
  a prime factor of 71 puts FFTW on its Rader path: the same convolution costs
  0.488 ms at 142 and 0.139 ms at 144. But `convolution_padding_` is what decides
  how much of the wrap-around a Clark residual correction keeps, so changing it
  changes the residual, and it is 0.35 ms x 6.4 major cycles = 2.2 ms an
  evaluation for a result that is no longer the one every archived run was
  scored under.
- **`-no-fast-subminor`.** It replaces the FFT correction with direct
  full-image subtraction per minor iteration. It is also not result-preserving:
  the sub-minor loop searches peaks over a pre-selected pixel set and the plain
  loop over the whole image, so the component lists can diverge.

## Reproducing

```sh
# 1. a corpus, on the filesystem a real evaluation uses
./ri search wsclean --nlive 20 --num-repeats 3 --max-ndead 20 \
    --keep-measurement-sets --output-dir results/nested-sampling/fftw-corpus

# 2. count the plans (the shim wraps fftwf_plan_dft_*; see the table above)
#    LD_PRELOAD=<shim>.so wsclean ... <ms>

# 3. two arms, one container
docker cp $(docker create <image>):/opt/wsclean/bin/wsclean-zygote <arm>/
docker cp $(docker create <image>):/opt/wsclean/lib <arm>/lib
# drive both over the corpus alternating which goes first, and read the
# zygote's own reply field for the child's wall clock
```
