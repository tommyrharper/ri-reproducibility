# The 5 ms a `wsclean` process spends before it looks at a visibility

**A forked `wsclean` child logs its version banner and then takes ~7 ms at
production concurrency to reach `=== IMAGING TABLE ===`. Instrumenting that
window shows it is two calls: cfitsio's one-time initialisation (0.47 ms) and
the first `casacore::MeasurementSet` construction in the process (the rest).
Both have a process-global half that a fork server's parent can pay once, so
`wsclean-zygote` now does: the phase falls from a median 7.15 ms to 4.78 ms in
all eight of eight swapped simultaneous search pairs, -1.9% on the `wsclean`
child over 230 interleaved replay pairs against a -0.26% null, with 230 of 230
output FITS data blocks byte-identical.**

Rig: 20-thread i5-13500, 65 W package limit; [details](nested-sampling-power-limit.md).

## Where the phase table said to look

At HEAD, `./ri profile <run> --phases` over a 2413-evaluation `--nlive 60`
search reads (76.7 ms of logged work an evaluation, 9 workers):

| ms/eval | share | phase |
| ---: | ---: | --- |
| 26.73 | 34.9% | gridding, 8.51 passes |
| 17.02 | 22.2% | degridding, 6.51 passes |
| 6.10 | 8.0% | the restoring-beam fit |
| **4.99** | **6.5%** | **banner -> `=== IMAGING TABLE ===`** |
| 7.7 | 10.0% | radler's minor loops |
| 2.5 | 3.3% | opening reordered parts |
| 1.75 | 2.3% | the reorder |

57% is ducc0's arithmetic and [is not moving](nested-sampling-gridder-floor.md).
Of what is left, the largest single item that is one contiguous piece of code is
the fourth row: one call to `WSClean::RunClean()` has not started, and 6.5% of
the evaluation is already gone.

## Instrumenting it

Nothing in that window logs, so the cheapest way to split it is to make it log.
A throwaway patch dropped into `docker/wsclean/patches/` as `0099-timing.patch`
(applied last, so it diffs against the tree with 0001-0005 already on it) adds
`Logger::Info << "TMARK ..."` lines around each candidate, and
`docker build --target build` gives a runnable tree in about four minutes
without touching the runtime image or `ri.build-inputs`. `-log-time` timestamps
every line, so the markers read straight off the log:

```
2026-Aug-29 15:02:09.112628                        (banner)
2026-Aug-29 15:02:09.112682 TMARK parsed           CommandLine::Parse   0.054 ms
2026-Aug-29 15:02:09.113154 TMARK cfitsio-init     fits_init_cfitsio    0.472 ms
2026-Aug-29 15:02:09.113170 TMARK runclean-entry   thread pool          0.016 ms
2026-Aug-29 15:02:09.116986 TMARK observation-info getObservationInfo   3.816 ms
2026-Aug-29 15:02:09.117002 TMARK imaging-table-entry  facets           0.016 ms
2026-Aug-29 15:02:09.117077 TMARK bands            open + MultiBandData 0.075 ms
2026-Aug-29 15:02:09.117093 === IMAGING TABLE ===                       0.016 ms
```

Two items, and no third: argument parsing is 54 us and the imaging table itself
is 91 us. `getObservationInfo()` is three lines
(`main/wsclean.cpp:626`) - open the set, read the phase centre, convert it to
J2000 - so a 90-line casacore probe compiled against the build stage
(`--target build`, since the runtime image has no headers) splits it:

| | ms |
| --- | ---: |
| `casacore::MeasurementSet` construction, first in the process | 2.15 |
| ... same set again | 0.14 |
| ... a *different* set, after one has already been opened | 0.62 |
| read `ANTENNA.POSITION`, first | 0.06 |
| read `FIELD.PHASE_DIR`, first | 0.04 |
| read `TIME`, first | 0.10 |
| `MDirection::Convert` to J2000, first | 0.05 |
| ... second | 0.005 |

The whole 3.8 ms is the open, and the measures conversion that looks expensive
is 45 us. That reproduces [the phase-profile doc's
number](nested-sampling-phase-profile.md): ~1.5 ms of the first open is
process-global casacore state (the difference between the first row and the
third), the rest is per-set work a child pays either way.

## What the zygote now warms

`docker/wsclean/src/zygote.cpp` already pays FFTW's planner state once per rank
([the planner doc](nested-sampling-fftw-planner.md)). It now pays these two as
well:

* `fits_init_cfitsio()` at start-up. It needs nothing from the request, and
  `CommandLine::Run` calls it as its first statement anyway, so a child that
  inherits an initialised cfitsio does exactly what it would have done.
* `WarmCasacore()` on the first request, over the Measurement Set that request
  names. casacore's process-global half only warms if a real Measurement Set is
  opened - a plain `casacore::Table` warms less than half of it and creating a
  throwaway one costs 26 ms - and the fork server has no set of its own, so it
  borrows the first one it is asked to image. The set is opened and closed
  before the fork, so no handle, lock file or cache entry crosses into a child.
  A path that turns out not to be a Measurement Set costs the warm-up and
  nothing else: the exception is swallowed and the child opens it again and
  reports the error itself.

Both are ordinary lazy initialisation of shared-library state. Neither changes
what a child computes, which is what the FITS comparison below checks.

## What it is worth

The phase this targets is in every evaluation's own log, so the primary
measurement needs no rig: run the two builds as two simultaneous 10-rank
searches and read the phase out of both. Eight pairs, swapping which arm starts
first, `--nlive 60 --mpi-procs 10`, ~2400 evaluations an arm:

| pair | A: banner -> imaging table | B: same | delta | A: minor loop | B: same |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.95 | 4.67 | -2.28 | 7.02 | 7.08 |
| 2 | 7.32 | 4.73 | -2.59 | 7.31 | 7.12 |
| 3 | 7.19 | 4.85 | -2.34 | 7.21 | 7.18 |
| 4 | 7.25 | 4.82 | -2.43 | 7.25 | 7.29 |
| 5 | 7.03 | 4.64 | -2.39 | 7.05 | 7.06 |
| 6 | 6.95 | 4.84 | -2.11 | 6.85 | 7.36 |
| 7 | 7.25 | 4.78 | -2.47 | 7.33 | 7.18 |
| 8 | 7.22 | 4.82 | -2.40 | 7.32 | 7.22 |

All in ms an evaluation. The last two columns are radler's minor loop, which
this change cannot touch, carried as an in-log null: it moves by -0.2 to +0.5 ms
against a signal of -2.4 ms, in the same logs. Median **-2.39 ms an
evaluation**, or -1.8% of the ~130 ms of logged work an evaluation at this
concurrency.

Second measurement, the fork server driven directly. Both `wsclean-zygote`
binaries `docker cp`'d out of their images (`libwsclean.so` is byte-identical
between them, so only the fork server differs), both started as long-lived
processes in one container - which is how a rank runs them, and the only way a
once-per-process warm-up can be measured honestly - and fed the same 46
Measurement Sets on tmpfs, each with its own `-scale`, alternating which arm
goes first:

| | paired median B/A | pairs |
| --- | ---: | ---: |
| A against a copy of itself | 0.9974 | 230 |
| A against B | 0.9787 | 230 |
| A against B, repeat | 0.9773 | 230 |

**-1.9%** on the child after the null, agreeing with the in-log number.

End-to-end evaluations per second does not resolve this. The eight swapped
pairs give a bucketed `image_binary_seconds` ratio of 0.9774 median, but the
per-pair values run 0.936 to 1.048 and track the sign of their own
`simulate_seconds` null - as
[the gridder-floor doc](nested-sampling-gridder-floor.md) already records, a 2%
effect is under the resolution of a two-minute search.

Identity: 230 of 230 output FITS data blocks byte-identical between the arms
over the 46-set corpus (five images a set, two rounds). Compare data blocks
rather than files - the header carries the `-name` path.

`peak_memory_bytes` moves: 36.6 MB median before, 35.8 MB after. It is
`wait4()`'s `ru_maxrss` for the child, and a child that inherits warm pages
touches fewer of its own.

## The restoring-beam fit, priced and not taken

The row above the one this change took is the beam fit, 6.1 ms an evaluation on
the mean and 3.3 ms on the median. It is one `DetermineBeamSize` call
(`math/imageoperations.cpp:105`) that runs *two* `Fit2DGaussianCentred` fits:
the fitting box is sized from the theoretical beam, this parameter space's
fitted beam is a very stable 1.65x that (min 1.63, max 1.77 over 522
evaluations), and the retry triggers at 1.25x, so the first fit runs in a 30x30
box and the second in a 50x50 one, every time.

Reading `schaapcommon/src/fitters/gaussianfitter.cc` to price the fit turns up
something else. The residual and Jacobian loops compute the pixel offset as
`(xi - x_mid) * scale`, where `xi` is a `size_t` and `x_mid` an `int`, so the
subtraction is done in unsigned arithmetic and every pixel left of - or above -
the box centre gets a coordinate of **+6.1e17** instead of a small negative one:

```
xi= 0  (xi - x_mid)*scale = 6.14891e+17
xi=14  (xi - x_mid)*scale = 6.14891e+17
xi=15  (xi - x_mid)*scale = 0
xi=16  (xi - x_mid)*scale = 0.0333333
```

The Gaussian underflows to exactly 0 there and its Jacobian row is exactly 0, so
those pixels are parameter-independent and contribute nothing to the fit:
**WSClean fits its restoring beam on the bottom-right quadrant of the box it
means to use.** A standalone rebuild of the fit confirms it at run time - 38121
of 50640 model evaluations in a fit pair underflow, i.e. 75.3%. This is upstream
code, present in the revision WSClean v3.7 pins and in `schaapcommon` master,
and it is a genuine WSClean failure mode of exactly the kind this repo exists to
find - but fixing it changes the fitted beam, hence the restored image, hence
every metric, so it is recorded here rather than patched.

The bit-identical half of it was measured and rejected. `std::exp(a)` is exactly
0.0 for `a < -745.2`, so guarding the call with `if (a < -746.0) return 0.0;`
removes three quarters of the transcendental work and reproduces `sx`, `sy` and
`beta` to all 17 digits - and is worth only **-14%** of the fit (1.049 ms to
0.897 ms a fit pair), because the fit is dominated by the divisions in the
Jacobian and by GSL's linear algebra over the full n x 3 system, not by `exp`.
That is ~1% of an evaluation, under what any rig here resolves, for a patch
against a vendored submodule. Not worth carrying.

## Reproducing

```bash
# 1. the phase table this started from
./ri search wsclean --nlive 60 --mpi-procs 10 --max-ndead -1 --output-dir results/nested-sampling/probe
./ri profile results/nested-sampling/probe --phases

# 2. split the pre-imaging phase (throwaway; delete the patch afterwards)
#    add Logger::Info << "TMARK ..." lines, then
docker build --target build -t wsclean-build -f docker/wsclean/Dockerfile .
docker run --rm -v "$PWD:$PWD" --entrypoint bash wsclean-build -lc \
  'LD_LIBRARY_PATH=/opt/wsclean/lib:/opt/casacore/lib /opt/wsclean/bin/wsclean ... -log-time <ms>'

# 3. A/B the fork server: tag the current image aside, rebuild, drive both
docker tag ri-reproducibility/wsclean:v3.7 wsclean-base
./scripts/build.sh wsclean
WSCLEAN_IMAGE=wsclean-base ./ri search wsclean --no-build --output-dir ... &
./ri search wsclean --no-build --output-dir ... &
wait
./ri profile <each run> --phases | grep 'IMAGING TABLE'
```

A search with `--keep-measurement-sets` supplies the replay corpus; `./ri
self-check zygote` covers the fork-server protocol after any change to
`zygote.cpp`.
