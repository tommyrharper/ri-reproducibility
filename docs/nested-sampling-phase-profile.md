# The per-evaluation phase profile

**Every evaluation's `wsclean.stdout.log` is now a microsecond phase timeline -
`polychord_wsclean.py` passes `-log-time`, which costs nothing measurable - so
`./ri profile <run> --phases` breaks the 84% of an evaluation that is the
`wsclean` binary into its phases without a rig, a patched tree or a replay
corpus. This page is what that reads on the post-zygote tree, and the two
avenues it closes.**

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65 W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## Where an evaluation goes now

A 5312-evaluation search, `--mpi-procs 20 --nlive 25 --num-repeats 10
--max-ndead 250`, on the tree as of this page
(`./ri profile <run>`):

| stage | per evaluation | share of worker-time |
| --- | ---: | ---: |
| simulate (MeqTrees) | 17 ms | 8.3% |
| `wsclean` binary | 170 ms | 84.0% |
| container overhead around it | 2 ms | 1.0% |
| metrics | 2 ms | 0.8% |
| **accounted** | **191 ms** | **94.0%** |
| unaccounted (PolyChord sampling + idle) | | 6.0% |

The 6% unaccounted is the ~6 s per-run startup constant plus 0.8%, not a rate
(see [the throughput doc](nested-sampling-throughput.md)) - at `--nlive 200` the
same run accounts for 98.3%. There is nothing left in the harness: the whole of
it outside the `wsclean` binary is 21 ms, and 17 ms of that is a simulator whose
own breakdown is 2.3 ms of visibilities, 2.2 ms of copying a cached Measurement
Set skeleton and ~1 ms of everything else.

Of the 170 ms `wsclean` process, 165 ms lands between its first and last log
line. The remaining ~5 ms is `fork()` from the zygote plus the tail past the
last log line; the 27 ms that used to run before `main()` is
[gone](nested-sampling-wsclean-zygote.md).

## Inside the binary

`./ri profile <run> --phases`, same run, largest first. A log line's timestamp
is written when the line *starts*, so the work a line announces sits in the gap
between it and the next line - and gaps are bucketed by (line, next line),
because `Loading data in memory...` appears once per gridding pass and means
something different each time.

| ms/eval | share | n/eval | ms each | phase |
| ---: | ---: | ---: | ---: | --- |
| 48.83 | 29.6% | 8.51 | 5.737 | `Gridding N rows...` -> `Gridded visibility count` |
| 31.26 | 18.9% | 6.51 | 4.801 | `Predicting N rows...` -> `Writing...` |
| 11.64 | 7.1% | 3.98 | 2.923 | minor loop (`Performed N iterations` -> `Stopped on peak N mJy`) |
| 9.70 | 5.9% | 0.85 | 11.426 | `Fitting beam...` -> `Writing psf image... DONE` |
| 7.46 | 4.5% | 1.00 | 7.457 | the reorder |
| 6.30 | 3.8% | 1.00 | 6.295 | `No corrected data...` -> `=== IMAGING TABLE ===` |
| 4.99 | 3.0% | 6.51 | 0.767 | `Opening reordered part N` -> `Loading metadata...` |
| 4.94 | 3.0% | 1.00 | 4.936 | first log line -> `No corrected data...` |
| 4.33 | 2.6% | 0.87 | 4.978 | `Rendering sources to restored image` -> `Writing restored image... DONE` |
| 4.25 | 2.6% | 1.00 | 4.250 | imaging table -> `WARNING: ... WEIGHT_SPECTRUM` |
| 3.79 | 2.3% | 8.51 | 0.446 | `Loading data in memory...` -> `Gridding N rows...` |
| 3.00 | 1.8% | 1.53 | 1.960 | minor loop (µJy) |
| 2.74 | 1.7% | 5.51 | 0.498 | `Gridded visibility count` -> `== Deconvolving (N) ==` |
| 2.32 | 1.4% | 1.00 | 2.316 | `Opening reordered part N` -> `Determining min and max w` |

Read as four groups:

- **Gridding and degridding, 80 ms (48%).** 8.51 gridding passes (PSF, dirty,
  and one per major cycle) and 6.51 degridding passes, all of them inside
  ducc0's w-gridder. 5.7 ms to grid 4140 visibilities into 128x128 is ~1000x the
  arithmetic, which is the same conclusion iteration 16 reached from the other
  end: the pass is dominated by per-call setup, not by the data. The only lever
  on it is *fewer passes*, which is `-mgain`
  ([the clean-loop doc](nested-sampling-clean-loop.md)), deliberately left at
  0.8.
- **Deconvolution, 16 ms (10%).** The minor loops themselves.
- **The beam fit, 11.1 ms (6.7%).** See below - this is the largest item that is
  not imaging arithmetic, and it had never been attributed before.
- **Metadata and I/O, ~35 ms (21%).** The reorder (7.5 ms, bounded and not
  reusable - iteration 16), the pre-imaging path (16 ms), the reordered-part
  opens (9 ms, already cheap since
  [the antenna-name patch](nested-sampling-wsclean-patches.md)), and the FITS
  writes (~1 ms).

## The beam fit

11.1 ms per evaluation, once per evaluation, mean over 5312 - and it is fitting
a 2D Gaussian to a 128x128 PSF, not touching visibilities. It is heavy tailed:
median 6.5 ms, p90 18.5 ms, worst 78 ms. About half of that spread is the
machine rather than the fit (the slowest decile's *gridding* is also 1.9x the
fastest decile's), but the fit itself grows 6.3x across the deciles where
gridding grows 1.9x, so most of the tail is real.

The mechanism is in `schaapcommon::fitters::Fit2DGaussianCentred`. It fits
inside a box `ceil(beam_estimate * box_scale_factor)` pixels wide, where
`beam_estimate` is WSClean's *theoretical* beam and `box_scale_factor` is
`-beam-fitting-size`; if the fitted major axis needs more than 80% of that box
it doubles down and refits. In this parameter space the fitted beam is
consistently ~1.75x the theoretical one (`major=45.99''` against
`theoretical=26.24''`), so **every evaluation pays the retry**: a ~30x30 fit
that is thrown away, then a ~52x52 one. Each fit is GSL's `lmsder` over
`width*height` residuals with three parameters, and each residual is an
`std::exp`.

No result-preserving lever was found here. The exact retry condition, and why
no value of `-beam-fitting-size` avoids the second fit, is in
[the cost-model doc](nested-sampling-cost-model.md). The first fit cannot be
skipped - its output is what sizes the second box and seeds its estimate - and every
knob that changes the box (`-beam-fitting-size`, `-circular-beam`,
`-no-fit-beam`, `-beam-size`) changes the fitted beam, hence the restored image,
hence every metric. It is recorded here because it is 6.7% of an evaluation
sitting in ~200 lines of straight-line arithmetic, which is a better shape for
a future upstream contribution than anything else left in the profile.

## Two avenues closed

### Pre-warming the zygote parent is worth ~1%, not ~7%

[The zygote doc](nested-sampling-wsclean-zygote.md) left "a zygote that imaged
one throwaway Measurement Set before serving would fork children that had
already paid casacore's lazy initialisation" as the next thing to measure, and
estimated it at the ~11 ms between a child's first log line and
`=== IMAGING TABLE ===`. Measured, that estimate is ~7x too high.

A probe that opens the same Measurement Set repeatedly in one process, against
a current-shape MS on tmpfs:

| | ms |
| --- | ---: |
| `casacore::MeasurementSet` open, first in the process | 1.64 |
| ... second and after | 0.70 |
| ... first, after the process has already opened a plain `casacore::Table` | 1.30 |
| ... first, after the process has already opened one `MeasurementSet` | 0.70 |

So the whole process-global premium a parent could pre-pay is **0.94 ms serial**,
or ~2 ms at production concurrency: ~1% of an evaluation, and below what the
rig below resolves. The rest of the 11 ms is per-Measurement-Set work the child
would pay anyway.

Two things make that number small and worth knowing: casacore's *expensive*
lazy initialisation is the measures tables, and the first
`MDirection J2000 -> AZEL` conversion in a process costs **34 ms** - but WSClean
never asks for one in this configuration (no beam, no primary-beam correction),
so it is not in the budget and pre-warming it would buy nothing. And a warm-up
needs a real Measurement Set to open: creating a throwaway casacore table
instead costs 26 ms and warms less than opening one does.

### The MeasurementSet class costs 0.51 ms an open more than a Table

`casacore::MeasurementSet ms(path)` attaches every subtable; `casacore::Table
t(path)` does not:

| open | ms |
| --- | ---: |
| `casacore::MeasurementSet` | 0.71 |
| `casacore::Table` | 0.20 |

WSClean still constructs a `MeasurementSet` over the parent ~4 times per
evaluation outside the gridding loop: `Settings::determineDataColumn` (which
only wants `tableDesc().isColumn("CORRECTED_DATA")`, so a plain `Table` would
do), `WSClean::makeImagingTable`, `WSClean::getObservationInfo` and the row
provider that drives the reorder. Making all four cheap is worth at most
~2 ms serial - ~4 ms at concurrency, ~2% - and only the first of the four could
drop to a `Table` without restructuring. That is the floor left after
[the six dropped subtables](nested-sampling-ms-open.md) and
[the antenna-name cache](nested-sampling-wsclean-patches.md): the parent-MS-open
avenue is now bounded on both sides.

## `-log-time` is free

Interleaved replay over 63 Measurement Sets on tmpfs, four passes, three arms
(baseline, a duplicate of the baseline as the null, and `-log-time`), all 756
commands shuffled into one `xargs -P 19` list so the arms compete with each
other rather than with the clock:

| arm | mean ms | median ms |
| --- | ---: | ---: |
| baseline | 128.95 | 123.15 |
| null (same command) | 126.66 | 121.24 |
| `-log-time` | 130.08 | 122.71 |

The null pair puts this rig's resolution at 1.8%; `-log-time` reads +0.9% on the
mean and -0.4% on the median, i.e. nothing. On disk it is ~1 KB on a 400 KB
evaluation.

## Reproducing it

```sh
./ri search wsclean --mpi-procs 20 --nlive 25 --num-repeats 10 --max-ndead 250
./ri profile <run>                # the stage budget
./ri profile <run> --phases       # inside the binary
```

`--phases` needs a run made since `-log-time` became a default; on an older run
it says so rather than printing an empty table. `--top N` widens the table.

The replay rig - one line per (Measurement Set, arm), shuffled, 19 at a time,
each timed by its own `date +%s%N`, with a duplicated arm as the null pair - is
the shape [the clean-loop doc](nested-sampling-clean-loop.md) describes, run
against a corpus kept by `./ri search wsclean --keep-measurement-sets` and
copied onto `/dev/shm` so an open costs what it costs in a run
([why](nested-sampling-wsclean-patches.md)).

The casacore probes are ~40 lines of C++ each, compiled against the headers the
`wsclean` runtime image already carries:

```sh
docker build -t wsclean-probe - <<'EOF'
FROM ri-reproducibility/wsclean:v3.7
RUN apt-get update && apt-get install -y --no-install-recommends g++
EOF
docker run --rm -v /tmp/probe:/probe -w /probe --entrypoint sh wsclean-probe -c '
  g++ -O2 -std=c++17 -I/opt/casacore/include probe.cpp -o /tmp/p \
      -L/opt/casacore/lib -lcasa_ms -lcasa_measures -lcasa_tables -lcasa_casa
  cat /proc/cpuinfo > /dev/null   # a cold open of it blocks ~20 ms
  /tmp/p /corpus/<something>.ms'
```
