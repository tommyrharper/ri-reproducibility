# Opening the Measurement Set

**WSClean re-opens the parent Measurement Set ~16 times per evaluation, and
casacore attaches every subtable each time. Dropping unread subtables is worth
+14.9% end to end, then another -3.2% on `wsclean` for `FEED`, with identical
image data.** The remaining re-opens were removed in
[local WSClean patches](nested-sampling-wsclean-patches.md).

This was hidden in WSClean's untimed phase; the [evaluation budget](nested-sampling-evaluation-budget.md)
measured it with process start. Results below use 19 concurrent WSClean
processes on the documented 65W-limit host, 29 August 2026.

## Finding it

`wsclean -log-time` prefixes every output line with a microsecond timestamp,
which turns any run's stdout into a phase timeline for free. Over 200 real
Measurement Sets replayed 19-way concurrent with each evaluation's own recorded
argv:

| phase | mean | share of the interior |
|---|---:|---:|
| open the MS, build the imaging table | 16.9 ms | 10.5% |
| reorder | 5.2 ms | 3.2% |
| model init + weight precalculation | 1.5 ms | 0.9% |
| PSF (one gridding pass) | 17.2 ms | 10.7% |
| first dirty image (one gridding pass) | 8.0 ms | 5.0% |
| major cycles + restore | 112.4 ms | 69.8% |

Getting to the first byte of data costs as much as gridding the PSF. But the
setup block is not the whole of it: every `Opening reordered part 0 for <ms>`
line in the log is a fresh open of the *parent* MS, and there are 15.7 of them
per evaluation (PSF, dirty, then two per major cycle). Timing the gap between
that line and the next one measures the open directly.

## The measurement

Three arms, all commands in one `xargs -P 19` list so they run simultaneously,
order shuffled, one full discarded warm-up pass first (the burst-clock window -
see the power-limit doc). Arm C is a byte-identical copy of arm A: the null
pair that calibrates the rig.

| | A: as makems writes it | B: five subtables dropped | C: null (= A) |
|---|---:|---:|---:|
| subtables | 12 | 7 | 12 |
| per MS open | 3.84 ms | 2.79 ms | 3.85 ms |
| opens per evaluation | 15.67 | 15.67 | 15.67 |
| **all opens** | **60.1 ms** | **43.7 ms** | 60.3 ms |
| setup block | 21.5 ms | 16.2 ms | 21.6 ms |
| major cycles | 141.3 ms | 128.2 ms | 142.5 ms |
| interior wall | 203.2 ms | 183.6 ms | 205.0 ms |
| wall, geometric mean vs A | - | **0.914** | 1.003 |

The 13 ms that comes off the *major cycles* is the point: nothing about the
deconvolution changed (6.33 major cycles in both arms, images bit-identical),
it is the re-opens inside the loop. All 200 pairs of `recon-image.fits` are
identical byte for byte in the data block; only the header differs, because
WSClean records its own command line and the two arms have different paths.

The five dropped are `FLAG_CMD`, `HISTORY` (both empty), `POINTING` (one row
per antenna, all pointing at the phase centre `FIELD` already gives),
`PROCESSOR` and `STATE` (one default row each). None is read by WSClean, by
`ms_to_r2d2_mat.py`, or by anything else in this repo.

## End to end

Three interleaved pairs of real searches (`--nlive 60 --max-ndead 700
--mpi-procs 20 --seed 4242`), arm order swapped between pairs, both images
rebuilt for each arm:

| pair | evaluations/s, dropped | evaluations/s, as-is | ratio | wsclean binary, dropped | as-is |
|---|---:|---:|---:|---:|---:|
| 1 | 70.95 | 62.34 | 1.138 | 226.0 ms | 265.9 ms |
| 2 | 69.38 | 58.00 | 1.196 | 231.4 ms | 278.8 ms |
| 3 | 69.83 | 62.74 | 1.113 | 235.6 ms | 259.3 ms |
| **geometric mean** | | | **1.149** | | **0.862** |

The simulate stage is unmoved (17.4 ms against 17.5 ms), which is the other
half of the result: the drop is free because it happens on a table this
evaluation already has open read-write.

Note that the replay rig understates the win - 0.914 there against 0.862 on the
binary in a real search. A replay corpus removes the simulate stage and the
harness from the machine, and an MS open is metadata- and syscall-bound rather
than arithmetic, so it is exactly the kind of work that gets more expensive
when the rest of the run is competing for the same cores. Treat the replay
number as a lower bound for anything latency-shaped.

## Where the drop happens, and why not earlier

In `fill_point_source_visibilities()` (`simulate_point_source_ms.py`), inside
the block that is already open read-write to write `DATA`, immediately after
the predict.

The obvious cheaper place is the cached MS skeleton - it is built once per
`(NTimes, NFrequencies)` shape at image build time and copied per evaluation,
so stripping it there would also make the copy smaller. It does not work:
casacore refuses to open an MS that is missing any required subtable, and the
MeqTrees predict path opens it that way. Dropping any *one* of the five from
the skeleton fails the predict with `meqserver reported 1 error(s)`; all five
were tried individually. That path only runs for a source off the phase centre
(`source_offset_fraction`, currently `enabled = false`), which is exactly the
kind of dependency that would have gone unnoticed until someone enabled the
parameter - `self_check_dropped_subtables()` runs both predict paths and is the
guard.

WSClean is more permissive than casacore's `MeasurementSet` class and images
the stripped MS without complaint.

## FEED, and the end of the avenue

The first pass left seven subtables and guessed at what they were worth. This
section measures the lot and takes the last one that moves.

### The screen

Six drop arms plus a baseline, all in one `xargs -P 19` list over the same 12
Measurement Sets. The question is only whether WSClean still runs:

| dropped | WSClean exit, 12/12 |
|---|---|
| `ANTENNA` | 255 |
| `DATA_DESCRIPTION` | 255 |
| `FIELD` | 255 |
| `POLARIZATION` | 255 |
| `SPECTRAL_WINDOW` | 255 |
| `OBSERVATION` | 255 (measured in the first pass) |
| **`FEED`** | **0** |

`FEED` is the only one, which is what the first pass assumed but never tested
for the other five. There is no eighth subtable to find.

### What FEED is worth

40 Measurement Sets, two arms in one 19-way `xargs` list, each evaluation's own
recorded argv with `-log-time` added, one discarded pass first:

| | as the first pass left it | `FEED` dropped |
|---|---:|---:|
| subtables | 7 | 6 |
| per MS open | 2.161 ms | 1.936 ms |
| opens per evaluation | 15.60 | 15.60 |
| **all opens** | **33.7 ms** | **30.2 ms** |
| interior wall | 131.80 ms | 127.51 ms |

-0.225 ms an open, -4.3 ms an evaluation, -3.3% of the replay interior. All
120 `recon-image.fits` / `recon-dirty.fits` / `recon-residual.fits` data blocks
are byte-identical between the arms.

### End to end

Three interleaved pairs of real searches (`--nlive 25 --num-repeats 10
--max-ndead -1 --mpi-procs 20`, one seed per pair, arm order swapped in the
middle pair, a discarded warm-up search first, both `meqtrees` images built and
tagged before either arm ran):

| pair (seed) | wsclean binary, kept | dropped | ratio | evaluations/s, kept | dropped | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 4242 | 235.94 ms | 236.22 ms | 1.001 | 61.06 | 67.81 | 1.111 |
| 7 | 239.97 ms | 230.06 ms | 0.959 | 66.96 | 69.17 | 1.033 |
| 99 | 243.33 ms | 229.72 ms | 0.944 | 66.27 | 70.15 | 1.059 |
| **geometric mean** | | | **0.968** | | | **1.067** |

Read the binary column, not the wall clock: the three runs differ by 5955,
6078 and 6520 evaluations at the same settings, which is the sampler's own
variance and swamps a 3% effect (this repo has made that mistake twice
already). -3.2% on the binary against -3.3% on the replay interior is the
result, and the two rigs agreeing that closely is the reason to believe a
number this small at all. Pair 4242 read zero on its own, which is what a 3%
effect looks like against ~4% run-to-run spread; a fourth pair would not have
been cheaper than the replay rig that already agrees.

The receptor geometry the first pass kept it for does not survive contact with
what the MS now is. A scored evaluation's MS is deleted as its `metrics.json`
is written, and a `--keep-measurement-sets` one was already missing five
subtables MSv2 calls required before `FEED` joined them, so no beam or polarisation tool would open it
anyway - `FEED` was being preserved inside an artefact nothing conformant can
read. The search itself images one unpolarised point source in Stokes I with no
beam model.

### Why the six that stay cannot be made cheaper either

A casacore table open is ~0.056 ms of fixed cost plus ~0.0026 ms per column
(measured on synthetic 1/2/4/8/16-column tables), so the 47 columns across the
six survivors are about 40% of their attach cost. Removing the columns WSClean
has no use for - `OBSERVATION`'s `LOG`/`SCHEDULE`/`OBSERVER`/`PROJECT`,
`ANTENNA`'s `OFFSET`/`TYPE`/`MOUNT`/`STATION`, `SPECTRAL_WINDOW`'s
`FREQ_GROUP`/`NAME`/`NET_SIDEBAND`, `FIELD`'s `DELAY_DIR`/`REFERENCE_DIR` -
fails exactly the way dropping the table does: WSClean opens each one through
casacore's `MSObservation`/`MSAntenna`/`MSSpectralWindow` classes, which
validate their required column set, and all 8 stripped Measurement Sets exited
255 with `An exception occured` against 8/8 clean in the paired baseline. The same validation that makes a subtable
undroppable makes its columns unremovable.

`IncrementalStMan` also refuses to give up a single column, so the main
table's 17 unused-by-WSClean scalar columns (`FLAG_CATEGORY`, `SIGMA`,
`EXPOSURE`, `FEED1`, `FEED2`, ...) cannot be dropped either -
`Table::removeColumn - column FLAG_CATEGORY cannot be removed`.

So the floor is ~1.9 ms an open and ~30 ms an evaluation, and getting under it
means not re-opening the MS 15.6 times, which is WSClean's business rather than
this repo's.

## Where the evaluation stands after both drops

The same free measurement, on a whole 6525-evaluation search rather than a
replay corpus (`--nlive 25 --num-repeats 10 --seed 4242 --mpi-procs 20`, 266 ms
an evaluation by `./ri profile`, 208.5 ms of it inside WSClean's own logged
timeline):

| item | per evaluation | count | share of the evaluation |
|---|---:|---:|---:|
| gridding passes | 51.0 ms | 8.52 | 19.2% |
| **MS re-opens** | **48.3 ms** | **16.04** | **18.2%** |
| degridding passes | 32.7 ms | 6.52 | 12.3% |
| process start (not in the timeline) | ~30 ms | 1 | ~11% |
| simulate (not in the timeline) | 18 ms | 1 | 6.8% |
| deconvolution | 16.4 ms | 6.52 | 6.2% |
| beam fit + restore | ~13.6 ms | 1 | 5.1% |
| first open + imaging table | 11.6 ms | 2 | 4.4% |
| reorder | 7.6 ms | 1 | 2.9% |

Even after both drops the re-opens are still the second line of the table, and
they are the only one of the top three the harness has any reach into at all.

## Consequence for a kept Measurement Set

`./ri search --keep-measurement-sets` now keeps an MS that is missing six
subtables the MSv2 specification calls required. Anything that opens it as a
`casacore::MeasurementSet` (CASA, `casabrowser`, MeqTrees) will refuse it;
anything that opens it as a plain casacore table (`python-casacore`'s
`table()`, `ms_to_r2d2_mat.py`, WSClean) is fine. The six carry no science for
a single-field unpolarised point-source simulation, but a kept MS is a
debugging artefact now rather than a portable one.

## Reproducing any of this

The phase timeline needs no rig at all: add `-log-time` to the argv a run
already records in every `metrics.json` under `commands.wsclean`, and every
line of `wsclean.stdout.log` carries a timestamp. Time the gap between
`Opening reordered part 0 for` and the line after it to get the open cost.

For an A/B, take a `--keep-measurement-sets` corpus, copy it once per arm onto
`/dev/shm`, rewrite `-name`/`-temp-dir`/the MS path per arm, put every arm's
commands in one shuffled `xargs -P 19` list with `/usr/bin/time -f %e` on each,
discard the first pass, and always include a null arm that is a copy of the
baseline. The null read 1.003 here; anything under ~1% is not a result. Compare
the FITS *data blocks* rather than whole files - WSClean writes its command
line into the header, so identical images have different checksums.
