# Opening the Measurement Set

**WSClean re-opens the parent Measurement Set once per gridding and degridding
pass - 15.7 times in a median evaluation - and casacore attaches every subtable
on every one of those opens. Deleting the five subtables nothing downstream
reads is worth +14.9% evaluations per second end to end, with bit-identical
images.**

This is the largest single item outside the gridding arithmetic itself, and it
was invisible to every previous decomposition because WSClean's own phase line
(`Inversion:, prediction:, deconvolution:`) does not count it -
[the evaluation budget](nested-sampling-evaluation-budget.md) filed it under
the 27% "untimed" residual along with process start.

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65W package limit [the power-limit doc](nested-sampling-power-limit.md)
describes, 19 concurrent WSClean processes throughout. 29 August 2026.

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

## What is left

Seven subtables remain: `ANTENNA`, `DATA_DESCRIPTION`, `FEED`, `FIELD`,
`OBSERVATION`, `POLARIZATION`, `SPECTRAL_WINDOW`.

- `OBSERVATION` is **not** droppable: WSClean dies with
  `MSObservation(const MSObservation &) - table is not a valid MSObservation`,
  exit 255.
- `FEED` is droppable - a stripped MS images fine - but the two measured points
  (12 subtables at 3.84 ms an open, 7 at 2.79 ms) put a subtable at ~0.21 ms,
  so it is worth ~3 ms of a 236 ms binary, ~1.5%, and it is the receptor
  geometry any future beam or polarisation work would need. Not taken.
- The remaining five are the phase centre, the frequencies, the polarisation
  setup and the array, all of which WSClean reads.

So the floor is ~2.8 ms an open and ~44 ms an evaluation, and getting under it
means not re-opening the MS 15.7 times, which is WSClean's business rather than
this repo's.

## Consequence for a kept Measurement Set

`./ri search --keep-measurement-sets` now keeps an MS that is missing five
subtables the MSv2 specification calls required. Anything that opens it as a
`casacore::MeasurementSet` (CASA, `casabrowser`, MeqTrees) will refuse it;
anything that opens it as a plain casacore table (`python-casacore`'s
`table()`, `ms_to_r2d2_mat.py`, WSClean) is fine. The five carry no science for
a single-field point-source simulation, but a kept MS is a debugging artefact
now rather than a portable one.

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
