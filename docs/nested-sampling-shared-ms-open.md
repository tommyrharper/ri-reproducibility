# Opening the parent Measurement Set once

**A `wsclean` run constructs a `casacore::MeasurementSet` over its input five
separate times before it reads a single visibility, and casacore re-reads the
table header and re-attaches every subtable on each one. Handing out a copy of
one already-open handle instead is `docker/wsclean/patches/0003`, worth -3.7%
on the `wsclean` binary in an interleaved replay and -4.4%/-6.1% in two
simultaneous swapped searches, with 800 output FITS data blocks
bit-identical.**

This is the third and last of the Measurement-Set-open findings.
[The MS-open doc](nested-sampling-ms-open.md) closed the avenue from the
simulator's side (delete the subtables nothing reads), and
[the patches doc](nested-sampling-wsclean-patches.md) removed WSClean's
per-pass re-open for the antenna names. What was left was the block of metadata
WSClean reads before the first visibility, which
[the simulate-stage doc](nested-sampling-simulate-stage.md) named as the next
target off its median phase table: 8.22 ms from process start to
`=== IMAGING TABLE ===` plus 3.89 ms from there to the reorder, 8.8% of an
evaluation.

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65 W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## Counting the opens, rather than guessing at them

Reading the source found four `casacore::MeasurementSet ms(path)` sites on this
configuration's path, which would have made the whole avenue worth about
2 ms. Counting them found five, because one of the four runs twice.

The counting needs no rebuild and no profiler, both of which this host is short
of: casacore opens a table through `open64("<ms>/table.dat")`, so a 30-line
`LD_PRELOAD` shim that wraps `open`/`open64`/`fopen` and counts the paths equal
to `$MS_ROOT/table.dat` prices the whole thing.

```sh
MS_ROOT=/work/corpus/ms1.ms LD_PRELOAD=./count.so wsclean ... /work/corpus/ms1.ms
```

| build | `table.dat` opens per run |
| --- | ---: |
| baseline | 21 |
| with 0003 | 4 |

One `casacore::MeasurementSet` construction is four of those opens (calibrated
by running the same shim over a program that opens `n` of them and dividing),
so the baseline is five constructions plus a stray `Table::isReadable`, and the
patched build is one construction.

What one construction costs, on the same warm tmpfs Measurement Set:

| | ms |
| --- | ---: |
| first `MeasurementSet` in a process | 1.81 |
| every one after it | 0.60 |
| copying a handle that is already open | 0.02 |

So the four repeat constructions were ~2.4 ms of serial work, and the copies
that replace them are ~0.1 ms.

## The patch

`OpenMeasurementSet(path)` in `msproviders/msprovider.{h,cpp}`: a mutex and a
`std::map<std::string, casacore::MeasurementSet>`, returning a copy of the
handle. `casacore::Table` is reference-counted, so the copy is the same table.
The four call sites - `WSClean::getObservationInfo()`,
`WSClean::makeImagingTable()`, `ReorderMS()` and `MakeMsRowProvider()` - each
lose their own `casacore::MeasurementSet ms(path)`.

Two details are load-bearing:

- **The cache is deliberately leaked** (`static std::map<...>& cache = *new
  std::map<...>()`). casacore's own table cache is a static as well, and if it
  is torn down first the sets left in this one close against freed state: the
  first build of this patch wrote all five correct FITS images and then
  segfaulted in its exit handlers. Production would never have seen it - the
  zygote's children `_exit()` past the global destructors - which is exactly
  why it is worth naming.
- **Access is serialised, the handles are not.** Two `casacore::Table` copies
  of one table are no more thread-safe than one, and the cache does not change
  that. It is safe here because each path's sites run in sequence, and safe
  upstream for the same reason: `MsHelper::PerformReordering()` parallelises
  over Measurement Sets, so two threads never hold the same path.

## Interleaved replay

160 Measurement Sets kept out of a real search (`./ri search --keep-measurement-sets`),
copied onto `/dev/shm` so that they sit on the same tmpfs a scored evaluation's
`sim.ms` does - [the patches doc](nested-sampling-wsclean-patches.md) records
that a corpus on ext4 overstates an MS-open win - and replayed with each
evaluation's own `-scale`, computed from `simulation.json` the way
`image_pixel_size_arcsec()` does. Both builds' `wsclean` and `libwsclean.so`
were copied out of their images and run from one container with
`LD_LIBRARY_PATH`, alternating arm and alternating which arm goes first, so the
two are interleaved at the process level rather than run as two blocks.

10 rounds x 160 sets x 2 arms = 3200 runs:

| | baseline | patched | paired median ratio |
| --- | ---: | ---: | ---: |
| median ms | 84.64 | 81.90 | **0.9632** |
| null (baseline against a copy of itself) | 84.56 | 85.29 | 1.0003 |

-3.0 ms an evaluation, -3.7%, against a null that reads 0.03% on the same
statistic and over the same 1600 pairs.

## End to end

Two 10-rank searches started at the same instant, one per image
(`WSCLEAN_IMAGE` selects the build), `--nlive 50 --num-repeats 5 --max-ndead
300 --seed 4242`, then the pair repeated with the start order swapped.

The two arms do not sample the same points - asynchronous MPI makes the
sampler's path depend on completion order - and an evaluation's cost is mostly
set by its visibility count
([the cost model](nested-sampling-cost-model.md)), so comparing raw medians
compares regions rather than builds: pair 2's arms differ by 37% in median
`observation_minutes x channel_count`. Bucketing both arms' evaluations into
twelve equal-count bins of that size proxy and taking the median of the
per-bin median ratios removes it.

| pair | evaluations | median of bin ratios on `image_binary_seconds` | `simulate_seconds` null |
| --- | ---: | ---: | ---: |
| 1 | 3264 / 3355 | **0.9558** | 0.997 |
| 2 | 3479 / 3351 | **0.9386** | 1.056 |

Every one of the 24 bins is below 1.0. The win is larger than the replay's
-3.7% because a saved open is syscall-bound and this host inflates such work
~2.2x at production concurrency
([the throughput doc](nested-sampling-throughput.md)); pair 2's null says its
patched arm was on the slow side of the pair by 5.6% and it still won, so
0.9386 is if anything pessimistic.

## Images

800 FITS files (5 per evaluation x 160 evaluations) written by each build over
the same corpus. Every data block is byte-identical; the only difference
anywhere in the files is the `HISTORY` card, which records the output path each
arm was given.

## What is left

Nothing, on this avenue. The input Measurement Set is now opened once per
`wsclean` process, and the remaining first-open premium - 1.81 ms against
0.60 ms for a repeat - is casacore's process-global lazy init. A fork server
*could* pay that in the parent: forking children off a parent that has already
opened a Measurement Set drops their first open from 1.65 ms to 0.96 ms
(median over 96 forks each). That is ~0.7 ms an evaluation, ~0.4%, and it needs
the zygote parent to hold a Measurement Set open that no evaluation asked for -
not worth it while there are larger items, and recorded here so the number does
not have to be measured again.

## Reproducing

```sh
# the corpus: any search, with its measurement sets kept
./ri search wsclean --nlive 8 --max-ndead 30 --mpi-procs 4 \
  --keep-measurement-sets --output-dir results/nested-sampling/corpus
# baseline image beside the working tree's
docker tag ri-reproducibility/wsclean:v3.7 ri-reproducibility/wsclean:base
./ri build wsclean
# two simultaneous searches, one image each, then swap the start order
WSCLEAN_IMAGE=ri-reproducibility/wsclean:base ./ri search wsclean \
  --nlive 50 --num-repeats 5 --max-ndead 300 --mpi-procs 10 --seed 4242 \
  --no-build --output-dir results/nested-sampling/a &
./ri search wsclean --nlive 50 --num-repeats 5 --max-ndead 300 --mpi-procs 10 \
  --seed 4242 --no-build --output-dir results/nested-sampling/b &
wait
```

`./ri build wsclean` always tags `ri-reproducibility/wsclean:v3.7`, so tag the
baseline aside **first** or the second build replaces the arm you meant to
keep.
