# Reading the Measurement Set in row blocks

The reorder is the last item in [the phase
table](nested-sampling-gridder-floor.md) that is not gridding arithmetic. It is
where WSClean reads every visibility of its input set once and writes it out as
the flat files the gridder streams from, and it is 5.4% of an evaluation.

It reads that set **one row at a time**, and casacore charges by the call.
Fetching the same rows in blocks is `docker/wsclean/patches/0005`, worth
**-2.2% on the `wsclean` binary** in an interleaved tmpfs replay and **-4.1%
over four simultaneous swapped searches**, with 530 of 530 output FITS data
blocks byte-identical.

## What the reorder was doing per row

`ReorderMS()` (`msproviders/reorderedmsprovider.cpp`) drives a
`MsRowProviderBase` forwards one row at a time, and `DirectMSRowProvider` /
`MSRowProvider` answer each row out of casacore column objects:

| site | calls per row |
| --- | --- |
| `MSRowProvider::NextRow` | `ANTENNA1`, `ANTENNA2`, `FIELD_ID`, `DATA_DESC_ID`, `TIME`, `UVW` |
| `DirectMSRowProvider::ReadData` | `DATA`, `FLAG`, and `ANTENNA1`, `ANTENNA2`, `FIELD_ID`, `TIME` **again** |
| `MSRowProvider::getCurrentWeights` | `WEIGHT_SPECTRUM` or `WEIGHT` |

Thirteen column reads per row, four of which repeat a value `NextRow()` read a
moment earlier and threw away, plus one whole extra pass over `TIME` in
`Initialize()` to find the last timestep. For the 2106-row set of a median
evaluation that is **29484 casacore column reads**.

## What a column read costs

A 90-line probe compiled against the build stage's casacore
(`docker build --target build`) times each loop over a real set with no
WSClean in the picture. Milliseconds for a whole 2106-row set, best of five,
on tmpfs:

| column | storage manager | row at a time | blocks of 256 | blocks of 1024 | whole column |
| --- | --- | --- | --- | --- | --- |
| `DATA` | `TiledColumnStMan` | 0.490 | 0.059 | 0.053 | 0.032 |
| `FLAG` | `TiledColumnStMan` | 0.318 | 0.026 | 0.023 | 0.002 |
| `UVW` | `TiledColumnStMan` | 0.134 | 0.032 | 0.031 | 0.007 |
| `WEIGHT` | `StandardStMan` | 0.303 | 0.247 | 0.239 | 0.241 |

A tiled column costs ~10x more read one row at a time than read in a block,
and **a block of 256 rows already has almost all of it** - the remaining gap to
a whole-column read is not worth the memory. `StandardStMan` is the exception:
it is per-row storage, so blocking `WEIGHT` buys 20% rather than 900%.

Summed over the six scalar columns, `UVW`, `DATA`, `FLAG` and `WEIGHT`, the
read side of the reorder is **2.0 ms** of a ~48 ms `wsclean` process
row-at-a-time and ~0.45 ms in blocks.

## The patch

`docker/wsclean/patches/0005-read-measurement-set-rows-in-blocks.patch` gives
`MSRowProvider` one forward-only block:

- `fillBlock()` reads `TIME`, `ANTENNA1`, `ANTENNA2`, `FIELD_ID`,
  `DATA_DESC_ID` and `UVW` for a block of rows with `getColumnRange()`;
  `fillBlockVisibilities()` does the same for `DATA`, `FLAG` and the weights,
  but only when a row provider actually asks for them - `AveragingMSRowProvider`
  reads its own visibilities and would otherwise pay for a block it never uses.
- `NextRow()` and `ReadData()` serve their values out of the block. The four
  values `ReadData()` used to re-read are now the `_currentAntenna1` /
  `_currentAntenna2` / `_currentFieldId` / `_currentTime` that `NextRow()`
  already knows.
- `EndTimestep()` is computed on demand instead of in `Initialize()`. Only
  baseline-dependent averaging asks for it, and answering it means walking the
  whole `TIME` column.

29484 column reads become **27** `getColumnRange()` calls (three blocks x nine
columns) plus a `memcpy` per row - the same copy `ArrayColumn::get()` was
making anyway.

Two details are load-bearing:

- **The block is copied out, not referenced.** `NoiseMSRowProvider` mutates the
  `data` array it is handed in place; handing it a view into the block would
  corrupt the block. Copying is also exactly what upstream's
  `ArrayColumn::get(row, array, true)` did, so the patch removes casacore's
  per-call overhead and nothing else.
- **A set whose rows differ in shape cannot be read in blocks at all.**
  `getColumnRange()` throws on a multi-spectral-window set with different
  channel counts, so the first failure drops the block to a single row, which
  is upstream's behaviour. Forcing that path (`kMaxBlockRows = 1`) over the
  whole corpus gives 530 of 530 identical FITS data blocks and runs **7.8%
  slower** than the unpatched binary - the fallback is correct, and the block
  size is where the win comes from.

The block is capped at 1024 rows *and* at 4M values, so a wide set
(4 polarisations x 4096 channels) uses 256-row blocks rather than turning one
block into hundreds of megabytes. This parameter space's sets are 4 x 4 to
4 x 8, so they take the 1024-row cap.

## Measurements

Both arms' `wsclean-zygote` and `libwsclean.so` were `docker cp`'d out of their
images and driven from one container over the same 106-Measurement-Set corpus
on tmpfs, each replay using that evaluation's own `-scale`, alternating arm and
alternating which arm goes first.

| | n pairs | base median | new median | paired median ratio |
| --- | --- | --- | --- | --- |
| 0005 against HEAD | 1060 | 48.010 ms | 46.783 ms | **0.9781** |
| null (HEAD against a copy of itself) | 1060 | 48.026 ms | 47.911 ms | 0.9998 |

i.e. **-2.2% on the `wsclean` binary** uncontended.

The reorder phase itself, out of each run's own `-log-time` log over 424 paired
replays, with the untouched model-initialisation phase carried as an in-log
null:

| phase | base | 0005 |
| --- | --- | --- |
| reorder | 2.463 ms | **1.274 ms** |
| model init (null) | 0.050 ms | 0.055 ms |

**-48% of the reorder**, which is the 1.19 ms the replay reads as -2.2% of a
48 ms process.

At production concurrency, four swapped pairs of simultaneous 10-rank searches
(`WSCLEAN_IMAGE` selects each arm's build), bucketed into twelve equal-count
bins of `observation_minutes x channel_count` because two arms never sample the
same points, with `simulate_seconds` as the in-run null:

| pair | slot a | slot b | `image_binary` | `simulate` (null) | corrected |
| --- | --- | --- | --- | --- | --- |
| 1 | HEAD | 0005 | 0.9511 | 0.9971 | 0.9538 |
| 2 | 0005 | HEAD | 0.9673 | 1.0121 | 0.9558 |
| 3 | HEAD | 0005 | 0.9550 | 0.9924 | 0.9623 |
| 4 | 0005 | HEAD | 0.9816 | 0.9987 | 0.9828 |

Median **0.959**, i.e. **-4.1% on the `wsclean` binary**; averaging within each
swap direction first gives 0.958 and 0.969, mean 0.964. The binary is ~85% of
an evaluation, so that is ~+3.5% evaluations per second.

The real search reads a *larger* win than the uncontended replay (-4.1% against
-2.2%), which is the same way round as
[the shared-open patch](nested-sampling-shared-ms-open.md) and the opposite of
[the FFTW planner](nested-sampling-fftw-planner.md): what this patch removes is
casacore bookkeeping, and this host inflates that kind of work ~2.2x at
production concurrency while stretching the arithmetic around it less.

**Bit-identity**: 530 of 530 output FITS data blocks byte-identical across the
whole 106-Measurement-Set corpus (5 images each). Compare *data blocks*, not
files - the header records the `-name` path the run was given.

## What is left

Nothing on the read side. What remains of the reorder is the write side -
`FileWriter::WriteMetaRow` / `WriteDataRow` into the flat part files - which is
already a `memcpy` into a buffered stream.

The ranking from [the gridder-floor doc](nested-sampling-gridder-floor.md)
stands otherwise: 58% of an evaluation is ducc0's gridding and degridding
arithmetic, and its only levers are the accuracy and pass-count trades priced
there (`do_wgridding`, `-wgridder-accuracy`, `-mgain`).

## Reproducing

```sh
# a corpus of real Measurement Sets to replay against
./ri search wsclean --nlive 25 --num-repeats 2 --mpi-procs 14 --max-ndead 30 \
    --keep-measurement-sets --output-dir results/nested-sampling/corpus

# the column-cost probe: the build stage has casacore's headers, the runtime
# image does not
docker build --target build -f docker/wsclean/Dockerfile -t wsclean-build .
docker run --rm -v /tmp/probe:/w -w /w -e LD_LIBRARY_PATH=/opt/casacore/lib \
    wsclean-build bash -lc 'g++ -O2 -std=c++17 -I/opt/casacore/include \
    probe.cpp -o probe -L/opt/casacore/lib -lcasa_ms -lcasa_tables \
    -lcasa_casa -lcasa_measures && ./probe /ms/ms0'

# two builds, one container, interleaved - see the shared-open doc for the rig
docker tag ri-reproducibility/wsclean:v3.7 ri-reproducibility/wsclean:base
./ri build wsclean

# and at production concurrency
WSCLEAN_IMAGE=ri-reproducibility/wsclean:base ./ri search wsclean --nlive 60 \
    --num-repeats 2 --mpi-procs 10 --max-ndead 1000 --no-build \
    --output-dir results/nested-sampling/pair1a &
WSCLEAN_IMAGE=ri-reproducibility/wsclean:v3.7 ./ri search wsclean --nlive 60 \
    --num-repeats 2 --mpi-procs 10 --max-ndead 1000 --no-build \
    --output-dir results/nested-sampling/pair1b &
```

Note that `scripts/build.sh` hard-codes each image's tag, so `WSCLEAN_IMAGE=...
./ri build wsclean` *overwrites* `ri-reproducibility/wsclean:v3.7` rather than
tagging the build aside. Tag the baseline first, as above.
