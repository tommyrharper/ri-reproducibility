# Local WSClean patches

**WSClean asks the reordered Measurement Set provider for its antenna names
once per gridding and degridding pass, and every ask re-opens the parent
Measurement Set through casacore. Caching those names inside the process is
worth +10.5% evaluations per second end to end, -13.1% on the `wsclean` binary,
with bit-identical images.**

This is the other half of
[the Measurement Set open](nested-sampling-ms-open.md). That page closed the
avenue from the simulator's side: every subtable that could be deleted has
been, the ones that stay are load-bearing, their columns cannot be stripped,
and what is left is a ~1.9 ms floor per open paid ~16 times an evaluation. What
it did not do was ask why WSClean opens the Measurement Set 16 times in the
first place. It does not need to.

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## How a patch is carried

`docker/wsclean/patches/*.patch` is applied to the pinned upstream tree by
`docker/wsclean/Dockerfile`, between the `git clone` and `cmake`:

```
git -C /usr/src/wsclean apply --verbose /usr/src/wsclean-patches/*.patch
```

`git apply` is strict, so a `WSCLEAN_GIT_TAG` bump that moves the patched lines
fails the image build with the rejected hunk in the log. That is the intended
failure mode: a patch that silently stopped applying would be a silent
slowdown, and the whole point of the directory is that the number below stays
true. The patch directory is part of the `ri.build-inputs` hash in
`scripts/build.sh`, so editing a patch rebuilds the image and nothing else does.

Each patch is upstream-shaped - it should be something WSClean would take -
rather than a change to what WSClean computes. Both properties matter: a patch
that alters results makes every archived run incomparable, and a patch upstream
would reject is one this repo carries forever. 0002 is the one exception to the
first half and is called out as such below.

`docker/wsclean/src/*.cpp` is the companion directory: source this repo *adds*
to the tree, copied in by the same Dockerfile line, so a new file does not have
to be written as a diff. It is hashed into `ri.build-inputs` alongside the
patches.

## 0001: cache the reordered provider's antenna names

`MsProviderCollection::InitializeMeasurementSet()` runs once per gridding and
degridding pass and starts with

```cpp
ms_data.antenna_names = ms_provider.GetAntennaNames();
```

A fresh `ReorderedMsProvider` is constructed for each of those passes, and its
implementation is

```cpp
std::vector<std::string> ReorderedMsProvider::GetAntennaNames() {
  return wsclean::GetAntennaNames(MS()->antenna());
}
```

where `MS()` is `SynchronizedMS(handle_.data_->ms_path_.data())` - a fresh
`casacore::MeasurementSet`, which attaches every subtable. So each pass opens
the whole Measurement Set to read one column of the `ANTENNA` table, and in
this search's configuration the result is used only for its `.size()` (the
names themselves are read only on the H5Parm solution path, which is off).

The patch reads it once per Measurement Set path and hands out copies from a
mutex-guarded `std::map`. The `ANTENNA` table cannot change while WSClean holds
the reordered data, and each wsclean process here images exactly one
Measurement Set.

`GetObservationInfo()` and `Interval()` re-open the same way, but only under
`-apply-time-frequency-smearing`, which this search does not use. They are left
alone.

### What it removes

`wsclean -log-time` timestamps every output line, so the gap between
`Opening reordered part 0 for <ms>` and the next line measures the open
directly. One evaluation's real argv, serial, warm page cache:

| | logged span | sum of the 17 open gaps |
|---|---:|---:|
| stock v3.7 | 59.37 ms | 17.15 ms |
| patched | 49.77 ms | 3.76 ms |

The gaps do not go to zero because the first pass still does the one real open,
and the log line itself is emitted by the `ReorderedMsProvider` constructor,
which does its own (cheap) reads of the reordered temp files.

### Replay

79 Measurement Sets kept by `./ri search --keep-measurement-sets`, replayed
with each evaluation's own recorded argv, 19-way concurrent, six passes, the
first discarded. The two arms are two long-lived containers - one per image -
with the commands from all three arms shuffled into a single work queue, so
both arms see the same machine at the same time. The third arm is a second
container off the *baseline* image, which calibrates the floor.

| arm | n | mean | ratio to baseline |
|---|---:|---:|---:|
| baseline | 395 | 193.29 ms | - |
| patched | 395 | 159.29 ms | **0.8200** |
| null (baseline again) | 395 | 194.25 ms | 1.0054 |

### End to end

Two real searches started at the same instant, 10 ranks each (20 in total, one
per hardware thread), `--nlive 25 --num-repeats 10 --max-ndead 600`, arm
assignment swapped between the two pairs. Each arm's numbers are restricted to
the window in which both runs of the pair were live, because they do not finish
together and a run that outlives its partner gets a quieter machine.

| pair | arm | n | mean binary | evaluations/s |
|---|---|---:|---:|---:|
| 1 | baseline | 4968 | 230.27 ms | 34.71 |
| 1 | patched | 5374 | 204.91 ms | 37.55 |
| 2 | patched | 5322 | 197.49 ms | 38.78 |
| 2 | baseline | 4717 | 232.75 ms | 34.37 |
| null | baseline | 5074 | 236.33 ms | 33.86 |
| null | baseline | 4930 | 237.61 ms | 32.90 |

Geometric mean over the two swapped pairs: **0.8691 on `image_binary_seconds`**
and **1.1048 on evaluations per second**. The null pair puts the floor at 0.5%
on the binary column and 2.8% on evaluations per second, which is the usual
ordering - the throughput column also carries the sampler's own variance.

### Images

Every FITS file of two full replay passes - 790 files, five per evaluation -
compared by data block (the header carries the command line, so whole-file
checksums differ between directories). All 790 identical. The patch changes
what WSClean reads, not what it computes.

### Why the replay overstates it and the search understates it

-18.0% on the replay corpus against -13.1% in a real search is the opposite
skew to the one [the subtable drop](nested-sampling-ms-open.md) showed, and the
cause is where the Measurement Set lives. A replayed corpus sits on ext4 under
`results/`; a scored evaluation's `sim.ms` is built in the shared tmpfs
(`NS_SCRATCH_DIR`, see [I/O placement](nested-sampling-io-placement.md)) and
never reaches the disk. A casacore open is metadata- and syscall-bound, so it
is cheaper on tmpfs, and removing it therefore buys less in the run than on the
corpus. Take the end-to-end number; use the replay only to confirm the sign and
the images.

## Reproducing it

Keep a corpus, then replay it against two images:

```sh
./ri search wsclean --nlive 25 --num-repeats 5 --max-ndead 600 --mpi-procs 8 \
  --keep-measurement-sets --output-dir results/nested-sampling/corpus
docker tag ri-reproducibility/wsclean:v3.7 wsclean-baseline:local   # before editing a patch
scripts/build.sh wsclean                                           # ~11 minutes
```

`WSCLEAN_IMAGE` is read by `scripts/run-nested-sampling.sh` and reaches
`polychord_wsclean.py`, so two searches can run simultaneously against two
different `wsclean` builds - which is the only honest way to A/B a binary on
this host (see the false-positive rate of sequential arms in
[I/O placement](nested-sampling-io-placement.md)).

Note that `NS_MAX_NDEAD` defaults to 12 in `defaults.toml` - a smoke-test
value. A benchmark run that forgets `--max-ndead` finishes in four seconds and
measures the process start of ten MPI ranks.

## 0002: build `wsclean-zygote`

Two `CMakeLists.txt` hunks - an `add_executable`/`target_link_libraries` pair
and an `install` line - for `main/zygote.cpp`, which
`docker/wsclean/src/zygote.cpp` supplies. The binary is a fork server: it links
the same `wsclean-lib`, runs the static initialisers once, and forks an
already-initialised child per imaging request, which is worth +8.4%
evaluations per second end to end with bit-identical images. What it is, why
27ms of every `wsclean` process runs before `main()` does, and the measurements
are in [the zygote doc](nested-sampling-wsclean-zygote.md).

This is the patch that is not upstream-shaped: WSClean has no reason to ship a
fork server, and the code it builds is this repo's rather than a change to
WSClean's. It keeps the property that actually protects the archive - it alters
nothing WSClean computes - and it is deliberately as small as a patch can be,
so that a `WSCLEAN_GIT_TAG` bump has almost no surface to break against.

## 0003: open the parent Measurement Set once

`OpenMeasurementSet()` in `msproviders/msprovider.cpp` - a mutex, a map from
path to an open `casacore::MeasurementSet`, and a returned copy of the
reference-counted handle - replacing the four `casacore::MeasurementSet
ms(path)` sites a run reaches before its first visibility. That is 21
`table.dat` opens down to 4, worth -3.7% on the `wsclean` binary in an
interleaved replay and -4.4%/-6.1% in two simultaneous swapped searches, with
bit-identical images. The counting method, the per-open costs, the two
load-bearing details of the cache and the measurements are in
[the shared-open doc](nested-sampling-shared-ms-open.md).

Upstream-shaped: it changes nothing WSClean computes, and it is the same shape
as 0001 one level up - cache what does not change for the life of a process
instead of re-deriving it.

## 0004: cache the FFTW transform plans

`CachedPlan1D()` and `CachedPlan2D()` in
`external/schaapcommon/src/math/convolution.cc` - a `std::map` from transform
shape to plan behind a mutex - replacing the four plans
`schaapcommon::math::Convolve()` built and destroyed on every call and the two
`Resampler` built and destroyed per instance. That is 64 plan builds an
evaluation down to 12 and 64 destroys down to 0, worth -2.7% on the `wsclean`
binary in an interleaved replay and -3.2% over four simultaneous swapped
searches, with bit-identical images. Why the plans are safe to share, why the
cache is leaked, and the measurements are in
[the gridder-floor doc](nested-sampling-gridder-floor.md).

The first patch here that touches the `schaapcommon` submodule rather than
WSClean itself, so its paths are `external/schaapcommon/...` and it is the one
most likely to need regenerating against a `WSCLEAN_GIT_TAG` bump. Otherwise
the same shape as 0001 and 0003: cache what does not change for the life of a
process.
