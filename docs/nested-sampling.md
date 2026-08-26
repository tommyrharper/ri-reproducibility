# Nested Sampling PoC

This repo uses PolyChord as a targeted search tool, not as a Bayesian posterior
fit. PolyChord maximizes a configurable objective metric (default
`off_source_rms_jy`).

Ground truth for every run is one unpolarized 1 Jy point source at phase centre.
Dynamic range is controlled by complex Gaussian thermal noise in the simulated
visibilities.

## Images

```bash
make build-wsclean
make build-r2d2
make build-meqtrees
make build-polychord
```

`make build` also builds the MeqTrees and PolyChord images.

The MeqTrees image uses KERN 10 packages on Ubuntu 24.04. The VLA.A antenna table
is unpacked from makems' bundled `VLAA_ANT` example inside the image, so antenna
positions are not hand-rolled in this repo. Visibilities for that skeleton are
predicted by an actual MeqTrees/Meow point-source RIME run
(`scripts/lib/nested_sampling/point_source_forest.py`, driven through
`meqtree-pipeliner.py`), not a hand-rolled formula; thermal noise is added on
top of that clean MeqTrees prediction.

## Run the PoC

Both PoCs share the same `NS_*` and `OUTPUT_DIR` overrides (see WSClean below).
Each target builds its required images first and starts one long-lived sidecar
container per image; the PolyChord container mounts the Docker socket and
drives those sidecars. The WSClean target starts the PolyChord container the
same way and `docker exec`s the run into it (see "The PolyChord container is a
sidecar too" and "Long-lived sidecar containers, one per image" below).

### WSClean

```bash
make nested-sampling-poc
```

Outputs:

```text
results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>/
```

Useful overrides:

```bash
NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 make nested-sampling-poc
NS_MPI_PROCS=4 make nested-sampling-poc
NS_METRIC=badness make nested-sampling-poc
NS_METRIC=snr make nested-sampling-poc
NS_METRIC=total_rms_jy make nested-sampling-r2d2-poc
NS_METRIC=sigma_res make nested-sampling-r2d2-poc
OUTPUT_DIR=results/nested-sampling-poc/manual make nested-sampling-poc
```

PolyChord likelihood evaluations run in parallel across MPI ranks inside the
PolyChord container. `NS_MPI_PROCS` sets the rank count (default
`min(NS_NLIVE, host CPUs)`). Set `NS_MPI_PROCS=1` to disable parallel
evaluations for debugging.

The target builds any missing WSClean, MeqTrees, and PolyChord images first.
Each likelihood evaluation runs one MeqTrees simulate and one WSClean imaging
step in this rank's already-running sidecar containers.

### R2D2

```bash
make nested-sampling-r2d2-poc
```

Outputs:

```text
results/nested-sampling-poc/r2d2-vlaa-<UTC timestamp>/
```

The target builds R2D2, MeqTrees, and PolyChord images first. Each likelihood
evaluation runs one MeqTrees simulate, one MeqTrees-hosted MS-to-`.mat`
conversion, and one R2D2 imaging container.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `make fetch-r2d2-checkpoints` and `make smoke-test-r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
scripts/check-ms-to-r2d2-mat.sh
```

`run-nested-sampling-r2d2-poc.sh` runs `NS_MPI_PROCS` PolyChord ranks
concurrently, each launching its own R2D2 container. Each R2D2 imaging
container is launched with OpenMP/BLAS thread env vars (`OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) set from the host's available CPU
count, overridable via `R2D2_OMP_THREADS`. The previous image default of
`OMP_NUM_THREADS=4` capped finufft/OpenMP work when the Docker VM exposed more
CPUs than four. To avoid CPU oversubscription, the script defaults
`R2D2_OMP_THREADS` to `host CPUs / NS_MPI_PROCS` (minimum `1`) when not set
explicitly, so each rank's R2D2 container gets a fair share of the host's cores
instead of all of them. Set `R2D2_OMP_THREADS` explicitly to override this
per-rank default.

## Parameter space

VLA configuration is an outer-loop dimension. This PoC only runs `VLA.A`.

PolyChord dimensions for both algorithm PoCs:

| Dimension | PoC range | Meaning |
|---|---:|---|
| `dynamic_range` | `1e2` to `1e3` | One-Jy source divided by thermal-noise sigma |
| `observation_minutes` | `4` to `10` | Total requested observing time |
| `channel_count` | `2` to `6` | Number of frequency channels |
| `start_frequency_hz` | `1.0e9` to `1.1e9` | First channel frequency |
| `channel_width_hz` | `0.5e6` to `2.0e6` | Uniform spacing between channels |

Channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

Fixed hyperparameters (not searched) on every evaluation:

**WSClean:** `-niter 100` and `-auto-threshold 3.0`, recorded in
`poc-summary.json` under `wsclean_fixed_hyperparameters`.

**R2D2:** `128x128` image size (matching the WSClean PoC's `-size 128 128
-scale 1asec` footprint), `num_iter 25`, `architecture unet`, `num_chans 64`,
`ckpt_path /checkpoints/R2D2_A1`, and `ckpt_realisations 1`, recorded in
`poc-summary.json` under `r2d2_fixed_hyperparameters`.

## MS to R2D2 `.mat` bridge

R2D2-RI reads visibilities from a MATLAB `.mat` file via `load_data_to_tensor()`
in the upstream `src/utils.py`. The nested-sampling simulator produces a CASA
Measurement Set (`sim.ms`) that WSClean consumes directly. The R2D2 PoC adds
`scripts/lib/nested_sampling/ms_to_r2d2_mat.py`, which runs inside the MeqTrees
image (python3-casacore plus scipy) and writes the minimal field set R2D2 loads
without flag metadata:

| Field | Meaning |
|---|---|
| `u`, `v` | UV coordinates in wavelengths, flattened across rows and channels |
| `y` | Complex visibilities for correlation index 0 (parallel-hand Stokes I) |
| `nW` | `sqrt(WEIGHT)` from the MS (sqrt of inverse variance) |

Imaging weights are generated inside R2D2 when `data_weighting: True` in the
per-evaluation YAML config. The converter does not replicate the bundled
`data_3c353.mat` pruning or tau-compressed weight fields.

## Metrics and objective

For each sample, the pipeline records:

| Metric | Source |
|---|---|
| `snr` | Reconstructed image peak divided by off-source RMS |
| `log_snr` | `log10(snr)` |
| `off_source_rms_jy` | Off-source RMS in Jy/beam |
| `total_rms_jy` | RMS of (reconstructed image − one-pixel truth) over all pixels |
| `peak_jy_per_beam` | Peak absolute flux in the reconstructed image |
| `relative_l2_error` | Image residual versus the one-pixel point-source truth |
| `peak_flux_abs_error_jy` | Absolute centre-pixel flux error |
| `sigma_res` | Paper data-fidelity \(\overline{\sigma}_{\textrm{res.}}=\|\widehat{\mathbf{r}}\|_2/\|\mathbf{x}_{\textrm{d}}\|_2\) (final residual dirty over dirty) |
| `wall_seconds` | Imaging container runtime |
| `peak_memory_bytes` | Peak imaging memory: GNU `time -v` for WSClean, sampled Docker stats for R2D2 |

PolyChord maximizes whatever value the run returns as its log-likelihood. The
default objective is `off_source_rms_jy` (off-source RMS in Jy/beam).

An optional composite `badness` score is also available (higher means worse
reconstruction or a more expensive run):

```text
max(0, 3 - log_snr)
+ min(relative_l2_error, 10)
+ 0.05 * min(wall_seconds / 60, 5)
+ 0.02 * min(peak_memory_bytes / 2 GiB, 5)
```

### Choosing the objective (`--metric` / `NS_METRIC`)

Both `polychord_wsclean_poc.py` and `polychord_r2d2_poc.py` accept
`--metric <value>` (default `off_source_rms_jy`). The shell wrappers forward
`NS_METRIC` with the same default. Resolution order:

1. `badness` - the composite formula above.
2. Any bare metric name from the table - use that raw value directly as the
   objective (including the default `off_source_rms_jy`).
3. Any other string - treat it as an arithmetic expression over the same metric
   names (for example `log_snr + 0.1 * wall_seconds`, or the composite formula
   rewritten by hand).

Expressions are compiled once at startup (before any Docker evaluations) and
evaluated in a restricted namespace: no Python builtins, metric names as locals,
and `math` module functions available by name. A typo or unsafe expression fails
immediately at startup.

PolyChord always maximizes the returned value with no automatic sign flip. The
`badness` composite is oriented so higher is worse. Raw metrics keep their
natural orientation: the default `off_source_rms_jy` search prefers higher
off-source RMS, `--metric snr` searches for the highest-SNR corner, and a
worst-SNR search must negate explicitly (`--metric "-snr"` or
`--metric "1/snr"`). `total_rms_jy` and `sigma_res` are also higher-is-worse
(noisier reconstruction / worse data fidelity); search for the best corner
with `--metric "-total_rms_jy"` or `--metric "-sigma_res"`. Failed simulations
or imaging runs still receive objective `100.0`.

Each evaluation record and `poc-summary.json` store the chosen value in an
`objective` field. `poc-summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

## Profiling: where does the wall time actually go?

Both `polychord_wsclean_poc.py` and `polychord_r2d2_poc.py` time every stage of
each likelihood evaluation with plain `time.perf_counter()` calls around the
existing subprocess/docker invocations already in `evaluate()` - no separate
profiling framework, no changes to the container images' entrypoints. Each
evaluation's `metrics.json` (and the aggregated `poc-summary.json`) gets a
`timing` block:

| Field | Meaning |
|---|---|
| `simulate_seconds` | Wall time for the MeqTrees `docker run` that produces `sim.ms` (container start + RIME simulation, not split further) |
| `convert_seconds` | R2D2 only: wall time for the MS -> `.mat` conversion `docker run` |
| `image_container_seconds` | Wall time for the imaging `docker run` round trip (WSClean or R2D2) |
| `image_binary_seconds` | WSClean only: the binary's own elapsed time from `/usr/bin/time -v` inside the container, i.e. excluding docker create/start/teardown |
| `metrics_seconds` | Wall time for `compute_image_metrics()` (FITS read + numpy) |

`image_container_overhead_seconds` (container round trip minus binary time) is
only available for WSClean, because only its image installs GNU `time`; R2D2
and MeqTrees containers report only the round-trip `docker run` time as one
blob.

`poc-summary.json` also gets a run-level `profiling` block: each field above
summed across every evaluation, plus:

- `accounted_worker_seconds` - sum of every stage total across all evaluated
  points.
- `accounted_seconds` - same value as `accounted_worker_seconds`, but emitted
  only for serial runs where `NS_MPI_PROCS=1`.
- `polychord_overhead_seconds` = `total_wall_seconds - accounted_seconds`.
  This is whatever PolyChord itself is doing outside likelihood calls (its own
  slice-sampling bookkeeping, live-point management, I/O to `chains/`). It is
  emitted only for serial runs where `NS_MPI_PROCS=1`; at higher MPI process
  counts, ranks run likelihood evaluations concurrently, so summed
  worker-seconds cannot be subtracted from rank-0 elapsed wall time.

### Running the profiler

The instrumentation runs automatically as part of every PoC run - there's no
separate flag. To read the breakdown of a completed run:

```bash
make nested-sampling-profile RUN=results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
# or directly:
uv run scripts/profile-nested-sampling-run.py results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
uv run scripts/profile-nested-sampling-run.py results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp> --json
```

`scripts/profile-nested-sampling-run.py` only reads `poc-summary.json` and
prints a table (or the raw `profiling` dict with `--json`); it does not launch
anything itself. Runs written before this instrumentation existed have no
`profiling` block and must be re-run to get one.

### What a real bounded run showed

A single-rank (`NS_MPI_PROCS=1`), 5-dimensional run at the default sampler
settings (`NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12`, 62 likelihood
evaluations, not committed) profiles as:

| Stage | Total | Share |
|---|---:|---:|
| WSClean image container (total) | 6.8s | 66.5% |
| &nbsp;&nbsp;of which: `wsclean` binary itself | 6.4s | 62.9% |
| &nbsp;&nbsp;of which: container overhead | 0.37s | 3.6% |
| MeqTrees simulate | 3.0s | 29.8% |
| Metrics computation | 0.12s | 1.1% |
| PolyChord overhead (unaccounted) | 0.26s | 2.5% |

Total wall time 10.2s (~0.16s/eval; ~1.8s on the default 8 ranks). That is
`poc-summary.json`'s `total_wall_seconds`, measured around `run_polychord()`
inside the PolyChord container - the end-to-end `time` of the run script is
~1.1s more on one rank and ~1.2s more on eight, for starting and removing the
containers (8-rank end to end is ~2.95s). No fixed overhead of any
size is left in either sidecar: what remains is the science.
Warm, an evaluation is ~0.05s of simulate (~0.022s RIME predict, ~0.007s
re-pointing the forest at the new MS, ~0.008s of casacore table I/O, ~0.004s to
copy a cached `makems` skeleton and ~0.005s to move the finished MS out of
`/dev/shm`; no evaluation runs `makems` any more, the image ships every
skeleton) and ~0.10s of `wsclean`, which is now well over half the run.
The rest is one-off startup - the simulate worker, meqserver and the one TDL
compile, now started concurrently before the sampler runs rather than serially
inside the first evaluation - plus PolyChord's own sampling and bookkeeping.

`wsclean` itself is at its floor for this problem size: it self-reports
0.035s inversion + 0.023s prediction + 0.008s deconvolution per evaluation
against ~0.018s of process startup, and moving `-temp-dir` to `/dev/shm`
changes nothing measurable (the reordered scratch files never reach the ext4
journal). `-j 4` buys ~5ms but would multiply threads by the MPI rank count
and make the gridding sum order - and so the image - non-deterministic.

#### The PolyChord ranks run with one BLAS thread each

`numpy`'s bundled OpenBLAS spawns one busy-waiting worker thread per host CPU,
in every process that touches it. Each MPI rank is one such process, so on this
20-CPU host the default 8 ranks carried ~160 spinning threads between them and
burnt ~10 cores waiting for work that never arrives - nothing in this pipeline
has a BLAS call big enough to want them, the largest being a norm over a
128x128 image. The starvation showed up as every stage inflating under MPI:
`metrics_seconds` averaged 0.28s per evaluation on 8 ranks against 0.005s on
one, for 1.5ms of actual arithmetic.

Both run scripts therefore pass `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` to the PolyChord container. Measured over 3 runs each,
the default 8-rank run went 8.25s -> 5.26s (-36%) and its summed metrics
computation 11.3s -> 1.3s. This is not worth pushing down into the sidecars:
the MeqTrees image's system `numpy` spawns no BLAS threads at all, and `wsclean`
already runs `-j 1`.

Single-rank is ~3% *slower* with the pin (10.95s -> 11.28s, consistent across
interleaved A/B runs) - with no contention to remove, the spinning threads were
keeping cores out of idle states. The default configuration is 8 ranks, so the
pin stays.

`log(Z)` and every evaluation's objective are unchanged. `sigma_res` moves in
its last bit, because a single-threaded `np.linalg.norm` reduces in a different
order than a threaded one; that also makes it reproducible across hosts with
different CPU counts, which it previously was not.

#### `WEIGHT`/`SIGMA` are written with one TaQL `UPDATE`, not `putcol`

`putcol` on these two columns was 42ms of the 81ms simulate - more than the
RIME predict. Both are *variable-shaped* array columns in the `ISMData`
`IncrementalStMan` group of a makems MS, and python-casacore's `putcol` on
that combination is quadratic in rows: at this MS size 100 rows cost 0.06ms,
500 rows 1.4ms and all 1755 rows 17ms per column. It is the storage manager,
not the disk - a `TiledColumnStMan` column of the same size (`DATA`, `FLAG`,
`UVW`) writes in 0.1ms, and an `IncrementalStMan` *scalar* column (`TIME`) in
0.05ms. `setmaxcachesize()` does not help, and the columns cannot be dropped
and re-added under another storage manager because the whole ISM group would
have to go with them.

A `putcell()` row loop is linear where `putcol` is quadratic - 7.9ms for the
pair - and that was the fix for a while: measured 13.1s -> 10.9s single-rank
(-16%, simulate 5.5s -> 3.4s) and 8.5s -> 8.2s on the default 8 ranks, with
every column of all 62 Measurement Sets, the artifact trees, all 62
evaluations' science metrics and `log(Z)` identical.

`fill_point_source_visibilities()` now does the same row loop in C++ instead,
as a single `taql("UPDATE $ms SET WEIGHT=..., SIGMA=...")`: 3.3ms for the pair
against 7.9ms for the Python loop and 43ms for `putcol`, best of four on a
1755-row MS, with both columns bit-identical to what the loop wrote. That is
~4.6ms off every evaluation's ~0.06s of simulate. The `removecols` route stays
closed: casacore refuses outright - `column WEIGHT cannot be removed from
table` - because makems put `WEIGHT` and `SIGMA` in one ISM group with 15 other
columns.

#### The compiled TDL forest is reused across evaluations

`Compile.compile_file()` was ~0.034s of every evaluation, and the forest it
builds does not depend on the Measurement Set's shape: the antenna layout and
phase centre come from the fixed antenna table and the hardcoded
`RightAscension`/`Declination` in `write_makems_config()`, and the time and
frequency axes are runtime data the `VisDataMux` reads per request. So
`run_meqtrees_predict()` keys a process-level cache on the generated `.tdlconf`
text with the `ms_sel.msname` line removed, and on a hit calls
`point_to_measurement_set()` - which re-points the `MSSelector` at the new MS -
instead of recompiling. Over a 62-evaluation run that is one compile and 61
reuses.

`MSSelector._select_new_ms()` re-lists the MS's data columns, which resets the
output-column option that `_define_forest()` set to `DATA`; the reuse path has
to re-assert it or the sinks quietly write `CORRECTED_DATA` and `DATA` comes
back all zeros with no error anywhere. `simulate_point_source_ms.py
--self-check` guards exactly that: it predicts three MS shapes off one cached
forest and off three fresh compiles and asserts the `DATA` columns are equal.

Measured over three runs each of the default single-rank configuration: 16.9s
before, 13.2s after (-22%), with the simulate stage down 39% (9.0s -> 5.5s -
more than the compile alone, because a reuse also skips the MS-metadata reads
the compile did). The 8-rank default went 9.1s to 8.4s (-8%; with ~5
evaluations per rank there is much less to amortise). All 62 images were
pixel-identical, every science metric matched, the evaluation directories held
the same file tree and `log(Z)` was unchanged.

#### The `makems` skeleton is cached per MS shape

`makems` is ~0.05s of every simulate and its output depends on the whole
`makems.cfg` except `StartFreq`/`StepFreq`, which move exactly six
`SPECTRAL_WINDOW` columns (`CHAN_FREQ`, `CHAN_WIDTH`, `EFFECTIVE_BW`,
`RESOLUTION`, `REF_FREQUENCY`, `TOTAL_BANDWIDTH`) and nothing else - verified
by comparing every column of every subtable across two frequency settings.
`make_ms_skeleton()` in `simulate_point_source_ms.py` therefore keys a cache on
the config text with those two lines removed, and on a hit copies the cached
skeleton inside `/dev/shm` (~0.002s) and rewrites those six columns instead of
running `makems`. Only `observation_minutes` and `channel_count` reach the key,
so the parameter space has 20 distinct shapes.

The cache is a directory in `/dev/shm`, not a dict in the worker process, and
that matters on the default 8 ranks: all eight `--serve` workers `docker exec`
into the same meqtrees sidecar, so they share `/dev/shm`, and a shape any one of
them has built is a `copytree` away for the other seven. A default run makes 41
evaluations over only ~12 distinct shapes, so per-process caches missed on most
of them - each rank sees ~5 evaluations and almost every one was a fresh
`makems`. Entries are staged in a scratch directory and `rename`d into place
under `sha256(key)`, so a concurrent worker either does not see an entry or sees
a complete one; losing that race is normal and the loser just drops its copy.

Measured over six interleaved A/B pairs of the default 8-rank run (rebuilding
the meqtrees image between arms): summed simulate worker-seconds 5.38s -> 4.68s
(-13%, 6/6 pairs) and end to end 3.65s -> 3.38s (-7%, 5/6 pairs). All 41
evaluations matched on every science metric and `log(Z)` was bit-identical.

##### The image ships every skeleton, so no run calls `makems`

Waiting for an evaluation to miss puts the ~0.11s of a fresh `makems` in the
middle of the sampler's critical path. Building the shapes in the workers'
background threads once the run had started only half-fixed that: a default
8-rank run still took ~7 misses (0.06-0.10s each, in-worker timings) because
the sampler asks for its first evaluations well before ~3 shapes per worker
have been built, and all eight workers additionally raced on the same fresh
`makems` inside `warm_forest()`.

The parameter space only has 20 shapes and they cost ~1.2s and ~18MB to build,
so the meqtrees image builds all of them at `docker build` time -
`simulate_point_source_ms.py --prebuild-skeletons` into `/opt/ms-skeletons` -
and `skeleton_dir()` prefers that directory when it exists. It is an ordinary
writable container path, so a shape the image was not built with is still built
and published there at runtime: the baked set is a head start, not a fixed set.
`poc_common.py` is `--mount=type=bind`ed for that one build step rather than
copied, so the shapes come from the single authoritative `PARAMETER_SPACE` and
the runtime image still carries only the three simulate-side scripts.

The MS name is part of the cache key, so a prebuilt entry is only useful if it
is built under the name a real evaluation uses (`sim.ms`).
`self_check_skeleton_prebuild()` is the guard: it prebuilds a two-by-two slice
of shapes and then asserts a real `make_ms_skeleton()` call for one of them
reports a cache hit. It fails if the prebuild builds under any other name.

Measured over 40 interleaved A/B pairs of the default 8-rank run against
pre-tagged `:ab-old`/`:ab-new` images (which is also the TaQL `WEIGHT`/`SIGMA`
change above): end to end 3.148s -> 2.955s (-6.1%, -0.193s +/- 0.029s, t = 6.7,
33/40 pairs), summed simulate worker-seconds 2.73s -> 1.89s (-31%), and zero
skeleton cache misses against seven before. All 41 evaluations' objectives were
bit-identical.

Because the workers still build into the cache directory when they miss, and
because `/dev/shm` is where an unbaked image's cache lives, the sidecars keep
`--shm-size 512m`; docker's 64MB default is only about 3x what 20 skeletons
need.

`simulate_point_source_ms.py --self-check` is the guard on the rewrite formula
(and on the forest reuse below): it builds each shape both ways and asserts a
patched cache hit matches a fresh `makems` run column for column. Run it in the
meqtrees image:

```bash
docker run --rm --network none ri-reproducibility/meqtrees:kern-10 --self-check
```

Measured over three runs each of the default single-rank configuration above:
19.8s before, 16.9s after (-15%), with the simulate stage down 26% (12.2s ->
9.0s). All 62 images were pixel-identical, every science metric matched, the
evaluation directories held the same file tree, and log(Z) was unchanged.
`copytree(..., symlinks=True)` matters here: `makems` leaves `vis.DATA`,
`vis.uvw` and `vis.flg` as symlinks into the tiled storage manager files, and
copying them as regular files leaves stale duplicates of the visibilities in
every evaluation directory.

#### Sidecar commands go through one long-lived `sh` per rank

`docker exec` costs ~0.033s on this host, a third of the `wsclean` binary's own
~0.107s, and every evaluation paid it again. `sidecar_shell()` in
`poc_common.py` therefore `docker exec -i`s a single `sh` into the rank's
sidecar on first use, and `sidecar_run()` sends each later evaluation one
command line - `cd <eval_dir> && <cmd> >stdout.log 2>stderr.log; echo $?` - and
reads the exit code back. Arguments are `shlex.quote`d, the command's own output
goes to the log files, so nothing a sidecar prints can be mistaken for a reply,
and a shell that dies without answering is dropped from the cache the same way
the simulate worker is.

A round trip costs ~0.0003s against ~0.033s for `docker exec`, taking WSClean
container overhead from 0.78s to 0.18s over 19 evaluations and the profiled run
from 7.88s to 7.04s (medians of three runs each, -10.6%). A 4-rank
54-evaluation run went 14.0s to 13.4s. Metrics, `log(Z)` and the reconstructed
FITS images are pixel-identical; only the recorded `commands.wsclean` changes,
from the `docker exec` argv to the in-container command it wrapped.

#### The simulate sidecar is a long-lived worker process

Even inside a reused sidecar container, a per-evaluation `docker exec` of
`simulate_point_source_ms.py` spent ~0.45s of its ~0.7s on startup the next
evaluation would immediately repeat: 0.10s of Python plus numpy/casacore
imports, 0.14s of Timba imports, 0.04s starting a meqserver and ~0.10s reaping
it again, against ~0.14s of actual RIME predict and ~0.05s of `makems`.

`simulate_point_source_ms.py --serve` therefore reads one JSON request per
stdin line - `{"argv": [...], "stdout": path, "stderr": path}` - and replies
with `{"returncode": int}` on its original stdout, with fds 1 and 2 pointed at
the request's log files for the duration so `makems` and the meqserver still log
per-evaluation exactly as they did when each was its own process.
`simulate_worker()` in `poc_common.py` starts one such process per rank on first
use and writes to its stdin from then on; a worker that dies without replying is
dropped from the cache so the next evaluation starts a fresh one instead of
inheriting the corpse.

The predict itself moved in-process with it: `run_meqtrees_predict()` now runs
the same `TDLOptions`/`Compile.compile_file`/job sequence `meqtree-pipeliner.py`
runs, against a meqserver that survives between requests. `mqs.get_error_log()`
flushes, so each request only ever sees its own errors. Those errors are
printed with `!r`, not `str()`: Timba's DMI record `__str__` is still py2
(`string.join`) and raises `AttributeError`, which used to replace the
meqserver's actual error with a traceback from the error-reporting path itself.
`__repr__` on the same class is py3-clean.

Measured cost per simulate dropped from 0.62s one-shot to ~0.18s served. On the
profiled single-rank run that is 16.8s to 7.7s total (-54%), and on a 4-rank
54-evaluation run 25.2s to 13.7s (-45%), with identical science metrics,
identical `log(Z)` and identical per-evaluation artifact file sets. Only
`wall_seconds` and `peak_memory_bytes` - the WSClean timing metrics - differ,
as they do between any two runs.

The worker starts its meqserver before it reads its first request. It used to
be started lazily, inside `meqserver_session()`, which meant the ~0.3s of Timba
imports plus server startup was paid *inside* evaluation one on every rank -
and since PolyChord asks all ranks for their initial live points at once, all of
it landed on the wall clock. Nothing has been asked of a freshly spawned worker,
so `serve()` pays it up front instead, concurrently with the rank's own
PolyChord import and setup. Interleaved A/B over 9 pairs (rebuilding the
`meqtrees` image between arms) put the default 8-rank run at 4.52s before and
4.11s after, -9%, with B faster in all 9 pairs and bit-identical `log(Z)` and
per-evaluation objectives.

That warm-up runs under `redirect_fds(os.devnull)`: Timba prints to fd 1 while
it starts (`Qt not available, substituting proxy types for QObject` and
friends), and fd 1 is the reply pipe, so without the redirect the first
`{"returncode": ...}` line arrives behind three lines of chatter and
`json.loads` fails. `--self-check` covers exactly that: it runs a worker,
sends one deliberately invalid request and asserts its stdout is a single JSON
line - verified to fail when the redirect is removed.

##### The workers are started by the container, not by the ranks

Starting the meqserver eagerly still leaves the worker unable to answer for
~0.5s after it is launched, measured across eight launched at once: ~0.11s of
`docker exec`, ~0.07s of interpreter and imports, ~0.11s of meqserver, and the
rest the first TDL compile and first predict (a fresh worker's first simulate
is ~0.17s against ~0.03s for its second, even against a warm skeleton cache).
No amount of eager work *inside* the worker can hide that, because the rank
that launched it asks for its first evaluation ~0.2s later.

So the ranks no longer launch them. `run-nested-sampling-poc.sh` creates one
`<rank>.in`/`<rank>.out` FIFO pair per rank under
`<output-dir>/.simulate-workers`, and the meqtrees sidecar's *container
command* - `sidecar_launch ... -- sh -c ...`, in place of the default `sleep
infinity` - spawns one `simulate_point_source_ms.py --serve --fifo <base>` per
pair. `poc_common._connect_shell_started_worker()` opens that pair instead of
spawning anything, and `FifoWorker` presents the same
`.stdin`/`.stdout`/`.terminate()` surface as the `subprocess.Popen` it replaces,
so nothing downstream changed. The FIFOs reach across containers because the
PolyChord container and the meqtrees sidecar both bind-mount `REPO_ROOT`: a FIFO
on a bind mount is one host inode both of them open.

It has to be the container's command rather than a `docker exec` into it. A
`docker exec` cannot be issued until `docker run` has returned, which is ~0.02s
after the container's own command has already started and ~0.1s before the
exec's process does; an earlier version that retried `docker exec --detach`
into the container as soon as it would accept one measured -0.06s end to end
over 14 pairs on a ~2.95s baseline, against -0.15s for this one. Under `--fifo` the worker also
compiles the forest and runs one throwaway predict before it opens its request
pipe (`warm_forest()`); on the stdin path it deliberately does not, because
there the rank that started it is already waiting.

The price is that `docker info` moves back in front of the sidecar launches
(~0.06s of serial delay, undone from an earlier iteration) because the FIFOs
have to exist before the container's command globs for them.

Measured on the default 8-rank run: the eight `eval_id == 1` `simulate_seconds`
records go from 0.18-0.41s (median ~0.33s) to 0.05-0.11s against a ~0.05s
steady state, `total_wall_seconds` from ~2.4s to ~1.8s, and end to end 3.40s to
3.25s - -4.4%, 20 of 24 interleaved pairs, sd of the paired difference 0.18s -
with bit-identical `.stats`, `.txt` and `_dead-birth.txt` chains and identical
per-evaluation params and metrics.

Two sharp edges. Opening a FIFO blocks until the other end opens, so both sides
must open the request pipe first and the reply pipe second; reverse either and
the run hangs with no error at all. `--self-check`'s `self_check_serve_fifo()`
is the guard, verified to fail (rather than hang) when `serve()`'s two opens are
swapped. And the rank's side opens with `O_NONBLOCK`, which is how a FIFO
write-open reports "no reader yet" (`ENXIO`) instead of blocking forever: it
retries for 10s and then falls back to starting its own worker, so a missing or
broken pool costs latency, not the run. That fallback is also what the R2D2 PoC
uses - it does not set `NS_SIMULATE_FIFO_DIR` at all.

One sharp edge: Timba registers `stop_default_mqs()` with `atexit`, but CPython
joins non-daemon threads - including octopussy's event thread, which only exits
once the server is stopped - *before* it runs `atexit` handlers, so a process
that leaves meqserver teardown to `atexit` finishes all its work and then hangs
at exit forever. `meqtree-pipeliner.py` avoids that by calling
`stop_default_mqs()` explicitly, and so does `stop_meqserver_session()` here.

#### The Measurement Set is built in tmpfs

`makems` and casacore `fsync` on nearly every table write, so on the
bind-mounted repo the simulate sidecar spends most of its wall time parked in
`jbd2_log_wait_commit` waiting for ext4 journal commits - sampling
`/proc/<pid>/wchan` during a `makems` run put it in journal wait for ~75% of
the samples. The cost is entirely fixed overhead, not data volume: `makems`
takes 0.54s for a 1-time-sample, 1-channel MS and 0.55s for the largest MS this
parameter space produces, but only 0.046s when the same run happens on tmpfs.

`simulate_point_source_ms.py`'s `main()` therefore builds everything -
`makems.cfg`, the unpacked `VLAA_ANT` table, the MS, the MeqTrees predict and
the noise fill - inside a `tempfile.TemporaryDirectory(dir="/dev/shm")`, then
moves the finished directory contents to the real output path in one go. The
whole MS is ~1MB, so the copy out is ~2ms, and every artifact a run used to
leave in the evaluation directory (including `makems.log`,
`meqtree-pipeliner.log` and `point_source_forest.tdlconf`) still lands there -
verified by `find`-diffing evaluation directories before and after.

Measured per-simulate cost dropped from 1.12s to 0.55s standalone, and on the
profiled run from 23.7s to 13.3s of simulate (27.5s to 16.9s total, -38%) with
identical per-evaluation metrics and identical `log(Z)`. Under 8-way MPI the
win is larger still - the ranks were contending for the same journal - taking
an 8-rank 41-evaluation run from 19.9s to 13.8s.

Docker gives a container 64MB of `/dev/shm` by default, which is ~30x the
largest MS this parameter space produces; a bigger parameter space needs
`--shm-size` on the sidecar. `SCRATCH_ROOT` falls back to the `tempfile`
default when `/dev/shm` is not writable.

#### Each rank warms its sidecar attachments before the sampler starts

The first evaluation on a rank used to cost ~0.7s that later ones did not, and
every rank paid it at the same moment - PolyChord asks all `nlive` initial live
points at once - so the whole thing landed on the wall clock in front of
evaluation one. It was four independent startups run one after the other inside
`evaluate()`:

| Startup | Cost |
|---|---:|
| `simulate_point_source_ms.py --serve` worker (`docker exec`, Python, Timba, meqserver) | ~0.45s, now started eagerly by the worker itself |
| `astropy.io.fits` import, on the first metrics call | ~0.45s on 8 ranks, now gone - see below |
| `docker inspect` of each image's `ENTRYPOINT`, twice | ~0.05s |
| the WSClean sidecar's `sh` (`docker exec`) | ~0.03s |

`poc_common.prewarm()` starts all of them in threads, so a rank pays the slowest
instead of the sum. `main()` calls it before `import pypolychord` and joins it
immediately before `run_polychord()`, so the remainder also overlaps the
sampler's own import and setup. Nothing may touch a sidecar between the call and
the join: `_SIMULATE_WORKERS`, `_SIDECAR_SHELLS` and `_IMAGE_ENTRYPOINTS` are
plain dicts with no lock, and a lazy start racing the prewarm thread would leave
a second, orphaned worker.

**What is still left in evaluation one.** Per-evaluation `simulate_seconds`
from `poc-summary.json` used to show the eight `eval_id == 1` records (one per
rank, all issued at the same moment) at 0.18-0.41s against a ~0.05s median for
the rest of the run - the `--serve` worker's startup, finishing inside the first
request because the rank that started it had nothing else to do meanwhile. The
run script now starts those workers as the meqtrees container's own command
(see above) and the same records read 0.05-0.11s, so this is no longer the
largest item on the critical path.

What replaced it is the prewarm join itself, and two-sided timestamps have since
shown that the join is the whole critical path - see "The critical path is the
simulate worker's warm-up, not the rank" below. Closing it needs the worker to
be *ready* sooner, not started sooner: the remaining ~0.4s is ~0.07s of
interpreter and imports, ~0.10s of meqserver and ~0.25s of first compile and
first predict. Baking a ready-made Measurement Set into the image to skip the
warm-up's `makems` was measured and moves worker-ready time by nothing.

#### FITS images are read without astropy

`from astropy.io import fits` was the single largest per-rank startup left, and
it dominated the prewarm join: ~0.45s when the 8 default ranks import it at
once, against ~0.07s for both sidecar startups put together. Instrumenting the
prewarm threads is what showed it - the other two only `Popen`, so they return
in milliseconds and the join was pure astropy.

All it was doing is reading a single-HDU, uncompressed, `BITPIX = -32` image and
two header cards (`CRPIX1`/`CRPIX2`). `poc_common.load_fits_2d()` now does that
directly: 2880-byte header blocks of 80-column cards, then big-endian samples in
C order. Anything outside that shape - an integer or `BSCALE`/`BZERO`-scaled
image, a short data block - raises instead of being guessed at. astropy is still
installed in the image and still used by the self-check.

The trap is card parsing, not the data block: a quoted value may contain the `/`
that otherwise starts the comment (`BUNIT = 'JY/BEAM '`), so `_fits_card_value()`
closes the quote before cutting the comment. `self_check_fits_reader()` (run by
`POLYCHORD_WSCLEAN_POC_SELF_CHECK=1`) writes exactly that card with astropy and
asserts the reader agrees; it fails if the quote handling is removed.

Verified against astropy on all 16833 FITS files this repo's results tree
contains - identical pixels and identical values for every non-comment header
card. Eight interleaved A/B pairs (rebuild between arms) gave 3.91s -> 3.61s end
to end on the default 8 ranks, -7.8%, 8/8 pairs, with a bit-identical
`chains/wsclean_vlaa_poc.stats` and bit-identical metrics for all 41
evaluations.

#### Sidecar teardown does not block the run

The EXIT trap's `docker rm --force` of the three containers costs ~0.4s, spent
after every result is already on disk. It is now backgrounded (`... &`); the
orphaned `docker rm` outlives the shell and finishes. On a `SIGKILL` of the run
script the containers survive as before, and
`docker rm -f $(docker ps -q --filter name=ri-ns-sidecar-)` still clears them.

Measured together with four interleaved A/B runs of the default 8-rank
configuration, end-to-end script wall time went 5.15s -> 4.42s (-14%), split
roughly 0.5s to the prewarm and 0.2s to the backgrounded teardown.
`total_wall_seconds` went 3.6s -> 2.8s, but only part of that is real: the
prewarm happens *before* `run_start`, so it moves cost out of that window as
well as shrinking it. All eight runs produced identical `log(Z)` and identical
objectives for all 41 evaluations.

#### The PolyChord container is a sidecar too

The run scripts do not `docker run` the PolyChord container either: they start
it detached alongside the two data-plane sidecars and enter it with `docker
exec mpirun ...`. A `docker run` of this image costs ~0.7s of create, start,
wait and `--rm` teardown; the `docker exec` costs ~0.03s, and the container's
own startup now happens concurrently with the MeqTrees and WSClean ones.

One more thing rides on that: **the manifest write moved into the gap.**
`scripts/record-environment.sh` is ~0.4s of `git` and `docker image inspect`,
and now runs between `sidecar_launch` and `sidecar_wait` instead of after the
containers are up. `NS_SIDECARS` is exported by `sidecar_launch` rather than
`sidecar_wait` because the container names are known as soon as the launches
are issued.

The `docker info` that resolves `HOST_CPUS` (and doubles as the
daemon-availability check) moved *after* the launches for the same reason -
nothing between the launches and `sidecar_wait` touches a sidecar - which put
`launches-issued` at ~0.005s after script start instead of ~0.075s. The WSClean
script has since moved it back in front of them, because the FIFO pairs the
meqtrees container's command globs for have to exist first (see "The workers are
started by the container, not by the ranks" above).

Measured with four interleaved A/B runs of the default 8-rank configuration,
end-to-end script wall time went 6.82s -> 5.29s (-22%); single-rank went 13.1s
-> 12.2s (-7%). All eight runs produced identical `log(Z)` and byte-identical
objectives for all 41 evaluations.

Note which clock that is. `poc-summary.json`'s `total_wall_seconds` - the
number the profile table above totals - is measured around
`run_polychord()` *inside* the container, so it does not see container startup
or teardown at all: it is 3.65s before and after this change. Only
`time scripts/run-nested-sampling-poc.sh` shows it. Anything that moves fixed
setup cost has to be measured end to end.

Only the WSClean run script does this. The R2D2 one still uses
`start_sidecars`, which is now a wrapper over `sidecar_launch` + `sidecar_wait`
and behaves exactly as before; ~0.7s is not worth re-validating a PoC whose
runs take 20 minutes.

#### The images ship byte-compiled, so no container needs a warm-up

`sidecar_launch` used to run a throwaway `python3 -c "import numpy,
pypolychord, poc_common, argparse; argparse.ArgumentParser()"` in the fresh
PolyChord container before `sidecar_wait` returned, because the *first* Python
process in a container cost far more than the next one: the real 8-rank
`mpirun python3` exec measured 0.99s cold against 0.22s warm.

That was byte compilation, not the overlay mount. `python:3.11-slim` ships no
`.pyc` for most of the stdlib, and pip left `/opt/venv` with `.pyc` whose
recorded source mtimes no longer match after the image's `COPY`. `docker diff`
on a container that has done nothing but one import shows 60 freshly written
`.pyc` files; eight ranks starting at once each compile the same modules and
race to write them. Reading every file in `/opt/venv` first (`cat` over the
`.so` and `.pyc` files, 0.12s) leaves the 8-rank exec at 0.91s, so it is not
page cache; a `python3 -c pass` first does not help either.

So both images now run `python3 -m compileall` at build time
(`docker/polychord/Dockerfile`, `docker/meqtrees/Dockerfile`). A fresh
container's 8-rank cold import is then 0.22-0.26s with no warm-up at all and
writes zero `.pyc`, and `sidecar_launch` lost its warm-up hook entirely -
`sidecar_wait` now only waits for `docker run`, which the manifest write
already hides. Shell markers put `sidecars-ready` at ~0.47s after script start
instead of ~0.70s.

The MeqTrees image only wrote two `.pyc` (Ubuntu ships them for its own
`python3` packages, and one of the two is the `Timba` module the Dockerfile
patches), so `compileall` there is cheap insurance rather than a measured win.

Ten interleaved A/B pairs of the default 8-rank run, rebuilding both images
between arms: 3.31s -> 3.11s end to end (-6%), 9/10 pairs in the right
direction. The 41 evaluations' parameter sets, science metrics and objectives
and the PolyChord `.stats` output are identical; only `eval_id` ordering and
the per-evaluation `wall_seconds`/`peak_memory_bytes` differ, as they do
between any two runs.

Also measured and rejected: warming the MeqTrees and WSClean sidecars is not
worth it even before `compileall` (cold-vs-warm first exec 0.17s vs 0.13s and
0.09s vs 0.08s, and warming the Timba/casacore imports moved eight concurrent
`--serve` worker startups only 0.34s -> 0.32s for 0.22s of warm-up); warming
with `mpirun -np 8` cost 1.0s instead of 0.28s with no gain in the real exec.

Also measured and rejected: **`wsclean -j 2` and above.** The imaging binary is
the largest per-evaluation stage (~0.11s of a ~0.16s warm evaluation) and
`-j 4` runs it in ~0.09s on an idle host, but multi-threaded gridding changes
the summation order: the image, dirty, residual and PSF pixels differ from the
`-j 1` run at the float32 rounding level (~1e-7 on a peak of 1.0). The
objective is `off_source_rms_jy` ~ 8e-6, so a 1e-7 pixel shift moves the
likelihood and takes the whole sampler down a different path. This repository
is about reproducibility; `-j 1` stays.

#### `mpi_rank()` reads the launcher's environment first

`from mpi4py import MPI` initialises MPI, and eight ranks doing that at once
costs 0.24s each. `prewarm()` needs the rank before anything else has touched
MPI - it is how a rank finds its own FIFO pair - so calling
`mpi_rank()` there added that 0.24s to every rank's pre-sampler startup, where
it hid from `total_wall_seconds` (measured around `run_polychord()`) and showed
up only end to end. `mpi_rank()` therefore reads `OMPI_COMM_WORLD_RANK`, which
OpenMPI's launcher exports, and only falls back to `mpi4py`.

#### MPI picks its transport by search unless it is told not to

`from mpi4py import MPI` runs `MPI_Init`, and on this host that cost 0.25s on
every one of the eight ranks at the same moment. Nothing in this repo imports it
on the sampler path any more (see above), so where it lands is inside PolyChord:
`pypolychord` imports `mpi4py` lazily, and the whole 0.25s therefore shows up as
the first `run_polychord()` call taking that long to reach its first likelihood,
with a second call in the same process reaching it in ~0.3ms. `strace` on the
first call is what identifies it - 2796 `openat`s and 903 `clock_nanosleep`s
worth 0.17s, none of them PolyChord's. It is not the transport itself:
`OMPI_MCA_pml=ob1` takes the slowest rank's `MPI_Init` to 0.05s, and
`OMPI_MCA_pml=^ucx,cm` - the image's `/etc/openmpi/openmpi-mca-params.conf`
already excludes `ucx` - to 0.046s, so the 0.19s is Open MPI opening the `cm`
PML, which opens the MTL framework, which has libfabric probe every provider it
can find. This job never leaves one container, and `ob1` over shared memory is
what the search settles on anyway, so both PoC run scripts name it:

```
docker exec ... -e OMPI_MCA_pml=ob1 ...
```

Adding `-e OMPI_MCA_btl=self,vader` on top buys a further ~0.007s and was left
out. Output is bit-identical: the same 41 evaluations with the same metrics, and
`.stats`, `.txt`, `_dead-birth.txt`, `_phys_live.txt` and `_equal_weights.txt`
all compare equal against a run without it.

Forty interleaved A/B pairs: 3.205s -> 3.068s median, paired difference
-0.136s +/- 0.029s, 29/40 pairs. Instrumented runs put the rank's
`run_polychord()`-to-first-likelihood gap at 0.234-0.266s without the setting
and 0.022-0.053s with it, 6/6 pairs; end to end recovers rather less than that
because the ranks reach the join spread over ~0.2s.

An earlier 24-pair A/B of this same change read -0.007s and nearly sent it to
the bin. Both arms had been run against a meqtrees image left over from a
reverted experiment that moved `warm_forest()` to the other side of the worker's
FIFO open - which is precisely where the rank's wait comes from, so it masked
the effect. **Rebuild every image the arms depend on before an A/B, not just the
one being changed**; `scripts/build.sh` is ~2s per image against runs that are
~3s each.

#### The critical path is the simulate worker's warm-up, not the rank

Absolute timestamps taken on both sides of the FIFO at once - in the rank's
`prewarm()` and inside the `--serve --fifo` worker - line up like this on a
default 8-rank run (seconds from the run script starting, worker running from
the bind mount so its imports are ~0.15s slower than the baked-in copy):

| Marker | When |
|---|---:|
| worker process enters `serve()` | 0.58-0.66 |
| worker's `meqserver_session()` returns | 0.69-0.88 |
| rank reaches `prewarm()`'s join | 0.76-0.93 |
| worker's `warm_forest()` returns, worker opens its FIFOs | **1.00-1.20** |
| rank's `simulate_worker()` returns | 1.00-1.20, to the millisecond |
| worker reads request one | 1.00-1.20, to the millisecond |

Every rank-side marker lands *before* the worker is ready, and the rank's
connect and the worker's FIFO open are the same instant, so the rank spends
0.2-0.3s blocked in the join. Anything a rank does *before* the join is
therefore free, and anything after it is not: `OMPI_MCA_pml=ob1` (above) sits
after the join and is worth 0.136s end to end, while paying PolyChord's one-time
setup with a throwaway `run_polychord()` before the join was measured at
-0.007s +/- 0.026s over 20 pairs and dropped, as was opening the worker's FIFOs
before its warm-up instead of after (0.35s off the join, 0.00s end to end - it
only moves the same wait into evaluation one).

It also explains an apparent regression: `OMPI_MCA_pml=ob1` takes the eight
`eval_id == 1` `simulate_seconds` records from ~0.06s to ~0.14s and summed
simulate worker-seconds from 2.25 to 2.88. Nothing got slower. The rank starts
timing when it writes the request and the worker reads it at the same absolute
moment either way, so a rank that reaches evaluation one earlier simply measures
more of the wait. Steady-state `simulate_seconds` is ~0.05-0.06s in both.

The next lever is therefore inside the worker: ~0.32s of `docker run` before its
command starts, then ~0.07s of interpreter and imports (with the baked-in,
byte-compiled copy), ~0.10s of meqserver, and ~0.25s of `warm_forest()`. Only
the last two are ours.

##### How much slack each branch of the startup has

Two independent branches converge before the first likelihood, and only the
slower one is on the clock. Measured with `PS4='+[${EPOCHREALTIME}] ' bash -x`
on the run script plus in-process markers, seconds from the script starting:

| Branch | Steps | Ready at |
|---|---|---:|
| worker | `docker info` 0.045 -> `docker run -d` issued -> container command starts +0.26-0.35 -> worker ready +0.51 | **0.85-0.95** |
| rank | ... -> all three `docker run -d` return +0.30-0.53 (the manifest's 0.24-0.42s hides inside it) -> `docker exec` + `mpirun` + interpreter + imports +0.20-0.27 | 0.55-0.80 |

So the rank branch has 0.25-0.40s of slack: work moved onto it is free until it
becomes the binding branch, and work taken off it buys nothing. The worker
branch is what to attack, and inside the worker's 0.51s the split is ~0.13s of
interpreter and imports, ~0.15s of `meqserver_session()`, ~0.12s of `makems` and
~0.20s of TDL compile plus first predict.

The tail is 0.085s: that is what elapses between the last rank's `atexit` and
`docker exec` returning - CPython finalisation, `MPI_Finalize`, `mpirun` reaping
and the exec stream closing. Nothing in the repo runs during it.

##### Measured and rejected inside the worker's warm-up

- **Building the warm-up MS while the meqserver starts.** `makems` is a
  subprocess this process only waits on and `meqserver_session()` is Timba
  imports plus another child, so running them in two threads should have taken
  ~0.12s off the serial path. It does - for the *fastest* worker. Time to ready
  across eight concurrent workers went min 0.489 -> 0.413 but max 0.504 ->
  0.508, and 30 interleaved end-to-end pairs read +0.020s +/- 0.030s. The eight
  workers contend (one worker alone is ready in 0.35-0.40s against 0.50-0.54s
  for eight), so reordering within a worker just concentrates the contention
  instead of removing work. This is the same null result as baking a ready-made
  MS into the image, for the same reason.
- **Importing `mpi4py` eagerly, before `prewarm()`'s join.** `mpirun -np 8
  python3 -c "from mpi4py import MPI"` costs 0.20s against 0.07s for `-c pass`,
  so `MPI_Init` looked like ~0.13s sitting after the join. It is not: markers
  show only ~0.02s between the last rank leaving the join and the first
  likelihood call, and forcing the import early made first-likelihood 0.417 ->
  0.447. Whatever the standalone `mpirun` measurement is paying for, the real
  run does not pay it there.
- **Further Open MPI MCA tuning.** On top of `OMPI_MCA_pml=ob1`, none of
  `btl=self,vader`, `osc=sm`, `coll=basic,libnbc,self`,
  `hwloc_base_binding_policy=none` or `rmaps_base_mapping_policy=slot` moves
  `mpirun -np 8 python3 -c "from mpi4py import MPI"` outside 0.19-0.22s.
- **Container `docker run` options.** Best-of-four time from `docker run
  --detach` to the container's command running is 0.26s minimal, 0.26s with
  `--shm-size 512m`, 0.26s with the repo bind mount added and 0.34s on the
  default bridge network. `--network none` is the only one that matters and is
  already in use; 0.26s is this host's rootless-Docker floor.

#### Long-lived sidecar containers, one per image

`docker run` costs ~0.40s of create/start/teardown on this host regardless of
image, mounts or `--platform`, while `docker exec` into an already-running
container costs ~0.03s. The MeqTrees simulate and WSClean imaging sidecars are
both short work against bind-mounted paths, so each runs in a detached `sleep
infinity` container that lives for the whole run - through the long-lived `sh`
above for WSClean, through the `--serve` worker for simulate. Both run scripts
source `scripts/lib/start-sidecars.sh` and start those containers before the
PolyChord container, handing the names to every rank in `NS_SIDECARS`;
`sidecar_container()` in `poc_common.py` still starts one itself for any image
that is not in there.

That container mounts `REPO_ROOT` at its own host path (the same trick the
PolyChord container already uses), so sidecar arguments are plain absolute
paths instead of the old per-evaluation `-v {eval_dir}:/work` plus `/work/...`.
`sidecar_command()` reads the image's `ENTRYPOINT` back with `docker inspect`
rather than restating the Dockerfile, since neither `docker exec` nor the
sidecar shell applies it, and each evaluation runs in its own working directory
so anything a sidecar writes relative to the cwd still stays per-evaluation -
except the simulate worker, which outlives any one evaluation and so runs in
`REPO_ROOT` and writes only absolute paths.
Pre-started containers are removed by the run script's `EXIT` trap, and any a
rank started itself via `atexit`; a `SIGKILL`ed run leaks sleeping containers,
cleaned up with `docker rm -f $(docker ps -q --filter name=ri-ns-sidecar-)`.

One container per image, rather than one per rank, is what makes the start
cheap. A single `docker run` of these images costs ~0.36s here, but 16 of them
at once - which is what 8 ranks x 2 images did the moment the ranks came up -
costs 1.3s, and all of it lands in front of the first evaluation. Separate
`docker exec` processes in one container are already isolated: 8 concurrent
`--serve` workers in a single MeqTrees container, each with its own meqserver,
run without interfering. Measured over 3 runs each, the default 8-rank run went
5.44s -> 3.64s (-33%) with identical `log(Z)`, the same 41 evaluations and
byte-identical metrics for every one of them.

The WSClean call also dropped `run_docker_monitored()`'s `docker stats`
sampler. GNU `time -v` already runs inside that container and reports an exact
peak RSS, where the 0.2s-interval sampler both missed short peaks and delayed
noticing the process had exited. R2D2 still uses `run_docker_monitored()`; its
image has no GNU `time`.

Together these took the profiled run from 43.5s to 27.5s (-37%) with identical
per-evaluation metrics, identical `log(Z)`, and the same set of files in every
evaluation directory. The R2D2 PoC picks up the simulate half of this through
the shared `simulate_measurement_set()`; its own convert and imaging containers
are still one `docker run` each - the imaging one needs the extra
`checkpoints/` mount and per-run thread env vars, and is dominated by R2D2's own
runtime anyway.

#### Sidecar containers run with `--network none`

Every per-evaluation sidecar (MeqTrees simulate, MS-to-`.mat` convert, WSClean,
R2D2) is launched with `--network none`. None of them talks to the network:
inputs and outputs are bind-mounted, and MeqTrees' `meqserver` only needs the
loopback interface, which `none` still provides.

Docker's default bridge network costs ~0.65s per container to set up and tear
down here versus ~0.45s with `--network none` - about 0.2s per container, or
0.44s per evaluation across the simulate and imaging sidecars. On the profiled
run that cut total wall time from 51.5s to 43.0s (-16.5%) with byte-identical
per-evaluation metrics and the same log(Z). The gap is largest under rootless
Docker, where bridge setup goes through a userspace network stack.

#### The meqserver shutdown sleep

An equivalent run before this fix profiled at 162.7s of MeqTrees simulate
(~11.6s/eval, 92.6% of wall time). Almost none of that was RIME work: makems takes ~0.5s, the
container start ~0.8s, and the TDL compile plus predict ~0.4s. The remaining
~10s per evaluation was Timba's `stop_default_mqs()`, which reaps the meqserver
child with a single `waitpid(WNOHANG)` and then sleeps a fixed 10 seconds before
re-checking - so every evaluation paid a full 10s of pure shutdown wait.

`docker/meqtrees/Dockerfile` rewrites that poll loop in the installed
`Timba/Apps/meqserver.py` to sleep 0.1s per iteration over 2000 iterations,
keeping the same ~200s ceiling before the SIGKILL fallback. Simulate wall time
drops from ~11.8s to ~1.9s per evaluation with bit-identical `DATA`, `UVW`,
`WEIGHT`, `SIGMA` and `FLAG` columns. The patch asserts on the exact upstream
source lines, so a KERN package bump that changes them fails the image build
rather than silently reverting the speedup.

## Output files

### WSClean

Each likelihood evaluation:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/wsclean/recon-image.fits
evaluations/eval-*/wsclean/recon-dirty.fits
evaluations/eval-*/wsclean/recon-residual.fits
evaluations/eval-*/metrics.json
```

### R2D2

Each likelihood evaluation:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/r2d2_data.mat
evaluations/eval-*/r2d2_config.yaml
evaluations/eval-*/r2d2/r2d2_data/R2D2_model_image.fits
evaluations/eval-*/r2d2/r2d2_data/dirty_normalised.fits
evaluations/eval-*/r2d2/r2d2_data/R2D2_residual_dirty_image.fits
evaluations/eval-*/metrics.json
```

### Run summary and reports

Run-level summary:

```text
poc-summary.json
```

View completed runs (settings, evidence, per-evaluation metrics and
reconstructions) in the nested-sampling HTML report:

```bash
make nested-sampling-report
# open benchmarks/nested-sampling-report.html

make nested-sampling-report LAST=1
# open benchmarks/nested-sampling-report-last.html

make nested-sampling-report RUN=results/nested-sampling-poc/r2d2-vlaa-merged-20260818T125604Z
# open benchmarks/nested-sampling-report-r2d2-vlaa-merged-20260818T125604Z.html
```

`LAST=N` renders only the newest N runs (timestamp sort) into
`benchmarks/nested-sampling-report-last.html`. `RUN=` renders one named
run (directory, repo-relative path, or directory name) into
`benchmarks/nested-sampling-report-<run>.html`. Neither overwrites the
full report. They cannot be combined. Make cannot take `--last`; use
`LAST=1`.

The report globs `results/nested-sampling-poc/*/poc-summary.json` directly
(no manifest join), so a merged run directory (see **Merge runs** below) shows
up as its own card automatically. Evidence prefers a `log_z` /
`log_z_err` pair already in the summary (written for merged runs); otherwise
it parses PolyChord `chains/*.stats` for log(Z). It shows
each run's total wall-clock duration (from `total_wall_seconds`, when present)
top-right in the card header. Per-run images - the shared synthesized
ground-truth image and a per-evaluation card gallery (reconstruction,
objective, and searched parameters) - sit in an Images tab, and the
best-effort `anesthetic` KDE contour corner plot sits in a Likelihood tab, both inside
one collapsed-by-default details block, separate from the collapsed raw
metrics table. Corner plots are weighted by the raw log-likelihood (the
failure score), not by nested-sampling posterior mass. Runs are ordered newest-first by the UTC timestamp in the
run directory name.

### Replay a run in anesthetic's GUI

For an interactive nested-sampling replay (live points vs \(\ln X\), \(\beta\)
tempering) with human-readable parameter labels, run on the **host** (needs a
display; not inside Docker/Colima):

```bash
make anesthetic-gui
make anesthetic-gui RUN=results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
uv run scripts/anesthetic-gui.py results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
```

With no `RUN=`, the latest *completed* run under `results/nested-sampling-poc/*/`
is used - either a plain run (`poc-summary.json` and `chains/`) or a merged
run (`poc-summary.json` with `merged_from`, no local `chains/`). For a plain
run the script writes/refreshes `chains/<root>.paramnames` from that run's
`poc-summary.json` / `parameter-space.json`; either way it passes only the
searched Fourier parameter names into `samples.gui(params=...)` (not `logL` /
`logL_birth` / `nlive`). Close the GUI window to return to the shell.
Requires the host `uv` project dependency `anesthetic`
(`uv add anesthetic` if missing).

The run also writes a standard environment manifest through
`scripts/record-environment.sh`.

## Merge runs

Independent PolyChord runs of the **same likelihood and prior** can be fused
after the fact into one run directory, without re-running PolyChord. This is
post-processing only: it concatenates nested-sampling dead points with
`anesthetic.samples.merge_nested_samples` and recomputes live-point weights.
Evaluation directories, FITS images, and PolyChord chain files are never
copied; the merged summary just points back at the absolute evaluation paths
and source run directories already on disk.

Sampler effort may differ between sources; the search itself must not:

| May differ | Must match |
|---|---|
| `NS_NLIVE` / `polychord.nlive` | `algorithm` |
| `NS_NUM_REPEATS` / `polychord.num_repeats` | `vla_config` |
| `NS_MAX_NDEAD` / `polychord.max_ndead` | `metric` |
| `seed`, `mpi_procs` | `parameter_space` (name/min/max/kind) |
| | `r2d2_fixed_hyperparameters` **or** `wsclean_fixed_hyperparameters` |

WSClean and R2D2 runs never merge with each other, nor do runs with a
different `--metric` / `NS_METRIC` or a different prior box (the prior box is
`PARAMETER_SPACE` in `scripts/lib/nested_sampling/poc_common.py`, copied into
every `poc-summary.json` as `parameter_space`).

With no directories, every completed source run under
`results/nested-sampling-poc/` is grouped by the must-match fields above
and one merged directory is written per group of 2+. Incomplete dirs,
previous merges (`merged_from`), and singleton groups are skipped.
Zero groups of 2+ exits non-zero. `--out` is only valid with an explicit
directory list.

```bash
uv run scripts/merge-nested-sampling-runs.py
make merge-nested-sampling

uv run scripts/merge-nested-sampling-runs.py \
  results/nested-sampling-poc/r2d2-vlaa-AAA \
  results/nested-sampling-poc/r2d2-vlaa-BBB

make merge-nested-sampling RUNS="results/nested-sampling-poc/r2d2-vlaa-AAA results/nested-sampling-poc/r2d2-vlaa-BBB"
```

Writes `results/nested-sampling-poc/<algorithm>-vlaa-merged-<UTC>/poc-summary.json`
(pass `--out DIR` on the explicit form to pick a different output directory).
The explicit form refuses with a non-zero exit on fewer than two runs, a run
missing `poc-summary.json` or `chains/`, or any must-match field above differing.
`polychord.nlive` in the merged summary is the sum of source nlives;
`num_repeats` / `max_ndead` / `seed` stay a single value when all sources
agree, else become a list. Pooled `evaluations` keep source argument order,
are renumbered `eval_id` `1..N` (originals kept as `source_eval_id` /
`source_run`), and keep their original absolute `paths`.

`make nested-sampling-report` and `make anesthetic-gui RUN=<merged-dir>` both
treat the merged directory as a completed run - see the sections above.

## Deferred

Deferred deliberately:

- VLA.B and VLA.D.
- Full parameter-space exploration.
- VLA.C production exploration, although VLA.A and VLA.C are the prioritized
  follow-up configurations.
- A general multi-algorithm or multi-VLA orchestrator.

To add the next run family, reuse the simulator and metrics modules, keep VLA
configuration as the outer loop, and add only the missing algorithm-specific
runner.
