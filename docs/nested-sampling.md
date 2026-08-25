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

Total wall time 10.2s (~0.16s/eval; ~2.5s on the default 8 ranks). That is
`poc-summary.json`'s `total_wall_seconds`, measured around `run_polychord()`
inside the PolyChord container - the end-to-end `time` of the run script is
~1.1s more on one rank and ~1.6s more on eight, for starting and removing the
containers (8-rank end to end is ~4.1s). No fixed overhead of any
size is left in either sidecar: what remains is the science.
Warm, an evaluation is ~0.05s of simulate (~0.02s RIME predict, the rest
casacore table I/O, plus ~0.05s of `makems` on the evaluations that miss the MS
skeleton cache) and ~0.10s of `wsclean`, which is now well over half the run.
The rest is one-off startup - the simulate worker, meqserver, the one TDL
compile and the `astropy` import, now started concurrently before the sampler
runs rather than serially inside the first evaluation - plus PolyChord's own
sampling and bookkeeping.

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

#### `WEIGHT`/`SIGMA` are written row by row, not with `putcol`

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

`fill_point_source_visibilities()` therefore writes each column with a
`putcell()` loop, which is linear: 7.9ms for the pair. Keep the two loops
separate - interleaving the columns costs 19ms, because each column's ISM
buckets get evicted between rows. Measured 13.1s -> 10.9s single-rank (-16%,
simulate 5.5s -> 3.4s) and 8.5s -> 8.2s on the default 8 ranks, where wsclean
dominates and each rank runs only ~5 evaluations. Verified by comparing every
column of all 62 Measurement Sets between a before and after run (identical),
the artifact trees (identical), all 62 evaluations' science metrics
(identical) and `log(Z)` (identical).

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
so the parameter space has 20 distinct shapes and a long-lived `--serve` worker
hits the cache for most of its evaluations.

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

Because the cache lives in `/dev/shm` alongside the working MS, the sidecar
containers are started with `--shm-size 512m`; docker's 64MB default is only
about 3x what 20 cached skeletons need.

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
| `astropy.io.fits` import, on the first metrics call | ~0.20s |
| `docker inspect` of each image's `ENTRYPOINT`, twice | ~0.05s |
| the WSClean sidecar's `sh` (`docker exec`) | ~0.03s |

`poc_common.prewarm()` starts all of them in threads, so a rank pays the slowest
instead of the sum. `main()` calls it before `import pypolychord` and joins it
immediately before `run_polychord()`, so the remaining ~0.45s also overlaps the
sampler's own import and setup. Nothing may touch a sidecar between the call and
the join: `_SIMULATE_WORKERS`, `_SIDECAR_SHELLS` and `_IMAGE_ENTRYPOINTS` are
plain dicts with no lock, and a lazy start racing the prewarm thread would leave
a second, orphaned worker.

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

Two more things ride on that:

- **A warm-up command.** The first process in a fresh container faults the
  whole dynamic-linker path of whatever it runs through the image's overlay
  mount. The real 8-rank `mpirun python3` exec measured 0.85s cold against
  0.19s once one throwaway `python3 -c "import numpy, pypolychord"` had paid
  that; `docker exec <container> true` first does *not* help, so it is the
  libraries, not the container. `sidecar_launch` takes that warm-up after a
  `--` and runs it inside the same background job as the `docker run`, so it is
  hidden behind the other containers coming up. The MeqTrees and WSClean
  sidecars are not worth warming: their cold-vs-warm first exec is 0.17s vs
  0.13s and 0.09s vs 0.08s.
- **The manifest write moved into the gap.** `scripts/record-environment.sh` is
  ~0.4s of `git` and `docker image inspect`, and now runs between
  `sidecar_launch` and `sidecar_wait` instead of after the containers are up.
  `NS_SIDECARS` is exported by `sidecar_launch` rather than `sidecar_wait`
  because the container names are known as soon as the launches are issued.

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
