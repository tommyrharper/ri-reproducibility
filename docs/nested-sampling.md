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
Each target builds its required images first, then runs the PolyChord container,
which mounts the Docker socket and launches per-evaluation sidecar containers.

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
Each likelihood evaluation launches one MeqTrees simulation container plus one
WSClean imaging container.

### R2D2

```bash
make nested-sampling-r2d2-poc
```

Outputs:

```text
results/nested-sampling-poc/r2d2-vlaa-<UTC timestamp>/
```

The target builds R2D2, MeqTrees, and PolyChord images first. Each likelihood
evaluation launches one MeqTrees simulation container, one MeqTrees-hosted
MS-to-`.mat` conversion, and one R2D2 imaging container.

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
| `peak_memory_bytes` | Peak imaging memory from Docker stats (WSClean also records GNU `time` when available) |

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

A single-rank (`NS_MPI_PROCS=1`), 5-dimensional, `NS_NLIVE=3 NS_MAX_NDEAD=4`
WSClean PoC run (19 likelihood evaluations, not committed) profiled as:

| Stage | Total | Share |
|---|---:|---:|
| MeqTrees simulate | 29.7s | 69.0% |
| WSClean image container (total) | 12.8s | 29.7% |
| &nbsp;&nbsp;of which: `wsclean` binary itself | 2.3s | 5.3% |
| &nbsp;&nbsp;of which: container overhead | 10.5s | 24.4% |
| Metrics computation | 0.3s | 0.7% |
| PolyChord overhead (unaccounted) | 0.27s | 0.6% |

Total wall time 43.0s (~2.3s/eval). The MeqTrees RIME simulation is still the
single largest stage (~1.6s/eval), but docker container start/teardown for the
imaging sidecar is the second (~0.55s/eval, roughly 4.5x the `wsclean` binary's
own runtime). PolyChord's own sampling/bookkeeping overhead stays negligible
(0.6%) at this scale.

The remaining container overhead is `docker run` create/start/teardown itself,
measured at ~0.45s per container on this host regardless of image, mounts, or
`--platform`. `docker exec` into an already-running container costs ~0.02s, so a
long-lived per-rank sidecar container is the natural next step - two `docker
run` calls per evaluation currently cost ~0.9s of the ~2.3s.

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
