# Nested sampling

This repo uses PolyChord as a targeted search tool, not as a Bayesian posterior
fit. PolyChord maximizes a configurable objective metric (default
`total_rms_jy`).

Ground truth for every run is one unpolarized 1 Jy point source at phase centre.
Dynamic range is controlled by complex Gaussian thermal noise in the simulated
visibilities.

## Images

```bash
./ri build wsclean
./ri build r2d2
./ri build meqtrees
./ri build polychord
```

`./ri build` also builds the MeqTrees and PolyChord images.

The MeqTrees image uses KERN 10 packages on Ubuntu 24.04. The VLA.A antenna table
is unpacked from makems' bundled `VLAA_ANT` example inside the image, so antenna
positions are not hand-rolled in this repo. Visibilities for that skeleton are
predicted by an actual MeqTrees/Meow point-source RIME run
(`scripts/lib/nested_sampling/point_source_forest.py`, driven through
`meqtree-pipeliner.py`), not a hand-rolled formula; thermal noise is added on
top of that clean MeqTrees prediction.

## Run it

Both algorithms share the same `NS_*` and `OUTPUT_DIR` overrides (see "Environment
overrides" below). Each target builds its required images first and starts one
long-lived sidecar container per image; the PolyChord container mounts the
Docker socket and drives those sidecars. The WSClean target starts the
PolyChord container the same way and `docker exec`s the run into it (see "The
PolyChord container is a sidecar too" and "Long-lived sidecar containers, one
per image" in
[nested-sampling-profiling.md](nested-sampling-profiling.md)).

### WSClean

```bash
./ri search wsclean
```

Outputs:

```text
results/nested-sampling/wsclean-vlaa-<UTC timestamp>/
```

Useful overrides:

```bash
./ri search wsclean --nlive 8 --num-repeats 2 --max-ndead 12
./ri search wsclean --mpi-procs 4
./ri search wsclean --metric badness
./ri search wsclean --metric snr
./ri search r2d2 --metric off_source_rms_jy
./ri search r2d2 --metric sigma_res
./ri search wsclean --output-dir results/nested-sampling/manual
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
./ri search r2d2
```

Outputs:

```text
results/nested-sampling/r2d2-vlaa-<UTC timestamp>/
```

The target builds R2D2, MeqTrees, and PolyChord images first. Each likelihood
evaluation runs one MeqTrees simulate, one MeqTrees-hosted MS-to-`.mat`
conversion, and one R2D2 imaging container.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `./ri fetch-checkpoints` and `./ri smoke r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
./ri smoke ms-to-mat                 # or: scripts/check-ms-to-r2d2-mat.sh
```

`run-nested-sampling-r2d2.sh` runs `NS_MPI_PROCS` PolyChord ranks
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

### Environment overrides

Both run scripts read the same `NS_*` variables and forward them to
`polychord_wsclean.py` / `polychord_r2d2.py` as command-line flags.
The defaults live in `defaults.toml` at the repository root, loaded by
`scripts/lib/defaults.sh`, which both scripts source.

`./ri search` exposes each of them as a flag (`--nlive`, `--num-repeats`,
`--max-ndead`, `--seed`, `--metric`, `--mpi-procs`, `--omp-threads`,
`--output-dir`) which sets the variable for that run. Setting the variable
yourself still works and still wins over `defaults.toml`; a flag wins over both.

| Variable | Meaning | Default |
|---|---|---|
| `NS_NLIVE` | Number of PolyChord live points (`--nlive`) | `8` |
| `NS_NUM_REPEATS` | How much PolyChord explores inside the likelihood constraint before generating a replacement live point (`--num-repeats`) | `2` |
| `NS_MAX_NDEAD` | Dead-point budget that terminates the run (`--max-ndead`) | `12` |
| `NS_SEED` | PolyChord random seed (`--seed`) | `41` |
| `NS_METRIC` | Objective (`--metric`): `badness`, a bare metric name, or an expression over metric names - see "Choosing the objective" below | `total_rms_jy` |
| `NS_MPI_PROCS` | PolyChord rank count (`mpirun -np`); `1` disables parallel evaluations | `min(NS_NLIVE, host CPUs)`, host CPUs from `docker info` |
| `NS_SIDECARS` | JSON map from image name to that image's long-lived sidecar container | Exported by `scripts/lib/start-sidecars.sh`. Unset means `{}`: each rank starts its own container per image |
| `NS_SIMULATE_FIFO_DIR` | Directory holding the per-rank `<rank>.in` / `<rank>.out` FIFOs of the pre-warmed simulate workers | `${OUTPUT_DIR}/.simulate-workers`, set by the WSClean run script only. No default: unset (as in the R2D2 run script) means each rank starts its own simulate worker |

`NS_SIDECARS` and `NS_SIMULATE_FIFO_DIR` are wiring the run scripts export for
the containers they start, not knobs to set by hand.

## Parameter space

VLA configuration is an outer-loop dimension. The runs here only use `VLA.A`.

PolyChord dimensions for both algorithms:

| Dimension | Range | Meaning |
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
`summary.json` under `wsclean_fixed_hyperparameters`.

**R2D2:** `128x128` image size (matching the WSClean run's `-size 128 128
-scale 1asec` footprint), `num_iter 25`, `architecture unet`, `num_chans 64`,
`ckpt_path /checkpoints/R2D2_A1`, and `ckpt_realisations 1`, recorded in
`summary.json` under `r2d2_fixed_hyperparameters`.

## MS to R2D2 `.mat` bridge

R2D2-RI reads visibilities from a MATLAB `.mat` file via `load_data_to_tensor()`
in the upstream `src/utils.py`. The nested-sampling simulator produces a CASA
Measurement Set (`sim.ms`) that WSClean consumes directly. The R2D2 run adds
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
default objective is `total_rms_jy` (RMS of the reconstructed image minus
the one-pixel truth, over all pixels).

An optional composite `badness` score is also available (higher means worse
reconstruction or a more expensive run):

```text
max(0, 3 - log_snr)
+ min(relative_l2_error, 10)
+ 0.05 * min(wall_seconds / 60, 5)
+ 0.02 * min(peak_memory_bytes / 2 GiB, 5)
```

### Choosing the objective (`--metric` / `NS_METRIC`)

Both `polychord_wsclean.py` and `polychord_r2d2.py` accept
`--metric <value>` (default `total_rms_jy`). The shell wrappers forward
`NS_METRIC`, whose default lives in `defaults.toml`, with the same value.
Resolution order:

1. `badness` - the composite formula above.
2. Any bare metric name from the table - use that raw value directly as the
   objective (including the default `total_rms_jy`).
3. Any other string - treat it as an arithmetic expression over the same metric
   names (for example `log_snr + 0.1 * wall_seconds`, or the composite formula
   rewritten by hand).

Expressions are compiled once at startup (before any Docker evaluations) and
evaluated in a restricted namespace: no Python builtins, metric names as locals,
and `math` module functions available by name. A typo or unsafe expression fails
immediately at startup.

PolyChord always maximizes the returned value with no automatic sign flip. The
`badness` composite is oriented so higher is worse. Raw metrics keep their
natural orientation: the default `total_rms_jy` search prefers higher
whole-image RMS error, `--metric snr` searches for the highest-SNR corner, and
a worst-SNR search must negate explicitly (`--metric "-snr"` or
`--metric "1/snr"`). `off_source_rms_jy` and `sigma_res` are also
higher-is-worse (noisier reconstruction / worse data fidelity); search for the
best corner with `--metric "-total_rms_jy"`, `--metric "-off_source_rms_jy"`
or `--metric "-sigma_res"`. Failed simulations or imaging runs still receive
objective `100.0`.

Each evaluation record and `summary.json` store the chosen value in an
`objective` field. `summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

## Profiling

Every run times each stage of every likelihood evaluation automatically -
there is no separate flag. To read the breakdown of a completed run:

```bash
./ri profile results/nested-sampling/wsclean-vlaa-<UTC timestamp>
# or directly:
uv run scripts/profile-nested-sampling-run.py results/nested-sampling/wsclean-vlaa-<UTC timestamp> [--json]
```

The numbers come from the `timing` block in each evaluation's `metrics.json`
and the run-level `profiling` block in that run's `summary.json`.

The same breakdown is available without the CLI: a run's HTML report page has a
collapsible "Profiling (where the run's time went)" section, shown whenever that
run's `summary.json` carries a `profiling` block (runs predating the profiler
instrumentation simply omit the section). It leads with a stacked bar of where
the worker-time went and then the same rows as the CLI table, so the report page
and `profile-nested-sampling-run.py` always agree - both call
`profiling_breakdown()` in `scripts/lib/nested_sampling/common.py`.

Both views show, per stage: the total, the mean per evaluation, the share of the
run's worker-time budget, and the evaluation count. Durations are rendered in
whatever unit carries their digits (`33ms`, `1.44s`, `39m 15s`, `1h 00m 45s`).

See [nested-sampling-profiling.md](nested-sampling-profiling.md) for what each
field means and for every measured (and rejected) optimisation behind the
current run scripts and images.

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
summary.json
```

View completed runs (settings, evidence, per-evaluation metrics and
reconstructions) in the nested-sampling HTML report:

```bash
./ri report
# open reports/nested-sampling-report/index.html

./ri report --last 1
./ri report --run results/nested-sampling/r2d2-vlaa-merged-20260818T125604Z
./ri report --upgrade
./ri report --force
```

Each run gets its own page, `reports/nested-sampling-report/<run>.html`,
plus an `index.html` that lists every run on disk and links into them; each
run page links back to the index. Rendering a run means reading its FITS
output, so **run pages that are already up to date are skipped** - a re-run
only builds pages for new runs.

The plots themselves are PNG files under
`reports/nested-sampling-report/images/`, named after a hash of what they
were drawn from, and the pages link to them rather than inlining them. That
is where almost all of the report's time goes, so rebuilding a page (below)
redraws nothing that its inputs still match - only deleting the report
directory forces a full redraw. The hash covers a plot's inputs, not how it
was drawn, so a change to the drawing code has to bump `IMAGE_RENDER_VERSION`
in `scripts/lib/generate_report.py` to retire the PNGs already on disk.

Evaluation rasters are colour-mapped straight into a PNG at the FITS data's
own resolution and scaled up by the browser, rather than drawn through a
matplotlib figure - roughly 16x cheaper per image and 5x smaller on disk, and
indistinguishable at the size the pages display them. What is left of a full
redraw's cost is the anesthetic corner plot. A rebuild that redraws nothing does not even
import astropy or matplotlib - they are loaded on the first missing PNG - which
is most of what is left of a page-only rebuild. The two halves load separately:
the corner plot needs only matplotlib, the eval rasters astropy and PIL on top,
so a cold build keeps astropy out of the parent's serial prologue and off the
corner plot's critical path (the longest task in the build). The index is always rebuilt, so
it picks up new runs immediately.

Run pages are built in parallel, and each run splits into two concurrent
tasks - the anesthetic corner plot and the rest of the page - so the pool has
twice as many pages to overlap. The two kinds of task go into two pools, forked
either side of the astropy import: the corner plots - the critical path - start
first, and the eval-raster workers forked afterwards inherit astropy rather than
each importing it again while the plots want the CPU. anesthetic is imported in
the parent for the same reason, just before the first fork, so every corner-plot
worker inherits it instead of repeating the same 0.34s. Together that is ~25%
less CPU on a five-run cold build for the same output, and once there are more
runs to draw than cores it is wall-clock too (20 runs: 5.25s -> 4.75s).
The container is given a single BLAS thread
(the work is matplotlib rasterisation, not linear algebra, and multi-threaded
BLAS only oversubscribes the CPU). Override with `R2D2_OMP_THREADS=`. It also
runs with `--network none`: the report only reads the repo and writes
`reports/`, and setting up the container's network is ~0.3s of every
invocation - most of the cost of a build that draws nothing.

The `r2d2` image bakes matplotlib's font list into `/opt/matplotlib`
(`MPLCONFIGDIR`). Containers run with `--rm`, so without it the first
`import matplotlib.pyplot` in every one of them rebuilds that list from the
installed fonts - ~0.07s, on the report's serial prologue, on every cold
build. Rebuild with `scripts/build.sh r2d2` to pick it up; an image without
it still works, just that much slower.

Every page carries the version of the report generator that wrote it (the
hash of `scripts/lib/generate_report.py`, in a
`<meta name="report-version">` tag), so changing the card design, the CSS or
anything else in that file makes existing pages **outdated** rather than
silently stale. Outdated pages are still skipped by a plain run - it says how
many it saw - and the index flags them with an `outdated page` badge.
`UPGRADE=1` rebuilds exactly those, bringing every page up to the current
design.

`LAST=N` only considers the newest N runs (timestamp sort). `RUN=` targets
one named run (directory, repo-relative path, or directory name) and always
rebuilds its page. `UPGRADE=1` rebuilds the pages an older report version
wrote. `FORCE=1` rebuilds every page in scope, up to date or not. `LAST=` and
`RUN=` cannot be combined. Make cannot take `--last`; use `LAST=1`.

The report globs `results/nested-sampling/*/summary.json` directly
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
./ri plot gui
./ri plot gui results/nested-sampling/wsclean-vlaa-<UTC timestamp>
uv run scripts/anesthetic-gui.py results/nested-sampling/wsclean-vlaa-<UTC timestamp>
```

With no `RUN=`, the latest *completed* run under `results/nested-sampling/*/`
is used - either a plain run (`summary.json` and `chains/`) or a merged
run (`summary.json` with `merged_from`, no local `chains/`). For a plain
run the script writes/refreshes `chains/<root>.paramnames` from that run's
`summary.json` / `parameter-space.json`; either way it passes only the
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
`PARAMETER_SPACE` in `scripts/lib/nested_sampling/common.py`, copied into
every `summary.json` as `parameter_space`).

With no directories, every completed source run under
`results/nested-sampling/` is grouped by the must-match fields above
and one merged directory is written per group of 2+. Incomplete dirs,
previous merges (`merged_from`), and singleton groups are skipped.
Zero groups of 2+ exits non-zero. `--out` is only valid with an explicit
directory list.

```bash
uv run scripts/merge-nested-sampling-runs.py
./ri merge

uv run scripts/merge-nested-sampling-runs.py \
  results/nested-sampling/r2d2-vlaa-AAA \
  results/nested-sampling/r2d2-vlaa-BBB

./ri merge results/nested-sampling/r2d2-vlaa-AAA results/nested-sampling/r2d2-vlaa-BBB
```

Writes `results/nested-sampling/<algorithm>-vlaa-merged-<UTC>/summary.json`
(pass `--out DIR` on the explicit form to pick a different output directory).
The explicit form refuses with a non-zero exit on fewer than two runs, a run
missing `summary.json` or `chains/`, or any must-match field above differing.
`polychord.nlive` in the merged summary is the sum of source nlives;
`num_repeats` / `max_ndead` / `seed` stay a single value when all sources
agree, else become a list. Pooled `evaluations` keep source argument order,
are renumbered `eval_id` `1..N` (originals kept as `source_eval_id` /
`source_run`), and keep their original absolute `paths`.

`./ri report` and `./ri plot gui <merged-dir>` both
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
