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
Docker socket and drives those sidecars. It is started the same way as the
others and the run is `docker exec`ed into it (see "The PolyChord container is
a sidecar too" and "Long-lived sidecar containers, one per image" in
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
conversion and one R2D2 imaging job, each a request to a long-lived process
inside one of the run's two sidecar containers.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `./ri fetch-checkpoints` and `./ri smoke r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
./ri smoke ms-to-mat                 # or: scripts/check-ms-to-r2d2-mat.sh
```

`run-nested-sampling-r2d2.sh` runs `NS_MPI_PROCS` PolyChord ranks
concurrently, each with its own `r2d2_serve.py` imaging worker inside the
shared R2D2 sidecar and its own simulate worker inside the MeqTrees one. Both
pools are started by their container's own command, over one FIFO pair per rank,
before the PolyChord container exists - the imaging pool as one `--fifo-dir`
process that imports torch once, forks a worker per pair, and opens every pair
before it starts importing so the ranks do not wait for it (see "R2D2 imaging runs in a long-lived
worker", "The workers are started by the container, not by the ranks" and "The
ranks attach to the pool before the warm-up" in
[nested-sampling-profiling.md](nested-sampling-profiling.md)). That process also
patches R2D2's `MeasOp.get_op_norm` to solve the operator norm with Lanczos
rather than upstream's power iteration - the same quantity, ~3.5x fewer NUFFT
pairs and ~3e-6 relative accuracy instead of ~1e-4, and no longer a different
answer on every run (see "The operator norm is solved with Lanczos" there) - and
it gives each measurement operator one FINUFFT plan per transform type instead
of the one-plan-per-transform `pytorch_finufft` builds, worth ~30% of a warm
imaging request (see "Each measurement operator keeps its FINUFFT plans"). Its
warm-up runs `imager.py`'s own import block - the file under a run name that is
not `__main__` - plus the NUFFT backend `create_meas_op` imports lazily, and
makes `utils` resolve its submodules on demand so the imaging path never pays
for `lightning` or `scipy.optimize` (see "The imaging worker warms what
`imager.py` imports, and no more"). The workers get
OpenMP/BLAS thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`) set from the host's available CPU count, overridable via
`R2D2_OMP_THREADS`. The previous image default of
`OMP_NUM_THREADS=4` capped finufft/OpenMP work when the Docker VM exposed more
CPUs than four. To avoid CPU oversubscription, the script defaults
`R2D2_OMP_THREADS` to `host CPUs / NS_MPI_PROCS` (minimum `1`) when not set
explicitly, so each rank's imaging worker gets a fair share of the host's cores
instead of all of them. Set `R2D2_OMP_THREADS` explicitly to override this
per-rank default. The same count is written into every per-evaluation
`r2d2_config.yaml` as `ncpus`, because those env vars alone do not reach
torch - see "R2D2 sizes its own torch thread pool" in
[nested-sampling-profiling.md](nested-sampling-profiling.md).

### Environment overrides

Both run scripts read the same variables and forward the sampler ones to
`polychord_wsclean.py` / `polychord_r2d2.py` as command-line flags.
Sampler defaults live in `defaults.toml` at the repository root, loaded by
`scripts/lib/defaults.sh`. Setting a variable yourself still wins over
`defaults.toml`; a flag wins over both.

#### Tweak these

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--nlive` | `NS_NLIVE` | Number of PolyChord live points | `8` |
| `--num-repeats` | `NS_NUM_REPEATS` | How much PolyChord explores inside the likelihood constraint before generating a replacement live point | `2` |
| `--max-ndead` | `NS_MAX_NDEAD` | Dead-point budget that terminates the run | `12` |
| `--seed` | `NS_SEED` | PolyChord random seed | `41` |
| `--metric` | `NS_METRIC` | Objective: `badness`, a bare metric name, or an expression over metric names - see "Choosing the objective" below | `total_rms_jy` |

#### Leave these alone

Flags exist, but the defaults are derived. Leave them unset unless you want
serial debugging (`--mpi-procs 1`), a different rank/thread split, or a
pinned run directory.

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--mpi-procs` | `NS_MPI_PROCS` | PolyChord rank count (`mpirun -np`); `1` is serial | `min(NS_NLIVE, host CPUs)`, host CPUs from `nproc` (`sysctl -n hw.ncpu` on macOS, which has no `nproc`) |
| `--omp-threads` | `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS/torch threads | `host CPUs / NS_MPI_PROCS`, min 1 |
| `--output-dir` | `OUTPUT_DIR` | Run directory | `results/nested-sampling/<algo>-vlaa-<UTC>` |

The run scripts also export these for the containers they start. They have
no `./ri` flags. Unset, the Python ranks fall back to starting their own
workers, which is slower.

| Variable | Meaning | Set by the run scripts to |
|---|---|---|
| `NS_SIDECARS` | JSON map from image name to that image's long-lived sidecar container | Exported by `scripts/lib/start-sidecars.sh` |
| `NS_SIMULATE_FIFO_DIR` | Directory holding the per-rank `<rank>.in` / `<rank>.out` FIFOs of the pre-warmed simulate workers | `${OUTPUT_DIR}/.simulate-workers` |
| `NS_R2D2_FIFO_DIR` | Same, for the pre-warmed `r2d2_serve.py` imaging workers | `${OUTPUT_DIR}/.r2d2-workers` (R2D2 run only) |

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
`scripts/lib/nested_sampling/ms_to_r2d2_mat.py`, which the rank's simulate
worker runs in-process inside the MeqTrees sidecar (python3-casacore plus
scipy), and which writes the minimal field set R2D2 loads without flag
metadata:

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
| `peak_memory_bytes` | Peak imaging memory: GNU `time -v` for WSClean; for R2D2 the imaging worker's own high-water RSS, which is a running maximum across that rank's evaluations |

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

The index has a toolbar above the run cards: filter by algorithm (R2D2 /
WSClean) or by merged/unmerged, and sort newest/oldest or by eval count. It is
plain client-side JavaScript over the cards already on the page - no rebuild
or server needed, and it works the same off a `file://` open as it does
through `./ri serve`.

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
is most of what is left of a page-only rebuild. numpy goes the same way:
`common.py` binds `np` to a shim that imports numpy on first attribute
access, because the report reaches into that module only for its formatting
helpers and a page-only rebuild never touches an array (0.19s -> 0.16s on a five-run page-only rebuild, and
`self_check_lazy_numpy()` there fails if a module-level `np.` use creeps back
in). The two halves load separately:
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
invocation - most of the cost of a build that draws nothing. The container is
removed from an `EXIT` trap rather than by `docker run --rm`, which blocks the
CLI for another ~0.12s tearing the rootfs down after the report has already
been written to the bind mount: five-run cold build 1.75s -> 1.63s on the host
and a rebuild that draws nothing 0.55s -> 0.48s, with byte-identical output.
The trap also fires on the failure paths, so a container is removed whether the
report succeeded or not.

The corner plot itself is the build's longest task, and roughly a fifth of it
was pandas repeating work. anesthetic draws the grid as 15 separate pandas plot
calls, and pandas re-runs its shared-axis tick housekeeping over every axis in
the figure after each one. `dedupe_pandas_tick_housekeeping()` in
`scripts/lib/generate_report.py` makes that run once per axis. It is not a plain
skip: reading an axis' tick labels un-stales matplotlib's shared view limits,
and that side effect is what keeps each diagonal panel's CDF twin on its
parent's x range, so the repeats still touch `viewLim` - drop that and the plot
visibly changes. Corner plot 1.01s -> 0.81s, five-run cold build 2.01s -> 1.80s
in-container and 15% less CPU, with byte-identical PNGs. The patch is
best-effort: if pandas moves the private helper it is a no-op and the plot is
just slower again.

With the labels themselves de-duplicated, what was left of a repeat call was the
scan that decides which axes to hand to the helper: for every axis, and for both
of its axes, walk the whole shared-axis group comparing positions - 420 of those
per corner plot. The same function records the axes the first scan selects and
replays only their `viewLim` touch afterwards, keyed on the figure's axes, their
visibility and the grid shape, so anything that changes the answer misses. That
takes the scan to 28 calls; corner plot 0.552s -> 0.538s, five-run cold build
1.449s -> 1.435s in-container and 1.5% less CPU, with byte-identical PNGs.
Note that pandas' plotting core binds `handle_shared_axes` by `from ... import`,
so the patch has to rebind the name in every module that holds it, not just in
`pandas.plotting._matplotlib.tools`.

Matplotlib repeats work of its own on the same figure. Every layout pass -
`tight_layout()`, the tight-bbox draw, the render draw, and each spine and axis
label placed in between - recomputes every axis' tick positions and label text
from scratch, ~930 times for one corner plot and about a third of its cost.
`memoize_matplotlib_tick_updates()` caches `Axis._update_ticks` per axis against
the axis' view interval, data interval and locator/formatter pair, which is
everything the result depends on; building that key reads `get_view_interval()`,
so a cache hit still performs the same `viewLim` un-staling the de-duplication
above relies on. Corner plot 0.86s -> 0.69s, five-run cold build 1.65s -> 1.43s
in-container and 18% less CPU, with byte-identical PNGs. Best-effort in the same
way: a matplotlib that renames the private method just draws slower.

Laying out a `Text` - splitting it into lines, measuring each against the font,
rotating the box - is repeated the same way. Those three passes ask for it 441
times per corner plot, for 70 distinct texts: the tick labels and axis titles
are re-measured in full on every pass, and the diagonal panels share their
labels with the panels below them. The result is position-free, so
`memoize_matplotlib_text_layout()` caches `Text._get_layout` in one process-wide
dict keyed on the string, the renderer class and figure dpi, the font
properties, the usetex/parse-math flags, the line spacing, the rotation and
rotation mode, and the three alignments - every input the layout reads. Texts
with wrapping enabled bypass the cache entirely, because their line breaks also
depend on the figure width. Corner plot 0.523s -> 0.496s (5% of every plot, so
5% of the CPU once there are more runs than cores), five-run cold build 1.443s
-> 1.423s in-container and 1.6% less CPU, with byte-identical PNGs. Best-effort
like the rest.

anesthetic repeats work of its own. Its labelled frames resolve every `df[key]`
by attempting the lookup against each of four label-stripped views of the frame
and keeping the best answer, and each attempt rebuilds that axis' paramname ->
label mapping from the index. One corner plot does that ~420 times.
`memoize_anesthetic_labels_map()` caches `_LabelledObject.get_labels_map` on the
pandas `Index` it is read off. An `Index` is immutable and anything that adds or
drops a column swaps in a new one, so a mutated frame misses the cache rather
than seeing a stale mapping; the `fill=False` variant, which `set_label()`
mutates in place, is deliberately never cached. Corner plot 0.66s -> 0.62s,
five-run cold build 1.43s -> 1.37s in-container and 5% less CPU, with
byte-identical PNGs. Best-effort like the two above.

Each of those four attempts also deep-copies the whole frame first, so one
corner plot makes ~380 copies of the same handful of frames.
`memoize_anesthetic_drop_labels()` caches `_LabelledObject.drop_labels` against
the frame's identity, both of its pandas `Index` objects, and which
`(axis, level)` pairs actually get dropped - so the specs that drop the same
levels share one copy and the rest stay apart. Pinning the indexes is what makes
it safe: an `Index` is immutable, so adding, dropping or relabelling a column
swaps in a new one and misses the cache instead of handing back a stale copy of
the frame. Rewriting an existing column's values in place would still slip past
that, but nothing on the plotting path does - the frame is read-only from load
to `savefig`. Corner plot 0.57s -> 0.53s, five-run cold build 1.55s -> 1.51s
in-container and 4% less CPU (7% at twenty runs), with byte-identical PNGs.
Best-effort like the rest.

Caching both halves still leaves anesthetic *running* all four attempts and
throwing three away, which is the expensive part: `ac` evaluates every
candidate, sorts them by dimensionality (fewest first, then most index levels,
ties going to the earliest candidate) and returns the first. For the case the
corner plot hits ~65 times per plot - a plain string column name on a frame
whose columns carry the labels level - the winner is decided in advance, so
`shortcut_anesthetic_labelled_column()` runs only that one. It is the candidate
that strips the labels off the columns: the two that keep them index a
two-or-more-level `MultiIndex` and so return a 2-D frame, which loses to any 1-D
result outright, and of the two that strip them this one leaves the *index*
alone, so it never has fewer index levels than the other and it wins the tie by
position. A `DataFrame` lookup cannot return a 0-D result, so nothing can
undercut a 1-D one. The guards are premises of that argument rather than safety
nets - a string key is what forces the label-keeping candidates to be 2-D - but
anything outside them, any candidate that raises, and any result that turns out
not to be 1-D all fall back to anesthetic's own search, so an uncovered shape is
slow rather than wrong. Corner plot 0.449s -> 0.413s (-8.2% CPU), five-run cold
build 1.168s -> 1.120s wall in-container and 4.2% less CPU, with byte-identical
PNGs. Best-effort like the rest. `_self_check_labelled_column_shortcut` builds
each frame twice before comparing the two resolutions, because pandas caches a
frame's column `Series` and resolving both ways on one frame would compare an
object with itself.

Saving the finished figure repeated a pass too. `savefig(bbox_inches="tight")`
cannot trust the artist positions it is handed - a layout engine may still be
pending - so it walks the whole figure once in a draw-disabled pass before
measuring the box. After `fig.tight_layout()` both reasons are already spent:
the layout has run, and all it leaves behind is a do-nothing placeholder engine.
`tight_bbox()` in `scripts/lib/generate_report.py` clears the engine, measures
the box itself and hands `savefig` a `Bbox` instead of the string, which drops
that pass. Save 0.108s -> 0.083s, corner plot 0.572s -> 0.545s, five-run cold
build 1.51s -> 1.47s in-container and 3% less CPU, with byte-identical PNGs.
Unlike the caches above this only ever asks matplotlib to do less, so there is
nothing to degrade gracefully; `_self_check_tight_bbox` asserts the two save
paths still produce the same bytes.

Two more repeats are structural rather than per-pass. Every read of an axis'
`viewLim` asks matplotlib whether anything sharing a limit with it still needs
autoscaling, and that question walks the whole shared-axis group through a
`WeakSet`, once per axis name - ~3200 times over 20 axes for one corner plot,
almost always to answer "nothing". Staleness is only ever created in one place,
`_request_autoscale_view`, so `skip_settled_matplotlib_viewlims()` counts calls
to it: an axis whose group was found settled at epoch *N* is still settled while
the counter reads *N*, and the scan is skipped outright. The epoch is re-read
after the wrapped call, so an autoscale that re-stales the group on its way out
is not recorded as settled. Nothing else in matplotlib or `mpl_toolkits` writes
`_stale_viewlims` to `True`, so the counter sees every transition.

And anesthetic gives each panel its limit-linking behaviour by defining a fresh
`Axes` subclass *inside* the per-axis helper and rebinding `__class__` to it, so
a 5x5 corner plot builds 15 one-instance classes. Each one pays matplotlib's
`Artist.__init_subclass__`, which regenerates `set()`'s signature and docstring
by parsing the docstring of all ~265 setters, and each leaves its panel with a
class no other panel shares, so no type-level cache is reused either. The class
bodies close over nothing but their base, so
`share_anesthetic_axes_subclasses()` lets the first panel build the class as
usual and rebinds the rest onto it, one class per (helper, base type).

Together: corner plot 0.499s -> 0.456s (-8.5%), five-run cold build 1.207s ->
1.171s in-container and 3.6% less CPU, with byte-identical PNGs. Best-effort
like the rest - if matplotlib or anesthetic renames either private hook, the
plot is just slower.

Two of matplotlib's own lookup helpers rebuild the same answer on every call,
which only shows up at this call volume. `_axis_map` - how an `Axes` finds its
`XAxis`/`YAxis`, read ~4500 times per corner plot - is a property that builds a
fresh dict from two f-strings and two `getattr`s each time;
`cache_matplotlib_axis_map()` keeps the dict on the instance and validates it by
identity against `self.xaxis`/`self.yaxis`, so the one place matplotlib swaps
those objects (`_init_axis`) misses the cache instead of getting a stale map.
Projections whose `_axis_names` is not `("x", "y")` fall through to the original
property. And `cbook.normalize_kwargs`, which every artist constructor and every
`.set()` call routes its kwargs through (~3400 times per corner plot), flattens
the artist class' alias map (`{'linewidth': ['lw'], ...}`) into an
alias -> canonical dict from scratch on each call; that flattening is a pure
function of the class, so `memoize_matplotlib_alias_maps()` memoises it per
class and keeps matplotlib's duplicate-alias `TypeError`. Callers that pass a
plain dict keep the original path. Like the shared-axis scan in iteration 16,
the patch has to rebind `normalize_kwargs` in every module that did
`from ... import normalize_kwargs`, because the hot call sites are not in the
defining module.

Together: corner plot 2.330s -> 2.262s CPU over the five runs (-2.6%), five-run
cold build 1.167s -> 1.157s wall and 1.0% less worker CPU in-container, with
byte-identical PNGs. The build's wall clock moves less than its CPU because
half of it is the parent's serial import prologue, which no plot-side change
touches.

`main()` disables the cyclic garbage collector. The report is a batch process -
it draws its figures, writes the pages and exits - so nothing it allocates has
to be reclaimed while it runs, and matplotlib figures are dense enough in
reference cycles that the collector wakes constantly and finds almost nothing
refcounting would not have freed anyway. Turning it off is 12% of a corner
plot's render (0.41s -> 0.37s in-process) and 9% of the whole build (five-run
cold build 1.060s -> 0.962s in-container), with byte-identical PNGs; the cycles
it would have collected are held to exit instead, which measured as +3MB on a
worker's 128MB peak. It is disabled in `main()` rather than at import, so
importing the module - the self-checks, host-side use - leaves the caller's gc
alone; the pool workers inherit the setting across fork. This is the one lever
that helps the serial import prologue as well as the plots.

`load_plot_libs()` skips matplotlib's 3d projection. `matplotlib.projections`
imports `mpl_toolkits.mplot3d` solely to register the `'3d'` projection, inside
a `try`/`except` that already degrades to "3d unavailable" when the import
fails. Nothing here draws in 3d - a corner plot is a grid of 2d axes - so a
`None` is parked in `sys.modules` for the duration of the `import
matplotlib.pyplot`, which makes that import raise `ImportError` and take the
branch matplotlib already handles. The entry is removed again immediately, so a
later `import mpl_toolkits.mplot3d` still works; only the registry entry stays
gone, which turns an accidental `projection='3d'` into a loud `ValueError`
rather than a silently wrong plot. Worth 17ms of the 258ms `import
matplotlib.pyplot` (-6.7%), median of 15 interleaved runs with the two
distributions not overlapping at all.

Alongside it, `multiprocessing` (3.4ms) is imported only once `main()` has runs
to fan out, and `tempfile`/`shutil` (2.0ms) only inside the self-checks that
use them, so a rebuild with nothing to do stops paying for either: its CPU
drops from 0.093s to 0.081s (-13%). Together with the 3d skip the parent's
serial import prologue - `import generate_report` plus matplotlib plus
anesthetic, which is over half a cold build's critical path - goes from 600.6ms
to 577.0ms (-3.9%), on every build that draws anything.

Two neighbouring import trims are measured dead and should not be retried:
`pyparsing` (29ms) cannot be dropped with the `_fontconfig_pattern` it first
appears under, because `matplotlib._mathtext` imports it anyway for the math
labels; and `importlib.metadata` (23ms) cannot be stubbed for anesthetic's
`__version__`, because pandas' plotting-backend loader needs its real
`entry_points()` on the same path.

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

### Read the report from another machine

Searches normally run on a headless remote host, where there is no browser to
open `index.html` with. `./ri serve` puts the report behind an HTTP server:

```bash
./ri serve                  # loopback:8000
./ri serve --port 9000      # [REPORT_PORT]
./ri serve --bind 0.0.0.0   # [REPORT_BIND]
```

It binds to `127.0.0.1` and prints the tunnel command to run on your own
machine:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<host>
# then open http://localhost:8000/
```

Nothing goes out to the network: the loopback bind means the only route in
from another machine is an SSH session the host already accepts, and the tunnel
runs at your end. It is not an access control, though - the report is
unauthenticated, so anyone with an account on that host, and any container
sharing its network namespace, can read it over loopback without SSH. On a
shared login node that is worth knowing before starting one. `--bind 0.0.0.0`
drops even that and serves the report to anything that can reach the host.

The `ssh -L` line forwards to whatever `--bind` actually bound, and guesses the
host address from `hostname -I` - right on a cloud box and wrong behind NAT, so
`REPORT_SSH_HOST` overrides it.

The server is `python3 -m http.server` from the host's standard library -
no Docker, no dependencies - and runs in the foreground until Ctrl-C. It is
also why copying a single page off the host does not work: the pages link to
sibling PNGs under `images/`, which is most of the report's bytes, so the whole
directory has to travel together (`rsync -a`) or be served in place.

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
`[[parameter_space]]` in `defaults.toml`, copied into
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
