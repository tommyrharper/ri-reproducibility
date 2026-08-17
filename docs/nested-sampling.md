# Nested Sampling PoC

This repo uses PolyChord as a targeted search tool, not as a Bayesian
posterior fit. The likelihood is a configurable objective metric (default
`off_source_rms_jy`). PolyChord maximizes whatever metric is selected.

Ground truth for every run is one unpolarized 1 Jy point source at phase
centre. Dynamic range is controlled by complex Gaussian thermal noise in the
simulated visibilities.

## Images

```bash
make build-wsclean
make build-r2d2
make build-meqtrees
make build-polychord
```

`make build` also builds the MeqTrees and PolyChord images.

The MeqTrees image uses KERN 10 packages on Ubuntu 24.04. The VLA.A antenna
table is unpacked from makems' bundled `VLAA_ANT` example inside the image,
so antenna positions are not hand-rolled in this repo. Visibilities for that
skeleton are predicted by an actual MeqTrees/Meow point-source RIME run
(`scripts/lib/nested_sampling/point_source_forest.py`, driven through
`meqtree-pipeliner.py`), not a hand-rolled formula; thermal noise is added on
top of that clean MeqTrees prediction.

## Run The PoC

### WSClean

```bash
make nested-sampling-poc
```

Outputs are written under:

```text
results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>/
```

Useful overrides:

```bash
NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 make nested-sampling-poc
NS_MPI_PROCS=4 make nested-sampling-poc
NS_METRIC=badness make nested-sampling-poc
NS_METRIC=snr make nested-sampling-poc
OUTPUT_DIR=results/nested-sampling-poc/manual make nested-sampling-poc
```

PolyChord likelihood evaluations run in parallel across MPI ranks inside the
PolyChord container. `NS_MPI_PROCS` sets the rank count (default
`min(NS_NLIVE, host CPUs)`). Set `NS_MPI_PROCS=1` to disable parallel
evaluations for debugging.

The target builds any missing WSClean, MeqTrees, and PolyChord images first,
then runs the PolyChord container. That container mounts the Docker socket and
launches one MeqTrees simulation container plus one WSClean imaging container
per likelihood evaluation.

### R2D2

```bash
make nested-sampling-r2d2-poc
```

Outputs are written under:

```text
results/nested-sampling-poc/r2d2-vlaa-<UTC timestamp>/
```

Use the same `NS_*` and `OUTPUT_DIR` overrides as the WSClean PoC. The target
builds R2D2, MeqTrees, and PolyChord images first, then runs the PolyChord
container. Each likelihood evaluation launches one MeqTrees simulation container,
one MeqTrees-hosted MS-to-`.mat` conversion, and one R2D2 imaging container.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `make fetch-r2d2-checkpoints` and `make smoke-test-r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
scripts/check-ms-to-r2d2-mat.sh
```

## Parameter Space

VLA configuration is an outer-loop dimension. This PoC only runs `VLA.A`.

PolyChord dimensions for both algorithm PoCs:

| Dimension | PoC range | Meaning |
|---|---:|---|
| `dynamic_range` | `1e2` to `1e3` | One-Jy source divided by thermal-noise sigma |
| `observation_minutes` | `4` to `10` | Total requested observing time |
| `channel_count` | `2` to `6` | Number of frequency channels |
| `start_frequency_hz` | `1.0e9` to `1.1e9` | First channel frequency |
| `channel_width_hz` | `0.5e6` to `2.0e6` | Uniform spacing between channels |

WSClean runs with fixed hyperparameters on every evaluation: `-niter 100`
and `-auto-threshold 3.0`. These are recorded in `poc-summary.json` under
`wsclean_fixed_hyperparameters`.

R2D2 runs with fixed hyperparameters on every evaluation: `128x128` image
size (matching the WSClean PoC's `-size 128 128 -scale 1asec` footprint),
`num_iter 25`, `architecture unet`, `num_chans 64`, `ckpt_path
/checkpoints/R2D2_A1`, and `ckpt_realisations 1`. These are recorded in
`poc-summary.json` under `r2d2_fixed_hyperparameters`.

Each R2D2 imaging container is launched with OpenMP/BLAS thread env vars
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) set from the
host's available CPU count, overridable via `R2D2_OMP_THREADS`. The
previous image default of `OMP_NUM_THREADS=4` capped finufft/OpenMP work
when the Docker VM exposed more CPUs than four.

`run-nested-sampling-r2d2-poc.sh` runs `NS_MPI_PROCS` PolyChord ranks
concurrently, each launching its own R2D2 container. To avoid CPU
oversubscription, the script defaults `R2D2_OMP_THREADS` to `host CPUs /
NS_MPI_PROCS` (minimum `1`) when not set explicitly, so each rank's R2D2
container gets a fair share of the host's cores instead of all of them. Set
`R2D2_OMP_THREADS` explicitly to override this per-rank default.

Channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

## MS To R2D2 `.mat` Bridge

R2D2-RI reads visibilities from a MATLAB `.mat` file via `load_data_to_tensor()`
in the upstream `src/utils.py`. The nested-sampling simulator produces a CASA
Measurement Set (`sim.ms`) that WSClean consumes directly. The R2D2 PoC adds
`scripts/lib/nested_sampling/ms_to_r2d2_mat.py`, which runs inside the MeqTrees
image (python3-casacore plus scipy) and writes the minimal field set R2D2
loads without flag metadata:

| Field | Meaning |
|---|---|
| `u`, `v` | UV coordinates in wavelengths, flattened across rows and channels |
| `y` | Complex visibilities for correlation index 0 (parallel-hand Stokes I) |
| `nW` | `sqrt(WEIGHT)` from the MS (sqrt of inverse variance) |

Imaging weights are generated inside R2D2 when `data_weighting: True` in the
per-evaluation YAML config. The converter does not replicate the bundled
`data_3c353.mat` pruning or tau-compressed weight fields.

## Metrics And Objective

For each sample, the pipeline records:

| Metric | Source |
|---|---|
| `snr` | Reconstructed image peak divided by off-source RMS |
| `log_snr` | `log10(snr)` |
| `off_source_rms_jy` | Off-source RMS in Jy/beam |
| `peak_jy_per_beam` | Peak absolute flux in the reconstructed image |
| `relative_l2_error` | Image residual versus the one-pixel point-source truth |
| `peak_flux_abs_error_jy` | Absolute centre-pixel flux error |
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

1. `badness` — the composite formula above.
2. Any bare metric name from the table — use that raw value directly as the
   objective (including the default `off_source_rms_jy`).
3. Any other string — treat it as an arithmetic expression over the same metric
   names (for example `log_snr + 0.1 * wall_seconds`, or the composite formula
   rewritten by hand).

Expressions are compiled once at startup (before any Docker evaluations) and
evaluated in a restricted namespace: no Python builtins, metric names as
locals, and `math` module functions available by name. A typo or unsafe
expression fails immediately at startup.

PolyChord always maximizes the returned value with no automatic sign flip. The
`badness` composite is oriented so higher is worse. Raw metrics keep their
natural orientation: the default `off_source_rms_jy` search prefers higher
off-source RMS, `--metric snr` searches for the highest-SNR corner, and a
worst-SNR search must negate explicitly (`--metric "-snr"` or
`--metric "1/snr"`). Failed simulations or imaging runs still receive objective
`100.0`.

Each evaluation record and `poc-summary.json` store the chosen value in an
`objective` field. `poc-summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

## Output Files

### WSClean

Each likelihood evaluation gets:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/wsclean/recon-image.fits
evaluations/eval-*/wsclean/recon-residual.fits
evaluations/eval-*/metrics.json
```

### R2D2

Each likelihood evaluation gets:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/r2d2_data.mat
evaluations/eval-*/r2d2_config.yaml
evaluations/eval-*/r2d2/r2d2_data/R2D2_model_image.fits
evaluations/eval-*/metrics.json
```

The run-level summary is:

```text
poc-summary.json
```

View completed runs (settings, evidence, per-evaluation metrics and
reconstructions) in the shared benchmark HTML report:

```bash
make benchmark-report
# open benchmarks/report.html
```

The report globs `results/nested-sampling-poc/*/poc-summary.json` directly
(no manifest join). It parses PolyChord `chains/*.stats` for log(Z), renders
the shared synthesized ground-truth image once per run, followed by a
per-evaluation card grid (reconstruction, objective, and searched
parameters, with a collapsed raw metrics table as secondary reference), and
best-effort posterior plots via `anesthetic` in the r2d2 image.

### Replay a run in anesthetic's GUI

For an interactive nested-sampling replay (live points vs \(\ln X\), \(\beta\)
tempering) with human-readable parameter labels, run on the **host** (needs a
display; not inside Docker/Colima):

```bash
make anesthetic-gui
make anesthetic-gui RUN=results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
uv run scripts/anesthetic-gui.py results/nested-sampling-poc/wsclean-vlaa-<UTC timestamp>
```

With no `RUN=`, the latest `results/nested-sampling-poc/*/` directory is used.
The script writes/refreshes `chains/<root>.paramnames` from that run's
`poc-summary.json` / `parameter-space.json` so anesthetic shows names such as
`log10_dynamic_range` instead of `0, 1, 2…`. Requires the host `uv` project
dependency `anesthetic` (`uv add anesthetic` if missing).

The run also writes a standard environment manifest through
`scripts/record-environment.sh`.

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
