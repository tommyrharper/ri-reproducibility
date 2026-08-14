# Nested Sampling PoC

This repo uses PolyChord as a targeted search tool, not as a Bayesian
posterior fit. The likelihood is a badness score: higher values mean worse
single-point-source reconstruction.

Ground truth for every run is one unpolarized 1 Jy point source at phase
centre. Dynamic range is controlled by complex Gaussian thermal noise in the
simulated visibilities.

## Images

```bash
make build-wsclean
make build-meqtrees
make build-polychord
```

`make build` also builds the MeqTrees and PolyChord images.

The MeqTrees image uses KERN 10 packages on Ubuntu 24.04. The VLA.A antenna
table is copied from Cattery's bundled `VLAAA_ANTENNA` makems example inside
the image, so antenna positions are not hand-rolled in this repo.

## Run The PoC

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
OUTPUT_DIR=results/nested-sampling-poc/manual make nested-sampling-poc
```

The target builds any missing WSClean, MeqTrees, and PolyChord images first,
then runs the PolyChord container. That container mounts the Docker socket and
launches one MeqTrees simulation container plus one WSClean imaging container
per likelihood evaluation.

## Parameter Space

VLA configuration is an outer-loop dimension. This PoC only runs `VLA.A`.

PolyChord dimensions for the WSClean PoC:

| Dimension | PoC range | Meaning |
|---|---:|---|
| `dynamic_range` | `1e2` to `1e3` | One-Jy source divided by thermal-noise sigma |
| `observation_minutes` | `4` to `10` | Total requested observing time |
| `channel_count` | `2` to `6` | Number of frequency channels |
| `start_frequency_hz` | `1.0e9` to `1.1e9` | First channel frequency |
| `channel_width_hz` | `0.5e6` to `2.0e6` | Uniform spacing between channels |
| `wsclean_niter` | `25` to `150` | CLEAN iteration cap |
| `wsclean_auto_threshold` | `1.5` to `5.0` | WSClean auto-threshold in sigma |

ponytail: channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

## Metrics And Badness

For each sample, the pipeline records:

| Metric | Source |
|---|---|
| `snr` | Reconstructed image peak divided by off-source RMS |
| `log_snr` | `log10(snr)` |
| `relative_l2_error` | Image residual versus the one-pixel point-source truth |
| `peak_flux_abs_error_jy` | Absolute centre-pixel flux error |
| `wall_seconds` | WSClean container runtime |
| `peak_memory_bytes` | Peak WSClean memory from GNU `time`, with Docker stats as a secondary source |

The badness score is:

```text
max(0, 3 - log_snr)
+ min(relative_l2_error, 10)
+ 0.05 * min(wall_seconds / 60, 5)
+ 0.02 * min(peak_memory_bytes / 2 GiB, 5)
```

PolyChord receives this badness score as its log-likelihood. That means
high-likelihood samples are bad reconstructions or expensive runs.

## Output Files

Each likelihood evaluation gets:

```text
evaluations/eval-*/sim.ms
evaluations/eval-*/simulation.json
evaluations/eval-*/wsclean/recon-image.fits
evaluations/eval-*/wsclean/recon-residual.fits
evaluations/eval-*/metrics.json
```

The run-level summary is:

```text
poc-summary.json
```

The run also writes a standard environment manifest through
`scripts/record-environment.sh`.

## Deferred

Deferred deliberately:

- R2D2 through PolyChord.
- VLA.B and VLA.D.
- Full parameter-space exploration.
- VLA.C production exploration, although VLA.A and VLA.C are the prioritized
  follow-up configurations.
- A general multi-algorithm or multi-VLA orchestrator.

To add the next run family, reuse the simulator and metrics modules, keep VLA
configuration as the outer loop, and add only the missing algorithm-specific
runner.
