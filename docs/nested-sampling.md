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
NS_METRIC=badness make nested-sampling-poc
NS_METRIC=snr make nested-sampling-poc
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

WSClean runs with fixed hyperparameters on every evaluation: `-niter 100`
and `-auto-threshold 3.0`. These are recorded in `poc-summary.json` under
`wsclean_fixed_hyperparameters`.

Channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

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
| `wall_seconds` | WSClean container runtime |
| `peak_memory_bytes` | Peak WSClean memory from GNU `time`, with Docker stats as a secondary source |

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

`scripts/lib/nested_sampling/polychord_wsclean_poc.py` accepts
`--metric <value>` (default `off_source_rms_jy`). The shell wrapper forwards
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
`--metric "1/snr"`). Failed simulations or WSClean runs still receive objective
`100.0`.

Each evaluation record and `poc-summary.json` store the chosen value in an
`objective` field. `poc-summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

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
