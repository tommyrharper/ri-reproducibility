# Cheatsheet

Every command in one place. Authoritative sources: `Makefile`, `scripts/`,
`defaults.toml`, `versions.env`, `README.md`, `docs/nested-sampling.md`.

Run everything from the repository root. Host prerequisites: Docker, `git`, `uv`.

## First-time setup

```bash
cp .env.example .env    # mounts, HOST_UID/HOST_GID, thread counts
make build              # all four images (slow; the r2d2 image alone is ~3.2 GB)
```

## Images

| Make target | Script | Tag |
|---|---|---|
| `make build` | `scripts/build.sh all` | all four below |
| `make build-wsclean` | `scripts/build.sh wsclean` | `ri-reproducibility/wsclean:v3.7` |
| `make build-r2d2` | `scripts/build.sh r2d2` | `ri-reproducibility/r2d2:cpu` |
| `make build-meqtrees` | `scripts/build.sh meqtrees` | `ri-reproducibility/meqtrees:kern-10` |
| `make build-polychord` | `scripts/build.sh polychord` | `ri-reproducibility/polychord:lite` |

```bash
# CPU-native WSClean for benchmarking on THIS machine -> tag :native
WSCLEAN_PORTABLE=OFF scripts/build.sh wsclean
```

Never report the default `PORTABLE=ON` build as optimized-WSClean performance.

After editing `scripts/lib/nested_sampling/*`, rebuild **both** `polychord` and
`meqtrees` - they bake those files in at build time and otherwise run stale code.

## Smoke tests

```bash
make smoke-test            # both
make smoke-test-wsclean    # version + a real tiny imaging run
make smoke-test-r2d2       # imports -> app modules -> data -> config -> inference
```

## Shells into an image

```bash
make shell-wsclean     # mounts data/, results/
make shell-r2d2        # mounts data/, checkpoints/, results/ + thread env
make shell-meqtrees
make shell-polychord   # mounts repo root + docker.sock
```

## Checkpoints

```bash
make fetch-r2d2-checkpoints REALISATION=R2D2_A1_T2_Realisation1.zip
```

Cannot be fully automated - the host is behind a Cloudflare challenge; the
script detects that and prints the URL plus placement instructions.
R2D2 runs need `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`.

## FITS -> PNG

```bash
make plot-fits                                                   # standard smoke-test set
make plot-fits FILES="results/smoke-test-wsclean/foo-image.fits"
make plot-fits FILES="/opt/r2d2/R2D2-RI/data/3c353_gdth.fits"    # bundled ground truth
```

PNGs land flat in `results/`, named after the source FITS file.

## Nested-sampling PoC

```bash
make nested-sampling-poc         # WSClean x VLA.A -> results/nested-sampling-poc/wsclean-vlaa-<UTC>/
make nested-sampling-r2d2-poc    # R2D2 x VLA.A    -> results/nested-sampling-poc/r2d2-vlaa-<UTC>/
scripts/check-ms-to-r2d2-mat.sh  # validate the MS -> .mat bridge before an R2D2 run
```

### Overrides (defaults in `defaults.toml`)

| Variable | Meaning | Default |
|---|---|---|
| `NS_NLIVE` | PolyChord live points | `8` |
| `NS_NUM_REPEATS` | Exploration per replacement live point | `2` |
| `NS_MAX_NDEAD` | Dead-point budget, terminates the run | `12` |
| `NS_SEED` | PolyChord seed | `41` |
| `NS_METRIC` | Objective, see below | `total_rms_jy` |
| `NS_MPI_PROCS` | Rank count; `1` disables parallel evaluations | `min(NS_NLIVE, host CPUs)` |
| `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS threads | `host CPUs / NS_MPI_PROCS`, min 1 |
| `OUTPUT_DIR` | Run directory | `results/nested-sampling-poc/<algo>-vlaa-<UTC>` |

`NS_SIDECARS` and `NS_SIMULATE_FIFO_DIR` are wiring the run scripts export, not
knobs to set by hand.

```bash
NS_NLIVE=8 NS_NUM_REPEATS=2 NS_MAX_NDEAD=12 make nested-sampling-poc
NS_MPI_PROCS=4 make nested-sampling-poc
NS_MPI_PROCS=1 make nested-sampling-poc              # serial, for debugging
NS_METRIC=badness make nested-sampling-poc
NS_METRIC=snr make nested-sampling-poc
NS_METRIC=sigma_res make nested-sampling-r2d2-poc
OUTPUT_DIR=results/nested-sampling-poc/manual make nested-sampling-poc
```

### `--metric` / `NS_METRIC` resolution

1. `badness` - the composite score (higher = worse).
2. A bare metric name, used raw: `snr`, `log_snr`, `off_source_rms_jy`,
   `total_rms_jy`, `peak_jy_per_beam`, `relative_l2_error`,
   `peak_flux_abs_error_jy`, `sigma_res`, `wall_seconds`, `peak_memory_bytes`.
3. Anything else - an arithmetic expression over those names, e.g.
   `"log_snr + 0.1 * wall_seconds"`. Compiled at startup, so a typo fails fast.

PolyChord always **maximizes** the returned value, with no automatic sign flip.
To search for the *best* corner of a higher-is-worse metric, negate explicitly:

```bash
NS_METRIC="-total_rms_jy" make nested-sampling-poc
NS_METRIC="-snr"          make nested-sampling-poc   # worst-SNR search
```

Failed simulate/imaging evaluations score `100.0`.

### Parameter space (VLA.A only)

| Dimension | Range |
|---|---|
| `dynamic_range` | `1e2` - `1e3` |
| `observation_minutes` | `4` - `10` |
| `channel_count` | `2` - `6` |
| `start_frequency_hz` | `1.0e9` - `1.1e9` |
| `channel_width_hz` | `0.5e6` - `2.0e6` |

Defined as `PARAMETER_SPACE` in `scripts/lib/nested_sampling/poc_common.py`,
copied into every `poc-summary.json`.

## Reports

```bash
make nested-sampling-report    # reports/nested-sampling-report/index.html + one page per run
```

Nested-sampling report selectors:

```bash
make nested-sampling-report LAST=1                                   # newest N runs only
make nested-sampling-report RUN=results/nested-sampling-poc/<run>    # one run, always rebuilt
make nested-sampling-report UPGRADE=1                                # rebuild pages from an older report version
make nested-sampling-report FORCE=1                                  # rebuild everything in scope
```

Up-to-date pages are skipped; the index is always rebuilt. `LAST=` and `RUN=`
cannot be combined. Both reports are generated and gitignored - `git add -f` a
copy if you want one version-controlled.

## Profiling

```bash
make nested-sampling-profile RUN=results/nested-sampling-poc/<run>
uv run scripts/profile-nested-sampling-run.py results/nested-sampling-poc/<run> [--json]
```

Every run is profiled automatically; the same breakdown appears in that run's
HTML report page. Field meanings: `docs/nested-sampling-profiling.md`.

## Merging runs

Post-processing only. Sources must match on `algorithm`, `vla_config`,
`metric`, `parameter_space` and fixed hyperparameters; sampler effort may differ.

```bash
uv run scripts/merge-nested-sampling-runs.py                    # auto-group every completed run
make merge-nested-sampling
uv run scripts/merge-nested-sampling-runs.py RUN_A RUN_B [--out DIR]
make merge-nested-sampling RUNS="results/nested-sampling-poc/A results/nested-sampling-poc/B"
```

Writes `results/nested-sampling-poc/<algorithm>-vlaa-merged-<UTC>/poc-summary.json`.
`--out` is only valid with an explicit run list. Merged directories are treated
as completed runs by the report and the GUI.

## anesthetic GUI (host, needs a display)

```bash
make anesthetic-gui                                     # latest completed run
make anesthetic-gui RUN=results/nested-sampling-poc/<run>
uv run scripts/anesthetic-gui.py results/nested-sampling-poc/<run>
```

Not inside Docker/Colima. Needs `anesthetic` (`uv add anesthetic` if missing).

```bash
uv run scripts/plot-merged-likelihood-compare.py   # newest comparable merged R2D2 vs WSClean
```

## Recording a benchmark run

`record-environment.sh` writes the manifest only - it does **not** execute the
command. Run the pipeline yourself, then record the same command verbatim.

```bash
# 1. run it
docker run --rm \
  -v "$(pwd)/checkpoints:/checkpoints:ro" \
  -v "$(pwd)/results/<experiment>:/results" \
  -v "$(pwd)/config/r2d2:/workspace/config:ro" \
  --entrypoint python3 ri-reproducibility/r2d2:cpu \
  ./src/imager.py --config /workspace/config/R2D2_U-Net.yaml --ckpt_path /checkpoints/R2D2_A1

# 2. record it
scripts/record-environment.sh --tool r2d2 \
  --image ri-reproducibility/r2d2:cpu \
  --config config/r2d2/R2D2_U-Net.yaml -- <the exact command above>

# or via make (manifest only, no trailing command)
make record-environment TOOL=r2d2 IMAGE=ri-reproducibility/r2d2:cpu CONFIG=config/r2d2/R2D2_U-Net.yaml
```

Then hand-add an `"experiment"` object (`purpose`, provenance, `results`) to the
written manifest in `reports/manifests/`.

## Housekeeping

```bash
make config          # docker compose config (validation)
make disk-usage      # docker system df -v
make clean           # this repo's images + smoke-test outputs
docker builder prune # BuildKit cache
docker builder prune -a && make build   # true cold rebuild
docker system prune  # Docker-wide, affects other projects - use with care
```

## Checks (what CI runs)

```bash
shellcheck -x scripts/*.sh
bash -n scripts/*.sh
python3 -m compileall -q scripts config
scripts/test-defaults.sh
```

## Layout

| Path | Contents |
|---|---|
| `defaults.toml` | Runtime defaults for every script; the environment always wins |
| `versions.env` | Pinned upstream revisions (record-keeping; Dockerfile ARGs kept in sync by hand) |
| `.env` | Mounts, HOST_UID/GID, thread counts - read by Docker Compose only |
| `config/r2d2/`, `config/wsclean/` | Per-tool configs |
| `data/` -> `/data` | Measurement Sets, `.mat` files, ground-truth FITS |
| `checkpoints/` -> `/checkpoints` | R2D2 pretrained checkpoints |
| `results/` -> `/results` | Run output, smoke-test output, PoC runs |
| `reports/manifests/` | One JSON manifest per recorded run |
| `docs/nested-sampling.md` | PoC design, metrics, output files |
| `docs/nested-sampling-profiling.md` | Profiling fields, measured optimisations |
| `r2d2-paper/`, `claims/`, `latex/` | Reference material: the R2D2 paper, published claims, our own write-up |

## Gotchas

- `DOCKER_DEFAULT_PLATFORM` should normally be **unset** - the platform is
  derived from `uname -m`. Setting it cross-builds under slow QEMU emulation.
- `torch.cuda.is_available() == False` is correct; the images are CPU-only.
- `exec format error` means something forced the non-host architecture.
- WSClean illegal-instruction: a `PORTABLE=OFF` image built on a different CPU.
- Root-owned files in `results/` on Linux: `sudo chown -R $(id -u):$(id -g) results/`.
- OOM during the casacore build: raise Docker memory, or `--build-arg BUILD_JOBS=1`.
- Docker Desktop on macOS numbers are not representative of native Linux
  (VM CPU/memory limits, virtualized bind-mount I/O, no GPU passthrough).
