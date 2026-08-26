# Cheatsheet

Every command in one place. `./ri` is the front door - `./ri --help` lists the
whole surface and every subcommand has its own `--help`. Authoritative sources:
`ri`, `scripts/`, `defaults.toml`, `versions.env`, `README.md`,
`docs/nested-sampling.md`.

Run everything from the repository root. Host prerequisites: Docker, `git`, `uv`.

Anything the CLI would run can be previewed instead:

```bash
./ri --dry-run search wsclean --nlive 8    # prints the env and commands, runs nothing
```

## First-time setup

```bash
cp .env.example .env    # mounts, HOST_UID/HOST_GID, thread counts
./ri build              # all four images (slow; the r2d2 image alone is ~3.2 GB)
```

## Images

| Command | Script | Tag |
|---|---|---|
| `./ri build` | `scripts/build.sh all` | all four below |
| `./ri build wsclean` | `scripts/build.sh wsclean` | `ri-reproducibility/wsclean:v3.7` |
| `./ri build r2d2` | `scripts/build.sh r2d2` | `ri-reproducibility/r2d2:cpu` |
| `./ri build meqtrees` | `scripts/build.sh meqtrees` | `ri-reproducibility/meqtrees:kern-10` |
| `./ri build polychord` | `scripts/build.sh polychord` | `ri-reproducibility/polychord:lite` |

```bash
# CPU-native WSClean for benchmarking on THIS machine -> tag :native
./ri build wsclean --native      # WSCLEAN_PORTABLE=OFF scripts/build.sh wsclean
```

Never report the default `PORTABLE=ON` build as optimized-WSClean performance.

After editing `scripts/lib/nested_sampling/*`, rebuild **both** `polychord` and
`meqtrees` - they bake those files in at build time and otherwise run stale code.

## Smoke tests

```bash
./ri smoke                 # both imagers
./ri smoke wsclean         # version + a real tiny imaging run
./ri smoke r2d2            # imports -> app modules -> data -> config -> inference
./ri smoke ms-to-mat       # the MS -> R2D2 .mat bridge, before an R2D2 search
```

## Shells into an image

```bash
./ri shell wsclean     # mounts data/, results/
./ri shell r2d2        # mounts data/, checkpoints/, results/ + thread env
./ri shell meqtrees
./ri shell polychord   # mounts repo root + docker.sock
```

## Checkpoints

```bash
./ri fetch-checkpoints R2D2_A1_T2_Realisation1.zip
```

Cannot be fully automated - the host is behind a Cloudflare challenge; the
script detects that and prints the URL plus placement instructions.
R2D2 runs need `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`.

## FITS -> PNG

```bash
./ri plot fits                                                   # standard smoke-test set
./ri plot fits results/smoke-test-wsclean/foo-image.fits
./ri plot fits /opt/r2d2/R2D2-RI/data/3c353_gdth.fits            # bundled ground truth
```

PNGs land flat in `results/`, named after the source FITS file.

## Searches

```bash
./ri search wsclean    # WSClean x VLA.A -> results/nested-sampling/wsclean-vlaa-<UTC>/
./ri search r2d2       # R2D2 x VLA.A    -> results/nested-sampling/r2d2-vlaa-<UTC>/
./ri smoke ms-to-mat   # validate the MS -> .mat bridge before an R2D2 search
```

Each search builds the images it needs first; `--no-build` skips that.

### Overrides (defaults in `defaults.toml`)

Each flag sets its variable for that run, so the two forms below are the same
run. A flag beats an exported variable, and both beat `defaults.toml`.

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--nlive` | `NS_NLIVE` | PolyChord live points | `8` |
| `--num-repeats` | `NS_NUM_REPEATS` | Exploration per replacement live point | `2` |
| `--max-ndead` | `NS_MAX_NDEAD` | Dead-point budget, terminates the run | `12` |
| `--seed` | `NS_SEED` | PolyChord seed | `41` |
| `--metric` | `NS_METRIC` | Objective, see below | `total_rms_jy` |
| `--mpi-procs` | `NS_MPI_PROCS` | Rank count; `1` disables parallel evaluations | `min(NS_NLIVE, host CPUs)` |
| `--omp-threads` | `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS/torch threads | `host CPUs / NS_MPI_PROCS`, min 1 |
| `--output-dir` | `OUTPUT_DIR` | Run directory | `results/nested-sampling/<algo>-vlaa-<UTC>` |

`NS_SIDECARS` and `NS_SIMULATE_FIFO_DIR` are wiring the run scripts export, not
knobs to set by hand.

```bash
./ri search wsclean --nlive 8 --num-repeats 2 --max-ndead 12
./ri search wsclean --mpi-procs 4
./ri search wsclean --mpi-procs 1                    # serial, for debugging
./ri search wsclean --metric badness
./ri search wsclean --metric snr
./ri search r2d2 --metric sigma_res
./ri search wsclean --output-dir results/nested-sampling/manual

NS_NLIVE=8 ./ri search wsclean                       # the same, from the environment
```

### `--metric` / `NS_METRIC` resolution

1. `badness` - the composite score (higher = worse).
2. A bare metric name, used raw: `snr`, `log_snr`, `off_source_rms_jy`,
   `total_rms_jy`, `peak_jy_per_beam`, `relative_l2_error`,
   `peak_flux_abs_error_jy`, `sigma_res`, `wall_seconds`, `peak_memory_bytes`.
3. Anything else - an arithmetic expression over those names, e.g.
   `"log_snr + 0.1 * wall_seconds"`. Compiled at startup, so a typo fails fast.

PolyChord always **maximizes** the returned value, with no automatic sign flip.
To search for the *best* corner of a higher-is-worse metric, negate explicitly.
A negated metric starts with a dash, so use the `=` form of the flag (or the
environment variable):

```bash
./ri search wsclean --metric=-total_rms_jy
./ri search wsclean --metric=-snr                    # worst-SNR search
NS_METRIC="-snr" ./ri search wsclean                 # the same
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

Defined as `PARAMETER_SPACE` in `scripts/lib/nested_sampling/common.py`,
copied into every `summary.json`.

## Reports

```bash
./ri report    # reports/nested-sampling-report/index.html + one page per run
```

Report selectors:

```bash
./ri report --last 1                                   # newest N runs only
./ri report --run results/nested-sampling/<run>    # one run, always rebuilt
./ri report --upgrade                                  # rebuild pages from an older report version
./ri report --force                                    # rebuild everything in scope
```

Up-to-date pages are skipped; the index is always rebuilt. `--last` and `--run`
cannot be combined. Both reports are generated and gitignored - `git add -f` a
copy if you want one version-controlled.

## Profiling

```bash
./ri profile results/nested-sampling/<run> [--json]
uv run scripts/profile-nested-sampling-run.py results/nested-sampling/<run> [--json]
```

Every run is profiled automatically; the same breakdown appears in that run's
HTML report page. Field meanings: `docs/nested-sampling-profiling.md`.

## Merging runs

Post-processing only. Sources must match on `algorithm`, `vla_config`,
`metric`, `parameter_space` and fixed hyperparameters; sampler effort may differ.

```bash
./ri merge                                                      # auto-group every completed run
./ri merge results/nested-sampling/A results/nested-sampling/B [--out DIR]
uv run scripts/merge-nested-sampling-runs.py RUN_A RUN_B [--out DIR]
```

Writes `results/nested-sampling/<algorithm>-vlaa-merged-<UTC>/summary.json`.
`--out` is only valid with an explicit run list. Merged directories are treated
as completed runs by the report and the GUI.

## anesthetic GUI (host, needs a display)

```bash
./ri plot gui                                           # latest completed run
./ri plot gui results/nested-sampling/<run>
uv run scripts/anesthetic-gui.py results/nested-sampling/<run>
```

Not inside Docker/Colima. Needs `anesthetic` (`uv add anesthetic` if missing).

```bash
./ri plot likelihood   # newest comparable merged R2D2 vs WSClean, into reports/
```

## Recording a run

`./ri record` writes the manifest only - it does **not** execute the command.
Run the pipeline yourself, then record the same command verbatim. The searches
call it themselves.

```bash
# 1. run it
docker run --rm \
  -v "$(pwd)/checkpoints:/checkpoints:ro" \
  -v "$(pwd)/results/<experiment>:/results" \
  -v "$(pwd)/config/r2d2:/workspace/config:ro" \
  --entrypoint python3 ri-reproducibility/r2d2:cpu \
  ./src/imager.py --config /workspace/config/R2D2_U-Net.yaml --ckpt_path /checkpoints/R2D2_A1

# 2. record it
./ri record --tool r2d2 \
  --image ri-reproducibility/r2d2:cpu \
  --config config/r2d2/R2D2_U-Net.yaml -- <the exact command above>
```

Then hand-add an `"experiment"` object (`purpose`, provenance, `results`) to the
written manifest in `reports/manifests/`.

## Housekeeping

```bash
./ri disk-usage        # docker system df -v
./ri clean             # this repo's images + smoke-test outputs
docker compose config  # compose file validation
docker builder prune   # BuildKit cache
docker system prune    # Docker-wide, affects other projects - use with care
docker builder prune -a && ./ri build   # true cold rebuild
```

## Checks (what CI runs)

```bash
shellcheck -x scripts/*.sh
bash -n scripts/*.sh
python3 -m compileall -q scripts config
scripts/test-defaults.sh
uv run --no-project scripts/test_cli.py
```

## Layout

| Path | Contents |
|---|---|
| `ri` | The CLI: argument parsing and dispatch into `scripts/`, nothing else |
| `defaults.toml` | Runtime defaults for every script; the environment always wins |
| `versions.env` | Pinned upstream revisions (record-keeping; Dockerfile ARGs kept in sync by hand) |
| `.env` | Mounts, HOST_UID/GID, thread counts - read by Docker Compose only |
| `config/r2d2/`, `config/wsclean/` | Per-tool configs |
| `data/` -> `/data` | Measurement Sets, `.mat` files, ground-truth FITS |
| `checkpoints/` -> `/checkpoints` | R2D2 pretrained checkpoints |
| `results/` -> `/results` | Run output, smoke-test output, nested-sampling runs |
| `reports/manifests/` | One JSON manifest per recorded run |
| `docs/nested-sampling.md` | Nested-sampling design, metrics, output files |
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
