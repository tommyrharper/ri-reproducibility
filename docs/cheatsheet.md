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

A build whose inputs (Dockerfile + whatever it copies from the context +
platform + build args) hash to what the existing image's `ri.build-inputs`
label already says is skipped: ~0.08s instead of ~2s. `FORCE_BUILD=1` builds
anyway, for the drift Docker's layer cache does not see either (`apt-get`/`pip`
output under a pinned base image, a moved upstream git ref).
`.dockerignore` keeps host `__pycache__` out of both the context and the hash,
so a `compileall` or a `--self-check` does not force a rebuild.

```bash
# CPU-native WSClean, ~6% more evaluations/s on THIS machine (same v3.7 tag,
# so the next plain build puts the portable binary back)
./ri build wsclean --native      # WSCLEAN_PORTABLE=OFF scripts/build.sh wsclean
./ri search wsclean --native     # ...and carry it into the search's own build
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

## Self-checks

CI runs everything that needs no Docker. These are the rest - a live meqserver,
a real TDL compile, numpy, casacore - so they only run when asked. The working
tree is what runs, inside the image, so no rebuild is needed to check a change;
a run executes the baked copy, so rebuild before starting one. They start no
search and write nothing into `results/`, so they are safe alongside a run.

```bash
./ri self-check            # host-side checks, all three images, then the self-heal kills
./ri self-check simulate   # MeqTrees: skeleton cache, forest reuse, deadlock recovery
./ri self-check wsclean    # the WSClean sampler's checks
./ri self-check r2d2       # the R2D2 sampler's checks
./ri self-check report     # the HTML report's checks (matplotlib fast paths, torn summaries)
./ri self-check self-heal  # kill and hang real searches, check they restart themselves (~90s)
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

#### Tweak these

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--nlive` | `NS_NLIVE` | PolyChord live points | `8` |
| `--num-repeats` | `NS_NUM_REPEATS` | Exploration per replacement live point | `2` |
| `--max-ndead` | `NS_MAX_NDEAD` | Dead-point budget, terminates the run | `12` |
| `--seed` | `NS_SEED` | PolyChord seed, random per run; set it to repeat a run | random |
| `--metric` | `NS_METRIC` | Objective, see below | `total_rms_jy` |

#### Leave these alone

Flags exist; defaults are derived. Leave unset unless you want serial
debugging (`--mpi-procs 1`) or a pinned run directory.

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--mpi-procs` | `NS_MPI_PROCS` | Rank count; `1` is serial | `min(NS_NLIVE, host CPUs)` |
| `--omp-threads` | `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS/torch threads | `host CPUs / NS_MPI_PROCS`, min 1 |
| `--output-dir` | `OUTPUT_DIR` | Run directory | `results/nested-sampling/<algo>-vlaa-<UTC>` |

No flags. The run scripts export these for the containers they start.

| Variable | Meaning |
|---|---|
| `NS_SIDECARS` | Image → sidecar container name |
| `NS_SIMULATE_FIFO_DIR` | Per-rank simulate worker FIFOs |
| `NS_R2D2_FIFO_DIR` | Per-rank R2D2 imaging worker FIFOs |

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
| `start_frequency_hz` | inside one receiver band |
| `channel_width_hz` | `0.5e6` - `2.0e6` |

Defined as `[[parameter_space]]` in `defaults.toml`, read by
`load_parameter_space()` in `scripts/lib/nested_sampling/common.py` and copied
into every `summary.json`.

`start_frequency_hz` is drawn from the `[[receiver_band]]` list in the same
file (the VLA's ten bands), each band getting an equal share of the prior.
Swap the list for another telescope's bands - no code change.

The window is then fitted to the room left in that band: keep it if it fits,
else narrow the channels, else hold the width at its min and drop channels,
else throw the start frequency away and draw another. So the mins are hard
floors, the maxes are never exceeded, and a run can measure narrower channels
than it drew. How often that happens, and what it costs, is in each run's
`summary.json` under `spectral_window_fitting` and printed when the run ends.
See docs/nested-sampling.md.

## Runs that stopped early

A run writes `summary.json` only when it finishes, so one without it stopped -
and the report, which globs for `summary.json`, will not show it at all.

```bash
./ri runs                  # every run, with whether it finished
./ri runs --incomplete     # only the ones that stopped
./ri resume <run>          # continue one, keeping the evaluations it has
```

`resume` takes no settings: the run recorded its own in `run.env`.

Every command that takes a run - `resume`, `health`, `profile`, `merge`,
`plot gui` - takes either a path or the bare name `./ri runs` prints, so the
name can be copied straight out of that table. A path of the same name in the
working directory still wins. `health` also takes the name of a run another
checkout started while it is running, because it reports those too.

## All of that from one screen

```bash
./ri tui                   # run table, live health, and a form that starts a search
```

A terminal interface over the three commands above: the table is `./ri runs`,
in its order (newest run at the top, whichever imager it belongs to) and with
its `started` column, `enter` shows `./ri health` for the selected run and
re-runs it every 5 seconds, `l` swaps that for the tail of the run's `run.log`,
`a` narrows the table to what is running, and `n` opens a form that starts a
search. Keys are listed along the bottom of every screen.

A search started there is detached, so quitting the interface leaves it going.
It joins the table as `starting` the moment it is launched - the search builds
its images before it claims a run directory, so `./ri runs` cannot see it for a
while - and `enter` on it shows the output of that build, in
`results/tui-<run>.log`, which is where anything that goes wrong before the run
directory exists is said. `l` swaps to `./ri health` from there too.

Needs Go on the host; nothing else in this repo does. The source is `tui/`,
and `go -C tui test ./...` checks it.

## Is the run that is going still worth going?

```bash
./ri health                # every live run, plus host memory and leaked sidecars
./ri health <run>
./ri health --all
./ri health --monitor      # redraw in place every 5s instead of printing once
```

`STARTING` / `RESTARTING` / `HEALTHY` / `STALLED` / `STOPPED` / `FINISHED`,
with the rate, the share of wall clock lost to stalls, and how many
evaluations scored `FAILURE_OBJECTIVE` - which PolyChord maximizes, so a run
whose imager is broken looks like a run finding spectacular failures. Exits 1
when something needs attention. Details: docs/run-health.md.

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

Reading the report from a browser on another machine:

```bash
./ri serve                       # loopback:8000, prints the ssh -L line to run locally
./ri serve --port 9000
./ri serve --bind 0.0.0.0        # no tunnel; unauthenticated to the network
```

The server binds to loopback, so nothing is served to the network - the
`ssh -L` tunnel it prints runs on your own machine. The report is
unauthenticated, so a loopback bind still leaves it readable by anyone with an
account on this host; `--bind 0.0.0.0` drops even that. Foreground; Ctrl-C
stops it.

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
shellcheck -x scripts/*.sh scripts/lib/*.sh   # info-level findings fail the build too
bash -n scripts/*.sh scripts/lib/*.sh
python3 -m compileall -q scripts config
NESTED_SAMPLING_RUNS_SELF_CHECK=1 uv run --no-project python3 scripts/nested-sampling-runs.py
NESTED_SAMPLING_HEALTH_SELF_CHECK=1 uv run --no-project python3 scripts/nested-sampling-health.py
for f in rank-budget start-sidecars run-config progress-bar; do bash scripts/lib/$f.sh --self-check; done
bash scripts/resume-nested-sampling-run.sh --self-check
uv run --no-project python3 scripts/test_watchdogs.py
uv run --no-project python3 scripts/test_self_checks.py
scripts/test-defaults.sh
uv run --no-project scripts/test_cli.py
```

Everything above needs no Docker. `./ri self-check` is the half that does.

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
| `docs/run-health.md` | `./ri health`: every line of the report and what warns |
| `docs/robustness.md` | Failure handling, self-healing restarts, `./ri resume` |
| `docs/nested-sampling-profiling.md` | Profiling fields, measured optimisations |
| `docs/nested-sampling-throughput.md` | Throughput: why the ranks idled, why rank 0 is not one, where an evaluation's time goes, and what does and does not make it cheaper |
| `docs/parameter-space-proposal.md` | What to add to the searched space next, ranked |
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
