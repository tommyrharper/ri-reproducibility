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
# WSClean is built for x86-64-v3 (AVX2+FMA) by default. --native goes one step
# further, to THIS machine only (same v3.7 tag, so the next plain build puts the
# default binary back); WSCLEAN_TARGET_CPU= goes back to the plain x86-64 baseline
./ri build wsclean --native      # WSCLEAN_TARGET_CPU=native scripts/build.sh wsclean
./ri search wsclean --native     # ...and carry it into the search's own build
```

Never report a `WSCLEAN_TARGET_CPU=` build as this repo's WSClean performance.

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
./ri self-check zygote     # the WSClean fork server's request/reply protocol
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
./ri profile results/nested-sampling/<run> --phases   # inside the wsclean binary
./ri profile results/nested-sampling/<run> --over-time  # evals/s against wall clock
uv run scripts/profile-nested-sampling-run.py results/nested-sampling/<run> [--json]
```

Every run is profiled automatically; the same breakdown appears in that run's
HTML report page. Field meanings: `docs/nested-sampling-profiling.md`.

`--phases` goes a level down, into the `wsclean` binary itself: every evaluation
runs with `-log-time`, so its `wsclean.stdout.log` is a microsecond phase
timeline and the flag aggregates all of them. What it reads on the current tree,
and what is and is not left to win there:
`docs/nested-sampling-phase-profile.md`.

`--over-time` answers the other question: why the run got slower. It prints
evaluations/second against wall clock beside the visibility count that sets it,
because an evaluation costs a constant plus a rate times its visibilities and
the sampler walks towards the most of them -
`docs/nested-sampling-cost-model.md`.

`--phases` reports each bucket's *mean*, which is right for "where did the run's
seconds go" and wrong for "what should I work on": several buckets are
heavy-tailed. `docs/nested-sampling-simulate-stage.md` has the same table on
per-evaluation medians, and which rows it reorders.

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
| `docs/nested-sampling-throughput.md` | Throughput: why the ranks idled, why rank 0 is not one, where an evaluation's time goes, why the concurrency wall is the all-core clock rather than memory bandwidth, and what does and does not make an evaluation cheaper (asynchronous MPI, the `x86-64-v3` WSClean build and `--native` on top of it, the phase-centre predict, and the two constant `WEIGHT`/`SIGMA` columns the simulator no longer writes); and why a run's disk footprint, its rank memory and its own polling loops, not its clock, are what cap a big one; and the `nlive` scan showing utilisation *rises* with run size (93.9% at `--nlive 25`, 98.3% at `--nlive 200`) because the unaccounted time is a ~6s per-run constant; and why the Measurement Set is built in a tmpfs every container shares (`NS_SCRATCH_DIR`) instead of on the disk, and why `image_container_overhead` was GNU `time`'s 10ms clock rather than a cost until the fork server replaced it |
| `docs/nested-sampling-power-limit.md` | Why the host is the bottleneck now that the software budget is exhausted: the package runs at a 65W long-term RAPL limit against its own 117W rated turbo, which is what sets the all-core clock, costs ~26% of the evaluations per second and (via the 8-second averaging window) is the real cause of the burst-clock measurement trap; with the root command that lifts it, the thermal evidence that there is headroom for it, and the fresh worker-count throughput scan |
| `docs/nested-sampling-io-placement.md` | Where an evaluation's bytes go: why the simulator now assembles the Measurement Set in the destination directory rather than copying it between two tmpfs mounts (-6.5ms an evaluation, 28% of the simulate stage), why WSClean's `-temp-dir` and its five FITS images are already free where they are, why `--mpi-procs` above `nproc` buys nothing (and will not even launch without `--oversubscribe`), and the simultaneous paired A/B design that stops this host's ~4% drift reading as a result |
| `docs/nested-sampling-evaluation-budget.md` | What one evaluation's ~280ms is actually made of at the concurrency a search runs at, from WSClean's own phase line in every `wsclean.stdout.log`: 69% clean loop, 20% first-inversion path, 11% process start, 6% simulate, 0.6% metrics - with the replay rig that splits the 24% WSClean does not time, why the 30ms of process start does not come off, the `/proc/cpuinfo` open that costs 20ms cold and 0.04ms warm, the simultaneous-arm screen of every result-preserving WSClean flag (all zero; `-wgridder-accuracy` is the one with headroom and it is unaffordable here), and the note that no R2D2 run has ever been profiled here |
| `docs/nested-sampling-clean-loop.md` | The 69% of an evaluation that is WSClean's clean loop, and the one lever that moves it: `-mgain` sets how deep each minor loop goes, so `-niter 100` buys ~6.5 major cycles at 0.8 and ~4.7 at 0.9, worth +20% evaluations per second over three interleaved pairs of real searches - with the 600-Measurement-Set replay table, the re-scored metric differences that make it result-preserving for `total_rms_jy` (1e-7 median) but not for `peak_flux_abs_error_jy` or `sigma_res`, and why it is left at 0.8 |
| `docs/nested-sampling-ms-open.md` | The largest per-evaluation item outside the gridding arithmetic: WSClean re-opens the parent Measurement Set 15.7 times an evaluation and casacore attaches every subtable on each open, so the simulator deletes the six nothing reads - +14.9% evaluations per second for the first five and another -3.2% on the `wsclean` binary for `FEED`, bit-identical images - with the `-log-time` phase timeline that found it, the replay tables and their nulls, why the drop cannot move into the cached skeleton, the screen showing every other subtable kills WSClean and its columns cannot be stripped either, the refreshed per-evaluation budget, and what a kept MS is no longer |
| `docs/nested-sampling-wsclean-patches.md` | The local patches `docker/wsclean/patches/` applies to the pinned WSClean tree, why the directory exists, and what each one is worth. 0001 caches the reordered provider's antenna names: WSClean re-opened the parent Measurement Set once per gridding and degridding pass purely to read the `ANTENNA` table, and caching it is +10.5% evaluations per second end to end, -13.1% on the `wsclean` binary, 790 FITS data blocks identical - with the `-log-time` before/after, the three-arm replay and its null, the swapped simultaneous-search pairs and their null, and why a replay corpus on ext4 overstates an MS-open win against a `sim.ms` on tmpfs |
| `docs/nested-sampling-wsclean-zygote.md` | Why 27ms of every 163ms `wsclean` process runs before `main()` does - casacore's static initialisers across 73 shared objects, priced per library with `LD_PRELOAD` - and the `wsclean-zygote` fork server that pays it once per rank instead: +8.4% evaluations per second end to end over eight simultaneous swapped pairs, 200 FITS data blocks identical, with the `exec`-to-first-log-line measurement that found it, why `image_binary_seconds` and peak RSS now come from `wait4()` rather than a forked `/usr/bin/time`, why the parent must stay single-threaded, and why the parent-warm-up follow-up is closed (0.94ms, not the ~11ms it was estimated at) |
| `docs/nested-sampling-phase-profile.md` | Where a post-zygote evaluation's 191ms goes, refreshed on a 5312-evaluation search: 84% is the `wsclean` binary, and inside it 48% is ducc0's gridding and degridding passes, 10% deconvolution, 6.7% *fitting the Gaussian beam to the PSF* (the largest item that is not imaging arithmetic, and one every evaluation pays twice because the theoretical beam under-estimates the fitted one), 21% metadata and I/O. `-log-time` is now passed by default - measured free against a 1.8%-resolution null pair - so `./ri profile <run> --phases` reads that table off any run with no rig. Also closes two avenues with numbers: pre-warming the zygote parent (0.94ms) and WSClean's remaining parent-MS opens (0.51ms each over a plain `Table`) |
| `docs/nested-sampling-cost-model.md` | Why a run's throughput falls as it goes, and what it costs: an evaluation is `100.4 ms + 5.64 us x visibilities` at production concurrency, and nested sampling compresses towards the long-observation, many-channel corner, so evaluations/second drops 15% over six minutes with nothing degrading (`./ri profile <run> --over-time`). Sets the priority for future work - the *fixed* half shrinks in share as a run goes deeper, so pass-count levers (`-mgain`) matter more than metadata ones. Ships `-data-column DATA` (-1.0% on the `wsclean` binary against a 0.1%-resolution null, 1000 FITS data blocks identical) and closes `-gridder tuned-wgridder`, `-gridder wtowers` and the beam fit's retry (which triggers iff fitted > 1.25x theoretical, independent of `-beam-fitting-size`) |
| `docs/nested-sampling-simulate-stage.md` | The MeqTrees stage: 8.4% of a run's worker time, a fifth of it work with no consumer - a DATA column read back only for its shape, a second open of the same table only for its correlation count, six subtables copied out of the skeleton cache only to be deleted, and three whole copies of DATA allocated to add noise. Removing all four is -20% on the stage over two swapped pairs of simultaneous searches (15.1ms -> 12.1ms an evaluation, ~1.7% end to end) with a bit-identical Measurement Set. Also: why the phase table has to be read on medians (the beam fit is 10.6ms on the mean and 6.25ms on the median), that there is no ducc0 warm-up for the zygote parent to inherit, and that WSClean's two parent-MS opens before the imaging table are now the largest non-arithmetic item at 8.8% of an evaluation |
| `docs/nested-sampling-shared-ms-open.md` | The last of the Measurement-Set-open findings: a `wsclean` run constructs a `casacore::MeasurementSet` over its input five times before it reads a visibility, and `docker/wsclean/patches/0003` hands out a copy of one open handle instead - 21 `table.dat` opens down to 4, -3.7% on the `wsclean` binary in an interleaved tmpfs replay against a 0.03% null and -4.4%/-6.1% in two simultaneous swapped searches, 800 FITS data blocks identical. Also: the 30-line `LD_PRELOAD` shim that counts opens with no rebuild, why the cache must be leaked (casacore's own static table cache is torn down first and the process dies *after* writing its images), and the closed follow-up of pre-opening a Measurement Set in the zygote parent (0.7ms an evaluation) |
| `docs/nested-sampling-fftw-planner.md` | Why a `wsclean` process spends 6.3ms of ~56ms asking FFTW for 63 transform plans it already knows - `schaapcommon`'s `Convolve()` builds and destroys four 1-D plans per call and `Resampler` two 2-D ones per gridding pass - and the warm-up in the `wsclean-zygote` parent that pays the once-per-size half for every child: -6.9% on the `wsclean` binary over 960 interleaved tmpfs replay pairs against a -0.2% null, -5.0% over four simultaneous swapped searches, 400 FITS data blocks identical. Also: the `LD_PRELOAD` plan-counting shim, why 142 is a bad FFT size and why fixing that is not result-preserving, and the 1.8ms of repeat plan builds `docker/wsclean/patches/0004` has since taken |
| `docs/nested-sampling-gridder-floor.md` | Where the other 58% of an evaluation goes and why it does not move: ducc0's own `TimerHierarchy` (reachable by changing one hard-coded `0` to a `2` in `wgridder/wgriddingmsgridder.cpp`) prices a pass at 57% gridding proper, 23% FFT, 11% corrections, 3% index, and every pass builds a six-plane w-cube for a w-range spanning 0.7% of one plane because `nplanes` has a kernel-support floor - turning it off is -29% on the `wsclean` binary for a 2.3e-5 change in the restored image, priced but not taken. Also ships `docker/wsclean/patches/0004`, the `schaapcommon` FFTW plan cache: 64 plan builds an evaluation down to 12, -2.7% in an interleaved tmpfs replay and -3.2% over four simultaneous swapped searches, 520 FITS data blocks identical - plus the refreshed median phase table and the trap that two arms' FITS files never byte-compare equal because the header records the `-name` path |
| `docs/nested-sampling-row-blocks.md` | The last non-arithmetic item in the phase table: WSClean's reorder reads its input Measurement Set one row at a time - 13 casacore column reads a row, four of them re-reading what `NextRow()` just threw away, plus a whole extra pass over `TIME` - and a tiled column costs ~10x more read that way than in blocks, so `docker/wsclean/patches/0005` gives the row providers one forward-only block: 29484 column reads an evaluation down to 27 `getColumnRange()` calls, -48% of the reorder phase, -2.2% on the `wsclean` binary in an interleaved tmpfs replay against a 0.02% null and -4.1% over four simultaneous swapped searches, 530 FITS data blocks identical. Also: the per-column cost table that says a 256-row block already has all of it, why the block is copied rather than referenced, and the forced single-row fallback for sets whose rows differ in shape (identical images, 7.8% slower) |
| `docs/nested-sampling-process-warm-up.md` | The 5 ms a forked `wsclean` child spends between its version banner and `=== IMAGING TABLE ===`: a throwaway `TMARK` patch splits it into `fits_init_cfitsio()` (0.47 ms) and the first `casacore::MeasurementSet` construction in the process (3.8 ms, of which ~1.5 ms is process-global), so the `wsclean-zygote` parent now pays both - the phase falls 7.15 ms to 4.78 ms in all eight of eight swapped simultaneous search pairs against a flat in-log null, -1.9% on the child over 230 interleaved replay pairs against a -0.26% null, 230 FITS data blocks identical. Also: why the restoring-beam fit runs twice per evaluation, and the unsigned `(xi - x_mid)` in `schaapcommon`'s Gaussian fitter that makes WSClean fit its beam on a quarter of the box (75.3% of model evaluations underflow to 0) - a real failure mode, not result-preserving to fix, and the bit-identical half of it is worth 1% |
| `docs/nested-sampling-disk-footprint.md` | What a run costs on disk, which is what caps how big it can be now that an evaluation is ~200 ms: a WSClean evaluation directory was 393.6 KB and 94% of it was FITS, of which WSClean's `model` and `psf` have never had a reader at all and its `dirty` and `residual` are read once, by `compute_image_metrics()`, before the record is written - so all four join `PRUNED_ARTEFACTS`, taking an evaluation to **99.9 KB (3.94x)** for 48.5 us of `unlink()`. Also: the measurement that says `evaluations/` does not need sharding (flat ~45 us to create the 400000th directory, 81 ms to list them), the evaluations-per-`nlive` scaling out of the four archived `--nlive` runs, and the disk ceiling this host now has (477k evaluations -> 1.88M) |
| `docs/nested-sampling-evaluation-floor.md` | Where the floor is, and the one lever above it: the refreshed budget (**126 evaluations/second at 19 workers**, 143 ms an evaluation, 5.3% unaccounted - of which 1.9% is the harness's own Python, measured serially - so there is no idle time left to find), the phase table that puts 59% of the binary in ducc0, and **`./ri search --mgain 0.9`**, which is the only double-digit harness-side lever left and is now a flag rather than a constant in `common.py` (6.5 major cycles down to 4.7, 125.3 ms of logged work down to 107.2, over a simultaneous pair). Also: six avenues closed with numbers - the allocator (4511 minor faults a process), syscalls (3618 calls, all start-up), the 19 idle threads `wsclean -j 1` still clones, the two unread correlations in every `sim.ms`, the harness Python, and the fact that Docker here is **rootless**, so `--privileged` does not reach the RAPL power limit either |
| `docs/parameter-space-proposal.md` | What to add to the searched space next, ranked |
| `r2d2-paper/`, `claims/`, `latex/` | Reference material: the R2D2 paper, published claims, our own write-up |

## Gotchas

- `DOCKER_DEFAULT_PLATFORM` should normally be **unset** - the platform is
  derived from `uname -m`. Setting it cross-builds under slow QEMU emulation.
- `torch.cuda.is_available() == False` is correct; the images are CPU-only.
- `exec format error` means something forced the non-host architecture.
- WSClean illegal-instruction: a `--native` image on another CPU, or the
  default `x86-64-v3` one on a pre-2013 CPU (`WSCLEAN_TARGET_CPU=` fixes it).
- Root-owned files in `results/` on Linux: `sudo chown -R $(id -u):$(id -g) results/`.
- OOM during the casacore build: raise Docker memory, or `BUILD_JOBS=1 ./ri build wsclean`
  (it otherwise compiles at `nproc`).
- Docker Desktop on macOS numbers are not representative of native Linux
  (VM CPU/memory limits, virtualized bind-mount I/O, no GPU passthrough).
