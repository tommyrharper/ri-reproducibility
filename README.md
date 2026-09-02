# ri-reproducibility

**PolyChord searches this repo's R2D2-RI and WSClean parameter spaces for failure modes**; Docker images, smoke tests, and pinned revisions keep searches runnable and results trustworthy.

**Which R2D2:** the U-Net incarnation (`series: R2D2`, `layers: 1`), from R2D2-RI
v2.0 - checkpoints `R2D2_A1_T2_Realisation1`, 25 terms. `A1` = U-Net (`A2` =
U-WDSR), `T2` = the generalized training set. That model is described in
[arXiv:2503.02554](https://arxiv.org/abs/2503.02554), *not* in the older
[arXiv:2403.05452](https://arxiv.org/abs/2403.05452) vendored under
`r2d2-paper/`, which is where `claims/` and `latex/` take their metric
definitions and published numbers from. Do not compare search output against
`claims/r2d2_claims.md` as if it were the same model.

`./ri` is the front door - one command for every part of that:

```bash
./ri --help            # the whole surface, and --help on every subcommand
./ri build             # the four Docker images
./ri search wsclean    # run a search
```

It is a thin dispatcher over `scripts/`, so anything it does can also be
run by hand, and `./ri --dry-run <command>` prints exactly what it would run.

Science write-up: `latex/notes.tex`. Operational detail:
[`docs/nested-sampling.md`](docs/nested-sampling.md). Command index:
[`docs/cheatsheet.md`](docs/cheatsheet.md).

Read-only reference material: `r2d2-paper/` defines reused R2D2 metrics and
`claims/` records published WSClean and R2D2 numbers. Neither is a
reproduction target.

## 1. Running a search

```bash
cp .env.example .env    # adjust HOST_UID/HOST_GID/paths if needed
./ri search wsclean     # WSClean search (builds its images first)
./ri search r2d2        # R2D2 search (needs checkpoints, section 5)

./ri search wsclean --nlive 20 --num-repeats 5 --max-ndead 20
./ri search r2d2 --metric sigma_res
NS_NLIVE=20 ./ri search wsclean    # same thing; every flag has a variable
```

Output lands in `results/nested-sampling/<tool>-vlaa-<UTC>/`: one
directory per likelihood evaluation (the MS, the reconstruction FITS, and
`metrics.json`), plus a run-level `summary.json`.

Two runs of the same shape can be combined, and the profiler breaks a
finished run down per stage:

```bash
./ri merge results/nested-sampling/A results/nested-sampling/B
./ri profile results/nested-sampling/<run>
```

Every finished search also adds a row to `benchmarks.jsonl`, so what this
commit costs on this machine can be compared with what the last one did:

```bash
./ri bench                          # the table, per commit and machine
./ri bench run wsclean --repeat 3   # the controlled measurement
```

## 2. Reading the results

```bash
./ri tui                # runs, live health, the profile, the benchmark table and a new-run form (needs Go)
./ri report             # all runs
./ri report --last 1    # newest run only
./ri serve              # read the report from a browser on another machine
./ri plot gui           # interactive corner plots (needs a display)
./ri plot likelihood    # R2D2 vs WSClean, overlaid both ways and side by side
./ri plot likelihood --last   # ...for the last two directly comparable runs
```

`./ri plot likelihood` (`scripts/plot-merged-likelihood-compare.py`) writes
merged-failure-score figures into `reports/`, included by `latex/notes.tex`.
`--last` compares the newest R2D2 and WSClean run that agree on VLA config,
metric, parameter space and sampler effort, merged or not, and leaves those
included figures alone. Either way the pair is also kept under its own name in
`reports/likelihood-comparisons/`, which `./ri report` collects onto a page
linked from the top of the index.

## 3. Host prerequisites

- Docker Desktop (built/verified against Docker 20.10.17 client /
  Docker Desktop's `desktop-linux` context on macOS).
- `git`.
- [`uv`](https://docs.astral.sh/uv/) for parsing `defaults.toml` and running
  host-side analysis (`./ri profile`, `./ri plot gui`, `./ri merge`).
- Python 3 for the stdlib-only `./ri` dispatcher.
- Nothing else.

## 4. Building the images

```bash
./ri build              # all four
./ri build wsclean     # or r2d2, meqtrees, polychord
```

`./ri search` builds required images first; use this to build ahead of time.
Unchanged inputs skip `docker build`; `FORCE_BUILD=1` overrides that.

WSClean defaults to portable `x86-64-v3`; use `--native` only when building
and searching on the same host. See the [throughput guide](docs/nested-sampling-throughput.md)
for CPU targets, compatibility, and measurements.

### Does the imager under test actually run?

```bash
./ri smoke              # both
./ri smoke wsclean      # wsclean --version + a real tiny imaging run
./ri smoke r2d2         # imports -> app modules -> bundled data load ->
                        # config validation -> (real inference if
                        # checkpoints are present)
./ri smoke ms-to-mat    # the MS -> R2D2 .mat bridge, before an R2D2 search
```

These verify that images can run their workloads. Run them after rebuilds,
before searches. `./ri plot fits` renders their FITS output (or supplied
paths) to PNG using the r2d2 image's astropy + matplotlib.

## 5. Fetching R2D2 checkpoints

```bash
./ri fetch-checkpoints R2D2_A1_T2_Realisation1.zip
```

**This cannot be fully automated.** The checkpoint host
(`researchportal.hw.ac.uk`) serves files behind a Cloudflare bot
challenge that rejects `curl`/`wget` (HTTP 403, verified 2026-08-03). The
script attempts the download, detects that failure precisely, and prints
the direct URL plus exact placement instructions instead of a stack
trace. See `checkpoints/README.md`. The R2D2 search needs
`checkpoints/R2D2_A1/R2D2_UNet_N<k>.ckpt`.

## 6. Mounts

Configured via `.env` (copy from `.env.example`), consumed by
`compose.yaml` and by the `scripts/*.sh` (plain `docker run -v`, reading
the same variables):

| Host path (default) | Container path | Purpose |
|---|---|---|
| `./data` | `/data` | Measurement Sets, `.mat` files, ground-truth FITS |
| `./checkpoints` | `/checkpoints` | R2D2 pretrained DNN checkpoints |
| `./results` | `/results` | Nested-sampling runs, smoke-test output |
| `./reports` | (host-side only) | Run manifests and the generated HTML report |

None of these are baked into an image layer or committed to Git (see
`.gitignore`) - **with one documented exception**: R2D2-RI's own ~100 MB
bundled example (`data/data_3c353.mat`, `data/3c353_gdth.fits`) ships
inside the upstream repository, so cloning it at build time unavoidably
bakes those two files into the `r2d2` image layer. That is upstream's
packaging decision. See `data/README.md`.

## 7. Apple Silicon and CPU-only notes

- Images use the host architecture (`linux/arm64` on Apple Silicon,
  `linux/amd64` on x86-64); set `DOCKER_DEFAULT_PLATFORM` only to cross-build.
- Imagers are CPU-only. R2D2 runs on CPU, and `finufft` builds from source on
  ARM64; see `docker/r2d2/Dockerfile` for pinned wheels and build packages.
- Docker Desktop searches use its allocated VM CPU, memory, and mount I/O;
  native Linux is faster. No GPU or Rosetta support is required.

## 8. How upstream revisions are pinned

`versions.env` is the source of truth for upstream URLs, revisions, packages,
and deliberately unpinned casacore data. Update matching Dockerfile defaults,
rebuild, and commit both when changing it.

## 9. Reproducibility limitations

Pinned revisions do not make images bit-exact: casacore fetches unversioned
IERS/leap-second/ephemeris data at build time; its image SHA-256 is recorded at
`/opt/casacore-data/WSRT_Measures.ztar.sha256`. The bundled 3c353 example is
also baked in (section 6). Compare `wall_seconds` within runs, not across
machines with different Docker resources (section 7).

## 10. Reclaiming disk space

```bash
./ri clean            # this repo's images + generated smoke-test outputs
./ri disk-usage       # docker system df -v
docker builder prune  # BuildKit cache (asks first)
docker system prune   # Docker-wide - affects OTHER projects too, use with care
```

`./ri clean` leaves `data/`, `checkpoints/`, `results/` and `reports/` alone.
`docker builder prune -a` before a rebuild forces a fully cold build.

## Troubleshooting

- **Architecture / CUDA / wheels:** clear or match `DOCKER_DEFAULT_PLATFORM`, avoid foreign `--platform`, expect CPU-only images, and check PyPI for `*aarch64*.whl` when ARM64 wheels are missing (`finufft` builds from source).
- **Stale code / submodules:** rebuild `polychord` and `meqtrees` after editing `scripts/lib/nested_sampling/`; build with `--recurse-submodules`.
- **WSClean build / CPU failure:** see `docker/wsclean/Dockerfile`; match `WSCLEAN_TARGET_CPU` to the host, or leave it empty for plain x86-64.
- **Root-owned mounts:** expected on Linux; use `sudo chown -R $(id -u):$(id -g) results/` if needed.
- **Build OOM / disk:** `BUILD_JOBS=1 ./ri build wsclean` (it otherwise compiles at `nproc`); see section 10 (`R2D2` image is ~3.2 GB before checkpoints).
