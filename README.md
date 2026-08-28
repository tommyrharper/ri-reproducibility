# ri-reproducibility

This repo runs **nested sampling (PolyChord) over the parameter space of
R2D2-RI and WSClean to find their failure modes**: which observing setups
make a learned DNN-series imager or classical CLEAN reconstruct the sky
badly. Everything else here - the Docker images, the smoke tests, the
pinned upstream revisions - exists to make those searches runnable and
their results trustworthy.

`./ri` is the front door - one command for every part of that:

```bash
./ri --help            # the whole surface, and --help on every subcommand
./ri build             # the four Docker images
./ri search wsclean    # run a search
./ri report            # read what came out
```

It is a thin dispatcher over `scripts/`, so anything it does can also be
run by hand, and `./ri --dry-run <command>` prints exactly what it would run.

The write-up of the science is `latex/notes.tex` ("Finding failure modes
of R2D2 and CLEAN"). The operational detail is
[`docs/nested-sampling.md`](docs/nested-sampling.md); this file is the
map. Every command in one place: [`docs/cheatsheet.md`](docs/cheatsheet.md).

Reference material for the imagers under test is kept alongside, read-only:
`r2d2-paper/` (the R2D2-RI paper source, which defines `sigma_res` and the
dynamic-range conventions the search reuses) and `claims/` (the published
WSClean and R2D2 numbers, including the digitised WSClean figure-7 curves).
Neither is a reproduction target - they are there to be read when deciding
what a bad reconstruction means.

## 1. What the search actually is

Each observing setup is a parameter vector - dynamic range, observation
duration, channel count, start frequency, channel width - searched by
PolyChord over the ranges in `docs/nested-sampling.md` ("Parameter
space"). For one sample:

1. MeqTrees/Cattery simulates a VLA.A Measurement Set of a single 1 Jy
   point source at phase centre, with complex Gaussian thermal noise
   setting the dynamic range.
2. WSClean or R2D2-RI reconstructs it.
3. The reconstruction is scored against the known truth.

That score is handed to PolyChord *as the log-likelihood*. This is not a
Bayesian posterior fit: PolyChord always maximises what it is given, so
choosing a higher-is-worse score turns nested sampling into a search that
concentrates its live points on the setups the imager handles worst. The
resulting samples are a map of failure regions in parameter space.

Which score is used is `NS_METRIC` (`--metric`), default `total_rms_jy` -
the RMS of (reconstructed image - one-pixel truth) over the whole image.
Other useful choices are `off_source_rms_jy`, `sigma_res` (the paper
data-fidelity ratio `||residual_dirty||_2 / ||dirty||_2`), the composite
`badness` score, or any arithmetic expression over the recorded metric
names. The full metric table and the sign conventions are in
`docs/nested-sampling.md` ("Metrics and objective"). Per-run knobs
(`NS_NLIVE`, `NS_NUM_REPEATS`, `NS_MAX_NDEAD`, `NS_MPI_PROCS`, ...) are
documented there too, with the values actually worth using in
`instructions-for-tom.md`.

## 2. Running a search

```bash
cp .env.example .env    # adjust HOST_UID/HOST_GID/paths if needed
./ri search wsclean     # WSClean search (builds its images first)
./ri search r2d2        # R2D2 search (needs checkpoints, section 6)

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

## 3. Reading the results

```bash
./ri tui                # runs, live health and a new-run form in one screen (needs Go)
./ri report             # all runs
./ri report --last 1    # newest run only
./ri serve              # read the report from a browser on another machine
./ri plot gui           # interactive corner plots (needs a display)
./ri plot likelihood    # R2D2 vs WSClean overlay
```

`./ri report` writes `reports/nested-sampling-report/`:
one page per run - PolyChord log(Z), the searched parameters and metrics
for every evaluation, the reconstructions rendered next to the truth, the
likelihood plot, and a collapsible per-stage timing table - plus an
`index.html` linking to them all and an `images/` directory of the PNGs
they reference. Open the index in a browser.

Rendering a page means reading that run's FITS output, so pages already
up to date are skipped and only new runs are built. Each page is stamped
with the report generator's version; `--upgrade` rebuilds the ones an
older version wrote, `--force` rebuilds everything, `--run <run>` rebuilds
one. Rebuilding reuses the PNGs already under `images/`, so a design
change costs page HTML rather than redrawing every evaluation - such a
rebuild skips the astropy/matplotlib import too. Each run
is built as two concurrent processes - its corner plot and the rest of its
page - so a full redraw scales with the cores you have. The report runs inside the r2d2 image (its astropy + matplotlib +
anesthetic), so no host Python environment is needed. Details in
`docs/nested-sampling.md` ("Run summary and reports").

Searches usually run on a headless remote host, so there is often no
browser there to open the index with. `./ri serve` serves the report
directory over HTTP on loopback and prints the `ssh -L` line that tunnels
it to your own machine, so nothing goes out to the network. Details in
`docs/nested-sampling.md` ("Read the report from another machine").

`./ri plot likelihood` (`scripts/plot-merged-likelihood-compare.py`) writes
the merged-failure-score figures into `reports/`, which is where
`latex/notes.tex` includes them from.

## 4. Host prerequisites

- Docker Desktop (built/verified against Docker 20.10.17 client /
  Docker Desktop's `desktop-linux` context on macOS).
- `git`.
- [`uv`](https://docs.astral.sh/uv/). The scripts read their shared
  defaults from `defaults.toml` and `uv` supplies the Python that parses
  it; it also runs the host-side analysis commands (`./ri profile`,
  `./ri plot gui`, `./ri merge`).
- Python 3 for `./ri` itself, which is stdlib-only argparse over the
  scripts below.
- Nothing else - see section 11.

## 5. Building the images

```bash
./ri build              # all four
./ri build wsclean      # WSClean
./ri build r2d2         # R2D2-RI
./ri build meqtrees     # MeqTrees/Cattery MS simulator
./ri build polychord    # PolyChord nested-sampling driver
```

`./ri search` builds what it needs first, so this is only for building
ahead of time. Equivalent to `scripts/build.sh
[all|wsclean|r2d2|meqtrees|polychord]`, which wraps `docker build`
directly (not `docker compose build`) so build args stay explicit.
`docker compose config` is for validating `compose.yaml`, not a build path.
Re-running a build whose inputs have not changed skips `docker build` outright
(~0.08s rather than ~2s of cache-walking); `FORCE_BUILD=1` overrides that.

**Portable vs. host-optimized WSClean**: `WSCLEAN_PORTABLE=ON` (default)
builds a binary that runs on any CPU of the build architecture, per
WSClean's own documented `-DPORTABLE` CMake option. `./ri build wsclean
--native` (`WSCLEAN_PORTABLE=OFF`) rebuilds the same
`ri-reproducibility/wsclean:v3.7` tag for the building machine's exact
instruction set - measurably faster (see
[docs/nested-sampling-throughput.md](docs/nested-sampling-throughput.md)),
but it will die with an illegal-instruction error anywhere else, and it
changes WSClean's timings, so do not mix the two within one search.
Because it is the same tag, the next build without `--native` puts the
portable binary back: to search with it, pass `./ri search wsclean
--native`, which carries the flag into the build the search does first.

### Does the imager under test actually run?

```bash
./ri smoke              # both
./ri smoke wsclean      # wsclean --version + a real tiny imaging run
./ri smoke r2d2         # imports -> app modules -> bundled data load ->
                        # config validation -> (real inference if
                        # checkpoints are present)
./ri smoke ms-to-mat    # the MS -> R2D2 .mat bridge, before an R2D2 search
```

These answer one question: is the image capable of imaging at all? Run
them after a rebuild, before starting a search that will call the imager
thousands of times. `./ri plot fits` renders their FITS output (or any
paths you pass) to PNG using the r2d2 image's own astropy + matplotlib.

## 6. Fetching R2D2 checkpoints

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

## 7. Mounts

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

## 8. Apple Silicon and CPU-only notes

- Images build and run natively for the host architecture - `linux/arm64`
  on Apple Silicon (Docker Desktop runs an arm64 Linux VM: no emulation,
  no Rosetta), `linux/amd64` on an x86_64 Linux host. The scripts derive
  this from `uname -m`; set `DOCKER_DEFAULT_PLATFORM` only to cross-build
  for the other one.
- **CUDA/GPU are not available.** Both imager images are CPU-only by
  design. R2D2 checkpoints trained on GPU load and run fine on CPU, just
  slower - which is the dominant cost of an R2D2 search, since every
  likelihood evaluation is a full DNN-series inference. See
  `docker/r2d2/Dockerfile` for why the CPU-only PyTorch wheel is
  installed explicitly: as of torch 2.13 the *default* PyPI
  `linux/aarch64` wheel bundles several GB of CUDA 13 packages
  (discovered while building this image, not assumed).
- `finufft` (R2D2's NUFFT backend) has **no** `linux/aarch64` wheel on
  PyPI (only macOS arm64 and various Linux x86_64 wheels exist, checked
  against the PyPI JSON API). It is compiled from source inside the
  `docker/r2d2` build stage, which needs `cmake>=3.25`, a C++17+OpenMP
  compiler, and `libfftw3-dev` - all installed there already.
- `torch` **does** have an official `linux/aarch64` CPU wheel
  (`torch==2.13.0+cpu`, from `https://download.pytorch.org/whl/cpu`), so
  no source build or emulation is needed for it.
- Nothing here requires `linux/amd64` or Rosetta emulation. A future
  CUDA/GPU image (not built here) would need a `linux/amd64` host with an
  NVIDIA GPU - Apple Silicon cannot run it at all.
- Docker Desktop on macOS runs containers in a Linux VM (`linuxkit`/`vz`),
  so a search's throughput is bounded by the CPU/memory you allocated to
  that VM, not by the host. Bind-mount I/O goes through VirtioFS (or older
  gRPC-FUSE/osxfs) and is measurably slower than native Linux, which
  matters because every evaluation writes an MS and reads FITS back. If a
  search is too slow to be useful, that - and the absence of GPU
  passthrough for R2D2 - is usually why; native Linux with an NVIDIA GPU
  is the route to scaling it up, and these images should build and run
  there unmodified.

## 9. How upstream revisions are pinned

Every upstream reference lives in `versions.env`, resolved by inspecting
each repository (tags, `.gitmodules`, commit history) rather than assumed:

- **WSClean**: tag `v3.7` (latest stable tag at time of writing; master
  intentionally not used, per WSClean's own installation docs), commit
  `4f395b28abb5eb0ceacfa05f61e3ee49d154d001`. Submodule pins
  (`aocommon`, `radler`, `schaapcommon`) resolved automatically by
  `git clone --recurse-submodules --branch v3.7` and recorded.
- **Casacore**: `v3.8.1` (WSClean v3.7 requires >=3.6; Debian bookworm's
  package is older, so it is built from source - see
  `docker/wsclean/Dockerfile`).
- **R2D2-RI**: no upstream Git tags exist (checked via `git ls-remote
  --tags`); pinned to `main` branch HEAD as inspected,
  `22669259f770a0cb3a3191a5d3e8dbad4ae5a70c`, which the repo's own README
  identifies as "v2.0". Submodule `RI-measurement-operator` (branch
  `python`) pinned to `3c8a93e9127ccaf115d1e3772fbee74aaaccf8e8`.
- **casacore measures data** (IERS/leap-second tables): explicitly **not**
  pinned - see section 10.

To adopt a new revision: edit `versions.env` deliberately, rebuild, and
commit both changes together.

## 10. Reproducibility limitations

The point of pinning revisions is that two searches run months apart are
comparing the same imagers. Two things stop that from being bit-exact:

- The casacore "measures" data (IERS/leap-second/ephemeris tables,
  fetched from `ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar`)
  has no versioned release and is updated continuously upstream - the
  file observed during this build was dated the day before it. Two builds
  months apart embed different measures data even with every other pin
  identical. This is a genuine gap in the upstream ecosystem that this
  repo cannot close unilaterally; the build records the fetched file's
  SHA-256 at `/opt/casacore-data/WSRT_Measures.ztar.sha256` inside the
  image for after-the-fact auditing.
- R2D2-RI's ~100 MB bundled 3c353 example is baked into the image layer
  by upstream's packaging (section 7), so it is not checksummed through
  the bind mounts the way your own data is.

Timings recorded by a search are additionally environment-specific: they
reflect the Docker VM's allocation and its virtualized filesystem, not
raw hardware (section 8). Comparing `wall_seconds` across machines is
meaningless; comparing it within one run is fine, which is what the
`badness` composite and the profiler assume.

## 11. Reproducibility metadata

`./ri record` (`scripts/record-environment.sh`) writes a JSON manifest per run to
`reports/manifests/` capturing: timestamp, this repo's Git revision,
Docker image ID/digest/creation time, host OS/arch/kernel/CPU/allocated
Docker resources, the config file used and its SHA-256, relevant
environment variables, and the exact command run. The nested-sampling run
scripts call it themselves. It only *writes the manifest* - it does not
execute the command - so calling it by hand means passing the same
command you run separately.

### Verifying nothing was installed on the host

Everything WSClean/R2D2/MeqTrees/PolyChord need (compilers, Casacore,
Python packages, PyTorch, finufft, ...) is installed inside the Docker
build stages only - grep any `Dockerfile` for the full list. To verify
nothing leaked: `which wsclean` and `python3 -c "import torch"` (system
Python, not a venv you made for something else) should both fail outside
a container built from this repo. The only host-side tools required are
Docker, Git, `uv` and a Python 3 for `./ri` itself - and `uv` installs
nothing into the system Python: the scripts invoke it as `uv run
--no-project`, which uses a self-contained interpreter under `uv`'s own
cache.

## 12. Reclaiming disk space

```bash
./ri clean            # this repo's images + generated smoke-test outputs
./ri disk-usage       # docker system df -v
docker builder prune  # BuildKit cache (asks first)
docker system prune   # Docker-wide - affects OTHER projects too, use with care
```

`./ri clean` leaves `data/`, `checkpoints/`, `results/` and `reports/`
alone. `docker builder prune -a` before a rebuild forces a fully cold
build.

## Troubleshooting

- **Docker architecture mismatch** (`exec format error`, or Docker
  silently emulating): the images are host-native, so this means
  something forced the other architecture. Check
  `echo $DOCKER_DEFAULT_PLATFORM` is empty (or matches `docker info | grep
  Architecture`) and that you did not pass `--platform` by hand. Forcing
  the non-host architecture triggers slow QEMU emulation this repo does
  not test against, and fails outright on a host with no `binfmt_misc`
  handler registered for it.
- **Unavailable/unsupported ARM64 Python wheel**: check the PyPI JSON API
  (`https://pypi.org/pypi/<pkg>/json`) for a `*aarch64*.whl` before
  assuming a package needs a source build. `finufft` is the one confirmed
  case here (section 8); check the same way for anything you add.
- **CUDA unavailable**: expected and by design on these CPU-only images -
  see section 8. `torch.cuda.is_available()` returning `False` is
  correct, not a bug.
- **R2D2 checkpoint missing**: `scripts/smoke-test-r2d2.sh` stage 5
  prints exact instructions rather than a stack trace; see section 6 and
  `checkpoints/README.md`.
- **A search silently runs stale code**: the `polychord` and `meqtrees`
  images bake in `scripts/lib/nested_sampling/` at build time. Rebuild
  both after editing those files - see `AGENTS.md`.
- **Git submodule missing** (`fatal: no submodule mapping found`, or
  R2D2-RI's `src/ri_measurement_operator` empty): the Dockerfiles clone
  with an explicit `git submodule update --init --recursive` /
  `--recurse-submodules` step - if building outside this repo's
  Dockerfiles (e.g. in `vendor/`), remember both flags.
- **WSClean CMake dependency failure**: WSClean v3.7 additionally
  requires `pybind11-dev`, and casacore's `LibDeflate`-gated `SISCO`
  storage manager needs `libdeflate>=1.19` (Debian bookworm ships 1.14) -
  this repo builds casacore with `-DBUILD_SISCO=OFF` to avoid a second
  from-source dependency, since WSClean does not use it. Both gotchas
  were found by running the build, not by reading docs - see
  `docker/wsclean/Dockerfile` comments.
- **WSClean illegal-instruction error**: you are almost certainly running
  a `WSCLEAN_PORTABLE=OFF` ("native") image built on a different CPU than
  the one running it. Rebuild with `PORTABLE=ON`, or rebuild `native` on
  the actual target machine - see section 5.
- **Incorrect mounted-file ownership**: containers here run as `root` by
  default (no non-root `USER` is created in either imager Dockerfile,
  since neither upstream build system expects one). Files written into
  `./results` etc. will be owned by `root` on Linux hosts; on Docker
  Desktop for macOS, VirtioFS generally maps this transparently already.
  If it does not, `sudo chown -R $(id -u):$(id -g) results/` is the
  pragmatic fix.
- **Insufficient Docker memory**: casacore's parallel build (`make -j4`)
  is the heaviest build step; if it or `pip install torch` gets
  OOM-killed, raise Docker Desktop's memory allocation (Settings ->
  Resources) or pass `--build-arg BUILD_JOBS=1` to `docker build` for the
  WSClean image.
- **Large image or build-cache disk usage**: see section 12. The R2D2
  image alone is ~3.2 GB (dominated by PyTorch CPU plus the unavoidable
  `tensorflow` transitive dependency - see below), before any checkpoints.

## Notable findings from actually building this (not assumed up front)

Discovered by running real builds, not predicted from documentation:

- **`tensorboard==2.16.1`** (pinned in R2D2-RI's `requirements.txt`)
  transitively requires `tf-keras>=2.15.0`, which requires
  `tensorflow<2.22,>=2.21` - so the R2D2 image ends up with a full
  TensorFlow install (~280 MB wheel) despite R2D2-RI being a PyTorch
  project. Upstream's pinned dependency graph, not a mistake here; left
  as-is, since the point is to run the imager upstream actually ships.
- **`scipy` is used directly by R2D2-RI's own code** (`src/utils/io.py`,
  `src/utils/*.py`: `scipy.io.loadmat`, `scipy.constants`,
  `scipy.optimize`) but is **not listed** in `requirements.txt`.
  Installed as an explicit extra step in `docker/r2d2/Dockerfile` with a
  comment explaining the gap, rather than silently patching the vendored
  `requirements.txt`.
- **PyTorch's default `linux/aarch64` wheel bundles CUDA** as of
  `torch==2.13.0` (`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton`, ...
  multiple GB), contradicting the CPU-only-by-default requirement. Fixed
  by installing explicitly from `https://download.pytorch.org/whl/cpu`
  before `requirements.txt`.
- **Casacore's `SISCO` compression storage manager** needs
  `libdeflate>=1.19`; Debian bookworm ships `1.14`. Disabled
  (`-DBUILD_SISCO=OFF`) rather than adding a second from-source
  dependency build, since WSClean does not use it.
- **Casacore's Python3 bindings need NumPy headers** that are not present
  by default; since WSClean only needs casacore's C++ libraries, this is
  avoided entirely with `-DBUILD_PYTHON3=OFF` rather than adding
  `python3-numpy` for a feature nothing here uses.
