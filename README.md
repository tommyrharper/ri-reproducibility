# ri-reproducibility

A reproducible, Docker-based local environment for two radio-interferometric
imaging pipelines - **WSClean** (classical CLEAN-family imager, C++) and
**R2D2-RI** (learned DNN-series imager, Python/PyTorch) - built for
Apple Silicon, with the goal of eventually benchmarking both against
their published papers under controlled conditions.

## 1. Scientific purpose

Radio interferometers measure visibilities (Fourier-domain samples of
the sky). Turning those into an image requires: gridding the
visibilities onto a regular Fourier grid, an inverse FFT to the image
domain, and a deconvolution step to remove the effect of incomplete
Fourier coverage (the point spread function / "dirty beam"). WSClean
implements this with the classical CLEAN algorithm family (multiscale,
wideband, w-stacking); R2D2-RI replaces the deconvolution step with a
series of pretrained deep neural networks. This repository exists to
run both, honestly and reproducibly, as a foundation for evaluating
where and how machine learning can improve the deconvolution stage of
this pipeline (the broader UROP research question this environment
supports).

## 2. Four distinct activities - keep them separate

1. **Installation verification** - does the software run at all?
   (`make smoke-test-wsclean` stage 1, `make smoke-test-r2d2` stage 1-2)
2. **Baseline execution** - does it run end-to-end on real (if tiny)
   input data and produce plausible output? (the rest of the smoke
   tests)
3. **Benchmarking** - controlled measurement of runtime, memory,
   configuration sensitivity (`benchmarks/` - see section 8 below and
   `benchmarks/README.md`).
4. **Paper reproduction** - matching a specific published figure/table
   under matched conditions (`benchmarks/REPRODUCTION_PLAN.md`).

Each is a strictly higher bar than the last. Do not conflate "it ran"
with "it reproduced the paper."

## 3. Host prerequisites

- Docker Desktop (this environment was built/verified against Docker
  20.10.17 client / Docker Desktop's `desktop-linux` context on macOS).
- `git`.
- Nothing else. See "15. Verifying no dependencies were installed on
  the host" below.

## 4. Apple Silicon limitations

- All images are built and run as `linux/arm64` natively (Docker
  Desktop on Apple Silicon runs an arm64 Linux VM - no emulation, no
  Rosetta needed for `linux/arm64` images).
- **CUDA/GPU are not available.** Both images are CPU-only by design.
  R2D2 checkpoints trained on GPU load and run fine on CPU (slower);
  see `docker/r2d2/Dockerfile` for why the CPU-only PyTorch wheel is
  installed explicitly (as of torch 2.13, the *default* PyPI
  `linux/aarch64` wheel bundles several GB of CUDA 13 packages -
  discovered while building this image, not assumed).
- `finufft` (R2D2's NUFFT backend) has **no** `linux/aarch64` wheel on
  PyPI (only macOS arm64 and various Linux x86_64 wheels exist, checked
  against the PyPI JSON API). It is compiled from source inside the
  `docker/r2d2` build stage, which needs `cmake>=3.25`, a C++17+OpenMP
  compiler, and `libfftw3-dev` - all installed there already.
- `torch` **does** have an official `linux/aarch64` CPU wheel
  (`torch==2.13.0+cpu`, from `https://download.pytorch.org/whl/cpu`), so
  no source build or emulation is needed for it.
- Nothing in this repository requires `linux/amd64` or Rosetta emulation
  for the default CPU images. A future CUDA/GPU image (not built here)
  would need a `linux/amd64` host with an NVIDIA GPU - Apple Silicon
  cannot run it at all, emulated or otherwise.
- Benchmark numbers gathered under Docker Desktop on macOS are **not**
  representative of native Linux performance - see section 16.

## 5. Building the images

```bash
cp .env.example .env   # adjust HOST_UID/HOST_GID/paths if needed
make build              # all images
make build-wsclean      # WSClean only
make build-r2d2         # R2D2-RI only
make build-meqtrees     # MeqTrees/Cattery MS simulator only
make build-polychord    # PolyChord nested-sampling driver only
```

Equivalent to `scripts/build.sh [all|wsclean|r2d2|meqtrees|polychord]`, which wraps `docker
build` directly (not `docker compose build`) so build args stay
explicit. `docker compose config` (`make config`) is provided for
validation and for `make shell-*`, not as the primary build path.

WSClean portability: `WSCLEAN_PORTABLE=ON` (default) builds a binary
that runs on any `linux/arm64` CPU, per WSClean's own documented
`-DPORTABLE` CMake option. For benchmarking *on the machine you will
actually run benchmarks on*, rebuild with `WSCLEAN_PORTABLE=OFF`
(`WSCLEAN_PORTABLE=OFF scripts/build.sh wsclean`, tagged
`ri-reproducibility/wsclean:native`) to let WSClean auto-detect and
target that CPU's instruction set. **Never** benchmark the portable
build and report it as representative optimized-WSClean performance.

## 6. Running the smoke tests

```bash
make smoke-test            # both
make smoke-test-wsclean    # wsclean --version + a real tiny imaging run
make smoke-test-r2d2       # staged: imports -> app modules -> bundled
                            # data load -> config validation -> (real
                            # inference only if checkpoints are present)
```

See `scripts/smoke-test-wsclean.sh` and `scripts/smoke-test-r2d2.sh` for
exactly what each stage does and why.

## 7. Visualizing FITS output

```bash
make plot-fits                        # renders the R2D2 smoke-test outputs
make plot-fits FILES="results/smoke-test-wsclean/foo-image.fits"
```

Renders FITS images to PNG (zscale + asinh stretch, via
`scripts/plot-fits.sh`) using the r2d2 image's own astropy + matplotlib,
so no host Python environment is needed. With no arguments it renders
the standard R2D2 diagnostic set (dirty image, PSF, cleaned model,
residual). Pass one or more paths via `FILES` (relative to the repo
root, or absolute paths inside the r2d2 image, e.g. the bundled
ground-truth `/opt/r2d2/R2D2-RI/data/3c353_gdth.fits` - see section 9
below) to render specific files instead. PNGs are written flat into
`results/`, named after the source FITS file.

## 8. Running a benchmark and producing the report

A benchmark run is a real (not smoke-test) pipeline invocation, recorded
so it can be inspected and compared later:

```bash
# 1. Run the pipeline for real, writing output to its own results/
#    subdirectory (not the smoke test's, so they don't collide), e.g.:
docker run --rm --platform linux/arm64 \
  -v "$(pwd)/checkpoints:/checkpoints:ro" \
  -v "$(pwd)/results/<experiment-name>:/results" \
  -v "$(pwd)/config/r2d2:/workspace/config:ro" \
  --entrypoint python3 ri-reproducibility/r2d2:cpu \
  ./src/imager.py --config /workspace/config/R2D2_U-Net.yaml \
  --ckpt_path /checkpoints/R2D2_A1

# 2. Record a manifest for that exact run
scripts/record-environment.sh --tool r2d2 \
  --image ri-reproducibility/r2d2:cpu \
  --config config/r2d2/R2D2_U-Net.yaml -- <the docker run command above>

# 3. Render every manifest in benchmarks/manifests/ into one HTML report
make benchmark-report
# Nested-sampling PoC runs have a separate report:
# make nested-sampling-report
```

`record-environment.sh` only *writes the manifest* - it does not execute
the command itself (see "Reproducibility metadata" below), so steps 1
and 2 both use the same exact command, run separately. Input/checkpoint
checksums and the run's actual results (SNR/logSNR, wall-clock time,
etc.) are experiment-specific and not auto-captured; add an
`"experiment"` object to the written manifest JSON with a `"purpose"`,
whatever provenance is relevant, and a `"results"` object - see any
existing `benchmarks/manifests/r2d2-*.json` for a worked example.
`benchmarks/REPRODUCTION_PLAN.md` tracks which specific benchmark (paper,
table, expected numbers) this environment currently targets, and its
"Current reproduction status" section should be updated by hand after a
run that moves that target forward.

For the WSClean and R2D2 x VLA.A nested-sampling infrastructure PoCs, see
`docs/nested-sampling.md` and run `make nested-sampling-poc` (WSClean) or
`make nested-sampling-r2d2-poc` (R2D2). Each run uses MeqTrees/Cattery's
VLA.A makems data to create noisy single-point-source Measurement Sets, then
PolyChord to search parameter space by a configurable objective metric
(`--metric`/`NS_METRIC`, default `off_source_rms_jy`; an optional composite
`badness` score is also available).

`make benchmark-report` builds `benchmarks/report.html` from
`benchmarks/manifests/` (one card per manifest: environment/provenance,
results metrics, output FITS). `make nested-sampling-report` builds
`benchmarks/nested-sampling-report/` from
`results/nested-sampling-poc/*/poc-summary.json` - one page per run
(PolyChord log(Z), evaluation image grid, likelihood plot) plus an
`index.html` linking to them all; re-running only builds pages for runs
that don't have one yet (see `docs/nested-sampling.md`).
Both reuse the r2d2 image's own astropy + matplotlib + anesthetic,
same approach as `make plot-fits` (section 7), so no host Python
environment is needed. Open the file directly in a browser afterward.
It's generated/gitignored, like `results/`; `git add -f` a specific copy
if you want one version-controlled (e.g. for a paper appendix).

## 9. Mounting datasets and output directories

Configured via `.env` (copy from `.env.example`), consumed by
`compose.yaml` and the `scripts/*.sh` (which use plain `docker run -v`,
reading the same variables). Host directories:

| Host path (default) | Container path | Purpose |
|---|---|---|
| `./data` | `/data` | Measurement Sets, `.mat` files, ground-truth FITS |
| `./checkpoints` | `/checkpoints` | R2D2 pretrained DNN checkpoints |
| `./results` | `/results` | Generated images, logs, smoke-test output |
| `./benchmarks` | (host-side only) | Manifests, benchmark scripts |

None of these are ever baked into an image layer or committed to Git
(see `.gitignore`) - **with one documented exception**: R2D2-RI's own
~100 MB bundled example (`data/data_3c353.mat`,
`data/3c353_gdth.fits`) ships inside the upstream repository itself, so
cloning it at build time (as instructed) unavoidably bakes those two
files into the `r2d2` image layer. This is upstream's packaging
decision, not this repo's; anything *you* add for further experiments
still goes through the bind mounts above. See `data/README.md`.

## 10. Fetching checkpoints

```bash
make fetch-r2d2-checkpoints REALISATION=R2D2_A1_T2_Realisation1.zip
```

**This cannot be fully automated.** The checkpoint host
(`researchportal.hw.ac.uk`) serves files behind a Cloudflare bot
challenge that rejects `curl`/`wget` (HTTP 403, verified 2026-08-03).
The script attempts the download, detects that failure precisely, and
prints the direct URL plus exact placement instructions instead of a
stack trace. See `checkpoints/README.md`.

## 11. How upstream revisions are pinned

Every upstream reference lives in `versions.env`, resolved by directly
inspecting each repository (tags, `.gitmodules`, commit history) rather
than assumed:

- **WSClean**: tag `v3.7` (latest stable tag at time of writing; master
  intentionally not used, per WSClean's own installation docs), commit
  `4f395b28abb5eb0ceacfa05f61e3ee49d154d001`. Submodule pins
  (`aocommon`, `radler`, `schaapcommon`) resolved automatically by
  `git clone --recurse-submodules --branch v3.7` and recorded for
  documentation.
- **Casacore**: `v3.8.1` (WSClean v3.7 requires >=3.6; Debian bookworm's
  packaged version is older, so it's built from source - see
  `docker/wsclean/Dockerfile`).
- **R2D2-RI**: no upstream Git tags exist (checked via `git ls-remote
  --tags`); pinned to `main` branch HEAD as inspected,
  `22669259f770a0cb3a3191a5d3e8dbad4ae5a70c`, which the repo's own
  README identifies as "v2.0". Submodule `RI-measurement-operator`
  (branch `python`) pinned to `3c8a93e9127ccaf115d1e3772fbee74aaaccf8e8`.
- **casacore measures data** (IERS/leap-second tables): explicitly
  **not** pinned to an immutable release - see section 16's note below
  and `docker/wsclean/Dockerfile`'s comments. This is a genuine, open
  reproducibility gap in the upstream ecosystem, not an oversight here.

To adopt a new revision: edit `versions.env` deliberately, rebuild, and
commit both changes together.

## 12. Rebuilding from scratch

```bash
make clean                    # removes this repo's images + smoke-test outputs
docker builder prune          # reclaims BuildKit cache (asks for confirmation)
make build
```

Docker layer caching means a plain `make build` after `make clean` will
still re-download apt/pip packages fresh but reuse nothing else
locally cached; `docker builder prune -a` before rebuilding forces a
fully cold build if you need to verify true from-scratch reproducibility.

## 13. Removing all generated Docker artefacts / reclaiming disk space

```bash
make clean                          # this repo's images + local smoke-test outputs
docker builder prune                # BuildKit cache
docker system prune                 # anything else Docker-wide (asks first; affects
                                     # containers/images from OTHER projects too - use
                                     # with care, not scoped to this repo)
```

## 14. Inspecting Docker disk usage

```bash
make disk-usage    # docker system df -v
```

## 15. Verifying no dependencies were installed on the host

Everything WSClean/R2D2 need (compilers, Casacore, Python packages,
PyTorch, finufft, ...) is installed inside the Docker build stages only
- grep any `Dockerfile` for the full list. To verify nothing leaked onto
the host: `which wsclean`, `python3 -c "import torch"` (using your
system Python, not a venv you made for something else) should both fail
outside a container built from this repo. The only host-side tools this
project's own instructions require are Docker and Git.

## 16. Docker Desktop on macOS - limitations for benchmarking

- Docker Desktop on macOS runs containers inside a lightweight Linux VM
  (`linuxkit`/`vz`). CPU/memory limits are whatever you've configured
  for that VM (Docker Desktop settings), not the full host - benchmark
  numbers reflect the VM's allocation, not raw Apple Silicon
  performance.
- File I/O through bind mounts (`-v host:container`) on macOS goes
  through a virtualized filesystem layer (VirtioFS or older
  gRPC-FUSE/osxfs), which is measurably slower than native Linux
  bind-mount I/O. Any I/O-bound timing (large MS reads, checkpoint
  loads) will look worse here than on native Linux with the same disk.
- No GPU passthrough exists for Docker Desktop on macOS at all (Apple
  Silicon GPU or otherwise) - R2D2 timings here can only ever reflect
  CPU inference, never the GPU numbers papers typically report.
- The casacore "measures" data (IERS/leap-second/ephemeris tables,
  fetched from `ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar`)
  has no versioned release and is updated continuously upstream (the
  file observed during this build was dated the day before the build).
  Two builds performed months apart will embed different measures data
  even with every other pin identical - a genuine, documented gap in
  bit-exact reproducibility that this repo cannot close unilaterally.
  The build records the fetched file's SHA-256 at
  `/opt/casacore-data/WSRT_Measures.ztar.sha256` inside the image for
  after-the-fact auditing.

**Recommended future route for authoritative benchmarks**: native Linux
(bare metal or a cloud VM) with an NVIDIA GPU for R2D2, using
`WSCLEAN_PORTABLE=OFF` rebuilt on that exact machine for WSClean. This
repository's images should build and run unmodified there (same
Dockerfiles, `--platform linux/amd64` or native architecture) - only the
benchmark *numbers* from this Apple Silicon / Docker Desktop setup
should not be treated as authoritative.

## Troubleshooting

- **Docker architecture mismatch** (`exec format error`, or Docker
  silently emulating): confirm `docker info | grep Architecture` says
  `aarch64` and you did not pass `--platform linux/amd64` anywhere by
  accident. Both Dockerfiles are `linux/arm64` native; forcing
  `linux/amd64` triggers slow, unsupported QEMU emulation this repo
  does not test against.
- **Unavailable/unsupported ARM64 Python wheel**: check the PyPI JSON
  API (`https://pypi.org/pypi/<pkg>/json`) for a `*aarch64*.whl` before
  assuming a package needs a source build. `finufft` is the one
  confirmed case in this project (see section 4); if you add new Python
  dependencies, check the same way rather than guessing.
- **CUDA unavailable**: expected and by design on this CPU-only image -
  see section 4. `torch.cuda.is_available()` returning `False` is
  correct, not a bug.
- **R2D2 checkpoint missing**: `scripts/smoke-test-r2d2.sh` stage 5
  prints exact instructions rather than a stack trace; see also section
  9 and `checkpoints/README.md`.
- **Git submodule missing** (`fatal: no submodule mapping found`, or
  R2D2-RI's `src/ri_measurement_operator` empty): the Dockerfiles clone
  with an explicit `git submodule update --init --recursive` /
  `--recurse-submodules` step - if building outside this repo's
  Dockerfiles (e.g. in `vendor/`), remember both flags.
- **WSClean CMake dependency failure**: WSClean v3.7 additionally
  requires `pybind11-dev` and casacore's `LibDeflate`-gated `SISCO`
  storage manager needs `libdeflate>=1.19` (Debian bookworm ships 1.14)
  - this repo builds casacore with `-DBUILD_SISCO=OFF` to avoid needing
  a from-source libdeflate too, since WSClean doesn't use it. Both
  gotchas were found by actually running the build, not by reading docs
  alone - see `docker/wsclean/Dockerfile` comments.
- **WSClean illegal-instruction error**: you are almost certainly
  running a `WSCLEAN_PORTABLE=OFF` ("native") image built on a different
  CPU than the one running it. Rebuild with `PORTABLE=ON`, or rebuild
  `native` on the actual target machine - see section 5.
- **Incorrect mounted-file ownership**: containers here run as `root`
  by default (no non-root `USER` is created in either Dockerfile, since
  neither upstream project's build system expects one). Files written
  into `./results` etc. will be owned by `root` on Linux hosts (harmless
  but occasionally annoying); on Docker Desktop for macOS, VirtioFS
  generally maps this transparently to your host user already. If it
  doesn't, `sudo chown -R $(id -u):$(id -g) results/` is the pragmatic
  fix; a proper non-root-user Dockerfile change is future work, not
  done here to avoid adding untested complexity to the first baseline.
- **Insufficient Docker memory**: casacore's parallel build (`make -j4`)
  is the heaviest build step; if it or `pip install torch` gets OOM-killed,
  raise Docker Desktop's memory allocation (Settings -> Resources) or
  pass `--build-arg BUILD_JOBS=1` to `docker build` for the WSClean image.
- **Large image or build-cache disk usage**: see sections 12-13. The
  R2D2 image alone is ~3.2 GB (dominated by PyTorch CPU + the
  unavoidable `tensorflow` transitive dependency - see "Notable findings"
  below), before any checkpoints are downloaded.

## Notable findings from actually building this (not assumed up front)

These were discovered by running real builds during development of this
environment, not predicted from documentation:

- **`tensorboard==2.16.1`** (pinned in R2D2-RI's `requirements.txt`)
  transitively requires `tf-keras>=2.15.0`, which requires
  `tensorflow<2.22,>=2.21` - so the R2D2 image ends up with a full
  TensorFlow install (~280 MB wheel) despite R2D2-RI being a PyTorch
  project. This is upstream's pinned dependency graph, not a mistake in
  this repository; left as-is per "preserve upstream requirements.txt
  as the source specification."
- **`scipy` is used directly by R2D2-RI's own code**
  (`src/utils/io.py`, `src/utils/*.py`: `scipy.io.loadmat`,
  `scipy.constants`, `scipy.optimize`) but is **not listed** in
  `requirements.txt`. Installed as an explicit extra step in
  `docker/r2d2/Dockerfile` with a comment explaining the gap, rather
  than silently patching the vendored `requirements.txt`.
- **PyTorch's default `linux/aarch64` wheel bundles CUDA** as of
  `torch==2.13.0` (`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton`,
  etc. - multiple GB), contradicting this project's CPU-only-by-default
  requirement. Fixed by installing explicitly from
  `https://download.pytorch.org/whl/cpu` before `requirements.txt`.
- **Casacore's `SISCO` compression storage manager** needs
  `libdeflate>=1.19`; Debian bookworm ships `1.14`. Disabled
  (`-DBUILD_SISCO=OFF`) rather than adding a second from-source
  dependency build, since WSClean does not use it.
- **Casacore's Python3 bindings need NumPy headers** that aren't present
  by default; since WSClean only needs casacore's C++ libraries, this is
  avoided entirely with `-DBUILD_PYTHON3=OFF` rather than adding
  `python3-numpy` for a feature nothing here uses.

## Reproducibility metadata

`scripts/record-environment.sh` writes a JSON manifest per run to
`benchmarks/manifests/` capturing: timestamp, this repo's Git revision,
Docker image ID/digest/creation time, host OS/arch/kernel/CPU/allocated
Docker resources, the config file used and its SHA-256, relevant
environment variables, and the exact command run. Input/output/checkpoint
checksums and random seeds are experiment-specific and are added by
whatever script drives that experiment (not yet wired up beyond the
smoke tests - see `benchmarks/README.md`).
