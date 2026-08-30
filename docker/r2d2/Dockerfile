# syntax=docker/dockerfile:1
#
# R2D2-RI, CPU-only. See ../../versions.env for the exact pinned commit
# (parent repo + RI-measurement-operator submodule) and README.md for the
# ARM64-specific dependency notes (finufft has no linux/aarch64 wheel and
# is compiled from source here; torch does have an official aarch64 CPU
# wheel on PyPI).
#
# This is a CPU-only image by design (see CLAUDE.md instructions): CUDA is
# treated as a separate, later, Linux+NVIDIA target, not bundled here.
#
# Build:
#   docker build --platform linux/arm64 -f docker/r2d2/Dockerfile \
#     -t ri-reproducibility/r2d2:cpu .

ARG R2D2_GIT_URL=https://github.com/basp-group/R2D2-RI.git
ARG R2D2_GIT_COMMIT=22669259f770a0cb3a3191a5d3e8dbad4ae5a70c
ARG PYTHON_VERSION=3.10

##############################################################################
# Stage: build - compiler toolchain needed only to build the venv and
# compile finufft's C++ extension from source (no aarch64 wheel exists).
##############################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

ARG R2D2_GIT_URL
ARG R2D2_GIT_COMMIT

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    libfftw3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pinned clone (parent repo + RI-measurement-operator submodule, both
# resolved to immutable commits - see versions.env).
RUN git clone ${R2D2_GIT_URL} /opt/r2d2/R2D2-RI \
    && cd /opt/r2d2/R2D2-RI \
    && git checkout ${R2D2_GIT_COMMIT} \
    && git submodule update --init --recursive

WORKDIR /opt/r2d2/R2D2-RI

COPY docker/r2d2/patches/remove-final-residual.patch /tmp/remove-final-residual.patch
RUN git apply /tmp/remove-final-residual.patch && rm /tmp/remove-final-residual.patch

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# cmake via pip: Debian bookworm's apt cmake can be older than the >=3.25
# finufft's CMake-based build requires; PyPI's manylinux aarch64 wheel is
# simpler and reproducible than mixing apt/pip cmake.
RUN pip install --upgrade pip "cmake>=3.25"

# Install the CPU-only torch build explicitly, from PyTorch's own "cpu"
# wheel index, BEFORE installing requirements.txt. As of torch 2.13, the
# default PyPI linux/aarch64 wheel bundles CUDA 13 (cudnn, nccl, triton,
# cusparselt, ...) - several GB of NVIDIA packages that have no use on
# this CPU-only Apple Silicon image and directly contradict this
# project's "no CUDA in the default image" requirement. Installing the
# matching "+cpu" wheel first satisfies requirements.txt's `torch>=2.0.0`
# without pip pulling the CUDA variant back in.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"

RUN pip install -r requirements.txt

# requirements.txt does not list scipy, but R2D2-RI's own src/utils/io.py
# (scipy.io.loadmat, scipy.constants) and src/utils/*.py (scipy.optimize)
# import it directly - an upstream gap in the pinned requirements file,
# confirmed by grepping the pinned source tree. Installed separately and
# recorded here rather than silently patching requirements.txt itself.
RUN pip install scipy

# Posterior/corner plots for nested-sampling run pages in
# scripts/lib/generate_report.py (make nested-sampling-report).
RUN pip install anesthetic

# Resolved, fully-pinned environment snapshot - the "lock file" mandated
# alongside upstream's floating-minimum requirements.txt. Inspect with
# `make shell-r2d2` then `cat /opt/r2d2/requirements.lock.txt`, or copy
# out with `docker cp`.
RUN pip freeze > /opt/r2d2/requirements.lock.txt

##############################################################################
# Stage: runtime - interpreter + venv only, no compilers.
##############################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# libfftw3-3 (double) + libfftw3-single3: runtime shared libraries for the
# finufft extension compiled in the build stage. libgomp1: OpenMP runtime
# used by both finufft and torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-double3 \
    libfftw3-single3 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/r2d2 /opt/r2d2

# The base image's Python ships no bytecode for its standard library (1483
# `.py`, 0 `.pyc`) and the R2D2 checkout has none either, so the first Python
# process in a fresh container compiled everything `import torch` and imager.py
# reach and wrote it into the container layer: `import torch` cost 0.89s cold
# against 0.62s once one process had paid it. The nested-sampling run starts
# this image fresh every time and its imaging worker's warm-up is what
# evaluation one waits on, so that 0.27s is straight off the run's wall clock.
# pip already compiled the venv. `|| true` for the same reason the PolyChord
# and MeqTrees images carry it: compileall returns non-zero on any file that is
# not valid source for this interpreter, and which those are is a property of
# whatever `PYTHON_VERSION` resolves to, not of this repository. Nothing under
# either path fails today.
RUN python3 -m compileall -q "/usr/local/lib/python${PYTHON_VERSION}" /opt/r2d2/R2D2-RI/src || true

ENV PATH=/opt/venv/bin:${PATH} \
    R2D2_HOME=/opt/r2d2/R2D2-RI \
    MPLCONFIGDIR=/opt/matplotlib

# Bake matplotlib's font list into the image. It is derived from the fonts
# installed here, so it never changes at runtime - but containers are run with
# --rm, so without this the first `import matplotlib.pyplot` in every one of
# them rebuilds it from scratch (~0.07s, on the report's serial prologue).
# World-writable so the same cache still works if the container runs as a
# non-root user; matplotlib silently falls back to a temporary directory (and
# rebuilds) when MPLCONFIGDIR is not writable.
RUN mkdir -p "${MPLCONFIGDIR}" \
    && python3 -c "import matplotlib.pyplot" \
    && chmod -R a+rwX "${MPLCONFIGDIR}"

WORKDIR /opt/r2d2/R2D2-RI

ENTRYPOINT ["python3"]
CMD ["--version"]
