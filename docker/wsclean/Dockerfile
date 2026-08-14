# syntax=docker/dockerfile:1
#
# WSClean, built from a pinned source revision. See ../../versions.env for
# the exact tag/commit and README.md for why casacore is also built from
# source rather than taken from Debian packages.
#
# Build (portable, works on any linux/arm64 CPU):
#   docker build --platform linux/arm64 -f docker/wsclean/Dockerfile \
#     --build-arg WSCLEAN_PORTABLE=ON -t ri-reproducibility/wsclean .
#
# Build (host-optimized, for benchmarking on THIS machine only - see
# README.md "portable vs optimized builds"):
#   docker build --platform linux/arm64 -f docker/wsclean/Dockerfile \
#     --build-arg WSCLEAN_PORTABLE=OFF -t ri-reproducibility/wsclean:native .

ARG WSCLEAN_GIT_URL=https://gitlab.com/aroffringa/wsclean.git
ARG WSCLEAN_GIT_TAG=v3.7
ARG CASACORE_GIT_URL=https://github.com/casacore/casacore.git
ARG CASACORE_GIT_TAG=v3.8.1
ARG CASACORE_MEASURES_URL=ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar
ARG WSCLEAN_PORTABLE=ON
ARG BUILD_JOBS=4

##############################################################################
# Stage: build - all compilers/headers needed only to produce the binaries.
##############################################################################
FROM debian:bookworm-slim AS build

ARG CASACORE_GIT_URL
ARG CASACORE_GIT_TAG
ARG CASACORE_MEASURES_URL
ARG WSCLEAN_GIT_URL
ARG WSCLEAN_GIT_TAG
ARG WSCLEAN_PORTABLE
ARG BUILD_JOBS

ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain + WSClean's documented dependencies (Casacore, FFTW,
# Boost filesystem/program_options, CFITSIO, GSL, HDF5, BLAS, LAPACK,
# Python3 dev, CMake, C++ compiler, Git, pkg-config, OpenMPI).
# pybind11-dev is additionally required by WSClean v3.7's CMakeLists.txt
# (not mentioned in the installation docs, confirmed by inspecting the
# pinned source tree).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    ca-certificates \
    wget \
    ftp \
    python3 \
    python3-dev \
    python3-pybind11 \
    pybind11-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-date-time-dev \
    libboost-system-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libgsl-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    wcslib-dev \
    flex \
    bison \
    gfortran \
    libncurses-dev \
    libreadline-dev \
    libdeflate-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Casacore, built from source (pinned) -----------------------------
# WSClean v3.7 requires Casacore >= 3.6; Debian bookworm ships an older
# version, so it is built here rather than installed from apt.
# BUILD_SISCO=OFF: the Sisco compression storage manager needs
# libdeflate >= 1.19, but Debian bookworm ships 1.14. Sisco is not
# required by WSClean, so it is disabled rather than building libdeflate
# from source too.
RUN git clone --depth 1 --branch ${CASACORE_GIT_TAG} ${CASACORE_GIT_URL} /usr/src/casacore \
    && mkdir -p /usr/src/casacore/build \
    && cd /usr/src/casacore/build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/casacore \
        -DBUILD_TESTING=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_HDF5=ON \
        -DDATA_DIR=/opt/casacore-data \
        -DBUILD_SISCO=OFF \
        -DBUILD_PYTHON3=OFF \
    && make -j${BUILD_JOBS} \
    && make install \
    && rm -rf /usr/src/casacore

# Casacore "measures" data (IERS/leap-second/ephemeris tables). This
# upstream artefact has no versioned release and is refreshed continuously
# by ASTRON - see README.md "Reproducibility limitations" for what that
# means for exact numerical reproducibility across rebuilds.
RUN mkdir -p /opt/casacore-data \
    && wget -q ${CASACORE_MEASURES_URL} -O /tmp/WSRT_Measures.ztar \
    && sha256sum /tmp/WSRT_Measures.ztar > /opt/casacore-data/WSRT_Measures.ztar.sha256 \
    && tar -xzf /tmp/WSRT_Measures.ztar -C /opt/casacore-data \
    && rm /tmp/WSRT_Measures.ztar

# --- WSClean, built from source (pinned) -------------------------------
RUN git clone --recurse-submodules --branch ${WSCLEAN_GIT_TAG} --depth 1 \
        --shallow-submodules ${WSCLEAN_GIT_URL} /usr/src/wsclean \
    && mkdir -p /usr/src/wsclean/build \
    && cd /usr/src/wsclean/build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/wsclean \
        -DCMAKE_PREFIX_PATH=/opt/casacore \
        -DPORTABLE=${WSCLEAN_PORTABLE} \
    && make -j${BUILD_JOBS} \
    && make install \
    && rm -rf /usr/src/wsclean

##############################################################################
# Stage: runtime - shared libraries only, no compilers/headers.
##############################################################################
FROM debian:bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Exact runtime package set below was derived by running `ldd` on the
# compiled /opt/wsclean/bin/wsclean binary against this same base image
# and resolving each "not found" entry with `apt-cache search`, not
# guessed - Debian bookworm ships multiple coexisting Boost/HDF5
# versions (e.g. both 1.74.0 and 1.81.0), so the package name has to
# match what the build stage actually linked against.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-filesystem1.74.0 \
    libcfitsio10 \
    libfftw3-double3 \
    libfftw3-single3 \
    libgsl27 \
    libhdf5-cpp-103-1 \
    libhdf5-103-1 \
    libblas3 \
    liblapack3 \
    libgfortran5 \
    libopenmpi3 \
    libwcs7 \
    libgomp1 \
    libpython3.11 \
    python3 \
    ca-certificates \
    time \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/casacore /opt/casacore
COPY --from=build /opt/casacore-data /opt/casacore-data
COPY --from=build /opt/wsclean /opt/wsclean

ENV LD_LIBRARY_PATH=/opt/casacore/lib:/opt/wsclean/lib \
    PATH=/opt/wsclean/bin:${PATH} \
    CASARCFILES=/opt/casacore-data/measures.d

RUN printf 'measures.directory: /opt/casacore-data\n' > /opt/casacore-data/measures.d

WORKDIR /workspace
VOLUME ["/data", "/results"]

ENTRYPOINT ["wsclean"]
CMD ["--version"]
