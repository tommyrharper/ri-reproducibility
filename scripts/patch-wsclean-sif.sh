#!/usr/bin/env bash
# Rebuild WSClean with this tree's patches and layer it over a wsclean.sif,
# on the cluster, without Docker or root.
#
#   scripts/patch-wsclean-sif.sh [OUT]   # default OUT: $SIF_DIR/wsclean.sif
#
# A SIF imported from an older Docker build carries only the patches that
# existed when it was built (the 21 Sep one stops at 0006), so later patches
# never reach a cluster run until someone rebuilds on a Docker host. This
# script compiles WSClean from the Dockerfile's pinned tag, plus every
# docker/wsclean/patches/*.patch and docker/wsclean/src/*.cpp, inside an
# `apptainer --fakeroot` Debian sandbox. It reuses /opt/casacore from the
# existing SIF, so casacore is not rebuilt, and writes a `Bootstrap: localimage`
# SIF whose /opt/wsclean is replaced. The old SIF is kept as OUT.orig.
# Run it on a compute node: the build takes ~2 minutes with 30 jobs.
# See docs/csd3-speed.md, round 43.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
BASE="${WSCLEAN_BASE_SIF:-${SIF_DIR}/wsclean.sif}"
OUT="${1:-${SIF_DIR}/wsclean.sif}"
JOBS="${BUILD_JOBS:-$(env -u OMP_NUM_THREADS nproc)}"
dockerfile="${REPO_ROOT}/docker/wsclean/Dockerfile"
arg_of() { sed -n "s/^ARG $1=//p" "${dockerfile}" | head -n1; }
TAG="${WSCLEAN_GIT_TAG:-$(arg_of WSCLEAN_GIT_TAG)}"
URL="$(arg_of WSCLEAN_GIT_URL)"
CPU="${WSCLEAN_TARGET_CPU:-$(arg_of WSCLEAN_TARGET_CPU)}"

[ -f "${BASE}" ] || { echo "FATAL: no base SIF at ${BASE}" >&2; exit 1; }
[ -n "${SLURM_JOB_ID:-}" ] || [ -n "${RI_ALLOW_LOGIN_BUILD:-}" ] \
  || { echo "FATAL: run this on a compute node (RI_ALLOW_LOGIN_BUILD=1 to override)" >&2; exit 1; }

work="$(mktemp -d "${TMPDIR:-/tmp}/ri-wsclean-XXXXXX")"
trap 'chmod -R u+w "${work}" 2>/dev/null; rm -rf "${work}"' EXIT
export APPTAINER_CACHEDIR="${work}/cache" APPTAINER_TMPDIR="${work}/tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
box="${work}/box"

echo "==> sandbox"
apptainer build --fakeroot --sandbox "${box}" docker://debian:bookworm-slim >/dev/null
# The Dockerfile's build list, less what casacore alone needed.
apptainer exec --fakeroot --writable "${box}" sh -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential cmake git pkg-config ca-certificates python3 python3-dev \
    python3-pybind11 pybind11-dev libboost-filesystem-dev \
    libboost-program-options-dev libboost-date-time-dev libboost-system-dev \
    libcfitsio-dev libfftw3-dev libgsl-dev libhdf5-dev libblas-dev liblapack-dev \
    libopenmpi-dev wcslib-dev gfortran >/dev/null'

echo "==> /opt/casacore from ${BASE}"
apptainer exec --bind "${work}" "${BASE}" cp -a /opt/casacore "${box}/opt/"

patches=("${REPO_ROOT}"/docker/wsclean/patches/*.patch)
echo "==> WSClean ${TAG} + ${#patches[@]} patches, -march=${CPU}, ${JOBS} jobs"
# Copied in, not bound: a --writable sandbox cannot create a bind's mount point.
cp -r "${REPO_ROOT}/docker/wsclean" "${box}/ri"
apptainer exec --fakeroot --writable "${box}" sh -ec "
  git clone -q --recurse-submodules --branch '${TAG}' --depth 1 --shallow-submodules '${URL}' /usr/src/wsclean
  git -C /usr/src/wsclean apply /ri/patches/*.patch
  cp /ri/src/*.cpp /usr/src/wsclean/main/
  mkdir /usr/src/wsclean/build && cd /usr/src/wsclean/build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/wsclean \
    -DCMAKE_PREFIX_PATH=/opt/casacore -DPORTABLE=OFF -DTARGET_CPU='${CPU}' >/dev/null
  make -s -j${JOBS} >/dev/null
  make -s install >/dev/null"

echo "==> ${OUT}"
cat >"${work}/layer.def" <<EOF
Bootstrap: localimage
From: ${BASE}

%files
    ${box}/opt/wsclean /opt/wsclean.new

%post
    rm -rf /opt/wsclean && mv /opt/wsclean.new /opt/wsclean

%labels
    ri.wsclean-patches $(ns_wsclean_patches_hash)
EOF
apptainer build --fakeroot "${work}/new.sif" "${work}/layer.def" >/dev/null
apptainer exec "${work}/new.sif" wsclean --version | head -n3
if [ "${OUT}" = "${BASE}" ] && [ ! -e "${OUT}.orig" ]; then
  cp -p "${BASE}" "${OUT}.orig"
fi
mv -f "${work}/new.sif" "${OUT}"
echo "==> done: ${OUT}"
