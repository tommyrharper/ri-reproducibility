#!/usr/bin/env bash
# Build images with explicit args; repeated builds use Docker's layer cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

TARGET="${1:-all}"
# The instruction set WSClean's gridder is compiled for; see the header of
# docker/wsclean/Dockerfile for the three values that mean something. Empty is
# meaningful (the plain x86-64 baseline), so `-` rather than `:-`.
WSCLEAN_TARGET_CPU="${WSCLEAN_TARGET_CPU-x86-64-v3}"

# Cache build inputs in an image label; FORCE_BUILD=1 bypasses the check.
BUILD_INPUTS_LABEL="ri.build-inputs"

# Hash Dockerfiles and copied or bind-mounted files. Include names so renames
# invalidate builds; exclude __pycache__ to match .dockerignore.
# Usage: inputs_hash <flavour> <path...>
inputs_hash() {
  local flavour="$1"
  shift
  {
    printf '%s\n' "${flavour}"
    find "$@" -name '__pycache__' -prune -o -type f -print | LC_ALL=C sort
    find "$@" -name '__pycache__' -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 cat
  } | sha256sum | cut -d' ' -f1
}

# Build unless the tag already names an image carrying this input hash.
# Usage: build_image <tag> <hash> <docker build args...>
build_image() {
  local image="$1" hash="$2"
  shift 2
  if [ -z "${FORCE_BUILD:-}" ] && [ "$(
    docker image inspect -f "{{index .Config.Labels \"${BUILD_INPUTS_LABEL}\"}}" \
      "${image}" 2>/dev/null
  )" = "${hash}" ]; then
    echo "==> ${image} is up to date (inputs ${hash:0:12})"
    return 0
  fi
  echo "==> building ${image} (platform=${PLATFORM})"
  docker build --platform "${PLATFORM}" \
    --label "${BUILD_INPUTS_LABEL}=${hash}" \
    -t "${image}" "$@" .
}

# BUILD_JOBS controls source-build parallelism; default to `nproc` and exclude
# it from inputs_hash because it changes build time, not output.
build_wsclean() {
  build_image ri-reproducibility/wsclean:v3.7 \
    "$(inputs_hash "${PLATFORM} TARGET_CPU=${WSCLEAN_TARGET_CPU}" \
      docker/wsclean/Dockerfile docker/wsclean/patches docker/wsclean/src)" \
    -f docker/wsclean/Dockerfile \
    --build-arg BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}" \
    --build-arg WSCLEAN_TARGET_CPU="${WSCLEAN_TARGET_CPU}"
}

build_r2d2() {
  build_image ri-reproducibility/r2d2:cpu \
    "$(inputs_hash "${PLATFORM}" docker/r2d2/Dockerfile docker/r2d2/patches)" \
    -f docker/r2d2/Dockerfile
}

build_meqtrees() {
  build_image ri-reproducibility/meqtrees:kern-10 \
    "$(inputs_hash "${PLATFORM}" docker/meqtrees/Dockerfile \
      scripts/lib/nested_sampling/simulate_point_source_ms.py \
      scripts/lib/nested_sampling/point_source_forest.py \
      scripts/lib/nested_sampling/ms_to_r2d2_mat.py \
      scripts/lib/nested_sampling/common.py \
      defaults.toml)" \
    -f docker/meqtrees/Dockerfile
}

build_polychord() {
  build_image ri-reproducibility/polychord:lite \
    "$(inputs_hash "${PLATFORM}" docker/polychord/Dockerfile scripts/lib/nested_sampling)" \
    -f docker/polychord/Dockerfile
}

case "${TARGET}" in
  wsclean) build_wsclean ;;
  r2d2) build_r2d2 ;;
  meqtrees) build_meqtrees ;;
  polychord) build_polychord ;;
  all) build_wsclean && build_r2d2 && build_meqtrees && build_polychord ;;
  *) echo "usage: $0 [all|wsclean|r2d2|meqtrees|polychord]" >&2; exit 1 ;;
esac
