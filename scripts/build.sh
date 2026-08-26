#!/usr/bin/env bash
# Build one or more images. Wraps `docker build` directly rather than
# `docker compose build` so build args (portability, platform) stay
# explicit and idempotent - re-running just hits Docker's layer cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

TARGET="${1:-all}"
WSCLEAN_PORTABLE="${WSCLEAN_PORTABLE:-ON}"

# A `docker build` whose every step is CACHED is not free: buildkit still
# resolves the `docker/dockerfile:1` frontend and the base image metadata over
# the network, walks every CACHED step, and re-exports and re-tags the manifest
# and its provenance attestation - 1.5-1.9s per image on this host. The
# nested-sampling PoC targets build three images in front of a run whose sampler
# is ~1.2s, so that check used to be a third of the command's wall clock.
#
# So: record a hash of everything a build reads in a label on the image it
# produces, and skip the build when the image is already there with that exact
# hash. `docker image inspect` is ~0.05s and answers "is it built from these
# inputs" and "does it still exist" in one call. FORCE_BUILD=1 skips the skip.
BUILD_INPUTS_LABEL="ri.build-inputs"

# The files a build reads: its Dockerfile plus whatever that COPYs or
# bind-mounts from the context (`grep -n 'COPY\|--mount=type=bind' docker/*/Dockerfile`).
# Directories are walked. Names go into the hash as well as contents, so that
# renaming a file inside a COPYed directory is a rebuild. <flavour> carries the
# platform and any build args - same files, different flags, different image.
# Usage: inputs_hash <flavour> <path...>
inputs_hash() {
  local flavour="$1"
  shift
  {
    printf '%s\n' "${flavour}"
    find "$@" -type f -print | LC_ALL=C sort
    find "$@" -type f -print0 | LC_ALL=C sort -z | xargs -0 cat
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

build_wsclean() {
  build_image ri-reproducibility/wsclean:v3.7 \
    "$(inputs_hash "${PLATFORM} PORTABLE=${WSCLEAN_PORTABLE}" docker/wsclean/Dockerfile)" \
    -f docker/wsclean/Dockerfile \
    --build-arg WSCLEAN_PORTABLE="${WSCLEAN_PORTABLE}"
}

build_r2d2() {
  build_image ri-reproducibility/r2d2:cpu \
    "$(inputs_hash "${PLATFORM}" docker/r2d2/Dockerfile)" \
    -f docker/r2d2/Dockerfile
}

build_meqtrees() {
  build_image ri-reproducibility/meqtrees:kern-10 \
    "$(inputs_hash "${PLATFORM}" docker/meqtrees/Dockerfile \
      scripts/lib/nested_sampling/simulate_point_source_ms.py \
      scripts/lib/nested_sampling/point_source_forest.py \
      scripts/lib/nested_sampling/ms_to_r2d2_mat.py \
      scripts/lib/nested_sampling/poc_common.py)" \
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
