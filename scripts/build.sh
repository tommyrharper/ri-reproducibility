#!/usr/bin/env bash
# Build one or more images. Wraps `docker build` directly rather than
# `docker compose build` so build args (portability, platform) stay
# explicit and idempotent - re-running just hits Docker's layer cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-all}"
PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"
WSCLEAN_PORTABLE="${WSCLEAN_PORTABLE:-ON}"

build_wsclean() {
  echo "==> building ri-reproducibility/wsclean:v3.7 (platform=${PLATFORM}, PORTABLE=${WSCLEAN_PORTABLE})"
  docker build --platform "${PLATFORM}" \
    -f docker/wsclean/Dockerfile \
    --build-arg WSCLEAN_PORTABLE="${WSCLEAN_PORTABLE}" \
    -t ri-reproducibility/wsclean:v3.7 .
}

build_r2d2() {
  echo "==> building ri-reproducibility/r2d2:cpu (platform=${PLATFORM})"
  docker build --platform "${PLATFORM}" \
    -f docker/r2d2/Dockerfile \
    -t ri-reproducibility/r2d2:cpu .
}

build_meqtrees() {
  echo "==> building ri-reproducibility/meqtrees:kern-10 (platform=${PLATFORM})"
  docker build --platform "${PLATFORM}" \
    -f docker/meqtrees/Dockerfile \
    -t ri-reproducibility/meqtrees:kern-10 .
}

build_polychord() {
  echo "==> building ri-reproducibility/polychord:lite (platform=${PLATFORM})"
  docker build --platform "${PLATFORM}" \
    -f docker/polychord/Dockerfile \
    -t ri-reproducibility/polychord:lite .
}

case "${TARGET}" in
  wsclean) build_wsclean ;;
  r2d2) build_r2d2 ;;
  meqtrees) build_meqtrees ;;
  polychord) build_polychord ;;
  all) build_wsclean && build_r2d2 && build_meqtrees && build_polychord ;;
  *) echo "usage: $0 [all|wsclean|r2d2|meqtrees|polychord]" >&2; exit 1 ;;
esac
