#!/usr/bin/env bash
# Validate MS -> R2D2 .mat conversion using simulated or existing data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"
MEQTREES_IMAGE="${MEQTREES_IMAGE:-ri-reproducibility/meqtrees:kern-10}"
R2D2_IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"

if ! docker info >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi

if [ -z "${DOCKER_SOCKET:-}" ]; then
  DOCKER_SOCKET="/var/run/docker.sock"
fi

docker run --rm --platform "${PLATFORM}" \
  -v "${REPO_ROOT}:${REPO_ROOT}" \
  -v "${DOCKER_SOCKET}:/var/run/docker.sock" \
  -w "${REPO_ROOT}" \
  -e REPO_ROOT="${REPO_ROOT}" \
  -e MEQTREES_IMAGE="${MEQTREES_IMAGE}" \
  -e R2D2_IMAGE="${R2D2_IMAGE}" \
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}" \
  --entrypoint python3 \
  ri-reproducibility/polychord:lite \
  /opt/ri-nested-sampling/check_ms_to_r2d2_mat.py \
  --repo-root "${REPO_ROOT}" \
  --meqtrees-image "${MEQTREES_IMAGE}" \
  --r2d2-image "${R2D2_IMAGE}" \
  --platform "${PLATFORM}" \
  ${MS_PATH:+--ms-path "${MS_PATH}"}

echo "OK: MS -> R2D2 .mat bridge self-check passed"
