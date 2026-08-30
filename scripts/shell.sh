#!/usr/bin/env bash
# Open bash in a built image with this repo's usual mounts.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

case "${1:-}" in
  wsclean)
    docker run --rm -it \
      -v "${REPO_ROOT}/data:/data" -v "${REPO_ROOT}/results:/results" \
      --entrypoint bash "${WSCLEAN_IMAGE}"
    ;;
  r2d2)
    # shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
    source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"
    docker run --rm -it \
      "${R2D2_DOCKER_ENV_FLAGS[@]}" \
      -v "${REPO_ROOT}/data:/data" \
      -v "${REPO_ROOT}/checkpoints:/checkpoints" \
      -v "${REPO_ROOT}/results:/results" \
      --entrypoint bash "${R2D2_IMAGE}"
    ;;
  meqtrees)
    docker run --rm -it \
      -v "${REPO_ROOT}/data:/data" -v "${REPO_ROOT}/results:/results" \
      --entrypoint bash "${MEQTREES_IMAGE}"
    ;;
  polychord)
    # Mounts the repo at its host path and the Docker socket: this image drives
    # the other containers, so paths it passes on must resolve on the host too.
    docker run --rm -it \
      -v "${REPO_ROOT}:${REPO_ROOT}" -w "${REPO_ROOT}" \
      -e REPO_ROOT="${REPO_ROOT}" \
      -v /var/run/docker.sock:/var/run/docker.sock \
      --entrypoint bash "${POLYCHORD_IMAGE}"
    ;;
  *)
    echo "usage: $0 <wsclean|r2d2|meqtrees|polychord>" >&2
    exit 1
    ;;
esac
