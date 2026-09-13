#!/usr/bin/env bash
# Open bash in an image's SIF with this repo's usual binds.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

mkdir -p "${REPO_ROOT}/data" "${REPO_ROOT}/results" "${CHECKPOINTS_DIR}"

case "${1:-}" in
  wsclean)
    ns_require_sifs "${WSCLEAN_SIF}"
    "${APPTAINER}" shell \
      --bind "${REPO_ROOT}/data:/data" --bind "${REPO_ROOT}/results:/results" \
      "${WSCLEAN_SIF}"
    ;;
  r2d2)
    ns_require_sifs "${R2D2_SIF}"
    # shellcheck source=scripts/lib/r2d2-thread-env.sh
    source "${REPO_ROOT}/scripts/lib/r2d2-thread-env.sh"
    "${APPTAINER}" shell --pwd /opt/r2d2/R2D2-RI \
      "${R2D2_ENV_FLAGS[@]}" \
      --bind "${REPO_ROOT}/data:/data" \
      --bind "${CHECKPOINTS_DIR}:/checkpoints" \
      --bind "${REPO_ROOT}/results:/results" \
      "${R2D2_SIF}"
    ;;
  meqtrees)
    ns_require_sifs "${MEQTREES_SIF}"
    "${APPTAINER}" shell \
      --bind "${REPO_ROOT}/data:/data" --bind "${REPO_ROOT}/results:/results" \
      "${MEQTREES_SIF}"
    ;;
  polychord)
    # The repo at its host path with the working tree's nested_sampling over
    # the baked copy, the way a run sees it.
    ns_require_sifs "${POLYCHORD_SIF}"
    "${APPTAINER}" shell --pwd "${REPO_ROOT}" \
      --bind "${REPO_ROOT}" \
      --bind "${REPO_ROOT}/scripts/lib/nested_sampling:/opt/ri-nested-sampling" \
      --env "REPO_ROOT=${REPO_ROOT}" \
      "${POLYCHORD_SIF}"
    ;;
  *)
    echo "usage: $0 <wsclean|r2d2|meqtrees|polychord>" >&2
    exit 1
    ;;
esac
