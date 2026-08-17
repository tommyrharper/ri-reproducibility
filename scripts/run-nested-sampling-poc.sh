#!/usr/bin/env bash
# Run the cheap WSClean x VLA.A PolyChord proof of concept.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"
MEQTREES_IMAGE="${MEQTREES_IMAGE:-ri-reproducibility/meqtrees:kern-10}"
POLYCHORD_IMAGE="${POLYCHORD_IMAGE:-ri-reproducibility/polychord:lite}"
WSCLEAN_IMAGE="${WSCLEAN_IMAGE:-ri-reproducibility/wsclean:v3.7}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/nested-sampling-poc/wsclean-vlaa-${RUN_ID}}"
NS_NLIVE="${NS_NLIVE:-8}"
NS_NUM_REPEATS="${NS_NUM_REPEATS:-2}"
NS_MAX_NDEAD="${NS_MAX_NDEAD:-12}"
NS_SEED="${NS_SEED:-41}"

if [ -z "${DOCKER_SOCKET:-}" ]; then
  DOCKER_SOCKET="/var/run/docker.sock"
fi
if ! docker info >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

RUN_COMMAND=(
  docker run --rm --platform "${PLATFORM}"
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -v "${DOCKER_SOCKET}:/var/run/docker.sock"
  -w "${REPO_ROOT}"
  -e REPO_ROOT="${REPO_ROOT}"
  -e MEQTREES_IMAGE="${MEQTREES_IMAGE}"
  -e WSCLEAN_IMAGE="${WSCLEAN_IMAGE}"
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
  "${POLYCHORD_IMAGE}"
  --output-dir "${OUTPUT_DIR}"
  --repo-root "${REPO_ROOT}"
  --meqtrees-image "${MEQTREES_IMAGE}"
  --wsclean-image "${WSCLEAN_IMAGE}"
  --nlive "${NS_NLIVE}"
  --num-repeats "${NS_NUM_REPEATS}"
  --max-ndead "${NS_MAX_NDEAD}"
  --seed "${NS_SEED}"
  --platform "${PLATFORM}"
)

scripts/record-environment.sh \
  --tool polychord \
  --image "${POLYCHORD_IMAGE}" \
  --config docs/nested-sampling.md \
  -- "${RUN_COMMAND[@]}"

"${RUN_COMMAND[@]}"

echo "OK: nested-sampling PoC output in ${OUTPUT_DIR}"
