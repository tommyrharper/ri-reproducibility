#!/usr/bin/env bash
# Run the cheap R2D2 x VLA.A PolyChord proof of concept.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"
MEQTREES_IMAGE="${MEQTREES_IMAGE:-ri-reproducibility/meqtrees:kern-10}"
POLYCHORD_IMAGE="${POLYCHORD_IMAGE:-ri-reproducibility/polychord:lite}"
R2D2_IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/nested-sampling-poc/r2d2-vlaa-${RUN_ID}}"
NS_NLIVE="${NS_NLIVE:-8}"
NS_NUM_REPEATS="${NS_NUM_REPEATS:-2}"
NS_MAX_NDEAD="${NS_MAX_NDEAD:-12}"
NS_SEED="${NS_SEED:-41}"
NS_METRIC="${NS_METRIC:-off_source_rms_jy}"

if [ -z "${DOCKER_SOCKET:-}" ]; then
  # Rootless Docker listens on $XDG_RUNTIME_DIR/docker.sock, not
  # /var/run/docker.sock, and the sidecar containers this run launches need
  # the real host path to bind-mount. DOCKER_HOST is what points the CLI at
  # it, so derive from that and fall back to the rootful default.
  case "${DOCKER_HOST:-}" in
    unix://*) DOCKER_SOCKET="${DOCKER_HOST#unix://}" ;;
    *) DOCKER_SOCKET="/var/run/docker.sock" ;;
  esac
fi
if ! docker info >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi

HOST_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
if [ -z "${NS_MPI_PROCS:-}" ]; then
  if [ "${NS_NLIVE}" -lt "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="${NS_NLIVE}"
  else
    NS_MPI_PROCS="${HOST_CPUS}"
  fi
fi

if [ -z "${R2D2_OMP_THREADS:-}" ]; then
  R2D2_OMP_THREADS="$(( HOST_CPUS / NS_MPI_PROCS ))"
  if [ "${R2D2_OMP_THREADS}" -lt 1 ]; then
    R2D2_OMP_THREADS=1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

RUN_COMMAND=(
  docker run --rm --platform "${PLATFORM}"
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -v "${DOCKER_SOCKET}:/var/run/docker.sock"
  -w "${REPO_ROOT}"
  -e REPO_ROOT="${REPO_ROOT}"
  -e MEQTREES_IMAGE="${MEQTREES_IMAGE}"
  -e R2D2_IMAGE="${R2D2_IMAGE}"
  -e CHECKPOINTS_DIR="${CHECKPOINTS_DIR}"
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
  -e NS_MPI_PROCS="${NS_MPI_PROCS}"
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -e R2D2_OMP_THREADS="${R2D2_OMP_THREADS}"
  --entrypoint mpirun
  "${POLYCHORD_IMAGE}"
  --allow-run-as-root
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_r2d2_poc.py
  --output-dir "${OUTPUT_DIR}"
  --repo-root "${REPO_ROOT}"
  --meqtrees-image "${MEQTREES_IMAGE}"
  --r2d2-image "${R2D2_IMAGE}"
  --checkpoints-dir "${CHECKPOINTS_DIR}"
  --nlive "${NS_NLIVE}"
  --num-repeats "${NS_NUM_REPEATS}"
  --max-ndead "${NS_MAX_NDEAD}"
  --seed "${NS_SEED}"
  --metric "${NS_METRIC}"
  --platform "${PLATFORM}"
)

scripts/record-environment.sh \
  --tool polychord \
  --image "${POLYCHORD_IMAGE}" \
  --config docs/nested-sampling.md \
  -- "${RUN_COMMAND[@]}"

"${RUN_COMMAND[@]}"

echo "OK: nested-sampling R2D2 PoC output in ${OUTPUT_DIR}"
