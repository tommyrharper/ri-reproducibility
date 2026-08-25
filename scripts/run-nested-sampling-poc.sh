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
# Before the launches, because the meqtrees container's command below needs one
# FIFO pair per rank to already exist - which costs ~0.06s of serial `docker
# info` in front of a ~0.4s container start that does not otherwise need its
# answer. It doubles as the daemon-availability check - a dead daemon fails the
# launches too, but this is where the run says so.
if ! HOST_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null)"; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi
if [ -z "${NS_MPI_PROCS:-}" ]; then
  if [ "${NS_NLIVE}" -lt "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="${NS_NLIVE}"
  else
    NS_MPI_PROCS="${HOST_CPUS}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
# The simulate workers are reached over FIFOs, so this has to sit on the bind
# mount the rank's container and the meqtrees sidecar both see - REPO_ROOT,
# which OUTPUT_DIR is under by default. Point OUTPUT_DIR outside the repo and
# the ranks simply fall back to starting their own workers.
SIMULATE_FIFO_DIR="${OUTPUT_DIR}/.simulate-workers"
rm -rf "${SIMULATE_FIFO_DIR}"
mkdir -p "${SIMULATE_FIFO_DIR}"
for ((rank = 0; rank < NS_MPI_PROCS; rank++)); do
  mkfifo "${SIMULATE_FIFO_DIR}/${rank}.in" "${SIMULATE_FIFO_DIR}/${rank}.out"
done

# Shared by every rank, started here so the daemon is not hit by one
# `docker run` per rank per image the moment the ranks come up. The PolyChord
# container joins them: `docker run` of it costs ~0.7s where `docker exec` into
# a running one costs ~0.03s, and starting it here overlaps that cost with the
# sidecars and the manifest write instead of paying it in front of rank 0.
#
# The meqtrees container's command is one simulate worker per rank instead of
# the default `sleep infinity`. A worker is not ready to answer for ~0.5s -
# interpreter, Timba, meqserver, the first TDL compile and the first predict -
# and every rank asks for its first evaluation at the same moment, so all of
# that used to sit on the wall clock inside evaluation one. Started as the
# container's own command it runs while the other two containers, the manifest,
# `docker exec`, mpirun and PolyChord's own setup still have to happen. It is
# the container's command and not a `docker exec` because an exec cannot be
# issued until `docker run` has returned, ~0.1s after the container's command
# has already started, and head start is the entire point here.
. "${REPO_ROOT}/scripts/lib/start-sidecars.sh"
sidecar_launch "${PLATFORM}" "${MEQTREES_IMAGE}" -- sh -c '
  for fifo in "$1"/*.in; do
    [ -e "${fifo}" ] || continue
    python3 /opt/ri-nested-sampling/simulate_point_source_ms.py \
      --serve --fifo "${fifo%.in}" &
  done
  exec sleep infinity
' sh "${SIMULATE_FIFO_DIR}"
sidecar_launch "${PLATFORM}" "${WSCLEAN_IMAGE}"
sidecar_launch "${PLATFORM}" "${POLYCHORD_IMAGE}" \
  -v "${DOCKER_SOCKET}:/var/run/docker.sock"
POLYCHORD_CONTAINER="${SIDECAR_NAME}"

RUN_COMMAND=(
  docker exec
  -w "${REPO_ROOT}"
  -e REPO_ROOT="${REPO_ROOT}"
  -e MEQTREES_IMAGE="${MEQTREES_IMAGE}"
  -e WSCLEAN_IMAGE="${WSCLEAN_IMAGE}"
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
  -e NS_MPI_PROCS="${NS_MPI_PROCS}"
  -e NS_SIDECARS="${NS_SIDECARS}"
  -e NS_SIMULATE_FIFO_DIR="${SIMULATE_FIFO_DIR}"
  # numpy's OpenBLAS in this image spawns one busy-waiting worker thread per
  # host CPU, in every rank. Nothing here has a BLAS call big enough to want
  # them (the largest is a norm over a 128x128 image), so on a 20-CPU host the
  # 8 default ranks spent ~10 cores spinning and starved the real work.
  -e OMP_NUM_THREADS=1
  -e OPENBLAS_NUM_THREADS=1
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  "${POLYCHORD_CONTAINER}"
  mpirun
  --allow-run-as-root
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_wsclean_poc.py
  --output-dir "${OUTPUT_DIR}"
  --repo-root "${REPO_ROOT}"
  --meqtrees-image "${MEQTREES_IMAGE}"
  --wsclean-image "${WSCLEAN_IMAGE}"
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

# Only now: writing the manifest above is ~0.4s of `docker image inspect` and
# `git` that overlaps with the containers coming up.
sidecar_wait

"${RUN_COMMAND[@]}"

rm -rf "${SIMULATE_FIFO_DIR}"
echo "OK: nested-sampling PoC output in ${OUTPUT_DIR}"
