#!/usr/bin/env bash
# Run the WSClean x VLA.A PolyChord nested-sampling search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/progress-bar.sh
source "${REPO_ROOT}/scripts/lib/progress-bar.sh"

if command -v nproc >/dev/null 2>&1; then
  HOST_CPUS="$(nproc)"
else
  HOST_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null || sysctl -n hw.ncpu)"
fi
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"
if [ -z "${NS_MPI_PROCS:-}" ]; then
  if [ "${NS_NLIVE}" -lt "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="${NS_NLIVE}"
  else
    NS_MPI_PROCS="${HOST_CPUS}"
  fi
  NS_MPI_PROCS="$(ns_budget_ranks "${NS_MPI_PROCS}" "${NS_WSCLEAN_MB_PER_RANK}" wsclean)"
else
  ns_budget_warn_if_over "${NS_MPI_PROCS}" "${NS_WSCLEAN_MB_PER_RANK}" wsclean
fi

# shellcheck source=scripts/lib/run-config.sh
. "${REPO_ROOT}/scripts/lib/run-config.sh"
if [ -n "${OUTPUT_DIR:-}" ]; then
  ns_refuse_live_run "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  ns_refuse_unmounted_run "${OUTPUT_DIR}"
else
  OUTPUT_DIR="$(ns_claim_run_dir "${REPO_ROOT}/results/nested-sampling" wsclean-vlaa-)"
fi
write_run_config "${OUTPUT_DIR}" wsclean
SIMULATE_FIFO_DIR="${OUTPUT_DIR}/.simulate-workers"
rm -rf "${SIMULATE_FIFO_DIR}"
mkdir -p "${SIMULATE_FIFO_DIR}"
for ((rank = 0; rank < NS_MPI_PROCS; rank++)); do
  mkfifo "${SIMULATE_FIFO_DIR}/${rank}.in" "${SIMULATE_FIFO_DIR}/${rank}.out"
done

. "${REPO_ROOT}/scripts/lib/start-sidecars.sh"
# Single quotes defer $1 and ${fifo} expansion to the sidecar shell.
# shellcheck disable=SC2016
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

if ! docker info --format '{{.NCPU}}' >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi

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
  -e NS_SCRATCH_DIR="${NS_SCRATCH_DIR}"
  -e NS_ENABLE_PARAMS="${NS_ENABLE_PARAMS:-}"
  -e NS_DISABLE_PARAMS="${NS_DISABLE_PARAMS:-}"
  -e NS_SYNCHRONOUS="${NS_SYNCHRONOUS}"
  -e NS_KEEP_MEASUREMENT_SETS="${NS_KEEP_MEASUREMENT_SETS}"
  -e NS_WSCLEAN_MGAIN="${NS_WSCLEAN_MGAIN}"
  -e OMP_NUM_THREADS=1
  -e OPENBLAS_NUM_THREADS=1
  -e OMPI_MCA_pml=ob1
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  "${POLYCHORD_CONTAINER}"
  mpirun
  --allow-run-as-root
  --use-hwthread-cpus
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_wsclean.py
  --output-dir "${OUTPUT_DIR}"
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

# Record environment after sidecars start, overlapping its inspection cost.
sidecar_wait

mkdir -p "${OUTPUT_DIR}/evaluations"
run_with_retries "${NS_RETRIES}" "${OUTPUT_DIR}" "${NS_MAX_NDEAD}" "${NS_NLIVE}" -- "${RUN_COMMAND[@]}"

# One row in benchmarks.jsonl per finished search, so a change to this repo can
# be shown to have helped rather than argued about; see
# docs/nested-sampling-benchmarks.md. Best effort: the run is already done, and
# no measurement of it is worth failing it after the fact.
uv run scripts/bench.py record "${OUTPUT_DIR}" || true

rm -rf "${SIMULATE_FIFO_DIR}"
echo "OK: nested-sampling output in ${OUTPUT_DIR}"
