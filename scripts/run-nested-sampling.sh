#!/usr/bin/env bash
# Run the WSClean x VLA.A PolyChord nested-sampling search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/progress-bar.sh
source "${REPO_ROOT}/scripts/lib/progress-bar.sh"

ns_require_sifs "${MEQTREES_SIF}" "${WSCLEAN_SIF}" "${POLYCHORD_SIF}"

# Inside a Slurm job this is the allocation, not the node.
HOST_CPUS="$(nproc)"
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"
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
# On a cluster login node the search leaves here: the claimed directory and the
# whole environment go to a Slurm job that runs this script again inside the
# allocation, where the cores and memory below are the job's (docs/cluster.md).
# shellcheck source=scripts/lib/slurm.sh
. "${REPO_ROOT}/scripts/lib/slurm.sh"
if ns_should_submit; then
  export OUTPUT_DIR
  ns_submit_run "${OUTPUT_DIR}" "${NS_WSCLEAN_MB_PER_RANK}" scripts/run-nested-sampling.sh \
    || { rmdir "${OUTPUT_DIR}" 2>/dev/null; exit 1; }
  exit 0
fi
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

write_run_config "${OUTPUT_DIR}" wsclean
# One FIFO pair per rank per worker kind, under the run directory so the
# pools and the ranks - all bound to REPO_ROOT - see them at the same path.
SIMULATE_FIFO_DIR="${OUTPUT_DIR}/.simulate-workers"
WSCLEAN_FIFO_DIR="${OUTPUT_DIR}/.wsclean-workers"
rm -rf "${SIMULATE_FIFO_DIR}" "${WSCLEAN_FIFO_DIR}"
mkdir -p "${SIMULATE_FIFO_DIR}" "${WSCLEAN_FIFO_DIR}"
for ((rank = 0; rank < NS_MPI_PROCS; rank++)); do
  mkfifo "${SIMULATE_FIFO_DIR}/${rank}.in" "${SIMULATE_FIFO_DIR}/${rank}.out"
  mkfifo "${WSCLEAN_FIFO_DIR}/${rank}.in" "${WSCLEAN_FIFO_DIR}/${rank}.out"
done

. "${REPO_ROOT}/scripts/lib/start-sidecars.sh"
# One simulate worker per rank, each kept alive by its own loop: a worker
# that dies - or exits when its rank's end of the FIFO closes - is started
# again and reopens the same pair, and the rank reconnects (common.py). The
# pid file is how a rank kills a worker that has wedged (FifoWorker.kill).
# The working tree is bound over the baked copy so the run executes the code
# in this checkout; there is no `docker build` here to keep them in step.
#
# Single quotes defer $1, ${fifo} and ${base} to the container's own sh.
# shellcheck disable=SC2016
sidecar_launch "${MEQTREES_SIF}" \
  --bind "${REPO_ROOT}/scripts/lib/nested_sampling:/opt/ri-nested-sampling" \
  -- sh -c '
  for fifo in "$1"/*.in; do
    [ -e "${fifo}" ] || continue
    base="${fifo%.in}"
    ( while :; do
        python3 /opt/ri-nested-sampling/simulate_point_source_ms.py \
          --serve --fifo "${base}" & echo $! >"${base}.pid"; wait $!
      done ) &
  done
  wait
' sh "${SIMULATE_FIFO_DIR}"
# The WSClean fork server, one per rank over a FIFO pair like the simulate
# workers: a rank inside polychord.sif cannot start another SIF, so the zygote
# the Docker branch spawned per rank is a host-started pool here.
# shellcheck disable=SC2016
sidecar_launch "${WSCLEAN_SIF}" -- sh -c '
  for fifo in "$1"/*.in; do
    [ -e "${fifo}" ] || continue
    base="${fifo%.in}"
    ( while :; do
        wsclean-zygote <"${fifo}" >"${base}.out" & echo $! >"${base}.pid"; wait $!
      done ) &
  done
  wait
' sh "${WSCLEAN_FIFO_DIR}"

mapfile -t POLYCHORD_BINDS < <(sidecar_binds)
RUN_COMMAND=(
  env
  REPO_ROOT="${REPO_ROOT}"
  MEQTREES_IMAGE="${MEQTREES_SIF}"
  WSCLEAN_IMAGE="${WSCLEAN_SIF}"
  NS_MPI_PROCS="${NS_MPI_PROCS}"
  NS_IMAGE_DIM="${NS_IMAGE_DIM}"
  NS_MPI_OVERSUBSCRIBE="${NS_MPI_OVERSUBSCRIBE:-}"
  NS_SIMULATE_FIFO_DIR="${SIMULATE_FIFO_DIR}"
  NS_WSCLEAN_FIFO_DIR="${WSCLEAN_FIFO_DIR}"
  NS_SCRATCH_DIR="${NS_SCRATCH_DIR}"
  NS_ENABLE_PARAMS="${NS_ENABLE_PARAMS:-}"
  NS_DISABLE_PARAMS="${NS_DISABLE_PARAMS:-}"
  NS_SYNCHRONOUS="${NS_SYNCHRONOUS}"
  NS_KEEP_MEASUREMENT_SETS="${NS_KEEP_MEASUREMENT_SETS}"
  NS_WSCLEAN_MGAIN="${NS_WSCLEAN_MGAIN}"
  NS_WSCLEAN_NITER="${NS_WSCLEAN_NITER}"
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  OMPI_MCA_pml=ob1
  # mpirun forks every rank itself on this one node. Inside a Slurm job Open
  # MPI would otherwise read the job's task count (one, for a job that asks
  # for cores rather than tasks) as the slot count and refuse -np.
  OMPI_MCA_ras=^slurm
  OMPI_MCA_plm=^slurm
  "${APPTAINER}" exec --pwd "${REPO_ROOT}"
  "${POLYCHORD_BINDS[@]}"
  --bind "${REPO_ROOT}/scripts/lib/nested_sampling:/opt/ri-nested-sampling"
  "${POLYCHORD_SIF}"
  mpirun
  --use-hwthread-cpus
  ${NS_MPI_OVERSUBSCRIBE:+--oversubscribe}
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_wsclean.py
  --output-dir "${OUTPUT_DIR}"
  --meqtrees-image "${MEQTREES_SIF}"
  --wsclean-image "${WSCLEAN_SIF}"
  --nlive "${NS_NLIVE}"
  --num-repeats "${NS_NUM_REPEATS}"
  --max-ndead "${NS_MAX_NDEAD}"
  --seed "${NS_SEED}"
  --metric "${NS_METRIC}"
  --platform "${PLATFORM}"
)

scripts/record-environment.sh \
  --tool polychord \
  --image "${POLYCHORD_SIF}" \
  --config docs/nested-sampling.md \
  -- "${RUN_COMMAND[@]}"

mkdir -p "${OUTPUT_DIR}/evaluations"
run_with_retries "${NS_RETRIES}" "${OUTPUT_DIR}" "${NS_MAX_NDEAD}" "${NS_NLIVE}" -- "${RUN_COMMAND[@]}"

# One row in benchmarks.jsonl per finished search, so a change to this repo can
# be shown to have helped rather than argued about; see
# docs/nested-sampling-benchmarks.md. Best effort: the run is already done, and
# no measurement of it is worth failing it after the fact.
uv run scripts/bench.py record "${OUTPUT_DIR}" || true

# Pools first: their loops start a worker again the moment the ranks let go
# of the FIFOs, and that worker writes its pid file into the directory being
# removed. The EXIT trap would kill them too, but only after this rm.
_sidecar_remove
rm -rf "${SIMULATE_FIFO_DIR}" "${WSCLEAN_FIFO_DIR}"
echo "OK: nested-sampling output in ${OUTPUT_DIR}"
