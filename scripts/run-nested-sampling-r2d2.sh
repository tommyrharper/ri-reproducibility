#!/usr/bin/env bash
# Run the R2D2 x VLA.A PolyChord nested-sampling search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/progress-bar.sh
source "${REPO_ROOT}/scripts/lib/progress-bar.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/nested-sampling/r2d2-vlaa-${RUN_ID}}"

# Needed before the launches, because the containers' commands below want one
# FIFO pair per rank to already exist. On Linux `nproc` is what mpirun will
# see too, since the daemon is the host kernel. On macOS the daemon runs
# inside its own VM (Docker Desktop / Colima), which can be handed fewer
# vCPUs than `sysctl` reports for the host - `-np` gets sized from a CPU
# count mpirun's container never has, and it refuses to launch at all
# ("not enough slots"). Ask the daemon what it actually has instead.
if command -v nproc >/dev/null 2>&1; then
  HOST_CPUS="$(nproc)"
else
  HOST_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null || sysctl -n hw.ncpu)"
fi
# Rank count is what costs memory - 3.4GB of warm imaging worker each - and
# `min(NS_NLIVE, host CPUs)` knows nothing about how much the host has left,
# so on its own it will happily ask for more than the box holds and let the
# OOM killer score the difference as the search's best points. See
# scripts/lib/rank-budget.sh.
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"
if [ -z "${NS_MPI_PROCS:-}" ]; then
  if [ "${NS_NLIVE}" -lt "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="${NS_NLIVE}"
  else
    NS_MPI_PROCS="${HOST_CPUS}"
  fi
  # Kept for the thread split below: this is the run's fair share of the
  # host, and it should not grow just because memory forced the rank count
  # down. It is memory pressure that clamps the ranks, and memory pressure
  # means another run is already using the cores this one would take.
  NS_MPI_PROCS_UNCLAMPED="${NS_MPI_PROCS}"
  NS_MPI_PROCS="$(ns_budget_ranks "${NS_MPI_PROCS}" "${NS_R2D2_MB_PER_RANK}" r2d2)"
else
  # Explicitly asked for, so it is honoured - but said out loud if it will
  # not fit, because the way this run fails is silent.
  ns_budget_warn_if_over "${NS_MPI_PROCS}" "${NS_R2D2_MB_PER_RANK}" r2d2
  NS_MPI_PROCS_UNCLAMPED="${NS_MPI_PROCS}"
fi

if [ -z "${R2D2_OMP_THREADS:-}" ]; then
  R2D2_OMP_THREADS="$(( HOST_CPUS / NS_MPI_PROCS_UNCLAMPED ))"
  if [ "${R2D2_OMP_THREADS}" -lt 1 ]; then
    R2D2_OMP_THREADS=1
  fi
fi

mkdir -p "${OUTPUT_DIR}"
# Written before anything can go wrong, so that a run which stops - out of
# memory, Ctrl-C, reboot - still says how to start it again exactly.
# shellcheck source=scripts/lib/run-config.sh
. "${REPO_ROOT}/scripts/lib/run-config.sh"
write_run_config "${OUTPUT_DIR}" r2d2
# The workers are reached over FIFOs, so these have to sit on the bind mount the
# rank's container and the sidecars both see - REPO_ROOT, which OUTPUT_DIR is
# under by default. Point OUTPUT_DIR outside the repo and the ranks simply fall
# back to starting their own workers.
SIMULATE_FIFO_DIR="${OUTPUT_DIR}/.simulate-workers"
R2D2_FIFO_DIR="${OUTPUT_DIR}/.r2d2-workers"
rm -rf "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
mkdir -p "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
for ((rank = 0; rank < NS_MPI_PROCS; rank++)); do
  mkfifo "${SIMULATE_FIFO_DIR}/${rank}.in" "${SIMULATE_FIFO_DIR}/${rank}.out"
  mkfifo "${R2D2_FIFO_DIR}/${rank}.in" "${R2D2_FIFO_DIR}/${rank}.out"
done

# Shared by every rank, started here so the daemon is not hit by one
# `docker run` per rank per image the moment the ranks come up.
. "${REPO_ROOT}/scripts/lib/start-sidecars.sh"
# Every stage of an evaluation runs in one of these two: simulate and the
# MS-to-`.mat` convert in the MeqTrees container, imaging in the R2D2 one. The
# checkpoint mount point stays `/checkpoints` so that the `ckpt_path` every
# `poc-summary.json` records - and merge-nested-sampling-runs.py compares - is
# not a host path.
#
# Both containers run one worker per rank as their own command instead of the
# default `sleep infinity` - the meqtrees one as a worker per FIFO pair, the
# R2D2 one as a single process that imports torch once and forks the pool.
# Neither worker can answer for a while after it starts - ~0.5s of Timba,
# meqserver, the first TDL compile and the first predict for simulate, ~0.9s of
# `import torch` and the R2D2 modules for imaging - and PolyChord asks every rank for its first live point at the same
# moment, so all of it used to land on the wall clock inside evaluation one
# (measured ~2.3s of imaging on evaluation one against a ~0.25s steady state).
# Started as the containers' commands it runs while the PolyChord container, the
# manifest, mpirun and PolyChord's own setup still have to happen. It is the
# command and not a `docker exec` because an exec cannot be issued until `docker
# run` has returned, ~0.1s after the container's command has already started,
# and head start is the entire point here.
#
# The single quotes are deliberate: $1, $2 and ${fifo} are for the containers'
# own sh, which gets its arguments below, not for this one.
# shellcheck disable=SC2016
sidecar_launch "${PLATFORM}" "${MEQTREES_IMAGE}" -- sh -c '
  for fifo in "$1"/*.in; do
    [ -e "${fifo}" ] || continue
    python3 /opt/ri-nested-sampling/simulate_point_source_ms.py \
      --serve --fifo "${fifo%.in}" &
  done
  exec sleep infinity
' sh "${SIMULATE_FIFO_DIR}"
# The thread caps are on the container here rather than on a per-rank `docker
# exec`: torch and finufft read them at import time and every rank gets the same
# value anyway.
#
# OMP_WAIT_POLICY=PASSIVE because the parallel regions here are tiny - a 128x128
# NUFFT - and libgomp's default is to spin for the rest of its timeslice after
# each one. With one worker per rank that spinning is a second thread per rank
# burning a core it never uses: 8 workers imaging at 2 threads each measured
# 27.7 requests/s spinning against 50.4 passive, and the sampler's wall clock
# fell 17-22% (10 of 10 interleaved A/B pairs). Do not translate this into a
# lower R2D2_OMP_THREADS - passive 2 threads matches 1 thread here and the
# checkpointed UNet passes, which this parameter space cannot run, want them.
# shellcheck disable=SC2016
sidecar_launch "${PLATFORM}" "${R2D2_IMAGE}" \
  -v "${CHECKPOINTS_DIR}:/checkpoints:ro" \
  -e OMP_NUM_THREADS="${R2D2_OMP_THREADS}" \
  -e MKL_NUM_THREADS="${R2D2_OMP_THREADS}" \
  -e OPENBLAS_NUM_THREADS="${R2D2_OMP_THREADS}" \
  -e OMP_WAIT_POLICY=PASSIVE \
  -- sh -c '
  python3 "$2" --fifo-dir "$1" &
  exec sleep infinity
' sh "${R2D2_FIFO_DIR}" "${REPO_ROOT}/scripts/lib/nested_sampling/r2d2_serve.py"
# The PolyChord container joins the sidecars: `docker run` of it costs ~0.7s
# where `docker exec` into a running one costs ~0.03s, and starting it here
# overlaps that cost with the two workers' containers and the manifest write
# instead of paying it in front of rank 0. It keeps the socket mount because a
# rank whose FIFO pool is missing falls back to `docker exec`ing its own worker.
sidecar_launch "${PLATFORM}" "${POLYCHORD_IMAGE}" \
  -v "${DOCKER_SOCKET}:/var/run/docker.sock"
POLYCHORD_CONTAINER="${SIDECAR_NAME}"

# A dead daemon fails the launches above too, but they are backgrounded, so this
# is where the run says so. Here rather than in front of them because at this
# point the ~0.06s overlaps the containers starting.
if ! docker info --format '{{.NCPU}}' >/dev/null 2>&1; then
  echo "FATAL: Docker daemon is not available" >&2
  exit 1
fi

RUN_COMMAND=(
  docker exec
  -w "${REPO_ROOT}"
  -e REPO_ROOT="${REPO_ROOT}"
  -e MEQTREES_IMAGE="${MEQTREES_IMAGE}"
  -e R2D2_IMAGE="${R2D2_IMAGE}"
  -e CHECKPOINTS_DIR="${CHECKPOINTS_DIR}"
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
  -e NS_MPI_PROCS="${NS_MPI_PROCS}"
  -e NS_SIDECARS="${NS_SIDECARS}"
  -e NS_SIMULATE_FIFO_DIR="${SIMULATE_FIFO_DIR}"
  -e NS_R2D2_FIFO_DIR="${R2D2_FIFO_DIR}"
  -e NS_ENABLE_PARAMS="${NS_ENABLE_PARAMS:-}"
  -e NS_DISABLE_PARAMS="${NS_DISABLE_PARAMS:-}"
  # numpy's OpenBLAS in this image spawns one busy-waiting worker thread per
  # host CPU, in every rank. Nothing here has a BLAS call big enough to want
  # them (the largest is a norm over a 128x128 image), so on a 20-CPU host the
  # 8 default ranks spent ~10 cores spinning and starved the real work.
  -e OMP_NUM_THREADS=1
  -e OPENBLAS_NUM_THREADS=1
  # Open MPI's default point-to-point selection opens the cm PML, which opens
  # the MTL framework, which has libfabric scan every provider it can find -
  # ~0.19s of MPI_Init on this host, on every rank at the same moment, for a job
  # that never leaves one container. ob1 over shared memory is what it settles on
  # anyway; naming it skips the search. Measured: slowest rank's `from mpi4py
  # import MPI` 0.25s -> 0.05s at 8 ranks.
  -e OMPI_MCA_pml=ob1
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -e R2D2_OMP_THREADS="${R2D2_OMP_THREADS}"
  "${POLYCHORD_CONTAINER}"
  mpirun
  --allow-run-as-root
  # The rank count comes from `nproc`, which counts hardware threads, but Open
  # MPI's default slot count is physical cores. On any SMT host the two
  # disagree and mpirun refuses to launch ("not enough slots"). This makes Open
  # MPI count the same units the rank count was derived from.
  --use-hwthread-cpus
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_r2d2.py
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

# Only now: writing the manifest above is ~0.4s of `docker image inspect` and
# `git` that overlaps with the containers coming up.
sidecar_wait

mkdir -p "${OUTPUT_DIR}/evaluations"
run_with_progress "${OUTPUT_DIR}/evaluations" "${NS_MAX_NDEAD}" -- "${RUN_COMMAND[@]}"

rm -rf "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
echo "OK: nested-sampling R2D2 output in ${OUTPUT_DIR}"
