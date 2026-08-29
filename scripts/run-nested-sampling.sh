#!/usr/bin/env bash
# Run the WSClean x VLA.A PolyChord nested-sampling search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/progress-bar.sh
source "${REPO_ROOT}/scripts/lib/progress-bar.sh"

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
# A WSClean rank is cheap next to an R2D2 one (~0.2GB against ~3.4GB), but it
# is not free, and this host runs both at once from several agent sessions.
# Same guard, same reservation, so the two size themselves around each other
# rather than both assuming an empty box. See scripts/lib/rank-budget.sh.
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
# Claimed here rather than named at the top of the script, so a run refused by
# the memory guard above leaves no empty directory for `./ri runs` and the
# health report to puzzle over. An OUTPUT_DIR given on the command line is the
# caller's to name and may already exist - but not while a job is still in it;
# the default one is claimed, because two searches started in the same second
# would otherwise share it.
if [ -n "${OUTPUT_DIR:-}" ]; then
  ns_refuse_live_run "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  # Absolute and `..`-free from here on, so that the containment test below is
  # a string comparison and so that run.env, run.log and the health report all
  # name the run the same way whatever the caller typed.
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  ns_refuse_unmounted_run "${OUTPUT_DIR}"
else
  OUTPUT_DIR="$(ns_claim_run_dir "${REPO_ROOT}/results/nested-sampling" wsclean-vlaa-)"
fi
# Written before anything can go wrong, so that a run which stops - out of
# memory, Ctrl-C, reboot - still says how to start it again exactly.
write_run_config "${OUTPUT_DIR}" wsclean
# The simulate workers are reached over FIFOs, so this has to sit on the bind
# mount the rank's container and the meqtrees sidecar both see - REPO_ROOT,
# which OUTPUT_DIR is always under, because ns_refuse_unmounted_run above is
# what makes that true.
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
# The single quotes are deliberate: $1 and ${fifo} are for the container's
# own sh, which gets the fifo directory as its argument below, not for this one.
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
  -e WSCLEAN_IMAGE="${WSCLEAN_IMAGE}"
  -e DOCKER_DEFAULT_PLATFORM="${PLATFORM}"
  -e NS_MPI_PROCS="${NS_MPI_PROCS}"
  -e NS_SIDECARS="${NS_SIDECARS}"
  -e NS_SIMULATE_FIFO_DIR="${SIMULATE_FIFO_DIR}"
  -e NS_ENABLE_PARAMS="${NS_ENABLE_PARAMS:-}"
  -e NS_DISABLE_PARAMS="${NS_DISABLE_PARAMS:-}"
  -e NS_SYNCHRONOUS="${NS_SYNCHRONOUS}"
  -e NS_KEEP_MEASUREMENT_SETS="${NS_KEEP_MEASUREMENT_SETS}"
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
  "${POLYCHORD_CONTAINER}"
  mpirun
  --allow-run-as-root
  # The rank count comes from `nproc`, which counts hardware threads, but Open
  # MPI's default slot count is physical cores. On any SMT host the two
  # disagree and mpirun refuses to launch ("not enough slots"). This makes Open
  # MPI count the same units the rank count was derived from.
  --use-hwthread-cpus
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_wsclean.py
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

mkdir -p "${OUTPUT_DIR}/evaluations"
run_with_retries "${NS_RETRIES}" "${OUTPUT_DIR}" "${NS_MAX_NDEAD}" "${NS_NLIVE}" -- "${RUN_COMMAND[@]}"

rm -rf "${SIMULATE_FIFO_DIR}"
echo "OK: nested-sampling output in ${OUTPUT_DIR}"
