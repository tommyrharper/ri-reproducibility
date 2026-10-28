#!/usr/bin/env bash
# Run the R2D2 x VLA.A PolyChord nested-sampling search.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/progress-bar.sh
source "${REPO_ROOT}/scripts/lib/progress-bar.sh"

# R2D2_DEVICE=cuda runs the U-Net on a GPU: the CUDA build of the R2D2 image,
# `--nv` on its pool, and on a login node a job on the ampere partition with
# one GPU (slurm.sh). Everything else about the run is the same.
case "${R2D2_DEVICE:=cpu}" in
  cpu) ;;
  cuda) R2D2_SIF="${R2D2_CUDA_SIF}" ;;
  *) echo "FATAL: R2D2_DEVICE must be cpu or cuda, got '${R2D2_DEVICE}'" >&2; exit 1 ;;
esac
export R2D2_DEVICE
# A CPU forward runs through ONNX Runtime unless R2D2_RUNTIME=torch: 33-44%
# more evals/s, log(Z) within 0.25 sigma of torch's at convergence
# (docs/csd3-speed.md, rounds 140-145). GPU runs keep torch. Set before
# rank-budget.sh, which budgets onnx ranks larger.
: "${R2D2_RUNTIME:=$([ "${R2D2_DEVICE}" = cpu ] && echo onnx || echo torch)}"
export R2D2_RUNTIME
ns_require_sifs "${MEQTREES_SIF}" "${R2D2_SIF}" "${POLYCHORD_SIF}"

# Inside a Slurm job this is the allocation, not the node. OMP_NUM_THREADS and
# OMP_THREAD_LIMIT are dropped because GNU nproc honours them, and CSD3's login
# environment exports OMP_NUM_THREADS=1, which sbatch carries into the job: a
# bare nproc there reads 1 on a 76-core node and the run gets one rank.
# Rank count comes first because the FIFO setup needs it.
HOST_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
# Rank count drives memory use; rank-budget.sh clamps it to available memory.
# shellcheck source=scripts/lib/rank-budget.sh
. "${REPO_ROOT}/scripts/lib/rank-budget.sh"
# shellcheck source=scripts/lib/run-config.sh
. "${REPO_ROOT}/scripts/lib/run-config.sh"
ns_refuse_missing_checkpoints "${CHECKPOINTS_DIR}" "${R2D2_CKPT_NAME}"
# Claim the default only after guards, so refused runs leave no empty result;
# named directories may exist, but not while a job is still in them.
if [ -n "${OUTPUT_DIR:-}" ]; then
  ns_refuse_live_run "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  # Normalize once so containment and recorded paths are consistent.
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  ns_refuse_unmounted_run "${OUTPUT_DIR}"
else
  OUTPUT_DIR="$(ns_claim_run_dir "${REPO_ROOT}/results/nested-sampling" r2d2-vlaa-)"
fi
ns_warn_file_quota "${OUTPUT_DIR}" "${NS_NLIVE}" "${NS_NUM_REPEATS}" "${NS_MAX_NDEAD}"
# On a cluster login node the search leaves here: the claimed directory and the
# whole environment go to a Slurm job that runs this script again inside the
# allocation, where the cores and memory below are the job's (docs/cluster.md).
# shellcheck source=scripts/lib/slurm.sh
. "${REPO_ROOT}/scripts/lib/slurm.sh"
if ns_should_submit; then
  export OUTPUT_DIR
  if [ "${R2D2_DEVICE}" = cuda ]; then
    NS_SLURM_GPUS=1
  fi
  NS_SLURM_GPUS="${NS_SLURM_GPUS:-0}" ns_submit_run "${OUTPUT_DIR}" "${NS_R2D2_MB}" scripts/run-nested-sampling-r2d2.sh \
    || { rmdir "${OUTPUT_DIR}" 2>/dev/null; exit 1; }
  exit 0
fi
# Inside an allocation it did not submit (sintr), the run carries on in the
# job's own environment rather than the login node's (slurm.sh).
export OUTPUT_DIR
ns_enter_job_env "${OUTPUT_DIR}" scripts/run-nested-sampling-r2d2.sh
# On a node without a GPU every worker would fail its first request, and a
# failed evaluation is scored rather than fatal - so it is checked here, once,
# before anything starts.
if [ "${R2D2_DEVICE}" = cuda ]; then
  if ! "${APPTAINER}" exec --nv "${R2D2_SIF}" python3 -c \
      'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "FATAL: R2D2_DEVICE=cuda but no GPU is visible here - submit from a login node (it asks for one), or sintr -p ampere --gres=gpu:1" >&2
    rmdir "${OUTPUT_DIR}" 2>/dev/null || true
    exit 1
  fi
  NS_R2D2_MAX_RANKS="${NS_R2D2_CUDA_MAX_RANKS}"
  # A TensorRT worker's 25 engines hold ~5.1GB on the card against eager's
  # ~3.1GB of weights, so an 80GB A100 fits 12, not 19 (csd3-speed.md, round 170).
  # R2D2_TRT_SHARED=1 holds them once, in one forward server, and its workers
  # stay off the card, so eager's sizing is an over-estimate (round 208).
  if [ "${R2D2_RUNTIME}" = trt ] && [ "${R2D2_TRT_SHARED:-0}" != 1 ]; then
    NS_R2D2_CUDA_MB_PER_RANK="${NS_R2D2_CUDA_TRT_MB_PER_RANK}"
  fi
  # Each worker holds all 25 checkpoints on the GPU. Past what the card holds,
  # workers die with CUDA OOM and their evaluations score as failures - a
  # silent fake worst case - so an explicit rank count over it is refused.
  gpu_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "${gpu_mb}" ]; then
    gpu_ranks="$((gpu_mb * 95 / 100 / NS_R2D2_CUDA_MB_PER_RANK))"
    if [ -n "${NS_MPI_PROCS:-}" ] && [ "${NS_MPI_PROCS}" -gt "${gpu_ranks}" ]; then
      echo "FATAL: ${NS_MPI_PROCS} R2D2 workers need ~$((NS_MPI_PROCS * NS_R2D2_CUDA_MB_PER_RANK))MB of GPU memory; this GPU has ${gpu_mb}MB, enough for ${gpu_ranks} (NS_R2D2_CUDA_MB_PER_RANK)" >&2
      exit 1
    fi
    if [ "${NS_R2D2_MAX_RANKS}" -gt "${gpu_ranks}" ]; then
      NS_R2D2_MAX_RANKS="${gpu_ranks}"
    fi
  fi
fi
if [ -z "${NS_MPI_PROCS:-}" ]; then
  if [ "${NS_NLIVE}" -lt "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="${NS_NLIVE}"
  else
    NS_MPI_PROCS="${HOST_CPUS}"
  fi
  # R2D2 saturates a host well below one rank per core, so the memory clamp
  # below is not the right stopping point: it would take every rank the memory
  # allows for throughput that stopped rising. See NS_R2D2_MAX_RANKS in
  # defaults.toml and docs/nested-sampling-throughput.md.
  if [ "${NS_MPI_PROCS}" -gt "${NS_R2D2_MAX_RANKS}" ]; then
    NS_MPI_PROCS="${NS_R2D2_MAX_RANKS}"
  fi
  # PolyChord's root rank only hands out points: it never evaluates, so a
  # rank per CPU leaves one CPU without an R2D2 worker. One more rank fills it:
  # 38 -> 39 ranks on 38 icelake CPUs won 5 of 5 pairs, +3.7% evals/s, with
  # the same time per evaluation (csd3-speed.md, round 186).
  if [ "${R2D2_DEVICE}" = cpu ] && [ "${NS_MPI_PROCS}" = "${HOST_CPUS}" ]; then
    NS_MPI_PROCS="$((HOST_CPUS + 1))"
  fi
  NS_MPI_PROCS="$(ns_budget_ranks "${NS_MPI_PROCS}" "${NS_R2D2_MB}" r2d2)"
else
  # Explicitly asked for, so it is honoured - but said out loud if it will
  # not fit, because the way this run fails is silent.
  ns_budget_warn_if_over "${NS_MPI_PROCS}" "${NS_R2D2_MB}" r2d2
fi
# mpirun counts one slot per CPU and refuses -np past it.
if [ "${NS_MPI_PROCS}" -gt "${HOST_CPUS}" ]; then
  NS_MPI_OVERSUBSCRIBE="${NS_MPI_OVERSUBSCRIBE:-1}"
fi

if [ -z "${R2D2_OMP_THREADS:-}" ]; then
  R2D2_OMP_THREADS="$(( (HOST_CPUS + NS_MPI_PROCS - 1) / NS_MPI_PROCS ))"
  if [ "${R2D2_OMP_THREADS}" -lt 1 ]; then
    R2D2_OMP_THREADS=1
  fi
fi
R2D2_INTEROP_THREADS="${R2D2_INTEROP_THREADS:-0}"
# Ceiling for r2d2_thread_count() in common.py, which splits it over the
# evaluations in flight. Async only: any count but 1 moves R2D2's output by
# ~1e-7, so a synchronous run would stop repeating itself from its seed.
# At most 16: a lone request is fastest there and slower at 38, and capping
# the tail at 16 shortened it from 42-91s to 30-47s (csd3-speed.md, round 20).
if [ "${NS_SYNCHRONOUS}" = 0 ]; then
  R2D2_MAX_THREADS="${R2D2_MAX_THREADS:-$(( HOST_CPUS < 16 ? HOST_CPUS : 16 ))}"
else
  R2D2_MAX_THREADS="${R2D2_OMP_THREADS}"
fi

# Written before anything can go wrong, so that a run which stops - out of
# memory, Ctrl-C, reboot - still says how to start it again exactly.
write_run_config "${OUTPUT_DIR}" r2d2
# The workers are reached over FIFOs, so these have to sit on the bind mount the
# ranks and the pools both see - REPO_ROOT, which OUTPUT_DIR is always under,
# because ns_refuse_unmounted_run above is what makes that true.
SIMULATE_FIFO_DIR="${OUTPUT_DIR}/.simulate-workers"
R2D2_FIFO_DIR="${OUTPUT_DIR}/.r2d2-workers"
rm -rf "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
mkdir -p "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
for ((rank = 0; rank < NS_MPI_PROCS; rank++)); do
  mkfifo "${SIMULATE_FIFO_DIR}/${rank}.in" "${SIMULATE_FIFO_DIR}/${rank}.out"
  mkfifo "${R2D2_FIFO_DIR}/${rank}.in" "${R2D2_FIFO_DIR}/${rank}.out"
done

. "${REPO_ROOT}/scripts/lib/start-sidecars.sh"
# Each evaluation uses MeqTrees for simulate and MS-to-`.mat`, and R2D2 for
# imaging. One worker per rank in each pool, started here so their ~0.5-0.9s
# warm-up overlaps PolyChord startup and its first live-point request.
#
# The simulate workers are each kept alive by their own loop: a worker that
# dies - or exits when its rank's end of the FIFO closes - is started again
# and reopens the same pair, and the rank reconnects (common.py). The pid
# file is how a rank kills a worker that has wedged (FifoWorker.kill). The
# working tree is bound over the baked copy so the run executes the code in
# this checkout; there is no `docker build` here to keep them in step.
#
# The single quotes are deliberate: $1, ${fifo} and ${base} are for the
# container's own sh, which gets its arguments below, not for this one.
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
# The R2D2 pool warms torch once and forks a worker per rank, starting any
# that dies again (serve_pool in r2d2_serve.py, read live off the bind mount).
# Keep `/checkpoints` stable: summaries record this path and merge uses it.
#
# The thread caps are on the pool rather than per rank: torch and finufft read
# them at import time and every rank gets the same value anyway.
#
# OMP_WAIT_POLICY=PASSIVE because the parallel regions here are tiny - a 128x128
# NUFFT - and libgomp's default is to spin for the rest of its timeslice after
# each one. With one worker per rank that spinning is a second thread per rank
# burning a core it never uses: 8 workers imaging at 2 threads each measured
# 27.7 requests/s spinning against 50.4 passive, and the sampler's wall clock
# fell 17-22% (10 of 10 interleaved A/B pairs). Do not translate this into a
# lower R2D2_OMP_THREADS - passive 2 threads matches 1 thread here and the
# checkpointed UNet passes, which this parameter space cannot run, want them.
R2D2_POOL_FLAGS=()
if [ "${R2D2_DEVICE}" = cuda ]; then
  R2D2_POOL_FLAGS=(--nv)
  # NS_CUDA_MPS=0 leaves the workers time-slicing the card, for an A/B.
  # R2D2_TRT_SHARED=1 never starts MPS: its one server is the only process on
  # the card, and through an MPS daemon it ran 32% slower (steady 61.9 against
  # 90.9 evals/s, 3 of 3 interleaved pairs; docs/csd3-speed.md, round 211).
  if [ "${NS_CUDA_MPS}" != 0 ] && [ "${R2D2_TRT_SHARED:-0}" != 1 ]; then
    sidecar_cuda_mps_flags
    R2D2_POOL_FLAGS+=(${CUDA_MPS_FLAGS[@]+"${CUDA_MPS_FLAGS[@]}"})
  fi
fi
# R2D2_RUNTIME=onnx runs the CPU U-Net through ONNX Runtime, ~34% faster a
# forward but not bit-identical (patch_onnx_network in r2d2_serve.py,
# docs/csd3-speed.md rounds 140-141); the CPU default since round 145. r2d2.sif has no onnxruntime, so it is
# pip-installed once into the work-dir cache - compute nodes reach PyPI - and
# its exported checkpoints are cached beside it. numpy is dropped from the
# install so the image's own is the one imported. A failed install (no PyPI)
# runs the pool on torch rather than ending the search.
# r2d2_pip_install DIR MODULE PKG...: pip-install PKG... once into DIR/py,
# which must then hold MODULE; numpy is dropped so the image's own is imported,
# and TensorRT's Windows builder resources (2GB) with it.
r2d2_pip_install() {
  local dir="$1" module="$2"
  shift 2
  [ -d "${dir}/py/${module}" ] && return 0
  mkdir -p "${dir}"
  rm -rf "${dir}/py.$$"
  if "${APPTAINER}" exec "${R2D2_SIF}" python3 -m pip install --quiet --disable-pip-version-check \
    --target "${dir}/py.$$" "$@"; then
    rm -rf "${dir}/py.$$"/numpy "${dir}/py.$$"/numpy-* "${dir}/py.$$"/numpy.libs \
      "${dir}/py.$$"/tensorrt_libs/libnvinfer_builder_resource_win_*
    mv -T "${dir}/py.$$" "${dir}/py" 2>/dev/null || rm -rf "${dir}/py.$$"
  else
    rm -rf "${dir}/py.$$"
    return 1
  fi
}
case "${R2D2_DEVICE}:${R2D2_RUNTIME}" in
  *:torch) ;;
  cpu:onnx | cuda:trt)
    R2D2_ONNX_DIR="${_ri_cache}/r2d2-onnx"
    if ! r2d2_pip_install "${R2D2_ONNX_DIR}" onnxruntime onnxruntime==1.23.2 onnx==1.23.2; then
      echo "WARNING: could not pip-install onnxruntime; this run's U-Net runs on torch (R2D2_RUNTIME=torch)" >&2
      R2D2_RUNTIME=torch
    fi
    # R2D2_RUNTIME=trt (GPU only) builds each exported checkpoint into a
    # TensorRT fp32 engine: ~+84% forwards through an A100 at 12 workers
    # against eager's best (patch_trt_network, csd3-speed.md rounds 168-171).
    # tensorrt-cu13 matches the CUDA 13 torch in r2d2-cuda.sif; its builds are
    # cached beside it, per GPU model and image size.
    R2D2_TRT_DIR="${_ri_cache}/r2d2-trt"
    if [ "${R2D2_RUNTIME}" = trt ] && ! r2d2_pip_install "${R2D2_TRT_DIR}" tensorrt --no-deps \
      tensorrt-cu13==11.4.0.106 tensorrt-cu13-bindings==11.4.0.106 tensorrt-cu13-libs==11.4.0.106; then
      echo "WARNING: could not pip-install tensorrt; this run's U-Net runs on torch (R2D2_RUNTIME=torch)" >&2
      R2D2_RUNTIME=torch
    fi
    if [ "${R2D2_RUNTIME}" != torch ]; then
      R2D2_POOL_FLAGS+=(--bind "${R2D2_ONNX_DIR}:/opt/r2d2-onnx")
    fi
    if [ "${R2D2_RUNTIME}" = trt ]; then
      R2D2_POOL_FLAGS+=(--bind "${R2D2_TRT_DIR}:/opt/r2d2-trt" --env R2D2_TRT_STATUS="${OUTPUT_DIR}/r2d2-engine.txt"
        --env NS_IMAGE_DIM="${NS_IMAGE_DIM}" --env R2D2_TRT_SHARED="${R2D2_TRT_SHARED:-0}")
    fi
    ;;
  *) echo "FATAL: R2D2_RUNTIME must be torch or onnx on a CPU, torch or trt on a GPU; got '${R2D2_RUNTIME}' for R2D2_DEVICE=${R2D2_DEVICE}" >&2; exit 1 ;;
esac
sidecar_launch "${R2D2_SIF}" \
  ${R2D2_POOL_FLAGS[@]+"${R2D2_POOL_FLAGS[@]}"} \
  --env R2D2_DEVICE="${R2D2_DEVICE}" \
  --env R2D2_RUNTIME="${R2D2_RUNTIME}" \
  --bind "${CHECKPOINTS_DIR}:/checkpoints:ro" \
  --env OMP_NUM_THREADS="${R2D2_OMP_THREADS}" \
  --env MKL_NUM_THREADS="${R2D2_OMP_THREADS}" \
  --env OPENBLAS_NUM_THREADS="${R2D2_OMP_THREADS}" \
  --env R2D2_INTEROP_THREADS="${R2D2_INTEROP_THREADS:-0}" \
  --env R2D2_INTERLEAVE_WEIGHTS="${R2D2_INTERLEAVE_WEIGHTS:-1}" \
  --env R2D2_NUMPY_WEIGHTS="${R2D2_NUMPY_WEIGHTS:-1}" \
  --env R2D2_FAST_IO="${R2D2_FAST_IO:-1}" \
  --env R2D2_NUFFT_SORT="${R2D2_NUFFT_SORT:-0}" \
  --env R2D2_WEIGHTS_PER_NODE="${R2D2_WEIGHTS_PER_NODE:-auto}" \
  --env R2D2_CUDA_BLOCKING_SYNC="${R2D2_CUDA_BLOCKING_SYNC:-1}" \
  --env OMP_WAIT_POLICY=PASSIVE \
  -- python3 "${REPO_ROOT}/scripts/lib/nested_sampling/r2d2_serve.py" --fifo-dir "${R2D2_FIFO_DIR}"

sidecar_binds
RUN_COMMAND=(
  env
  REPO_ROOT="${REPO_ROOT}"
  MEQTREES_IMAGE="${MEQTREES_SIF}"
  R2D2_IMAGE="${R2D2_SIF}"
  CHECKPOINTS_DIR="${CHECKPOINTS_DIR}"
  NS_MPI_PROCS="${NS_MPI_PROCS}"
  NS_IMAGE_DIM="${NS_IMAGE_DIM}"
  NS_MPI_OVERSUBSCRIBE="${NS_MPI_OVERSUBSCRIBE:-}"
  NS_SIMULATE_FIFO_DIR="${SIMULATE_FIFO_DIR}"
  NS_SCRATCH_DIR="${NS_SCRATCH_DIR}"
  NS_R2D2_FIFO_DIR="${R2D2_FIFO_DIR}"
  NS_ENABLE_PARAMS="${NS_ENABLE_PARAMS:-}"
  NS_DISABLE_PARAMS="${NS_DISABLE_PARAMS:-}"
  NS_SYNCHRONOUS="${NS_SYNCHRONOUS}"
  NS_KEEP_MEASUREMENT_SETS="${NS_KEEP_MEASUREMENT_SETS}"
  # numpy's OpenBLAS in this image spawns one busy-waiting worker thread per
  # host CPU, in every rank. Nothing here has a BLAS call big enough to want
  # them (the largest is a norm over a 128x128 image), so on a 20-CPU host the
  # 8 default ranks spent ~10 cores spinning and starved the real work.
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  # Open MPI's default point-to-point selection opens the cm PML, which opens
  # the MTL framework, which has libfabric scan every provider it can find -
  # ~0.19s of MPI_Init on this host, on every rank at the same moment, for a job
  # that never leaves one node. ob1 over shared memory is what it settles on
  # anyway; naming it skips the search. Measured: slowest rank's `from mpi4py
  # import MPI` 0.25s -> 0.05s at 8 ranks.
  OMPI_MCA_pml=ob1
  # mpirun forks every rank itself on this one node. Inside a Slurm job Open
  # MPI would otherwise read the job's task count (one, for a job that asks
  # for cores rather than tasks) as the slot count and refuse -np.
  OMPI_MCA_ras=^slurm
  OMPI_MCA_plm=^slurm
  R2D2_OMP_THREADS="${R2D2_OMP_THREADS}"
  R2D2_MAX_THREADS="${R2D2_MAX_THREADS}"
  R2D2_INTEROP_THREADS="${R2D2_INTEROP_THREADS:-1}"
  "${APPTAINER}" exec --pwd "${REPO_ROOT}"
  "${SIDECAR_BINDS[@]}"
  --bind "${REPO_ROOT}/scripts/lib/nested_sampling:/opt/ri-nested-sampling"
  "${POLYCHORD_SIF}"
  mpirun
  # Match Open MPI's slot units to `nproc`'s hardware-thread rank count.
  --use-hwthread-cpus
  ${NS_MPI_OVERSUBSCRIBE:+--oversubscribe}
  -np "${NS_MPI_PROCS}"
  python3 /opt/ri-nested-sampling/polychord_r2d2.py
  --output-dir "${OUTPUT_DIR}"
  --repo-root "${REPO_ROOT}"
  --meqtrees-image "${MEQTREES_SIF}"
  --r2d2-image "${R2D2_SIF}"
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
rm -rf "${SIMULATE_FIFO_DIR}" "${R2D2_FIFO_DIR}"
echo "OK: nested-sampling R2D2 output in ${OUTPUT_DIR}"
