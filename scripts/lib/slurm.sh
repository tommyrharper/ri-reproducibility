# shellcheck shell=bash  # sourced, so no shebang
# On a cluster login node a run submits itself as a Slurm job.
#
# The run scripts call ns_should_submit after claiming their run directory and
# before sizing anything: inside an allocation (sintr, a batch script) and on a
# host with no Slurm they carry on in place; on a login node they hand the
# same command to sbatch and leave. The job runs the script again with the
# whole environment (NS_*, OUTPUT_DIR, SBATCH_*), so it resolves the same
# settings the login node would have, against the allocation's cores and memory.
#
# Account, partition and time limit are sbatch's own SBATCH_* input variables
# (`./ri search --account X` sets SBATCH_ACCOUNT); the two that have a sane
# default get one here. There is no default account: `mybalance` lists yours.

ns_should_submit() {
  [ -z "${SLURM_JOB_ID:-}" ] && [ "${NS_SBATCH:-1}" = 1 ] && command -v sbatch >/dev/null 2>&1
}

# ns_submit_run <run dir> <MB per rank> <command...>
#
# One node, one task (mpirun forks the ranks), named after the run so squeue
# and ns_run_is_live can find it, stdout beside the run's own logs. With an
# explicit NS_MPI_PROCS the job is sized to it; otherwise it takes the node.
ns_submit_run() {
  local run_dir="$1" mb_per_rank="$2" cmd size
  shift 2
  printf -v cmd '%q ' "$@"
  if [ -n "${NS_MPI_PROCS:-}" ]; then
    size=(--cpus-per-task "${NS_MPI_PROCS}"
          --mem "$((NS_MPI_PROCS * mb_per_rank + ${NS_RANK_BUDGET_HEADROOM_MB:-4096}))")
  else
    size=(--exclusive --mem 0)
  fi
  export SBATCH_PARTITION="${SBATCH_PARTITION:-icelake}"
  export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-36:00:00}"
  echo "Submitting ${run_dir##*/} to Slurm (${SBATCH_PARTITION}, ${SBATCH_TIMELIMIT}," \
    "${SBATCH_ACCOUNT:-default account}); ./ri runs and squeue -u ${USER:-$(id -un)} track it."
  OUTPUT_DIR="${run_dir}" sbatch \
    --job-name "${run_dir##*/}" \
    --chdir "${REPO_ROOT}" \
    --output "${run_dir}/slurm-%j.out" \
    --nodes 1 --ntasks 1 "${size[@]}" \
    --wrap "exec ${cmd}"
}

# `bash scripts/lib/slurm.sh --self-check`: when a run submits, and what it
# submits, against a fake sbatch that records its arguments and environment.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  _dir="$(mktemp -d)"
  trap 'rm -rf "${_dir}"' EXIT
  mkdir -p "${_dir}/bin"
  # shellcheck disable=SC2016  # the $vars are for the fake sbatch's own sh
  printf '#!/bin/sh\nprintf "%%s\\n" "$@" >"%s/args"\necho "$OUTPUT_DIR $SBATCH_PARTITION $SBATCH_TIMELIMIT" >"%s/env"\necho 4242\n' \
    "${_dir}" "${_dir}" >"${_dir}/bin/sbatch"
  chmod +x "${_dir}/bin/sbatch"
  REPO_ROOT="${_dir}/repo"
  unset SLURM_JOB_ID NS_SBATCH NS_MPI_PROCS SBATCH_PARTITION SBATCH_TIMELIMIT

  ns_should_submit && { echo "FAIL: no sbatch on PATH, so the run must stay in place"; exit 1; }
  export PATH="${_dir}/bin:${PATH}"
  ns_should_submit || { echo "FAIL: sbatch on PATH outside a job means submit"; exit 1; }
  SLURM_JOB_ID=1 ns_should_submit && { echo "FAIL: inside a job the run must stay in place"; exit 1; }
  NS_SBATCH=0 ns_should_submit && { echo "FAIL: NS_SBATCH=0 must keep the run in place"; exit 1; }

  _run="${_dir}/results/wsclean-vlaa-20260101T000000Z"
  _out="$(ns_submit_run "${_run}" 200 scripts/run-nested-sampling.sh)"
  case "${_out}" in
    *"Submitting wsclean-vlaa-20260101T000000Z"*"icelake, 36:00:00"*4242) ;;
    *) echo "FAIL: the submission must say what it did, got: ${_out}"; exit 1 ;;
  esac
  [ "$(cat "${_dir}/env")" = "${_run} icelake 36:00:00" ] \
    || { echo "FAIL: the job must inherit the run directory and the sbatch defaults, got: $(cat "${_dir}/env")"; exit 1; }
  _args="$(tr '\n' ' ' <"${_dir}/args")"
  for _want in "--job-name wsclean-vlaa-20260101T000000Z" "--chdir ${REPO_ROOT}" \
               "--output ${_run}/slurm-%j.out" "--nodes 1 --ntasks 1 --exclusive --mem 0" \
               "--wrap exec scripts/run-nested-sampling.sh"; do
    case "${_args}" in
      *"${_want}"*) ;;
      *) echo "FAIL: sbatch was not given '${_want}': ${_args}"; exit 1 ;;
    esac
  done

  # An explicit rank count sizes the job instead of taking the node, and the
  # caller's partition and time are left alone.
  SBATCH_PARTITION=sapphire SBATCH_TIMELIMIT=01:00:00 NS_MPI_PROCS=4 \
    ns_submit_run "${_run}" 3500 scripts/run-nested-sampling-r2d2.sh >/dev/null
  _args="$(tr '\n' ' ' <"${_dir}/args")"
  case "${_args}" in
    *"--nodes 1 --ntasks 1 --cpus-per-task 4 --mem 18096 --wrap exec scripts/run-nested-sampling-r2d2.sh"*) ;;
    *) echo "FAIL: NS_MPI_PROCS must size the job, got: ${_args}"; exit 1 ;;
  esac
  case "${_args}" in
    *--exclusive*) echo "FAIL: a sized job must not also be exclusive: ${_args}"; exit 1 ;;
  esac
  [ "$(cat "${_dir}/env")" = "${_run} sapphire 01:00:00" ] \
    || { echo "FAIL: the caller's partition and time must win, got: $(cat "${_dir}/env")"; exit 1; }

  echo "slurm self-check passed"
fi
