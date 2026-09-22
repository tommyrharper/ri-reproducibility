# shellcheck shell=bash  # sourced, so no shebang
# On a cluster login node a run submits itself as a Slurm job.
#
# The run scripts call ns_should_submit after claiming their run directory and
# before sizing anything: inside an allocation (sintr, a batch script) and on a
# host with no Slurm they carry on in place; on a login node they hand the
# same command to sbatch and leave. The job runs the script again with the
# run's settings (NS_*, R2D2_*, OUTPUT_DIR, ...; scripts/lib/job-env.sh), so it
# resolves the same settings the login node would have, against the
# allocation's cores and memory - and with nothing else of the login node's
# environment, which on CSD3 reaches into /home.
#
# Account, partition and time limit are sbatch's own SBATCH_* input variables
# (`./ri search --account X` sets SBATCH_ACCOUNT); the two that have a sane
# default get one here. The time defaults to 12 hours, the SL3 cap, because a
# limit above the caller's service level is refused outright by sbatch and
# SL1/SL2 can ask for their 36 hours; a run resumes across jobs either way.
# There is no default account: `mybalance` lists yours.

ns_should_submit() {
  [ -z "${SLURM_JOB_ID:-}" ] && [ "${NS_SBATCH:-1}" = 1 ] && command -v sbatch >/dev/null 2>&1
}

# ns_submit_run <run dir> <MB per rank> <command...>
#
# One node, one task (mpirun forks the ranks), named after the run so squeue
# and ns_run_is_live can find it, stdout beside the run's own logs. With an
# explicit NS_MPI_PROCS the job is sized to it; otherwise it takes the node.
# NS_SLURM_GPUS=N (R2D2_DEVICE=cuda sets 1) asks for N GPUs on the ampere
# partition instead, with the 32 cores per GPU CSD3 allows there: a GPU job is
# charged in GPU hours, so the cores and their 8000MB each cost nothing extra.
# The job starts from an empty environment (`--export=NIL`) in
# scripts/lib/job-env.sh, which builds its own and sources the caller's run
# settings from `.job-settings.env` in the run directory, so a run script
# exports OUTPUT_DIR before calling; bench.py submits itself the same way with
# a directory of its own. sbatch still reads SBATCH_* from this shell. Paths
# are physical: the job's REPO_ROOT is, and the settings' directories are.
ns_submit_run() {
  local run_dir="$1" mb_per_rank="$2" cmd size repo="${REPO_ROOT}" settings phys
  shift 2
  # A -CPU project has no GPU hours, and sbatch's own refusal does not say so.
  if [ "${NS_SLURM_GPUS:-0}" -gt 0 ]; then
    case "${SBATCH_ACCOUNT:-}" in
      *-CPU | *-cpu)
        echo "FATAL: a GPU run needs a -GPU project (mybalance lists yours), not ${SBATCH_ACCOUNT}" >&2
        return 1 ;;
    esac
  fi
  phys="$(cd "${repo}" 2>/dev/null && pwd -P)" && repo="${phys}"
  phys="$(cd "${run_dir}" 2>/dev/null && pwd -P)" && run_dir="${phys}"
  settings="${run_dir}/.job-settings.env"
  # shellcheck source=scripts/lib/job-env.sh
  . "$(dirname "${BASH_SOURCE[0]}")/job-env.sh"
  ns_write_job_settings "${settings}" || return
  printf -v cmd '%q ' /bin/bash "${repo}/scripts/lib/job-env.sh" "${settings}" "$@"
  if [ "${NS_SLURM_GPUS:-0}" -gt 0 ]; then
    size=(--gres "gpu:${NS_SLURM_GPUS}" --cpus-per-task "$((32 * NS_SLURM_GPUS))")
    export SBATCH_PARTITION="${SBATCH_PARTITION:-ampere}"
  elif [ -n "${NS_MPI_PROCS:-}" ]; then
    size=(--cpus-per-task "${NS_MPI_PROCS}"
          --mem "$((NS_MPI_PROCS * mb_per_rank + ${NS_RANK_BUDGET_HEADROOM_MB:-4096}))")
  else
    # Not --mem 0: CSD3's sbatch refuses it. --exclusive already brings every
    # core's default memory, i.e. the node.
    size=(--exclusive)
  fi
  export SBATCH_PARTITION="${SBATCH_PARTITION:-icelake}"
  # CSD3's `intr` QoS starts at once but caps a job at one hour, and sbatch
  # refuses the 12-hour default under it rather than trimming it.
  if [ "${SBATCH_QOS:-}" = intr ]; then
    export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-01:00:00}"
  else
    export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-12:00:00}"
  fi
  echo "Submitting ${run_dir##*/} to Slurm (${SBATCH_PARTITION}, ${SBATCH_TIMELIMIT}," \
    "${SBATCH_QOS:+qos ${SBATCH_QOS}, }${SBATCH_ACCOUNT:-default account}); squeue -u ${USER:-$(id -un)} tracks it."
  sbatch \
    --export=NIL \
    --job-name "${run_dir##*/}" \
    --chdir "${repo}" \
    --output "${run_dir}/slurm-%j.out" \
    --nodes 1 --ntasks 1 "${size[@]}" \
    --wrap "exec ${cmd}" || return
  # The readers' shared squeue cache (scripts/slurm_queue.py) predates this job;
  # dropped so `./ri runs` lists it as queued straight away.
  rm -f "${XDG_CACHE_HOME:-${HOME}/.cache}/ri/squeue-${USER:-$(id -un)}.txt"
}

# ns_enter_job_env <run dir> <command...>
#
# A run started inside an allocation it did not submit - `sintr`, a batch
# script of the caller's own - carries that shell's environment, which is the
# login node's, /home and all. So it hands itself to job-env.sh the way a
# submitted job starts, with the run's settings saved beside it, once:
# job-env.sh exports RI_JOB_ENV. Outside an allocation this does nothing;
# NS_JOB_ENV=0 keeps the caller's environment. Callers export OUTPUT_DIR first.
ns_enter_job_env() {
  local run_dir="$1" repo="${REPO_ROOT}" phys
  shift
  [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${RI_JOB_ENV:-}" ] && [ "${NS_JOB_ENV:-1}" = 1 ] || return 0
  phys="$(cd "${repo}" 2>/dev/null && pwd -P)" && repo="${phys}"
  phys="$(cd "${run_dir}" 2>/dev/null && pwd -P)" && run_dir="${phys}"
  # shellcheck source=scripts/lib/job-env.sh
  . "$(dirname "${BASH_SOURCE[0]}")/job-env.sh"
  ns_write_job_settings "${run_dir}/.job-settings.env" || return
  echo "Inside Slurm job ${SLURM_JOB_ID}: carrying on in the job's own environment (scripts/lib/job-env.sh; NS_JOB_ENV=0 to skip)."
  cd "${repo}" || return
  exec /bin/bash "${repo}/scripts/lib/job-env.sh" "${run_dir}/.job-settings.env" "$@"
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
  mkdir -p "${REPO_ROOT}"
  unset SLURM_JOB_ID NS_SBATCH NS_MPI_PROCS SBATCH_PARTITION SBATCH_TIMELIMIT SBATCH_QOS

  # A PATH of only the fake: the real cluster has sbatch on its own PATH.
  PATH="${_dir}/none" ns_should_submit && { echo "FAIL: no sbatch on PATH, so the run must stay in place"; exit 1; }
  export PATH="${_dir}/bin:${PATH}"
  ns_should_submit || { echo "FAIL: sbatch on PATH outside a job means submit"; exit 1; }
  SLURM_JOB_ID=1 ns_should_submit && { echo "FAIL: inside a job the run must stay in place"; exit 1; }
  NS_SBATCH=0 ns_should_submit && { echo "FAIL: NS_SBATCH=0 must keep the run in place"; exit 1; }

  _run="${_dir}/results/wsclean-vlaa-20260101T000000Z"
  mkdir -p "${_run}"
  export OUTPUT_DIR="${_run}" NS_NLIVE=8 UNRELATED=1
  _out="$(ns_submit_run "${_run}" 200 scripts/run-nested-sampling.sh)"
  case "${_out}" in
    *"Submitting wsclean-vlaa-20260101T000000Z"*"icelake, 12:00:00"*4242) ;;
    *) echo "FAIL: the submission must say what it did, got: ${_out}"; exit 1 ;;
  esac
  [ "$(cat "${_dir}/env")" = "${_run} icelake 12:00:00" ] \
    || { echo "FAIL: the job must inherit the run directory and the sbatch defaults, got: $(cat "${_dir}/env")"; exit 1; }
  _args="$(tr '\n' ' ' <"${_dir}/args")"
  for _want in "--export=NIL" "--job-name wsclean-vlaa-20260101T000000Z" "--chdir ${REPO_ROOT}" \
               "--output ${_run}/slurm-%j.out" "--nodes 1 --ntasks 1 --exclusive --wrap" \
               "--wrap exec /bin/bash ${REPO_ROOT}/scripts/lib/job-env.sh ${_run}/.job-settings.env scripts/run-nested-sampling.sh"; do
    case "${_args}" in
      *"${_want}"*) ;;
      *) echo "FAIL: sbatch was not given '${_want}': ${_args}"; exit 1 ;;
    esac
  done

  # The job gets the run's settings and nothing else of this shell.
  if ! grep -qx "export NS_NLIVE=8" "${_run}/.job-settings.env" \
     || ! grep -qx "export OUTPUT_DIR=${_run}" "${_run}/.job-settings.env"; then
    echo "FAIL: the run's settings must be saved for the job: $(cat "${_run}/.job-settings.env")"; exit 1
  fi
  grep -q UNRELATED "${_run}/.job-settings.env" \
    && { echo "FAIL: only run settings may reach the job: $(cat "${_run}/.job-settings.env")"; exit 1; }
  unset UNRELATED

  # An explicit rank count sizes the job instead of taking the node, and the
  # caller's partition and time are left alone.
  SBATCH_PARTITION=sapphire SBATCH_TIMELIMIT=01:00:00 NS_MPI_PROCS=4 \
    ns_submit_run "${_run}" 3500 scripts/run-nested-sampling-r2d2.sh >/dev/null
  _args="$(tr '\n' ' ' <"${_dir}/args")"
  case "${_args}" in
    *"--nodes 1 --ntasks 1 --cpus-per-task 4 --mem 18096 --wrap exec /bin/bash ${REPO_ROOT}/scripts/lib/job-env.sh ${_run}/.job-settings.env scripts/run-nested-sampling-r2d2.sh"*) ;;
    *) echo "FAIL: NS_MPI_PROCS must size the job, got: ${_args}"; exit 1 ;;
  esac
  case "${_args}" in
    *--exclusive*) echo "FAIL: a sized job must not also be exclusive: ${_args}"; exit 1 ;;
  esac
  [ "$(cat "${_dir}/env")" = "${_run} sapphire 01:00:00" ] \
    || { echo "FAIL: the caller's partition and time must win, got: $(cat "${_dir}/env")"; exit 1; }

  # The intr QoS caps a job at an hour, so without a time that is the default;
  # a submission also drops the squeue cache so the job is seen at once.
  unset SBATCH_PARTITION SBATCH_TIMELIMIT
  export XDG_CACHE_HOME="${_dir}/cache"
  mkdir -p "${XDG_CACHE_HOME}/ri"
  touch "${XDG_CACHE_HOME}/ri/squeue-${USER:-$(id -un)}.txt"
  SBATCH_QOS=intr ns_submit_run "${_run}" 200 scripts/run-nested-sampling.sh >/dev/null
  [ "$(cat "${_dir}/env")" = "${_run} icelake 01:00:00" ] \
    || { echo "FAIL: qos intr must default the time to its one-hour cap, got: $(cat "${_dir}/env")"; exit 1; }
  [ -e "${XDG_CACHE_HOME}/ri/squeue-${USER:-$(id -un)}.txt" ] \
    && { echo "FAIL: a submission must drop the squeue cache"; exit 1; }

  # Inside an allocation it did not submit, a run re-execs itself under
  # job-env.sh once; anywhere else it carries on.
  mkdir -p "${REPO_ROOT}/scripts/lib"
  printf '#!/bin/sh\nprintf "%%s\\n" "$@" >"%s/entered"\n' "${_dir}" >"${REPO_ROOT}/scripts/lib/job-env.sh"
  ( unset SLURM_JOB_ID; ns_enter_job_env "${_run}" scripts/run-nested-sampling.sh; echo stayed ) | grep -qx stayed \
    || { echo "FAIL: outside an allocation a run must carry on in place"; exit 1; }
  ( SLURM_JOB_ID=5 RI_JOB_ENV=1 ns_enter_job_env "${_run}" x; echo stayed ) | grep -qx stayed \
    || { echo "FAIL: a run already under job-env.sh must not re-enter it"; exit 1; }
  ( SLURM_JOB_ID=5 NS_JOB_ENV=0 ns_enter_job_env "${_run}" x; echo stayed ) | grep -qx stayed \
    || { echo "FAIL: NS_JOB_ENV=0 must keep the caller's environment"; exit 1; }
  ( SLURM_JOB_ID=5 ns_enter_job_env "${_run}" scripts/run-nested-sampling.sh; echo stayed ) >/dev/null
  [ "$(tr '\n' ' ' <"${_dir}/entered")" = "${_run}/.job-settings.env scripts/run-nested-sampling.sh " ] \
    || { echo "FAIL: sintr must re-enter the run under job-env.sh, got: $(cat "${_dir}/entered" 2>/dev/null)"; exit 1; }

  # A GPU run asks for its GPU and the cores that come with it on ampere, and
  # a -CPU project is refused before anything is written.
  unset SBATCH_PARTITION SBATCH_TIMELIMIT SBATCH_QOS NS_MPI_PROCS
  SBATCH_ACCOUNT=PROJ-GPU NS_SLURM_GPUS=1 ns_submit_run "${_run}" 3500 scripts/run-nested-sampling-r2d2.sh >/dev/null
  _args="$(tr '\n' ' ' <"${_dir}/args")"
  case "${_args}" in
    *"--nodes 1 --ntasks 1 --gres gpu:1 --cpus-per-task 32 --wrap"*) ;;
    *) echo "FAIL: a GPU run must ask for one GPU and its 32 cores, got: ${_args}"; exit 1 ;;
  esac
  case "$(cat "${_dir}/env")" in
    *" ampere "*) ;;
    *) echo "FAIL: a GPU run defaults to the ampere partition, got: $(cat "${_dir}/env")"; exit 1 ;;
  esac
  rm -f "${_dir}/args"
  unset SBATCH_PARTITION
  SBATCH_ACCOUNT=PROJ-CPU NS_SLURM_GPUS=1 ns_submit_run "${_run}" 3500 x >/dev/null 2>&1 \
    && { echo "FAIL: a GPU run on a -CPU project must be refused"; exit 1; }
  [ -e "${_dir}/args" ] && { echo "FAIL: a refused GPU run must not reach sbatch"; exit 1; }

  echo "slurm self-check passed"
fi
