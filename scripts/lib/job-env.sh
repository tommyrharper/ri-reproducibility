#!/bin/bash
# shellcheck shell=bash  # sourced by slurm.sh for ns_write_job_settings, run by every job
# The entry point of every Slurm job a run submits (scripts/lib/slurm.sh).
#
# CSD3 asks that jobs do no I/O on /home, and the interactive tooling on the
# account lives there (a nix-portable store, uv's own Python, the login
# shell's dotfiles). So a job is submitted with `--export=NIL` - not NONE,
# which rebuilds the environment by running the login shell, i.e. ~/.bashrc,
# on the compute node - and starts here with nothing but SLURM_*. This builds
# the job's environment out of the module system, the base OS and hpc-work,
# sources the run settings the login node saved beside the run, refuses to
# start if any tool or path it would use resolves into /home or a Nix store,
# and execs the run script.
#
#   job-env.sh <settings file> <command...>   what sbatch runs
#   job-env.sh --check                         the same environment, checked, on a login node
#   job-env.sh --self-check                    the check itself, against a fake /home
#
# Where things go (RI_WORK_DIR defaults to the directory holding the
# checkout, i.e. hpc-work):
#
#   $RI_WORK_DIR/.local/bin                uv, installed there once (docs/cluster.md)
#   $RI_WORK_DIR/.local/share/uv/python    uv's Pythons, which .venv points into
#   $RI_WORK_DIR/.cache                    XDG_CACHE_HOME, uv's cache, apptainer's cache
#   $RI_WORK_DIR/.ri-job/home              HOME, and the container home (APPTAINER_HOME)
#
# APPTAINER_HOME matters as much as HOME: apptainer mounts the passwd home
# (/home/<user>) into every container whatever $HOME says, unless told a home
# of its own (measured on CSD3 with HOME alone, and with APPTAINER_HOME).

# Run settings a job carries over from the login node: the search's own knobs
# (NS_*, R2D2_*, ...), where its run and images are, and nothing else - not
# PATH, HOME or anything else describing the login node's environment.
RI_JOB_SETTINGS_PATTERN='^(NS|R2D2|RI|WSCLEAN|MEQTREES|POLYCHORD)_[A-Za-z0-9_]*$|^(OUTPUT_DIR|SIF_DIR|CHECKPOINTS_DIR|MS_PATH)$'

# ns_write_job_settings <file>
#
# The caller's run settings as `export NAME=value` lines. A setting naming an
# existing directory, or a file in one, is written as its physical path: the
# job's REPO_ROOT is the checkout's physical path, ns_refuse_unmounted_run
# compares the two as spelled, and ~/rds/hpc-work is spelled through /home.
ns_write_job_settings() {
  local file="$1" name value phys
  : >"${file}" || return
  while IFS= read -r name; do
    [[ "${name}" =~ ${RI_JOB_SETTINGS_PATTERN} ]] || continue
    value="${!name}"
    if [[ "${value}" == /* && -d "${value}" ]]; then
      value="$(cd "${value}" && pwd -P)"
    elif [[ "${value}" == /* ]] && phys="$(cd "$(dirname "${value}")" 2>/dev/null && pwd -P)"; then
      value="${phys%/}/${value##*/}"
    fi
    printf 'export %s=%q\n' "${name}" "${value}" >>"${file}"
  done < <(compgen -e)
}

# `readlink -f` is GNU; BSD readlink (macOS, where CI also runs) has no -f and
# silently resolves nothing, which would let a symlink into a forbidden tree
# look clean. Resolve by hand when it is missing.
_ri_realpath() {
  local path="$1" dir base hops=0 resolved
  if resolved="$(readlink -f -- "${path}" 2>/dev/null)" && [ -n "${resolved}" ]; then
    printf '%s' "${resolved}"
    return
  fi
  while [ -L "${path}" ] && [ "${hops}" -lt 40 ]; do
    dir="$(dirname -- "${path}")"
    base="$(readlink -- "${path}")"
    case "${base}" in
      /*) path="${base}" ;;
      *) path="${dir}/${base}" ;;
    esac
    hops=$((hops + 1))
  done
  dir="$(cd "$(dirname -- "${path}")" 2>/dev/null && pwd -P)" || dir="$(dirname -- "${path}")"
  printf '%s/%s' "${dir%/}" "$(basename -- "${path}")"
}

_ri_forbidden_prefixes() {
  local home
  home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6)"
  tr -s ' ' '\n' <<<"${RI_JOB_FORBIDDEN:-/home /nix ${home}}"
}

# Whether a path, as spelled or once resolved, is under a forbidden prefix.
_ri_is_forbidden() {
  local path="$1" real prefix
  real="$(_ri_realpath "${path}")"
  while IFS= read -r prefix; do
    [ -n "${prefix}" ] || continue
    case "${path}/" in "${prefix%/}"/*) return 0 ;; esac
    case "${real}/" in "${prefix%/}"/*) return 0 ;; esac
  done < <(_ri_forbidden_prefixes)
  return 1
}

# ri_job_env_check: every tool the job runs, every search path and directory
# it uses, and every value it carries, against /home and Nix stores. Prints
# what it checked; returns 1 after listing every problem.
ri_job_env_check() {
  local bad=0 tool path real var entry required
  local -
  set -f  # the value split below must not glob
  required=" ${RI_JOB_REQUIRED_TOOLS:-bash uv apptainer} "
  for tool in bash sh env python3 gcc uv apptainer sbatch squeue; do
    if ! path="$(command -v "${tool}")"; then
      case "${required}" in
        *" ${tool} "*) echo "job-env: ${tool}: not on PATH" >&2; bad=1 ;;
      esac
      continue
    fi
    real="$(_ri_realpath "${path}")"
    echo "job-env: ${tool} -> ${real}"
    if _ri_is_forbidden "${path}"; then
      echo "job-env: ${tool} resolves into a forbidden place: ${path} -> ${real}" >&2; bad=1
    elif grep -aq '/nix/store/' "${real}" 2>/dev/null; then
      # A copied Nix-built binary loads its libraries from the store.
      echo "job-env: ${tool} (${real}) links against a Nix store" >&2; bad=1
    fi
  done

  # The interpreters uv hands the job: the project's .venv (bench.py record)
  # and a bare >=3.11 (the defaults loader).
  local venv_python="${REPO_ROOT_PHYS}/.venv/bin/python"
  if [ -e "${venv_python}" ] || [ -L "${venv_python}" ]; then
    real="$(_ri_realpath "${venv_python}")"
    echo "job-env: .venv python -> ${real}"
    _ri_is_forbidden "${real}" \
      && { echo "job-env: .venv's python is ${real}; rebuild it with uv under hpc-work (docs/cluster.md)" >&2; bad=1; }
  fi
  if [ "${RI_JOB_CHECK_PYTHON:-1}" = 1 ] && command -v uv >/dev/null 2>&1; then
    if real="$(uv python find --no-project '>=3.11' 2>/dev/null)"; then
      real="$(_ri_realpath "${real}")"
      echo "job-env: uv python -> ${real}"
      _ri_is_forbidden "${real}" && { echo "job-env: uv's python resolves into a forbidden place: ${real}" >&2; bad=1; }
    else
      echo "job-env: uv finds no Python >=3.11 under ${UV_PYTHON_INSTALL_DIR:-?}; run \`uv python install 3.13\` there (docs/cluster.md)" >&2
      bad=1
    fi
  fi

  # Every exported value, split into the paths it names - PATH-style lists,
  # directories like HOME, UV_CACHE_DIR or APPTAINER_HOME, a run setting like
  # NS_SCRATCH_DIR. SLURM_* only describe the submission (SLURM_SUBMIT_DIR).
  while IFS= read -r var; do
    case "${var}" in SLURM_*|SPANK_*|BASH_FUNC_*|RI_JOB_FORBIDDEN) continue ;; esac
    case "${!var}" in
      */nix/store/*) echo "job-env: ${var}=${!var} names a Nix store" >&2; bad=1; continue ;;
    esac
    for entry in ${!var//[:=,;]/ }; do
      [[ "${entry}" == /* ]] || continue
      if _ri_is_forbidden "${entry}"; then
        echo "job-env: ${var} names ${entry}" >&2; bad=1
      fi
    done
  done < <(compgen -e)

  [ "${bad}" = 0 ] && echo "job-env: nothing resolves into /home or a Nix store"
  return "${bad}"
}

# The job's environment, in this (already emptied) shell.
_ri_job_env_build() {
  local work
  work="${RI_WORK_DIR:-$(dirname "${REPO_ROOT_PHYS}")}"
  export PATH=/usr/local/bin:/usr/bin:/bin
  USER="$(id -un)"
  export USER LOGNAME="${USER}" SHELL=/bin/bash LANG=C.UTF-8
  export HOME="${work}/.ri-job/home"
  export APPTAINER_HOME="${HOME}"
  export XDG_CACHE_HOME="${work}/.cache"
  export XDG_CONFIG_HOME="${HOME}/.config" XDG_DATA_HOME="${HOME}/.local/share"
  export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"
  export UV_PYTHON_INSTALL_DIR="${work}/.local/share/uv/python"
  # Only uv's own Pythons (the system python3 is 3.6), and never a download
  # from a compute node: `uv python install` is a login-node step.
  export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_DOWNLOADS=never
  export PYTHONNOUSERSITE=1
  export APPTAINER_CACHEDIR="${XDG_CACHE_HOME}/apptainer"
  mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${APPTAINER_CACHEDIR}" || return

  # CSD3's modules, from nothing: `module purge` first, so the default
  # login set (Intel compilers, CUDA, OMP_NUM_THREADS=1 from rhel8/global)
  # does not come along, then only what a run uses - Slurm's client commands.
  # Apptainer is the base OS's.
  if [ -r /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh
    module purge >/dev/null 2>&1
    local mod
    for mod in ${RI_JOB_MODULES-rhel8/slurm}; do
      module load "${mod}" || { echo "job-env: module load ${mod} failed" >&2; return 1; }
    done
  fi
  export PATH="${work}/.local/bin:${PATH}"
}

# job-env.sh <settings> <command...>: empty the environment down to SLURM_*,
# then build, check and exec. The re-exec under env -i makes this true however
# the job was submitted.
_ri_job_env_main() {
  if [ -z "${_RI_JOB_ENV_CLEAN:-}" ]; then
    local keep=() name
    while IFS= read -r name; do
      case "${name}" in SLURM_*|SPANK_*) keep+=("${name}=${!name}") ;; esac
    done < <(compgen -e)
    exec /usr/bin/env -i "${keep[@]}" _RI_JOB_ENV_CLEAN=1 /bin/bash "${BASH_SOURCE[0]}" "$@"
  fi
  unset _RI_JOB_ENV_CLEAN
  export PATH=/usr/bin:/bin
  local settings="$1"
  shift
  REPO_ROOT_PHYS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  # shellcheck disable=SC1090
  . "${settings}" || { echo "job-env: cannot read ${settings}" >&2; exit 1; }
  _ri_job_env_build || exit 1
  # A run script inside the job knows not to hand itself over again (slurm.sh).
  export RI_JOB_ENV=1
  cd "${REPO_ROOT_PHYS}" || exit 1
  if ! ri_job_env_check; then
    echo "FATAL: the job's environment reaches /home or a Nix store; see above." >&2
    exit 1
  fi
  exec "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -uo pipefail
  case "${1:-}" in
    --check)
      # From a login node: the environment a job would get, with the settings
      # this shell would hand it, checked the same way.
      _tmp="$(cd "$(mktemp -d)" && pwd -P)"
      trap 'rm -rf "${_tmp}"' EXIT
      ns_write_job_settings "${_tmp}/settings" || exit 1
      /bin/bash "${BASH_SOURCE[0]}" "${_tmp}/settings" /bin/true
      ;;
    --self-check)
      set -e
      _tmp="$(mktemp -d)"
      # Physical: the check resolves what it finds, and macOS hands out
      # temporary directories under /var, a symlink to /private/var, so a
      # forbidden tree named logically would never match.
      _tmp="$(cd "${_tmp}" && pwd -P)"
      trap 'rm -rf "${_tmp}"' EXIT
      mkdir -p "${_tmp}/home/bin" "${_tmp}/ok/bin" "${_tmp}/work" "${_tmp}/run"
      printf '#!/bin/sh\necho fake\n' >"${_tmp}/home/bin/python3"
      chmod +x "${_tmp}/home/bin/python3"
      ln -s "${_tmp}/home/bin/python3" "${_tmp}/ok/bin/python3"
      export RI_JOB_FORBIDDEN="${_tmp}/home" RI_JOB_REQUIRED_TOOLS=bash RI_JOB_CHECK_PYTHON=0
      REPO_ROOT_PHYS="${_tmp}/repo"

      # A tool that only looks allowed: a symlink into the forbidden tree.
      PATH="${_tmp}/ok/bin:/usr/bin:/bin" ri_job_env_check >/dev/null 2>&1 \
        && { echo "FAIL: python3 symlinked into the forbidden tree must fail the check"; exit 1; }
      PATH="${_tmp}/home/bin:/usr/bin:/bin" ri_job_env_check >/dev/null 2>&1 \
        && { echo "FAIL: a forbidden PATH entry must fail the check"; exit 1; }
      NS_SCRATCH_DIR="${_tmp}/home/scratch" PATH=/usr/bin:/bin ri_job_env_check >/dev/null 2>&1 \
        && { echo "FAIL: a run setting naming the forbidden tree must fail the check"; exit 1; }
      _nix_bin="${_tmp}/ok/bin/nixy"
      printf '#!/bin/sh\n# /nix/store/abc-glibc/lib/ld-linux.so\n' >"${_nix_bin}"
      chmod +x "${_nix_bin}"
      ln -s "${_nix_bin}" "${_tmp}/ok/bin/gcc"
      PATH="${_tmp}/ok/bin:/usr/bin:/bin" ri_job_env_check >/dev/null 2>&1 \
        && { echo "FAIL: a binary carrying Nix store paths must fail the check"; exit 1; }
      rm "${_tmp}/ok/bin/gcc" "${_tmp}/ok/bin/python3"
      env -u NS_SCRATCH_DIR PATH=/usr/bin:/bin bash -c ". '${BASH_SOURCE[0]}'; REPO_ROOT_PHYS='${_tmp}/repo' ri_job_env_check" >/dev/null \
        || { echo "FAIL: the base OS alone must pass the check"; exit 1; }

      # Settings: only the run's knobs, directories as physical paths.
      ln -s "${_tmp}/run" "${_tmp}/run-link"
      NS_NLIVE='8 live' OUTPUT_DIR="${_tmp}/run-link" UNRELATED=1 \
        ns_write_job_settings "${_tmp}/settings"
      grep -q '^export NS_NLIVE=8\\ live$' "${_tmp}/settings" \
        || { echo "FAIL: NS_NLIVE must be carried, quoted: $(cat "${_tmp}/settings")"; exit 1; }
      grep -qx "export OUTPUT_DIR=$(printf %q "$(cd "${_tmp}/run" && pwd -P)")" "${_tmp}/settings" \
        || { echo "FAIL: OUTPUT_DIR must be carried as its physical path: $(cat "${_tmp}/settings")"; exit 1; }
      MS_PATH="${_tmp}/run-link/x.ms.tar" ns_write_job_settings "${_tmp}/ms"
      grep -qx "export MS_PATH=$(printf %q "$(cd "${_tmp}/run" && pwd -P)/x.ms.tar")" "${_tmp}/ms" \
        || { echo "FAIL: a file setting must be carried under its physical directory: $(cat "${_tmp}/ms")"; exit 1; }
      grep -Eq '^export (UNRELATED|PATH|HOME)=' "${_tmp}/settings" \
        && { echo "FAIL: only run settings may be carried: $(cat "${_tmp}/settings")"; exit 1; }

      # The whole entry point, from a dirty environment: what the command sees.
      printf 'export RI_WORK_DIR=%q RI_JOB_FORBIDDEN=%q RI_JOB_REQUIRED_TOOLS=bash RI_JOB_CHECK_PYTHON=0 RI_JOB_MODULES=\n' \
        "${_tmp}/work" "${_tmp}/home" >>"${_tmp}/settings"
      _seen="$(env LEAK=1 PATH="${_tmp}/home/bin:${PATH}" SLURM_JOB_ID=7 \
        /bin/bash "${BASH_SOURCE[0]}" "${_tmp}/settings" /usr/bin/env)"
      for _want in "HOME=${_tmp}/work/.ri-job/home" "APPTAINER_HOME=${_tmp}/work/.ri-job/home" \
                   "UV_CACHE_DIR=${_tmp}/work/.cache/uv" "NS_NLIVE=8 live" "SLURM_JOB_ID=7"; do
        grep -qxF "${_want}" <<<"${_seen}" || { echo "FAIL: the job must see ${_want}, saw: ${_seen}"; exit 1; }
      done
      grep -q "^PATH=${_tmp}/work/.local/bin:" <<<"${_seen}" \
        || { echo "FAIL: the job's PATH must start at hpc-work's tools, saw: $(grep ^PATH= <<<"${_seen}")"; exit 1; }
      grep -q "^PATH=.*${_tmp}/home" <<<"${_seen}" \
        && { echo "FAIL: the submitting PATH leaked into the job"; exit 1; }
      grep -q '^LEAK=' <<<"${_seen}" && { echo "FAIL: the submitting environment leaked into the job"; exit 1; }
      # ...and a job whose settings point it at the forbidden tree never starts.
      echo "export NS_SCRATCH_DIR=${_tmp}/home/scratch" >>"${_tmp}/settings"
      /bin/bash "${BASH_SOURCE[0]}" "${_tmp}/settings" /bin/true >/dev/null 2>&1 \
        && { echo "FAIL: a job reaching the forbidden tree must refuse to start"; exit 1; }

      echo "job-env self-check passed"
      ;;
    "")
      echo "usage: job-env.sh <settings file> <command...> | --check | --self-check" >&2
      exit 2
      ;;
    *)
      _ri_job_env_main "$@"
      ;;
  esac
fi
