# shellcheck shell=bash  # sourced, so no shebang
# Start the worker pools as host processes under Apptainer. Requires REPO_ROOT
# and APPTAINER (scripts/lib/defaults.sh); reads OUTPUT_DIR for the logs.
# API: sidecar_launch <sif> [apptainer exec flags] -- command...; sidecar_restore.
#
# The Docker branch started each pool as a detached container and the ranks
# `docker exec`ed into it. Here a pool is `apptainer exec <sif> ...` in the
# background, in its own process group (setsid), so the whole pool - the
# container's shell, its workers, their meqservers - is one `kill -- -pgid`.
SIDECAR_PIDS=()
_SIDECAR_COMMANDS=()
_SIDECAR_LOGS=()

# One host /dev/shm directory for every in-flight Measurement Set; bound into
# every pool and the ranks. Disabled when host /dev/shm is unavailable.
if [ -z "${NS_SCRATCH_DIR:-}" ] && [ -w /dev/shm ]; then
  NS_SCRATCH_DIR="/dev/shm/ri-ns-scratch-$$"
  mkdir -p "${NS_SCRATCH_DIR}"
fi
export NS_SCRATCH_DIR="${NS_SCRATCH_DIR:-}"

_sidecar_remove() {
  local pid
  for pid in ${SIDECAR_PIDS[@]+"${SIDECAR_PIDS[@]}"}; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  if [ -n "${NS_SCRATCH_DIR:-}" ]; then
    # Best-effort cleanup must not replace a successful search's exit status.
    rm -rf "${NS_SCRATCH_DIR}" 2>/dev/null || true
  fi
}

# The bind mounts every pool and every rank gets: the repo at its own path
# (the FIFOs, the run directory and r2d2_serve.py live there) and the MS
# scratch tmpfs, which the simulator also reads from its environment so that
# its closing move is a rename rather than a copy - see scratch_root_for() in
# simulate_point_source_ms.py.
sidecar_binds() {
  printf '%s\n' --bind "${REPO_ROOT}"
  if [ -n "${NS_SCRATCH_DIR:-}" ]; then
    printf '%s\n' --bind "${NS_SCRATCH_DIR}" --env "NS_SCRATCH_DIR=${NS_SCRATCH_DIR}"
  fi
}

_sidecar_start() {
  # `$!` is the process group only because a non-interactive bash puts a
  # background job in its own group and setsid then execs rather than forks;
  # the self-check below asserts it.
  setsid "$@" </dev/null &
  SIDECAR_PIDS+=("$!")
  # Not a job this shell reports on or waits for; the pools outlive every
  # command it runs and are killed as groups.
  disown "$!"
}

sidecar_launch() {
  local sif="$1" name
  local -a args=() command=() binds=()
  shift
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then
      shift
      command=("$@")
      break
    fi
    args+=("$1")
    shift
  done
  [ "${#command[@]}" -gt 0 ] || { echo "sidecar_launch: no command after --" >&2; return 1; }
  name="$(basename "${sif}" .sif)"
  # Which shell owns these pools, for the leak rule in rank-budget.sh: a pool
  # whose run has no ranks is only leaked once this pid is gone too - between
  # retries the ranks are dead and the pools are not.
  if [ -n "${OUTPUT_DIR:-}" ]; then
    echo "$$" >"${OUTPUT_DIR}/.launcher.pid"
  fi
  mapfile -t binds < <(sidecar_binds)
  local -a run=(
    "${APPTAINER}" exec --pwd "${REPO_ROOT}"
    "${binds[@]}"
    ${args[@]+"${args[@]}"}
    "${sif}" "${command[@]}"
  )
  # Kept so `sidecar_restore` can start this exact pool again. Quoted with %q
  # into one string because bash has no array of arrays; nothing else reads it.
  _SIDECAR_COMMANDS+=("$(printf '%q ' "${run[@]}")")
  # The workers' own stderr - a meqserver crash, a torch import error - is the
  # only record of why a pool died, and it lands beside the run.
  _SIDECAR_LOGS+=("${OUTPUT_DIR:-${TMPDIR:-/tmp}}/workers-${name}.log")
  _sidecar_start "${run[@]}" >>"${_SIDECAR_LOGS[-1]}" 2>&1
  # INT and TERM as well as EXIT: bash does not run an EXIT trap when it dies
  # on an uncaught signal, so a Ctrl-C or a `timeout` would leave the pools
  # running with no parent - an R2D2 pool holds ~3.4GB per rank of warm
  # worker that nothing else would ever free.
  trap '_sidecar_remove' EXIT
  trap '_sidecar_remove; exit 130' INT
  trap '_sidecar_remove; exit 143' TERM
}

# Start any pool that died before a retry; the workers reopen the same FIFOs.
sidecar_restore() {
  local i
  for i in "${!SIDECAR_PIDS[@]}"; do
    kill -0 "${SIDECAR_PIDS[$i]}" 2>/dev/null && continue
    echo "sidecar_restore: pool ${SIDECAR_PIDS[$i]} is gone, starting it again" >&2
    local -a run=()
    eval "run=(${_SIDECAR_COMMANDS[$i]})"
    _sidecar_start "${run[@]}" >>"${_SIDECAR_LOGS[$i]}" 2>&1
    SIDECAR_PIDS[i]="${SIDECAR_PIDS[-1]}"
    unset 'SIDECAR_PIDS[-1]'
  done
}

# `bash scripts/lib/start-sidecars.sh --self-check` - the exec line each pool
# gets, that pools are process groups, and that restore only touches dead ones.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  REPO_ROOT="${REPO_ROOT:-$(pwd)}"
  _tmp="$(mktemp -d)"
  OUTPUT_DIR="${_tmp}"
  # A stand-in apptainer that records its argv and then lives like a pool.
  APPTAINER="${_tmp}/apptainer"
  printf '#!/bin/sh\nprintf "%%s\\n" "$*" >>"%s/log"\nexec sleep 30\n' "${_tmp}" >"${APPTAINER}"
  chmod +x "${APPTAINER}"
  sidecar_launch /img/a.sif --bind /ckpt:/checkpoints:ro -- sh -c 'echo hi' sh /some/dir
  sidecar_launch /img/b.sif -- python3 serve.py
  sleep 0.5
  grep -q -- "exec --pwd ${REPO_ROOT} --bind ${REPO_ROOT} .*--bind /ckpt:/checkpoints:ro /img/a.sif sh -c echo hi sh /some/dir" "${_tmp}/log"
  grep -q -- "--bind ${REPO_ROOT} .*/img/b.sif python3 serve.py" "${_tmp}/log"
  grep -q -- '/img/b.sif.*/ckpt' "${_tmp}/log" && { echo "FAIL: a.sif's bind leaked into b.sif"; exit 1; }
  if [ -n "${NS_SCRATCH_DIR}" ]; then
    [ "$(grep -c -- "--bind ${NS_SCRATCH_DIR} --env NS_SCRATCH_DIR=${NS_SCRATCH_DIR}" "${_tmp}/log")" = 2 ] \
      || { echo "FAIL: the scratch mount did not reach both pools"; exit 1; }
  fi
  [ "${#SIDECAR_PIDS[@]}" = 2 ]
  for _pid in "${SIDECAR_PIDS[@]}"; do
    [ "$(ps -o pgid= -p "${_pid}" | tr -d ' ')" = "${_pid}" ] \
      || { echo "FAIL: pool ${_pid} is not its own process group"; exit 1; }
  done
  [ -f "${_tmp}/workers-a.log" ] && [ -f "${_tmp}/workers-b.log" ] \
    || { echo "FAIL: pool logs not written beside the run"; exit 1; }
  [ "$(cat "${_tmp}/.launcher.pid")" = "$$" ] || { echo "FAIL: launcher pid not recorded"; exit 1; }

  # Restore starts a dead pool again with its own command, and leaves the
  # live one alone.
  : >"${_tmp}/log"
  sidecar_restore
  [ ! -s "${_tmp}/log" ] || { echo "FAIL: restarted a pool that was still running"; exit 1; }
  _live="${SIDECAR_PIDS[0]}"
  kill -KILL -- "-${SIDECAR_PIDS[1]}"
  sleep 0.2
  sidecar_restore 2>/dev/null
  sleep 0.5
  grep -q -- "/img/b.sif python3 serve.py" "${_tmp}/log" || { echo "FAIL: the dead pool was not started again"; exit 1; }
  grep -q -- "/img/a.sif" "${_tmp}/log" && { echo "FAIL: restarted the pool that was still running"; exit 1; }
  [ "${SIDECAR_PIDS[0]}" = "${_live}" ] && [ "${#SIDECAR_PIDS[@]}" = 2 ] \
    || { echo "FAIL: pids after restore: ${SIDECAR_PIDS[*]}"; exit 1; }
  kill -0 "${SIDECAR_PIDS[1]}" || { echo "FAIL: the restored pool is not running"; exit 1; }

  # Teardown kills whole groups, children included.
  _sidecar_remove
  sleep 0.2
  for _pid in "${SIDECAR_PIDS[@]}"; do
    kill -0 "${_pid}" 2>/dev/null && { echo "FAIL: pool ${_pid} survived _sidecar_remove"; exit 1; }
  done
  trap - EXIT INT TERM
  rm -rf "${_tmp}"
  echo "start-sidecars self-check passed"
fi
