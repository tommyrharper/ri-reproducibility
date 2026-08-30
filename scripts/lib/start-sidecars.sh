# shellcheck shell=bash  # sourced, so no shebang
# Start sidecars for `docker exec`; export NS_SIDECARS. Requires REPO_ROOT.
# API: start_sidecars, sidecar_launch, sidecar_wait; `--` replaces sleep.
SIDECAR_NAMES=()
_SIDECAR_PIDS=()
_SIDECAR_COMMANDS=()
_SIDECAR_JSON=""

# Bind each container to one host /dev/shm directory for its in-flight
# Measurement Set. It avoids copying between private container tmpfses and is
# disabled when host /dev/shm is unavailable; readers then use the old path.
if [ -z "${NS_SCRATCH_DIR:-}" ] && [ -w /dev/shm ]; then
  NS_SCRATCH_DIR="/dev/shm/ri-ns-scratch-$$"
  mkdir -p "${NS_SCRATCH_DIR}"
fi
export NS_SCRATCH_DIR="${NS_SCRATCH_DIR:-}"

# Backgrounded: `docker rm --force` of the three containers costs ~0.4s, and
# waiting for it is 8% of the run's wall clock spent after every result is
# already on disk. The orphaned `docker rm` outlives this shell and finishes.
_sidecar_remove() {
  if [ "${#SIDECAR_NAMES[@]}" -gt 0 ]; then
    docker rm --force "${SIDECAR_NAMES[@]}" >/dev/null 2>&1 &
  fi
  if [ -n "${NS_SCRATCH_DIR:-}" ]; then
    # Containers may leave root-owned evaluation directories. Best-effort
    # cleanup must not replace a successful search's exit status.
    rm -rf "${NS_SCRATCH_DIR}" 2>/dev/null || true
  fi
}

sidecar_launch() {
  local platform="$1" image="$2" name
  local -a args=() command=(sleep infinity)
  shift 2
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then
      shift
      command=("$@")
      break
    fi
    args+=("$1")
    shift
  done
  name="ri-ns-sidecar-$$-${#SIDECAR_NAMES[@]}"
  SIDECAR_NAME="${name}"
  # Which run owns this container, for the leak rule in rank-budget.sh. The pid
  # in the name is this shell's, and this shell can die while the run it
  # started carries on - the ranks are children of containerd-shim, not of it -
  # so the pid alone called a live run's containers abandoned and offered them
  # up to `docker rm --force`. OUTPUT_DIR is read from the caller's scope
  # because both run scripts have it by the time they source this file, the
  # same way REPO_ROOT is read below; a caller without one (the self-check)
  # simply gets no label and falls back to the pid rule.
  local -a label=()
  if [ -n "${OUTPUT_DIR:-}" ]; then
    label=(--label "ri.run-dir=${OUTPUT_DIR}")
  fi
  # macOS ships bash 3.2, where expanding an empty array under `set -u` is an
  # "unbound variable" error rather than nothing - so a sidecar with no extra
  # docker arguments, which is most of them, could not start there at all. The
  # `${a[@]+"${a[@]}"}` form expands to nothing when empty on both. The slice
  # needs its own array because the guard cannot wrap a slice.
  local -a entrypoint_args=()
  if [ "${#command[@]}" -gt 1 ]; then
    entrypoint_args=("${command[@]:1}")
  fi
  local -a scratch_args=()
  if [ -n "${NS_SCRATCH_DIR:-}" ]; then
    # The path is also in the environment, not just on the argv the rank sends:
    # the simulator assembles its Measurement Set in the destination directory
    # when that is already this tmpfs, so that its closing move is a rename
    # rather than a copy off the container's own /dev/shm - see
    # scratch_root_for() in simulate_point_source_ms.py.
    scratch_args=(-v "${NS_SCRATCH_DIR}:${NS_SCRATCH_DIR}" -e "NS_SCRATCH_DIR=${NS_SCRATCH_DIR}")
  fi
  local -a run=(
    docker run --detach --rm --name "${name}"
    ${label[@]+"${label[@]}"}
    --network none
    --shm-size 512m
    --platform "${platform}"
    -v "${REPO_ROOT}:${REPO_ROOT}"
    ${scratch_args[@]+"${scratch_args[@]}"}
    ${args[@]+"${args[@]}"}
    --entrypoint "${command[0]}" "${image}" ${entrypoint_args[@]+"${entrypoint_args[@]}"}
  )
  # Kept so `sidecar_restore` can start this exact container again. Quoted with
  # %q into one string because bash has no array of arrays; nothing else reads
  # it, so the eval that replays it only ever sees this shell's own quoting.
  _SIDECAR_COMMANDS+=("$(printf '%q ' "${run[@]}")")
  "${run[@]}" >/dev/null &
  _SIDECAR_PIDS+=("$!")
  SIDECAR_NAMES+=("${SIDECAR_NAME}")
  _SIDECAR_JSON="${_SIDECAR_JSON}${_SIDECAR_JSON:+,}\"${image}\":\"${SIDECAR_NAME}\""
  # Exported here, not in sidecar_wait: the name is known as soon as the launch
  # is issued, so a caller can build the command that consumes it while the
  # containers are still coming up.
  export NS_SIDECARS="{${_SIDECAR_JSON}}"
  # INT and TERM as well as EXIT: bash does not run an EXIT trap when it dies
  # on an uncaught signal, so a Ctrl-C or a `timeout` left the sidecars
  # running with no parent. An abandoned R2D2 sidecar holds its whole warm
  # worker pool - measured at 33.7GB - which then counts against every later
  # run's memory budget (see scripts/lib/rank-budget.sh) until someone
  # notices. Cleaning up twice is harmless; the second `docker rm` just fails.
  trap '_sidecar_remove' EXIT
  trap '_sidecar_remove; exit 130' INT
  trap '_sidecar_remove; exit 143' TERM
}

# Restore gone containers before retry; workers recreate over bind-mounted FIFOs.
sidecar_restore() {
  local i restarted=0
  for i in "${!SIDECAR_NAMES[@]}"; do
    if [ "$(docker inspect --format '{{.State.Running}}' "${SIDECAR_NAMES[$i]}" 2>/dev/null)" = "true" ]; then
      continue
    fi
    echo "sidecar_restore: ${SIDECAR_NAMES[$i]} is gone, starting it again" >&2
    # `--rm` leaves nothing behind on a clean death, but a container the daemon
    # still has a record of (created, exited, being removed) keeps the name.
    docker rm --force "${SIDECAR_NAMES[$i]}" >/dev/null 2>&1 || true
    eval "${_SIDECAR_COMMANDS[$i]}" >/dev/null &
    _SIDECAR_PIDS+=("$!")
    restarted=1
  done
  if [ "${restarted}" = 1 ]; then
    sidecar_wait
  fi
}

sidecar_wait() {
  local pid
  for pid in ${_SIDECAR_PIDS[@]+"${_SIDECAR_PIDS[@]}"}; do
    wait "${pid}"
  done
  _SIDECAR_PIDS=()
}

start_sidecars() {
  local platform="$1" image
  shift
  for image in "$@"; do
    sidecar_launch "${platform}" "${image}"
  done
  sidecar_wait
}

# `bash scripts/lib/start-sidecars.sh --self-check` - checks isolated container
# arguments and NS_SIDECARS mapping with a stub `docker`.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  REPO_ROOT="${REPO_ROOT:-$(pwd)}"
  _log="$(mktemp)"
  # SC2329 is 0.11.0's spelling of the same finding SC2317 raised through
  # 0.10.0; both are listed so the gate stays green across the spread apt
  # gives across runner images.
  # shellcheck disable=SC2317,SC2329  # called indirectly, by sidecar_launch below
  docker() { printf '%s\n' "$*" >>"${_log}"; }
  sidecar_launch linux/amd64 img:a -v /sock:/sock
  sidecar_launch linux/amd64 img:b
  sidecar_launch linux/amd64 img:c -- sh -c 'echo hi' sh /some/dir
  sidecar_wait
  grep -q -- '-v /sock:/sock --entrypoint sleep img:a infinity' "${_log}"
  grep -q -- '--entrypoint sleep img:b infinity' "${_log}"
  grep -q -- 'img:b.*/sock' "${_log}" && {
    echo "FAIL: img:a's -v flag leaked into img:b"; exit 1
  }
  grep -q -- "--entrypoint sh img:c -c echo hi sh /some/dir" "${_log}"
  # Every container gets the MS scratch tmpfs or the imager cannot open it.
  if [ -n "${NS_SCRATCH_DIR}" ]; then
    [ "$(grep -c -- "-v ${NS_SCRATCH_DIR}:${NS_SCRATCH_DIR}" "${_log}")" = 3 ] \
      || { echo "FAIL: the scratch mount did not reach all three containers"; exit 1; }
    # The simulator reads its destination from this environment variable.
    [ "$(grep -c -- "-e NS_SCRATCH_DIR=${NS_SCRATCH_DIR}" "${_log}")" = 3 ] \
      || { echo "FAIL: NS_SCRATCH_DIR did not reach all three containers"; exit 1; }
  fi
  [ "${NS_SIDECARS}" = '{"img:a":"ri-ns-sidecar-'"$$"'-0","img:b":"ri-ns-sidecar-'"$$"'-1","img:c":"ri-ns-sidecar-'"$$"'-2"}' ]
  # Without OUTPUT_DIR, the pid rule in rank-budget.sh applies.
  grep -q -- 'ri.run-dir' "${_log}" && { echo "FAIL: labelled with no OUTPUT_DIR"; exit 1; }
  # With one, the label stops rank-budget.sh reaping a live run's containers.
  : >"${_log}"
  OUTPUT_DIR=/some/run sidecar_launch linux/amd64 img:d
  sidecar_wait
  grep -q -- '--label ri.run-dir=/some/run' "${_log}"

  # Restore keeps retries from execing into a dead container; live ones stay.
  SIDECAR_NAMES=()
  _SIDECAR_PIDS=()
  _SIDECAR_COMMANDS=()
  _SIDECAR_JSON=""
  _gone=""
  docker() {
    printf '%s\n' "$*" >>"${_log}"
    if [ "$1" = inspect ]; then
      case " $* " in *" ${_gone} "*) return 1 ;; esac
      echo true
    fi
  }
  OUTPUT_DIR=/some/run sidecar_launch linux/amd64 img:e -v /sock:/sock
  OUTPUT_DIR=/some/run sidecar_launch linux/amd64 img:f
  sidecar_wait
  : >"${_log}"
  sidecar_restore
  grep -q -- 'run --detach' "${_log}" && {
    echo "FAIL: restarted a sidecar that was still running"; exit 1
  }
  _gone="${SIDECAR_NAMES[1]}"
  : >"${_log}"
  sidecar_restore 2>/dev/null
  grep -q -- "run --detach --rm --name ${SIDECAR_NAMES[1]} .*--entrypoint sleep img:f" "${_log}" \
    || { echo "FAIL: a sidecar that is gone was not started again"; exit 1; }
  grep -q -- 'img:e' "${_log}" && {
    echo "FAIL: restarted the sidecar that was still running"; exit 1
  }
  rm -f "${_log}"
  echo "start-sidecars self-check passed"
fi
