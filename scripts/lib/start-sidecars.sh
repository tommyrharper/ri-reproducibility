# Start the long-lived containers a run `docker exec`s into, and export
# NS_SIDECARS.
#
# The ranks only ever `docker exec`, and separate `docker exec` processes are
# already isolated, so one container per image serves the whole run. Letting
# each rank start its own on first use meant 16 concurrent `docker run`s on the
# default 8 ranks, measured at 1.3s against 0.36s for a single one, all of it in
# front of the first evaluation.
#
# `--network none`: no sidecar needs networking, and docker's default bridge
# setup costs ~0.2s per container under rootless Docker; "none" still gives the
# loopback interface meqserver and MPI want. `--shm-size 512m`: the simulate
# builds its working MS and its cached makems skeletons in /dev/shm, and
# docker's 64MB default is only ~3x the largest cache this parameter space
# fills.
#
# Source this, then either
#
#   start_sidecars <platform> <image>...              # start these, then wait
#
# or, when a container needs extra `docker run` arguments or the caller has
# work to overlap with the startup,
#
#   sidecar_launch <platform> <image> [docker args...] [-- warm-up command...]
#   ...other setup...
#   sidecar_wait                                         # before the first exec
#
# `sidecar_launch` returns immediately and leaves the container's name in
# SIDECAR_NAME. Requires REPO_ROOT. Every container started is removed by an
# EXIT trap.
#
# A warm-up command after `--` is run once in the new container, still inside
# the background job, before `sidecar_wait` returns. The first process in a
# fresh container faults the whole dynamic-linker path of whatever it runs
# through the image's overlay mount: for the PolyChord container an 8-rank
# `mpirun python3` costs 0.85s cold against 0.19s once a single throwaway
# `python3 -c "import numpy, pypolychord"` has paid that, and doing it here
# hides it behind the other containers coming up.
SIDECAR_NAMES=()
_SIDECAR_PIDS=()
_SIDECAR_JSON=""

sidecar_launch() {
  local platform="$1" image="$2" name
  shift 2
  local docker_args=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    docker_args+=("$1")
    shift
  done
  if [ $# -gt 0 ]; then
    shift  # the "--" itself; anything left is the warm-up command
  fi
  name="ri-ns-sidecar-$$-${#SIDECAR_NAMES[@]}"
  SIDECAR_NAME="${name}"
  {
    docker run --detach --rm --name "${name}" \
      --network none \
      --shm-size 512m \
      --platform "${platform}" \
      -v "${REPO_ROOT}:${REPO_ROOT}" \
      "${docker_args[@]}" \
      --entrypoint sleep "${image}" infinity >/dev/null
    if [ $# -gt 0 ]; then
      docker exec "${name}" "$@" >/dev/null 2>&1
    fi
  } &
  _SIDECAR_PIDS+=("$!")
  SIDECAR_NAMES+=("${SIDECAR_NAME}")
  _SIDECAR_JSON="${_SIDECAR_JSON}${_SIDECAR_JSON:+,}\"${image}\":\"${SIDECAR_NAME}\""
  # Exported here, not in sidecar_wait: the name is known as soon as the launch
  # is issued, so a caller can build the command that consumes it while the
  # containers are still coming up.
  export NS_SIDECARS="{${_SIDECAR_JSON}}"
  # Backgrounded: `docker rm --force` of the three containers costs ~0.4s, and
  # waiting for it is 8% of the run's wall clock spent after every result is
  # already on disk. The orphaned `docker rm` outlives this shell and finishes.
  trap 'docker rm --force "${SIDECAR_NAMES[@]}" >/dev/null 2>&1 &' EXIT
}

sidecar_wait() {
  local pid
  for pid in "${_SIDECAR_PIDS[@]}"; do
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

# `bash scripts/lib/start-sidecars.sh --self-check` - guards the `--` split, the
# one piece of parsing here: extra `docker run` arguments must not leak into the
# warm-up command or vice versa. Stubs `docker` so nothing is actually started.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  REPO_ROOT="${REPO_ROOT:-$(pwd)}"
  _log="$(mktemp)"
  docker() { printf '%s\n' "$*" >>"${_log}"; }
  sidecar_launch linux/amd64 img:a -v /sock:/sock -- python3 -c "import numpy"
  sidecar_launch linux/amd64 img:b
  sidecar_wait
  grep -q -- '-v /sock:/sock --entrypoint sleep img:a infinity' "${_log}"
  grep -q -- 'exec ri-ns-sidecar-'"$$"'-0 python3 -c import numpy' "${_log}"
  grep -q -- '--entrypoint sleep img:b infinity' "${_log}"
  ! grep -q -- 'img:b.*python3' "${_log}"
  [ "${NS_SIDECARS}" = '{"img:a":"ri-ns-sidecar-'"$$"'-0","img:b":"ri-ns-sidecar-'"$$"'-1"}' ]
  rm -f "${_log}"
  echo "start-sidecars self-check passed"
fi
