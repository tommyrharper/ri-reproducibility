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
#   sidecar_launch <platform> <image> [docker args...] [-- command...]
#   ...other setup...
#   sidecar_wait                                         # before the first exec
#
# Everything after `--` replaces the container's default `sleep infinity`. That
# is how the meqtrees sidecar starts its simulate workers: a `docker exec` can
# only be issued once `docker run` has returned, and the container's own command
# starts ~0.1s earlier than that, on work whose whole value is head start.
#
# `sidecar_launch` returns immediately and leaves the container's name in
# SIDECAR_NAME. Requires REPO_ROOT. Every container started is removed by an
# EXIT trap.
#
# There is deliberately no warm-up hook here. A fresh container's first Python
# process used to cost ~0.8s more than the next one, which this file paid for
# with a throwaway `docker exec` before `sidecar_wait` returned; that cost was
# the interpreter byte-compiling modules the image shipped without a valid
# .pyc, and the images now compile them at build time instead.
SIDECAR_NAMES=()
_SIDECAR_PIDS=()
_SIDECAR_JSON=""

# Backgrounded: `docker rm --force` of the three containers costs ~0.4s, and
# waiting for it is 8% of the run's wall clock spent after every result is
# already on disk. The orphaned `docker rm` outlives this shell and finishes.
_sidecar_remove() {
  if [ "${#SIDECAR_NAMES[@]}" -gt 0 ]; then
    docker rm --force "${SIDECAR_NAMES[@]}" >/dev/null 2>&1 &
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
  # macOS ships bash 3.2, where expanding an empty array under `set -u` is an
  # "unbound variable" error rather than nothing - so a sidecar with no extra
  # docker arguments, which is most of them, could not start there at all. The
  # `${a[@]+"${a[@]}"}` form expands to nothing when empty on both. The slice
  # needs its own array because the guard cannot wrap a slice.
  local -a entrypoint_args=()
  if [ "${#command[@]}" -gt 1 ]; then
    entrypoint_args=("${command[@]:1}")
  fi
  docker run --detach --rm --name "${name}" \
    --network none \
    --shm-size 512m \
    --platform "${platform}" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    ${args[@]+"${args[@]}"} \
    --entrypoint "${command[0]}" "${image}" ${entrypoint_args[@]+"${entrypoint_args[@]}"} >/dev/null &
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

# `bash scripts/lib/start-sidecars.sh --self-check` - guards the two things a
# caller depends on: per-container `docker run` arguments must reach that
# container's command line and no other's, and NS_SIDECARS must map every image
# to its container. Stubs `docker` so nothing is actually started.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = "--self-check" ]; then
  set -euo pipefail
  REPO_ROOT="${REPO_ROOT:-$(pwd)}"
  _log="$(mktemp)"
  docker() { printf '%s\n' "$*" >>"${_log}"; }
  sidecar_launch linux/amd64 img:a -v /sock:/sock
  sidecar_launch linux/amd64 img:b
  sidecar_launch linux/amd64 img:c -- sh -c 'echo hi' sh /some/dir
  sidecar_wait
  grep -q -- '-v /sock:/sock --entrypoint sleep img:a infinity' "${_log}"
  grep -q -- '--entrypoint sleep img:b infinity' "${_log}"
  ! grep -q -- 'img:b.*/sock' "${_log}"
  grep -q -- "--entrypoint sh img:c -c echo hi sh /some/dir" "${_log}"
  [ "${NS_SIDECARS}" = '{"img:a":"ri-ns-sidecar-'"$$"'-0","img:b":"ri-ns-sidecar-'"$$"'-1","img:c":"ri-ns-sidecar-'"$$"'-2"}' ]
  rm -f "${_log}"
  echo "start-sidecars self-check passed"
fi
