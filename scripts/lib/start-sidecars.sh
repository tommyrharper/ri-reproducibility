# Start one long-lived sidecar container per image and export NS_SIDECARS.
#
# The ranks only ever `docker exec` into these, and separate `docker exec`
# processes are already isolated, so one container per image serves the whole
# run. Letting each rank start its own on first use meant 16 concurrent
# `docker run`s on the default 8 ranks, measured at 1.3s against 0.36s for a
# single one, all of it in front of the first evaluation.
#
# `--network none`: no sidecar needs networking, and docker's default bridge
# setup costs ~0.2s per container under rootless Docker; "none" still gives the
# loopback interface meqserver wants. `--shm-size 512m`: the simulate builds its
# working MS and its cached makems skeletons in /dev/shm, and docker's 64MB
# default is only ~3x the largest cache this parameter space fills.
#
# Source this, then call `start_sidecars <platform> <image>...` before the
# PolyChord container starts. Requires REPO_ROOT. Registers an EXIT trap that
# removes the containers.
start_sidecars() {
  local platform="$1"
  shift
  local image name pid json="" pids=() index=0
  SIDECAR_NAMES=()
  for image in "$@"; do
    name="ri-ns-sidecar-$$-${index}"
    docker run --detach --rm --name "${name}" \
      --network none \
      --shm-size 512m \
      --platform "${platform}" \
      -v "${REPO_ROOT}:${REPO_ROOT}" \
      --entrypoint sleep "${image}" infinity >/dev/null &
    pids+=("$!")
    SIDECAR_NAMES+=("${name}")
    json="${json}${json:+,}\"${image}\":\"${name}\""
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  export NS_SIDECARS="{${json}}"
  trap 'docker rm --force "${SIDECAR_NAMES[@]}" >/dev/null 2>&1 || true' EXIT
}
