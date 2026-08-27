#!/usr/bin/env bash
# Run the self-checks that need one of this repo's images to mean anything.
#
# The host-only checks run in CI on every change. These cannot: they need a live
# meqserver, a real TDL compile, numpy and casacore. Before this existed nothing
# ran them at all - they were written, wired into each module's own --self-check
# entry point, and then only ever invoked by hand, which is the same as not
# having them.
#
# The repo is bind-mounted and the working tree's copy is what runs, so these
# check the code you are about to ship using the image's interpreter and
# libraries - no rebuild needed to check a change. A run executes the copy baked
# into the image instead, so rebuild before running one.
#
# Nothing here starts a search or writes into results/, so it is safe to run
# while someone else's run is in progress.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

TARGET="${1:-all}"

# The repo is mounted at its own host path, the way the sidecars mount it, so
# paths inside a check mean the same thing they would during a run.
docker_run() {
  docker run --rm \
    --shm-size 512m \
    --platform "${DOCKER_DEFAULT_PLATFORM}" \
    -v "${REPO_ROOT}:${REPO_ROOT}" \
    -w "${REPO_ROOT}" \
    "$@"
}

nested_sampling() { echo "${REPO_ROOT}/scripts/lib/nested_sampling/$1"; }

echo "=== host-side checks ==="
python3 "${REPO_ROOT}/scripts/test_watchdogs.py"

if [[ "${TARGET}" == "all" || "${TARGET}" == "simulate" ]]; then
  echo
  echo "=== simulate (${MEQTREES_IMAGE}) ==="
  docker_run --entrypoint python3 "${MEQTREES_IMAGE}" -u "$(nested_sampling simulate_point_source_ms.py)" --self-check
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "wsclean" ]]; then
  echo
  echo "=== wsclean sampler (${POLYCHORD_IMAGE}) ==="
  docker_run -e POLYCHORD_WSCLEAN_SELF_CHECK=1 --entrypoint python3 \
    "${POLYCHORD_IMAGE}" -u "$(nested_sampling polychord_wsclean.py)"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "r2d2" ]]; then
  echo
  echo "=== r2d2 sampler (${POLYCHORD_IMAGE}) ==="
  docker_run -e POLYCHORD_R2D2_SELF_CHECK=1 --entrypoint python3 \
    "${POLYCHORD_IMAGE}" -u "$(nested_sampling polychord_r2d2.py)"
fi

echo
echo "OK: self-checks passed"
