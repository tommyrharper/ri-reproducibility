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
# All but the last of these start no search. The self-heal check does - two ~45
# second, ~0.6GB WSClean searches, one killed and one hung by freezing a rank,
# both watched until they recover - because the thing it checks only exists in
# a real run. It is safe alongside another run:
# 3 ranks at ~200MB, sized and refused by scripts/lib/rank-budget.sh like any
# other, on a throwaway directory the report and `./ri runs` never see.
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

# Through uv, like every other host-side entry point in `ri`, and pinned the
# way scripts/lib/defaults.sh pins it: test_self_checks.py imports common.py,
# which needs tomllib to read defaults.toml. A host whose `python3` predates
# 3.11 - Ubuntu 22.04's 3.10 is one - took the whole of `./ri self-check` down
# on that import, self-heal check included, so the only check that starts a
# real search was unreachable through its own documented front door.
# --no-project because these two need nothing from pyproject.toml.
host_python() { uv run --no-project --python ">=3.11" python "$@"; }

echo "=== host-side checks ==="
host_python "${REPO_ROOT}/scripts/test_watchdogs.py"
host_python "${REPO_ROOT}/scripts/test_self_checks.py"

if [[ "${TARGET}" == "all" || "${TARGET}" == "simulate" ]]; then
  echo
  echo "=== simulate (${MEQTREES_IMAGE}) ==="
  docker_run --entrypoint python3 "${MEQTREES_IMAGE}" -u "$(nested_sampling simulate_point_source_ms.py)" --self-check
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "r2d2-serve" ]]; then
  echo
  echo "=== r2d2 imaging worker (${R2D2_IMAGE}) ==="
  # The R2D2 sidecar runs this file off the bind mount, so what runs here is
  # what a run would run. Its numpy and measurement-operator checks skip
  # themselves outside this image, which is what they do in CI.
  docker_run --entrypoint python3 "${R2D2_IMAGE}" -u "$(nested_sampling r2d2_serve.py)" --self-check
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

if [[ "${TARGET}" == "all" || "${TARGET}" == "self-heal" ]]; then
  echo
  echo "=== self-healing (real searches, killed and hung) ==="
  # Host-side, unlike the checks above: what is under test - run_with_retries in
  # scripts/lib/progress-bar.sh and `./ri health` - runs on the host from the
  # working tree. The search it drives executes the copy baked into the images.
  bash "${REPO_ROOT}/scripts/test_self_heal.sh"
fi

echo
echo "OK: self-checks passed"
