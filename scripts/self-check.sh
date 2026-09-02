#!/usr/bin/env bash
# Run image checks; the working tree is mounted, but searches use baked copies
# and need a rebuild. Self-heal uses throwaway searches with rank/memory limits.
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

# Use uv's pinned Python >=3.11: host `python3` may lack tomllib, which
# `test_self_checks.py` needs through common.py. `--no-project` keeps checks
# independent of pyproject.toml, and `--isolated` keeps them independent of
# `.venv` as well - without it, `uv run` picks up whatever a `./ri` invocation
# left there, so a HOST_RUNNABLE self-check that reaches for numpy passes here
# and fails in CI, which has no `.venv` to find. These four are stdlib-only by
# contract; that is the contract this enforces.
host_python() { uv run --no-project --isolated --python ">=3.11" python "$@"; }

echo "=== host-side checks ==="
host_python "${REPO_ROOT}/scripts/profile-nested-sampling-run.py" --self-check
host_python "${REPO_ROOT}/scripts/lib/report_server.py" --self-check
# Only the run picking; drawing the figures needs the r2d2 image.
host_python "${REPO_ROOT}/scripts/plot-merged-likelihood-compare.py" --self-check
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
  docker_run --entrypoint python3 "${R2D2_IMAGE}" -u "$(nested_sampling r2d2_serve.py)" --self-check
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "zygote" ]]; then
  echo
  echo "=== wsclean fork server (${WSCLEAN_IMAGE}) ==="
  docker_run --entrypoint python3 "${WSCLEAN_IMAGE}" -u "${REPO_ROOT}/scripts/test_zygote.py"
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

if [[ "${TARGET}" == "all" || "${TARGET}" == "report" ]]; then
  echo
  echo "=== HTML report (${R2D2_IMAGE}) ==="
  docker_run -e GENERATE_REPORT_SELF_CHECK=1 --entrypoint python3 \
    "${R2D2_IMAGE}" -u "${REPO_ROOT}/scripts/lib/generate_report.py"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "self-heal" ]]; then
  echo
  echo "=== self-healing (real searches, killed and hung) ==="
  bash "${REPO_ROOT}/scripts/test_self_heal.sh"
fi

echo
echo "OK: self-checks passed"
