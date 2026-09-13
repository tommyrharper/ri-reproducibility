#!/usr/bin/env bash
# Run the checks that need the images, inside their SIFs, on the working tree.
# Self-heal uses throwaway searches with rank/memory limits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

TARGET="${1:-all}"
ns_require_sifs "${MEQTREES_SIF}" "${R2D2_SIF}" "${WSCLEAN_SIF}" "${POLYCHORD_SIF}"

# The repo is bound at its own host path, the way the pools bind it, so paths
# inside a check mean the same thing they would during a run. Flags before
# the SIF are apptainer's (`--env X=1`), as `-e X=1` was docker's.
image_run() {
  "${APPTAINER}" exec --pwd "${REPO_ROOT}" --bind "${REPO_ROOT}" "$@"
}

nested_sampling() { echo "${REPO_ROOT}/scripts/lib/nested_sampling/$1"; }

# Use uv's pinned Python >=3.11: host `python3` may lack tomllib, which
# `test_self_checks.py` needs through common.py. `--no-project` keeps checks
# independent of pyproject.toml, `--isolated` of the `.venv` a `./ri` run left
# behind - without it these four look stdlib-only here and fail in CI.
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
  echo "=== simulate (${MEQTREES_SIF##*/}) ==="
  image_run "${MEQTREES_SIF}" python3 -u "$(nested_sampling simulate_point_source_ms.py)" --self-check
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "r2d2-serve" ]]; then
  echo
  echo "=== r2d2 imaging worker (${R2D2_SIF##*/}) ==="
  image_run "${R2D2_SIF}" python3 -u "$(nested_sampling r2d2_serve.py)" --self-check
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "zygote" ]]; then
  echo
  echo "=== wsclean fork server (${WSCLEAN_SIF##*/}) ==="
  image_run "${WSCLEAN_SIF}" python3 -u "${REPO_ROOT}/scripts/test_zygote.py"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "wsclean" ]]; then
  echo
  echo "=== wsclean sampler (${POLYCHORD_SIF##*/}) ==="
  image_run --env POLYCHORD_WSCLEAN_SELF_CHECK=1 \
    "${POLYCHORD_SIF}" python3 -u "$(nested_sampling polychord_wsclean.py)"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "r2d2" ]]; then
  echo
  echo "=== r2d2 sampler (${POLYCHORD_SIF##*/}) ==="
  image_run --env POLYCHORD_R2D2_SELF_CHECK=1 \
    "${POLYCHORD_SIF}" python3 -u "$(nested_sampling polychord_r2d2.py)"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "report" ]]; then
  echo
  echo "=== HTML report (${R2D2_SIF##*/}) ==="
  image_run --env GENERATE_REPORT_SELF_CHECK=1 \
    "${R2D2_SIF}" python3 -u "${REPO_ROOT}/scripts/lib/generate_report.py"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "self-heal" ]]; then
  echo
  echo "=== self-healing (real searches, killed and hung) ==="
  bash "${REPO_ROOT}/scripts/test_self_heal.sh"
fi

echo
echo "OK: self-checks passed"
