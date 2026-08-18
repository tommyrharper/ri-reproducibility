#!/usr/bin/env bash
# Build HTML reports inside the r2d2 image (astropy + matplotlib + anesthetic).
#
# Usage:
#   scripts/generate-benchmark-report.sh benchmarks
#   scripts/generate-benchmark-report.sh nested-sampling
#   scripts/generate-benchmark-report.sh nested-sampling 1
#
# Outputs:
#   benchmarks/report.html
#   benchmarks/nested-sampling-report.html
#   benchmarks/nested-sampling-report-last.html  (when a limit is given)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"
KIND="${1:-}"
LIMIT="${2:-}"

case "${KIND}" in
  benchmarks)
    OUT_NAME="report.html"
    ;;
  nested-sampling)
    if [[ -n "${LIMIT}" ]]; then
      OUT_NAME="nested-sampling-report-last.html"
    else
      OUT_NAME="nested-sampling-report.html"
    fi
    ;;
  *)
    echo "usage: $0 {benchmarks|nested-sampling} [limit]" >&2
    exit 1
    ;;
esac

# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

REPORT_ARGS=(--kind "${KIND}" "/workspace/out/${OUT_NAME}")
if [[ -n "${LIMIT}" ]]; then
  REPORT_ARGS+=(--limit "${LIMIT}")
fi

docker run --rm --platform linux/arm64 \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/benchmarks:/workspace/out:rw" \
  --entrypoint python3 \
  "${IMAGE}" /workspace/repo/scripts/lib/generate_benchmark_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/benchmarks/${OUT_NAME} in a browser"
