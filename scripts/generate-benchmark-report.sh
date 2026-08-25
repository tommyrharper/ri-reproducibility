#!/usr/bin/env bash
# Build HTML reports inside the r2d2 image (astropy + matplotlib + anesthetic).
#
# Usage:
#   scripts/generate-benchmark-report.sh benchmarks
#   scripts/generate-benchmark-report.sh nested-sampling
#   LAST=1 scripts/generate-benchmark-report.sh nested-sampling
#   RUN=results/nested-sampling-poc/<id> scripts/generate-benchmark-report.sh nested-sampling
#
# Outputs:
#   benchmarks/report.html
#   benchmarks/nested-sampling-report.html
#   benchmarks/nested-sampling-report-last.html       (LAST=N)
#   benchmarks/nested-sampling-report-<run>.html      (RUN=)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"
KIND="${1:-}"
LIMIT="${LAST:-}"
RUN_SEL="${RUN:-}"

case "${KIND}" in
  benchmarks)
    OUT_NAME="report.html"
    ;;
  nested-sampling)
    if [[ -n "${LIMIT}" && -n "${RUN_SEL}" ]]; then
      echo "refuse: LAST= and RUN= cannot be used together" >&2
      exit 1
    fi
    if [[ -n "${RUN_SEL}" ]]; then
      run_base="${RUN_SEL%/}"
      run_base="${run_base##*/}"
      if [[ "${run_base}" == "poc-summary.json" ]]; then
        run_base="$(basename "$(dirname "${RUN_SEL}")")"
      fi
      run_base="${run_base//[^A-Za-z0-9._-]/_}"
      OUT_NAME="nested-sampling-report-${run_base}.html"
    elif [[ -n "${LIMIT}" ]]; then
      OUT_NAME="nested-sampling-report-last.html"
    else
      OUT_NAME="nested-sampling-report.html"
    fi
    ;;
  *)
    echo "usage: $0 {benchmarks|nested-sampling}" >&2
    echo "  nested-sampling also reads LAST=N and RUN=<run dir or name>" >&2
    exit 1
    ;;
esac

# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

REPORT_ARGS=(--kind "${KIND}" "/workspace/out/${OUT_NAME}")
if [[ -n "${LIMIT}" ]]; then
  REPORT_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${RUN_SEL}" ]]; then
  REPORT_ARGS+=(--run "${RUN_SEL}")
fi

docker run --rm --platform "${DOCKER_DEFAULT_PLATFORM:-linux/arm64}" \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/benchmarks:/workspace/out:rw" \
  --entrypoint python3 \
  "${IMAGE}" /workspace/repo/scripts/lib/generate_benchmark_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/benchmarks/${OUT_NAME} in a browser"
