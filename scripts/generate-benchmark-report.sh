#!/usr/bin/env bash
# Build HTML reports inside the r2d2 image (astropy + matplotlib + anesthetic).
#
# Usage:
#   scripts/generate-benchmark-report.sh benchmarks
#   scripts/generate-benchmark-report.sh nested-sampling
#   LAST=1 scripts/generate-benchmark-report.sh nested-sampling
#   RUN=results/nested-sampling-poc/<id> scripts/generate-benchmark-report.sh nested-sampling
#   FORCE=1 scripts/generate-benchmark-report.sh nested-sampling
#
# Outputs:
#   benchmarks/report.html
#   benchmarks/nested-sampling-report/index.html   (links to every run)
#   benchmarks/nested-sampling-report/<run>.html   (one page per run)
#
# Nested-sampling run pages that already exist are skipped; FORCE=1 (or a
# RUN= selection) rebuilds them. The index is always rebuilt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

KIND="${1:-}"
LIMIT="${LAST:-}"
RUN_SEL="${RUN:-}"
FORCE_SEL="${FORCE:-}"

REPORT_ARGS=(--kind "${KIND}")

case "${KIND}" in
  benchmarks)
    OUT_REL="report.html"
    REPORT_ARGS+=("/workspace/out/${OUT_REL}")
    ;;
  nested-sampling)
    if [[ -n "${LIMIT}" && -n "${RUN_SEL}" ]]; then
      echo "refuse: LAST= and RUN= cannot be used together" >&2
      exit 1
    fi
    OUT_REL="nested-sampling-report/index.html"
    # Create on the host so the directory isn't owned by the container's root.
    mkdir -p "${REPO_ROOT}/benchmarks/nested-sampling-report"
    REPORT_ARGS+=("/workspace/out/nested-sampling-report")
    if [[ -n "${LIMIT}" ]]; then
      REPORT_ARGS+=(--limit "${LIMIT}")
    fi
    if [[ -n "${RUN_SEL}" ]]; then
      REPORT_ARGS+=(--run "${RUN_SEL}")
    fi
    if [[ -n "${FORCE_SEL}" ]]; then
      REPORT_ARGS+=(--force)
    fi
    ;;
  *)
    echo "usage: $0 {benchmarks|nested-sampling}" >&2
    echo "  nested-sampling also reads LAST=N, RUN=<run dir or name> and FORCE=1" >&2
    exit 1
    ;;
esac

# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

docker run --rm --platform "${PLATFORM}" \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/benchmarks:/workspace/out:rw" \
  --entrypoint python3 \
  "${R2D2_IMAGE}" /workspace/repo/scripts/lib/generate_benchmark_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/benchmarks/${OUT_REL} in a browser"
