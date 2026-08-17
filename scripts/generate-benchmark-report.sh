#!/usr/bin/env bash
# Builds benchmarks/report.html from benchmarks/manifests/*.json and
# results/nested-sampling-poc/*/poc-summary.json - a single self-contained
# page (metrics + rendered output images) for browsing benchmark runs. See
# scripts/lib/generate_benchmark_report.py for what it does; runs inside the
# r2d2 image to reuse its astropy + matplotlib + anesthetic, same approach
# as scripts/plot-fits.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"

docker run --rm --platform linux/arm64 \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/benchmarks:/workspace/out:rw" \
  --entrypoint python3 \
  "${IMAGE}" /workspace/repo/scripts/lib/generate_benchmark_report.py /workspace/out/report.html

echo "OK: open ${REPO_ROOT}/benchmarks/report.html in a browser"
