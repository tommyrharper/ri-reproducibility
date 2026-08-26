#!/usr/bin/env bash
# Build the nested-sampling HTML report inside the r2d2 image (astropy +
# matplotlib + anesthetic).
#
# Usage:
#   scripts/generate-report.sh
#   LAST=1 scripts/generate-report.sh
#   RUN=results/nested-sampling/<id> scripts/generate-report.sh
#   UPGRADE=1 scripts/generate-report.sh
#   FORCE=1 scripts/generate-report.sh
#
# Outputs:
#   reports/nested-sampling-report/index.html   (links to every run)
#   reports/nested-sampling-report/<run>.html   (one page per run)
#   reports/nested-sampling-report/images/      (PNGs the pages reference)
#
# Each page records the report version that wrote it. Up-to-date pages are
# skipped; UPGRADE=1 rebuilds the ones an older report version wrote, and
# FORCE=1 (or a RUN= selection) rebuilds them all. The index is always rebuilt.
# Rebuilding a page reuses the PNGs under images/, which is most of the cost -
# delete the whole report directory to force those to be drawn again. A rebuild
# that draws nothing skips the astropy/matplotlib import as well. Pages that
# do need building are built in parallel: each run is two concurrent tasks, its
# corner plot and the rest of its page, in two pools forked either side of the
# astropy import so only the parent pays for it. anesthetic is imported in the
# parent too, so the corner-plot workers inherit it rather than each repeating
# the import. The corner plot also de-duplicates pandas' per-plot-call tick
# housekeeping and memoises matplotlib's per-axis tick updates - see
# docs/nested-sampling.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

LIMIT="${LAST:-}"
RUN_SEL="${RUN:-}"
FORCE_SEL="${FORCE:-}"
UPGRADE_SEL="${UPGRADE:-}"

if [[ -n "${LIMIT}" && -n "${RUN_SEL}" ]]; then
  echo "refuse: LAST= and RUN= cannot be used together" >&2
  exit 1
fi

OUT_REL="nested-sampling-report/index.html"
# Create on the host so the directory isn't owned by the container's root.
mkdir -p "${REPO_ROOT}/reports/nested-sampling-report"

REPORT_ARGS=(/workspace/out/nested-sampling-report)
if [[ -n "${LIMIT}" ]]; then
  REPORT_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${RUN_SEL}" ]]; then
  REPORT_ARGS+=(--run "${RUN_SEL}")
fi
if [[ -n "${FORCE_SEL}" ]]; then
  REPORT_ARGS+=(--force)
fi
if [[ -n "${UPGRADE_SEL}" ]]; then
  REPORT_ARGS+=(--upgrade)
fi

# The report is matplotlib rasters, not linear algebra: multi-threaded BLAS buys
# it nothing (measured slightly slower) and badly oversubscribes the CPU once the
# run pages are built in parallel processes. One thread each, unless overridden.
R2D2_OMP_THREADS="${R2D2_OMP_THREADS:-1}"
# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

# --network none: the report only reads the repo and writes reports/, and
# skipping the container network setup is ~0.3s of every invocation.
docker run --rm --network none --platform "${PLATFORM}" \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/reports:/workspace/out:rw" \
  --entrypoint python3 \
  "${R2D2_IMAGE}" /workspace/repo/scripts/lib/generate_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/reports/${OUT_REL} in a browser"
