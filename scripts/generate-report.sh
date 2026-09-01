#!/usr/bin/env bash
# Build report in r2d2 image. `LAST=1`, `RUN=...`, `LIVE=1`, `UPGRADE=1`, and
# `FORCE=1` select rebuilds; index always rebuilds. Outputs go under reports/.
# Container cleanup is asynchronous because `docker run --rm` blocks on teardown.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

LIMIT="${LAST:-}"
RUN_SEL="${RUN:-}"
FORCE_SEL="${FORCE:-}"
UPGRADE_SEL="${UPGRADE:-}"
LIVE_SEL="${LIVE:-}"

if [[ -n "${LIMIT}" && -n "${RUN_SEL}" ]]; then
  echo "refuse: LAST= and RUN= cannot be used together" >&2
  exit 1
fi

if [[ -n "${LIVE_SEL}" && ( -n "${LIMIT}" || -n "${RUN_SEL}" ) ]]; then
  echo "refuse: LIVE= selects the runs still going, so it cannot be used with LAST= or RUN=" >&2
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
# The summaries themselves are written on the host by scripts/live_runs.py -
# finding a run in progress needs `ps`, and this container has no host process
# table - so ./ri report --live runs that first.
if [[ -n "${LIVE_SEL}" ]]; then
  REPORT_ARGS+=(--live)
fi

# The report is matplotlib rasters, not linear algebra: multi-threaded BLAS buys
# it nothing (measured slightly slower) and badly oversubscribes the CPU once the
# run pages are built in parallel processes. One thread each, unless overridden.
R2D2_OMP_THREADS="${R2D2_OMP_THREADS:-1}"
# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

# The report writes to the bind mount, so once python3 has exited nothing about
# the container matters - but `docker run --rm` keeps the CLI blocked for ~0.13s
# of every invocation while it tears the rootfs down. Name it and remove it from
# an EXIT trap instead, so the teardown runs after the script has reported.
CONTAINER="nested-sampling-report-$$-${RANDOM}"
trap 'docker rm -f "${CONTAINER}" >/dev/null 2>&1 &' EXIT

# --network none: the report only reads the repo and writes reports/, and
# skipping the container network setup is ~0.3s of every invocation.
docker run --network none --platform "${PLATFORM}" --name "${CONTAINER}" \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  -v "${REPO_ROOT}/reports:/workspace/out:rw" \
  --entrypoint python3 \
  "${R2D2_IMAGE}" /workspace/repo/scripts/lib/generate_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/reports/${OUT_REL} in a browser"
