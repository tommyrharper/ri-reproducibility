#!/usr/bin/env bash
# Build report in the r2d2 SIF. `LAST=1`, `RUN=...`, `LIVE=1`, `UPGRADE=1`, and
# `FORCE=1` select rebuilds; index always rebuilds. Outputs go under reports/.
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
ns_require_sifs "${R2D2_SIF}"
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
# finding a run in progress asks squeue, which the SIF does not carry - so
# ./ri report --live runs that first.
if [[ -n "${LIVE_SEL}" ]]; then
  REPORT_ARGS+=(--live)
fi

# The report is matplotlib rasters, not linear algebra: multi-threaded BLAS buys
# it nothing (measured slightly slower) and badly oversubscribes the CPU once the
# run pages are built in parallel processes. One thread each, unless overridden.
R2D2_OMP_THREADS="${R2D2_OMP_THREADS:-1}"
# shellcheck source=scripts/lib/r2d2-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-thread-env.sh"
# A cluster login node is shared, and its administrators allow a few CPUs for a
# few seconds: outside a Slurm job on a host with sbatch, draw with two
# processes per pool rather than one per core. Override with RI_REPORT_WORKERS.
if [[ -z "${RI_REPORT_WORKERS:-}" && -z "${SLURM_JOB_ID:-}" ]] && command -v sbatch >/dev/null 2>&1; then
  RI_REPORT_WORKERS=2
fi
R2D2_ENV_FLAGS+=(--env "RI_REPORT_WORKERS=${RI_REPORT_WORKERS:-}")

# The report reads the repo and writes reports/, at the paths the generator
# hardcodes; the working tree is bound on top of the baked copy of itself.
"${APPTAINER}" exec "${R2D2_ENV_FLAGS[@]}" \
  --bind "${REPO_ROOT}:/workspace/repo:ro" \
  --bind "${REPO_ROOT}/reports:/workspace/out" \
  "${R2D2_SIF}" python3 /workspace/repo/scripts/lib/generate_report.py \
  "${REPORT_ARGS[@]}"

echo "OK: open ${REPO_ROOT}/reports/${OUT_REL} in a browser"
