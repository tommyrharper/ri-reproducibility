#!/usr/bin/env bash
# OpenMP/BLAS thread env for R2D2 `apptainer exec` invocations.
# Source this file, then expand "${R2D2_ENV_FLAGS[@]}" before the SIF.
set -euo pipefail

r2d2_thread_count() {
  if [[ -n "${R2D2_OMP_THREADS:-}" ]]; then
    echo "${R2D2_OMP_THREADS}"
    return
  fi
  if command -v nproc >/dev/null 2>&1; then
    # Not the inherited OMP_NUM_THREADS (1 on CSD3), which nproc would echo.
    env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu 2>/dev/null || echo 1
  else
    echo 1
  fi
}

R2D2_OMP_THREADS="${R2D2_OMP_THREADS:-$(r2d2_thread_count)}"
# shellcheck disable=SC2034  # expanded by the sourcing script
# MPLCONFIGDIR: the image's /opt/matplotlib cache is read-only under Apptainer.
R2D2_ENV_FLAGS=(
  --env "MPLCONFIGDIR=${TMPDIR:-/tmp}/ri-matplotlib-${USER:-$(id -u)}"
  --env "OMP_NUM_THREADS=${R2D2_OMP_THREADS}"
  --env "MKL_NUM_THREADS=${R2D2_OMP_THREADS}"
  --env "OPENBLAS_NUM_THREADS=${R2D2_OMP_THREADS}"
)
