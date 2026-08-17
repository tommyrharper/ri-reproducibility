#!/usr/bin/env bash
# OpenMP/BLAS thread env for R2D2 docker run invocations.
# Source this file, then expand "${R2D2_DOCKER_ENV_FLAGS[@]}" in docker run.
set -euo pipefail

r2d2_thread_count() {
  if [[ -n "${R2D2_OMP_THREADS:-}" ]]; then
    echo "${R2D2_OMP_THREADS}"
    return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu 2>/dev/null || echo 1
  else
    echo 1
  fi
}

R2D2_OMP_THREADS="${R2D2_OMP_THREADS:-$(r2d2_thread_count)}"
R2D2_DOCKER_ENV_FLAGS=(
  -e "OMP_NUM_THREADS=${R2D2_OMP_THREADS}"
  -e "MKL_NUM_THREADS=${R2D2_OMP_THREADS}"
  -e "OPENBLAS_NUM_THREADS=${R2D2_OMP_THREADS}"
)
