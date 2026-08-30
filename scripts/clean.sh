#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

docker rmi "${WSCLEAN_IMAGE}" "${WSCLEAN_IMAGE%:*}:native" "${R2D2_IMAGE}" \
  "${MEQTREES_IMAGE}" "${POLYCHORD_IMAGE}" 2>/dev/null || true

rm -rf results/.smoke-test-fixtures results/smoke-test-wsclean results/smoke-test-r2d2

echo "Images and generated smoke-test outputs removed."
echo "data/, checkpoints/, and results/ contents you added yourself are left untouched."
echo "Run 'docker builder prune' to reclaim build-cache disk space if needed."
