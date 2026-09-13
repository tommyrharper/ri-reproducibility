#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

rm -rf "${WSCLEAN_SIF}" "${R2D2_SIF}" "${MEQTREES_SIF}" "${POLYCHORD_SIF}" "${SIF_DIR}/archives"
rm -rf results/.smoke-test-fixtures results/smoke-test-wsclean results/smoke-test-r2d2

echo "SIFs, their archives and generated smoke-test outputs removed."
echo "data/, checkpoints/, and results/ contents you added yourself are left untouched."
echo "Run './ri images import' to bring the images back."
