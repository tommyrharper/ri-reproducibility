#!/usr/bin/env bash
# Validate MS -> R2D2 .mat conversion: simulate an MS (or take MS_PATH),
# convert it in the MeqTrees SIF, load the .mat with R2D2's own loader.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/r2d2-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-thread-env.sh"

ns_require_sifs "${MEQTREES_SIF}" "${R2D2_SIF}"
NESTED_SAMPLING="${REPO_ROOT}/scripts/lib/nested_sampling"
SUPER_RESOLUTION="$(grep -E '^DEFAULT_SUPER_RESOLUTION *= *' "${NESTED_SAMPLING}/common.py" | sed 's/.*= *//')"

if [[ -n "${MS_PATH:-}" ]]; then
  MS_PATH="$(cd "$(dirname "${MS_PATH}")" && pwd)/$(basename "${MS_PATH}")"
  WORK_DIR="$(dirname "${MS_PATH}")"
else
  mkdir -p "${REPO_ROOT}/results"
  WORK_DIR="$(mktemp -d "${REPO_ROOT}/results/ms-mat-bridge-XXXXXX")"
  MS_PATH="${WORK_DIR}/sim.ms"
  "${APPTAINER}" exec --bind "${REPO_ROOT}" "${MEQTREES_SIF}" \
    python3 "${NESTED_SAMPLING}/simulate_point_source_ms.py" \
    --output-ms "${MS_PATH}" --metadata-json "${WORK_DIR}/simulation.json" \
    --vla-config VLA.A --observation-minutes 4 \
    --channel-count 2 --start-frequency-hz 1.0e9 --channel-width-hz 1.0e6 \
    --source-flux-jy 1.0 --dynamic-range 100 --seed 42
fi
MAT_PATH="${WORK_DIR}/r2d2_data.mat"

"${APPTAINER}" exec --bind "${REPO_ROOT}" --bind "${WORK_DIR}" "${MEQTREES_SIF}" \
  python3 "${NESTED_SAMPLING}/ms_to_r2d2_mat.py" --ms-path "${MS_PATH}" --mat-path "${MAT_PATH}"

"${APPTAINER}" exec --pwd /opt/r2d2/R2D2-RI "${R2D2_ENV_FLAGS[@]}" \
  --bind "${WORK_DIR}" "${R2D2_SIF}" python3 -c "
import sys; sys.path.insert(0, 'src')
from utils import load_data_to_tensor
data = load_data_to_tensor(uv_file_path='${MAT_PATH}', super_resolution=${SUPER_RESOLUTION}, verbose=False)
print('visibility_count', data['y'].numel())
print('u_count', data['u'].numel())
print('nW_count', data['nW'].numel())
"

echo "OK: MS -> R2D2 .mat bridge self-check passed (${MAT_PATH})"
