#!/usr/bin/env bash
# Staged R2D2-RI smoke test, per the acceptance criteria in the project
# brief: each stage must fail with a precise message, not a stack trace,
# and stage 5 (real inference) only runs once checkpoints are present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${R2D2_IMAGE:-ri-reproducibility/r2d2:cpu}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CONFIG_DIR="${REPO_ROOT}/config/r2d2"

# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

mkdir -p "${RESULTS_DIR}/smoke-test-r2d2"

run() {
  docker run --rm --platform linux/arm64 \
    "${R2D2_DOCKER_ENV_FLAGS[@]}" \
    -v "${CHECKPOINTS_DIR}:/checkpoints:ro" \
    -v "${RESULTS_DIR}/smoke-test-r2d2:/results" \
    -v "${CONFIG_DIR}:/workspace/config:ro" \
    --entrypoint python3 \
    "${IMAGE}" "$@"
}

echo "==> [1/5] importing critical third-party Python packages"
run -c "
import astropy, torch, lightning, tensorboard, torchkbnufft, torchmetrics
import tqdm, pydantic, matplotlib, finufft, pytorch_finufft, psutil, h5py
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('OK: all critical third-party packages import cleanly')
"

echo "==> [2/5] importing R2D2-RI application modules"
run -c "
import sys
sys.path.insert(0, 'src')
from ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights
from optimiser import R2D2
from utils import load_data_to_tensor, parse_args_imaging, vprint, create_meas_op, snr, to_log
print('OK: R2D2-RI application modules import cleanly')
"

echo "==> [3/5] loading + validating the bundled example measurement file"
run -c "
import sys, os
sys.path.insert(0, 'src')
from utils import load_data_to_tensor
path = 'data/data_3c353.mat'
if not os.path.exists(path):
    raise SystemExit(f'FATAL: bundled example measurement file not found at {path!r} inside the image')
data = load_data_to_tensor(uv_file_path=path, super_resolution=1.52, verbose=False)
print(f'OK: {path} loaded via R2D2-RI\\'s own load_data_to_tensor(), {data[\"y\"].numel()} visibilities')
"

echo "==> [4/5] validating an example imaging configuration"
run -c "
import yaml, os
path = '/workspace/config/R2D2_U-Net.yaml'
with open(path) as f:
    cfg = yaml.safe_load(f)
required = ['data_file', 'output_path', 'im_dim_x', 'im_dim_y', 'num_iter', 'ckpt_path']
missing = [k for k in required if k not in cfg]
if missing:
    raise SystemExit(f'FATAL: {path} is missing required field(s): {missing}')
print(f'OK: {path} is structurally valid ({cfg[\"architecture\"]}, {cfg[\"im_dim_x\"]}x{cfg[\"im_dim_y\"]}, num_iter={cfg[\"num_iter\"]})')
"

echo "==> [5/5] checking for pretrained DNN checkpoints"
CKPT_DIR_A1="${CHECKPOINTS_DIR}/R2D2_A1"
if [ ! -f "${CKPT_DIR_A1}/R2D2_UNet_N1.ckpt" ]; then
  cat <<EOF

SKIPPED: no pretrained checkpoints found at ${CKPT_DIR_A1}/R2D2_UNet_N1.ckpt
Real inference was NOT attempted. This is expected out of the box.

To run real inference:
  1. make fetch-r2d2-checkpoints   (prints manual-download instructions -
     the checkpoint host is behind Cloudflare and cannot be curl'd)
  2. Extract the realisation you downloaded so that
     ${CKPT_DIR_A1}/R2D2_UNet_N<k>.ckpt exists for k = 1..25
  3. Re-run this script, or: make smoke-test-r2d2
EOF
  echo "OK: smoke test stages 1-4 passed; stage 5 (inference) skipped as documented above."
  exit 0
fi

echo "    checkpoints found, running one real R2D2 imaging pass"
run ./src/imager.py --config /workspace/config/R2D2_U-Net.yaml --ckpt_path /checkpoints/R2D2_A1
echo "OK: R2D2-RI smoke test complete, including real inference. Output in ${RESULTS_DIR}/smoke-test-r2d2"
