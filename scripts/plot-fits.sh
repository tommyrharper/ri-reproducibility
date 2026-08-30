#!/usr/bin/env bash
# Render FITS images to PNG with astropy + matplotlib in the R2D2 image.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"
# shellcheck source=scripts/lib/r2d2-docker-thread-env.sh
source "${REPO_ROOT}/scripts/lib/r2d2-docker-thread-env.sh"

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  # Use outputs from either smoke test; skip missing files.
  candidates=(results/smoke-test-r2d2/r2d2-unet/data_3c353/dirty_normalised.fits
              results/smoke-test-r2d2/r2d2-unet/data_3c353/PSF.fits
              results/smoke-test-r2d2/r2d2-unet/data_3c353/R2D2_model_image.fits
              results/smoke-test-r2d2/r2d2-unet/data_3c353/R2D2_residual_dirty_image.fits
              results/smoke-test-wsclean/smoke-dirty.fits
              results/smoke-test-wsclean/smoke-psf.fits
              results/smoke-test-wsclean/smoke-image.fits
              results/smoke-test-wsclean/smoke-model.fits
              results/smoke-test-wsclean/smoke-residual.fits)
  targets=()
  for c in "${candidates[@]}"; do
    if [ -f "${REPO_ROOT}/${c}" ]; then
      targets+=("$c")
    else
      echo "SKIPPED (not found - run the matching smoke test first): ${c}"
    fi
  done
  if [ "${#targets[@]}" -eq 0 ]; then
    echo "FATAL: no smoke-test FITS output found. Run ./ri smoke first." >&2
    exit 1
  fi
fi

docker run --rm --platform "${PLATFORM}" \
  "${R2D2_DOCKER_ENV_FLAGS[@]}" \
  -v "${RESULTS_DIR}:/results" \
  -v "${REPO_ROOT}:/workspace/repo:ro" \
  --entrypoint python3 \
  "${R2D2_IMAGE}" -c "
import sys, os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for rel in sys.argv[1:]:
    src = rel if rel.startswith('/') else os.path.join('/workspace/repo', rel)
    if not os.path.isfile(src):
        raise SystemExit(f'FATAL: {src!r} not found (looked for {rel!r})')
    data = fits.getdata(src)
    data = np.squeeze(np.asarray(data, dtype=float))
    if data.ndim != 2:
        raise SystemExit(f'FATAL: {src!r} is not a 2-D image after squeezing (shape {data.shape})')

    vmin, vmax = ZScaleInterval().get_limits(data)
    if vmin == vmax:
        # ZScaleInterval degenerates to (0, 0) on sparse images (e.g. a
        # CLEAN component model that's mostly zeros) - fall back to the
        # actual data range rather than feeding AsinhStretch a zero-width
        # interval (which divides by zero and renders every pixel NaN).
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(data, origin='lower', cmap='inferno', norm=norm)
    ax.set_title(os.path.basename(src))
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = os.path.join('/results', os.path.splitext(os.path.basename(src))[0] + '.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'OK: {rel} -> results/{os.path.basename(out)}')
" "${targets[@]}"
