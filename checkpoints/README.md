# checkpoints/

Bind-mounted into the R2D2 container at `/checkpoints`. Nothing in this
directory (other than this file) is tracked by Git.

## R2D2 VLA-trained DNN series

Fetched with `make fetch-r2d2-checkpoints` (wraps
`scripts/fetch-r2d2-checkpoints.sh`).

Source: Aghabiglou et al., DOI
[10.17861/e3060b95-4fe6-4b61-9f72-d77653c305bb](https://doi.org/10.17861/e3060b95-4fe6-4b61-9f72-d77653c305bb),
Heriot-Watt Research Portal, license CC BY.

**This download cannot be fully automated.** The host
(`researchportal.hw.ac.uk`) serves files through a Cloudflare bot
challenge that rejects non-browser requests (`curl`/`wget` get an HTTP 403
with a Cloudflare challenge page, verified 2026-08-03). The fetch script
detects this, prints the direct URLs below, and tells you to download the
realisation ZIP(s) through a browser and place them in this directory.
The upstream landing page does not publish per-file checksums, so the
script records the SHA-256 of whatever you place here (self-recorded, not
verified against an upstream value) into `checkpoints/CHECKSUMS.sha256`.

Ten realisation archives exist (5 U-Net "A1_T2", 5 U-WDSR "A2_T2"), each
~3.5-5.3 GB. The smoke test and the R2D2 nested-sampling search both run
with **one** realisation of one architecture (`ckpt_realisations: 1`),
e.g. `R2D2_A1_T2_Realisation1.zip` (~5.25 GB):

- https://researchportal.hw.ac.uk/files/146289536/R2D2_A1_T2_Realisation1.zip

After extraction, each `.ckpt` must be placed so that
`checkpoints/R2D2_A1/R2D2_UNet_N<k>.ckpt` (k = 1..25) exists - the
architecture subdirectory is part of the path, not optional: it is what
`config/r2d2/R2D2_U-Net.yaml`'s `ckpt_path: /checkpoints/R2D2_A1`,
`scripts/lib/nested_sampling/polychord_r2d2_poc.py`, and
`scripts/smoke-test-r2d2.sh`'s stage-5 probe all look for (the
`checkpoints/` directory itself is what gets bind-mounted at
`/checkpoints`). The U-WDSR series is the same shape one directory over:
`checkpoints/R2D2_A2/R2D2_UWDSR_N<k>.ckpt`, per
`config/r2d2/R2D2_U-WDSR.yaml`. See R2D2-RI's README section "VLA-trained
DNN series" for the full upstream naming convention.
