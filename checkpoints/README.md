# checkpoints/

Bind-mounted at `/checkpoints`; only this README is tracked.

Fetch with `./ri fetch-checkpoints`. The portal may require a browser; the
script prints the URL and records a local SHA-256. Source: [Heriot-Watt
Research Portal](https://doi.org/10.17861/e3060b95-4fe6-4b61-9f72-d77653c305bb),
CC BY.

One archive is enough for smoke tests and the default search:

`https://researchportal.hw.ac.uk/files/146289536/R2D2_A1_T2_Realisation1.zip`

Extract it so `R2D2_A1/R2D2_UNet_N<k>.ckpt` (`k = 1..25`) exists here. For
U-WDSR, use the corresponding `R2D2_A2/R2D2_UWDSR_N<k>.ckpt` paths.

**`num_chans` is 64, not 32.** Upstream's README says 32 for the pre-trained
U-Nets. The A1/T2 checkpoints here are 64, confirmed by their own
`down_sample_layers.0` weight shape (`[64, 2, 3, 3]`); loading them into a
32-channel model fails on every layer. `config/r2d2/R2D2_U-Net.yaml` already
sets 64 - do not "correct" it back to upstream's demo value.
