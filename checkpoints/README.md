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
