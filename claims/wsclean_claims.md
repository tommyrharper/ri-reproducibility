# [WSClean](https://arxiv.org/abs/1407.1943) Claims

## Parameters used during benchmarks (unless otherwise mentioned)

Table 2, page 9:
| Parameter                         | Value                       |
|------------------------------------|-----------------------------|
| Array                             | MWA                         |
| Number of elements                | 128                         |
| Image size                        | 3072 × 3072                 |
| Angular pixel size                | 0.72′                       |
| Number of visibilities            | 3.5 × 10<sup>8</sup>        |
| Time resolution                   | 2 s                         |
| Frequency resolution              | 40 kHz                      |
| Observation duration              | 112 s                       |
| Bandwidth                         | 30.72 MHz (768 channels)    |
| Central frequency                 | 182 MHz                     |
| Zenith angle at phase centre      | 10°                         |
| Max w-value for phase centre      | 172 λ (283 m)               |
| Number of polarisations in set    | 4                           |
| Imaged polarisation               | _pp_ (~XX)                  |
| Imaging mode                      | multi-frequency synthesis   |
| Weighting                         | uniform                     |
| Data size                         | 18 GB                       |


## Accuracy

Table 3, page 9:
|                                            | WSCLEAN            | WSCLEAN + recentre | CASA              |
|--------------------------------------------|--------------------|--------------------|-------------------|
| **Zenith angle 0° (12 w-layers/planes)**   |                    |                    |                   |
| Source flux standard error                 | 1.31%              | —                  | 1.34%             |
| RMS in residual image (mJy/b)              | 0.94               | —                  | 1.90              |
| Computational time                         | 8.5 min            | —                  | 19.3 min          |
|                                            |                    |                    |                   |
| **Zenith angle 0° (128 w-layers/planes)**  |                    |                    |                   |
| Source flux standard error                 | 1.39%              | —                  | 2.08%             |
| RMS in residual image (mJy/b)              | 0.94               | —                  | 0.94              |
| Computational time                         | 10.3 min           | —                  | 19.6 min          |
|                                            |                    |                    |                   |
| **Zenith angle 10° (195 w-layers/planes)** |                    |                    |                   |
| Source flux standard error                 | 1.75%              | 1.40%              | 2.41%             |
| RMS in residual image (mJy/b)              | 0.90               | 1.03               | 1.07              |
| Computational time                         | 15.3 min           | 6.6 min            | 178.2 min         |


## Performance



## Verbal claims

### WSClean is about an order of magnitude faster than CASA w-projection on MWA data

Made in:
- Abstract ✅
- Table 3, pg 9 ✅
- Conclusion (section 5) ✅
