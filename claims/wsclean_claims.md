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

Based on simulated MWA observations (page 8, 4.1 Accuracy):
```
To assess the accuracy of WSCLEAN and CASA’s clean task, we
simulate a MWA observation with 100 sources of 1 Jy in a 20◦
diameter area, without adding system noise. A unitary primary beam
is assumed. We image the simulated set with WSCLEAN and CASA
using Cotton-Schwab cleaning to a threshold of 10 mJy. The two
imagers calculate slightly different restoring (synthesised) beams,
hence to avoid bias the restoring beams are fixed. Other imaging
parameters are given in Table 2. The AEGEAN program (Hancock
et al. 2012) is used to perform source detection on the produced
images. Sidelobe noise of the residual 10 mJy source structures
triggers a few false detections. These are ignored."
```

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

Based on real MWA observations:
```
We measure the performance of the imagers using several MWA
data sets. Each specific configuration is run five times and standard deviations are calculated. The variation in duration between
runs is typically a few seconds. In each benchmark, the wall-clock
time is measured that is required to produce the synthesised pointspread function and the image itself. No cleaning or prediction is
performed, and the optimisation of §3.5 is not used. The results are
given in Fig. 7.
```

Digitised approximate values from Fig. 7 (page 10). Sources:
`wsclean_figure_7a_digitised.csv` … `wsclean_figure_7e_digitised.csv`.
`R+WSClean` = WSClean with the recentring technique. Missing cells (`—`)
are series/points not present in that panel of the figure.

### Figure 7a — runtime vs number of visibilities

| Number of visibilities | WSClean ZA=10° (min) | WSClean ZA=0° (min) | CASA ZA=10° (min) | CASA ZA=0° (min) |
|---|---|---|---|---|
| 2.146×10⁷ | 2.111 | 0.3201 | 5.355 | 1.012 |
| 3.981×10⁷ | 2.163 | 0.414 | 6.675 | 1.089 |
| 8.545×10⁷ | 2.19 | 0.5623 | 10 | 1.462 |
| 1.855×10⁸ | 2.599 | 0.9406 | 16.52 | 2.217 |
| 3.981×10⁸ | 4.295 | 2.035 | 29.38 | 3.847 |
| 8.545×10⁸ | 7.273 | 3.618 | 57.63 | 9.522 |
| 1.280×10⁹ | 10.25 | 5.099 | 82.2 | 13.42 |

### Figure 7b — runtime vs zenith angle

| Zenith angle (°) | WSClean FOV=36° (min) | WSClean FOV=24° (min) | R+WSClean FOV=36° (min) | R+WSClean FOV=24° (min) | CASA FOV=36° (min) | CASA FOV=24° (min) |
|---|---|---|---|---|---|---|
| 0 | 1.4 | 1.6 | 3.8 | 4.1 | 3.5 | 3.6 |
| 2 | 1.8 | 1.6 | 4.3 | 4.2 | 7 | — |
| 5 | 2.7 | 1.55 | 4.6 | 4.5 | 13 | 6 |
| 10 | 4 | 1.6 | 4.5 | 4.4 | 29 | 10 |
| 15 | 5.2 | 1.7 | 4.5 | 4.2 | — | 15 |
| 25 | 7.8 | 2 | 4 | 4 | — | 30 |
| 45 | 11.5 | 2.4 | 4.4 | 4 | — | 65 |
| 80 | 17 | 4.2 | 4.8 | 4.5 | — | 100 |

### Figure 7c — runtime vs image resolution (pixels along one side)

| Resolution (px) | WSClean ZA=10° (min) | WSClean ZA=0° (min) | R+WSClean ZA=10° (min) | CASA ZA=10° (min) | CASA ZA=0° (min) |
|---|---|---|---|---|---|
| 1000 | 1.7 | 1.3 | 2.2 | 18 | 4 |
| 1400 | 1.4 | 1.25 | 2.35 | 21 | 3.5 |
| 1700 | 1.5 | — | 2.35 | 27 | 3.9 |
| 1800 | — | 1.28 | — | — | — |
| 1900 | 1.7 | — | 2.4 | 26.5 | 5.5 |
| 2100 | 2 | — | 2.4 | 26.5 | 3.8 |
| 2200 | — | 1.27 | — | — | — |
| 2500 | 2.5 | — | 2.4 | 26.3 | 3.9 |
| 3000 | 4.2 | 1.35 | — | — | 4 |
| 3100 | — | — | 2.5 | — | — |
| 3200 | — | — | — | 28 | — |
| 3600 | 5.3 | 1.4 | 2.6 | 27 | 4.1 |
| 4100 | 7 | 1.45 | 2.8 | — | 4.3 |
| 5200 | 10 | 1.9 | 2.8 | — | 4.8 |
| 6100 | 18 | 2.2 | 3 | — | 5 |
| 7200 | 26 | 2.8 | 3.4 | — | 5.7 |
| 8200 | 34 | 4.3 | 4.7 | — | 6.1 |
| 9200 | 40 | 4.2 | — | — | 6.6 |
| 10000 | 43 | 3.4 | 4.9 | — | 6.8 |
| 11000 | 56 | 4.7 | — | — | — |
| 12800 | 112 | 5.8 | 9.2 | — | 8.8 |

### Figure 7d — runtime vs number of frequency channels

| Frequency channels | WSClean ZA=10° (min) | WSClean ZA=0° (min) | R+WSClean ZA=10° (min) | R+WSClean ZA=0° (min) | CASA ZA=10° (min) | CASA ZA=0° (min) |
|---|---|---|---|---|---|---|
| 1 | 4 | 1.4 | 4.5 | 3.8 | 30 | 3.7 |
| 2 | 6.2 | 1.8 | 6 | 4.8 | 29.5 | 5.2 |
| 4 | 10.8 | 2 | 8.7 | 4.3 | 30 | 5.1 |
| 8 | 21 | 2.6 | 13.5 | 5 | — | 6.2 |
| 16 | 40 | 3.8 | 25 | 6 | — | 7.8 |
| 24 | 58 | 5.2 | 38 | 7.2 | — | 11 |
| 48 | 115 | 9.5 | 68 | 11.5 | — | 12 |
| 92 | 230 | 16 | 125 | 18.5 | — | 28 |

### Figure 7e — runtime vs field of view

| FOV (°) | WSClean ZA=10° (min) | WSClean ZA=0° (min) | R+WSClean ZA=10° (min) | CASA ZA=10° (min) | CASA ZA=0° (min) |
|---|---|---|---|---|---|
| 5 | 2.2 | 1.8 | 4.4 | 3.6 | 3.6 |
| 10 | 1.8 | 1.7 | 4.4 | 4.4 | — |
| 20 | 1.7 | 1.6 | 4.3 | 6.8 | 3.6 |
| 37 | 3.8 | 1.4 | — | 30 | 3.7 |
| 58 | 6.8 | 1.38 | 4.3 | — | 4.8 |
| 80 | 14.5 | 1.45 | 4.6 | — | 5.4 |
| 110 | 25 | 1.9 | 4.6 | — | 8.5 |
| 150 | 25.2 | 1.8 | 4.6 | — | 18 |
| 180 | 25.3 | 1.75 | 4.6 | — | 42 |

## Verbal claims

### WSClean is about an order of magnitude faster than CASA w-projection on MWA data

Made in:
- Abstract ✅
- Table 3, pg 9 ✅
- Conclusion (section 5) ✅
