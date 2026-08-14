# Table: Training computational cost

- **Label:** `table:training_cost`
- **Source:** `latex/R2D2.tex` lines 463–479

## Caption

Training computational details of R2D2, R3D3 realizations (R3D3<sup>3L</sup>, R3D3<sup>6L</sup>), and the end-to-end DNNs corresponding to the first components in their series, U-Net and R2D2-Net (R2D2-Net<sup>3L</sup>, R2D2-Net<sup>6L</sup>), respectively. Results are reported in terms of: the number of iterations (*I*), the number of learnable parameters of their network components (*Q*), the cumulative number of epochs (*n*<sub>epochs</sub>), the number of CPU cores (*n*<sub>cpu</sub>) deployed for generating the dirty images and updating the residual dirty images, and the number of GPUs (*n*<sub>gpu</sub>), deployed for DNN training and updating the image estimates. The training computational cost is provided in GPU hr and CPU hr.

## Data

| Algorithm | *I* | *Q* (×10<sup>6</sup>) | *n*<sub>epochs</sub> | *n*<sub>gpu</sub> | *n*<sub>cpu</sub> | GPU hr | CPU hr |
|-----------|-----|----------------------|---------------------|------------------|------------------|--------|--------|
| U-Net | 1 | 31 | 264 | 4 | 6 | 82 | 336 |
| R2D2-Net<sup>3L</sup> | 1 | 93 | 405 | 12 | 6 | 873 | 336 |
| R2D2-Net<sup>6L</sup> | 1 | 186 | 192 | 12 | 6 | 414 | 336 |
| R2D2 | 15 | 31 | 398 | 4 | 6 | 160 | 4757 |
| R3D3<sup>3L</sup> | 7 | 93 | 605 | 12 | 6 | 1291 | 2165 |
| R3D3<sup>6L</sup> | 8 | 186 | 591 | 12 | 6 | 1276 | 2244 |

## Notes

- No `\tablenotetext` attached in the LaTeX.
- *Q* column header is \(Q (\times 10^6)\) (millions of learnable parameters).
