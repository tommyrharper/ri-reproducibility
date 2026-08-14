# Table: Generic experiment results

- **Label:** `table:results`
- **Source:** `latex/R2D2.tex` lines 543–630 (`rotatetable*` / `deluxetable*`)

## Caption

Results of the experiment in generic image and data settings obtained by the imaging algorithms. Reconstruction quality metrics are SNR, logSNR, and \(\overline{\sigma}_{\textrm{res.}}\). Computational details are presented in terms of: the total number of iterations (*I*), the total reconstruction time (*t*<sub>tot.</sub>), the average time per iteration of the data fidelity step (*t*<sub>dat.</sub>) and the regularization step (*t*<sub>reg.</sub>), and the allocated resources in GPUs (*n*<sub>gpu</sub>) and CPU cores (*n*<sub>cpu</sub>). Reported values are computed as averages over 200 inverse problems. Information on the programming languages underlying the algorithms implementations is provided.

## Column units (from table head)

| Column | Unit / note |
|--------|-------------|
| SNR | ± std (dB) |
| logSNR | ± std (dB) |
| \(\overline{\sigma}_{\textrm{res.}}\) | ± std (×1E-4) |
| *I* | ± std |
| *t*<sub>tot.</sub> | ± std (s) |
| *t*<sub>dat.</sub> | ± std (s) |
| *t*<sub>reg.</sub> | ± std (s) |
| *n*<sub>cpu</sub>, *n*<sub>gpu</sub> | resource counts (`-` = none) |
| Programming language | C++ / MATLAB / Python |

## Data

LaTeX multirows are expanded to one row per implementation/resource setting. Where quality metrics are shared across Python CPU/GPU rows (via `\multirow`), the shared values are repeated on both rows.

| Algorithm | SNR ± std (dB) | logSNR ± std (dB) | \(\overline{\sigma}_{\textrm{res.}}\) ± std (×1E-4) | *I* ± std | *t*<sub>tot.</sub> ± std (s) | *t*<sub>dat.</sub> ± std (s) | *t*<sub>reg.</sub> ± std (s) | *n*<sub>cpu</sub> | *n*<sub>gpu</sub> | Programming language |
|-----------|----------------|-------------------|-----------------------------------------------------|-----------|------------------------------|------------------------------|------------------------------|------------------|------------------|----------------------|
| CLEAN | 13.6±3.6 | 10.3±3.5 | 5.1±5.2 | 9±1 | 65.9±18.8 | 3.8±1.1 | 3.8±1.0 | 1 | - | C++ |
| uSARA | 30.8±1.9 | 21.9±3.3 | 6.5±8.2 | 1103±373 | 4184.2±1548.9 | 1.4±0.7 | 2.3±1.1 | 1 | - | MATLAB |
| AIRI | 31.3±2.3 | 21.9±4.4 | 6.4±8.0 | 5000±0.0 | 3478.8±1531.4 | 0.66±0.3 | 0.03±0.2 | 1 | 1 | MATLAB |
| U-Net | 20.5±2.7 | 6.6±3.3 | 777.5±467.4 | 1 | 2.2±0.6 | - | 2.2±0.6 | - | 1 | MATLAB |
| U-Net | 20.5±2.7 | 6.6±3.3 | 777.6±467.4 | 1 | 1.1±0.1 | - | 1.1±0.1 | - | 1 | Python |
| R2D2-Net<sup>3L</sup> | 32.6±1.5 | 19.6±5.5 | 21.5±13.8 | 1 | 2.7±0.4 | - | 2.7±0.4 | - | 1 | MATLAB |
| R2D2-Net<sup>3L</sup> | 32.6±1.5 | 19.6±5.4 | 21.5±13.8 | 1 | 1.1±0.1 | - | 1.1±0.1 | - | 1 | Python |
| R2D2-Net<sup>6L</sup> | 33.7±1.7 | 24.0±4.7 | 9.3±6.9 | 1 | 3.8±1.5 | - | 3.8±1.5 | - | 1 | MATLAB |
| R2D2-Net<sup>6L</sup> | 33.7±1.7 | 24.0±4.7 | 9.3±7.0 | 1 | 1.1±0.1 | - | 1.1±0.1 | - | 1 | Python |
| R2D2 | 33.7±1.5 | 25.1±4.9 | 13.5±46.9 | 15 | 12.2±2.8 | 0.4±0.2 | 0.2±0.07 | 1 | 1 | MATLAB |
| R2D2 (Python hybrid CPU+GPU) | 33.7±1.5 | 25.0±4.9 | 13.5±46.9 | 15 | 18.6±5.9 | 1.1±0.4 | 0.1±0.1 | 1 | 1 | Python |
| R2D2 (Python full GPU) | 33.7±1.5 | 25.0±4.9 | 13.5±46.9 | 15 | 2.9±0.3 | 0.05±0.01 | 0.1±0.2 | - | 1 | Python |
| R3D3<sup>3L</sup> | 33.8±1.4 | 25.3±4.6 | 7.6±7.6 | 7 | 9.4±1.5 | 0.5±0.2 | 0.5±0.1 | 1 | 1 | MATLAB |
| R3D3<sup>3L</sup> (Python hybrid CPU+GPU) | 33.8±1.4 | 25.3±4.6 | 7.6±7.6 | 7 | 9.8±4.3 | 1.1±0.4 | 0.2±0.3 | 1 | 1 | Python |
| R3D3<sup>3L</sup> (Python full GPU) | 33.8±1.4 | 25.3±4.6 | 7.6±7.6 | 7 | 1.9±0.5 | 0.05±0.02 | 0.2±0.3 | - | 1 | Python |
| R3D3<sup>6L</sup> | 34.0±1.6 | 25.3±4.7 | 7.9±7.8 | 8 | 15.2±2.1 | 0.5±0.2 | 1.3±0.6 | 1 | 1 | MATLAB |
| R3D3<sup>6L</sup> (Python hybrid CPU+GPU) | 34.0±1.6 | 25.3±4.7 | 7.9±7.8 | 8 | 11.3±3.9 | 1.1±0.4 | 0.2±0.3 | 1 | 1 | Python |
| R3D3<sup>6L</sup> (Python full GPU) | 34.0±1.6 | 25.3±4.7 | 7.9±7.8 | 8 | 2.2±0.3 | 0.05±0.02 | 0.2±0.3 | - | 1 | Python |

## Multirow structure (as in LaTeX)

| Algorithm | Rows in LaTeX | Notes |
|-----------|---------------|-------|
| U-Net | 2 | MATLAB then Python; quality metrics nearly identical (residual 777.5 vs 777.6) |
| R2D2-Net<sup>3L</sup> | 2 | MATLAB then Python; logSNR std 5.5 (MATLAB) vs 5.4 (Python) |
| R2D2-Net<sup>6L</sup> | 2 | MATLAB then Python; residual std 6.9 (MATLAB) vs 7.0 (Python) |
| R2D2 | 3 | MATLAB; then Python hybrid (*n*<sub>cpu</sub>=1, *n*<sub>gpu</sub>=1); then Python full GPU (*n*<sub>cpu</sub>=`-`, *n*<sub>gpu</sub>=1). Python quality metrics shared via `\multirow` (logSNR 25.0±4.9, vs MATLAB 25.1±4.9) |
| R3D3<sup>3L</sup> | 3 | Same MATLAB / Python hybrid / Python full-GPU pattern; quality metrics identical across the three rows |
| R3D3<sup>6L</sup> | 3 | Same pattern; quality metrics identical across the three rows |

Parenthetical row labels `(Python hybrid CPU+GPU)` / `(Python full GPU)` are **not** in the LaTeX algorithm column; they encode the resource allocation implied by `n_cpu`/`n_gpu` and the paper’s implementation section.

## Notes / footnotes (from `\tablenotetext`)

**Note.** For enhanced readability, a color-coding is considered to categorize the performance of the imaging algorithms: high in green, sub-optimal in red. Specific to CLEAN, the reported number of iterations corresponds to the number of “major cycles” reached at convergence. Two inverse problems from the test dataset diverged and are therefore excluded from the reported results. These instances of instability in the CLEAN implementation could potentially stem from the *deep* cleaning.

### Color coding (LaTeX macros; stripped from numeric cells above)

In the source, `\highq{...}` = green (high / favorable) and `\lowq{...}` = maroon/red (sub-optimal). Mapping by cell:

| Algorithm | SNR | logSNR | \(\overline{\sigma}_{\textrm{res.}}\) | *I* | *t*<sub>tot.</sub> |
|-----------|-----|--------|---------------------------------------|-----|-------------------|
| CLEAN | low | low | high | high | high |
| uSARA | high | high | high | low | low |
| AIRI | high | high | high | low | low |
| U-Net (both) | low | low | low | high | high |
| R2D2-Net<sup>3L</sup> / <sup>6L</sup> (both) | high | high | high | high | high |
| R2D2 / R3D3<sup>3L</sup> / R3D3<sup>6L</sup> (all impl.) | high | high | high | high | high |

Uncolored in LaTeX: *t*<sub>dat.</sub>, *t*<sub>reg.</sub>, *n*<sub>cpu</sub>, *n*<sub>gpu</sub>, programming language (and some `-` placeholders).
