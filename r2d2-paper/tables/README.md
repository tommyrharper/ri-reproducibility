# R2D2 paper tables (extracted)

Agent-friendly markdown extracts of **all** numerical tables from `latex/R2D2.tex` (arXiv:2403.05452v3).

**Source of truth:** `latex/R2D2.tex` — do not invent numbers; quote from these files or the LaTeX.

## Index

| Label | File | Caption (short) | LaTeX lines |
|-------|------|-----------------|-------------|
| `table:training_cost` | [`table_training_cost.md`](table_training_cost.md) | Training compute (GPU/CPU hr, epochs, params) for U-Net, R2D2-Net, R2D2, R3D3 | 463–479 |
| `table:results` | [`table_results.md`](table_results.md) | Generic experiment: SNR/logSNR/residual + timing over 200 problems | 543–630 |
| `table:testset_config` | [`table_testset_config.md`](table_testset_config.md) | Experiments I–IV image/observation settings | 646–662 |

## Coverage

A search of `R2D2.tex` found exactly three float tables:

- 1× `table` → `table:training_cost`
- 1× `rotatetable*` wrapping `deluxetable*` → `table:results`
- 1× `table` → `table:testset_config`

No other `\begin{table}`, `\begin{deluxetable}`, or `\begin{rotatetable}` environments are present.

## Guidance for agents

1. Prefer these markdown files for quantitative claims (metrics, timings, experiment configs).
2. If a number is critical (e.g. for reproducibility claims), double-check the cited cell against `latex/R2D2.tex`.
3. Figures remain outside this folder; see `latex/fig/` and captions in `R2D2_paper.md`.
