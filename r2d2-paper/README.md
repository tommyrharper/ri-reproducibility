# R2D2 paper (arXiv:2403.05452v3)

**Title:** The R2D2 deep neural network series paradigm for fast precision imaging in radio astronomy  
**Authors:** Amir Aghabiglou, Chung San Chu, Arwa Dabbech, Yves Wiaux  
**arXiv:** [2403.05452](https://arxiv.org/abs/2403.05452) (v3, 01 May 2024)  
**HTML (experimental):** https://arxiv.org/html/2403.05452v3

## Layout

| Path | What |
|------|------|
| `r2d2.pdf` | Published PDF |
| `arXiv-2403.05452v3.tar.gz` | Original arXiv source tarball |
| `latex/` | Extracted LaTeX source (authoritative) |
| `R2D2_paper.md` | Agent-friendly markdown view of the paper body |

## Main LaTeX entrypoint

- **`latex/R2D2.tex`** — single main manuscript (no `\input`/`\include` of body sections)
- **`latex/R2D2.bib`** / **`latex/R2D2.bbl`** — bibliography
- **`latex/aastex631.cls`**, **`latex/aasjournal.bst`** — AASTeX class + BibTeX style
- **`latex/fig/`** — figures (~96 files; leave as-is)

## Build (optional)

From `latex/`, with a TeX distribution that provides AASTeX / revtex dependencies:

```bash
cd latex
pdflatex R2D2.tex
bibtex R2D2
pdflatex R2D2.tex
pdflatex R2D2.tex
```

A prebuilt PDF is already at `r2d2.pdf`; rebuilding is usually unnecessary for reading.

## What agents should read

1. **`R2D2_paper.md`** — fastest greppable overview (sections, abstract, body text; math/citations simplified; figure/table bodies replaced by captions).
2. **`latex/R2D2.tex`** — authoritative equations, macros, and exact wording.
3. **`latex/R2D2.bib`** — reference keys/details.
4. **`r2d2.pdf`** — only when figures/layout matter.

## Paper structure (sections)

1. Introduction  
2. R2D2 algorithm — data model; algorithmic structure; DNN series training; normalization; incarnations (R2D2 / R3D3)  
3. Training approach — U-Net; ground-truth database; VLA-specific training; implementation; cost  
4. Simulations and results — benchmarks; metrics; generic/specific experiments; compute  
5. Conclusions  
6. Data Availability (+ software, acknowledgements, bibliography)

## Notes

- Source is self-contained in one `.tex` file; figures live under `latex/fig/`.
- `R2D2_paper.md` is a derived convenience view, not a substitute for the LaTeX/PDF when quoting equations.
