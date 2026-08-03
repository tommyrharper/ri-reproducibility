# data/

Bind-mounted into both containers at `/data`. Nothing in this directory
(other than this file) is tracked by Git.

Expected contents, added by you as experiments require them:

- Measurement Sets (`*.ms/`) for WSClean runs beyond the built-in smoke
  test fixture (which is downloaded on demand into a Docker volume by
  `scripts/smoke-test-wsclean.sh`, not placed here).
- R2D2 `.mat` measurement files, if you generate your own beyond the
  example bundled with R2D2-RI itself (`data/data_3c353.mat` inside the
  cloned repo - see README.md "Reproducibility limitations" for why that
  ~100 MB file ends up baked into the R2D2 image rather than here).
- Ground-truth FITS images used for evaluation metrics.

Large files placed here should be documented (source, checksum, license)
in `benchmarks/REPRODUCTION_PLAN.md` or a sibling note, since Git will
never see them.
