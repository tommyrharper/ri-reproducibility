# reports/

Host-side output of the nested-sampling searches. Everything in here
except this file and `manifests/.gitkeep` is generated and gitignored.

- `nested-sampling-report/` - built by `make nested-sampling-report` from
  `results/nested-sampling-poc/*/poc-summary.json`. One page per run
  (PolyChord log(Z), searched parameters and metrics per evaluation,
  reconstructions rendered next to the truth, likelihood plot, and a
  collapsible per-stage profiling table when the run recorded one), plus
  an `index.html` listing and linking to every run.

  Rendering a page reads that run's FITS output, so pages that are already
  up to date are skipped and only new runs are built. Each page is stamped
  with the report generator's version: `UPGRADE=1` rebuilds the pages an
  older version wrote, `FORCE=1` rebuilds them all, `RUN=<run>` rebuilds
  one, `LAST=N` limits to the newest N runs. The report runs inside the
  r2d2 image (its astropy + matplotlib + anesthetic) via
  `scripts/generate-report.sh`, same approach as `scripts/plot-fits.sh`,
  so no host Python environment is needed. Open the index in a browser.
  See `docs/nested-sampling.md` for the full description.

- `merged-r2d2-wsclean-*.png` - the merged and side-by-side failure-score
  figures written by `scripts/plot-merged-likelihood-compare.py`, which is
  where `latex/notes.tex` includes them from.

- `manifests/` - one JSON file per run, written by
  `scripts/record-environment.sh`: repo revision, image digests, host
  CPU/memory, config checksums, and the exact command. Gitignored except
  for the directory itself; they are metadata (small, text) so *could* be
  committed, but are treated as run-local output, consistent with
  `results/`. `git add -f` a specific one if you want it version
  controlled.
