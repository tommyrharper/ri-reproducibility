# benchmarks/

Minimal scaffolding only - a full benchmarking framework is deliberately
*not* built until both WSClean and R2D2-RI have verified smoke tests and
at least one baseline run each (see `REPRODUCTION_PLAN.md`).

- `REPRODUCTION_PLAN.md` - which paper figures/tables this environment
  targets reproducing, what's required, and current status.
- `manifests/` - one JSON file per experiment run, written by
  `scripts/record-environment.sh`. Gitignored except for this directory
  itself; manifests are metadata (small, text) so *could* be committed,
  but are treated as run-local outputs here, consistent with
  `results/`. If you want specific manifests version-controlled (e.g.
  for a paper appendix), copy them out and `git add -f`.
- `scripts/` - benchmark drivers, added once there is more than one
  thing to benchmark. Empty for now.
- `report.html` - generated, gitignored; run `make benchmark-report` to
  build it from whatever is currently in `manifests/` plus any
  `results/nested-sampling-poc/*/poc-summary.json` nested-sampling PoC runs.
  One card per manifest (or per PoC run): environment/provenance, the
  `experiment.results` metrics table (if the manifest has one - see the
  `r2d2-*.json` example), and the run's output FITS files rendered inline.
  Nested-sampling cards also show PolyChord log(Z) evidence, a card grid of
  per-evaluation reconstruction/truth image pairs (see
  `docs/nested-sampling.md` for detail), and a best-effort posterior plot.
  Uses the r2d2
  image's own astropy + matplotlib + anesthetic
  (`scripts/generate-benchmark-report.sh`, same approach as
  `scripts/plot-fits.sh`), so no host Python environment is needed; open
  the file directly in a browser afterward.

## What "benchmarking" means here vs. what it doesn't

WSClean and R2D2-RI solve the same underlying inverse problem
(visibilities -> image) but are not directly comparable on wall-clock
time alone: different pipelines, preprocessing, hardware paths (WSClean
is multithreaded CPU C++; R2D2 is a DNN series that only becomes
competitive on GPU), and output definitions (WSClean produces a CLEAN
restored image + residual; R2D2 produces a DNN-series image estimate +
residual, with optional epistemic-uncertainty maps). Any comparison
must control for and report: image size, pixel scale, weighting scheme,
input data, hardware, and precision - see `REPRODUCTION_PLAN.md` and the
root README's benchmarking section for the full list of axes this
environment is designed to keep separable.
