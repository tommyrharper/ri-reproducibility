# Reproduction plan

Scope, per `CLAUDE.md`: **one modest verified target per project** for
this first pass - not a survey of every figure in every paper. Broader
figure reproduction is future work, tracked as open items below.

## R2D2-RI target

### Identifying the right paper for the v2.0 repository

R2D2-RI's `README.md` (pinned commit `22669259f770a0cb3a3191a5d3e8dbad4ae5a70c`)
lists four papers. Of these, **[1] is the one that corresponds to the
pinned v2.0 code**: the README explicitly says "This repository
corresponds to the latest version of the R2D2 algorithm (v2.0)", and
paper [1] is the one that introduces the v2.0-era contributions this
repository actually implements (the `T2` generalized training set, the
data-fidelity convergence/pruning criterion visible as `prune` and
`sigma_res_tol` in `config/imaging/*.yaml`, and the U-WDSR architecture
selectable via `architecture: uwdsr`). Papers [2]-[4] describe earlier
work (uncertainty quantification methodology, the original R2D2 series,
and the Cygnus A application) that predate or are subsumed by [1].

- **Paper**: Aghabiglou, A., Chu, C. S., Tang, C., Dabbech, A. & Wiaux,
  Y., "Towards a robust R2D2 paradigm for radio-interferometric imaging:
  revisiting DNN training and architecture", ApJS 280, 63 (2025).
  arXiv:2503.02554, DOI (from ADS bibcode 2025ApJS..280...63A).
  Prior summary on file: `../r2d2-citations-review/papers/2025ApJS..280...63A.md`.
- **Figure/Table**: Table 4 (benchmark comparison, test set E2) - SNR and
  logSNR for CLEAN, uSARA, AIRI, standalone U-Net/U-WDSR, and
  `R2D2_A1,T2` / `R2D2_A2,T2` (the two DNN series shipped as
  `config/imaging/R2D2_U-Net.yaml` / `R2D2_U-WDSR.yaml` in this repo's
  vendored R2D2-RI checkout).
- **Purpose**: sanity-check that this environment's R2D2 imaging path
  (finufft measurement operator, pretrained checkpoints, `imager.py`
  driver) reproduces fidelity numbers in the same ballpark as the
  published ones - not bit-exact reproduction of Table 4 itself, which
  reports means over 200 independent simulated inverse problems.
- **Required input dataset**: one of the paper's test images. **3C 353
  is one of the four source images** the paper's test sets (E1/E2) were
  simulated from (per the paper summary), and `data/data_3c353.mat` +
  `data/3c353_gdth.fits` ship inside the pinned R2D2-RI repository
  itself - likely the exact artefact backing that part of the paper's
  test set, though this is inferred (not confirmed against the paper's
  supplementary data) and marked as an open ambiguity below.
- **Required checkpoint**: `R2D2_A1_T2` (U-Net, `config/r2d2/R2D2_U-Net.yaml`)
  or `R2D2_A2_T2` (U-WDSR, `config/r2d2/R2D2_U-WDSR.yaml`), one
  realisation each is sufficient for a point estimate (`ckpt_realisations: 1`);
  all 5 are needed to reproduce the paper's epistemic-uncertainty (MRU)
  numbers. See `checkpoints/README.md` for the manual download (Cloudflare
  blocks automation).
- **Competing methods in the paper**: uSARA, AIRI (both BASPLib,
  GPU-accelerated Python, not part of this environment), multiscale
  CLEAN via WSClean.
- **Parameters**: 512x512 image, VLA-simulated visibilities, Briggs
  weighting robustness varied in the paper (this repo's config pins
  `weight_type: briggs`, `natural_weight: True` - see open ambiguity
  below on whether that matches the specific ρ_br the 3C 353 example was
  generated at).
- **Hardware reported by the paper**: not verified in this session -
  paper reports GPU-hours for training (Table 2) and per-iteration
  timings by NUFFT backend (Section title implies TorchKbNufft, PyNUFFT,
  FINUFFT, PSF-approximation compared), but the exact GPU/CPU model used
  is an open item to check against the paper text directly.
- **Software revision reported by paper**: not stated in the summary on
  file; the paper predates or coincides with this repo's pinned commit,
  exact correspondence not verified.
- **Evaluation metric**: SNR (dB) and logSNR (dB) against the ground
  truth FITS, computed by R2D2-RI's own `utils.snr` / `utils.to_log`
  (already exercised, without ground truth comparison printed, by
  `scripts/smoke-test-r2d2.sh` stage 3). RDR (residual-to-dirty-image
  ratio) is the paper's data-fidelity metric; whether `imager.py` reports
  it directly needs checking once a real inference run is possible.
- **Expected output**: `SNR ~= 30.0 dB` (`R2D2_A1,T2`) or
  `SNR ~= 31.2 dB` (`R2D2_A2,T2`), per Table 4's *mean* over 200 problems
  - a single 3C 353 run should land in the same broad range but will not
  match exactly (see "Unresolved ambiguities").
- **Publicly available artefacts**: MS/`.mat` example - yes (bundled).
  Ground truth - yes (bundled). Checkpoints - yes, but gated behind a
  Cloudflare-protected manual download (see `checkpoints/README.md`).
  Exact 3C 353 simulation parameters (noise realisation, ρ_br, super
  resolution used in the paper's own test set vs. the bundled example)
  - not confirmed as identical.
- **Current reproduction status**: **First baseline run complete**
  (2026-08-03, `R2D2_A1_T2` Realisation1, `ckpt_realisations: 1`, 512x512,
  25/25 iterations, CPU-only on an Apple M1 Max). Manifest:
  `benchmarks/manifests/r2d2-20260803T201519Z.json`; full log and output
  FITS: `results/benchmark-r2d2-3c353-unet-A1/`.
  - **SNR: 37.4454 dB, logSNR: 29.8406 dB** on the single bundled 3C 353
    example, vs. the paper's Table 4 mean of **~30.0 dB** for
    `R2D2_A1,T2` over 200 simulated test problems. In the same broad
    range and, if anything, higher - plausible for one image rather than
    a 200-problem mean, but not yet a controlled comparison: none of the
    three "Unresolved ambiguities" below have been checked against the
    paper text, so this is a **baseline execution**, not a verified
    **reproduction**, per the distinction in `benchmarks/README.md`.
  - Not yet done: the U-WDSR series (`R2D2_A2,T2`, paper mean ~31.2 dB),
    and resolving the ambiguities below before elevating this from
    "plausible ballpark" to "reproduction."
- **Unresolved ambiguities** (do not guess - resolve before treating a
  future run as a "reproduction" rather than a "baseline execution"):
  1. Whether `data/data_3c353.mat` is literally the same visibility
     realisation used in the paper's Table 4/3C 353 test problems, or a
     separately generated example with the same source image.
  2. The exact ρ_br (Briggs robustness) and super-resolution factor the
     paper used for its 3C 353 test cases, vs. this repo's
     `data/README.md`-recorded values (super resolution 1.52, robustness
     2.5303e-01) for the bundled file - these may or may not be the same
     pairing the paper reports.
  3. Which NUFFT backend (of the paper's four compared) its headline
     Table 4 numbers use, vs. this environment's default `finufft`.

## WSClean target

- **Paper**: Offringa, A. R. et al., "WSClean: an implementation of a
  fast, generic wide-field imager for radio astronomy", MNRAS, 2014.
  DOI: 10.1093/mnras/stu1368, arXiv:1407.1943. (Exact volume/page numbers
  not independently verified in this session - use the DOI.)
- **Figure/Table**: not selected for reproduction in this pass. The
  paper's benchmarks (w-stacking vs. w-projection speed, "order of
  magnitude faster") are run on **Murchison Widefield Array (MWA)**
  observations, which are not publicly bundled at a size suitable for a
  first baseline, and reproducing them is explicitly out of scope for
  "focus primarily on building a clean, reproducible foundation" per
  this repo's brief.
- **Chosen initial target instead**: an installation-verification +
  baseline-execution smoke test (not a paper-figure reproduction) using
  the tiny, checksummed, ASTRON-hosted `JVLA-MultiBand-S1_C5-minimal.ms`
  fixture that WSClean's own CI test suite uses (see
  `scripts/smoke-test-wsclean.sh`, `tests/CMakeLists.txt` in the pinned
  WSClean source). This exercises the full gridding -> iFFT ->
  deconvolution pipeline end-to-end on real (if minimal) VLA visibility
  data, without requiring an external, uncontrolled dataset download.
- **Required input dataset**: `JVLA-MultiBand-S1_C5-minimal.ms.tar.bz2`,
  SHA256 `7c8d41b5ff59c8736b1223e6b855a96e410f27fb4be05d179c8292fdb78cdc7e`
  (matches the checksum WSClean's own build system pins for the same
  file), fetched automatically by the smoke test script.
- **Parameters**: see `config/wsclean/smoke-test.args` (256x256, 1asec
  scale, niter 1000, mgain 0.8, auto-threshold 3, 1 thread).
- **Evaluation**: `wsclean --version` succeeds, and imaging produces the
  expected `-image.fits` / `-residual.fits` / `-dirty.fits` /
  `-psf.fits` outputs without error - a runnability/baseline check, not
  a fidelity metric, since there is no published ground truth for this
  CI fixture.
- **Current reproduction status**: pending first full build (see root
  README "Verification and acceptance criteria" for the latest run).
- **Next concrete step toward the first real WSClean benchmark**: pick
  one MWA or LOFAR public archive dataset small enough to bind-mount
  (order 100 MB-1 GB), and use it to attempt a modest reproduction of
  the Offringa et al. 2014 w-stacking vs. w-projection timing comparison
  under this environment's Docker/Apple-Silicon constraints (documented
  as *not* representative of the paper's native-hardware numbers - see
  root README "Docker Desktop on macOS" limitations).

## Explicitly out of scope for this pass

Per `CLAUDE.md`: "do not attempt to reproduce every paper figure in the
first pass." The following are noted but deliberately deferred:

- Any of the other 32 papers surveyed in `../r2d2-citations-review/`
  that cite R2D2 but are not the v2.0 paper itself.
- R2D2's epistemic-uncertainty figures (Fig. 3-5 of the target paper),
  which need all 5 checkpoint realisations (~26 GB) rather than 1.
- Cygnus A real-data results (Section 5 of the target paper) - a
  different, non-bundled dataset.
- Any direct WSClean-vs-R2D2 runtime/quality comparison - see
  `benchmarks/README.md` "What benchmarking means here vs. what it
  doesn't."
