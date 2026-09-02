# Proposed additions to the nested-sampling parameter space

The searched space today is five dimensions (`[[parameter_space]]` in
`defaults.toml`): `log10_dynamic_range`, `observation_minutes`,
`channel_count`, `start_frequency_hz` (band-constrained), `channel_width_hz`.
Everything else that shapes an evaluation is pinned in `cube_to_params()` or in
the runners.

This document proposes what to add next, ranked by failure-mode value per unit
of plumbing. WSClean's cell size following the sampled frequency instead of a
hard-coded `-scale` - the confound that had to be fixed before any of this was
worth measuring - is implemented; see "Cell size" in `docs/nested-sampling.md`.

Nothing here is implemented beyond section 1; it is a plan, not a changelog -
section 1's own writeup below is kept as the design rationale, not updated
into a changelog entry. Every dimension `[[parameter_space]]` takes `enabled`
(default true) for pinning it back out without deleting it - see "Toggling
dimensions on and off" in docs/nested-sampling.md - which is what makes
landing one dimension at a time here safe: the next section's addition does
not have to also be the next section's permanent commitment.

## 1. Source offset from the phase centre - cheapest, highest value

`cube_to_params()` pins `source_l_arcsec = source_m_arcsec = 0.0`. That is the
single most benign configuration available, and it disables the physics the
existing spectral and temporal dimensions are supposed to probe:

- **No bandwidth smearing.** Smearing scales with offset x fractional
  bandwidth. At the phase centre it is exactly zero, so `channel_width_hz` and
  `channel_count` currently buy visibility count and nothing else.
- **No time smearing.** Same argument for `observation_minutes`.
- **No w-term, no gridding-kernel error.** `point_source_forest.py` skips
  K-Jones entirely at `l == m == 0` (see the comment there), and the source
  lands exactly on the reference pixel, so pixel-interpolation error is zero.
- **R2D2's learned prior is centred-source-friendly.** A source away from the
  centre is a distribution shift the network was not asked about.

Proposal: one dimension, `source_offset_fraction`, 0.0 to 0.35 of the image
half-width, converted to `l`/`m` at a fixed non-axis-aligned position angle
(e.g. 30 deg, to avoid the symmetries of a purely horizontal or vertical
offset). One dimension rather than separate `l` and `m`: the array's response
is close enough to radially symmetric that a second angular dimension mostly
buys cube volume.

Plumbing:

- `cube_to_params()`: derive `source_l_arcsec` / `source_m_arcsec` from the
  fraction and the image geometry. The simulator CLI flags already exist.
- `compute_image_metrics()` in `common.py` places truth at `CRPIX` and masks
  off-source with a radius-5 disc around it (lines ~522-533). It has to take
  the offset and put truth at the right pixel, otherwise every off-centre
  evaluation scores as a total failure for a trivial reason.
- No MS skeleton cache impact: the offset only changes the MeqTrees predict.

Caveat: `ms_to_r2d2_mat.py` writes only `u` and `v` - `w` is dropped, so R2D2
is a coplanar 2D operator while WSClean is not. Keep offsets inside the
small-field regime, or the comparison becomes apples to oranges for a reason
that has nothing to do with either algorithm.

## 2. Integration time - implemented, shipped disabled

`DEFAULT_INTEGRATION_SECONDS = 120.0` was fixed, so `observation_minutes` of
0.3 to 20 yielded **1 to 10 time samples**. The time axis of the search was
that short. Total observing time and visibility count were also welded
together: there was no way to ask for "long track, sparse sampling" versus
"short track, dense sampling", which are very different uv-coverage regimes.

Now `integration_seconds`, `kind = "choice"` over `[10, 30, 60, 120, 300]`,
`enabled = false` with `default = 120`. Discrete, not continuous, because
`StepTime` is inside the makems skeleton cache key. A `choice` dimension is
new machinery this section justified: what has to be bounded is the count of
distinct skeletons, and `kind = "integer"` over a seconds range would reach
thousands of dump times rather than five.

The shape-count formula above undercounts, because the reachable `NTimes`
range itself grows as the dump time shrinks - the true count is a sum over the
list, not a product. Measured: the baked cache is 80 shapes and 87 MB today;
these five values reach 193 `(NTimes, dump time)` pairs, so 1544 shapes and
~5 GB. That lands in the sidecar's writable layer, not its 512 MB `/dev/shm`,
and is built lazily at ~0.05 s a shape - so it costs disk and a few minutes
spread across a run rather than failing, but it is why this ships disabled.
`prebuild_skeletons()` still bakes at 120 s only; extend it to iterate the
list if the lazy first-use cost shows up in a real run.

Pairs with 1: time smearing needs both a long dump and an off-centre source.
It is also the *only* way to reach time smearing - smearing goes as the dump
time times the source's offset in beams, so `observation_minutes` cannot reach
it however long the track. Enabling this without `source_offset_fraction` (or
the cartesian pair) buys visibility count and uv sampling density only.

## 3. Calibration error (antenna-based complex gain corruption)

The dominant real-world failure mode for radio-interferometric imaging, and
completely absent: visibilities are currently perfectly calibrated, so both
imagers see the ideal measurement equation. R2D2 in particular is trained on
calibrated data, so this is where a learned reconstructor should degrade
sharply and non-gracefully.

Proposal: `log10_gain_error_sigma`, a per-antenna per-time-slot complex gain
with amplitude and phase jitter, applied where the thermal noise is already
applied (`simulate_point_source_ms.py`, around line 432 - it is a numpy block
over `DATA`). Roughly ten lines: draw per-antenna gains, index by
`ANTENNA1`/`ANTENNA2`, multiply.

## 4. Non-Gaussian noise / low-level RFI

Same numpy block, same cost. The noise model today is exactly circular Gaussian
at `source_flux_jy / dynamic_range`, which is what R2D2's training assumed.

Proposal: `outlier_fraction` (fraction of visibilities replaced by a
high-amplitude draw), 0 to a few percent, log-scaled. Deconvolution algorithms
and learned reconstructors fail differently here - CLEAN chases the outlier
into a spurious component, a network may absorb or hallucinate it - which is
precisely the kind of divergence this repo exists to find.

## 5. Declination of the phase centre - implemented, shipped disabled

Was pinned at `Declination=65.0.0` in the makems config - near-optimal for the
VLA, giving a full circular uv track and a near-circular PSF. Low declination
foreshortens the array, produces a highly elliptical beam and much worse
sidelobes, and it is a routine observing condition rather than an exotic one.

Now `declination_deg`, `kind = "integer"` over -30 to 80, `enabled = false`
with `default = 65`. Discrete for the skeleton-cache reason: `Declination` is
part of the makems config, and only `StartFreq`/`StepFreq` are excluded from
the cache key, so a continuous dimension would build a fresh skeleton per
evaluation. Whole degrees rather than a hand-picked grid because
`kind = "integer"` already exists and a six-value list would be new machinery
for the same bound.

It is disabled because the skeletons baked into the MeqTrees image
(`--prebuild-skeletons`) are all at +65: enabling it means every other
declination pays one makems build per (NTimes, NFrequencies) shape it reaches,
published to the live cache and reused after that. Extending `prebuild_skeletons()`
to a declination list multiplies the prebuild shape count by that list's length,
so cost it before choosing one.

## 6. A second source, with a flux ratio

One source at the phase centre never exercises deconvolution proper: there are
no sidelobes of one source sitting on another. The classic dynamic-range
failure is a bright source whose sidelobes bury a faint one.

Proposal: `log10_flux_ratio` (0 to 4) plus a separation, with the second source
placed by the same offset machinery as 1. Needs a multi-source
`point_source_forest.py` (Meow supports it directly) and a `compute_image_metrics()`
truth image with two pixels set. Moderate plumbing, high value - this is where
`log10_dynamic_range` becomes about deconvolution rather than just SNR.

## 7. Source extent

`Meow.GaussianSource` instead of `PointSource`, with an FWHM parameter from
unresolved up to a few beams. Pure model mismatch: R2D2_A1 was trained on a
particular source class, and the metrics assume a point-like truth. Note the
metric has to change with it (a delta-function truth is wrong for an extended
source), so this is the most invasive of the physics proposals - defer it until
1-6 have been run.

## 8. Missing antennas / flagged fraction

Real arrays lose antennas and lose data to RFI flagging, which punches holes in
uv coverage. Cheap on the WSClean side; on the R2D2 side there is no FLAG
plumbing (`ms_to_r2d2_mat.py` writes no flag metadata), so it has to be done by
dropping rows before the bridge rather than by flagging. Proposal:
`flagged_fraction`, 0 to ~0.3, applied as whole-antenna dropouts rather than
random rows (holes, not thinning - random thinning is nearly equivalent to
shortening the observation, which axis 2 already covers).

## 9. Imager hyperparameters - separate run family, not this space

`PARAMETER_TEX_LABELS` in `common.py` already carries `wsclean_niter` and
`wsclean_auto_threshold`, so this was anticipated. Currently fixed: WSClean
`-niter 100`, `-auto-threshold 3.0`, `-mgain 0.8`, `-weight natural`; R2D2
`num_iter 25`, `super_resolution 1.5`.

Recommendation: **do not** mix these into the observation-parameter space. The
objective rewards bad reconstructions, so a sampler given `niter` will find
`niter = 1` and report it as a failure mode, which is true and useless. It also
makes the WSClean and R2D2 spaces different shapes, and the two runners
currently share one `cube_to_params()` and one `parameter_space`, so a shared
space would need per-runner conditioning that does not exist yet.

If hyperparameter sensitivity is wanted, run it as its own family with the
observation parameters frozen at a few representative points. The one exception
worth considering inside the main space is **`briggs_robust`** (-2 to +2),
because natural weighting is the most forgiving setting available and every
real observer turns that knob; it changes the PSF rather than crippling the
algorithm.

## Reparameterization notes (no new dimensions)

- **Fractional bandwidth.** `channel_count x channel_width_hz` caps total
  bandwidth at 12 MHz, which is 22% fractional at 54 MHz and 0.02% at 50 GHz.
  Multi-frequency effects are therefore only reachable in the lowest band.
  Sampling `fractional_bandwidth` and deriving `channel_width_hz` from the
  sampled centre frequency would make the dimension mean the same thing across
  the whole band list.
- **`observation_minutes` as hour angle.** What matters for uv coverage is the
  fraction of the source's track observed, which depends on declination. If 5
  lands, express the time axis as hour-angle coverage rather than raw minutes.

## Dimension budget

PolyChord cost scales with the number of dimensions: standard practice is
`nlive >= 25 x nDims` and `num_repeats` of 2-5 x nDims for a converged run
(the committed `NS_NLIVE = 8`, `NS_NUM_REPEATS = 2` are PoC values). Going from
5 to 10 dimensions is more than a doubling of evaluation count. Reordering or
inserting into `[[parameter_space]]` also invalidates existing chains, and
`merge-nested-sampling-runs.py` refuses to merge runs whose `parameter_space`
differs, so batch additions rather than dripping them in.

Suggested first batch, all cheap and all unlocking physics the existing
dimensions already pay for: **source offset (1)** - implemented - then
**integration time (2)** and **gain error (3)**.
