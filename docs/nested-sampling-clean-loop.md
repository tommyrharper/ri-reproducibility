# The clean loop: the last 20% on this host, and what it costs

The [evaluation-budget doc](nested-sampling-evaluation-budget.md) ends by
saying that 69% of an evaluation is WSClean's clean loop and that no
*result-preserving* WSClean flag recovers any of it. This doc goes into that
69% and reports the one lever that does: **`-mgain 0.9` in place of `-mgain
0.8` is worth +20% evaluations per second end to end**, measured over three
interleaved pairs of real searches, and it changes the default `total_rms_jy`
objective by 8e-8 in the median and 9e-5 in the worst of 600 real
evaluations.

It is **not shipped**. `-mgain` is part of the experiment definition all 45
archived runs were scored under, and it is not free for every metric (see
[the caveat](#the-catch-it-is-only-result-preserving-for-total_rms_jy)).
Taking it is `./ri search wsclean --mgain 0.9` (`NS_WSCLEAN_MGAIN`), which
needs no rebuild and is carried through a resume by the run's `run.env`; every
run's `summary.json` records the value it ran under in
`wsclean_fixed_hyperparameters.mgain`. `DEFAULT_WSCLEAN_MGAIN` in
`scripts/lib/nested_sampling/common.py` is what the flag defaults to, and
moving *that* (plus `scripts/build.sh polychord`) is how the default itself
would change.

## Why the clean loop costs what it does

The clean loop is a **count of gridding passes**, not arithmetic. One
evaluation's `wsclean.stdout.log` shows the shape:

```
 == Constructing PSF ==            <- gridding pass 1
 == Constructing image ==          <- gridding pass 2 (the dirty image)
 == Deconvolving (1) ==
Next major iteration at: 200 mJy
Performed 16 iterations in total, 16 in this major iteration
 == Converting model image to visibilities ==   <- degridding pass
 == Constructing image ==                       <- gridding pass 3
 == Deconvolving (2) ==
...
6 major iterations were performed.
```

With `-mgain 0.8` each minor loop stops once the peak has come down by 80%,
which on this problem is ~16 minor iterations; `-niter 100` therefore buys
`100 / 16` = ~6.5 major cycles, and each of those is one gridding plus one
degridding pass over 4140 visibilities into 128x128. **`-mgain` sets the
depth of a minor loop, `-niter` sets the total, and their ratio is the number
of gridder invocations an evaluation pays for.** Each invocation costs
~16 ms at production concurrency and is dominated by ducc0's per-pass kernel
setup rather than by the visibilities (which is also why
[`-wgridder-accuracy 1e-2` was worth 13.8%](nested-sampling-evaluation-budget.md#what-the-remaining-wsclean-flags-are-worth)
- it shrinks the kernel, not the arithmetic).

Raising `-mgain` therefore buys throughput by spending the same 100 iterations
in fewer, deeper minor loops.

## The measurement

600 Measurement Sets and their recorded `commands.wsclean` argv, taken every
third evaluation from a 2070-evaluation `--nlive 50 --seed 4242` search run
with `--keep-measurement-sets`. Every arm images the same 600 sets with the
same argv except for `-mgain`, all arms interleaved in one 19-way
`xargs -P 19` inside one `wsclean` container, each command timed by its own
`/usr/bin/time -f %e`. Two full repeats. A duplicated `0.8` arm is carried
through as the null pair.

| arm | major cycles | mean wall, run 1 | mean wall, run 2 | speedup |
|---|---:|---:|---:|---:|
| `-mgain 0.8` (baseline) | 6.54 | 243.8 ms | 258.3 ms | - |
| `-mgain 0.8` (null pair) | 6.54 | 245.7 ms | 256.4 ms | 0.99 / 1.01 |
| `-mgain 0.85` | 5.58 | 223.1 ms | 235.7 ms | **1.093 / 1.096** |
| `-mgain 0.9` | 4.69 | 208.1 ms | 215.4 ms | **1.172 / 1.199** |
| `-mgain 0.95` | 1.19 | 128.7 ms | 133.1 ms | 1.894 / 1.941 |

The null pair reads 0.99 and 1.01, so this rig resolves ~1%.

`0.95` is not a candidate: `-auto-threshold` recomputes the 3-sigma stopping
level from the *current* residual, and after a single deep minor loop that
residual is still full of sidelobes, so the threshold lands three orders of
magnitude high and the clean stops after one major cycle. That is why the
speedup is so large and why the science is not comparable - `snr` moves by
49% in the median.

### End to end

Three interleaved pairs of real 70-second searches, `--nlive 100
--mpi-procs 20 --max-ndead -1`, built as two `polychord` images differing only
in the constant, with the first 20 seconds of each run discarded (the
[power-limit burst window](nested-sampling-power-limit.md)):

| pair | 0.8 evals/s | 0.9 evals/s | ratio | 0.8 binary | 0.9 binary |
|---|---:|---:|---:|---:|---:|
| 1 | 59.24 | 72.96 | 1.232 | 282.0 ms | 223.1 ms |
| 2 | 60.69 | 69.33 | 1.142 | 273.1 ms | 237.4 ms |
| 3 | 58.67 | 72.60 | 1.237 | 273.1 ms | 221.9 ms |

Geometric mean **1.203**, i.e. +20% evaluations per second. (Pair 2's `0.9`
arm is the outlier in both columns together, which is what host contention
looks like; the sign and rough size agree in all three.)

## The catch: it is only result-preserving for `total_rms_jy`

Every arm's five FITS images were re-scored through this repo's own
`compute_image_metrics`, against the `0.8` arm, on the same 600 evaluations.
Relative differences:

| metric | `0.85` med / p95 / max | `0.9` med / p95 / max |
|---|---|---|
| `total_rms_jy` (the default objective) | 6.7e-8 / 1.3e-5 / 1.6e-4 | 7.9e-8 / 9.9e-6 / 9.3e-5 |
| `relative_l2_error` | 6.7e-8 / 1.3e-5 / 1.6e-4 | 7.9e-8 / 9.9e-6 / 9.3e-5 |
| `snr`, `off_source_rms_jy` | 3.5e-7 / 5.2e-4 / 9.1e-3 | 4.4e-7 / 1.0e-3 / 5.4e-3 |
| `peak_jy_per_beam` | 1.2e-7 / 8.6e-6 / 1.2e-4 | 1.2e-7 / 6.4e-6 / 7.5e-5 |
| `sigma_res` | 1.5e-3 / 2.0e-2 / 3.8e-2 | 2.6e-3 / 3.3e-2 / 7.0e-2 |
| `peak_flux_abs_error_jy` | 5.2e-3 / 1.0e-1 / 9.6e-1 | 5.3e-3 / 1.4e-1 / 6.7e-1 |

Read the last two rows before taking the win. `peak_flux_abs_error_jy` is
`|peak - source_flux|`, a difference of two numbers that agree to five
figures, so a 7.5e-5 change in `peak_jy_per_beam` is a 67% change in it; the
same cancellation drives `sigma_res`. Both are selectable with `--metric`, and
a search run against either of them would be a materially different search
under `-mgain 0.9`. Against the default `total_rms_jy` - which is what every
archived run used - the objective is unchanged to 1e-4 in the worst case and
1e-7 typically.

The worst differences sit at the *low* end of the parameter space, where the
metric is noise-dominated anyway: over the 60 highest-`dynamic_range`
evaluations in the corpus (up to 9.8e5) the objective agrees to 2.2e-7.

## What this does not fix

`-mgain` does not make the clean loop converge - 83% of evaluations at 0.8 and
75% at 0.9 stop because `-niter 100` ran out, not because the threshold was
reached. Both settings are "100 CLEAN iterations", spent differently. If a
future change wants the deconvolution to actually converge, that is a `-niter`
decision and it costs throughput rather than buying it.

## That `-niter` change has now shipped

`NS_WSCLEAN_NITER` defaults to 1000. The prediction above was right about the
diagnosis and wrong about the price: it costs almost nothing.

Measured over one interleaved pair on this host, same seed, `--nlive 8
--num-repeats 2 --max-ndead 40 --mpi-procs 4`:

| `-niter` | Evaluations | Wall | Evaluations/s | Stopped on the cap | Median iterations |
| --- | --- | --- | --- | --- | --- |
| 100 | 291 | 6.8 s | 42.9 | 69.1% | 100 |
| 1000 | 300 | 7.1 s | 42.2 | **0%** | 122 |

**1.6% fewer evaluations per second, and nothing is truncated any more.** The
cap was never expensive to lift because deconvolution is a small part of an
evaluation and the median evaluation only wants ~122 iterations - it was
sitting just above 100, which is exactly where a binding cap does most damage
for least benefit. Runs across the wider archived parameter space wanted a
median of 177.

This is one pair on one host, not the interleaved triple the `-mgain`
measurement above used; treat 1.6% as "small", not as a precise figure.

Why it matters beyond throughput: at `-niter 100`, 77-84% of evaluations across
five archived runs scored the residual after 100 components rather than what
CLEAN converges to, and the worst 200 evaluations of a run were 90% capped
against an 80% base rate - so the cap was shaping the failure map, not just the
cost. Each record now carries `clean_stop_reason`, `clean_iterations` and
`clean_major_iterations`, so this is answerable from `summary.json` instead of
from logs that no longer exist.

Current flag-based measurement procedure lives in the
[evaluation-floor guide](nested-sampling-evaluation-floor.md#-mgain-measured-again-as-a-flag).
