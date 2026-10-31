# Many-seed validation of `NS_ADOPT_CONTOUR` and `NS_EARLY_STOP`

Status: **complete.** 150 of 150 runs finished, none failed, none excluded. 3,105 of
the 5,000 core-hours were spent. The criteria were committed before any result was
read and were not changed. Nothing becomes a default: both settings stay opt-in, this
report recommends, and the user decides.

| arm | shape | verdict | wall vs stock | what it means |
| --- | --- | --- | --- | --- |
| `NS_EARLY_STOP=1` | A | **PASS** | -32.3% (-38.2 to -26.3) | no shift found anywhere, tail included |
| `NS_ADOPT_CONTOUR=1` | A | **FAIL** | -14.4% (-19.8 to -9.0) | one posterior statistic; tail volumes also shift |
| both | A | **FAIL** | -37.4% (-42.8 to -32.0) | three posterior statistics; tail volumes also shift |
| both | B | PASS, weak | -41.1% (-44.5 to -37.6) | this shape cannot show a posterior difference, and tail volumes shift more than in A |

**Recommendation.** Early stop alone is fit to use at production shape for CPU R2D2
on `total_rms_jy`. It was not tested alone at target size. Contour adoption should
stay off for this repo's purpose, with or without early stop: log(Z) does not notice
it, but it moves the high-RMS tail the search exists to map (see
[the tail](#the-tail-post-hoc)). The ~5 points of wall time it adds to early stop alone
are not worth that.

Both settings change results, not just speed. `NS_ADOPT_CONTOUR=1` lets a slice
adopt a newer contour mid-chain (patch 0001, round 312). `NS_EARLY_STOP=1` stops
an R2D2 evaluation predicted to land below the contour (round 355), so any wrong
stop approximates the objective. Earlier rounds compared one to three seeds; this
is the many-seed test.

## Design

**What runs.** One frozen tree (`4336fac`, worktree `../frozen-val`) and one image
set (`.cache/ri/sifs-r361`, whose `polychord.sif` has patches 0001-0002) for every
arm, so the stock arms use the same binaries with the settings off. CPU R2D2
(onnx), async, converged (`--max-ndead -1`), default ranks and threads.

| part | shape | seeds | arms |
| --- | --- | --- | --- |
| A, production | nlive 150, 15 repeats, 1 half icelake node (38 CPUs, 39 ranks) | 24: 9101-9124 | stock, stock-repeat, adopt, stop, adopt+stop |
| A, wrong stops | same | 6: 9101-9106 | shadow, adopt+shadow (`NS_EARLY_STOP=2`) |
| B, target size | nlive 500, 25 repeats, 8 half icelake nodes | 6: 9201-9206 | stock, stock-repeat, adopt+stop |

The seeds have not been used before. stock and stock-repeat have identical
settings and seed. They differ only through async timing, so stock-repeat is the
noise control. The shadow arms sit outside the pass rule: they exist only to count
wrong stops, which a stopping run cannot show because a stopped evaluation never
reaches a final RMS. Part B has no shadow arm; round 371's seed 344 is its only
wrong-stop reading (1 of 98,616).

**Scheduling.** Two Slurm arrays, `scripts/sampler-validation-job.sh`, throttled
(`%10` for A, `%2` for B). Task order is seed by seed with the arms shuffled within
each seed (`sampler_validation.py manifest`, fixed shuffle seed), so arms alternate
and share the same period and node population. Each task records its node list.
Run time limits are 80 minutes (A) and 40 minutes (B).

**Budget.** At most 5,000 CPU core-hours. Expected, from earlier runs at these
shapes: A ~22 minutes per stock run and ~15 per stopping run on 38 CPUs, ~1,700
core-hours; B ~20 and ~11 minutes on 304 CPUs, ~1,500 core-hours. If the deadline
or the budget cuts a part, it is reported with fewer seeds.

**Kept per run, before any deletion** (`docs/csd3-sampler-validation-runs.jsonl`):
log(Z), its error, ndead, nlike, wall, and the weighted posterior mean, sd and
5/50/95% quantiles of every parameter, plus the exit status, nodes and times.
Failed runs get a row too. Gzipped chains, stats and logs stay in `.probe/val/keep`
until the report is written.

**Analysis** (`sampler_validation.py analyse`), each arm against stock, paired by
seed, using only seeds where both runs finished:

- log(Z): per-seed difference over the combined quoted error, z. Mean, standard
  error, 95% t interval, exact sign test. The stock-repeat row is the null.
- Posterior: for each of mean, sd, q05, q50, q95 and each parameter, a two-sided
  paired sign-flip test on the per-seed differences (exact for 12 seeds or fewer,
  otherwise 4,000 random flips), Holm-corrected across parameters within each
  statistic, significant at 0.05.
- Cost: per-seed percentage change in wall and in evaluations, mean with 95% t
  interval.
- Calibration: per arm, scatter of log(Z) across seeds over the rms quoted error.

**Pass rule, per arm and part.** PASS needs all of:

1. |mean z| <= 0.25;
2. the 95% interval lies inside (-0.5, +0.5);
3. no more Holm-significant posterior statistics than stock-repeat shows;
4. log(Z) scatter across seeds no worse than stock's, tested one-sided at 0.05 by
   swapping arm labels within seeds. A larger sd that this test cannot tell from
   noise does not fail the condition.

FAIL if 3 or 4 fails, or if |mean z| > 0.25 with an interval excluding zero.
Anything else is INCONCLUSIVE. stock-repeat goes through the same rule as a check
on the rule itself. Runs excluded or cut off are listed, never dropped.

**Secondary, outside the pass rule, chosen before any result.** The best objective
found, and the objective at the dead-point contours ln X = -1 to -8. Reason: the
objective is an RMS of ~0.01-0.03, so L = exp(objective) is nearly flat, Z ~ 1 and
the posterior stays close to the prior. log(Z) and the posterior statistics are
therefore mostly a measure of the prior mean, and may be insensitive to errors in
the high-RMS tail, which is what this search exists to find. A pass on the rule
above must be read with that limit; these curves are where a tail error would show.
`analyse` reports each as a paired difference from stock with a 95% interval. That
table, added before any comparison was made, is descriptive only and has no verdict.

**Power, worked out before any result.** Condition 2 needs the t half-width plus
|mean z| under 0.5. With 24 seeds the half-width is 0.42 x sd(z), so sd(z) = 1 would
leave |mean z| <= 0.08, a test most honest arms would not pass. Seeds shared within
a pair should make sd(z) well under 1 (rounds 355-371 read 0.03-0.44 in |z|). With 6
seeds the half-width is 1.05 x sd(z), so Part B can pass only with sd(z) under ~0.48,
and its posterior tests have a smallest exact p of 2/64, which Holm across 9
parameters can never bring under 0.05. Part B is therefore a check on log(Z) and on
the tail, and cannot show a posterior difference.

## Results

Runs ran from 18:44 on 30 October to 05:01 on 31 October. Every run's record is in
[`csd3-sampler-validation-runs.jsonl`](csd3-sampler-validation-runs.jsonl), and the
tables below are `sampler_validation.py analyse` output, unedited. The first table of
each part is the pass rule.

**Failures: none.** All 150 tasks exited 0 and extracted, inside their time limits.
No task log holds a traceback, out-of-memory, cancellation, time-limit or EDQUOT line.
Nothing was excluded or cut off, and Part B reached its full 6 seeds.

### Part A, production shape

Part A: 132 of 132 recorded runs finished, 0 tasks have no record.

| arm | seeds | mean z | SE | 95% CI | sd z | sign (+/n, p) | scatter / stock's (p) | Holm hits | verdict |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | --- |
| stock-repeat | 24 | +0.073 | 0.051 | -0.032 to +0.179 | 0.251 | 16/24, 0.152 | 0.975 (0.696) | 0 | PASS |
| adopt | 24 | -0.017 | 0.061 | -0.144 to +0.110 | 0.300 | 12/24, 1 | 1.000 (0.49) | 1 | FAIL |
| stop | 24 | +0.003 | 0.048 | -0.096 to +0.102 | 0.235 | 10/24, 0.541 | 1.008 (0.431) | 0 | PASS |
| adopt+stop | 24 | -0.013 | 0.064 | -0.145 to +0.120 | 0.315 | 10/24, 0.541 | 1.020 (0.387) | 3 | FAIL |

| arm | wall vs stock (%, 95% CI) | evaluations vs stock (%, 95% CI) | Holm hits named |
| --- | --- | --- | --- |
| stock-repeat | -2.1 (-9.8 to +5.5) | +2.0 (-2.7 to +6.7) | none |
| adopt | -14.4 (-19.8 to -9.0) | -10.3 (-13.4 to -7.3) | source_l_pixels sd |
| stop | -32.3 (-38.2 to -26.3) | -0.2 (-4.6 to +4.2) | none |
| adopt+stop | -37.4 (-42.8 to -32.0) | -8.7 (-12.0 to -5.5) | source_l_pixels sd, source_m_pixels q95, source_m_pixels sd |

Calibration (scatter of log(Z) across seeds against the quoted error):

| arm | runs | mean log(Z) | scatter | rms quoted error | scatter / quoted (+/- approx) |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 24 | 0.017426 | 5.130e-04 | 3.986e-04 | 1.287 +/- 0.190 |
| stock-repeat | 24 | 0.017468 | 5.000e-04 | 3.959e-04 | 1.263 +/- 0.186 |
| adopt | 24 | 0.017418 | 5.132e-04 | 3.929e-04 | 1.306 +/- 0.193 |
| stop | 24 | 0.017429 | 5.171e-04 | 3.950e-04 | 1.309 +/- 0.193 |
| adopt+stop | 24 | 0.017421 | 5.232e-04 | 3.939e-04 | 1.328 +/- 0.196 |

Per arm, mean over seeds:

| arm | runs | log(Z) error | ndead | nlike | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 24 | 3.977e-04 | 1461 | 69374 | 1430 |
| stock-repeat | 24 | 3.954e-04 | 1460 | 70334 | 1371 |
| adopt | 24 | 3.925e-04 | 1286 | 61977 | 1209 |
| stop | 24 | 3.945e-04 | 1459 | 68806 | 945 |
| adopt+stop | 24 | 3.934e-04 | 1288 | 63002 | 877 |
| shadow | 6 | 3.871e-04 | 1472 | 71470 | 1524 |
| adopt+shadow | 6 | 3.878e-04 | 1285 | 63601 | 1244 |

Smallest Holm-adjusted p across parameters, per statistic:

| arm | mean | sd | q05 | q50 | q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock-repeat | 0.472 | 0.405 | 1 | 0.0855 | 1 |
| adopt | 0.643 | 0.018 | 0.081 | 0.497 | 0.0742 |
| stop | 0.151 | 0.466 | 1 | 0.22 | 1 |
| adopt+stop | 0.198 | 0.00675 | 0.0945 | 0.562 | 0.00225 |

Secondary, outside the pass rule: the tail, which is what the search is for.

| arm | runs | mean max objective | mean contour at ln X -4 | at -6 | at -8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 24 | 0.03342 | 0.02666 | 0.03055 | 0.03109 |
| stock-repeat | 24 | 0.03350 | 0.02670 | 0.03054 | 0.03110 |
| adopt | 24 | 0.03310 | 0.02789 | 0.03082 | 0.03138 |
| stop | 24 | 0.03325 | 0.02667 | 0.03056 | 0.03110 |
| adopt+stop | 24 | 0.03305 | 0.02776 | 0.03078 | 0.03134 |

Paired difference from stock (mean, 95% CI, * = excludes 0):

| point | stock-repeat | adopt | stop | adopt+stop |
| --- | --- | --- | --- | --- |
| max objective | +7.91e-05 (-3.23e-04 to +4.81e-04) | -3.20e-04 (-6.66e-04 to +2.65e-05) | -1.68e-04 (-4.96e-04 to +1.59e-04) | -3.72e-04 (-6.74e-04 to -7.01e-05)* |
| ln X -1 | -3.80e-05 (-1.27e-04 to +5.10e-05) | +5.98e-04 (+4.76e-04 to +7.21e-04)* | -6.18e-05 (-1.58e-04 to +3.45e-05) | +5.87e-04 (+4.67e-04 to +7.08e-04)* |
| ln X -2 | +6.57e-05 (-4.93e-05 to +1.81e-04) | +8.01e-04 (+6.67e-04 to +9.35e-04)* | +1.60e-05 (-8.98e-05 to +1.22e-04) | +8.74e-04 (+7.58e-04 to +9.89e-04)* |
| ln X -3 | +4.68e-05 (-8.02e-05 to +1.74e-04) | +9.72e-04 (+8.33e-04 to +1.11e-03)* | +3.45e-05 (-1.19e-04 to +1.88e-04) | +9.76e-04 (+8.43e-04 to +1.11e-03)* |
| ln X -4 | +3.08e-05 (-2.03e-04 to +2.64e-04) | +1.22e-03 (+9.56e-04 to +1.49e-03)* | +3.00e-06 (-2.36e-04 to +2.42e-04) | +1.10e-03 (+8.30e-04 to +1.36e-03)* |
| ln X -5 | +5.57e-05 (-1.97e-04 to +3.08e-04) | +8.89e-04 (+6.17e-04 to +1.16e-03)* | +5.33e-05 (-2.33e-04 to +3.39e-04) | +7.89e-04 (+5.19e-04 to +1.06e-03)* |
| ln X -6 | -1.04e-05 (-8.52e-05 to +6.44e-05) | +2.67e-04 (+1.95e-04 to +3.39e-04)* | +7.20e-06 (-7.52e-05 to +8.96e-05) | +2.32e-04 (+1.59e-04 to +3.06e-04)* |
| ln X -7 | -2.87e-06 (-3.34e-05 to +2.77e-05) | +1.47e-04 (+1.11e-04 to +1.83e-04)* | -6.25e-06 (-4.42e-05 to +3.17e-05) | +1.19e-04 (+8.74e-05 to +1.52e-04)* |
| ln X -8 | +5.26e-06 (-2.44e-05 to +3.49e-05) | +2.88e-04 (+2.08e-04 to +3.69e-04)* | +6.58e-06 (-2.95e-05 to +4.27e-05) | +2.50e-04 (+1.95e-04 to +3.04e-04)* |

Post hoc, outside the design: the same on PolyChord's volumes (contour_at_volume), stock mean first:

| point | stock mean |
| --- | ---: |
| ln X -1 | 0.02011 |
| ln X -2 | 0.02331 |
| ln X -3 | 0.02590 |
| ln X -4 | 0.02913 |
| ln X -5 | 0.03064 |
| ln X -6 | 0.03098 |
| ln X -7 | 0.03118 |
| ln X -8 | 0.03144 |

| point | stock-repeat | adopt | stop | adopt+stop |
| --- | --- | --- | --- | --- |
| max objective | +7.91e-05 (-3.23e-04 to +4.81e-04) | -3.20e-04 (-6.66e-04 to +2.65e-05) | -1.68e-04 (-4.96e-04 to +1.59e-04) | -3.72e-04 (-6.74e-04 to -7.01e-05)* |
| ln X -1 | +4.13e-06 (-1.24e-04 to +1.33e-04) | -1.32e-05 (-1.51e-04 to +1.24e-04) | -5.21e-05 (-1.63e-04 to +5.93e-05) | -5.50e-05 (-1.76e-04 to +6.59e-05) |
| ln X -2 | +9.63e-05 (-2.20e-05 to +2.15e-04) | +2.30e-05 (-1.16e-04 to +1.62e-04) | +6.72e-05 (-4.15e-05 to +1.76e-04) | +1.11e-04 (-4.00e-06 to +2.26e-04) |
| ln X -3 | +9.81e-05 (-1.20e-04 to +3.16e-04) | -1.58e-04 (-3.57e-04 to +4.22e-05) | +7.55e-05 (-1.27e-04 to +2.78e-04) | -1.76e-04 (-3.96e-04 to +4.29e-05) |
| ln X -4 | +1.08e-04 (-1.70e-04 to +3.86e-04) | -6.73e-04 (-9.89e-04 to -3.56e-04)* | +9.86e-05 (-2.15e-04 to +4.12e-04) | -7.55e-04 (-1.11e-03 to -3.96e-04)* |
| ln X -5 | -1.03e-05 (-7.78e-05 to +5.73e-05) | -1.86e-04 (-2.67e-04 to -1.04e-04)* | -7.25e-06 (-7.21e-05 to +5.77e-05) | -2.75e-04 (-3.86e-04 to -1.65e-04)* |
| ln X -6 | -2.24e-06 (-3.09e-05 to +2.64e-05) | -7.50e-05 (-1.09e-04 to -4.10e-05)* | -3.42e-06 (-3.59e-05 to +2.91e-05) | -9.79e-05 (-1.33e-04 to -6.23e-05)* |
| ln X -7 | +1.56e-05 (-2.55e-05 to +5.67e-05) | -5.73e-05 (-9.89e-05 to -1.58e-05)* | +6.47e-06 (-3.35e-05 to +4.65e-05) | -8.08e-05 (-1.12e-04 to -4.93e-05)* |
| ln X -8 | +3.58e-05 (-5.87e-05 to +1.30e-04) | -1.00e-04 (-1.84e-04 to -1.70e-05)* | -8.80e-06 (-9.50e-05 to +7.73e-05) | -1.13e-04 (-1.91e-04 to -3.37e-05)* |

Spent 1,669 core-hours. Distinct node lists per arm: adopt 18, adopt+shadow 6, adopt+stop 18, shadow 5, stock 20, stock-repeat 17, stop 18.

Wrong stops, shadow, 6 runs: 4 of 131512 above-contour evaluations (0.003%), per run 0, 0, 0, 0, 2, 2.

Wrong stops, adopt+shadow, 6 runs: 2 of 109684 above-contour evaluations (0.002%), per run 0, 0, 0, 0, 1, 1.

### Part B, target size

Part B: 18 of 18 recorded runs finished, 0 tasks have no record.

| arm | seeds | mean z | SE | 95% CI | sd z | sign (+/n, p) | scatter / stock's (p) | Holm hits | verdict |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | --- |
| stock-repeat | 6 | +0.049 | 0.084 | -0.167 to +0.266 | 0.206 | 3/6, 1 | 1.056 (0.373) | 0 | PASS |
| adopt+stop | 6 | -0.221 | 0.097 | -0.470 to +0.027 | 0.237 | 1/6, 0.219 | 0.815 (0.961) | 0 | PASS |

| arm | wall vs stock (%, 95% CI) | evaluations vs stock (%, 95% CI) | Holm hits named |
| --- | --- | --- | --- |
| stock-repeat | +0.2 (-1.4 to +1.9) | -0.5 (-4.0 to +3.0) | none |
| adopt+stop | -41.1 (-44.5 to -37.6) | -17.2 (-22.4 to -11.9) | none |

Calibration (scatter of log(Z) across seeds against the quoted error):

| arm | runs | mean log(Z) | scatter | rms quoted error | scatter / quoted (+/- approx) |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 6 | 0.017452 | 2.232e-04 | 1.954e-04 | 1.142 +/- 0.361 |
| stock-repeat | 6 | 0.017466 | 2.356e-04 | 1.959e-04 | 1.203 +/- 0.380 |
| adopt+stop | 6 | 0.017391 | 1.820e-04 | 1.930e-04 | 0.943 +/- 0.298 |

Per arm, mean over seeds:

| arm | runs | log(Z) error | ndead | nlike | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 6 | 1.954e-04 | 5875 | 414912 | 1034 |
| stock-repeat | 6 | 1.958e-04 | 5902 | 412539 | 1037 |
| adopt+stop | 6 | 1.930e-04 | 4668 | 342931 | 609 |

Smallest Holm-adjusted p across parameters, per statistic:

| arm | mean | sd | q05 | q50 | q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock-repeat | 1 | 0.281 | 1 | 1 | 0.281 |
| adopt+stop | 0.281 | 0.281 | 0.281 | 0.562 | 0.281 |

Secondary, outside the pass rule: the tail, which is what the search is for.

| arm | runs | mean max objective | mean contour at ln X -4 | at -6 | at -8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 6 | 0.03364 | 0.02513 | 0.02932 | 0.03081 |
| stock-repeat | 6 | 0.03357 | 0.02515 | 0.02939 | 0.03080 |
| adopt+stop | 6 | 0.03341 | 0.02695 | 0.03057 | 0.03110 |

Paired difference from stock (mean, 95% CI, * = excludes 0):

| point | stock-repeat | adopt+stop |
| --- | --- | --- |
| max objective | -6.99e-05 (-4.70e-04 to +3.30e-04) | -2.33e-04 (-6.56e-04 to +1.89e-04) |
| ln X -1 | -1.52e-05 (-1.31e-04 to +1.00e-04) | +6.71e-04 (+6.03e-04 to +7.39e-04)* |
| ln X -2 | -3.72e-05 (-1.35e-04 to +6.08e-05) | +1.06e-03 (+9.02e-04 to +1.22e-03)* |
| ln X -3 | -3.29e-06 (-1.27e-04 to +1.20e-04) | +1.52e-03 (+1.27e-03 to +1.77e-03)* |
| ln X -4 | +2.67e-05 (-1.28e-04 to +1.81e-04) | +1.83e-03 (+1.49e-03 to +2.16e-03)* |
| ln X -5 | +1.02e-04 (-2.51e-04 to +4.56e-04) | +2.30e-03 (+1.76e-03 to +2.84e-03)* |
| ln X -6 | +7.41e-05 (-2.43e-04 to +3.91e-04) | +1.25e-03 (+9.60e-04 to +1.55e-03)* |
| ln X -7 | +8.66e-06 (-1.06e-04 to +1.23e-04) | +4.81e-04 (+3.81e-04 to +5.81e-04)* |
| ln X -8 | -3.02e-06 (-4.74e-05 to +4.14e-05) | +2.95e-04 (+2.62e-04 to +3.28e-04)* |

Post hoc, outside the design: the same on PolyChord's volumes (contour_at_volume), stock mean first:

| point | stock mean |
| --- | ---: |
| ln X -1 | 0.02006 |
| ln X -2 | 0.02328 |
| ln X -3 | 0.02588 |
| ln X -4 | 0.02920 |
| ln X -5 | 0.03064 |
| ln X -6 | 0.03097 |
| ln X -7 | 0.03116 |
| ln X -8 | 0.03134 |

| point | stock-repeat | adopt+stop |
| --- | --- | --- |
| max objective | -6.99e-05 (-4.70e-04 to +3.30e-04) | -2.33e-04 (-6.56e-04 to +1.89e-04) |
| ln X -1 | -6.57e-06 (-1.15e-04 to +1.02e-04) | -8.26e-06 (-1.29e-04 to +1.12e-04) |
| ln X -2 | +2.28e-05 (-6.78e-05 to +1.13e-04) | -1.06e-04 (-2.71e-04 to +5.90e-05) |
| ln X -3 | +1.26e-04 (-2.14e-04 to +4.65e-04) | -3.40e-04 (-6.55e-04 to -2.38e-05)* |
| ln X -4 | +1.65e-04 (-3.06e-04 to +6.36e-04) | -1.22e-03 (-1.85e-03 to -5.89e-04)* |
| ln X -5 | +1.17e-05 (-9.17e-05 to +1.15e-04) | -3.55e-04 (-5.73e-04 to -1.36e-04)* |
| ln X -6 | +5.56e-06 (-3.53e-05 to +4.64e-05) | -1.35e-04 (-2.02e-04 to -6.79e-05)* |
| ln X -7 | +4.56e-06 (-1.98e-05 to +2.89e-05) | -1.03e-04 (-1.36e-04 to -6.97e-05)* |
| ln X -8 | -2.02e-06 (-5.95e-05 to +5.54e-05) | -1.27e-04 (-2.00e-04 to -5.35e-05)* |

Spent 1,436 core-hours. Distinct node lists per arm: adopt+stop 4, stock 4, stock-repeat 4.

### Reading the rule's results

**log(Z).** No arm in A moved it: every mean z is within 0.02 sigma of zero except
stock-repeat's +0.073, the noise control. In B, adopt+stop reads -0.221 sigma, 5 of 6
seeds negative (sign test p = 0.219), interval -0.470 to +0.027. It passes, but only
just inside the +/-0.25 and +/-0.5 limits, and on 6 seeds that does not rule out a
small real shift. Part A at the same settings shows none.

**The posterior failures are small.** Stock-repeat has no Holm hit, so any hit fails
an arm. The hits are all narrower source-position spreads:

| arm | statistic | stock | shift | shift / stock | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
| adopt | `source_l_pixels` sd | 9.218 | -0.112 | -1.21% | 0.018 |
| adopt+stop | `source_l_pixels` sd | 9.218 | -0.111 | -1.21% | 0.026 |
| adopt+stop | `source_m_pixels` sd | 9.417 | -0.129 | -1.37% | 0.0067 |
| adopt+stop | `source_m_pixels` q95 | 14.581 | -0.343 | -2.36% | 0.0022 |

No statistic of any arm moved by more than 0.08 of a posterior sd in A. These are FAILs
under the committed rule and stay FAILs. The same `source_l_pixels` shift appears in
both adopt arms and in neither non-adopt arm, which points to adoption, though the arms
share seeds, so that is not independent evidence. Holm runs within each statistic, so
five families are tested and one lone hit could be chance. adopt+stop's three cannot
easily be.

**Calibration.** Quoted log(Z) errors are too small in Part A, by a factor of ~1.3
(scatter over quoted 1.26-1.33, +/- ~0.19), in every arm including stock, so neither
setting causes it. Part B reads 0.94-1.20 +/- ~0.35, which cannot separate 1.0 from
1.3. Arms share seeds, so the five A ratios are not five independent measurements.
Log(Z) scatter is no worse than stock's in any arm.

**Wrong stops.** 6 of 241,196 above-contour evaluations across the shadow arms, all in
seeds 9105 and 9106, four in shadow and two in adopt+shadow. Roughly half of evaluations
would stop, so early stop acts on tens of thousands per run and errs a few times at
most. Part B had no shadow arm.

**Early stop saves wall time only.** It does not change the number of evaluations
(-0.2%, -4.6 to +4.2), it makes them shorter. Adoption saves evaluations (-10.3%),
because fewer chains return below the contour and get thrown away.

**Scheduling.** Part A spread over 17-20 distinct nodes per arm. Part B saw 4 distinct
8-node lists per arm, with heavy overlap across arms, and arms alternated within each
seed. Stock-repeat's wall time (+0.2% in B, -2.1% in A) shows no gap between two
identical arms; the saving of a treated arm rests on that same scheduling.

### The tail, post hoc

**The pre-specified tail table is on the wrong axis.** Its contours come from
`dead[-x * nlive]`, which assumes each dead-file row is one step of compression. It is
not: `_dead.txt` also holds rows that never reach the posterior file `.txt`. In four
runs inspected, 260 such rows in stock and 82 with adoption at A, 1,832 and 630 at B;
38 (A) and 311 (B) of those, the same with or without adoption, are failed points at
logzero, which the table already skips. The rest are most likely the chains returning below the contour and thrown away, since
adoption exists to cut them; that identification is an inference from the counts and
the patch comment, not traced row by row. `.txt` itself steps by exactly 1/nlive per
row. So the "+1e-3" shifts in that table, and the matching stock difference between
nlive 150 and 500, mostly measure how many chains were discarded. They stay above as
committed, but should not be read as tail shifts. This was found after reading the
results, and no verdict depends on it: the table never had one.

**The corrected table** (`contour_at_volume`, the "Post hoc" tables above, also added
after results were read) takes volumes from PolyChord's own posterior weights. Stock
gives a built-in check here, with no analytic truth available: nlive 150 and nlive 500,
with different numbers of workers, should trace one curve, and they do.

| ln X | stock + repeat, A | stock + repeat, B | B - A | its SE |
| ---: | ---: | ---: | ---: | ---: |
| -1 | 0.02011 | 0.02006 | -5.3e-05 | 8.7e-05 |
| -2 | 0.02335 | 0.02329 | -6.0e-05 | 6.7e-05 |
| -3 | 0.02595 | 0.02594 | -9.9e-06 | 1.0e-04 |
| -4 | 0.02918 | 0.02929 | +1.0e-04 | 1.3e-04 |
| -5 | 0.03064 | 0.03065 | +9.1e-06 | 2.9e-05 |
| -6 | 0.03098 | 0.03098 | -6.5e-06 | 1.2e-05 |
| -7 | 0.03119 | 0.03116 | -2.9e-05 | 1.3e-05 |
| -8 | 0.03146 | 0.03134 | -1.2e-04 | 3.3e-05 |

Only -7 and -8 separate, right at the end of the search, where the two shapes stop at
slightly different points. That agreement is why stock is treated as the reference
below. It is a consistency check, not a proof.

On this axis:

- **Early stop alone matches stock at every point** in A, with intervals as tight as
  stock-repeat's.
- **Adoption shifts contours down from ln X -4 onward**, in both adopt arms, with every
  interval from -4 to -8 excluding zero, and B's at -3 too. Stock-repeat excludes zero
  nowhere.
- **The shift grows with scale.** At ln X -4 adopt+stop reads -7.6e-04 in A and
  -1.22e-03 in B. Divided by the local slope of the curve it is a roughly constant lag
  past ln X -4 of ~0.2-0.3 nats in A and ~0.4-0.5 nats in B, so adoption reports ~25-40%
  less volume above a given objective than stock does. Those lag figures are hand
  estimates from the table, not a fitted result.
- **log(Z) cannot see this,** because below ln X -4 is under 2% of the prior and L is
  ~1 everywhere. That is exactly the limit the design warned about.

The cause was not identified. The size of the effect grows with workers/nlive, which
fits adoption being the source, but the mechanism is not established. Target size is
what this repo aims for, and it is where the shift is largest.

**Best objective found.** adopt+stop's is lower than stock's in A by 3.7e-04, about 1.1%
(-6.7e-04 to -7.0e-05). adopt reads -3.2e-04 and stop -1.7e-04 with intervals that
include zero. B reads -2.3e-04, interval including zero. The maximum is one point per
run, so this is a weak signal, and it points the same way as the volumes.

## Not tested

GPU R2D2, WSClean, other nlive, repeats or node counts between the two parts, and
parameter-space or metric changes. Early stop only applies to `total_rms_jy`.
Early stop alone and adoption alone were not run at target size, and target size has
no many-seed wrong-stop count: round 371's seed 344 (1 of 98,616) is its only reading.
The likelihood has no analytic log(Z) or posterior, so every comparison is against
stock, and nothing here shows stock itself is right beyond the nlive agreement above.
Part B's posterior tests could never reach significance, as the design said.

## Conclusion

Early stop passed everything at production shape: log(Z) +0.003 sigma, no posterior
statistic moved, the tail matches stock, 6 wrong stops in 241,196, and -32% wall time.
Use it there if wanted. Before relying on it at target size, run it alone against
stock there, since this validation only saw it combined with adoption.

Contour adoption failed at production shape on the committed rule, and the failure is
small on its own terms: ~1-2% in source-position spread and nothing in log(Z). The
post-hoc volume curve is the larger concern, a lag of up to ~0.5 nats in the tail at
target size, growing with scale, with its cause unknown. It buys ~5 points of wall time
on top of early stop. The recommendation is to leave it off. The Part B PASS for the
pair should not be read as clearing it, because that test could not see the posterior
and the tail table says otherwise.

Reproduce with `python3 scripts/sampler_validation.py analyse A
docs/csd3-sampler-validation-runs.jsonl` (or `B`). The gzipped chains and task logs are
in `.probe/val/` (1.9 GB, untracked), left in place for now in case the tail question
is pursued.
