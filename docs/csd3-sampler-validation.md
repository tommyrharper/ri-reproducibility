# Many-seed validation of `NS_ADOPT_CONTOUR` and `NS_EARLY_STOP`

Status: **design fixed, runs in progress, no comparison made.** The results section
is added below once the runs finish. Arrays 39721682 (Part A, 132 tasks) and
39721865 (Part B, 18 tasks) were submitted on 2026-10-30. Part A started at 18:44
with 10 tasks; by 19:08 its first 9 runs, covering stock, stock-repeat and all three
treated arms, had finished and extracted cleanly, so the pipeline works end to end.
Those rows were opened only to check they are complete. At 19:53 Part A had 25 of
132 runs recorded, all status ok, with 33 tasks started; Part B had not started.
At 19:54 none of the 33 task logs held a traceback, out-of-memory, cancellation,
time-limit or EDQUOT line, and Part B had no log yet.
Budget check at that point, from aggregate timing with no split by arm: 24 runs in 65
minutes on 10 slots is about 27 minutes per slot per run, against the 15-22 expected
below, or ~2,260 core-hours for Part A. That estimate is rough because it ignores
in-flight work. If Part B overruns by the same third, ~2,000 core-hours, the total is
~4,260, still under 5,000. At this rate Part A ends around 00:40. The file quota stood
at ~528k of 1,048,576 at 19:53 (it reads ±20k with in-flight runs) with ~100-120k of that from the runs, and that part does not grow
with finished runs: a task deletes its run directory once the record is written, and
`keep/` held 187 files at 19:52. So it is ~12k per in-flight run, and ten slots hold
it level; Part B's in-flight count is unmeasured, with ~500k of headroom. The
criteria here were committed before any result was read and do not change afterwards.
Nothing becomes a default: both settings stay opt-in, this report recommends, and the
user decides.

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

## Not tested

GPU R2D2, WSClean, other nlive, repeats or node counts between the two parts, and
parameter-space or metric changes. Early stop only applies to `total_rms_jy`.
