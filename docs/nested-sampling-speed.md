# Making a nested-sampling search faster: the index

One hundred fifty-nine profiling rounds cut WSClean from ~2.3 s to ~143 ms/evaluation.
The latest three-repeat async measurement reached **110.4 +/- 2.7
evaluations/second at 20 workers**; the historical peak is 126
evaluations/second at 19 workers.
Values below are
per-iteration baselines, except the end-to-end figure above; pre-iteration-18
rates are historical and roughly half the current rate.

## What shipped

| change | where it lives | measured worth |
|---|---|---|
| PolyChord's asynchronous MPI mode by default | `polychord_wsclean.py` | -27-33% wall clock per dead point ([throughput](nested-sampling-throughput.md)) |
| WSClean built for the host CPU, `x86-64-v3` default | `WSCLEAN_TARGET_CPU` | +8.6% evals/s |
| MeqTrees no longer predicts a constant | `point_source_forest.py` | +16-20% evals/s |
| `WEIGHT`/`SIGMA` columns no longer written | `simulate_point_source_ms.py` | +3.8% evals/s |
| Measurement Set built in a shared host tmpfs | `NS_SCRATCH_DIR` | -1.2 ms/eval ([I/O placement](nested-sampling-io-placement.md)) |
| Cross-device Measurement Set copy removed | `scratch_root_for()` | -6.4 ms/eval, ~+2.2% evals/s |
| Six unread MS subtables dropped | `simulate_point_source_ms.py` | +14.9% evals/s ([MS open](nested-sampling-ms-open.md)) |
| `FEED` dropped too | same | -3.2% on the `wsclean` binary |
| Antenna names cached in-process | `patches/0001` | +10.5% evals/s ([patches](nested-sampling-wsclean-patches.md)) |
| `wsclean-zygote` fork server | `docker/wsclean/src/zygote.cpp` | +8.4% evals/s ([zygote](nested-sampling-wsclean-zygote.md)) |
| `-data-column DATA` | `polychord_wsclean.py` | -1.0% on the binary ([cost model](nested-sampling-cost-model.md)) |
| Dead work cut from the simulate stage | `simulate_point_source_ms.py` | -20% of that stage ([simulate](nested-sampling-simulate-stage.md)) |
| One parent-MS open shared five ways | `patches/0003` | -3.7% on the binary ([shared open](nested-sampling-shared-ms-open.md)) |
| FFTW planner warmed in the zygote parent | `zygote.cpp` | -6.9% on the binary ([FFTW](nested-sampling-fftw-planner.md)) |
| `schaapcommon` FFTW plan cache | `patches/0004` | -2.7% on the binary ([gridder floor](nested-sampling-gridder-floor.md)) |
| Measurement Set read in row blocks | `patches/0005` | -2.2% on the binary ([row blocks](nested-sampling-row-blocks.md)) |
| cfitsio and casacore init moved to the zygote parent | `zygote.cpp` | -2.39 ms/eval ([warm-up](nested-sampling-process-warm-up.md)) |
| NaN-free image RMS uses a BLAS dot product | `common.py` | 25.2 to 7.8 us per RMS call (container microbenchmark) |
| Sigma-res FITS files load only when the objective uses `sigma_res` | `polychord_*.py` | 1.03 to 0.46 ms metrics stage on the current WSClean benchmark; objective-preserving for other metrics |
| Dirty FITS output is disabled when the objective does not use `sigma_res` | `polychord_wsclean.py` | 111.3 +/- 1.8 eval/s median in three patched runs versus 109.3 +/- 2.5 previously; within normal variance, with ~33.9 MB peak memory |
| Residual metrics reuse the loaded image buffer | `common.py` | Removes one full-image allocation per evaluation; memory-only change at current image size |
| Off-source metric masks are bounded and reused | `common.py` | 25.2 to 3.6 us per mask (container microbenchmark); no measurable end-to-end gain |
| Sigma-res norms use direct dot products | `common.py` | 11.97 to 11.41 us per 512x512 metric call (host microbenchmark, identical result); too small to affect default throughput |
| R2D2 `.mat` conversion skips compression on tmpfs | `ms_to_r2d2_mat.py` | 8.59 to 0.39 ms for 12k visibilities (microbenchmark) |
| R2D2 `.mat` conversion avoids broadcast temporaries | `ms_to_r2d2_mat.py` | 1.13 to 1.09 ms for 1404 visibilities (10 warm calls, container microbenchmark) |
| R2D2 `.mat` conversion flattens weights directly | `ms_to_r2d2_mat.py` | 1.802 to 1.657 us per weight expansion (100k calls, container microbenchmark; 8.0%) |
| R2D2 checkpoints loaded once and shared by forked workers | `r2d2_serve.py` | Production `optimiser.R2D2` alias now hits the cache (self-check); removes repeated 25-checkpoint loads per evaluation; first real benchmark is 0.438 +/- 0.002 eval/s at 3.47 GB peak worker memory |
| R2D2 FINUFFT plans reused across sequential evaluations | `r2d2_serve.py` | Synthetic 128x128 plan setup: 0.86 ms fresh versus 0.091 ms retargeted (9.5x); real R2D2 benchmark shows no measurable end-to-end gain |
| R2D2 model-load garbage collection removed | `docker/r2d2/patches/skip-model-load-gc.patch` | 25 checkpoint swaps: 1.05 s to 0.235 s; R2D2 throughput 0.4372 to 0.5622 eval/s in three-repeat-scale runs, with unchanged 3.47 GB peak memory |
| R2D2 checkpoint weights assigned by reference | `docker/r2d2/patches/assign-checkpoint-weights.patch` | R2D2 throughput 0.5622 to 0.5948 eval/s in three controlled runs (5.8%), with unchanged 3.47 GB peak memory |
| R2D2 checkpoint key normalization cached | `r2d2_serve.py` | 4.36 to 0.01 us per repeated 24-key lookup in a microbenchmark; three fresh end-to-end runs measured 0.7205, 0.7768, and 0.7350 eval/s, so no isolated throughput gain is claimed |
| R2D2 auto thread count rounds up per-rank CPU share | `run-nested-sampling-r2d2.sh` | 0.6198 to 0.7249 eval/s at 8 ranks (16.9%), with unchanged 3.47 GB peak memory |

Compiling WSClean for this exact CPU (`-march=native`) was rejected: three
throughput repeats measured 136.9, 36.9, and 111.6 evaluations/second (median
111.6), versus 114.8, 119.2, and 111.2 (median 114.8) for matched
`x86-64-v3`. The native build is not retained as the default.

The R2D2 CPU backend probe closed two inference alternatives: disabling
MKLDNN was 1.853 s versus 1.363 s for a warmed 512x512 U-Net forward at three
Torch threads, while TorchScript tracing was 1.313 s versus 1.324 s eager.
The latter is below the observed run variance and also emits shape-dependent
trace warnings, so the eager MKLDNN path remains the portable default.

Two changes bought run *size* rather than speed - see [disk footprint](nested-sampling-disk-footprint.md)
and [run scaling](nested-sampling-run-scaling.md):

| change | measured worth |
|---|---|
| `sim.ms` deleted as the evaluation is scored | evaluation directory 3.4x smaller |
| WSClean's four unread FITS images pruned | another 3.94x - 393.6 KB to 99.9 KB, so this host's ceiling went from 477k evaluations to 1.88M |
| A resume keeps each adopted objective, not each record | 62 GB down to 5.4 GB, which is the difference between the target run finishing and being OOM-killed |
| Progress-bar redraw backed off to 9x its own cost | 44% of a core down to ~12%, and no longer growing with the run |
| GNU progress scans classify total and post-checkpoint evaluations in one `find` walk | 0.232 s to 0.073 s for 20 scans of a 635-evaluation run; BSD `find` keeps the portable fallback |

The current async WSClean throughput check measures **110.4 +/- 2.7
evaluations/second** at 20 workers, 145.3 +/- 1.9 ms/evaluation, and 34.4 MB
peak imaging-worker memory across three repeats (30 August 2026). The three
rates were 101.2, 110.4, and 113.8 evaluations/second; this refresh is
consistent with the host's observed run-to-run variance and is not a claimed
code speedup. The image binary remains the dominant stage at 129.6 +/- 1.0
ms/evaluation.
The latest three-repeat 20-worker refresh measured **105.6 +/- 4.2
evaluations/second**, with 144.7 +/- 2.5 ms/evaluation and 34.3 MB peak
memory. Rates were 108.0, 92.3, and 105.6 evaluations/second; image binary
remains dominant at 128.6 +/- 2.4 ms/evaluation, so this is a control refresh,
not a claimed regression or speedup.
The next three-repeat 20-worker control measured **112.0 +/- 3.3
evaluations/second**, with 139.5 +/- 0.87 ms/evaluation and 34.3 MB peak
memory. Rates were 112.0, 117.4, and 105.3 evaluations/second; image binary
remains dominant at 124.2 +/- 0.54 ms/evaluation, so this is a baseline refresh,
not an isolated code speedup.
The new rank-scaling probe measured **101.2 +/- 1.5 evaluations/second** at 15
workers, with 34.5 MB peak memory; 20 workers remains faster on this host.
The fresh three-repeat 10-worker probe measured **81.1 +/- 0.8 evaluations/second**
(81.1, 83.1, and 80.7), with 34.5 MB peak memory and 97.5 ms/evaluation.
It confirms that reducing concurrency below 20 workers lowers throughput without
reducing memory; no worker-count default change is justified.
The fresh three-repeat 9-worker probe measured **76.1 +/- 2.5 evaluations/second**
(70.2, 76.1, and 79.6), with 34.2 MB peak memory and 92.6 ms/evaluation.
It further confirms that reducing concurrency below 20 workers lowers throughput
without reducing memory; no worker-count default change is justified.
The 17-worker probe measured **115.9 evaluations/second** by the robust median
of three repeats, with 34.5 MB peak memory; it does not displace the 20-worker
default.
Repeating that probe at 19 workers measured **107.1 +/- 2.8 evaluations/second**
with 34.5 MB peak memory, statistically indistinguishable from 20 workers and
not enough to change the default.
The fresh three-repeat 19-worker probe measured **113.4 +/- 3.6
evaluations/second**, 134.9 +/- 6.0 ms/evaluation, and 34.4 MB peak memory.
It remains consistent with the current 20-worker result and does not justify a
runtime default change.
The fresh three-repeat 18-worker probe measured **109.3 +/- 3.0
evaluations/second**, 133.4 +/- 2.6 ms/evaluation, and 34.3 MB peak memory.
It remains below the current 20-worker result and does not justify changing the
default.
The new 18-worker probe measured **105.1 evaluations/second** by the robust
median of three repeats, with 34.3 MB peak memory. It is also statistically
indistinguishable from 19 and 20 workers; 20 remains the default.
The fresh three-repeat 17-worker probe measured **106.4 +/- 2.9
evaluations/second**, 126.8 +/- 2.3 ms/evaluation, and 34.5 MB peak memory.
It is below the current 20-worker result and does not justify changing the
default.
The fresh three-repeat 16-worker probe measured **107.8 +/- 3.1
evaluations/second**, 128.8 +/- 1.2 ms/evaluation, and 34.5 MB peak memory.
It is below the current 20-worker result and does not justify changing the
default.
The fresh three-repeat 14-worker probe measured **96.0 +/- 2.9
evaluations/second**, 119.2 +/- 1.2 ms/evaluation, and 34.3 MB peak memory.
It is below the 15-20-worker range and does not displace the 20-worker default.
The fresh three-repeat 13-worker probe measured **101.4 +/- 1.5
evaluations/second**, 106.6 +/- 2.3 ms/evaluation, and 34.4 MB peak memory.
It remains below the 20-worker result and does not displace the default.
The new 12-worker probe measured **89.14 +/- 2.5 evaluations/second** by the
robust median of three repeats, with 34.5 MB peak memory. Its 94.98 +/- 0.64
ms/evaluation image-binary stage is slower than the 14-20-worker range, so it
does not displace the 20-worker default.
The fresh three-repeat 11-worker probe measured **79.84 +/- 5.8
evaluations/second** by the robust median, with 34.2 MB peak memory. Its
94.07 +/- 0.31 ms/evaluation image-binary stage is slower than the 12-20-worker
range, so it does not displace the 20-worker default.
The new 10-worker probe measured **87.43 +/- 1.5 evaluations/second** by the
robust median of three repeats, with 34.3 MB peak memory. Its 84.7 +/- 2.2
ms/evaluation image-binary stage does not displace the 20-worker default.
The new 9-worker probe measured **77.1 evaluations/second** by the robust
median of three repeats, with 34.4 MB peak memory. Its 84.4 +/- 0.6
ms/evaluation image-binary stage does not displace the 20-worker default.
The fresh three-repeat 8-worker probe measured **74.6 +/- 0.43
evaluations/second** (74.6, 74.9, and 73.3), with 34.1 MB peak memory and
87.1 +/- 1.2 ms/evaluation. Its 75.4 +/- 1.1 ms/evaluation image-binary stage
is slower than the 9-20-worker range and does not displace the 20-worker
default.
The matched fresh three-repeat R2D2 controls measured **0.7254 evaluations/second**
at three threads (0.7253-0.7291) and **0.7671 evaluations/second** at four
threads (0.7660-0.7708), both at 8 ranks and **3.47 GB** peak worker memory.
Four threads is a **5.7%** median improvement over the automatic three-thread
setting, but remains an explicit candidate: the benchmark CLI runs each thread
setting in a separate invocation, so this result is not an interleaved A/B.
The new interleaved runner completed two paired repeats per arm at **0.7276
eval/s** for three threads and **0.7730 eval/s** for four threads (3.47 GB
peak memory in both arms), a 6.2% four-thread median advantage. This remains
candidate evidence rather than a portable default because two pairs are not a
full production-strength sample.
The current three-repeat 7-worker WSClean probe measured **71.8 +/- 1.2
evaluations/second** (68.3, 71.8, and 72.9), with **34.4 MB** peak memory and
79.8 ms/evaluation. It is slower than the current 20-worker control and does
not justify changing the default.
The current three-repeat 6-worker WSClean probe measured **67.7 +/- 2.5
evaluations/second** (63.2, 67.7, and 72.4), with **34.2 MB** peak memory and
69.8 ms/evaluation. Its 60.3 ms image-binary stage is slower than the
7-20-worker range and does not justify changing the default.
The new 7-worker probe measured **70.17 evaluations/second** by the robust
median of three repeats (66.37, 70.17, and 73.04), with 34.2 MB peak memory.
Its 70.6 +/- 2.3 ms/evaluation image-binary stage is slower than the
8-20-worker range and does not displace the 20-worker default.
The new 6-worker probe measured **64.32 +/- 0.69 evaluations/second** by the
robust median of three repeats (62.18, 64.32, and 64.75), with 34.2 MB peak
memory. Its 65.26 +/- 1.0 ms/evaluation image-binary stage is slower than the
7-20-worker range and does not displace the 20-worker default.
The new 5-worker probe measured **54.71 +/- 1.7 evaluations/second** by the
robust median of three repeats (53.19, 54.71, and 59.35), with 34.2 MB peak
memory. Its 62.82 +/- 2.1 ms/evaluation image-binary stage is slower than the
6-20-worker range and does not displace the 20-worker default.
The current three-repeat 5-worker control measured **55.8 evaluations/second**
(53.6, 55.8, and 55.8), with **34.2 MB** peak memory and 68.2-72.4
ms/evaluation. It confirms the earlier result and does not displace the
20-worker default.
The new 4-worker probe measured **45.73 evaluations/second** by the robust
median of three repeats (43.00, 45.73, and 51.56), with 34.0 MB peak memory.
Its 56.82 +/- 5.8 ms/evaluation image-binary stage is slower than the 5-20-worker
range and does not displace the 20-worker default.
The fresh three-repeat synchronous 4-worker control measured **35.9
evaluations/second** (35.8-36.1), with 34.0 MB peak memory. It is recorded as a
mode-specific control and does not alter the asynchronous default.
The new 3-worker probe measured **32.20 +/- 0.42 evaluations/second** by the
robust median of three repeats (32.20, 33.73, and 32.17), with 34.2 MB peak
memory. Its 54.18 +/- 0.74 ms/evaluation image-binary stage is slower than the
4-20-worker range and does not displace the 20-worker default.
The new 2-worker probe measured **17.27 +/- 0.008 evaluations/second** by the
robust median of three repeats (17.27, 17.27, and 17.30), with 34.2 MB peak
memory. Its 50.72 +/- 0.02 ms/evaluation image-binary stage is slower than the
3-20-worker range and does not displace the 20-worker default.
The new 1-worker probe measured **15.91 +/- 0.036 evaluations/second** by the
robust median of three repeats (15.91, 15.82, and 15.96), with 34.0 MB peak
memory. Its 55.77 +/- 0.14 ms/evaluation image-binary stage is slower than the
2-20-worker range and does not displace the 20-worker default.
The prior three-repeat group measured 117.2 +/- 2.5 evaluations/second. The production preset remains
the comparable target-scale record: 114.7 +/- 0.71 evaluations/second over
~39,900 evaluations at 150 live points, 15 repeats, and unlimited dead points.
The sigma-res lazy-load change now measures 108.2 +/- 4.6 evaluations/second
over three current throughput repeats, within normal end-to-end variance,
while the metrics stage itself remains at 0.47 +/- 0.003 ms/evaluation versus
about 1.0 ms before the change.
The latest default synchronous baseline measures 71.6 evaluations/second
(72.3, 71.4, and 71.6 across three repeats) at 148.7 +/- 0.4 ms/evaluation;
imaging remains 132.8 +/- 0.4 ms/evaluation. The prior baseline measured 71.1
 +/- 0.85 evaluations/second at 150.9 +/- 0.011 ms/evaluation over three
repeats; imaging remained 133.3 +/- 0.15 ms/evaluation.
The current three-repeat synchronous refresh measured 71.8, 70.6, and 68.1
evaluations/second, for a median of **70.6 +/- 0.82 evaluations/second** and
150.5 +/- 0.39 ms/evaluation. Its 134.0 +/- 0.96 ms image-binary stage is
consistent with the existing baseline; no optimization claim is made.
The latest three-repeat synchronous control measured 72.4, 67.1, and 68.9
evaluations/second, for a median of **68.9 evaluations/second** and 149.8
ms/evaluation. Its 134.8 ms median image-binary stage is consistent with the
existing baseline; no optimization claim is made.
The first three-repeat R2D2 benchmark with the checkpoint cache enabled measures
**0.438 +/- 0.002 evaluations/second**, 11.8 +/- 0.03 seconds/evaluation, and
3.47 GB peak worker memory at 8 ranks (`nlive=8`, `max_ndead=12`). The older
pre-cache baseline was 0.37 +/- 0.002 evaluations/second, but it also predates
the `.mat` conversion changes, so this is baseline evidence rather than an
isolated cache A/B result. R2D2 production-scale measurement remains too
expensive for this iteration; the checkpoint archive is now available through
`CHECKPOINTS_DIR` for future controlled comparisons.
An interleaved thread-count probe with real checkpoints measured **0.414 +/-
0.015 evaluations/second** at one R2D2 thread versus 0.438 +/- 0.002 at the
default two threads, with unchanged 3.47 GB peak memory. One thread is therefore
a regression and does not improve the memory budget.
The cross-evaluation FINUFFT cache reuses plan allocations while calling
`setpts` for each new trajectory; a synthetic 128x128 measurement measured
0.86 ms for fresh construction versus 0.091 ms for retargeting (8 samples,
9.5x). A checkpoint-backed three-repeat run now measures **0.4372 +/- 0.004
evaluations/second** versus 0.4381 +/- 0.002 before plan reuse, with unchanged
3.47 GB peak memory; the real end-to-end result is statistically unchanged.
R2D2 phase profiling on the same 41-evaluation run attributes **7219.7
ms/evaluation** to 25 model updates and **80.1 ms/evaluation** to residual
computations; model inference is therefore the next measured target.
Removing two full garbage collections from each model-update loop reduced the
checkpoint-swap microbenchmark from 1.05 s to 0.235 s over 25 swaps. Three
controlled default R2D2 runs then measured **0.5622 +/- 0.0025
evaluations/second**, versus **0.4372 +/- 0.004** before the change, with
unchanged 3.47 GB peak memory. The benchmark runs used the same 41-evaluation
smoke workload, so this is a warm production-path speed result, not a
production-scale 150-live-point claim.
Using PyTorch `load_state_dict(assign=True)` for each checkpoint swaps tensor
references instead of copying checkpoint weights into the live U-Net. Three
fresh controlled runs measured **0.5948 +/- 0.008 evaluations/second** versus
0.5622 +/- 0.0025 before the change, a **5.8% throughput gain**, with unchanged
3.47 GB peak worker memory. Outputs remain on the same model path and the
benchmark used the same 41-evaluation smoke workload.

A fresh three-repeat measurement of the unchanged checkpoint-swap path measured
**0.6198 evaluations/second** (0.6200, 0.6198, and 0.6196), with **3.47 GB**
peak worker memory. This refresh is not a claimed speedup: the 4.2% difference
from the prior 0.5948 group was not interleaved and remains within observed
run variance.
Rounding the automatic per-rank thread allocation up from `20 / 8 = 2` to 3
threads was then measured in three controlled runs at **0.7249 +/- 0.0018
evaluations/second**, a **16.9%** gain over the two-thread baseline, with
unchanged 3.47 GB peak memory. Explicit `R2D2_OMP_THREADS` still overrides the
automatic choice. A three-repeat explicit four-thread probe then measured
**0.766 +/- 0.002 evaluations/second** at the same 8 ranks and **3.47 GB** peak
memory, about **5.4%** above the three-thread result. This is a candidate rather
than a new automatic default: four threads oversubscribe the nominal 20-CPU
host, and the older pre-optimization sweep rejected four threads.
A matched five-thread probe measured **0.818 +/- 0.013 evaluations/second**
versus **0.813 +/- 0.001** at four threads, with unchanged memory; the 0.65%
difference is within run variance, so five threads is not a new candidate.
An asynchronous six-thread probe measured **0.7866 evaluations/second** (three
runs) versus the matched four-thread median **0.8127**, a 3.2% regression, with
unchanged 3.47 GB peak memory. Six threads is rejected.
An explicit seven-thread probe measured **0.7956 evaluations/second** (three
runs) versus the matched four-thread median **0.8127**, a 2.1% regression, with
unchanged 3.47 GB peak memory. Seven threads is rejected as well.
An explicit eight-thread probe measured **0.7493 +/- 0.011 evaluations/second**
(three runs) versus the matched four-thread median **0.8127**, a 7.8% regression,
with unchanged 3.47 GB peak memory. Eight threads is rejected too.

A fresh controlled preset probe repeated the comparison on the current commit:
the automatic three-thread setting measured **0.7251 eval/s** median across
three runs, while explicit four threads measured **0.7738 eval/s** median
(+6.7%), with both at **3.47 GB** peak worker memory. Four threads remains a
useful explicit setting for this 8-rank, 20-CPU host, but is not made automatic:
the best thread count depends on the rank count and available CPUs, and forcing
four threads would oversubscribe smaller-rank or larger-host configurations.

A fresh three-repeat asynchronous R2D2 throughput probe measured **0.7633 +/-
0.016 eval/s** median (0.7179, 0.7780, and 0.7633) at 8 ranks with the
automatic three-thread setting. Each run used 39-43 evaluations and recorded
**3.47 GB** peak worker memory; the 7.25-7.30 s/evaluation imaging stage remains the
dominant cost. This refresh confirms the prior 0.7251-0.7738 range rather than
isolating a new code speedup.

A fresh current-commit three-repeat control measured **0.7779 +/- 0.0009
eval/s** (0.7812, 0.7779, and 0.7779) at 8 ranks with the automatic
three-thread setting. The imaging stage remained **7.30-7.35 s/evaluation**
and peak worker memory stayed **3.47 GB**; this is a baseline refresh, not an
isolated code speedup.

A fresh three-repeat default R2D2 probe measured **0.728 eval/s** median
(0.7274, 0.7280, and 0.7296) at 8 ranks, with **3.47 GB** peak worker memory
and 7.43 s/evaluation in the image container. This is a baseline refresh, not
a claimed regression or speedup; model inference remains the dominant cost.

A fresh explicit four-thread probe measured **0.7747 eval/s** median (0.7666,
0.7747, and 0.7753) at the same 8 ranks, versus **0.728 eval/s** for the
three-thread baseline, with unchanged **3.47 GB** peak worker memory. Four
threads remains an explicit candidate, not the automatic default: this was a
non-interleaved probe and oversubscribes the nominal 20-CPU host.

A matched explicit four-thread probe measured **0.7595 eval/s** median (0.7548,
0.7595, and 0.7652) at the same 8 ranks, with **3.47 GB** peak worker memory.
It remains a useful explicit setting but does not justify changing the portable
automatic three-thread default.

The latest R2D2 phase profile measures **6734.6 ms/evaluation** for 25 model
updates and **99.0 ms/evaluation** for residual computation. A 512x512 CPU
`_model_prev` clone costs about **0.221 ms** with one Torch thread, or roughly
0.00013% of the model-update stage; this dead-work removal is therefore not a
useful speed target.

The latest three-repeat explicit four-thread R2D2 probe measured **0.7676
eval/s** median (0.7670-0.7686) at 8 ranks, with **3.47 GB** peak worker memory.
This is consistent with the prior 0.7595-0.7738 range and does not justify
changing the portable automatic three-thread default.

A fresh three-repeat explicit nine-thread R2D2 probe measured **0.7554 +/-
0.0055 eval/s** at the same 8 ranks, versus **0.7676 eval/s** at four threads,
with unchanged **3.47 GB** peak worker memory. Nine threads is rejected: extra
oversubscription increases model-update time without reducing memory.

A fresh three-repeat explicit eight-thread R2D2 probe measured **0.7702 +/-
0.0144 eval/s** (0.7509-0.7796) at the same 8 ranks, with unchanged **3.47 GB**
peak worker memory. It is consistent with four-thread performance and does not
justify changing the portable automatic three-thread default.

A fresh three-repeat explicit ten-thread R2D2 probe measured **0.7398 eval/s**
(0.7251, 0.7398, and 0.7406) at the same 8 ranks, with unchanged **3.47 GB**
peak worker memory. Ten threads is slower than the four-thread candidate and
closes the next point in the thread-count sweep; four threads remains the
fastest measured explicit setting on this host.

A fresh three-repeat checkpoint-backed R2D2 control measured **0.7769 +/-
0.0034 eval/s** (0.7735, 0.7843, and 0.7769) at 8 ranks and **3.47 GB** peak
worker memory. It is consistent with the existing best result and does not
justify a runtime change.

A fresh three-repeat explicit eleven-thread R2D2 probe measured **0.7204 eval/s**
(0.7130, 0.7204, and 0.7281) at the same 8 ranks and **3.47 GB** peak worker
memory. Eleven threads is 6.2% slower than the latest four-thread control at
0.7676 eval/s, so the thread-count sweep continues to reject further
oversubscription.

## What is priced but deliberately not taken

| lever | worth | why not |
|---|---|---|
| `-mgain 0.9` | +20% evals/s | Not result-preserving for `peak_flux_abs_error_jy` or `sigma_res`, and it is the experiment definition every archived run was scored under. Now `./ri search --mgain 0.9`, default still 0.8. [clean loop](nested-sampling-clean-loop.md) |
| Dropping the w-gridding cube | -29% on the binary | **Closed.** The ignored-`w` phase error exceeds ducc0's own 1e-4 epsilon on 5962 of 5962 evaluations, so no lossless per-evaluation rule can skip it. [run scaling](nested-sampling-run-scaling.md) |
| Raising the 65W RAPL package limit | ~+26% evals/s | **Closed.** Docker here is rootless, so `--privileged` still maps to an unprivileged user and cannot write it. [power limit](nested-sampling-power-limit.md) |

## Where the remaining time goes

At 126 evals/s and 143 ms/evaluation, 5.3% is unaccounted; a
`--mpi-procs 1` control attributes 1.9% to harness Python. **No idle time
remains:** PolyChord costs 3-7 us per likelihood call at every `nlive`.

59% of an evaluation is ducc0's gridding and degridding, and
[the gridder floor](nested-sampling-gridder-floor.md) prices that as
arithmetic. The pass count is the only handle on it, which is what `-mgain`
moves. See [the evaluation floor](nested-sampling-evaluation-floor.md) for the
current budget and the six further avenues closed with numbers.

## Reading your own run

Every run already carries these profiles:

```bash
./ri profile <run>              # per-stage budget and worker utilisation
./ri profile <run> --phases     # inside the wsclean binary, from its own -log-time output
./ri profile <run> --over-time  # why evaluations/second falls as a run goes deeper
```

Throughput falls because nested sampling reaches longer observations and more
channels: `70.7 ms + 4.58 us x visibilities` per evaluation. See [the cost
model](nested-sampling-cost-model.md).

## Keeping what is on this page

The table above is history: each row was measured once, against the commit
before it. `./ri bench` is the standing version - what this commit costs on
this machine now, with an error bar that narrows as repeats accumulate, so the
next change is measured rather than argued about and a regression shows up as
a column rather than a surprise. See
[benchmarks](nested-sampling-benchmarks.md).

## The rest of the pages

Chronological; each page starts where the last stopped.

| page | what it settles |
|---|---|
| [benchmarks](nested-sampling-benchmarks.md) | `./ri bench`: throughput per commit, machine and settings, and the three measurement traps its preset is pinned against - asynchronous draw variation, the cold first run, and a window too short to mean anything. |
| [throughput](nested-sampling-throughput.md) | The running log of iterations 1-14: idle ranks, the asynchronous-MPI switch, rank 0, the AVX2 build, the constant predict, the two constant columns, the scratch tmpfs. Long, and the older figures in it are superseded. |
| [run scaling](nested-sampling-run-scaling.md) | What a bigger run costs. A 10x `--nlive` costs nothing per evaluation; the ceiling on this host is ~850k evaluations, set by rank 0's memory while it builds `summary.json`. |
| [power limit](nested-sampling-power-limit.md) | The host's 65W RAPL cap, why it is not thermal, and the 8-second averaging window behind the burst-clock measurement trap. |
| [evaluation budget](nested-sampling-evaluation-budget.md) | The first production-concurrency decomposition, and the flag screen that closed WSClean's own knobs. |
| [I/O placement](nested-sampling-io-placement.md) | Where an evaluation's 2.4 MB goes, and why A/B arms must run *simultaneously* - sequential ones give consistent ~4% false positives on this host. |
| [clean loop](nested-sampling-clean-loop.md) | `-mgain`, and its exact science cost across all eight metrics on 600 real evaluations. |
| [MS open](nested-sampling-ms-open.md) | casacore re-attaching every subtable on each of the ~16 parent-MS opens, and why the avenue is closed from the simulator's side. |
| [WSClean patches](nested-sampling-wsclean-patches.md) | How `docker/wsclean/patches/` works - `git apply` in the Dockerfile, hashed into `ri.build-inputs`, and a `WSCLEAN_GIT_TAG` bump that breaks a patch fails the build on purpose. |
| [zygote](nested-sampling-wsclean-zygote.md) | The fork server: 27 ms of every 163 ms process ran before `main()`. Also why an archived run's `image_binary` column is ~5 ms low against a run made since. |
| [phase profile](nested-sampling-phase-profile.md) | `-log-time` by default, and `./ri profile --phases`. Read it on *medians*. |
| [cost model](nested-sampling-cost-model.md) | Why throughput falls through a run, and `--over-time`. |
| [simulate stage](nested-sampling-simulate-stage.md) | The MeqTrees half, and the confirmation that nothing measurable is left in it. |
| [shared MS open](nested-sampling-shared-ms-open.md) | `patches/0003`, counted with a 30-line `LD_PRELOAD` shim over `open64` - no rebuild, no profiler. |
| [FFTW planner](nested-sampling-fftw-planner.md) | 63 transform plans built and thrown away per process. Note the four warmed sizes derive from `DEFAULT_IMAGE_DIM = 128`; a stale list costs the speedup, not a result. |
| [gridder floor](nested-sampling-gridder-floor.md) | ducc0's own timer tree, and `patches/0004`. Two arms' FITS files never byte-compare equal - the header carries the `-name` path, so compare data blocks. |
| [row blocks](nested-sampling-row-blocks.md) | `patches/0005`: 29484 casacore column reads an evaluation down to 27. |
| [process warm-up](nested-sampling-process-warm-up.md) | The 5 ms before the first visibility - and the unsigned-arithmetic bug that makes WSClean fit its restoring beam on a quarter of the box it means to use. A real failure mode, not result-preserving to fix. |
| [disk footprint](nested-sampling-disk-footprint.md) | Bytes rather than clock. A run is `~17 x nlive x num_repeats` evaluations, and `evaluations/` needs no sharding. |
| [evaluation floor](nested-sampling-evaluation-floor.md) | The current budget, `--mgain` as a flag, and the last six avenues closed. |
eval/s** for three threads and **0.7730 eval/s** for four threads (3.47 GB
