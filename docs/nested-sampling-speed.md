# Making a nested-sampling search faster: the index

Current operating points on this host, from `./ri bench`:

| search | evaluations/second | per-worker memory | ranks x threads |
|---|---:|---:|---|
| WSClean | **109** | 34.5 MB | 20 x 1 |
| R2D2 | **0.91** | 3.47 GB | 15 x 2 |

Thirty-three profiling rounds got there. Values in the tables below are
per-round baselines, except the two figures above; pre-round-18 rates are
historical and roughly half the current rate.

## Round 33: R2D2 is 2.1x faster, WSClean is at its floor

R2D2 went from **0.436 to 0.913 evaluations/second** on the default benchmark
preset - a **2.1x** speedup - with per-worker memory unchanged at 3.47 GB, so
the rank budget a host can afford is unchanged too. Ten changes contribute; the
four that carry it are all in how a checkpoint reaches the U-Net, which the
phase profiler identified as **98.3%** of logged image time.

WSClean did not move, and the honest reading of thirty-odd controls is that it
should not have: it entered this round at ~110 evaluations/second after
thirty-two rounds of work, and every lever tried since is either below this
host's ~5% run-to-run variance or not result-preserving. What that variance is,
and the probes it swallowed, are in [what was measured and rejected](#what-was-measured-and-rejected).

The measurement machinery is the other half of the round. Group medians replaced
means, the ledger records memory and idle fraction, arms alternate run by run
instead of running as blocks, and the group key now includes the image ids - so
a rebuild between two rows can no longer make them look like one comparison.
Without those, several of the R2D2 results below would have been unreadable.

## What shipped

Cumulative. Rounds 1-32 first, then this round.

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

### Round 33: R2D2

Read the column as a chain: each row is measured against the row above it, on
the default benchmark preset at 8 ranks unless it says otherwise.

| change | where it lives | measured worth |
|---|---|---|
| R2D2 checkpoints loaded once and shared by forked workers | `r2d2_serve.py` | Removes a 25-checkpoint `torch.load` per evaluation; first checkpoint-backed control 0.4372 eval/s |
| Model-load garbage collection removed | `patches/skip-model-load-gc.patch` | 25 checkpoint swaps 1.05 s to 0.235 s; **0.4372 to 0.5622 eval/s (+28.6%)** |
| Checkpoint weights assigned by reference, not copied | `patches/assign-checkpoint-weights.patch` | **0.5622 to 0.5948 eval/s (+5.8%)**, three controlled runs |
| Per-rank thread share rounds up instead of down | `run-nested-sampling-r2d2.sh` | **0.6198 to 0.7249 eval/s (+16.9%)** |
| Automatic threads follow the memory-clamped rank count | `run-nested-sampling-r2d2.sh` | 15 ranks x 2 threads instead of 20 x 1: **0.7374 to 0.9092 eval/s (+23.3%)** |
| The unread final residual is no longer computed | `patches/remove-final-residual.patch` | Removes one forward and one adjoint transform per evaluation |
| Checkpoint key normalisation cached | `r2d2_serve.py` | 4.36 to 0.01 us per repeated 24-key lookup |
| FINUFFT plans reused across sequential evaluations | `r2d2_serve.py` | 128x128 plan setup 0.86 ms fresh to 0.091 ms retargeted (9.5x) |
| Deterministic MKLDNN kernels | `r2d2_serve.py` | 128x128 U-Net 65.0-70.6 to 59.7-60.3 ms/forward (~8%), bit-identical output |
| `torch.inference_mode` on the inference paths | `r2d2_serve.py` | 8.589 to 8.467 ms/forward (1.4%) |
| `.mat` conversion skips zlib compression | `ms_to_r2d2_mat.py` | 8.59 to 0.39 ms for 12k visibilities |
| `.mat` output follows the MS onto tmpfs scratch | `polychord_r2d2.py` | Removes the conversion's cross-device write |
| `.mat` conversion avoids broadcast temporaries | `ms_to_r2d2_mat.py` | 1.13 to 1.09 ms for 1404 visibilities; weight expansion 1.802 to 1.657 us (8.0%) |

The three `patches/` entries are `git apply`ed into the R2D2 tree at image build
time, the same mechanism WSClean's patches use, and `scripts/build.sh` now hashes
`docker/r2d2/patches` into the image's build-input label so editing one rebuilds.

### Round 33: WSClean and the harness

Individually small and, apart from the metrics stage, none of them is separable
from host variance end to end. They are here because they are strictly less
work for an identical result.

| change | where it lives | measured worth |
|---|---|---|
| `sigma_res` FITS files load only when the objective uses `sigma_res` | `polychord_wsclean.py`, `polychord_r2d2.py` | Metrics stage 1.03 to 0.46 ms/evaluation |
| The dirty image is not written when the objective ignores it | `polychord_wsclean.py` | One fewer FITS write per evaluation |
| NaN-free image RMS uses a BLAS dot product | `common.py` | 25.2 to 7.8 us per RMS call |
| Off-source metric masks are cached per (shape, source) | `common.py` | 25.2 to 3.6 us per mask |
| RMS and relative-L2 share one residual norm and one buffer | `common.py` | Removes a full-image copy per evaluation |
| Row-block copy metadata cached | `patches/0005` follow-up | 71.06 to 71.90 eval/s over six controls; inside variance |
| Progress scans classify both counts in one GNU `find` walk | `progress-bar.sh` | 0.232 to 0.073 s for 20 scans of a 635-evaluation run; BSD `find` keeps the two-pass fallback |

### Round 33: the benchmark and profiler

The reason the R2D2 numbers above are trustworthy and the WSClean ones are
honest about being noise.

| change | where it lives | why |
|---|---|---|
| Groups report a median with an IQR-based error bar | `bench.py` | One long-tail run used to move a three-repeat mean by more than most effects are worth |
| `--interleave SETTING A B` | `bench.py`, `ri` | Arms alternate run by run, so both meet the same host. Sequential arms give consistent ~4% false positives here |
| The ledger records peak memory, busy wall and idle fraction | `bench.py` | A 42.8 eval/s outlier turned out to be 69.0% idle at a *normal* 101.1 ms/evaluation image cost - a scheduler stall, not a regression |
| The group key includes the three image ids | `run-config.sh`, `bench.py` | The images bake the code, so two rows at one commit could otherwise compare different binaries |
| `--timeout` stops the whole process tree | `bench.py` | MeqTrees can hang; a hung probe used to cost the session |
| `throughput` and `production` presets | `defaults.toml` | `production` is the real workload: `nlive 150`, `num-repeats 15`, no `max-ndead` |
| `./ri profile --r2d2-phases` | `profile-nested-sampling-run.py` | Splits model update from residual computation. This is what identified inference as 98.3% of R2D2 image time |
| Phase tables report a median and a p90 | `profile-nested-sampling-run.py` | One slow evaluation no longer chooses the next optimization target |
| `run.env` is parsed with `shlex` | `bench.py` | A metric expression with spaces used to truncate its group key |
| `./ri bench \| head` exits cleanly | `bench.py` | `BrokenPipeError` is normal report consumption |

Two rounds bought run *size* rather than speed - see
[disk footprint](nested-sampling-disk-footprint.md) and
[run scaling](nested-sampling-run-scaling.md):

| change | measured worth |
|---|---|
| `sim.ms` deleted as the evaluation is scored | evaluation directory 3.4x smaller |
| WSClean's four unread FITS images pruned | another 3.94x - 393.6 KB to 99.9 KB, so this host's ceiling went from 477k evaluations to 1.88M |
| A resume keeps each adopted objective, not each record | 62 GB down to 5.4 GB, which is the difference between the target run finishing and being OOM-killed |
| Progress-bar redraw backed off to 9x its own cost | 44% of a core down to ~12%, and no longer growing with the run |

## What is priced but deliberately not taken

| lever | worth | why not |
|---|---|---|
| `-mgain 0.9` | +15-20% evals/s | Not result-preserving for `peak_flux_abs_error_jy` or `sigma_res`, and it is the experiment definition every archived run was scored under. Now `./ri search --mgain 0.9` and `./ri bench run wsclean --interleave NS_WSCLEAN_MGAIN 0.8 0.9`; default still 0.8. [clean loop](nested-sampling-clean-loop.md) |
| Dropping the w-gridding cube | -29% on the binary | **Closed.** The ignored-`w` phase error exceeds ducc0's own 1e-4 epsilon on 5962 of 5962 evaluations, so no lossless per-evaluation rule can skip it. [run scaling](nested-sampling-run-scaling.md) |
| Raising the 65W RAPL package limit | ~+26% evals/s | **Closed.** Docker here is rootless, so `--privileged` still maps to an unprivileged user and cannot write it. [power limit](nested-sampling-power-limit.md) |
| R2D2 at four OpenMP threads | +5-6% evals/s at 8 ranks | Fastest measured explicit setting on this 20-CPU host, but it oversubscribes: the automatic `ceil(cpus / ranks)` has to be right on smaller-rank and larger hosts too. `--omp-threads 4` still asks for it |

## What was measured and rejected

Nothing here changed the tree. Each line is the shortest form of a probe that
was run, so it is not run again.

**This host's resolution.** Fifteen WSClean control repeats at one commit
measured 109.0 +/- 1.5 evaluations/second over a 102.1-115.9 range, and a
three-repeat control spans ~5% routinely. Anything smaller than that needs an
interleaved probe or a direct stage timing, not an end-to-end A/B.

**WSClean rank scaling** (`--interleave NS_MPI_PROCS`, three repeats each,
34.0-34.9 MB peak worker memory at every point, so no rank count buys memory):

| ranks | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16 | 18 | 19 | **20** | 21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evals/s | 15.9 | 17.3 | 32.2 | 45.7 | 55.8 | 67.7 | 71.8 | 74.6 | 87.4 | 89.1 | 96.0 | 107.8 | 109.3 | 113.4 | **109** | 104.7 |

Throughput saturates at ~18-20 ranks and 21 is 2.8% slower. 20 stays the
default; 18, 19 and 20 are statistically indistinguishable, and none of the
lower counts trades throughput for memory.

**R2D2 thread scaling** at 8 ranks, 3.47 GB per worker at every point:
2 threads 0.657, 3 (automatic) 0.725, **4 0.813**, 5 0.818, 6 0.787, 7 0.796,
8 0.770, 9 0.755, 10 0.740, 11 0.720 evaluations/second. Beyond four, extra
oversubscription costs model-update time and returns no memory.

**R2D2 rank/thread packing.** 16 x 2 measured 0.9077 against 8 x 4 at 0.8133
(+11.6%), and 16 x 2 against 16 x 1 at 0.7374 (+23.3%). 15 x 2 and 16 x 2 tie
at 0.911, so 15 is the default: it saves 3.47 GB for nothing.

**R2D2 inference alternatives, all closed.** TorchScript tracing 1.313 s
against 1.324 s eager, and `optimize_for_inference` cannot execute this U-Net
at all (`Currently Mkldnn tensor does not support view`). Disabling MKLDNN is a
36% regression. Denormal flushing is 6.65% slower. `OMP_PROC_BIND=spread` is
2.2x slower and `close` is identical to the unset default. A larger oneDNN
primitive cache is 0.3%. `torch.compile` needs a `g++` the runtime image does
not have. PyTorch eval mode measured 0.8103 against 0.8126 and its patch was
removed. Removing legacy weight-norm reparameterisation: 114.96 against
115.08 ms/forward.

**R2D2 swap machinery is finished.** All 25 checkpoint swaps now take 11.0 ms,
about 0.44 ms/evaluation, under 0.2% of the model-update phase. The 512x512
`_model_prev` clone it might have replaced costs 0.221 ms. Model inference is
the only R2D2 target left.

**WSClean `-march=native`** measured 111.6 against 114.8 evaluations/second for
matched `x86-64-v3`, with one 36.9 eval/s outlier in the native arm. Not
retained; `./ri bench run wsclean --native` still builds it.

**WSClean `-log-time`** costs nothing measurable: 130.08 ms against a 128.95 ms
baseline and a 126.66 ms null in a 756-command shuffled replay, and 109.4
against 115.8 evaluations/second in an interleaved three-repeat probe - both
inside the null. It stays on, because `--phases` and `--over-time` need it.

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

With the checkpoint archive supplied through external `CHECKPOINTS_DIR`, an
interleaved four-thread R2D2 rank probe measured **0.6934 eval/s** at 5 ranks
and **0.7983 eval/s** at 8 ranks, both at **3.47 GB** peak worker memory. The
two-repeat sample is directionally clear but below production-strength size;
8 ranks remains the setting to test next. The latest phase profile attributes
**6568 ms** per evaluation to 25 model updates and **126 ms** to residual
computation, so U-Net inference remains the dominant optimization target.

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
