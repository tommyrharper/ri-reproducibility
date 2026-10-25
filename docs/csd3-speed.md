# Search speed on CSD3

Speed work for the 9-parameter space on CSD3. Newest round last.
[The speed index](nested-sampling-speed.md) covers the 5-parameter work on
the Hetzner host.

## Where the time went

`./ri profile` on the two most recent finished 9-parameter runs:

| run | simulate (MeqTrees) | imager | idle |
| --- | ---: | ---: | ---: |
| `wsclean-vlaa-20260922T144310Z` (4 ranks, 50 evals) | 1.40s, 44.6% | 0.95s, 30.2% | 22.5% |
| `r2d2-vlaa-20260922T174616Z` (24 ranks, 71,395 evals) | 1.94s, 34.7% | 2.87s, 51.4% | 11.5% |

On the Hetzner host simulate was ~13ms. The cause is the parameter space, not
CSD3. `source_l_pixels`/`source_m_pixels` move the source off the phase centre,
so every evaluation now runs a real MeqTrees predict. Before, the phase-centre
case wrote a constant. `integration_seconds` and `declination_deg` also make
nearly every evaluation a new MS shape, so the skeleton cache misses and
`makems` runs every time.

## Round 1: analytic predict

Probe: `scripts/probe_simulate_stage.py`, `simulate()` with each step timed, serial, one warm worker, on an icelake
node (Xeon 8368Q). Input is 48 parameter sets taken evenly from the 71k-eval
R2D2 run (mean 53.5k rows).

| step | before mean / median | after mean / median |
| --- | ---: | ---: |
| `makems` (cache miss, every eval) | 0.662s / 0.456s | 0.667s / 0.444s |
| MeqTrees predict | 0.794s / 0.632s | - |
| rest of the fill (noise, write) | 0.046s / 0.034s | 0.055s / 0.036s |
| **simulate total** | **1.515s / 1.133s** | **0.734s / 0.492s** |
| worker warm-up before the first request | 2.6-3.3s | 0.07s |

Both `makems` and the predict are linear in rows, at ~11us per row each. The
predict also has a ~0.25s floor.

The change: `point_source_visibilities()` writes
`I exp(+2 pi i f/c (u l + v m + w (n-1)))` on XX and YY with numpy, and the
worker no longer starts a meqserver. MeqTrees Meow reads UVW from the MS and
applies no smearing, so this is the whole of its predict.
`self_check_analytic_predict()` runs the real MeqTrees predict over 7 cases
(on and off centre, declinations -15 to 75, 1-120s integrations, up to 632k
rows). Worst difference: **4.7e-10 of the flux**, i.e. float32 rounding. The
phase-centre cases are bit-identical.

Other effects: MeqTrees deadlocks (one every 2,000-5,000 predicts,
[robustness](robustness.md)) and the `VisDataMux not found` failures that
survived their retry in the big run can no longer happen at run time. The
meqserver code is still there, used only by the self-check.

Expected run-level effect (measured in round 3): ~0.8-1.0s fewer
worker-seconds per evaluation, i.e. ~+25% evals/s for R2D2 on CPU and ~+45%
for WSClean.

Next: `makems` is now ~90% of simulate.

## Round 2: makems only for unseen observations

makems costs ~0.07s plus ~3.3ms a timestep: it converts all 27 antenna
positions to J2000 UVW every timestep. Timestep k of an observation does not
depend on its length, so the rows of a short observation are, bit for bit, the
first rows of a long one at the same declination and integration.

`make_ms()` caches, per (declination, integration), makems' per-antenna UVW
(relative to antenna 0) and the per-timestep times in an `.npz` beside the
skeleton cache. A later evaluation no longer than the cached one copies a
one-timestep template for its channel count, `addrows`, and writes UVW, TIME,
ANTENNA1/2, INTERVAL, EXPOSURE, the FIELD directions and OBSERVATION.TIME_RANGE.
A miss runs makems as before, then caches what it built (13ms), replacing a
shorter entry. `save_observation()` refuses to cache anything it cannot rebuild
bit for bit. `self_check_observation_prefix()` compares every column of MAIN
and the kept subtables, and the storage layout, with a fresh makems build after
the fill, over 8 hit/miss/grow steps.

Same probe and 48 parameter sets, icelake 8368Q. "Warm" first caches each set's
(declination, integration) at 20 minutes, the steady state of a long run:

| | cold cache | warm cache |
| --- | ---: | ---: |
| `makems` | 0.607s | 0.013s (templates: one per channel count) |
| extend template | - | 0.023s |
| fill | 0.047s | 0.048s |
| **simulate total** | **0.679s / 0.475s median** | **0.088s / 0.056s median** |

The hit rate grows with run length. Replaying uniform prior draws (the worst
case: a converging run revisits a narrower region) through the cache, the
share of timesteps still built by makems is 94% at 500 evaluations, 75% at
2,000, 37% at 10,000 and 9.5% at 71,395, the size of the last big R2D2 run.
There are 910 (declination, integration) keys; at full coverage the cache is
~200MB of `/dev/shm`.

## Round 3: end to end

`./ri bench run <imager> --repeat 2` on rounds 1+2 (`c7133fb`) and on the
commit before them (`076e395`, a second worktree), both started in one icelake
job so the arms race on the same node. Same seed, so both arms draw the same
evaluations; log(Z) was identical.

| | before | after | change |
| --- | ---: | ---: | ---: |
| WSClean, 16 ranks, 726 evals | 3.64 evals/s | 6.66 evals/s | **+83%** |
| - simulate per eval | 1.30s | 0.48s | |
| - WSClean per eval | 0.36s | 0.35s | |
| R2D2, 8 ranks, 50 evals | 2.18 evals/s | 2.66 evals/s | **+22%** |
| - simulate per eval | 1.01s | 0.48s | |
| - R2D2 per eval | 1.22s | 1.34s | |

Caveats. The WSClean baseline's second repeat ran alone after the other arm
finished, which flatters the baseline. Each R2D2 arm sized its threads for all
36 CPUs of the job (5 threads x 8 ranks, twice), so R2D2 was oversubscribed
in both arms, and more so in the faster one. These runs are short, so the
round-2 observation cache is cold: simulate is 0.48s here against ~0.09s warm.
Long runs will gain more.

How to repeat it: a job script `cd`s into each worktree and runs
`./ri bench run` in the background in both, with `NS_SBATCH=0`,
`RI_WORK_DIR=/rds/user/$USER/hpc-work` (job-env takes uv from the checkout's
parent dir, one level too deep for a nested worktree), and `SIF_DIR` and
`CHECKPOINTS_DIR` pointing at the main checkout.

`bench.py` fix: every CSD3 node shares one `/etc/machine-id`, so icelake and
ampere rows were grouped as one machine. Inside a Slurm job, the id now also
hashes the CPU model. Older CSD3 rows keep the shared id `39edb438`.

Next: WSClean is now 42% of a short WSClean run and R2D2 is 70% of an R2D2
run. In a long run the R2D2 share will be higher still.

## Round 4: R2D2 request overhead

`./ri profile --r2d2-phases` on round 3's R2D2 bench run: the 25 R2D2
iterations (model update plus residual) took 0.37s of a 1.35s imaging request.
Most of the request was spent around the algorithm, not in it.

Probe: `scripts/probe_r2d2_request.py`, `.mat` files from 24 of the 48 round-1 parameter sets, imaged serially by
one warm worker (`warm_imports()` as the pool runs it), 2 threads, under
cProfile, icelake 8368Q. Output on RDS, as in a run.

| per request | before | after |
| --- | ---: | ---: |
| U-Net forwards (~15 of them) | 0.47s | 0.46s |
| operator norm (Lanczos, ~21 NUFFT pairs) | 0.38s | - |
| other NUFFTs (dirty image, residuals) | ~0.2s | 0.22s |
| U-Net built with random weights | 0.17s | once per worker |
| FITS writes (4 files on RDS) | 0.28s | 0.23s |
| **request, mean of 23 warm** | **1.81s** | **1.10s (-39%)** |

- `imager.py` computes `target_dynamic_range` from the operator norm when the
  config leaves it unset. R2D2 reads that value only against a ground truth
  (`gdth_file`) or across several checkpoint realisations. We use neither, so
  the config now sets it and the solve never runs. This also removes the Lanczos
  patch, its upsampling plumbing and its self-check from `r2d2_serve.py`.
- `create_net_imaging` Kaiming-initialises a fresh U-Net every request. Each
  iteration's `load_net` then strictly assigns a checkpoint over every weight.
  `patch_net_reuse()` builds the net once per worker.

All 24 model images are bit-identical to before. 95 of the 96 output files are
too; one residual differs by 1.1e-16, which is FINUFFT's adjoint summation
order (see the nufft self-check).

End to end, `./ri bench run r2d2 --mpi-procs 8 --omp-threads 2 --repeat 3`,
both arms in one icelake job (the recipe from round 3; baseline `c6784f5`),
identical log(Z):

| | before | after | change |
| --- | ---: | ---: | ---: |
| evals/s (3 repeats) | 2.32 / 2.37 / 2.40 | 2.74 / 2.83 / 2.87 | **+19%** |
| R2D2 per eval | 1.55s | 1.23s | -0.32s |

The ranges do not overlap, and the baseline's last two repeats ran after the
new arm had finished, which flatters the baseline. The saving is under half the
serial probe's 0.7s. Two likely reasons: 32 R2D2 threads plus simulate workers
shared 36 cores, and these draws ran all 25 iterations where the probe's
averaged 15.

Next: the FITS writes are ~0.23s of the 1.1s. They go to the evaluation
directory on RDS/Lustre (open, stat and unlink are 5-15ms each there) and are
then mostly deleted. Writing them to the `/dev/shm` scratch and moving only the
retained ones should remove most of that.

## Round 5: evaluation files on tmpfs

The run directory is on RDS, which is Lustre. Single-file costs from a compute
node, 40 repeats each:

| op | RDS | `/dev/shm`, `/local` |
| --- | ---: | ---: |
| create + write 8KB | 42ms | 0.04ms |
| create empty | 25ms | 0.02ms |
| rename / unlink | 11ms | 0.02ms |
| mkdir | 4ms | 0.02ms |
| 4 FITS via astropy + 2 mkdirs (R2D2's output) | 212ms | 7ms |

An evaluation wrote ~14 files to its RDS directory: 6-8 logs,
`simulation.json`, `r2d2_config.yaml`, 4-5 FITS. Then, beyond its rank's 20
best and worst so far and 1 in 100, it deleted all but `metrics.json`. So
nearly every evaluation of a long run paid for ~14 creates and ~13 unlinks and
kept nothing.

Now everything an evaluation writes goes to the scratch tmpfs
(`NS_SCRATCH_DIR`) beside its MS. At scoring, `publish_evaluation_scratch()`
moves it to the evaluation directory only if the retention policy keeps it,
and rewrites the record's paths. A dead worker's logs are moved before the
run aborts (`salvage_evaluation_logs()`). `self_check_streaming_retention`
runs with and without scratch and asserts the same result.

End to end, both arms in one icelake job (baseline `3b778c8`), identical
log(Z). "Steady state" sets `NS_IMAGE_KEEP_ENDS=0 NS_KEEP_DETAIL_EVERY=1000000`.
Without that, a 50-eval bench keeps every evaluation, which a long run does not:

| | before | after | change |
| --- | ---: | ---: | ---: |
| R2D2 steady state, evals/s (8 ranks x 2 threads) | 2.74 / 2.78 / 2.80 | 3.03 / 3.21 / 2.95 | **+10%** |
| R2D2 per eval (stage sum) | 1.81s | 1.60s | -0.21s |
| WSClean steady state, evals/s (16 ranks) | 7.47 / 7.66 / 7.34 | 8.90 / 9.00 / 8.75 | **+19%** |
| WSClean `image_binary` | 310ms | 158ms | -49% |
| R2D2 defaults, all kept, evals/s | 2.76 / 2.80 | 2.79 / 2.86 | +1% |

WSClean gains most because its binary wrote five FITS and its reorder files
to RDS. With every evaluation kept, the moves cost about what the direct
writes did, so a short run neither gains nor loses.

Next: in these short benches simulate is still ~0.4s/eval, about a quarter of
R2D2's evaluation and 70% of WSClean's. That is round 2's observation cache
starting cold in every run. A long run warms it (0.09s), but a cache that
outlives the run would give short runs the same.

## Round 6: observation cache that outlives the run

Round 2's cache lived in the run's scratch, so every run started cold and a
short one paid makems for nearly every evaluation. `save_observation()` now
also publishes to `NS_OBSERVATION_CACHE_DIR`, and `load_observation()` reads
from it when the run's copy is missing or too short. `start-sidecars.sh`
points it at `$XDG_CACHE_HOME/ri/ms-observations/<meqtrees image id>`, since
makems and the image's IERS tables decide the rows; set it empty to turn it
off. `self_check_observation_prefix` now also runs a second "run" with a fresh
run cache and checks served entries against makems bit for bit.

Every (declination, integration) of the 9-param space at the longest
observation (20 min) is 910 observations, 183MB, built in 38s by 34 processes
each running `run_makems` + `save_observation` on one icelake node. That is what a long history of runs converges to.

End to end, both arms at once in one icelake job, steady-state retention
(`NS_IMAGE_KEEP_ENDS=0 NS_KEEP_DETAIL_EVERY=1000000`), identical log(Z). "off"
sets `NS_OBSERVATION_CACHE_DIR=` (round 5 behaviour); "full" points it at the
prefilled cache:

| | off | full | change |
| --- | ---: | ---: | ---: |
| WSClean evals/s (16 ranks) | 8.95 / 9.40 / 9.42 | 19.0 / 20.4 / 20.5 | **+115%** |
| WSClean `simulate` | 385-394ms | 66-74ms | -82% |
| R2D2 evals/s (8 ranks x 2 threads) | 2.93 / 2.96 / 2.88 | 3.52 / 3.43 / 3.55 | **+20%** |
| R2D2 `simulate` | 398-403ms | 69-71ms | -83% |

The "full" WSClean arm finished first, so the "off" arm's last repeats ran
alone: the gain is if anything understated.

Without a prefill the cache fills as runs go: each run adds the observations
it drew, at the longest length it drew. Next: prefill it once per meqtrees
image (38s on a node) so the first run is warm too.

## Round 7: record writes off the rank's critical path

Production runs are async (`NS_SYNCHRONOUS=0`), so this round measured that mode:
`./ri search wsclean` with 16 ranks, nlive 150 and steady-state retention, on 16
icelake CPUs, sampling per-process CPU from `/proc` every 10s:

- The node is CPU-bound. simulate workers use ~3.8 cores and WSClean ~9, out of 16.
  The 16 PolyChord ranks together use 1.25 cores, and rank 0 does not spin in MPI (0.00).
- Workers spent 85% of wall inside `evaluate()`. After `ended_epoch`, each
  evaluation still wrote its `metrics.json` (create, write, rename) to Lustre,
  ~50ms by round 5's numbers, with the rank idle throughout.

`write_records_in_background()` now hands that write to one thread per rank.
Records are serialized on the rank's own thread, and a failed write re-raises
there. The queue drains after `run_polychord` (before rank 0 reads the records
back), at exit, and in `abort_run` (bounded). `_strip_to_record` now skips
`metrics.json.partial`, so evicting an older evaluation cannot delete a write
that is in flight. See robustness.md for what a kill can lose.

A/B against `f96ff4b`, both arms in one 36-CPU icelake job and each pinned to
its own 18 CPUs with `taskset`, halves swapped every round. Async, so the
evaluation mix differs between arms. "Outside" is worker wall time not spent in
an evaluation, per evaluation:

| round | base outside | new outside | base util | new util | base evals/s | new evals/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.6ms | 14.7ms | 0.915 | 0.955 | 28.0 (cold) | 46.1 |
| 2 | 27.7ms | 15.0ms | 0.900 | 0.942 | 53.9 | 58.0 |
| 3 | 30.2ms | 10.9ms | 0.902 | 0.943 | 48.5 | 79.1 |

The saved 15-30ms/eval is 6-10% of a ~250ms WSClean evaluation, and ~2% of
R2D2's 1.35s (too small to show end to end). evals/s mostly tracks which half an
arm ran on: busy time per eval was 179-244ms on one half and 250-311ms on the
other. So use the utilisation and outside-time columns, not evals/s.

Dead ends this round, none of them bit-exact or worth it:

- R2D2 request at 1 thread (production shape: 24 ranks x 1 on sapphire) is
  1.64s. U-Net forwards are 0.81s (conv2d 0.6s) and NUFFTs 0.43s.
  `scripts/probe_r2d2_request.py` (now with `OUT_ROOT` for tmpfs outputs).
- U-Net per forward, cycling the 25 checkpoints: 43.7ms, and 41.8ms with one
  checkpoint reused, so weights missing cache cost little. The 2x2 and 4x4 layers
  (1024 channels) take ~16ms, and are bound by weight bandwidth.
  `torch.utils.mkldnn.to_mkldnn` prepacking: 78ms and not bit-identical.
  channels_last: 42.3ms against 42.6ms, not bit-identical. Keeping one net per
  checkpoint, instead of `load_state_dict` per iteration, saves ~1.2ms per
  forward (~1% of a request).
- FINUFFT 2.5.1 options (`spread_sort`, `kerevalmeth`, `fftw`, `nthreads`)
  timed on the real points of 3 requests. All but `spread_sort=0` on type 1 are
  bit-identical, and none is a consistent win: per-call times varied 2-3x on the
  shared node.
- Warm simulate (`PROFILE=1 scripts/probe_simulate_stage.py`, 48 sets) is
  ~70ms: `rng.normal` 32ms, casacore `putcol` 21ms, phase ramp 16ms. The noise
  draws cover all 4 correlations and fix the seeded stream, so drawing fewer
  would change every noise sample.

## Round 8: reading records back after the run

The 71,395-evaluation R2D2 run (job 36031480, sapphire, 24 ranks) spent 88
minutes, 23% of its 6h20m job, after PolyChord returned. PolyChord returned at
~23:37 (`total_wall_seconds` 17,331). `run.log`'s last write, `summary.json`,
was at 01:05. In between, rank 0 read every record back
(`load_evaluations_from_dir`) and pruned (`prune_run_artefacts`), one Lustre
round trip at a time, while all 24 cores sat idle. `./ri resume` pays the
same read on every rank before its first evaluation
(`adopt_completed_evaluations`).

Per-record cost on an icelake node, reading that run
(3000-6000 records per row):

| step | serial | 32 threads |
| --- | ---: | ---: |
| `glob("eval-*/metrics.json")`, per directory | 18.9ms | - |
| read one `metrics.json` (cold) | 85.8ms | 1.98ms (64 threads: 1.18ms) |
| prune's `is_dir` + 10 `is_file` per stripped record | 26.4ms | 4.3ms |

The glob was the surprise. `listdir` of the 71k names takes 0.15s, but the
glob opens every directory to find `metrics.json`: 22 minutes on its own.

Change: `read_evaluation_records()` lists the names once and reads
`<name>/metrics.json` directly on a 32-thread pool (`RECORD_READ_THREADS`). A
missing file is an in-flight evaluation, as before. The summary load and resume
adoption both use it. `prune_run_artefacts` runs its per-record work on the
same size of pool. Each record touches only its own paths.

Measured on the same node:

| | before | after |
| --- | ---: | ---: |
| load all 71,395 records | ~2h (serial estimate), 88 min end of run incl. prune | 122.5s |
| prune, 3000 real records | 85.4s (28.5ms each) | 13.0s (4.3ms each) |
| load, same 3000 | 17.5s | 2.5s |

Outputs are identical: the new loader returned the same 71,395 records, in
the same order, as the `evaluations` the old code wrote into that run's
`summary.json`. Old and new prune leave identical records. For a run that size,
the end of run falls from ~88 minutes to ~7, and a resume starts evaluating
~1.5h sooner.

Still open from the same run: PolyChord's own tail. From 22:50 to 23:37,
concurrency fell from ~23 workers to ~1, and each evaluation took 18-28s instead of
4-6s (sampled from every 20th evaluation id). That is the last ~15% of wall
time, spent almost idle. Round 9 takes it on.

## Round 9: idle cores in the final chains

All 71,395 records of the same run, binned by end time (10 min bins):

| minutes | evals | concurrency | s/eval | simulate | R2D2 | kilo-timesteps x channels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-250 (typical bin) | ~2,500-3,500 | 22.7 | 4-6 | 1.5-2.3s | 2.2-3.5s | 0.5-0.9 |
| 250-260 | 1,810 | 16.7 | 5.6 | 2.2s | 3.2s | 0.7 |
| 260-270 | 165 | 3.0 | 11.0 | 4.3s | 6.5s | 2.2 |
| 270-289 | 44 | 1.0 | 26 | 7.8s | 18s | 7.0 |

The last ~30 minutes were one rank finishing one chain. PolyChord waits for every
worker's chain once it stops handing out seeds. Eval ids 3248-3304 of one rank
ran back to back, ~75 evaluations (`<nlike>` 62-75 per dead point). That chain sat
in the expensive corner the posterior had converged on (1s integration,
7 channels, ~380k rows) and imaged on the 1 thread a 24-rank run gives each
rank, while 23 cores idled.

The same six parameter sets on an icelake node with the current code:
simulate is 0.60s (rounds 1-6), and the R2D2 request scales with threads:

| `ncpus` | 1 | 2 | 4 | 12 | 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| heavy request | 10.4-11.0s | 5.5s | 3.4s | 2.1s | 1.28-1.36s |

Change: each rank touches `$NS_SCRATCH_DIR/.busy/<pid>` for the duration of
an evaluation (`mark_evaluation_start` to `write_evaluation_record`).
`r2d2_thread_count()` writes `ncpus = max(R2D2_OMP_THREADS,
R2D2_MAX_THREADS // evaluations in flight)` into the request's config.
`R2D2_MAX_THREADS` defaults to the host CPUs in async runs. In synchronous
runs it defaults to `R2D2_OMP_THREADS` (off), because of the next paragraph.
R2D2 hands `ncpus` to torch. The sidecar's FINUFFT plans now take
`nthreads` from torch and are cached per count, because FINUFFT's own
OpenMP runtime stays at the sidecar's `OMP_NUM_THREADS`. Each record carries
`r2d2_threads`.

Not bit-identical: 1 thread is reproducible run to run, but 2 or 24 threads
move the model and residual images by up to 4e-7 relative, and the RMS
objective by <1e-7 relative. Dirty images and PSFs are unchanged. A
synchronous run keeps its fixed-seed repeatability. An async run already
depends on scheduling for its evaluation order.

End to end, production shape: 24 ranks x 1 thread on 24 icelake CPUs, async,
nlive 25, 15 repeats, 40 dead points, seed 4242, run in the order
base/new/new/base in one job against 8d03534:

| | base | new |
| --- | ---: | ---: |
| PolyChord wall | 144s (cold first run), 111s | 93s, 91s |
| tail (concurrency < 11 until the end) | 52s, 37s | 21s, 14s |
| R2D2 per eval in the tail | 1.81s, 1.83s | 1.00s, 0.86s |
| evals/s | 5.1, 6.7 | 7.9, 8.4 |
| log(Z) | -0.00195 | -0.00195 |

In those runs, 85% of evaluations got 1 thread and the rest 2-24, in dips as
well as at the end. The run is -17% wall against the warm baseline, but the
saving is a fixed tail, so its share falls as runs grow. For the 71k run's
final chain of ~75 heavy evaluations: ~26s each as run, ~12s with rounds 1-8
(0.6s simulate + ~11s R2D2 at 1 thread), ~2.5s with this round. That is a
projected ~15 minutes down to ~3. This projection is not measured end to end.

## Round 10: residuals without NUFFTs

A fresh profile of a round-9 run (24 ranks x 1 thread, async) put the R2D2
request at 93% of an evaluation. At 1 thread a typical request is ~1.2s: the
U-Net ~0.7s, and ~33 FINUFFT executes at ~11ms each, ~0.36s. Every R2D2
iteration computes its residual as `adjoint_op(forward_op(model))`, a type-2
then a type-1 NUFFT over every visibility, so that cost is linear in rows. On
the heavy corner the posterior converges on (~380k rows) it is ~80% of a 14s
request.

For a real image x, `Re(A^H A x)` is x correlated with
`g(d) = sum_k w_k^2 exp(+i k.d)` for offsets d in `[-(N-1), N-1]^2`. One
type-1 NUFFT onto 2N x 2N modes builds g (the Toeplitz embedding). Each
application after that is a 64x64 real FFT pair, independent of the visibility
count. The PSF is the same operator on a dirac.

Change: `patch_normal_operator()` in `r2d2_serve.py`. For the duration of
`R2D2.run`, `forward_op` defers and `adjoint_op` applies the convolution.
`get_psf` uses it too. The kernel is built once per request at FINUFFT eps
1e-9. Operators it cannot represent (GPU, complex images, preconditioning,
complex weights) stay on upstream's NUFFTs. The dirty image is unchanged.

Per request, one warm worker at 1 thread on an icelake node (`scripts/probe_r2d2_request.py`,
first request excluded):

| set | before | after | FINUFFT executes |
| --- | ---: | ---: | ---: |
| 24 typical (from the 71k run) | 1.16-1.18s | 0.80s (-32%) | 796 -> 48 |
| 6 heavy (1s integration, 7 channels) | 13.7-13.9s | 2.43s (-82%) | 318 -> 12 |

Not bit-identical, at the level of upstream's own NUFFT error (eps 1e-6 per
transform). Across those 30 requests: identical iteration counts in every
request, model images within 1.2e-6 of their peak, residual RMS within 1.5e-6
relative. The PSF and dirty normalisation move by <6e-7 through the PSF peak.
Kernel eps 1e-12 instead of 1e-9 moves results by <4e-7 more and costs
6ms (typical) to 0.12s (heavy) more per request.
`self_check_normal_operator` checks the convolution against upstream's
operator (3.4e-7 of peak) and that a preconditioned operator is refused.

End to end, round 9's production shape (24 ranks x 1 thread on 24 icelake
CPUs, async, nlive 25, 15 repeats, 40 dead points, seed 4242), against
bd0aa6e in two jobs, orders base/new/new/base and new/base/base/new:

| | base | new |
| --- | ---: | ---: |
| PolyChord wall, warm | 88s, 90s, 91s | 44s, 48s, 50s |
| PolyChord wall, first run in the job (cold) | 105s | 73s |
| R2D2 per eval, warm | 1.47-1.52s | 0.81-0.93s |
| evals/s, warm | 8.1, 8.3, 8.5 | 15.4, 15.8, 16.8 |
| log(Z) | -0.00195 | -0.00195 |

About -45% wall, ~1.9x throughput. The saving per evaluation grows with its
visibility count, so long runs, whose posterior sits in the heavy corner, gain
more than this short one.
Worker idle in these short runs is still 30-40%: nlive 25 caps concurrency at
~18 of 23 workers mid-run, plus the prior-to-sampling transition and the tail.
The 71k run was 88.5% busy, so this is a short-run artefact.
The U-Net is now ~85% of a typical request and the remaining target.

## Round 11: the U-Net on a full CPU node

At `NS_IMAGE_DIM=32` the U-Net (64 channels, 4 pools) runs its deep layers at
2x2 and 4x4 pixels against 124MB of float32 weights per checkpoint. So a
forward is mostly a stream of weights through the core, not arithmetic.
`scripts/probe_unet_forward.py`, one checkpoint, 1 thread, on this half
icelake node (38 CPUs, 34 of them on one socket), P copies at once:

| P copies | 1 | 12 | 24 | 36 |
| --- | ---: | ---: | ---: | ---: |
| U-Net ms/forward | 27.4 | ~30 | 40-41 | 41-54 |
| cache-resident 64ch conv, ms | 0.76 | | 0.84 | |
| streaming read, total GB/s | 14.6 | 113 | 185 | 263 |

A compute-bound conv loses only 10% at 24 copies (clock), but the U-Net loses
48%. The node is not out of bandwidth: 24 U-Nets pull ~74 GB/s where a plain
streaming read reaches 185. So each core's own weight stream is what slows
down (per-core memory latency and bandwidth under load). At 24 ranks a typical
request's ~15 forwards cost ~0.6s, not the 0.41s a lone probe shows.

Tried, all result-preserving or measured to the bit, none a win:

- oneDNN reorders every conv's weights from OIHW to `ABcd16b16a` on each call
  (`ONEDNN_VERBOSE=1`: ~4ms of a ~30ms forward). Prepacking with
  `mkldnn_reorder_conv2d_weight` + `torch.mkldnn_convolution` picks
  `Acdb64a`, which is reordered again on each call: 46-51ms and not
  bit-identical (4e-6).
- im2col + MKL `mm` with prepacked `[O, I*9]` weights for convs at <= 4x4
  pixels: bit-identical (!) but 27.4-27.7ms against 27.4. At <= 8x8: not
  bit-identical (3.9e-6). Under 24 copies: 40.0-41.1ms against 40.7, noise.
- `THP_MEM_ALLOC_ENABLE=1`: the node already runs THP `always`; 39.6-40.5ms
  against 40.2-41.1 under load, 27.2 against 27.4 solo.
- Batching B images that share a checkpoint reads the weights once: 27.2
  (B=2), 20.9 (B=4), 17.5 (B=8) ms/image solo, 46.8 (B=2) and 31.3 (B=4)
  ms/image under 24 copies. Not bit-identical (3-6e-6), and ranks would have
  to meet at the same R2D2 iteration of different evaluations through a shared
  server. A ~22% U-Net ceiling at B=4 is not worth that machinery.

On CPU the U-Net is at its memory floor for this layout. Rank x thread layout
does not change the bytes read per evaluation. What is left is the GPU.

### The GPU, re-measured after rounds 6-10

One `ampere` job (`gpu-q-14`: A100-SXM4-80GB, 32 cores of an AMD EPYC 7763),
round 9's production shape (async, nlive 25, 15 repeats, 40 dead points,
seed 4242), `R2D2_DEVICE=cuda` except the CPU arm. 18 ranks is the most the
card admits (`NS_R2D2_CUDA_MB_PER_RANK` 4096: every worker keeps all 25
checkpoints on the GPU).

| arm (order run) | evals/s | R2D2 per eval | simulate | log(Z) |
| --- | ---: | ---: | ---: | ---: |
| GPU 18 ranks (first in job, cold) | 8.6 | 1.61s | 110ms | -0.00021 |
| CPU 24 x 1 on the EPYC (first on `r2d2.sif`) | 6.0 | 1.77s | 137ms | -0.00195 |
| GPU 18 ranks | 12.5 | 0.92s | 97ms | -0.00018 |
| GPU 12 ranks | 13.2 | 0.54s | 121ms | 0.00221 |
| *icelake CPU 24 x 1, round 10* | *15.4-16.8* | *0.81-0.93s* | | *-0.00195* |

One A100 with its 32 cores does 12.5-13.2 evals/s: less than 24 icelake
cores (round 10), and GPU hours cost far more than CPU hours. The CPU arm's
log(Z) matches round 10's to the digit; the GPU arms differ inside the error
bar because the GPU's float32 conv moves objectives by ~5e-6 (see
`docs/cluster.md`), which reorders an async run. Per-eval R2D2 time rises
from 0.54s at 12 ranks to 0.92s at 18: separate CUDA contexts time-slice the
card, so ranks queue for it rather than share it. So for production at
`NS_IMAGE_DIM=32`, stay on icelake CPUs. The GPU would only pay with the
contexts merged (CUDA MPS, or one GPU server batching forwards for all ranks)
and the checkpoints held once instead of per worker; neither was tried here.

## Round 12: a production-shaped run, and the convert stage

Short benches use nlive 25. Here is round 9's production shape at the real
nlive: `NS_NLIVE=150 NS_NUM_REPEATS=15`, 24 ranks x 1 thread, async,
steady-state retention, 250 dead points, seed 4242. It ran on this job's half
icelake node (38 CPUs) at `d67cbef`. That was 8,512 evaluations in 423s,
log(Z) 0.0165 +/- 0.0017. Per 30s window (`.probe/timeline12.py`, which
reads the run's `metrics.json` records):

| window | evals/s | workers busy (of 23) | simulate | convert | R2D2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-30s | - | 22.4 | 158ms | 78ms | 3.20s (cold) |
| 30-360s | 21.8-26.1 | 22.9 | 64-97ms | 23-38ms | 0.80-0.88s |
| 360-390s | 21.8 | 19.1 | 94ms | 38ms | 0.76s |
| 390-420s (tail) | 9.3 | 5.9 | 107ms | 41ms | 0.51s |

- At nlive 150 the workers are never idle mid-run (22.9 of 23). Round 10's
  ~18-of-23 cap was a property of nlive 25. Worker time per evaluation
  (0.95s) explains the whole rate: 23 / 0.95 = 24 evals/s. So nothing is lost
  between evaluations. The R2D2 request is 87% of it, and round 11 showed
  that this is the U-Net's memory floor.
- The tail (the last chains) is ~45s, and the adaptive threads of round 9
  pick it up (5.9 threads per eval in that window).
- Every worker's first evaluation took ~40s (0.8s warm). This was the first
  R2D2 run on this node, so the pool's parent read the 6.2GB of checkpoints
  cold from Lustre. Run again warm, `warm_imports()` takes 4.5s and a first
  evaluation 4.4s. So it is once per job on a new node, ~35s, which is 0.2%
  of a 6h job. Not pursued: `posix_fadvise(DONTNEED)` does not reliably evict
  Lustre pages (a serial re-read took 19.8s once and 2.5s the next time), so
  a parallel-prefetch fix could not be measured cold here.

Dead end: the noise draws. They are 20ms of a 73ms typical simulate and
0.27s of a 0.6s heavy one. They are 11ns a sample (numpy 1.26
`Generator.normal`; `random_raw` alone is 4ns). `standard_normal(out=)` plus
an in-place scale is bit-identical but only 6% faster. A vectorised
reimplementation of numpy's ziggurat from `random_raw` is slower (287ms
against 121ms for 10.6M samples) because of the temporaries. The seeded
stream is sequential, so threads cannot split it. Only compiled code would
help, and that means rebuilding the image.

**Change.** The convert (`ms_to_r2d2_mat.py`) read all four correlations of
DATA and widened them to complex128 to use one. It now reads that one with
`getcolslice`. Timed in `meqtrees.sif` on the 48 typical and 6 heavy
parameter sets, old and new alternating, 2 timed repeats each:

| sets | old convert | new convert | `.mat` arrays |
| --- | ---: | ---: | --- |
| 48 typical (mean 53.5k rows) | 27.5ms | 25.1ms | identical, 48/48 |
| 6 heavy (~377k rows, 7 channels) | 211.6ms | 168.4ms | identical, 6/6 |

The files match byte for byte after savemat's 128-byte header, which holds
a creation timestamp. That is ~1.3% of a heavy evaluation (the converged
region of long runs) and 0.3% of a typical one: too small to see end to end,
so it is timed at the stage. `self_check_noise_weighting` now checks `y`
against the full DATA column for the first and last correlation.

What is left of the convert is WEIGHT: an IncrementalStMan column whose
`getcol` costs 15ms typical and 38ms heavy, every value 1.0. The simulator
never writes it. If simulate wrote the `.mat` itself from the arrays it
already holds, with the WEIGHT it knows, most of the remaining
25-170ms/eval would go, along with one worker round trip. That is the next
small R2D2 target. Its byte-identity check is the same as above.

## Round 13: simulate writes the `.mat`

**Change.** Round 12's next target. The R2D2 run used to send each
evaluation's simulate worker two requests: simulate (write the MS), then
convert (open the MS again, read DATA, UVW, WEIGHT and the SPW, write
`r2d2_data.mat`). Now `simulate_measurement_set(..., r2d2_mat=True)` adds
`--mat-path`, and the simulator writes the `.mat` from the arrays it just put
in the MS: the complex64 DATA (correlation 0), the UVW it read, its own
frequencies and noise sigma, and WEIGHT = 1.0, which makems leaves there and
`self_check_noise_weighting` asserts. `write_r2d2_mat()` in
`ms_to_r2d2_mat.py` is the one writer for both paths. The script still
converts any MS standalone. The convert request, its worker action and
`convert_ms_to_mat()` are gone, and so is the `convert` stage in new runs'
records. Reports already handle a missing stage.

**Stage timing.** `.probe/mat13.py` in `meqtrees.sif`, run in-process on
the round-1 parameter sets. Each set gets an untimed warm-up run of both
arms, then 2 timed repeats with old and new alternating. The old arm is
simulate plus `ms_to_r2d2_mat()`. The new arm is simulate with `--mat-path`.
Neither arm includes the worker round trip that the change also removes:

| sets | old (simulate + convert) | new (simulate --mat-path) | paired saving | `.mat` arrays |
| --- | ---: | ---: | ---: | --- |
| 48 typical, 96 pairs | 69.1ms mean / 49.2 median | 50.3 / 34.0 | 18.8ms | identical, 96/96 |
| 6 heavy, 12 pairs | 753.8ms / 753.8 | 649.3 / 670.7 | 104.4ms | identical, 12/12 |

"Identical" means that `u`, `v`, `y` and `nW` match in dtype and byte for
byte. `self_check_noise_weighting` now checks the same thing on every
`./ri self-check simulate`.

**End to end.** Production shape (24x1, async, nlive 25, 40 dead points,
seed 4242, steady-state retention) on cpu-q-553, with arms in the order
new/base/base/new. Base is `81bec61` unpacked with `git archive`.

| arm | evals/s | simulate | convert | simulate + convert | R2D2 | log(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| new (first in job, cold) | 12.9 | 110.1ms | - | 110.1ms | 954ms | -0.00195 |
| base | 14.2 | 93.8 | 32.3 | 126.1 | 860 | -0.00195 |
| base | 16.0 | 86.3 | 29.2 | 115.5 | 883 | -0.00195 |
| new | 16.1 | 104.3 | - | 104.3 | 887 | -0.00195 |

With 24 ranks running at once, the warm in-run saving is 11-22ms of worker
time per evaluation, ~1.5% of a ~1s evaluation. That is too small to show in
evals/s at the ~5% spread of short runs, so the stage timing
above is the evidence. log(Z) is the same in all four runs. The R2D2
request is now ~88% of worker time per evaluation, and simulate is ~10%.

## Round 14: WSClean at 36 ranks, and Lustre on the rank's critical path

**Profile.** A production-shaped WSClean run (36 ranks, async, nlive 150,
seed 4242) on half an icelake node ran at 88 evals/s with 20% of worker time
idle. Most of that idle time was not startup or tail. Mid-run, the number of
busy workers fell from 34 to 1 for up to 45s at a time, while one
evaluation's `simulate` took 14-45s. PolyChord's async root has no barrier
(`catch_babies` is an `MPI_RECV` from any source), so the root was not what
held them.

**Cause.** Sampling `/proc/<pid>/syscall` every 0.5s during a stall showed 34
ranks in `D` state in `openat` and one in `mkdir`, all on the run's
`evaluations/` directory on Lustre. strace on a rank showed two synchronous
Lustre metadata calls per evaluation on its main thread:

- `mkdir(eval_dir)` at the start of `simulate_measurement_set()`, inside the
  timed simulate. This is where the 14-45s simulates came from.
- `_strip_to_record(eval_dir)` opening the same, still empty, directory after
  scoring, to strip an evaluation that is not kept.

Both cost a few ms each, but when Lustre's metadata server is slow, every rank
blocks on them at once. With tmpfs scratch (round 5), everything an evaluation
writes goes to scratch anyway, so neither call does any work. Now, when there
is scratch, `simulate_measurement_set()` does not make the eval dir, the
retention step skips stripping the evaluation's own dir, and the dir is made
by whatever first writes into it: the record writer's thread (`_write_record`)
or publishing a kept evaluation. A 10s strace of a rank's main thread on the
new code covered 65 evaluations and showed zero syscalls on `/rds`; the old
code made 31 over 17 evaluations. Without scratch, nothing changes.

A Lustre write the size of a PolyChord update (1.9MB resume plus five 200KB
files, each written and renamed), done once a second for 10 minutes from this
node, took 91ms median, 354ms p99 and 0.51s at most. So rank 0's chain writes
are not where the stalls came from.

**Evidence.** All runs were on cpu-q-553, used 36 ranks with 400 dead points
unless noted, and had async draws. "Stalled" means 3s windows after startup,
before the last full window, with fewer than 10 of 35 workers busy. "Steady"
means evals/s over evaluations that ended 10-70s into the run.

| code | runs | runs with a stall | stalled / steady wall | steady evals/s (unstalled runs) |
| --- | ---: | ---: | ---: | --- |
| `ad78f5c` | 13 | 5 | 120s / 1326s (9.0%) | 106.9-126.3 |
| new | 5 | 0 | 0s / 426s | 105.2-118.4 |

The baseline column includes the diagnostic runs at 36, 38 and 40 ranks and
600 dead points. The last 4 were an interleaved new/base/base/new A/B against
`ad78f5c` from `git archive`. No stall hit either arm in that window, so the
A/B on its own shows only that the steady rate did not change (new 111.4 and
105.2, base 111.0 and 107.8). log(Z) agrees within its error in every run
(0.0945-0.0965, +/- 0.0009-0.0016). Async draws differ from run to run, so
these runs are not identical.

The stalls struck at random. Two runs in a row could be clean, or one run
could lose 69s of 153s. Their cost grows with the rate of evaluations, which
is why the R2D2 runs at ~24 evals/s (round 12) never showed one. Measured
against total worker time, the old code lost ~9% of WSClean throughput.

Notes from this round:

- More ranks than CPUs: 36, 38 and 40 ranks interleaved gave 116.8, 111.7/106.9
  and 126.3/118.9 steady evals/s. That spread is within draw noise, so no change
  was made (the default is the job's CPU count).
- In short runs the tail is set by whichever rank holds the last slice chain,
  and in these runs it varied from 15s to 60s. That decides `./ri bench`'s
  whole-run evals/s (72.5-97.5 here) far more than any code change. Use steady
  evals/s for A/Bs this short.
- In-flight evaluations on a run with scratch no longer have an eval dir on
  Lustre, so `./ri health`'s "in flight" count, which counts dirs with no
  `metrics.json`, now reads close to 0 while the run is live.

## Round 15: the simulate stage at 36 ranks (no change)

At 36 ranks the in-run simulate is ~86ms. In one process it is ~52ms: make_ms
19ms, noise 20ms, visibilities 6.4ms, putcol 1.9ms. With 36 probes running at
once it rises ~35% to ~70ms. A 2-correlation (XX, YY) MS gives byte-identical
WSClean images (144/144) but makes WSClean only 0.2-1.5% faster, and the noise
still has to be drawn over 4 correlations to stay bit-identical. So it was
dropped. Every simulate lever left changes results: drawing noise only for the
correlations that are used (about -13ms), or a float32 phase ramp.

## Round 16: one CPU per rank

**Profile.** At 36 ranks a WSClean evaluation costs a rank ~300ms of worker
time: `wsclean` 210ms (70%) and simulate 86ms. Inside the job's 38-CPU
cpuset, `mpirun` bound each rank to a NUMA node. 32 ranks got the 34 CPUs on
socket 0 and 4 ranks got the 4 CPUs the job has on socket 1. The rank's
simulate worker and WSClean zygote, which do the work, were not bound at all
and moved around all 38 CPUs.

**Change.** A rank waits while its workers run, so all three can share one
CPU. `pin_rank_cpu()` in `common.py` claims one CPU per rank by holding a
`flock` on `/dev/shm/ri-cpu-<n>.lock` for the rank's life. Two searches in one
allocation, such as simultaneous A/B arms, therefore never land on the same
CPU, and a crashed rank lets go of its CPU. The rank pins itself to that CPU,
and pins each pooled worker when it connects (`_connect_shell_started_worker`,
again after every reconnect). WSClean children forked by the zygote inherit
the pin. `mpirun` now runs with `--bind-to none`, so every rank sees the full
cpuset. If there are more ranks than CPUs, nothing is pinned.
`NS_PIN_CPUS=0` turns pinning off. This is WSClean runs only. R2D2 ranks run
multi-threaded U-Nets and are left alone.

**Evidence.** All runs were on cpu-q-553 with 36 ranks, nlive 150, 400 dead
points, async and seed 4242, and arms were interleaved. The other half of the
node was busy (load average ~60), so steady evals/s (evaluations ending 10-70s
in) is noisy. The stage times are compared with a per-run linear fit,
`time = a + b x (timesteps x channels)`, read at 1000 timestep-channels
(`.probe/fit16.py`). That removes the differences between async draws.

| arm | runs | steady evals/s | simulate @1k | `wsclean` @1k |
| --- | ---: | --- | --- | --- |
| unpinned | 8 | 99.7-119.0 (mean 111.2) | 115.7-117.9ms | 263.8-284.5ms (mean 278) |
| pinned (external `sched_setaffinity` probe) | 5 | 110.2-131.8 (mean 118.9) | 111.1-112.7ms | 268.1-280.3ms (mean 275) |
| pinned (`pin_rank_cpu`) | 1 | 113.3 | 112.7ms | 279.7ms |

The in-code arm also had two unpinned controls in the same run of four
(104.5, 106.8 evals/s; 116.2 and 117.5ms simulate). Its fourth run died when
the RDS file quota filled (below), and is not counted.

So simulate is ~4.5ms (4%) faster when pinned, with no overlap between arms in
14 runs. `wsclean` does not change. That is ~1.5% of worker time. The steady
evals/s means differ by +6%, but that is within the run-to-run spread, so the
claim is the 1.5%. log(Z) agrees within its error in every run
(0.0952-0.0964 +/- 0.0009-0.0014). The objective of a given parameter set does
not depend on which CPU it ran on.

Notes from this round:

- The run directories filled the RDS inode quota (1,048,576 files on
  `/rds-d6/user/th773`; check with `quota -s`). Each 36-rank, 400-dead-point
  WSClean benchmark leaves ~10k evaluation directories (~20k inodes) under
  `results/`, so ~50 of them fill the quota. Every write then fails with
  `EDQUOT`, and the search dies with no retry. Delete benchmark runs once
  their `benchmarks.jsonl` row is written.
- This round's 14 rows in `benchmarks.jsonl` have a `note` field naming the
  arm.

## Round 17: R2D2 ranks, memory and threads on a half node (no change)

All on cpu-q-553 (this job's 38 CPUs, 128GB cgroup limit) at `b081568`.

**WSClean at 36 ranks is all gridding and I/O inside WSClean.** `./ri profile
--phases` on a 150-dead-point run (45 logged evaluations): the `wsclean`
binary's wall is its logged phases plus ~3ms. Per evaluation: gridding 41ms
(4.07 passes), loading the reordered data 29ms, reordering 21ms, predicting
21ms (2.07 passes), loading metadata 9ms. Those are the same phases the
Hetzner [gridder floor](nested-sampling-gridder-floor.md) prices. No
CSD3-specific overhead is left outside them.

**A warm R2D2 request, one process, 1 thread** (`probe_r2d2_request.py`, 25
typical `.mat`s on tmpfs): 517ms. U-Net forwards ~432ms (15.5 per request),
checkpoint swaps 11ms (0.73ms each, timed without the profiler), Briggs
imaging weights 13.6ms, FINUFFT ~23ms. The Kaiming init happens once per
worker, not per request. With the `.mat`s on Lustre, as in earlier probes,
the same request is 760ms, so probe inputs must sit on tmpfs.

**Rank count.** nlive 150, 15 repeats, 150 dead points, async, seed 4242,
steady-state retention. `.probe/r17an.py` gives 30s windows:

| layout | steady evals/s (30-150s) | workers busy | R2D2 per eval | whole run | log(Z) |
| --- | --- | ---: | ---: | --- | --- |
| 38 x 1 | 31.7, 33.1 (then the tail) | 36.9 of 37 | 1.02-1.05s | 3,260 evals in 183s | 0.0145 +/- 0.0036 |
| 19 x 2 | 24.6, 25.7, 25.4, 26.0 | 17.9 of 18 | 0.60-0.63s | 3,969 evals in 171s | 0.0148 +/- 0.0030 |

One rank per CPU is ~27% faster mid-run than half the ranks at 2 threads.
The U-Net is memory-latency bound per core (round 11), so more cores pulling
weights at once beat more threads per forward. The 38 x 1 run lost its lead
in the tail (below), so its whole-run figure is lower. One pair only, so
treat the size as a pointer, not a measurement.

**Memory per rank is ~0.4GB, not 3.5GB.** Sampled every 5s from the job's
cgroup during the 38-rank runs: peak usage 30.7GB, of which RSS 15.6GB and
the rest page cache (mostly the 6.2GB of checkpoints). The pool parent loads
the checkpoints once and the workers share them after the fork, so each rank
adds ~0.4GB. `NS_R2D2_MB_PER_RANK` (3500) assumes every rank holds its own
copy. On this node that makes `--mpi-procs 38` print an OOM warning, and a
default run clamps itself to (128,060 - 4,096) / 3,500 = 35 ranks with
ceil(38/35) = 2 threads each, 70 threads on 38 CPUs. 35 x 2 was not measured
here. The 56-rank cap `NS_R2D2_MAX_RANKS` came from a sapphire scaling test
(E4) that was still rising at 56, and from the file quota. Retention now
leaves ~2 inodes per evaluation, so the quota no longer sets it.

**Thread count of a lone request** (the tail, where round 9's adaptive
threads give the last evaluations every CPU). Mean of 3 warm requests, alone
on this allocation:

| `ncpus` | 1 | 4 | 8 | 16 | 38 |
| --- | ---: | ---: | ---: | ---: | ---: |
| typical request | 0.59s | 0.29s | 0.23s | 0.22s | 0.34s |
| heavy request (1s integration, 7 channels) | 1.61s | 0.67s | 0.52s | 0.45s | 0.58s |

38 threads (34 on one socket, 4 on the other) are slower than 16. In the
38 x 1 run's tail, requests given 38 threads took 3.3-3.9s each, and 60s of
the 183s had 1-3 workers busy. The 19 x 2 tail capped at ~5 threads was
short. The lone-request table does not explain all of the 3.3-3.9s (the
tail's evaluations sit in the heavy corner), so it is not settled yet.

Next, in order:

1. Measure the default (35 x 2) against 38 x 1 here, and correct
   `NS_R2D2_MB_PER_RANK` to a shared-plus-per-rank model so the default
   becomes one rank per CPU.
2. Cap `R2D2_MAX_THREADS` (e.g. 16, or one socket) and A/B the tail.
3. On a whole node, test ranks above the 56 cap, up to one per CPU.

## Round 18: R2D2 memory is shared, so the default is one rank per CPU

All on cpu-q-553 (this job's 38 CPUs, 128GB) at `d7afc57` plus this change.
nlive 150, 15 repeats, 150 dead points, async, seed 4242 (`.probe/r18.sh`,
windows from `.probe/r17an.py`, tails from `.probe/tail18.py`).

**The old default against 38 x 1, interleaved.** With 3,500MB a rank the
budget clamped a default run to 35 ranks x 2 threads (70 threads on 38 CPUs):

| arm | run | steady evals/s (30-90s) | wall | log(Z) |
| --- | --- | --- | ---: | --- |
| old default, 35 x 2 | 1 | 26.6, 26.6 | 194s | 0.01481 +/- 0.00337 |
| 38 x 1 | 2 | 31.9, 31.9 | 149s | 0.01468 +/- 0.00340 |
| old default, 35 x 2 | 3 | 26.5, 27.4 | 164s | 0.01467 +/- 0.00343 |
| 38 x 1 | 4 | 30.9, 29.6 | 141s | 0.01465 +/- 0.00342 |

So one rank per CPU is ~15% faster at steady state, with 36.9 of 37 workers
busy against 33.9 of 34. log(Z) agrees in every run.

**Memory.** Peak RSS of the job's cgroup (`total_rss` in `memory.stat`,
which counts a page shared after a fork once), sampled every 5s:

| ranks | peak RSS |
| ---: | ---: |
| 8 | 7.7GB |
| 35 | 19.1GB |
| 38 | 18.2, 19.1, 21.0GB |

That is ~4.2GB paid once (the pool parent's checkpoints, which every
worker shares after the fork) plus 0.35-0.44GB a rank. The 5.8GB of
checkpoint files also sit in the cgroup's page cache, which is reclaimable.
The old 3,500MB a rank came from the Hetzner host, where each rank held its
own copy.

**The change.** `rank-budget.sh` now takes memory as `SHARED+PER`:
`NS_R2D2_SHARED_MB` 8000 plus `NS_R2D2_MB_PER_RANK` 600 (both rounded up
from the table, `NS_R2D2_MB` joins them). `ns_budget_ranks`,
`ns_budget_warn_if_over`, the retry re-sizing and `ns_submit_run`'s `--mem`
count the shared part once. A default run on this half node now takes 38
ranks at 1 thread (needing 30.8GB, against the 21GB measured), and the
`--mpi-procs 38` OOM warning is gone. A whole 76-core node still stops at
`NS_R2D2_MAX_RANKS` (56). WSClean is unchanged (one number is `PER`). A
default run on the new code: 38 x 1, steady 30.5 and 29.7 evals/s, log(Z)
0.01452 +/- 0.00351.

**The tail is now the larger cost in short runs.** After 95% of a run's
evaluations, the last few run with round 9's adaptive threads, up to 38 each.
Over five uncapped 38 x 1 runs that tail took 33, 44, 57, 113 and 119s, and
single late requests took up to 25s. With `R2D2_MAX_THREADS=16` (two runs,
interleaved with two of the uncapped ones) it took 38 and 45s, with no late
request over 4.2s. This matches round 17's lone-request table, where 16
threads beat 38. So whole-run wall times in this 3k-evaluation shape are
noisy: 141-225s for 38 x 1 against 164-194s for 35 x 2. In a production run
the tail is a fixed minute or two, and the steady state is what scales. Two
capped runs is not enough to ship the cap, so it is the next A/B.

Next, in order:

1. A/B `R2D2_MAX_THREADS=16` (or one socket) against the uncapped tail, with
   at least five runs an arm, since the uncapped tail is bimodal (33-57s or
   113-119s).
2. On a whole node, test ranks above the 56 cap, up to one per CPU. Memory
   no longer limits it (76 ranks is ~54GB by the model above).

## Round 20: the tail's thread cap is 16, and health counts in-flight work again

All on cpu-q-133 (this job's 38 CPUs) at `6829906`: the 38 x 1 default, nlive
150, 15 repeats, 150 dead points, async, seed 4242, with the two arms
interleaved (`.probe/r19.sh 16 38 ...`, read by `.probe/r20.py`). Round 19's
attempt died part-way and recorded nothing. "Tail" is the time from 95% of
the evaluations to the last one. "Slowest late" is the longest R2D2 request
in that tail.

| `R2D2_MAX_THREADS` | wall | steady evals/s (30-90s) | tail | slowest late | log(Z) |
| --- | ---: | ---: | ---: | ---: | --- |
| 16 | 206s | 15.7 | 42s | 4.4s | 0.01443 +/- 0.00351 |
| 38 | 192s | 28.9 | 54s | 7.2s | 0.01452 +/- 0.00345 |
| 16 | 147s | 30.7 | 35s | 2.8s | 0.01456 +/- 0.00349 |
| 38 | 207s | 29.8 | 91s | 9.0s | 0.01472 +/- 0.00329 |
| 16 | 147s | 28.1 | 30s | 2.6s | 0.01465 +/- 0.00349 |
| 38 | 184s | 29.2 | 67s | 5.9s | 0.01464 +/- 0.00354 |
| 16 | 154s | 28.8 | 41s | 3.8s | 0.01437 +/- 0.00347 |
| 38 | 180s | 29.7 | 68s | 7.7s | 0.01444 +/- 0.00349 |
| 16 | 164s | 28.3 | 47s | 4.0s | 0.01466 +/- 0.00342 |
| 38 | 155s | 29.5 | 42s | 6.9s | 0.01475 +/- 0.00333 |

With the cap the tail is 30-47s (median 41s), against 42-91s (median 67s)
uncapped. Round 18's runs agree: 38 and 45s capped, and 33-119s uncapped.
With the cap no late request took more than 4.4s, against 5.9-9.0s without
it. The steady state does not change, since the cap only matters once ranks
fall idle. (The first run's 15.7 is a slow start: the node's load went from
6 to 48 during it.) Median wall time for this shape fell from 184s to 154s.
All ten log(Z) values lie between 0.01437 and 0.01475, against errors of
~0.0034. This matches round 17's table for a lone request, where 16 threads
take 0.22/0.45s and 38 take 0.34/0.58s. Above 16, more threads cross the
socket and the request gets slower.

**The change.** In async runs `R2D2_MAX_THREADS` now defaults to the host's
CPU count capped at 16 (`run-nested-sampling-r2d2.sh`). It is also written to
`run.env`, so a resume keeps it and `./ri bench` groups runs by it. Before
this, two A/B arms of this setting landed in the same bench group.
A run on the new default recorded `R2D2_MAX_THREADS=16` and ran 160s at a
steady 29.3 evals/s, with a 43s tail and log(Z) 0.01451 +/- 0.00340.

**Health.** With tmpfs scratch (round 14), an evaluation in flight has no
directory on Lustre, so `./ri health` read "1 in flight" on a 38-rank run.
On a live run (local ranks, or a RUNNING Slurm job seen from a login node)
the count is now the larger of the leftover directories and the recent
imaging seconds per elapsed second (Little's law). This adds no IO. The same
live run read "29 in flight" with the new code and "1 in flight" with the
old. A dead run still counts its leftover directories as "abandoned".

Next: on a whole node, test ranks above the 56 cap, up to one per CPU
(round 18).

## Round 21: R2D2's simulate writes no MS

**Profile.** At `4b951f3` a default R2D2 run (38 x 1) spent ~120-130ms of
worker time per evaluation in simulate, ~10% of an evaluation. Since round 13
that request writes `r2d2_data.mat` itself, and nothing in an R2D2 run opens
the MS after that. Most evaluations copy a cached skeleton (baked into
`meqtrees.sif` or published earlier in the run) and patch, fill and prune it.
The rest extend a cached observation. All of that work fed only the `.mat`.

**Change.** In R2D2 runs the simulate request now passes `--mat-only`
(`common.py`), unless `NS_KEEP_MEASUREMENT_SETS=1`. `write_mat_only()` in
`simulate_point_source_ms.py` reads UVW from the cached skeleton (read-only,
no copy), or rebuilds it from the cached observation as `extend_template()`
does. It takes the channel frequencies from `patch_spectral_window()`'s
formula and draws the same noise stream over the full (row, channel,
correlation) shape, then adds only correlation 0. When neither cache holds
the observation it falls back to the full MS path, because that makems build
is what fills the caches. The WSClean path does not change.
`self_check_mat_only` runs a cache miss, a skeleton hit and an
observation-prefix hit, and checks each `.mat` against the full path byte for
byte, plus the metadata.

**Stage timing.** `.probe/mat21.py` ran in `meqtrees.sif` on round 15's
48 parameter sets: an untimed warm-up of both arms, then timed repeats with
the arm order alternating. "Base" is `simulate --mat-path`, "new" is the
prototype that became `write_mat_only()`.

| load | base | new | saving | `.mat` |
| --- | ---: | ---: | ---: | --- |
| 1 process, 2 x 48 | 45.9ms | 34.0ms | 11.9ms | `u`, `v`, `y`, `nW` identical, 96/96 |
| 36 at once, 3 x 48 each | 75.2ms (median of 36) | 50.6ms | 23.3-25.4ms in every copy | |

94 of the 96 sets hit a skeleton and 2 hit an observation prefix.

**End to end.** Run on cpu-q-133 with this job's 38 CPUs: the default
38 x 1, nlive 150, 15 repeats, 150 dead points, async, seed 4242. Arms ran in
the order new/base/base/new. Base is `4b951f3` unpacked with `git archive`.

| arm | evals | wall | steady evals/s (30-90s) | simulate mean / median | R2D2 mean | log(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| new | 3165 | 132s | 30.7 | 79.4 / 52.8ms | 1151ms | 0.01459 +/- 0.00345 |
| base | 3129 | 158s | 29.2 | 126.8 / 92.1ms | 1149ms | 0.01471 +/- 0.00352 |
| base | 3214 | 155s | 29.2 | 118.5 / 77.8ms | 1159ms | 0.01443 +/- 0.00361 |
| new | 3225 | 174s | 30.0 | 82.3 / 52.8ms | 1156ms | 0.01461 +/- 0.00352 |

In the full runs simulate drops by ~40ms (-35%) per evaluation, with no
overlap between the arms. R2D2 itself does not change. That is ~3% of worker
time. Steady evals/s is 30.0-30.7 against 29.2, which agrees but is inside
the ~5% spread of runs this short. Wall time follows the tail (round 20) and
says nothing here. log(Z) lies in round 20's band (0.01437-0.01475) for every
run. With the `.mat` byte-identical, a given parameter set scores the same.

What it costs: an R2D2 evaluation has no `sim.ms`, so a failed one keeps its
`.mat` but no MS. Re-running the recorded simulate command without
`--mat-only` rebuilds the MS. `NS_KEEP_MEASUREMENT_SETS=1` restores the MS
for a whole run.

Also tried, not shipped: reusing preallocated buffers for the noise draw
(`rng.standard_normal(out=)`, then multiply-and-cast into a float32 buffer)
is bit-identical, but only 20.0 -> 18.2ms solo and 28.5 -> 26.4ms at 36
copies. That is ~0.7% of a WSClean evaluation. PCG64's ziggurat itself is
the cost, and any faster generator changes the noise.

Next: WSClean's simulate (~90ms at 36 ranks) still needs the MS. What is
left there is noise ~20ms, the skeleton copy and fill, and the
visibility `exp`, and each of those levers changes bits (round 15).

## Round 23: WSClean reuses ducc0's gridder index (patch 0007)

**Profile.** A 38-rank WSClean search (nlive 150, 150 dead points, async,
seed 4242), read with `./ri profile --phases` over 56 logged evaluations:

| phase | ms/eval | passes/eval | ms each |
| --- | ---: | ---: | ---: |
| Gridding | 43.9 | 4.68 | 9.4 |
| Loading data in memory | 33.9 | 4.68 | 7.2 |
| Predicting | 25.7 | 2.68 | 9.6 |
| Reordering | 20.7 | 1.00 | 20.7 |
| Loading metadata | 11.9 | 2.68 | 4.4 |

A median evaluation here grids ~40k rows x 5 channels (200k visibilities),
against 4,914 on the Hetzner host. So ducc0's per-pass bookkeeping is no
longer small. `perf record` on one evaluation's `wsclean`, run 6 times pinned
to one CPU, with symbols from the SIF's own `/opt` and `/lib` copied out and
passed as `--symfs`: of the binary's CPU, the gridding and degridding kernels
take ~27%, ducc0's index build (`countRanges`) ~11%, its `scanData` ~5%, and
the reordered-file row reads plus weighting ~9%.

**Change.** `docker/wsclean/patches/0007` caches the last index built by
`countRanges` and reuses it when the visibility mask, the baselines, the
frequencies and every scalar the index depends on match byte for byte (see
[the patches doc](nested-sampling-wsclean-patches.md)). It applies only to
single-threaded gridding (`-j 1`, which the search uses), where the index is
deterministic.

**Building it without Docker.** This node can build and A/B a WSClean patch
on its own, in ~5 minutes. Nothing here needs root or a Docker host:

```sh
export APPTAINER_CACHEDIR=/dev/shm/c APPTAINER_TMPDIR=/dev/shm/t
apptainer build --fakeroot --sandbox /dev/shm/build docker://debian:bookworm-slim
apptainer exec --fakeroot --writable /dev/shm/build apt-get install -y g++ cmake ...  # the Dockerfile's build list
# copy /opt/casacore out of wsclean.sif into the sandbox, then clone v3.7,
# git apply the patches and cmake -DPORTABLE=OFF -DTARGET_CPU=x86-64-v3 (84s with 30 jobs)
apptainer build --fakeroot wsclean.sif idx.def  # Bootstrap: localimage, %files over /opt/wsclean/bin
```

A run picks the new image up from a `SIF_DIR` that holds it plus symlinks to
the other three SIFs. A derived SIF keeps its parent's `ri.build-inputs`
label, so both arms get the same image id in `benchmarks.jsonl`. Tell them
apart with the row's `note`. The base binary built this way gives output
byte-identical to the shipped `wsclean.sif` and the same solo time
(189.0 vs 189.5ms median).

**Evidence.**

- Byte-identity: the restored, model, residual and PSF FITS data blocks match
  the shipped binary's for 6 Measurement Sets taken from real draws
  (17k-202k rows, 3-4 major cycles). That is 24 of 24.
- Solo, pinned, 20 interleaved rounds on one MS: 175.0ms median against
  189.0 (local base) and 189.5 (shipped), -7.4%.
- In-run, 38 ranks, nlive 150, 300 dead points, async, seed 4242, ABBA order.
  Binary time is fitted per run as `a + b x timestep-channels` and read at
  1k (`.probe/fit16.py`), as in round 16:

| arm | evals | whole-run evals/s | simulate @1k | `wsclean` @1k | log(Z) |
| --- | ---: | ---: | ---: | ---: | --- |
| base | 6637 | 51.6 | 112.0ms | 268.8ms | 0.09574 +/- 0.00162 |
| 0007 | 6692 | 77.6 | 112.2ms | 245.3ms | 0.09549 +/- 0.00159 |
| 0007 | 6712 | 61.6 | 113.3ms | 252.2ms | 0.09527 +/- 0.00160 |
| base | 6788 | 52.5 | 108.9ms | 277.7ms | 0.09549 +/- 0.00164 |
| base, synchronous, 150 dead | 3415 | 23.9 | 111.0ms | 273.1ms | 0.09424 +/- 0.00279 |
| 0007, synchronous, 150 dead | 3415 | 24.9 | 110.0ms | 246.3ms | 0.09424 +/- 0.00279 |

The `wsclean` binary is 8-10% faster in every pair, and the arms do not
overlap. Simulate does not change. At ~300ms of worker time per evaluation
that is ~7-8% of WSClean throughput. Whole-run evals/s follows the tail
(round 14) and is not the measure. The synchronous pair is identical: same
log(Z), and every dead-point, prior and equal-weights chain file is
byte-identical. Only `phys_live.txt` differs, in its line order, not its
contents.

**To ship on CSD3:** the patch only takes effect once `wsclean.sif` is
rebuilt on a Docker host (`./ri build wsclean`, `./ri images export`) and
imported here. `./ri search` uses whatever SIF is in `SIF_DIR`.

Next: `scanData` (~5%) runs on every pass only to rebuild the mask the cache
compares against. Skipping it needs the mask proven unchanged some other way.
The reordered-file row reads ("Loading data in memory" and "Loading
metadata", ~46ms/eval in-run at 7.4 passes) are three small `istream` reads
per row. Reading each part into memory once per process is upstream-shaped
and would keep results byte-identical.

## Round 24: WSClean maps its reordered part files (patch 0008)

**Change.** Round 23's profile left ~46ms per evaluation in "Loading data in
memory" and "Loading metadata". Both read the reordered part files one row at a time,
with three small `istream::read` calls per row (metadata record, data,
weights). `docker/wsclean/patches/0008` maps the three files and copies each
record out with `memcpy`. The cursors advance exactly as the streams did, so
the gridder gets the same bytes ([the patches doc](nested-sampling-wsclean-patches.md)).
It was built and A/B'd on this node with round 23's no-Docker recipe. Both
arms came from one tree, base = 0001-0007, and differ only in 0008.

**Solo.** Six real-draw Measurement Sets from a short search run with
`NS_KEEP_MEASUREMENT_SETS=1`, the binary pinned to one CPU, 20 interleaved
rounds, median of rounds 2-20:

| MS | base | 0008 | change |
| --- | ---: | ---: | ---: |
| 1 (smallest) | 46.3ms | 45.8ms | -1.1% |
| 2 | 116.2ms | 107.8ms | -7.2% |
| 3 | 130.7ms | 121.9ms | -6.7% |
| 4 | 140.1ms | 133.5ms | -4.7% |
| 5 | 173.7ms | 163.5ms | -5.9% |
| 6 (largest) | 308.3ms | 288.1ms | -6.5% |
| sum | 915.3ms | 860.6ms | -6.0% |

The restored, model, residual and PSF data blocks are byte-identical for all
six (24 of 24).

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, ABBA
order. Binary time is fitted per run at 1k timestep-channels (`.probe/fit16.py`):

| arm | evals | simulate @1k | `wsclean` @1k | log(Z) |
| --- | ---: | ---: | ---: | --- |
| base | 6782 | 116.3ms | 253.3ms | 0.09560 +/- 0.00157 |
| 0008 | 6653 | 112.8ms | 227.6ms | 0.09565 +/- 0.00161 |
| 0008 | 6844 | 113.2ms | 225.2ms | 0.09556 +/- 0.00152 |
| base | 6742 | 112.3ms | 251.7ms | 0.09597 +/- 0.00157 |
| base, synchronous, 150 dead | 3415 | 111.4ms | 248.5ms | 0.09424 +/- 0.00279 |
| 0008, synchronous, 150 dead | 3415 | 113.7ms | 232.9ms | 0.09424 +/- 0.00279 |

The binary is ~10% faster in the async pairs (226 vs 252ms) and 6% faster in
the synchronous pair, and the arms do not overlap. Simulate is unchanged.
Under load the saving is larger than solo, as in round 23, because the
saved work is memory-bound. At ~300ms of worker time per evaluation it is
worth ~8% of WSClean throughput, on top of 0007. The synchronous pair matches
exactly: log(Z) is identical and every chain file is byte-identical except
`phys_live.txt`, which has the same lines in a different order.

**To ship on CSD3:** as with 0007, rebuild `wsclean.sif` on a Docker host
and import it. Next for WSClean: `WriteImagingWeights` still seeks, reads
and writes once per row on the first pass, and "Reordering" (~12-20ms per
evaluation) writes the part files that this patch maps.
