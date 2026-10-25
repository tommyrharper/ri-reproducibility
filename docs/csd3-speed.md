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
