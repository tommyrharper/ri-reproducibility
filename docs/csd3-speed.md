# Search speed on CSD3

Speed work for the 9-parameter space on CSD3. Newest round last. From round 30 on, a round's number is the gnhf iteration
whose `gnhf N:` commit shipped it, so gaps are iterations that shipped no round.
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

## Round 25: a fast path in ducc0's `scanData` (patch 0009)

**Profile.** `perf` over five runs of the 0001-0008 binary on the largest
real-draw MS from round 24 (self time, share of all samples): ducc0's
gridding kernel 20.7%, degridding kernel 13.0%, `scanData` 7.3%,
`ReorderedMsReader::CacheMeta` 4.0%, `CalculateWeightsImplementation` 3.2%,
`ApplyWeightsAndCorrections` 3.1%, `countRanges` 2.7% (the index cache's key
comparison). The annotated `scanData` loop is scalar and reloads the mav
pointers and strides on every visibility: the `lmask` store may alias them,
so the compiler cannot hoist them. WSClean's weights and mask are always a
uniform 1, so for that case (one thread, as the search runs) patch 0009 runs
the same test, `norm(vis) != 0`, over local pointers. That is exact, because
`x*1*1 == x`.

**Solo.** The same six real-draw MSs and protocol as round 24
(`/dev/shm/r24/solo.sh`, pinned, 20 interleaved rounds, median of rounds
2-20):

| MS | 0001-0008 | + 0009 | change |
| --- | ---: | ---: | ---: |
| 1 (smallest) | 45.3ms | 46.1ms | +1.7% |
| 2 | 108.5ms | 106.9ms | -1.5% |
| 3 | 122.7ms | 120.1ms | -2.1% |
| 4 | 133.8ms | 128.8ms | -3.8% |
| 5 | 163.1ms | 159.7ms | -2.1% |
| 6 (largest) | 288.7ms | 279.2ms | -3.3% |
| sum | 862.0ms | 840.6ms | -2.5% |

The restored, model, residual and PSF data blocks are byte-identical for all
six (24 of 24).

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, ABBA
order (`.probe/r25ab.sh`). Binary time fitted per run at 1k timestep-channels:

| arm | evals | simulate @1k | `wsclean` @1k | log(Z) |
| --- | ---: | ---: | ---: | --- |
| 0001-0008 | 6683 | 112.3ms | 235.6ms | 0.09566 +/- 0.00157 |
| + 0009 | 6775 | 110.2ms | 223.3ms | 0.09564 +/- 0.00166 |
| + 0009 | 6847 | 110.9ms | 225.3ms | 0.09558 +/- 0.00165 |
| 0001-0008 | 7804 | 113.7ms | 232.4ms | 0.09434 +/- 0.00205 |
| 0001-0008, synchronous, 150 dead | 3415 | 113.9ms | 232.6ms | 0.09424 +/- 0.00279 |
| + 0009, synchronous, 150 dead | 3415 | 112.0ms | 222.2ms | 0.09424 +/- 0.00279 |

The binary is ~4.5% faster in the async pairs (224 vs 234ms), and the arms
do not overlap. The synchronous pair is 4.5% faster too, and its log(Z) and
every chain file are identical (`phys_live.txt` once sorted). As in rounds
23 and 24, the in-run saving is larger than the solo one. At ~300ms of
worker time per evaluation it is worth ~3% of WSClean throughput.

**To ship on CSD3:** rebuild `wsclean.sif` on a Docker host and import it,
as for 0007 and 0008. The full 0001-0009 series applies cleanly to v3.7.
Next for WSClean: `CacheMeta` (4%) and the weighting passes (~6% together).

## Round 26: thermal noise drawn in C, bit for bit (`fastnoise.so`)

**Profile.** After rounds 23-25, the `wsclean` binary is ~220ms of a 38-rank
evaluation at 1k timestep-channels and simulate is ~110ms, so simulate is a
third of worker time. Round 15 found the noise draw to be its largest part.
On the largest real-draw MS from round 24 (105,300 rows x 5 channels x 4
correlations, 4.2M normals), numpy's `rng.normal(...).astype(np.float32)`
twice takes 55ms. The draw alone costs ~13ns a number: each draw is a call
through the bitgen function pointer, and each half allocates two float64
temporaries. Every evaluation has its own seed (`stable_seed()`), so the
draws cannot be cached.

**Change.** `docker/meqtrees/src/fastnoise.c` holds numpy 1.26's PCG64
(XSL-RR 128/64) and its 256-layer ziggurat (`random_standard_normal()`,
with numpy's tables in `ziggurat_double.h`), inlined, and adds
`(float)(0.0 + scale * z)` to a strided float32 destination in place. The
only change from numpy's code is the random sign: it is applied by
XOR-ing the sign bit, which gives the same double without a branch that
mispredicts half the time. That alone took the C loop from 37ms to 19ms.
`add_complex_noise()` in `simulate_point_source_ms.py` is the one noise
path for both the MS fill and R2D2's `--mat-only`. It hands the generator's
128-bit state to the C code, and writes the advanced state back. The
library is loaded once per process from `/opt/ri-lib/fastnoise.so`, which
is outside the bind-mounted `/opt/ri-nested-sampling`, and checked against
numpy on 100k draws (~1,200 of them through the rejection paths). If it is
missing or disagrees, numpy draws the noise. A builder stage of the
meqtrees Dockerfile compiles the library with `-O2 -ffp-contract=off`
against the runtime's base, so it links the same libm `exp`/`log1p`.
`./ri self-check simulate` checks three shapes, the every-4th
(`--mat-only`) stream, and the generator state afterwards. If the image has
no library, it says so and skips.

**Stage.** In `meqtrees.sif`, on the 4.2M-normal MS, median of 10-12
repeats: numpy 55.1-62.3ms, C 19.0ms (2.9x). Over 200 seeds and shapes, the
complex64 output is byte-identical to numpy's. A gcc 8.5 (host) build and a
gcc 13 (ubuntu:24.04, as the Dockerfile builds it) build time the same.

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, ABBA
order (`.probe/r26ab2.sh`). Both arms use the 0001-0009 `wsclean.sif`. The
fast arm's `meqtrees.sif` is the shipped one plus the library, built with
`Bootstrap: localimage`. Stage times are fitted per run at 1k
timestep-channels (`.probe/fit16.py`):

| arm | evals | simulate @1k | `wsclean` @1k | log(Z) |
| --- | ---: | ---: | ---: | --- |
| numpy | 6661 | 109.5ms | 221.3ms | 0.09558 +/- 0.00164 |
| fastnoise | 6874 | 80.2ms | 214.7ms | 0.09566 +/- 0.00158 |
| fastnoise | 6797 | 82.0ms | 222.2ms | 0.09581 +/- 0.00163 |
| numpy | 6821 | 109.9ms | 223.6ms | 0.09534 +/- 0.00167 |
| numpy, synchronous, 150 dead | 3415 | 110.6ms | 221.1ms | 0.09424 +/- 0.00279 |
| fastnoise, synchronous, 150 dead | 3415 | 82.3ms | 219.7ms | 0.09424 +/- 0.00279 |

Simulate is 28ms (26%) faster, and the arms do not overlap. In the
synchronous pair, log(Z) and every chain file are identical
(`phys_live.txt` once sorted). At ~300ms of worker time per evaluation,
that is ~9% of WSClean throughput. R2D2's `--mat-only` simulate draws the
same 4-correlation stream, so it saves about the same ~28ms of a ~1s
evaluation. Steady evals/s spread from 53 to 75 across these runs, because
the other half of the node was loaded (load average 30-37) and idle
fractions reached 23% in either arm, so evals/s cannot show this.

**Sharp edge.** `APPTAINER_BIND` cannot layer a file into a run's
containers. `job-env.sh` re-execs the run under `env -i`, so the variable
never reaches apptainer. The first four async runs and one synchronous
pair of this round were meant to A/B that way, and both arms drew with
numpy (simulate 107.5-111.1ms in all six). They are in `benchmarks.jsonl`
as controls. Use a derived SIF in a `SIF_DIR` of symlinks instead.

**To ship on CSD3:** rebuild `meqtrees.sif` on a Docker host and import it.
Until then, runs draw with numpy, and results are the same either way.

## Round 27: two fewer full-column writes in simulate (and AVX-512, no)

**Profile.** With round 26's C noise, simulate at a 1k-timestep-channel
shape (200 timesteps x 5 channels, 70,200 rows) takes 60ms solo on one
pinned icelake core (`/dev/shm` scratch, warm observation cache, 30 calls).
Timing every casacore call shows that the main table's scalar columns cost
more than DATA: `putcol` of TIME, TIME_CENTROID, INTERVAL and EXPOSURE (all
IncrementalStMan) at 2.0-2.5ms each, and ANTENNA1/2 (StandardStMan) at
1.9ms each, against 3.2ms for the 11MB DATA column. The rest is noise
(~17ms), the phase ramp (~10ms), 6 table open/close pairs (~5ms) and the
copy of the one-timestep template (~2.6ms). Every call also unpacked the
VLA antenna tarball (~1ms), which only makems reads.

**Change.** IncrementalStMan carries the last row's value into rows that
`addrows` adds, which is already how FIELD_ID, SCAN_NUMBER and the other
untouched ISM columns get their values. `extend_template()` now writes
INTERVAL and EXPOSURE on the template's one timestep (351 rows) before
`addrows`, instead of over all rows after it. That stays right when the
template was built with a different integration time, because its value is
overwritten before the rows are added. The antenna table is now unpacked by
`run_makems()`, so evaluations that extend a cached observation leave no
`VLAA_ANT` in their directory.

**Stage.** Every column of every table in the MS hashes the same before and
after the change, over 10 shapes, one of them an extension of a template
built at 3s for a 7s observation. Interleaved, 6 rounds of 30 calls each:
60.0-61.5ms before, 55.3-57.1ms after (-4.7ms, -8%).

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, ABBA.
Both arms use `/dev/shm/r26/sifs-fast`, which is round 26's `meqtrees.sif`
with fastnoise plus the 0001-0009 `wsclean.sif`. The base arm runs from a
`git worktree` at HEAD (`.probe/r27ab.sh`). Fits are from
`.probe/fit16.py`:

| arm | evals | simulate @1k | `wsclean` @1k | log(Z) |
| --- | ---: | ---: | ---: | --- |
| base | 6734 | 83.0ms | 222.7ms | 0.09550 +/- 0.00168 |
| new | 7959 | 77.6ms | 226.3ms | 0.09411 +/- 0.00210 |
| new | 7831 | 75.1ms | 215.6ms | 0.09478 +/- 0.00199 |
| base | 6743 | 81.5ms | 220.7ms | 0.09555 +/- 0.00162 |
| base, synchronous, 150 dead | 3415 | 84.9ms | 224.3ms | 0.09424 +/- 0.00279 |
| new, synchronous, 150 dead | 3415 | 76.8ms | 220.6ms | 0.09424 +/- 0.00279 |

Simulate is 6-8ms (7-10%) faster and the arms do not overlap. That is about
2% of WSClean worker time, which is below what evals/s can resolve here (it
spread from 66 to 90 across these runs). In the synchronous pair, log(Z) and
every chain file are identical (`phys_live.txt` once sorted). The change is
in the bind-mounted `nested_sampling` directory, so it ships with no
rebuild. R2D2's `--mat-only` path never extends an MS, so it is unaffected.

**Negative: an AVX-512 WSClean.** The patched tree (0001-0009) was built
with `-DTARGET_CPU=icelake-server` instead of the shipped `x86-64-v3`.
Gridding and degridding are ~31% of the binary's samples. On round 24's
six real-draw MSs, pinned, 10 interleaved rounds took 885.6ms against
906.6ms (-2.3%), and 6 of 24 FITS data blocks differ. It would also not run
on the ampere nodes' AMD EPYC CPUs, which have no AVX-512. Not pursued.

**Next.** What remains in simulate: TIME/TIME_CENTROID/ANTENNA1/2 writes
(~9ms solo). These could be cut by caching the extended main table's
scalar storage-manager files per (declination, integration, NTimes), but
that only pays if those shapes repeat. Also the second main-table open, and
the phase ramp's complex `exp`. For WSClean, the gridding kernels are ~31%,
and the Wgridder constructor, `CacheMeta`, `CalculateWeightsImplementation`
and `ApplyWeightsAndCorrections` are ~3-4% each.

## Round 28: a fifth of in-run WSClean time is the kernel (malloc tunables)

**Profile.** The previous rounds' fits only give each evaluation's wall time.
To split it, `/dev/shm/r28/zy.sh` drives the shipped `wsclean-zygote`
(0001-0009) the way a rank does. It sends N fork requests for one real-draw
MS on one pinned CPU, then reads the zygote's `cutime`, `cstime` and
`cminflt` from `/proc`. On round 24's MS 4 (a typical size), means over 20
requests after 2 warm-ups:

| | wall | user | sys | minor faults |
| --- | ---: | ---: | ---: | ---: |
| 1 zygote | 101.6ms | 86.5ms | 15.0ms | 9.4-10.7k |
| 37 zygotes at once, one per CPU | 136.4ms | 105.5ms | 29.5ms | 10.2k |

`perf stat` puts most of the user-time growth at 38-way down to the clock
(2.85 -> 2.43GHz, IPC 2.20 -> 2.02). Kernel time doubles, though, to ~21%
of the evaluation. Nearly all of it is first-touch page faults, ~40MB of
fresh pages per evaluation. `perf record -e page-faults` places them in
`PredictRegularMeasurementSet` and `GridRegularMeasurementSet`, whose
per-pass `uvw`/visibility buffers are large enough for glibc to `mmap` and
`munmap` them on every one of the ~7 passes. Further faults come from
`ReorderedMsProvider::WriteModel`, which writes into a model file
recreated as a sparse tmpfs file each major cycle, from the degridder's
output, and from `memcpy` while `metadata_buffer` grows.

**Change.** `run-nested-sampling.sh` starts each zygote with
`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1073741824:glibc.malloc.trim_threshold=4294967295:glibc.malloc.top_pad=67108864`.
Large blocks then come from the heap, and the heap is never trimmed, so
later passes reuse pages an earlier pass faulted in. The setting is per
process, and each forked child gets the zygote's malloc. No rebuild is
needed. (In glibc 2.36, `/proc/<pid>/environ` shows only the first tunable
afterwards, because the string is parsed in place. All three still apply:
`top_pad` alone moves the fault count.)

**Driver A/B** (MS 4, 37 concurrent zygotes, 20 requests each, ABBA):

| arm | wall | user | sys | minor faults |
| --- | ---: | ---: | ---: | ---: |
| base | 136.4ms | 105.5ms | 29.5ms | 10.2k |
| tunables | 132.9ms | 108.2ms | 23.1ms | 5.9k |
| tunables | 132.7ms | 108.4ms | 22.9ms | 5.9k |
| base | 137.2ms | 106.5ms | 29.3ms | 9.9k |

That is -3.9ms (-2.9%) with no overlap. Solo the gain is -3ms. Over the
six real-draw MSs, all 24 FITS data blocks are byte-identical.

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, with
`/dev/shm/r26/sifs-fast` in both arms. The base arm is a `git worktree` at
HEAD (`.probe/r28ab.sh`), run ABBA and then BAAB:

| arm | `wsclean` @1k, ABBA | `wsclean` @1k, BAAB | synchronous, 150 dead |
| --- | --- | --- | ---: |
| base | 227.5, 221.0ms | 215.7, 226.1ms | 221.2ms |
| tunables | 218.9, 221.4ms | 222.5, 217.7ms | 215.5ms |

The mean is -2.5ms async and -5.7ms in the synchronous pair. The arms
overlap, because per-run fits here spread by ~10ms, so in-run this is
consistent with the driver but not resolved by itself. In the synchronous
pair, log(Z) is 0.09424 +/- 0.00279 in both arms and every chain file is
identical (`phys_live.txt` once sorted).

**Next.** ~5.9k faults (~23ms of kernel time at 38-way) remain per
evaluation. A WSClean patch could reuse one `uvw`/visibility buffer across
passes and `reserve` `metadata_buffer`, and could map the model file with
`MAP_POPULATE`, or keep it and zero it instead of recreating it each major
cycle. All of these are bookkeeping changes that cannot alter the output.

## Round 30: map each reordered file once per evaluation (patch 0010)

**Profile.** After round 28, `perf record -e page-faults -d` on the zygote
driver (MS 4) attributed the remaining ~6k faults per evaluation to
addresses. The largest groups were the reordered temporary files:
`ReorderedMsProvider` and `ReorderedMsReader` are built afresh for each of
the ~7 gridding and degridding passes, and each one `mmap`s the model,
meta, part and weight files again and `munmap`s them on destruction, so
every pass faults the same pages in again. The model file is the worst
case, because writes into a shared mapping get no fault-around.

**Change.** Patch 0010 (`system/mappedfile.h`) keeps one `MAP_SHARED`
mapping per (device, inode, size) in a process-wide cache and never unmaps
before exit. Because the mapping is shared, every reader sees the same bytes
as before. Each evaluation is a fresh fork of a zygote that maps none of
these files, so the cache lives for exactly one evaluation.
[nested-sampling-wsclean-patches.md](nested-sampling-wsclean-patches.md)
covers the inode argument.

Rejected on the way, each measured with the driver: `MAP_POPULATE` on the
old per-pass mappings (-1k faults, but sys time unchanged), and
`malloc_trim(0)` in the zygote before each fork (no change, so the faults
are not copy-on-write of the parent's freed heap).

**Driver** (zygote driver from round 28, MS 4, 20 requests, ABBA, with
round 28's malloc tunables in both arms, on cpu-q-133):

| arm | solo wall | solo sys | solo faults | 37-way wall | 37-way sys | 37-way faults |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0001-0009 | 98.5ms | 11.0ms | 6.5k | 130.1ms | 23.3ms | 6.0k |
| + 0010 | 94.5ms | 8.0ms | 3.1k | 128.9ms | 20.5ms | 5.0k |
| + 0010 | 95.5ms | 8.5ms | 3.9k | 128.5ms | 19.9ms | 4.7k |
| 0001-0009 | 96.6ms | 12.0ms | 5.6k | 131.6ms | 23.9ms | 6.1k |

All 24 FITS data blocks over the six real-draw MSs are byte-identical.

**In-run** (cpu-q-226, 38 ranks, nlive 150, 300 dead points, async, seed
4242, `.probe/r30ab.sh`). Both arms are derived SIFs built from one
sandbox tree, 0001-0009 (`/dev/shm/r30/sifs-s9`) and the same plus 0010
(`sifs-mc`), with the shipped `meqtrees.sif` in both. Per-run `wsclean` fit
at 1k timestep-channels (`.probe/fit16.py`):

| arm | ABBA | BAAB |
| --- | --- | --- |
| 0001-0009 | 212.6, 214.6ms | 215.7, 211.4ms |
| + 0010 | 209.1, 209.9ms | 210.8, 206.2ms |

The means are 213.6ms and 209.0ms: -4.6ms (-2.2%), with no overlap across
the eight runs, matching the driver. In a synchronous pair (150 dead) the
fits were 212.0 and 211.7ms, so it is not resolved there. log(Z) is 0.09424
+/- 0.00279 in both arms of that pair, and every chain file is identical
(`phys_live.txt` and `clusters/` once sorted). End-to-end evals/s spread
from 55 to 97 across these runs because of load on the other half of the
node, so only the stage fits resolve an effect this size.

It takes effect once `wsclean.sif` is rebuilt on a Docker host and
imported.

**Rebuilding the sandbox.** /dev/shm does not survive a move to another
node. Round 23's recipe rebuilt the sandbox in ~6 minutes, with two
catches. With the host's git, `git apply` of the whole series from the
top-level repo applied 0001-0003 and then failed on 0004's submodule paths,
so apply with `patch -p1 -N` in order, from a clean tree. And `apt-get install` can drop out while
`update-ca-certificates` runs under `--fakeroot`. Finish it with
`dpkg --configure -a`.

## Round 31: the fill shares extend_template's table open

**Profile.** With a host-built `fastnoise.so` bound over `/opt/ri-lib`, the
48 real parameter sets of `.probe/params.jsonl` (`scripts/probe_simulate_stage.py`,
warm observation cache, one pinned core) take 36.9ms each. Timing every
casacore call (`.probe/colprof31.py`, run as `python3 .probe/colprof31.py scripts/probe_simulate_stage.py ...`) gives, per call: 8 table closes
2.4ms, TIME and TIME_CENTROID 2.0ms each, DATA 1.9ms, ANTENNA1/2 1.5ms each,
two main-table opens 1.0ms, two SPECTRAL_WINDOW opens 0.65ms, `addrows`
0.9ms, and a read back of UVW (0.5ms) and FLAG (0.26ms, only for its shape).
On a cache hit, `extend_template()` writes the main table and closes it, and
then `fill_point_source_visibilities()` opens it again, and opens
SPECTRAL_WINDOW to read back CHAN_FREQ, which `patch_spectral_window()` has
just written.

**Change.** `make_and_fill_ms()` asks `make_ms(fill=True)` to write the
visibilities inside `extend_template()`'s open, from the UVW it has just
computed and the CHAN_FREQ formula (`channel_frequencies()`, also used by
`--mat-only`). Both are float64 columns that read back unchanged, so the
values are the same. FLAG's zeros take DATA's shape. The skeleton and fresh
`makems` paths still fill as before.

**Stage.** Over 96 simulated MSs (each set at 20 minutes, then at its own
length, which extends the cached observation), every file is byte-identical
to HEAD except `table.lock`, where the open counters differ. `./ri
self-check simulate` passes. Interleaved, 6 rounds per arm: 36.8-37.1ms
before, 33.2-35.1ms after (-3.5ms, -9.5%).

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242, both
arms on `/dev/shm/r30/sifs-mc` (the shipped `meqtrees.sif`, so numpy noise,
and the 0001-0010 `wsclean.sif`). The base arm is a `git worktree` at HEAD
(`.probe/r31ab.sh`). Both sets ran ABBA. Per-run simulate fit at 1k
timestep-channels (`.probe/fit16.py`):

| arm | set 1 | set 2 | synchronous, 150 dead |
| --- | --- | --- | ---: |
| base | 102.6, 102.6ms | 102.2, 102.2ms | 104.3ms |
| fused | 93.9, 96.2ms | 97.3, 96.4ms | 97.0ms |

-6.4ms (-6.3%) on simulate, with no overlap, which is about 2% of a
WSClean evaluation. It is larger than solo, as round 27's was, because the
opens and closes cost more with 38 ranks at once. In the synchronous pair,
log(Z) is 0.09424 +/- 0.00279 in both arms and every chain file is
identical. The module is bind-mounted, so this ships with no rebuild. R2D2's
`--mat-only` path never opens an MS, so it is unaffected.

**Next.** What is left of simulate solo is mostly casacore: the TIME,
TIME_CENTROID and ANTENNA1/2 writes (~7ms), the DATA write, and the closes.
The phase ramp's complex `exp` and noise (with `fastnoise.so`) are ~7ms each.

## Round 32: the phase ramp in C (`phase_ramp_c64`)

**Profile.** With noise in C (round 26), the predict
`flux * np.exp((2j * pi / c) * np.outer(delay, freqs))` is the largest
numeric step left in simulate: 9.9ms solo for 21,060 rows x 16 channels.
numpy builds it through complex128 temporaries, and its complex `exp` is
7.6ms of that, against 3.7ms each for float64 `cos` and `sin`. Splitting it
into `np.cos` and `np.sin` is bit-identical but saves only 1.2ms.

**Change.** `phase_ramp_c64()` in `docker/meqtrees/src/fastnoise.c`
computes `sincos(k * (delay * freq))`, scales by the flux and writes
correlations 0 and n-1 of the zeroed complex64 array in one pass. numpy's
complex `exp` is libm's `cexp`, which for a zero real part is `sincos` of
the imaginary part times `exp(0) = 1`, and `k` is the imaginary part of
numpy's own constant. The result is the same in float64, not just after
rounding: 0 of 10.7M complex128 words differ, and 0 of 13.7M complex64
values over random shapes, fluxes and L to X band frequencies.
`point_source_visibilities()` keeps the delay `uvw @ (l, m, n-1)` in numpy.
The library is loaded and checked against numpy once per process, like the
noise filler (`_fast()`). A library without the symbol, or one that
disagrees, leaves the work to numpy. `./ri self-check simulate` adds three
shapes. It is solo 9.9ms -> 5.5ms on the shape above. R2D2's `--mat-only`
uses the same predict.

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242. Both
arms ran from this tree. They differed only in the meqtrees SIF:
`Bootstrap: localimage` over the shipped `meqtrees.sif`, adding
`/opt/ri-lib/fastnoise.so` built from HEAD's source (base) or this one
(new), with the 0001-0010 `wsclean.sif`. Both shared one
`NS_OBSERVATION_CACHE_DIR` in `/dev/shm`.

That cache started empty, and round 31's fit (`.probe/fit16.py`) is
dominated by makems misses: 279, 166, 129 and 140ms over the first four
runs as the cache warmed. `.probe/fit32.py` refits simulate after dropping
evaluations more than 50ms off the line (155-1041 per run, all misses).
Simulate at 1k timestep-channels, trimmed:

| arm | set 1 (ABBA) | set 2 (BAAB) | synchronous, 150 dead |
| --- | --- | --- | ---: |
| base | 63.9, 61.0ms | 61.7, 63.2ms | 61.5ms |
| ramp in C | 52.6, 51.4ms | 52.2, 51.8ms | 51.2ms |

That is -10.5ms (-17%) on simulate, with no overlap, and about 3.5% of
WSClean worker time (WSClean @1k: 206-213ms in both arms). It is twice
the solo saving, as rounds 27 and 31 were. In the synchronous pair, log(Z) is
0.09424 +/- 0.00279 in both arms. Every chain file is identical except
`phys_live.txt`, which holds the same 186 rows in a different order. It
takes effect once `meqtrees.sif` is rebuilt on a Docker host and imported.
Until then the shipped image has no `fastnoise.so`, so numpy does the
work.

**Learned.** A cold observation cache costs far more than this change in a
short run. Between 4% and 15% of evaluations hit makems in the first runs
on a fresh cache, at hundreds of ms each. Compare arms on a shared,
pre-warmed cache, and fit with outliers trimmed.

## Round 33: 32KB StandardStMan buckets in the one-timestep template

**Profile.** Per casacore call on the extend path, solo with `fastnoise.so`
(`.probe/colprof31.py` over `scripts/probe_simulate_stage.py`, 48 sets,
median simulate 23ms), the main-table writes were TIME and TIME_CENTROID
2.1ms each, DATA 1.9ms, ANTENNA1 and ANTENNA2 1.5ms each, `addrows` 0.9ms
and UVW 0.6ms. The int32 antenna columns were slower than the float64 UVW
with three times the bytes. makems puts ANTENNA1/2 in a StandardStMan with
a 256-byte bucket, 32 rows of the pair, behind a 2-bucket cache. Each
extended evaluation grows that table to tens of thousands of rows, so
`addrows` and both putcols walk about 1000 buckets.

**Change.** `one_timestep_template()` publishes the template through
`_copy_with_large_ssm_buckets()`, a deep `table.copy` with the
StandardStMan's `BUCKETSIZE` set to 32768. The template is built once per
channel count per run, so the copy costs a few ms once. Every column value
and keyword stays the same, and only the SSM file's layout changes (the
makems `vis.*` symlinks are not carried over, and nothing reads them). The
skeleton and fresh-makems paths are unchanged. Same-shaped extension, solo
(`/dev/shm/r33/relay.py`, 98 timesteps x 351 baselines):

| template | addrows | ANTENNA1 | ANTENNA2 | sum of writes |
| --- | ---: | ---: | ---: | ---: |
| makems (256B) | 0.60ms | 0.99ms | 0.96ms | 7.9ms |
| 4KB | 0.15ms | 0.14ms | 0.10ms | 5.7ms |
| 32KB | 0.11ms | 0.11ms | 0.07ms | 5.6ms |

Also tried and dropped: a 1MB ISM bucket made TIME slower (1.3 -> 1.7ms),
and UVW tiles of 3x1024 instead of 3x128 saved only 0.15ms while changing
the layout WSClean reads. Over 24 MSs from both trees (12 sets, cached and
extended), all 1608 columns and every keyword are equal. The observation-prefix
self-check now allows exactly this difference: the SSM bucket must be 32768
on an extended MS and the rest of the storage layout must match makems.
Probe simulate stage, interleaved, median of 48 sets: 23.6, 23.1 and 23.5ms
before, 20.2, 19.9 and 21.2ms after (-3.4ms, -14%).

**In-run.** 38 ranks, nlive 150, 300 dead points, async, seed 4242. Both
arms used `/dev/shm/r30/sifs-mc` (the shipped `meqtrees.sif`, so numpy noise
and ramp, plus the 0001-0010 `wsclean.sif`), with one shared
`NS_OBSERVATION_CACHE_DIR`. The base arm is a `git worktree` at HEAD
(`.probe/r33ab.sh`). Order was ABBA then BAAB. Simulate at 1k timestep-channels, makems misses
trimmed (`.probe/fit32.py`):

| arm | set 1 (ABBA) | set 2 (BAAB) | synchronous, 150 dead |
| --- | --- | --- | ---: |
| base | 93.3, 94.9ms | 94.2, 94.0ms | log(Z) 0.09424 +/- 0.00279 |
| 32KB SSM | 89.4, 87.4ms | 88.1, 88.9ms | log(Z) 0.09424 +/- 0.00279 |

That is -5.7ms (-6.1%) on simulate with no overlap. As in rounds 27, 31 and
31, it is larger than the solo saving. WSClean @1k was 210-212ms (base) and
204-214ms (new). In the synchronous pair every chain file is identical
except `phys_live.txt`, which holds the same rows in a different order. The
module is bind-mounted, so this ships with no rebuild.

**Learned.** The untrimmed means in `benchmarks.jsonl` are useless here.
This cache started cold, so whichever arm ran first paid the makems misses:
the synchronous pair's means were 134.5ms (base, first) against 98.2ms. On
this RDS, `fit32.py` read ~10 metrics.json a second, about 50 minutes for 8
runs. Fit in the background, or read the files in parallel.

## Round 35: meta records copied at their fixed size (patch 0011)

**Profile.** `perf record` of the 0001-0010 binary, solo and pinned, on the
largest real-draw MS from round 24 (`-size 32 32`, five runs) put
`ReorderedMsReader::CacheMeta` at 3.7% of the binary's samples. 37% of those
were on one instruction: the 32-byte vector load that reads back the meta
record `ReadMapped()` had just copied in with a run-time length. GCC emits
that copy as narrow stores, and a wide load over narrow stores cannot be
store-forwarded, so every row stalled.

**Change.** Patch 0011 adds `ReadMappedRecord<Record>()`, which copies a
whole record at `sizeof(Record)` (two vector moves) and falls back to
`ReadMapped()` only for a short record at the end of the file. Solo on that
MS, inversion plus prediction fell 25ms (-2.5%) in 11 of 11 interleaved
pairs.

**Driver.** A new zygote driver (`/dev/shm/r35/zy.sh`, `conc.sh`) runs one
`wsclean-zygote` per CPU on 37 of this job's 38 CPUs, with production's
malloc tunables. Each sends 21 rounds of six real-draw MSs from round 24
(0.05-0.40s binary times in their own run, the 10th-90th percentile) and
averages the reply wall times after the first round. That is 4.7k requests
per arm, about 25s. Both binaries were built from one sandbox tree, 0001-0010
and the same plus 0011, and run inside the shipped `wsclean.sif`. Mean
request wall time, 37 zygotes at once:

| set | order | 0001-0010 | + 0011 |
| --- | --- | ---: | ---: |
| 1 | ABBA BAAB ABBA BAAB | 165.6ms | 164.0ms |
| 2 | BAAB ABBA BAAB ABBA | 165.9ms | 164.2ms |

That is -1.6ms (-1.0%) in both sets, with the new binary ahead in 7 of the 8
four-arm blocks. The one exception was the first block of each set, where
both arms were slow. All 24 FITS data blocks over the six MSs are
byte-identical. (A first pass at 5 rounds per zygote, about 7s an arm,
drifted by 10ms over its 8 arms and resolved nothing.)

**In-run** (iteration 34, before its job ended: 38 ranks, nlive 150, 300
dead, async, seed 4242, layered SIFs differing only in the two binaries):
`wsclean` @1k was 205.0 and 207.5ms (base, the second run cut at 5.8k
evaluations) against 212.5 and 208.0ms. Per-run fits spread by ~10ms here,
so a 1.6ms effect is not visible in four runs. In the synchronous pair
(150 dead) log(Z) is 0.09424 +/- 0.00279 in both arms. Every chain file is
identical (`phys_live.txt` once sorted) except `.resume`, where rows
differ in a loglikelihood by ~1e-11. The one such evaluation traced
(eval 13 base, eval 17 new, same point) built its MS with makems in the
base run (2.8s simulate, cold observation cache) and from the template in
the new run, so the difference most likely comes from simulate's two
paths, not the patch, which gives byte-identical images on a fixed MS.

It takes effect once `wsclean.sif` is rebuilt on a Docker host and
imported.

**Learned.** The shipped build is reproducible: the sandbox rebuilt here
on a new node produced a binary with the same md5 as iteration 34's. The
37-way driver resolves 1% where in-run fits cannot, but needs ~25s an arm.
Shorter arms drift with load on the other half of the node.

## Round 36: unseen observations built once, at the horizon

**Profile.** Rounds 26-35 fitted simulate with makems misses trimmed, so
the fits never showed how big the misses are. A worker instrumented to log
every request (removed afterwards; layered `meqtrees.sif` with
`fastnoise.so`, 38 ranks, nlive 150, 300 dead, async) found the mean
simulate request is 5-10x the median, and the misses are why:

| run | requests | median | mean | > 100ms | share of simulate time |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold observation cache | 6,687 | 37ms | 333ms | 2,313 | 94% |
| cache warmed by that run | 6,725 | 31ms | 159ms | 1,045 | 83% |

The observation cache (round 2) is keyed on declination x integration
time, 91 x 10 = 910 keys in the default space. A miss ran makems for the
requested length only, so every new longest draw of a key missed again:
the record maxima of a stream of draws, about ln(k) misses for k draws of
a key, and still a third of evaluations in a 7k-evaluation run. A warm
cache did not fix it, because the second run draws longer observations
the first never did.

**Change.** The rank passes `--observation-horizon-minutes`, the
parameter space's `observation_minutes` max (`observation_horizon_minutes()`
in `common.py`, 0 when the dimension is pinned). On a miss below the
horizon the simulator runs makems once at the horizon with one channel
(`cache_observation_to_horizon()`; makems costs ~3.2ms a timestep whatever
the channel count: 0.45s at 10s integration, 3.9s at 1s), saves the
observation, and serves the evaluation by extension like any hit. Each key
now misses once per persistent cache, ever. The extended MS is the one
makems writes (the observation-prefix self-check, which now also covers a
horizon miss, its horizon-length hit and the fallback past the horizon).
The rank's reply timeout is sized on the horizon length. Both files are
bind-mounted, so this needs no rebuild.

**In-run.** Same settings, both arms on `/dev/shm/r36/sifs` (shipped SIFs
plus `fastnoise.so` layered into `meqtrees.sif`), each arm with its own
`NS_OBSERVATION_CACHE_DIR`, starting empty. Base is a `git worktree` at
HEAD. Order base, new, new, base, so each arm's second run had the cache
its first left:

| arm | cache | evals/s | simulate mean | WSClean binary | idle |
| --- | --- | ---: | ---: | ---: | ---: |
| base | cold | 39.9 | 304ms | 209ms | 2.3% |
| horizon | cold | 48.4 | 219ms | 222ms | 1.4% |
| horizon | warm (849 keys) | 74.7 | 59ms | 203ms | 1.0% |
| base | warm (830 keys) | 39.8 | 157ms | 225ms | 11.5% |

Cold: +21% evals/s, simulate -28%. Warm: +88% evals/s, simulate -62%;
simulate falls from ~40% of worker time to ~22%. The warm base run's 11.5%
idle is its own tail, not the change: without it, base's warm rate from
its busy time is 45.0 evals/s against 75.5 for the new arm (+68%). A production
run reuses one persistent cache across runs, so the warm row is the steady
state, and a long run on a cold cache converges to it after ~910 misses
instead of continuing to miss logarithmically.

In a synchronous pair (150 dead, cold caches) log(Z) is 0.09424 +/- 0.00279
in both arms, and every chain file is identical (`phys_live.txt` and
`clusters/` once sorted) except `.resume`. There two rows differ in their
last column by ~1e-11, the same signature round 35 traced to one evaluation
built by makems in one arm and from the template in the other, which this
change makes more common.

**Learned.** A mean that trims outliers hides the outliers: the trimmed
fits of rounds 32-35 were stable to 1.5ms while 83-94% of simulate time
sat in the trimmed tail. The shipped `meqtrees.sif` still predates
`fastnoise.so` (rounds 26 and 32): solo with numpy, noise is 23ms of a
53ms simulate against 7.6ms of 34ms with the library, so rebuilding the
image is worth ~20ms an evaluation on top of this round. A host-built
`fastnoise.so` (RHEL gcc 8.5, `-O2 -ffp-contract=off`) passes both
agreement checks inside the image and layers in with an 8-second
`apptainer build --fakeroot`.

## Round 37 (negative): streaming retention on the record-writer thread

**Profile.** Warm 38-rank WSClean, nlive 150, 300 dead: busy markers show
only ~25-33 of 37 workers evaluating mid-run. Sampling
`/proc/<pid>/syscall` on the rank main threads (37 ranks x 100 samples at
4Hz) gave 65.5% pselect6 (waiting on an evaluation), 16% running, 12.8%
mkdir and 5.4% newfstatat on Lustre. The Lustre calls are streaming
retention: each rank publishes every new top/bottom-20 objective to
`evaluations/` (~0.3s on the main thread) and evicts it later. 1,531 of
6,661 evaluations were kept at some point; the count grows as ~40 ln(k) a
rank, so this is a short-benchmark artefact and negligible in production
(wrong for WSClean at many ranks: 13% of evaluations at 36 ranks; round 47).

**Change tried.** Queue the publish/evict on the rank's FIFO record-writer
thread. The mkdir samples went, replaced by 13% futex; simulate was +24ms in
both new-arm runs (112 vs 87.5ms) and throughput did not move (ABBA: new
60.0/56.4, base 60.2/58.8 evals/s). The writer shares the rank's pinned
CPU and the Lustre client work is CPU, not just waiting, so it competes with
the simulate worker and the zygote. Reverted.

**Learnings.** Never strace a live rank: the ptrace attach interrupted an
`os.listdir` on Lustre with EINTR and the run aborted (`run_with_retries`
resumed it); sample `/proc/<pid>/syscall` or use perf. PolyChord's root
rank is 83.5% `mca_btl_vader` in perf - the documented MPI spin, not a
master bottleneck.

## Round 38: ranks stop claiming failed-artefact slots once they are full

**Profile.** Every failed evaluation calls `claim_failed_artefacts_slot()`
on the rank's main thread. Once the run's `NS_KEEP_FAILED_ARTEFACTS` (20)
slots are taken, that is one `mkdir(exist_ok)` plus 20 `mkdir`s that fail
with `EEXIST`, all ranks contending on one Lustre directory. A failure
storm (E5/E6 in [csd3-experiments](csd3-experiments.md) had ~6000
failures) pays it on every failure, and finding failures is this repo's
purpose. Timed directly on `/rds` with the slots full (cpu-q-166, load
~19):

| claimers | before: median / mean per call | after: median / mean |
| --- | ---: | ---: |
| 1 | 1.0 / 1.0ms | - |
| 36 at once | 36ms / 36ms (31.9-37.0 across processes) | 0.0 / 0.5-3.5ms |

The "after" mean is one ~30ms discovery per rank, amortised over 10 calls.
36ms is ~15% of a warm ~250ms WSClean evaluation, on the critical path of
that rank's next evaluation.

**Change.** Slots are never freed, so a rank that finds them full records
that (`_FAILED_ARTEFACT_SLOTS_FULL`, per slots directory) and answers
`False` from then on with no filesystem call. The first 20 failures still
keep their artefacts, claimed atomically as before; a fresh directory gives
20 `True` then `False` (checked on tmpfs). Scoring is untouched, so log(Z)
and the chains cannot change. Not run end to end: there is no failure
injection hook, and a search that fails often enough to show 15% needs a
failing region of the parameter space. No image rebuild is needed.

## Round 39 (diagnostic): where a short benchmark's idle workers go

**Question.** Round 37's cold run r37p-1 lost 66s of 235s to three gaps
where no rank was evaluating, and the cause was not found. This round reran
it to catch one: two cold 38-rank WSClean searches (nlive 150, 15 repeats,
300 dead, seed 4242, an empty `/dev/shm` observation cache per run, shipped
SIFs, cpu-q-166 with load ~21). A 2Hz watcher counted the scratch
`.busy` markers and dumped every rank's state after 2s at zero.

**Result.** Neither run had a gap (54.1 and 45.9 evals/s, rows `r39c-1/2`).
The watcher fired only after PolyChord had finished. Round 37's gaps are
still unexplained and did not recur in 2 of 2 runs, so they are rare at
most. The busy-marker counts (37 workers) show where the idle worker time
goes in a 146s run:

| phase | mean busy | idle worker-seconds (of ~5400) |
| --- | ---: | ---: |
| first 20s (prior phase, then the barrier into slice sampling) | 25.1 / 24.8 | 227 / 232 (4%) |
| middle | 30.4 / 28.6 | 682 / 861 (13-16%) |
| last 20s | 6.5 / 3.2 | 534 / 541 (10%) |

- The tail is one worker finishing its last slice-sampling chain alone.
  The last 8 evaluations of `r39c-1` (ids 270-277) ran back to back on one
  rank, ~0.9s each, while the other 36 sat idle. The barrier at ~12-15s
  works the same way: busy drops 37 -> 0 over ~4s before sampling starts.
  Both are paid once per run (~770 worker-seconds, ~21s of full-width
  wall), so a 300-dead benchmark understates steady throughput by ~14%. A
  production run of hours does not. `idle_fraction` in `benchmarks.jsonl`
  does not show this (0.005 here), because it counts wall time with *any*
  worker busy.
- The middle deficit is round 37's streaming-retention Lustre work. It too
  shrinks as a run grows.
- Cold-cache misses at 1s integration (1200 timesteps at the 20-minute
  horizon) cost 5.6-5.8s of simulate under load, against round 36's 3.9s
  solo. One 6s-integration miss took 10.7s, which is unexplained (200
  timesteps should be ~0.7s). The misses are spread through the run, not
  bunched at the end.

**Consequence for measurement.** Compare short benchmarks only with each
other (same `NS_MAX_NDEAD`). To read steady throughput, take evaluations
per second over the middle of the run, not `evals_per_s`. No code change.

## Round 40 (diagnostic): PolyChord's root is not a large-run bottleneck

**Question.** Does PolyChord's own work on rank 0 (dispatching points,
clustering, writing `.resume`, `_dead-birth.txt` and posteriors every
`update_files` = nlive dead points) grow into a bottleneck at production
sizes? None of the earlier rounds measured it apart from the imagers.

**Method.** A toy search in the shipped `polychord.sif`: the same settings as
`polychord_wsclean.py` (9 dimensions, 15 repeats, async, seed 4242,
`write_resume`), 37 ranks on cpu-q-166 (load ~30), and a likelihood that only
sleeps for a fixed time. Each worker logs the gap between the end of one
likelihood call and the start of the next, which is time spent waiting on
the root. 36 workers sleeping *t* seconds set the ceiling at 36/*t* evals/s.

| nlive | dead | sleep | evals | evals/s (ceiling) | mean wait | waits > 0.1s |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1500 | 250ms | 41,541 | 141.5 (144) | 0.14ms | none over 0.26s |
| 1000 | 1500 | 20ms | 41,472 | 1,744 (1,800) | 0.11ms | none over 0.23s |
| 1000 | 1500 | 5ms | 41,591 | 6,615 (7,200) | 0.20ms | none over 0.25s |
| 4000 | 20000 | 5ms | 731,612 | 6,492 (7,200) | 0.33ms | 144, 1.8-2.7s each |
| same, chains on /rds | | | 732,082 | 6,448 | 0.35ms | 144, 1.8-2.7s each |
| same, posteriors/equals/cluster_posteriors off | | | 732,822 | 6,552 | 0.28ms | 108, 1.4-2.2s each |

**Result.** The root keeps up with ~6,500 evals/s, ~50x what a 38-rank
WSClean search asks of it (~120) and ~200x for R2D2. Its only visible cost is
a whole-run pause at each `update_files` write: ~2s at nlive 4000 and 20k
dead, growing slowly with the dead count (1.8s on the first write, 2.7s on
the last). Chains on Lustre instead of `/dev/shm` make no difference, and
the posterior files are only ~20% of the pause, so it is mostly the
`.resume` dump (70MB here) and clustering. Real evaluations are ~50x slower
than the toy's, so the same nlive dead points take ~25 minutes of WSClean
searching between writes, and a 2-3s pause per write is ~0.1-0.2% of the
run. Turning the posterior outputs off would save a fifth of that, so there
is nothing here worth changing. `update_files` and the posterior settings
stay at PolyChord's defaults.

**Consequence.** Worker idle time in a real search (round 39's middle deficit,
13-16%) is not PolyChord's root. It is on the rank side: streaming retention
(round 37) plus the evaluation itself. Large runs will scale on the
evaluation cost. There is no benchmarks.jsonl row, because a toy likelihood
is not a search.

## Round 41: round 28's malloc threshold was below WSClean's buffers

**Profile.** The 0001-0011 zygote, rebuilt in a fakeroot sandbox, was driven
solo through six real-draw MSs (3-19MB, kept by a 6-rank search) with
`perf record -e page-faults -d`. Each child faulted 2.6-5.7k times but on
only 1.2-2.5k distinct pages, so about half its faults were pages it had
already touched. The repeats were all in anonymous mappings, not `[heap]`.
`strace` showed why: `GridRegularMeasurementSet` and
`PredictRegularMeasurementSet` size their `uvw`/visibility buffers from
`CalculateMaxRowsInMemory`, i.e. from the machine's RAM, not from the rows
in the MS. On a 250GB CSD3 node that is 47-110GB of virtual memory per
buffer. Only the rows used are touched, but such a block is far above round
28's 1GB `mmap_threshold`, so every pass still `mmap`ed it fresh and
`munmap`ed it afterwards. Round 28 only fixed the smaller blocks.

**Change.** `mmap_threshold` and `trim_threshold` are now 2^62 in
`run-nested-sampling.sh` (`top_pad` is unchanged). glibc 2.36 in
`wsclean.sif` accepts this: a 40GB `malloc` comes from the heap. Each pass
then reuses the heap pages the last pass touched. Resident memory does not
change, since only touched pages count. No rebuild is needed.

**Driver** (`/dev/shm/r41/zy.py`: 37 zygotes at once, one per CPU, 21 rounds
of the six MSs, 4,440 requests per arm, the shipped 0001-0011 binary, two
interleaved sets):

| set | order | 1GB thresholds | 2^62 | page faults per arm |
| --- | --- | ---: | ---: | --- |
| 1 | ABBA BAAB | 130.0-130.7ms | 126.4-127.0ms (one arm 171.5, load spike) | 17.8M -> 7.2M |
| 2 | BAAB ABBA | 131.2-133.2ms | 126.7-129.1ms | 17.9M -> 7.2M |

That is -3.5 to -4.5ms per request (-3%), with the new setting ahead in all
8 four-arm blocks. Faults per request fall from ~4.0k to ~1.6k (from 110k
to 57k for a whole solo run, -3ms). All 24 FITS data blocks are byte-identical.
Heap reuse hands WSClean's uninitialised `UVector`s old data instead of
zero pages, and the output does not change.

**In-run.** A synchronous pair (nlive 150, 150 dead, seed 4242, 18 ranks
per arm, both at once, each with its own cold observation cache, the base
arm a `git worktree` at HEAD) gives log(Z) 0.0947290581121557 +/-
0.00262747 in both arms, and every chain file is identical, `.resume`
included. That pair is not a speed comparison. At 38 ranks a 3% WSClean gain is
~2% of worker time, below what an in-run fit resolves (round 35).

**Learned.** A glibc tunable can do nothing and say nothing about it, and
here the reason was the allocation size, not glibc. Check
`strace -e mmap` for the sizes actually requested before choosing a
threshold. The remaining ~1.6k faults per request are first touches in a
fresh fork child, which a fork server cannot avoid: pre-faulting in the
zygote turns them into copy-on-write faults.

## Round 42: a host-built `fastnoise.so` for a `meqtrees.sif` without one

**Profile.** The shipped `meqtrees.sif` (21 Sep) predates rounds 26 and 32,
so it has no `/opt/ri-lib/fastnoise.so`, and every search on the cluster has
drawn thermal noise and the phase ramp with numpy. Round 36 measured the
gap solo (~20ms of simulate) but left the fix to an image rebuild, which
needs a Docker host.

**Change.** `start-sidecars.sh` asks the SIF whether it has the library
(`apptainer exec ... test -e`, 0.3s). If it does not, and the host has
`gcc`, it compiles `docker/meqtrees/src/fastnoise.c` with the Dockerfile's
flags into `~/.cache/ri/fastnoise/<hash of source + gcc --version>/` (since
gnhf 48, `$RI_WORK_DIR/.cache/ri/fastnoise/...`, so never `/home`). It
then binds that directory read-only at `/opt/ri-lib` in the meqtrees pool
only. The simulator's existing on-load agreement checks against numpy
(`_noise_agrees`, `_ramp_agrees`) still decide whether it is used, and the
host's RHEL gcc 8.5 build passes both inside the image. If the build fails,
or `NS_FASTNOISE_DIR` is set empty, the run keeps numpy, as before. A
rebuilt image has its own copy and skips all of this.

**In-run.** One tree, two arms raced at once, 18 ranks each (nlive 150,
15 repeats, 300 dead, seed 4242, shipped SIFs, cpu-q-166). Each arm had
its own `/dev/shm` observation cache, cold for its first run and warm for
the next two. Base is `NS_FASTNOISE_DIR=` (numpy):

| pair | arm | evals/s | simulate mean | WSClean binary | worker ms/eval |
| --- | --- | ---: | ---: | ---: | ---: |
| cold | base | 28.9 | 243.6ms | 206.2ms | 450.9 |
| cold | fastnoise | 29.8 | 218.4ms | 219.9ms | 439.4 |
| warm 1 | base | 40.1 | 86.8ms | 214.8ms | 302.7 |
| warm 1 | fastnoise | 42.5 | 54.9ms | 211.2ms | 267.2 |
| warm 2 | base | 43.7 | 81.3ms | 221.8ms | 304.3 |
| warm 2 | fastnoise | 50.7 | 48.3ms | 221.6ms | 271.1 |

Warm: simulate is 32-33ms shorter (-37 to -41%) and each evaluation costs
11-12% less worker time. evals/s rose 6% and 16%; an async run's evaluation
count varies, so read ms/eval rather than evals/s here. Cold: -25ms of
simulate, which makems misses swamp. The saving is larger than round 36's
solo 20ms because numpy's draws compete for memory bandwidth at 36-way.

In a synchronous pair (150 dead, warm caches) log(Z) is 0.09441 +/-
0.00264 in both arms, and every chain file is identical (`phys_live.txt`
once sorted), `.resume` included. That is expected, because the library
reproduces numpy's stream bit for bit.

**Learned.** Check what a shipped image actually contains before trusting
an earlier round's gain: rounds 26 and 32 were measured on layered SIFs, and
production never had them. Small C helpers that the simulator loads by
path and checks on load can be built on the host and bound in, so they
need no image rebuild.

## Round 43: the shipped `wsclean.sif` had none of rounds 23-35

**Profile.** The binaries in the shipped `wsclean.sif` (imported 21 Sep) are
dated 4 Sep, the day patch 0006 landed. So patches 0007-0011 (rounds 23, 24,
25, 29 and 33) were measured on layered test SIFs and never reached a cluster
run, just as round 42 found for `fastnoise.so`.

**Change.** `scripts/patch-wsclean-sif.sh` (`./ri images patch-wsclean`)
turns round 23's manual recipe into one command, run on a compute node:

1. Build a `debian:bookworm-slim` sandbox with `--fakeroot` and apt-install
   the Dockerfile's build list.
2. Copy `/opt/casacore` out of the existing SIF, so casacore is not rebuilt.
3. Clone the Dockerfile's pinned tag, `git apply` every patch, copy in
   `docker/wsclean/src/*.cpp`, and build with `-DTARGET_CPU` from the
   Dockerfile.
4. Write a `Bootstrap: localimage` SIF whose whole `/opt/wsclean` (binaries
   and `libwsclean.so`) is replaced, keeping the old SIF as `.orig`.

It took 1m54s with 30 jobs. A `--writable` sandbox cannot create a bind's
mount point, so the patches are copied into the sandbox, not bound.

**In-run.** The two arms raced at once, 18 ranks each (nlive 150, 15
repeats, 300 dead, seed 4242, async, cpu-q-166 under a load average of
33-54 from another job). Each arm had its own `/dev/shm` observation cache,
cold for the first pair. Both used the host-built `fastnoise.so` from round
40. Base is the shipped `wsclean.sif`, and the new arm is the same SIF after
`patch-wsclean-sif.sh`:

| pair | arm | evals | evals/s | simulate | WSClean binary | worker ms/eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cold | base | 8179 | 37.3 | 180.1ms | 193.7ms | 374.9 |
| cold | 0001-0011 | 7460 | 37.8 | 199.2ms | 161.2ms | 361.4 |
| warm 1 | base | 7664 | 50.5 | 55.5ms | 200.6ms | 257.2 |
| warm 1 | 0001-0011 | 7419 | 49.8 | 52.3ms | 168.9ms | 222.3 |
| warm 2 | base | 7476 | 55.1 | 43.2ms | 199.8ms | 244.1 |
| warm 2 | 0001-0011 | 8325 | 62.5 | 39.8ms | 146.9ms | 187.8 |
| sync, 150 dead | base | 3094 | 25.8 | 37.2ms | 174.3ms | 212.6 |
| sync, 150 dead | 0001-0011 | 3094 | 29.0 | 39.9ms | 141.6ms | 182.6 |

In every pair the WSClean binary is 32-53ms shorter (-16 to -27%), more than
rounds 23-35's in-run estimates added together, because those were measured
at 37-38-way and here two 18-rank searches share the node. With warm caches,
worker time per evaluation falls by 14-23%. The synchronous pair, which
controls for evaluation count, runs 11% faster end to end (120.0 -> 106.8s).
Whole-run evals/s in the async pairs follows the evaluation count and the
tail (round 14), so read ms/eval instead.

In the synchronous pair log(Z) is 0.09441 +/- 0.00264 in both arms, and every
chain file is identical (`phys_live*.txt` once sorted), including `.resume`
and `clusters/`. That is consistent with rounds 23-35, where each patch was
shown byte-identical on its own.

**To ship:** run `./ri images patch-wsclean` once in a compute-node job. It
replaces `images/wsclean.sif` in place, and a run already going keeps the
inode it opened. The shipped SIF was left as it is, because it is shared
with any run in flight. Rebuilding on a Docker host (`./ri build wsclean`,
export, import) gets the same result.

**Learned.** Rows from runs outside `results/nested-sampling/` are skipped
by `bench.py` ("not recording ... outside results/nested-sampling"). Move
the run there and `scripts/bench.py record <dir>` adds the row afterwards.

## Round 45: TIME and TIME_CENTROID out of the IncrementalStMan

**Profile.** At a warm cache with `fastnoise.so` (48 real sets, one pinned
core), `extend_template` has a median of 17.8ms. Per evaluation, the putcols
of TIME, TIME_CENTROID and DATA cost about 2ms each. TIME and TIME_CENTROID
are slow because makems stores them in an IncrementalStMan, and its putcol
compares each row with the previous one.

**Change.** When `_copy_with_large_ssm_buckets` copies makems' template, it
now also moves both columns into the template's StandardStMan (the one round
32 enlarged to 32KB buckets), using the new `_template_dminfo`. The values
are unchanged. The template is run-local, so there is nothing to invalidate.
On its own, the extension (copytree, addrows and four putcols at 150
timesteps x 351 baselines) has a median of 8.5ms, down from 11.5ms. The
observation-prefix self-check now compares a served MS's layout with a
makems build passed through `_template_dminfo`, and every value as before.

**In-run.** The two arms raced at once with 18 ranks each (nlive 150, 15
repeats, 300 dead, seed 4242, async, shipped `wsclean.sif`, host-built
`fastnoise.so`). Each arm had its own `/dev/shm` observation cache,
prefilled with the 910 keys of the production cache, which is the warm state
production runs see (iteration 44). The base arm is HEAD exported into
`/dev/shm`:

| pair | arm | evals | evals/s | simulate | WSClean binary | worker ms/eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a | base (tmpfs tree) | 8591 | 56.1 | 43.3ms | 220.9ms | 265.4 |
| a | SSM time (Lustre tree) | 7504 | 44.5 | 38.8ms | 221.5ms | 261.5 |
| b | base (tmpfs tree) | 7550 | 53.1 | 48.0ms | 238.1ms | 287.4 |
| b | SSM time (Lustre tree) | 7445 | 44.3 | 41.4ms | 229.7ms | 272.4 |
| c | base (tmpfs tree) | 7594 | 57.4 | 43.4ms | 220.9ms | 265.4 |
| c | SSM time (tmpfs tree) | 7376 | 58.4 | 38.4ms | 211.0ms | 250.5 |

Simulate is 4.5-6.6ms shorter in every pair (-10 to -14%), and the WSClean
binary does not get slower when it reads TIME from the SSM. Worker time per
evaluation falls by 1.5-5.6%.

In pairs a and b, whole-run evals/s favoured base by about 20%. This was the
tree's location, not the change. In those pairs, worker-busy time was ~83%
of wall for the tmpfs tree and ~66% for the Lustre one. Pair c put both
trees in `/dev/shm`, which brought both to 81-85% and gave the new arm +1.7%.

**Identity.** The synchronous pair (150 dead) gives log(Z) 0.09441 +/-
0.00264 in both arms, and every chain file is byte-identical
(`phys_live*.txt` once sorted), including `.resume` and `clusters/`. The
base arm got there through a `run_with_retries` resume. On both attempts,
some ranks hit "no wsclean-zygote pool for rank N" (the 300s
`NS_POOL_CONNECT_SECONDS`) after the prior phase, and the retry started a
fresh pool. A resumed run is not given a benchmark row.

**Learned.**
- The working tree's filesystem costs ~20% of whole-run evals/s. With the
  run directory and tree on Lustre (`/rds`), 18 ranks spend ~17% more of the
  wall outside worker time than with both in `/dev/shm`, and the
  per-evaluation stage timings do not show it. Every production run has its
  tree on `/rds`, which makes this the biggest open lead. The next round
  should find which per-evaluation file operations on Lustre (evaluation
  dirs, `metrics.json`, the `run.log` tee, chain writes) the rank waits on.
- A/B arms must share a filesystem. Export both trees to `/dev/shm`, or
  neither.
- Twice, a wsclean-zygote pool failed to connect within 300s when two
  18-rank arms started together, in the arm that started first. The retry
  recovered, but the cause is unknown.

## Round 47: a rank's extremes wait in tmpfs, not on Lustre

**Profile.** Round 45 found that a tree on Lustre costs ~20% of whole-run
evals/s. Round 37 had already caught one Lustre cost on the rank's main
thread: streaming retention publishes each evaluation that enters the rank's
top or bottom 20 objectives so far from tmpfs scratch to `evaluations/`
(~0.3s: mkdir and file creates on Lustre), and strips it again once it is
evicted. Round 37 called this a short-benchmark artefact, but the count is
~40 ln(k) for a rank with k evaluations, so it grows with the rank count.
Replaying the 71,393 objectives of the longest R2D2 run
(`r2d2-vlaa-20260922T174616Z`) through per-rank heaps, with evaluations dealt
to ranks at random, 9.1% of them enter an extreme at 24 ranks, 13.0% at 36
and 23.5% at 72. For WSClean, at ~0.25s of worker time per evaluation, that
is about 15% of the rank's time at 36 ranks. For R2D2, at ~20s, it is
negligible. A bound pair on HEAD code with both arms in one Lustre tree
(`NS_IMAGE_KEEP_ENDS=0`, so no extremes are published, against the default
20) gave 53.8 vs 42.5 and 51.0 vs 40.9 evals/s (+26%, +25%).

**Change.** `write_evaluation_record` now holds an extreme instead of
publishing it. The scratch directory is renamed to `.held/<eval>` in the
same tmpfs, and the record names its final Lustre paths as before. An
eviction of a held evaluation is an `rmtree` in tmpfs. Survivors move to
Lustre at most every `NS_RETAIN_FLUSH_SECONDS` (600) and when the rank
drains its records at the end. The 1 in 100 detail evaluations, failures,
and `NS_KEEP_ALL_IMAGES`/`NS_KEEP_MEASUREMENT_SETS` runs publish at once as
before. A flush moves at most 2 x `IMAGE_KEEP_ENDS` evaluations per rank. A
job that dies loses the images of at most one interval's held extremes.
Their records survive, and `prune_run_artefacts()` drops any path that no
longer exists. The streaming-retention self-check now covers holding,
eviction from `.held/`, the timed flush and the final drain.

**In-run.** Base is HEAD exported to `/rds`. New is this tree, also on
`/rds`. The arms were raced with 18 ranks each (nlive 150, 15 repeats, 300
dead, seed 4242, async), each with its own `/dev/shm` observation cache
prefilled from the production one, shipped SIFs and host-built
`fastnoise.so`. The starting arm alternated between pairs. Pairs c and d
were run in gnhf 46, which hit a usage limit before it could record them.
Pairs f and g were rerun here with the same code.

| pair | arm | evals | wall | evals/s | worker ms/eval |
| --- | --- | ---: | ---: | ---: | ---: |
| c | base | 7491 | 160.9s | 46.6 | 234.9 |
| c | held | 7944 | 131.2s | 60.5 | 248.1 |
| d | held | 8124 | 136.4s | 59.6 | 227.8 |
| d | base | 7473 | 137.6s | 54.3 | 237.9 |
| f | held | 7238 | 113.9s | 63.5 | 236.5 |
| f | base | 7857 | 131.8s | 59.6 | 223.7 |
| g | base | 7568 | 145.4s | 52.1 | 243.4 |
| g | held | 8070 | 133.0s | 60.7 | 248.9 |

The held arm was faster in all four pairs: +30%, +10%, +6.5% and +16.5%
evals/s. Worker time per evaluation did not fall, and was slightly higher
when the held arm had more evaluations in flight. The gain is all in rank
main-thread time between evaluations, which is where the Lustre publishes
were. The bound pair's code with holding (pair e, keep 56.4 vs nokeep 52.2
evals/s) now matches the no-publish bound. The node's load average reached
65 during pair g from another job sharing it, which is why only raced pairs
are compared.

**Identity.** In the synchronous pair (150 dead), log(Z) is 0.09441 +/-
0.00264 in both arms, and all 13 chain files are byte-identical
(`phys_live*.txt` once sorted). Neither arm restarted. After the final
prune, both async pair-c arms keep the same number of FITS images (94).

**Learned.**
- Round 37's "negligible in production" was wrong for WSClean. A rank's
  extremes so far grow as ~40 ln(k) per rank, so the fraction of evaluations
  published grows with the rank count, to 13% at 36 ranks and 23% at 72.
- A cost on the rank's main thread shows only in whole-run evals/s, not in
  any stage timing or in worker ms/eval. Keep comparing raced arms on
  evals/s, with worker ms/eval as a check that the workers did not change.
- gnhf 46's edits were recovered from its transcript
  (`.gnhf/runs/*/iteration-46.jsonl`, the `old=`/`new=` python edits) after
  a usage limit ended it unrecorded. Its result directories and the base
  tree on `/rds` were still there to re-score with `bench.row_for`.

## Round 50 (diagnostic): a tree on Lustre no longer costs anything

**Question.** Round 45 found that a tree on `/rds` ran ~20% slower than one
in `/dev/shm` (44 vs 56 evals/s at 18 ranks). Round 47 moved the biggest
known Lustre cost, the extremes' publish-then-strip, off the rank's main
thread. How much of the gap is left?

**Method.** The same code (HEAD `1aabf34`) in two trees: this worktree on
`/rds`, and `git archive HEAD` unpacked in `/dev/shm`. The arms were raced
with 18 ranks each (nlive 150, 15 repeats, 300 dead, seed 4242, async).
Each had its own `/dev/shm` observation cache, prefilled with the
production cache's 910 keys. The starting arm alternated between the two
pairs, with a 20s offset. "Middle" is the evaluations that ended between 20s
after the first start and 25s before the last end, read from each
evaluation's `timing` (`started_epoch`/`ended_epoch`). "Busy" is the mean
number of evaluations running then. The 18th rank is PolyChord's root, so
17 is the ceiling.

| pair | tree | evals | wall | evals/s | worker ms/eval | middle evals/s | busy (of 17) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | `/dev/shm` | 8151 | 131.7s | 61.9 | 252.4 | 64.8 | 16.86 |
| a | `/rds` | 7992 | 129.3s | 61.8 | 249.1 | 65.7 | 16.56 |
| b | `/rds` | 7604 | 120.9s | 62.9 | 246.7 | 64.8 | 16.64 |
| b | `/dev/shm` | 7561 | 128.2s | 59.0 | 256.0 | 64.8 | 16.86 |

**Result.** The gap is gone. The `/rds` arm matched or beat the tmpfs arm
on whole-run evals/s in both pairs (-0.1% and +6.6%), and the middle rates
are equal to within 1.4%. The `/rds` ranks are idle ~1.5% more of the time
(16.6 vs 16.86 busy). That is the per-evaluation `metrics.json` write and
the 1 in 100 detail publishes, which are too small to show end to end. In
the middle of the run, the workers are now 97-99% busy, against round 39's
84-87% (13-16% idle) before round 47. What is left of the whole-run
loss is the start (prior phase and the barrier into slice sampling) and the
tail (one rank finishing its last chain alone). In a 300-dead benchmark,
mean busy over the whole span is 15.1-15.6. Both are paid once per run, so
they shrink to nothing in production. log(Z) agrees within each pair (async
runs, so the chains differ): 0.09460/0.09466 in pair a, 0.09575/0.09590 in
pair b.

**Consequence.** A/B arms no longer have to share a filesystem, and the
cluster's default (a tree on `/rds`) pays no Lustre penalty on the rank's
main thread. A WSClean evaluation is now ~85% the WSClean binary (208-216ms)
and ~15% simulate (38-39ms), with the workers saturated, so further gains
have to come from inside the binary or simulate. No code change.

## Round 51: a run publishes only the held extremes it keeps

**Question.** Every benchmark so far stopped at 150-300 dead points. What
does a whole production search (the `production` preset: nlive 150, 15
repeats, to convergence) spend its wall on, now that the workers are
saturated mid-run (round 50)?

**Method.** The preset at 36 ranks (seed 1, async, tree on `/rds`, warm
production observation cache), timed from launcher start to launcher exit.
Per-evaluation `timing` gives the evaluation span. A second base run
carried timestamps around each step after `run_polychord` returns.

**Profile.** The search converges at 1427 dead points after ~44k
evaluations, and the whole run takes 6.5 minutes:

| phase | seconds | share |
| --- | ---: | ---: |
| launch to first evaluation | 5.8 | 1.5% |
| evaluations (132 evals/s, 33 of 35 workers busy) | 331.0 | 85% |
| of which the tail where one worker finishes the last chain alone | ~20 | ~5% |
| all ranks drain; each flushes its held extremes to Lustre | 22.7 | 5.8% |
| rank 0 reads 44k records | 7.2 | 1.8% |
| rank 0 prunes artefacts | 16.1 | 4.1% |
| rank 0 writes `summary.json` (176MB) | 3.8 | 1.0% |
| exit | 2.4 | 0.6% |

The post-run phase, 52.7s, is 13.5% of a production run, and `./ri bench`
never sees it: a row's `wall_s` is the evaluation span. Its biggest part is
round 47's held extremes. Each rank ends holding its own lowest and highest
20 so far, up to 36 x 40 evaluation dirs, and copied all of them to Lustre.
Only the run's 40 survive, so prune then unlinked the rest again. Prune also
made 11 Lustre stats per record (an `is_dir` and one `is_file` per log name)
for dirs that hold only `metrics.json`.

**Change.** `finished_run_records()` (`common.py`) replaces the drain at
the end of both `polychord_wsclean.py` and `polychord_r2d2.py`. Every rank
drains its records without flushing what it holds, then waits at a barrier.
Rank 0 reads the records and computes `evaluations_keeping_images()`, the
same decision prune makes, and broadcasts the result. Each rank publishes
only the held evaluations in that set and drops the rest in tmpfs. A second
barrier then lets rank 0 prune. The kept set is unchanged, because prune
would have deleted everything else anyway. Prune now lists each eval dir
once instead of stat-ing each log name. The streaming-retention self-check
adds another rank's held extreme that the run does not keep, and it fails
when the keep filter is removed.

**Result.** One run of the new code against the two base runs, same preset:

| run | evaluations | evaluation span | post-run | launch to exit |
| --- | ---: | ---: | ---: | ---: |
| base (`r51-prod`) | 44476 | 336.8s | ~51s | ~7.5 min |
| base, timed (`r51-prod2`) | 44212 | 331.0s | 52.7s | 389.5s |
| new (`r51-new`) | 44769 | 316.4s | 37.0s | 359.8s |

The post-run phase fell by 15.7s (-30%), or 4% of a production run. All 40
run extremes kept their images, no path in `summary.json` is missing, and
the 57-60 images named per run differ only because async runs draw
different points. log(Z) is 0.0966/0.0953 (base) against 0.0958 (new),
inside the async spread. This is a direct phase timing of one new run
against two base runs, not a raced pair. A 15.7s change on a fixed ~52s
phase is well outside the noise, but the evaluation spans are not
comparable.

**Left.** ~34s of post-run work, all on rank 0 while 35 ranks wait: the
record read, the prune's remaining Lustre walk, and a 176MB `summary.json`
re-read by every reader. These grow with the evaluation count, not the rank
count, so a multi-hour run pays minutes here. The next step is to time the
three again with the new code, then cut whichever is largest (for example,
ranks could hand their records to rank 0 over MPI instead of rank 0
re-reading them from Lustre).

## Round 52: rank 0 gets the records over MPI; prune skips record-only dirs

**Question.** Of round 51's ~34s left after the last evaluation, which step
is largest, and can it go?

**Method.** Both entrypoints now print `post-run seconds: drain, records,
flush, prune, summary` after the summary lands (`post_run_line()` in
`common.py`), so `run.log` shows the phase that `./ri bench` misses. Same
preset as round 51 (36 ranks, nlive 150, 15 repeats, to convergence, seed 1,
async, tree on `/rds`). One base run from a copy of a2b9b03 plus the timing,
then the new code three times. The runs were sequential, because two 36-rank
arms don't fit on 38 CPUs, but each phase is timed directly.

**Change.** Each rank keeps the text of every record it writes. At the end it
parses its own records, all ranks in parallel, and `comm.gather`s them to
rank 0. Rank 0 reads from Lustre only the eval dirs it wasn't handed, which
are the records of an earlier segment of a resumed run
(`gathered_and_earlier_records()`), and the result has the same order as
`load_evaluations_from_dir()`. Prune now skips listing the dir of a scored
record that names no image. `retain_evaluation_detail()` drops those paths
when it strips an evaluation as it is scored, so such a dir holds only
`metrics.json`. In the base run that was 34.5k of 43.2k records, and none of
them held anything else. A barrier before the gather keeps the drain wait out
of the `records` figure.

**Result.**

| run | drain | records | flush | prune | summary | log(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (`r52-base`) | 8.3 | 7.5 | 0.7 | 9.5 | 3.7 | 0.09647 |
| gather only, no barrier (`r52-new`) | 0.0 | 21.2* | 0.9 | 9.7 | 11.1** | 0.09640 |
| new (`r52-new2`) | 46.1 | 1.0 | 1.3 | 5.2 | 3.7 | 0.09634 |
| new (`r52-new3`) | 13.3 | 1.0 | 1.0 | 5.4 | 3.7 | 0.09624 |

\* includes the drain wait. \** a 176MB write to Lustre, which took 3.7s in
the other three runs. Dumping unpickled and JSON-parsed records to
`/dev/shm` took the same 3.2s, so this was noise.

Reading records fell from 7.5s to 1.0s and prune from 9.5s to 5.3s: 10.7s
off every production run, more on longer ones. Both new runs kept images
for all 40 run extremes, and no path in `summary.json` is missing. log(Z)
stayed within the async spread (rounds 47 and 51 saw 0.0953-0.0966). A
mutation that drops the earlier-segment read fails the resume self-check.

**Finding: the record writer runs minutes behind.** The drain step swung
from 8.3s to 46.1s with no change to its code. Per-rank timestamps in
`r52-new3` showed every rank leaving `run_polychord` at the same instant
with 415-1222 record writes still queued, out of ~1240 per rank. When
`metrics.json` mtimes are compared with each record's `ended_epoch`, the lag
grows to a median of ~190s by 90s into the run and then shrinks to ~40s by
the end. So for most of a run, a killed job loses minutes of records, which a
resume has to evaluate again. The likely cause is that
`_pypolychord.so` imports no `PyEval_SaveThread`/`PyGILState` symbols. The
sampler holds the GIL in Fortran and MPI between callbacks, so the
record-writer thread gets the GIL only while an evaluation blocks in
`zygote_run()`. This has not been tested yet. The next step is to write
records from a GIL-free helper (a child process fed over a pipe), or
otherwise inline, and then check the lag and the drain.
