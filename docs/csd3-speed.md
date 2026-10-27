# Search speed on CSD3

Speed work for the 9-parameter space on CSD3. Newest round last. From round 30 on, a round's number is the gnhf iteration
whose `gnhf N:` commit shipped it, so gaps are iterations that shipped no round.
[The speed index](nested-sampling-speed.md) covers the 5-parameter work on
the Hetzner host.

**Where it stands (round 115).** The 22 Sep production R2D2 search below,
rerun on the same 24 sapphire CPUs with the same images and seed, now
converges in **41 minutes instead of 6h20m (7.0x)**, at 26.6 evals/s
instead of 4.12, with log(Z) in agreement. Simulate (MeqTrees) went from
39% of worker time to 3.8%. See [round 115](#round-115-diagnostic-the-22-sep-production-search-rerun-on-todays-tree).

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

## Round 53 (rejected): a helper process for the record writes

**Question.** Round 52 found the record-writer thread running minutes behind
scoring and blamed the GIL. Does writing records from a GIL-free helper
process fix the lag without costing throughput?

**Method.** First a probe run: the production preset (36 ranks, nlive 150,
15 repeats, to convergence, seed 1, async), with every background write
timed in wall and thread-CPU time. A control process did the same mkdir,
write and rename every 0.25s in a sibling directory. Then the change, as one
36-rank production run and as raced 18-rank pairs (800 dead points,
seed 4242, ABBA) against a copy of 63e2074. Driver scripts:
`.probe/r53prod.sh`, `.probe/r53ab.sh`. Analysis: `.probe/r53an.py`,
`r53lag.py`, `r53busy.py`.

**The GIL part is confirmed.** `_pypolychord.cpp` calls back into Python
with `PyObject_CallFunctionObjArgs` and never releases the GIL around the
Fortran.

**But the slow step is Lustre.** In the probe, each background write spent
~600ms in `mkdir` of its eval dir and ~100ms writing and renaming, at 1ms of
CPU. That gave the writer ~1.4 records/s per rank against ~3.7/s scored, and
the queue grew to 174s by 200-300s into the run. The control in a sibling
directory took 0ms for mkdir and 11ms for the write. Measured from an
outside process during a raced pair, `mkdir` *inside* the live
`evaluations/` took 134ms (new arm) and 242ms (base arm), median, against
0ms in a sibling directory. Idle, 16 processes creating records in one
shared directory reach ~1000 records/s, the same as in 16 separate ones, so
the cost appears only while a search runs. Once the run stops scoring, the
same writers drain at ~1000 records/s with 0ms mkdirs. Node-wide on the
run's MDT (`rds-d6-MDT0001`, from `/proc/fs/lustre/mdc/*/md_stats`), two
raced arms made 640 intent_lock, 218 create, 134 revalidate_lock and 104
rename per second. What makes the live directory slow is not yet known. An
strace of a rank showed no Lustre path at all (everything is in tmpfs); its
workers were not traced.

**Change tried.** `write_records_in_background()` started a
`python -I -c` helper per rank that reads length-prefixed (path, text)
pairs from a pipe, writes each one atomically and reports failures as
errno lines. Failed writes still raised on the rank; the self-check and a
mutation test passed.

**Result: rejected.**

| run | evals | eval span (s) | evals/s | ms/eval | drain (s) | log(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base a (18 ranks) | 24210 | 354.1 | 68.38 | 224.8 | 18.4 | 0.09701 |
| new a | 24337 | 415.7 | 58.54 | 237.0 | 0.0 | 0.09710 |
| new b | 25110 | 403.1 | 62.29 | 223.8 | 0.0 | 0.09631 |
| base b | 24052 | 354.1 | 67.93 | 222.5 | 13.0 | 0.09716 |
| probe, thread (36) | 42889 | 300.1 | 142.90 | 213.8 | 17.7 | 0.09604 |
| new (36) | 45200 | 424.6 | 106.46 | 184.1 | 0.1 | 0.09556 |

The drain is gone, and the record lag stays flat at 3-5s for the whole run
instead of growing. That lag is about one full 64KB pipe (16 records of
~4KB) at ~270ms each. So the helper can't keep up either, and the bounded
pipe makes each rank wait on Lustre. At 36 ranks only ~20 of 35 workers were
busy. The thread's unbounded queue is what keeps the ranks scoring, so the
helper costs 11% of evaluations/s raced, and more at 36 ranks. log(Z) is
within the async spread. Reverted.

**What it means.** The lag can't be fixed by moving the writes to another
thread or process. One record per evaluation in a live `evaluations/` costs
~130-600ms of Lustre wait, and the evaluations come faster than that.
Candidates for a later round: find what slows the live directory (trace the
WSClean and MeqTrees workers, or `md_stats` per phase); fewer metadata
operations per record (no per-record eval dir for the 80% that hold only
`metrics.json`, or appending records in batches to a per-rank file); or
sharding `evaluations/`. All but the first change a layout that resume,
health and profile read.

## Round 54: the record writer no longer shares its rank's pinned CPU

**Question.** Round 53 found that `mkdir` inside a live `evaluations/` takes
130-600ms while a run scores, against ~0ms in a sibling directory, and left
the cause open. What makes the live directory slow?

**Method.** Three probe runs (HEAD a417def, 36 ranks, nlive 150, 15
repeats, 800 dead, seed 1, async). While each one scored, an outside process
timed mkdir, write and rename in the live `evaluations/` and in a series of
control directories. strace (`--seccomp-bpf`, every process I own) showed
what touches the directory. Then a raced 18-rank ABBA pair (600 dead,
seed 4242) against a tree copy of a417def. Scripts: `.probe/r54run.sh`,
`.probe/r54ab.sh`.

**What it is not.** It is not another reader. No process has a cwd or fd in
`evaluations/`, and strace shows only the record writers using it. The
stall watchdog's `find -maxdepth 2` doesn't slow concurrent creates either:
a 24k-entry synthetic dir kept 0.12ms mkdirs with `find` looping over it.
Load and concurrency alone don't explain it. Under the same live load, 36
writer processes in a fake run's `evaluations/` kept 0.12ms mkdirs (sibling
of the run, 1175 records/s from 36 threads in one directory). The decisive
control was a fresh subdirectory *inside* the live `evaluations/`, which
took 0.13ms while `evaluations/` itself took 560ms. Throughout, at most 0-4
modify RPCs were in flight from the node (`mdc.*.rpc_stats`). So the
creates were queueing in the client, not at the MDS.

**Cause.** The VFS takes the parent directory's lock for every `mkdir`, so
all 36 ranks' writers serialise on `evaluations/`, and each one holds that
lock across its Lustre round trip. `pin_rank_cpu()` (round 16) pins each
rank to one CPU, and the writer thread inherits the pin. So the CPU the
writer must wake on is the one where its rank's WSClean or MeqTrees worker
is running flat out. Each lock holder waits for that core while 35 others
wait for it, and an otherwise ~0.1ms create costs a scheduler wait times
the queue. When scoring stops (the end-of-run drain, or strace freezing the
workers) the same writers do ~1000 records/s, which is round 53's
observation.

**Change.** `write_records_in_background()` starts the writer with an
initializer that resets that thread's affinity to the CPUs the rank had
before it was pinned (`_unpin_thread`, `_UNPINNED_CPUS`). The rank and its
workers stay pinned. The CPU-pinning self-check now asserts that the writer
thread runs on the full set, and fails with the initializer removed.

**Result.** Raced 18-rank pairs, ABBA (base a, new a 20s later; then the
reverse):

| arm | evals | eval span (s) | evals/s | ms/eval | drain (s) | record lag median / p90 (s) | log(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base a | 17349 | 246.9 | 70.27 | 230.4 | 7.4 | 38.2 / 79.0 | 0.09656 |
| new a | 17672 | 261.6 | 67.54 | 234.3 | 0.0 | -0.5 / -0.1 | 0.09674 |
| new b | 17500 | 269.4 | 64.96 | 235.2 | 0.0 | -0.5 / -0.1 | 0.09705 |
| base b | 18280 | 284.0 | 64.37 | 240.7 | 8.9 | 42.6 / 87.6 | 0.09590 |

Measured from outside during pass a, mkdir in the live `evaluations/` took
180-240ms in the base arm and 0.14-9ms in the new one. The record lag
(metrics.json mtime minus `ended_epoch`) falls from a median of ~40s to
the clock-skew floor. The ~200s maximum in both arms is the held extremes,
which are flushed after scoring by design (round 51). The end-of-run drain
goes from 7-9s to 0s. evals/s is unchanged within this host's ~5% spread
(means 66.3 new, 67.3 base; ms/eval 234-235 against 230-241). log(Z) is
within the async spread. The gain is in what a run leaves behind, not in
its rate: a job killed mid-run now loses seconds of records, not minutes,
and the drain no longer grows with run length.

## Round 55: meta fields read straight from the mapping (patch 0012)

**Profile.** One 36-rank search (nlive 150, 15 repeats, 1500 dead, seed
4242, async) on a `wsclean.sif` rebuilt with patches 0001-0011 by
`./ri images patch-wsclean`, with the host-built `fastnoise.so`. `perf
record` was attached for 30s to five `wsclean-zygote`s and five simulate
servers. perf 4.18 here rejects a `-p` list longer than a few PIDs with a
bare usage message, so attach to a handful, not all. Symbols come from
`--symfs` over a tree untarred from the SIF. Zygote samples by function:

| function | share |
| --- | ---: |
| ducc0 `Wgridder` (all kernels) | ~38% |
| `ReorderedMsReader::CacheMeta` | 5.7% |
| `MsGridderData::CalculateWeightsImplementation` | 3.5% |
| `MsGridderData::ApplyWeightsAndCorrections` | 2.8% |
| `WGriddingMSGridder::GridRegularMeasurementSet` | 2.3% |
| `ImageWeights::Grid` | 2.0% |
| libc (memcpy/memset, malloc) | ~10% |

The simulate servers spend 38% in `fastnoise.so` and 14% in libm
(`sincos` for the phase ramp), 17% in the interpreter and 7% in numpy.

Two thirds of `CacheMeta`'s samples sit on one instruction, a 32-byte
`vmovapd` from the stack. Round 35's patch 0011 copies the 38-byte packed
`MetaRecordBuffer` at its fixed size, but GCC emits that as two 16-byte
stores plus an overlapping 8-byte store. `u`..`time` are then reloaded as
one 32-byte word, which spans two stores and cannot be forwarded, so every
row still stalls. The stall is worse under load: 2.9% of the binary solo,
5.7% at 36-way.

**Change.** Patch 0012 drops the stack record on the regular path and
`memcpy`s each field from the mapped meta file into `metadata_`
(`docs/nested-sampling-wsclean-patches.md`). The short-record fallback and
the BDA path are unchanged.

**Result.** Driver: six real-draw MSs from an `NS_KEEP_MEASUREMENT_SETS=1`
run (binary times at the 10th-90th percentile, 30-210ms in-run), each run
with the search's `wsclean` arguments and production's malloc tunables. A
sample is the summed wall time of one pass over the six MSs, after one warm
pass.

| driver | order | 0001-0011 | 0001-0012 | change |
| --- | --- | ---: | ---: | ---: |
| solo, pinned, 3 reps per MS | 6 ABBA/BAAB blocks | 719.3ms | 709.9ms | -1.3% (-1.0 to -1.8%, 6 of 6 blocks) |
| 37 copies at once, one per CPU, 4 reps | 4 ABBA/BAAB blocks | 986.2/983.8/989.5/985.5ms | 978.3/977.3/983.0/977.2ms | -0.75% (-0.67 to -0.85%, 4 of 4 blocks) |

Every one of the 16 concurrent arms beat the base arm next to it. Solo
`perf` puts `CacheMeta` at 0.9% of the binary's samples, down from 2.9%.
All 24 FITS data blocks over the six MSs are byte-identical, so the chains
cannot change. Each driver time includes process start and the reorder,
which the patch doesn't touch.

In a raced 18-rank ABBA pair (600 dead), whole-run evals/s was 78.8 -> 82.3
and 85.5 -> 79.5, and median binary time 116.1 -> 109.9ms and
102.8 -> 105.3ms. Both follow the draws, not the binary. A <1% effect can't
be seen end to end here (AGENTS.md: ~5% spread). log(Z) was
0.09676/0.09672 (base) and 0.09613/0.09643 (new), within the async spread.

It takes effect when `wsclean.sif` is rebuilt (`./ri images patch-wsclean`,
or a Docker build). No change for a run on the shipped SIF.

**Learned.**
- `images/wsclean.sif` is still the 21 Sep build with patches 0001-0006
  (round 43). This round's 36-rank profile search on a 0001-0011 SIF ran at
  204.6 evals/s (154ms worker time per evaluation, 122.7ms binary). Round
  53's 36-rank probe on the shipped SIF ran at 142.9 (213.8ms). Seeds and
  run lengths differ, so this is not an A/B, but it agrees with round 43's
  -16 to -27% binary time. Running `./ri images patch-wsclean` once is
  still the biggest unclaimed speedup for production.
- A fixed-size `memcpy` into a packed struct does not guarantee
  forwardable stores. Check the generated code (`perf annotate`) rather
  than trusting the source, and read fields from the mapping directly.
- In-run, a WSClean evaluation is now ~80% binary and ~20% simulate
  (30ms), with simulate dominated by `fastnoise.so` and `sincos`.

## Round 56: the noise generator's state stays in registers (`fastnoise.c`)

**Question.** Round 55's simulate-server profile puts `fastnoise.so` at 38%
of simulate, which is ~20% of a WSClean evaluation. Is the C loop as fast
as it can be?

**Diagnosis.** `perf stat` on a pinned driver (400k complex visibilities,
50 reps) gave 4.57ns a draw at 3.56 instructions per cycle. The
disassembly showed why it was slow. With the ziggurat's rejection paths
inlined, gcc 8.5 (the host compiler that builds the `fastnoise.so` runs
use, round 42) ran out of registers and kept the 128-bit PCG state on the
stack. So every draw's multiply chain went through a store and a reload.

Things that did not help:

- Four interleaved PCG lanes filling a 256-state buffer (jump-ahead by
  M^4) to break the multiply's latency chain: +7% at every=1, and +32% when
  the buffer was `__thread`. The loop is limited by instruction count, not
  multiply latency.
- Moving the rejection paths out of line on their own: -4%. The state
  was still spilled.

**Change.** In `docker/meqtrees/src/fastnoise.c`:

- The rare rejection paths (~1.2% of draws, idx 0 tail and wedge) moved
  to a `noinline slow_normal()` that takes and returns the state by value.
- The hot path steps the state as two `uint64_t` words (one 64x64->128
  `mul` plus two `imul`s).
- `add_normal_f32()` is one flat loop over all `n * every` draws, so only
  one copy of the draw is inlined. A pointer-based loop beats an
  index-based one (3.76 vs 4.08ns).

Every draw consumes the same stream words in the same order, so the
output is unchanged bit for bit.

**Result.**

| test | base | new | change |
| --- | ---: | ---: | ---: |
| C driver, pinned, every=1 (MS fill) | 4.56ns/draw | 3.76ns/draw | -17.6% |
| C driver, pinned, every=4 (R2D2 `--mat-only`) | 4.70ns/draw | 3.37ns/draw | -28.4% |
| C driver, 36 copies at once, every=1 | 5.68ns/draw | 4.71ns/draw | -17.1% (36/36 faster) |
| `add_complex_noise()` in `meqtrees.sif`, (351000,4,4), every=1, 30 interleaved pairs | 50.1ms | 41.5ms | -17.2% (30/30) |
| `add_complex_noise()`, (87750,4), every=4, 30 interleaved pairs | 13.0ms | 9.1ms | -29.6% (30/30) |
| raced 16-rank synchronous search, warm pair: simulate stage | 30.5ms | 29.0ms | -4.9% |
| same pair: evals/s | 31.47 | 31.70 | +0.7% (below resolution) |

The driver compares outputs and the final generator state over five
chained calls of odd lengths. The Python pairs compare the data bytes and
the next `rng.random()`. `simulate_point_source_ms.py --self-check` passes
inside `meqtrees.sif` with the new library bound at `/opt/ri-lib`. All
seven chain files are byte-identical across the four raced arms
(log(Z) 0.09667). The first raced pair was cold in both arms (226ms
container overhead per evaluation), so only the second is a speed
comparison.

**Shipping.** The shipped `meqtrees.sif` has no `/opt/ri-lib`, so every
cluster run uses the host-built copy that `start-sidecars.sh` caches by
source hash (round 42). The new source therefore reaches the next search
with no rebuild. A Docker build compiles it with Ubuntu 24.04's gcc, which
was not tried here. The load-time check against numpy falls back to numpy
if that build ever disagrees.

**Learned.**
- Check what gcc 8 does with an `unsigned __int128` loop-carried value:
  under register pressure it lives on the stack. Splitting the value into
  two 64-bit words and keeping only one inlined copy of the hot path fixed
  it. An instructions-per-cycle reading near 3.5 points to instruction
  count, not latency, so lanes don't help.
- `./ri bench run --interleave` takes only the settings the ledger
  groups by, so a library A/B (`NS_FASTNOISE_DIR`) is two raced
  `./ri search` runs. Their output directories must sit under
  `results/nested-sampling/` for `./ri bench record` to take them.

## Round 57: a vectorised sincos for the phase ramp (`fastnoise.c`)

**Question.** After round 56, libm's `sincos` in `phase_ramp_c64()` (round
32) was the largest piece of simulate that was not noise. Can it be faster
and still give libm's float32s bit for bit?

**Diagnosis.** A pinned C driver over (351000 rows, 4 channels) took
~34ms a call: ~21ns a `sincos` under `meqtrees.sif`'s glibc 2.39 (25-27ns
under the host's 2.28), at any phase size from ~10 to ~10^4 rad.

**Change.** In `docker/meqtrees/src/fastnoise.c`, `phase_ramp_c64()` now
works in blocks of 256 phases:

1. A scalar loop writes the phases, with the same `k * (delay * freq)`
   expression as before.
2. `poly_sincos4()` takes four at a time with GCC vector extensions:
   Cody-Waite reduction by pi/2 in three parts, then fdlibm's sin and cos
   kernel polynomials. It is cloned for AVX2 and for baseline SSE2
   (`target_clones`).
3. A scalar loop writes the float32s. It keeps a product only if every
   double within `1e-14 * |flux|` (plus 2^-50 relative, for the product's
   own rounding) rounds to the same float. Otherwise, or for |phase| >=
   2^19, that phase is redone with libm's `sincos`.

Over 2e8 draws (uniform in log magnitude up to 2^19, plus points next to
multiples of pi/2), the polynomial is within 2.2e-16 of `sincos` under
both glibc 2.28 and glibc 2.39. That leaves the 1e-14 margin ~40x headroom,
so the output is libm's whatever the polynomial's last bit. In practice
the fallback runs about 3 times in 1.4M phases. The blocks matter too:
packing four scalar phases straight into a vector stalled on store
forwarding and gave only 18.7ms, against 10.3ms with blocks.

**Result.**

| test | base | new | change |
| --- | ---: | ---: | ---: |
| C driver, pinned, (351000,4), AVX2 clone | 33.9ms | 10.3ms | -70% |
| same, SSE2 clone | 39.4ms | 13.6ms | -65% |
| same, all phases >= 2^19 (all fall back) | 39.2ms | 40.8ms | +4% (not reached in searches) |
| `_phase_ramp()` in `meqtrees.sif`, (351000,4), 30 interleaved triples (numpy 72.9ms) | 43.2ms | 15.9ms | -63% (30/30) |
| raced 16-rank synchronous search, warm pair 1: simulate | 44.6ms | 42.4ms | -4.9% |
| warm pair 2: simulate | 45.2ms | 43.2ms | -4.3% |
| warm pairs: evals/s | 22.30 / 23.95 | 22.74 / 24.10 | +1.9% / +0.6% (below resolution) |

The searches used nlive 150, 2 repeats, 150 dead and seed 4242. Every
arm's output is byte-identical to base in every C and Python test, at
delay spreads of 2, 200, 2000 and 20000m. The dead-point chains are
byte-identical across all six raced arms (log(Z) 0.09426 +/- 0.00255);
`phys_live.txt` matches up to row order. The first raced pair was cold
(simulate 108ms on makems misses in the new arm), so it is not a speed
comparison. `simulate_point_source_ms.py --self-check` passes with the
new library at `/opt/ri-lib`. The load-time check (`_ramp_agrees()`) now
also covers a partial block and a phase on libm's path.

**Shipping.** As in round 56, the change reaches cluster runs through the
host-built copy that `start-sidecars.sh` caches by source hash, so no
rebuild is needed.

**Learned.**
- gcc 8.5 has no `__builtin_convertvector`, and passing a 32-byte vector
  by value draws a `-Wpsabi` note. Pass vectors by pointer and do the
  float conversions per lane.
- Building a vector from four scalar stores and loading it straight back
  costs a store-forwarding stall on every vector, which ate half the
  gain. Stage the inputs in a block buffer first.
- "Bit-identical to libm" does not require a correctly rounded libm
  clone. The output is float32, so an error bound and a check for
  rounding ambiguity, with a fallback, are enough.

## Round 58: reordered rows buffered in memory (patch 0013)

**Question.** After rounds 56-57, simulate is ~18% of a 16-rank WSClean
evaluation and the `wsclean` binary ~80%. Where does a 0001-0012 binary
spend its time, and what can go next?

**Profile.** A 0001-0012 `wsclean.sif` built into `/dev/shm` by
`scripts/patch-wsclean-sif.sh OUT` (the shipped SIF was left alone), then a
16-rank synchronous search (nlive 150, 2 repeats, 150 dead, seed 4242) read
with `./ri profile --phases`:

| phase | ms/eval | share | per eval |
| --- | ---: | ---: | ---: |
| gridding | 24.9 | 27.6% | 4.35 passes |
| reordering | 16.6 | 18.4% | once |
| loading data in memory | 14.4 | 16.0% | 4.35 |
| predicting | 13.2 | 14.6% | 2.35 |
| process start to imaging table | 3.0 | 3.4% | once |
| loading metadata | 2.9 | 3.2% | 2.35 |
| fitting beam | 2.3 | 2.5% | once |

The reorder runs once per evaluation but costs as much as three gridding
passes. On a kept 46k-row MS it took 17.6ms solo, with only 0.01s of sys
time in the whole 0.15s run, so it is user CPU. Frame-pointer and DWARF call
graphs both lose their callers inside casacore, so perf attributes almost
nothing to `ReorderMS`. To get a profile of the phase itself, `perf record
-k monotonic` was run over ten runs and only the samples that fell between
each run's logged `Reordering:` and `Initializing model` lines were kept
(the realtime-monotonic offset was read just before and after the
recording). That gave 3584 samples, 17.9ms a run. About half were the
writer: `ostream::write`, its sentry and `filebuf::xsputn` (~14%), libc
copies and malloc/free (~16%), `ExtractWeights`/`ExtractData` (~12%), and
`FileWriter::WriteDataRow`/`WriteMetaRow` (~8%). The other half was
casacore and `MSRowProvider` reading the MS. Per row, `FileWriter` makes
three `ostream::write` calls of a few tens of bytes and allocates two
`std::vector`s.

**Change.** Patch 0013 has `FileWriter` append every row to an in-memory
buffer per file, extract the data and weights straight into it, and write a
buffer once it reaches 1MB. The header updates write what is left before
they seek, so each file gets the same bytes in the same order
(`docs/nested-sampling-wsclean-patches.md`).

**Result.** Driver: five kept real-draw MSs (binary 45-310ms in-run), each
run with the search's `wsclean` arguments and production's malloc
tunables, pinned. A sample is one pass over the five MSs.

| test | 0001-0012 | 0001-0013 | change |
| --- | ---: | ---: | ---: |
| solo, CPU 10, 3 reps per MS, 6 ABBA/BAAB blocks: reorder | 262ms | 211ms | -19.5% (6 of 6) |
| same: whole pass | 2364ms | 2287ms | -3.3% (6 of 6) |
| 36 copies at once, arms alternating across CPUs, 2 reps, block 1: pass | 2593ms | 2540ms | -2.0% |
| block 2 (CPUs swapped): pass | 2498ms | 2422ms | -3.0% |
| both blocks: reorder | 215ms | 177ms | -18%; new faster in 30 of 36 adjacent pairs |
| raced 16-rank synchronous search, pair 1 (cold): binary mean / median | 150.7 / 102.8ms | 146.5 / 100.4ms | -2.8% / -2.3% |
| pair 2 (warm): binary mean / median | 151.0 / 104.0ms | 147.1 / 99.7ms | -2.6% / -4.1% |
| pair 2: evals/s | 29.67 | 30.58 | +3.1% (below resolution) |

The whole-pass saving (~5ms a run solo) is larger than the reorder's alone
(3.4ms), probably the tail of the per-row allocations freed earlier. With
`-save-reordered`, the four reordered files of two MSs are byte-identical
(8 of 8), and so are all 20 FITS data blocks over the five MSs. In both
raced pairs every chain file is byte-identical, `phys_live.txt` matches up
to row order, and log(Z) is 0.09426 +/- 0.00255, the same as round 57.

**Shipping.** Takes effect when `wsclean.sif` is rebuilt (`./ri images
patch-wsclean` on a compute node, or a Docker build). The production SIF is
still the 0001-0006 build. A run on the shipped SIF does not change.

**Learned.**
- `scripts/patch-wsclean-sif.sh OUT` builds a test SIF anywhere in ~2
  minutes without touching `images/`. A `SIF_DIR` of symlinks (the other
  SIFs plus the test `wsclean.sif`) then lets two `./ri search` runs race
  two builds.
- perf on CSD3 can't unwind through casacore, so `--children` is useless
  for a phase that lives there. Windowing samples by the binary's own
  `-log-time` lines (`perf record -k monotonic`, plus the realtime offset)
  gives a per-phase profile instead.
- strace -f over `apptainer exec` hangs on this host (it stopped inside
  apptainer's startup). For syscall questions, compare `/usr/bin/time`'s
  user and sys times instead.
- What is left in the reorder is casacore reading the MS (~8ms on a 46k-row
  MS), and after it "loading data in memory" (14ms/eval over 4.35 passes),
  which copies the reordered files into the gridder's buffers on every pass.

## Round 59: no time-offset map lookups outside facets (patch 0014)

**Question.** After 0013, "loading data in memory" is the third-largest
WSClean phase (14.4ms/eval in round 58's table, ~4ms a pass on a 51k-row MS
against ~12ms of gridding). Where does it go?

**Profile.** A 0001-0013 SIF built into `/dev/shm` with
`scripts/patch-wsclean-sif.sh OUT`, and `perf record -F 20000` over ten
pinned runs of one kept 51k-row MS. Symbols came from a copy of the binary
pulled out of the SIF, matched to perf's file offsets with `nm -n`. The load
path was ~11% of the binary's samples, spread over a dozen functions:
`CalculateWeightsImplementation<1>` 2.8%, `ApplyWeightsAndCorrections<4>`
2.2%, `GridRegularMeasurementSet` 1.8%, `CacheMeta` 1.3%, `ReadMeta` 1.0%,
`NextInputRow` 0.9%, `ReadWeights` 0.7%, `std::map<unsigned long, unsigned
long>` 0.55%, `ReadData` 0.5%. The map is `VisibilityModifier::time_offsets_`,
read and written back for every row by `LoadAndApplyCorrections()`, though
only facets ever change it.

**Change.** Patch 0014 returns from `LoadAndApplyCorrections()` before the
lookups unless the gridder is a facet with a non-`kSingle` PSF mode
([the patches doc](nested-sampling-wsclean-patches.md)).

**Result.** Driver: five real-draw MSs from an `NS_KEEP_MEASUREMENT_SETS=1`
run (binary 57-388ms in-run, 10th-90th percentile), each run with the
search's `wsclean` arguments and production's malloc tunables, pinned to one
CPU, 3 reps per MS. A sample is one 15-run pass; phases are summed from
`-log-time` lines.

| test | 0001-0013 | 0001-0014 | change |
| --- | ---: | ---: | ---: |
| solo, 6 ABBA/BAAB blocks: "loading data in memory" | 697.6ms | 660.7ms | -5.3% (6 of 6) |
| same: whole pass | 4111.8ms | 4097.2ms | -0.35% (-0.13 to -0.54%, 6 of 6) |
| rejected variant (also caching per-channel wavelengths in `thread_local` vectors): load | 697.2ms | 766.6ms | +9.9% (6 of 6) |

All 20 FITS data blocks over the five MSs are byte-identical. No raced
search was run: 0.35% is far below an in-run A/B's ~5% resolution.

**Shipping.** Takes effect when `wsclean.sif` is rebuilt (`./ri images
patch-wsclean`, or a Docker build), together with 0007-0013.

**Learned.**
- The rejected variant saved one division per channel per row
  (`ChannelWavelength()` is `c / freq`), but the `thread_local std::vector`
  wrapper calls and the `std::equal` key check per row cost more than the
  division. The remaining `u / wavelength` divisions (two per visibility)
  are most of `CalculateWeightsImplementation`, and replacing them with a
  multiply would change rounding at the uv-limit boundary.
- The load path has no single hot spot left after 0008-0014. Each of its
  ~10 pieces is 0.5-3% of the binary, so further gains there mean
  restructuring the per-row reader-to-gridder pipeline, not another
  one-line skip.
- Pin CPUs from this job's mask (`taskset -cp $$`: even CPUs only here). A
  `taskset -c` outside it fails at once, and the driver's log is then empty.

## Round 60: the zygote warms FFTW for the image size actually used

**Question.** A whole-binary `perf` profile of a 0001-0014 WSClean (driven
through `wsclean-zygote`, symbols via `perf report --symfs` over a copy of the
SIF's `/opt` and `/usr/lib`) still had FFTW at 4.4% of samples, and most of it
was the planner: `fftwf_md5putc`, `fftwf_ct_applicable`, `fftwf_choose_radix`.
A forked child should find every plan's per-size tables already built by the
parent. Why was it planning?

**Diagnosis.** An `LD_PRELOAD` shim that times every `fftwf_plan_dft_*` call,
keyed by pid, over five real-draw MSs in one zygote: the parent warmed 108,
128, 142 and 156 - the sizes for `-size 128`, hard-coded in
`WarmFftwPlanner()` - but `NS_IMAGE_DIM` has since gone to 32, and every child
planned 32 (image and inversion), 36 (radler's `even(ceil(1.1 x 32))`
convolution) and 40 (the padded image) from cold: 12 plans, **~5.0 ms an
evaluation** of planning, against an ~87 ms mean binary on these MSs.

**Change.** `zygote.cpp` now warms at the first request, beside the casacore
warm-up, for every even size from 0.8x to 1.25x the request's `-size` plus 8
(24-48 at 32; 102-168 at 128, which still covers the old list). Planning in a
child fell to 0.28 ms; the parent pays ~36 ms once per rank.

**Result.** Solo driver as round 59 (five MSs from an
`NS_KEEP_MEASUREMENT_SETS=1` run, binary 10th-90th percentile, production
malloc tunables, pinned to CPU 10), but through `wsclean-zygote` rather than
one process per run, because the fork server is what holds the warm state.
A sample is one pass over the five MSs, mean of three warm passes.

| test | old zygote | new zygote | change |
| --- | ---: | ---: | ---: |
| solo, 6 ABBA/BAAB blocks: binary per pass | 425.8 ms | 400.5 ms | **-5.9%** (6 of 6) |
| same: first deconvolution ("Performed N iterations") | 13.69 ms | 0.99 ms | -92.8% |
| same: rendering the restored image | 5.23 ms | 0.66 ms | -87.3% |
| raced 16-rank sync search, pair 1 (cold): binary mean / median | 147.1 / 101.4 ms | 141.7 / 96.7 ms | -3.7% / -4.6% |
| pair 2 (warm): binary mean / median | 147.0 / 101.2 ms | 141.7 / 95.9 ms | -3.6% / -5.2% |
| pair 2: evals/s | 31.40 | 31.69 | +0.9% (below resolution) |

Searches: `--mpi-procs 16 --nlive 150 --num-repeats 2 --max-ndead 150 --seed
4242 --synchronous`, both arms at once from `SIF_DIR`s of symlinks. All 20
FITS data blocks of the solo driver are byte-identical, and all four searches'
chain files are byte-identical (`phys_live.txt` up to row order), log(Z)
0.09426 +/- 0.00255 as in rounds 57-58. The new arm's
`image_container_overhead` is ~1 ms higher: the 36 ms warm-up, 16 ranks over
613 evaluations, which a production-length run amortises away.

**Shipping.** The zygote is compiled into `wsclean.sif`, so this takes effect
on a rebuild (`./ri images patch-wsclean`, or a Docker build), with 0007-0014.

**Learned.**
- `perf report --symfs DIR` resolves symbols of a SIF binary once `DIR` holds
  copies of its `/opt` and `/usr/lib` (`apptainer exec SIF cp -a ...`), which is
  quicker than round 59's `nm` bisection. `squashfuse_ll`'s zlib shows up as
  ~10% of a short profile: SIF decompression at zygote start, not per eval.
- Anything a fork server warms by a hard-coded size goes stale silently when a
  default moves; derive it from the first request instead.
- The deconvolution's and restore's FFT plans were most of the planning cost
  (first "Performed N iterations" segment 2.7 ms an MS, restore 1 ms), so a
  log-time segment that is large only on the first major cycle is a sign of
  per-process setup, worth checking against a warm parent.

## Round 61: where WSClean binary time goes now, and a cached simulator parser

**Question.** Round 60's searches had a mean binary time of ~142 ms against a
median of ~96 ms. What drives the mean, and what is left of the fixed cost
per evaluation once every patch is in?

**Cost model.** A 16-rank synchronous search on the shipped SIFs (nlive 150,
450 dead, seed 4242, 2003 evaluations; still the 0001-0006 `wsclean.sif`)
fitted by least squares over all evaluations, with `Mvis` = time samples x
channels x 351 baselines / 1e6 and `major` = `clean_major_iterations`:

| term | coefficient | share of binary time |
| --- | ---: | ---: |
| constant | 31.7 ms | 19% |
| per visibility, once (reorder, PSF, first image) | 0.235 s/Mvis | 36% |
| per visibility per major cycle (load, grid, predict) | 0.101 s/Mvis | 44% |
| per minor iteration | 15 us | 1.5% |

R2 is 0.97. Major cycles are 3 or 4 in 75% of evaluations (counts 1: 465,
2: 43, 3: 754, 4: 737), which is `-mgain 0.8` against the drawn dynamic
range, so the cycle count is science, not overhead. The top 10% of
evaluations by binary time take 34% of it. Simulate fits a ~9 ms floor:
the smallest MSs (under 60 timestep-channels) took 8.3 / 9.2 / 10.5 ms in-run
(10th / 50th / 90th percentile), against 7.8 ms solo.

**What the fixed cost is made of.** The smallest MS of an
`NS_KEEP_MEASUREMENT_SETS=1` run (5967 rows, 3 channels, 3 major cycles)
through `wsclean-zygote`, pinned to one CPU, 20 requests: 26.8 ms median on
the shipped SIF, **18.3 ms** on a SIF built by `patch-wsclean-sif.sh` with
0001-0014 and round 60's zygote. On the patched build, the `-log-time`
segments over 0.25 ms are the MS metadata open between the banner and the
imaging table (2.3 ms), the reorder (2.2 ms), the PSF beam fit (2.1 ms),
then 0.4-0.8 ms for each load, grid and predict. A 300-request `perf`
profile puts the beam fit (GSL `lmsder` over reference BLAS: `dnrm2_`,
`ddot_`, `gsl_linalg_householder_hm`, `FittingDerivativeCentered`) at
~8.5% of samples, against 20% for ducc0's gridding kernels. The fit result
is the restoring beam, so swapping BLAS or the fitter would change image
bits; it is left alone.

**Change.** `simulate_point_source_ms.parse_args()` built its
`ArgumentParser` (22 options, each `add_argument` doing gettext lookups) on
every request. It is now built once per worker (`functools.lru_cache`).

| test | before | after | change |
| --- | ---: | ---: | ---: |
| solo simulate, smallest MS, 30 interleaved pairs in one meqtrees.sif process, median | 7.513 ms | 7.105 ms | **-0.41 ms** (30 of 30) |

The two MSs are byte-identical (`diff -r`), and `./ri self-check simulate`
passes. At ~200 ms of worker time per evaluation this is ~0.2%, far below
what an end-to-end A/B can see, so this is a direct stage timing only. The
module is bound over the SIF's copy, so it reaches the next cluster run with
no rebuild.

**Learned.**
- `cProfile` put the parser at 1.3 ms a request; timed without the
  profiler it is 0.41 ms. Size small Python wins with interleaved
  wall-clock pairs, not with profiler cumulative times.
- What remains of the per-evaluation floor is spread out (MS open, reorder,
  beam fit, ~2 ms each). The bulk of binary time scales with visibilities
  x major cycles, so the gridding and the per-cycle load/predict path are
  still where big wins would come from.
- The production `images/wsclean.sif` is still the 21 Sep 0001-0006 build:
  the smallest evaluation is 46% slower on it than on a 0001-0014 build
  (26.8 vs 18.3 ms). `./ri images patch-wsclean` is still the largest single
  speed-up open to a cluster run.

## Round 62: the one-timestep template copied from memory

**Profile.** `PROFILE=1 scripts/probe_simulate_stage.py .probe/params15.jsonl warm`
in `meqtrees.sif` (fastnoise.so bound, one pinned CPU, 48 real draws) puts a
warm extended simulate at 17.8 ms median. Of the parts left after rounds
26-61, `_add_complex_noise` is ~5.6 ms an evaluation, the 20 `putcol`s
~3.5 ms, `extend_template`'s `shutil.copytree` of the one-timestep template
~2.5 ms and the phase ramp ~1.7 ms. The copy is 39 files an evaluation, and
each is a source open, a `sendfile`, a `copystat` (`utime` + `chmod`) and
several `stat`s, all for ~400 KB.

**Change.** `_copy_template()` reads a template's files once per worker and
writes later copies from those bytes (`mkdir`, then `open(O_EXCL)` + one
`write` per file, with the source's mode bits). A published template is never
written again, and a republished one is a new directory, so the cache is keyed
by the template directory's inode and mtime. Each template is ~0.5 MB, and
there is one per channel count.

**Result.** `.probe/ab62.py` (its last argument is the directory holding `probe_simulate_stage.py`'s `argv()`, copied out): one process warms every
observation and template, checks both arms' MS for every draw, then times
the 48 draws per arm with the order alternating each round.

| test | `copytree` | from memory | change |
| --- | ---: | ---: | ---: |
| 1 process, pinned, 30 rounds, median of per-round means | 17.690 ms | 16.457 ms | **-1.25 ms (-7.0%)**, 30 of 30 |
| 16 processes at once on 16 pinned CPUs, 10 rounds each | 22.43-22.69 ms | 21.09-21.41 ms | **-1.26 to -1.35 ms (-5.7%)**, 160 of 160 |

All 48 draws give byte-identical MSs in every copy (768 of 768, `filecmp`
over every file), and the file modes match. `./ri self-check simulate`
passes; its observation-prefix check compares extended MSs with fresh makems
builds column by column. At ~200 ms of worker time per WSClean evaluation
this is ~0.6%, too small for an end-to-end A/B to see (round 61), so it is a
stage timing only and no search was run for the ledger. The module is bound
over the SIF's copy, so the next cluster run gets it without a rebuild.

**Learned.**
- On tmpfs, `shutil.copytree` of a small casacore table costs ~65 us a file,
  most of it metadata syscalls. For a source that never changes, keep the
  bytes in the process.
- Simulate in a 16-way load is ~27% slower than solo (22.5 against 17.7 ms)
  even though no imager is running. A 16-rank search's in-run simulate
  (~35 ms, round 61) is slower again, by about as much as the gap from solo
  to 16 copies.

## Round 63 (rejected): the noise generator split into lanes or vectors

**Profile.** A 16-rank synchronous search on a SIF built by
`patch-wsclean-sif.sh` with 0001-0014 and round 60's zygote (nlive 150, 450
dead, seed 4242, so the same 2003 evaluations as round 61's cost-model run;
`r63-prof` in `benchmarks.jsonl`) ran at 31.9 evals/s, against 25.3 for round
61's run on the shipped SIF. That is not an A/B: the runs were an hour apart
and the host load differed. Its worker time was 165.6 ms an evaluation:
binary 119.5 ms, simulate 44.9 ms. `./ri profile --phases` over the logs
prune kept puts gridding at 30% of logged binary work, loading and
predicting at 17% each, and the reorder at 17%. Simulate is now 27% of
worker time. In-run it fits 26 ms + 74 ms/Mvis, but makems misses for new
observations early in the run inflate that constant (the slowest 50
evaluations hold 20% of simulate time, at 0.4-0.8 s each). Solo on 51 of the
run's draws spread across sizes (one pinned CPU, warm cache, `PROFILE=1`),
`_add_complex_noise` took 8.2 ms of an evaluation, the `putcol`s 4.5 ms and
the phase ramp 2.3 ms.

**What was tried.** `add_normal_f32` draws ~4.0 ns a normal, one PCG64 step
after another. Two rewrites, both bit-identical to it on the outputs and the
final state, were tested in a C harness (gcc 8.5 `-O2 -ffp-contract=off`,
pinned, best of 30):

| variant | 1k draws | 100k | 1.1M |
| --- | ---: | ---: | ---: |
| current: a serial step and the ziggurat in one loop | 3.94 ns | 4.03 ns | 4.05 ns |
| raw outputs from 4 jump-ahead lanes into a 512-word buffer, then the ziggurat over the buffer | 4.69 ns | 3.95 ns | 3.98 ns |
| the same, with an AVX-512 ziggurat fast path (8 gathered `ki`/`wi`, scalar if any of the 8 is rejected) | 6.70 ns | 6.06 ns | 6.13 ns |

Timed apart, the lane fill is 1.28 ns a raw output and the scalar ziggurat
1.70 ns. So the serial step is not latency-bound: three 64-bit multiplies
for each 128-bit step already take the multiplier's throughput, and removing
the dependency chain frees nothing. The AVX-512 path loses on its two
gathers, on the 9% of 8-blocks that hold a rejection and fall back to
scalar, and on writing strided floats one at a time. Neither variant was
shipped.

**Learned.**
- Noise is at ~4 ns a normal against a ~3 ns floor (raw output plus
  ziggurat). Making it much faster would need fewer multiplies per step,
  which means a different generator and a different noise stream, so it is
  not byte-safe.
- On a 0001-0014 build, the remaining per-evaluation binary work is mostly
  gridding, load and predict, which scale with visibilities times major
  cycles (round 61). The reorder is the largest once-per-evaluation phase.

## Round 65: XY and YX drawn but not computed

**Why.** `-pol I` imaging reads XX and YY only, so half of the thermal noise
`write_visibilities()` adds to a 4-correlation `DATA` column is never read.
Noise was the largest part of simulate (8.2 ms of an evaluation solo, round
63). The draws are still needed: the stream is interleaved per value (all
reals, then all imaginaries), so skipping a draw would move every later one.

**Change.** `add_normal_f32_xxyy()` in `fastnoise.c` walks each 4-value group
as fast, skip, skip, fast. `skip_normal()` steps the PCG64 state and runs the
ziggurat's accept test, but builds the double only when the draw is rejected
and `slow_normal()` needs it. There is no float conversion, multiply or store
for XY and YX. `add_complex_noise(..., xxyy=True)` takes this path when the
library has the symbol and passes a check on load: XX and YY must match
numpy, XY and YX must be untouched, and the generator must end in the same
state. Otherwise it falls back to the full fill. So a kept MS's XY and YX
columns now hold only the predict (zero for this unpolarised source), not
noise. The `.mat` R2D2 reads is correlation 0 and is unchanged. In a C
harness, a generic per-draw `if (keep & bit)` loop was 35-38% slower than the
original, and putting `skip_normal` into the existing `every` loop was 14-16%
slower (register pressure, as in round 56). Only the fixed, unrolled body
wins. Cluster runs get the change with no rebuild, through the host-built
copy in `start-sidecars.sh`.

**Solo.** 168,480 complex values (351 x 120 rows, 4 channels, 4
correlations), in the image, mean of 20 seeds, two alternations: 5.10/5.00 ms
before, 4.27/4.27 ms after (-15%). The C harness gave 8.25 -> 7.05 ms at 1.1M
values.

**In run.** Raced 17-rank pairs, `NS_SEED=4242`, nlive 150. Base is a `git
archive HEAD` copy and new is this tree. The arms launched 10 s apart, and
each order was run twice:

| pass | first launched | base simulate ms | new simulate ms | base evals/s | new evals/s |
| --- | --- | ---: | ---: | ---: | ---: |
| sync, ndead 300 | base | 32.62 | 28.92 | 24.75 | 25.09 |
| async a, ndead 600 | base | 30.34 | 27.35 | 68.61 | 71.42 |
| async b, ndead 600 | new | 30.43 | 30.30 | 66.46 | 65.93 |
| sync2, ndead 300 | new | 29.68 | 28.94 | 8.41 | 8.52 |

Simulate was lower in the new arm in all four pairs. The arm launched second
gains ~1.5 ms, and averaging over both orders gives -1.9 ms (-6% of in-run
simulate, ~0.8% of worker time). That is too small to show in evals/s (sync2
ran with load ~59 on the node from outside this job, hence its 65% idle).
Both sync passes gave the same log(Z) (0.09607 +/- 0.00114) and all 8 chain
files byte-identical between the arms, so WSClean does not read XY or YX.

**Learned.**
- When raced arms start 10 s apart, launch order alone is worth ~1.5 ms of
  in-run simulate here. A single pair cannot resolve an effect of that size,
  so run both orders.

## Round 66: the MS holds XX and YY only

**Why.** Round 65 showed that `-pol I` never reads XY and YX: with their
noise gone, every chain file was still identical. They were still carried
through the pipeline, though. Simulate wrote them, and WSClean's reorder
read them: half of `DATA` and `FLAG`.

**Gate.** Six real draws (`params15`) were each simulated, and a copy was
cut to 2 correlations (`DATA` and `FLAG` re-added at `[nchan, 2]`,
`POLARIZATION` set to `[XX, YY]`). Both were imaged with the search's
`wsclean` arguments, using the shipped `wsclean.sif` with production malloc
tunables, one process per run, pinned to one CPU. All 24 FITS files
(image, model, residual and psf for each MS) were byte-identical. The time
per pass over the six MSs was 601.9 ms with 4 correlations and 587.5 ms
with 2 (**-2.4%**, median of 7 warm alternating passes, 6 of 7 lower).

**Change.** `_write_template()`, formerly `_copy_with_large_ssm_buckets()`,
now also cuts the one-timestep template to XX and YY. `DATA` and `FLAG` are
re-added at `[nchan, 2]`, keeping their tile size in bytes. `WEIGHT` and
`SIGMA` become `[2]`, and `POLARIZATION` becomes `NUM_CORR` 2 with
`CORR_TYPE` `[9, 12]`. Every extended evaluation, which in a long run is
nearly all of them, inherits this layout. Templates now sit in
`templates-xxyy/`, so a run of older code on the same node never reads one.
A new `add_normal_f32_xxyy2()` in `fastnoise.c`, sharing round 65's
unrolled body with YY at offset 1 instead of 3, draws the 4-correlation
stream and stores only XX and YY. The numpy fallback draws into a
4-correlation scratch array. Both are bit for bit what XX and YY were
before, as `_xxyy2_agrees()` at load time and the fast-noise self-check
confirm. The `.mat` path still draws its stream 4 wide (`STREAM_CORRELATIONS`).
A cached makems skeleton or a fresh makems build stays at 4 correlations,
and those MSs image the same. Cluster runs pick this up without a rebuild,
because the module is bound over the SIF's copy and `start-sidecars.sh`
compiles the new `fastnoise.c`.

**Identity.** The new simulator's `DATA` and `FLAG` match the gate's
converted MSs byte for byte, on both the template-miss and the template-hit
paths, and imaging them gives the same 24 FITS files. `./ri self-check
simulate` passes. The observation-prefix check now projects the makems
build onto XX and YY before it compares. In the raced sync pass, all chain
files were byte-identical between the arms, with log(Z) 0.09607 +/- 0.00114
in both.

**In run.** These were raced 17-rank pairs (`.probe/r66ab.sh`,
`NS_SEED=4242`, nlive 150). The base arm was a `git archive HEAD` copy:

| pass | first launched | base simulate ms | new simulate ms | base binary ms | new binary ms | base evals/s | new evals/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync, ndead 300 | base | 29.23 | 26.12 | 198.63 | 196.38 | 25.21 | 26.02 |
| async a, ndead 600 | base | 27.96 | 25.41 | 179.08 | 185.22 | 73.19 | 71.64 |
| async b, ndead 600 | new | 28.45 | 25.07 | 191.72 | 170.76 | 68.19 | 76.40 |

Simulate was lower by 3.1, 2.6 and 3.4 ms (**-10%**), in both launch
orders, so this is not round 65's ~1.5 ms launch-order effect. The binary
can only be compared in the sync pass, where both arms image the same
draws: there it was -1.1% (-2.25 ms) and evals/s +3.2%. The async arms draw
different points, so their binary and evals/s gaps (+3.4% and -11%) are
sampling, not code. Together, worker time per evaluation falls by about
3 ms of simulate plus ~1-2.5% of the binary, which is roughly 2.5% of
worker time.

**Learned.**
- An MS needs only the correlations the imager reads. WSClean `-pol I` on
  an `[XX, YY]` MS is byte-identical to one on `[XX, XY, YX, YY]`, including
  when `WEIGHT` keeps 4 entries.
- makems has no option for fewer correlations (`NPolarizations=2` is
  ignored). Re-adding the columns on the cached template is a one-off cost
  per channel count.
- casacore lists a re-added column's data manager last in `getdminfo()`, so
  a layout comparison has to sort by manager name.

## Round 67 (rejected): blocked u/v divisions for the imaging weights

**Profile.** This round profiled a SIF built by `patch-wsclean-sif.sh` with
0001-0014. `perf record` ran 3 passes over the three largest real-draw MSs
from round 24's `results/r24ms`, pinned to one CPU, with the search's
`wsclean` arguments (`-size 32 32`, natural weighting, `-pol I`, `-j 1`).
The self time splits like this:

| function | share |
| --- | ---: |
| ducc0 `x2grid_c_helper<6>` (gridding) | 26.8% |
| ducc0 `grid2x_c_helper<6>` (degridding) | 17.2% |
| ducc0 `Wgridder` constructor (scan, baselines) | 4.2% |
| `MsGridderData::CalculateWeightsImplementation<1>` | 4.1% |
| `ApplyWeightsAndCorrections<4>` | 2.6% |
| `WGriddingMSGridder::GridRegularMeasurementSet` | 2.3% |
| `ImageWeights::Grid` | 2.0% |
| `ReorderedMsReader` `CacheMeta` + `ReadMeta` + `ReadWeights` + `NextInputRow` | 5.8% |
| `countRanges` (three symbols) | 3.0% |

On the largest MS (388k rows, 6 channels, 4 major cycles, 1.1 s), the
`-log-time` timeline shows the reorder at 137 ms. The natural-weights pass
takes 29 ms and the w-limits pass 13 ms. Each major cycle spends 31-35 ms
"loading data in memory" before 57-89 ms of gridding, and each predict
spends 13 ms loading metadata, 58 ms predicting and 7-12 ms writing. The
per-cycle load, at ~15% of the binary, is the largest cost outside ducc0.

**What was tried.** `CalculateWeightsImplementation` and
`ImageWeights::Grid` compute `u / (c / f[ch])` and `v / (c / f[ch])` per
channel, row by row, on every pass. A candidate patch 0015 computed these
quotients first, into 16-channel stack blocks the compiler can vectorise,
then did the grid lookups. IEEE division gives the same quotient either
way. Solo ABBA (6 blocks x 3 reps x 5 MSs, pinned, median of reps 2-3 per
block):

| MS | 0001-0014 | + 0015 | change | 0015 faster |
| --- | ---: | ---: | ---: | ---: |
| largest | 1109.2 ms | 1127.8 ms | +1.7% | 0/6 |
| 2nd largest | 557.8 ms | 565.2 ms | +1.3% | 0/6 |
| 3rd largest | 404.2 ms | 410.5 ms | +1.6% | 0/6 |
| median-size | 125.7 ms | 127.6 ms | +1.5% | 2/6 |
| smallest | 54.5 ms | 54.8 ms | +0.5% | 2/6 |

All 20 FITS data blocks were byte-identical, but the patch is slower. Under
perf, `CalculateWeightsImplementation` rose from 4.1% to 5.5%, while
`ImageWeights::Grid` fell only from 2.0% to 1.7%. It was not shipped.

**Learned.**
- `perf annotate` on the shipped code shows the divisions were never the
  cost. With 1-8 channels per row, the function is per-row overhead: the
  `is_selected` zeroing loop takes ~15% of its samples, and the double to
  int conversions and the `_grid` load take most of the rest. Blocking adds
  stack traffic and vector prologues for a loop that runs at most 8 times.
- Per-channel restructuring doesn't pay at 1-8 channels per row. Any win
  in the weights path has to remove per-row work. One option is to
  compute image weights once per evaluation rather than on every major
  cycle, since they are identical on every pass.
- `perf` can resolve symbols for a SIF's binaries through `--symfs` over a
  directory holding copies taken with `apptainer exec <sif> cat <path>`.
  Under a per-run `apptainer exec`, ~6% of samples are the host's libz,
  which is squashfs decompression at container start. A zygote-run search
  doesn't pay this.

## Round 68: row weights reused across major cycles (patch 0015)

**Why.** Round 67's timeline showed every inversion of the largest
real-draw MS spending 31-35 ms "loading data in memory" before gridding,
~15% of the binary, and the weights part of that is recomputed identically
on every pass. Only the residual (data minus model) changes between cycles.

**Change.** Patch 0015 (`docker/wsclean/patches/`, described in
[nested-sampling-wsclean-patches.md](nested-sampling-wsclean-patches.md))
keeps each row's uvw and final weight from the first inversion of an MS,
plus the weight counters that pass ends with. Later inversions read only
data and model, subtract, multiply by the cached weight, and restore the
counters. It applies only where the full path reduces to exactly that
(one visibility polarization, no facets, beams, solutions or stored imaging
weights), and the cache is keyed on the process id, so a forked zygote child
starts empty. Built with `patch-wsclean-sif.sh` over 0001-0014 into a
scratch SIF; the A arm was the same script without 0015.

**Phase timing.** With `-log-time` on the largest MS (388k rows, 6
inversions), "loading data in memory" stays ~33 ms on the first inversion
and falls to 14-16 ms on the five that follow.

**Solo.** Four chains pinned to one CPU each ran ABBA blocks at the same
time (6 blocks per chain, 3 reps per MS, mean of reps 2-3, the search's
`wsclean` arguments). The node's load average from other jobs was 35-41:

| MS | 0001-0014 | + 0015 | change | 0015 faster |
| --- | ---: | ---: | ---: | ---: |
| largest | 1128.6 ms | 1045.8 ms | -7.3% | 24/24 |
| 2nd largest | 565.5 ms | 527.3 ms | -6.8% | 24/24 |
| 3rd largest | 413.9 ms | 388.2 ms | -6.2% | 24/24 |
| median-size | 128.4 ms | 123.4 ms | -3.9% | 22/24 |
| smallest | 42.3 ms | 42.5 ms | +0.3% | 11/24 |

All 1440 FITS files had byte-identical data blocks, and identical headers
apart from DATE and HISTORY (so `WSCNVIS`, `WSCENVIS`, `WSCVWSUM`,
`WSCNORMF` and the major-cycle counts match).

**In run.** Synchronous raced 17-rank pairs (`NS_SEED=4242`, nlive 150,
ndead 300), so both arms image the same draws; the arms differ only in
`SIF_DIR`'s `wsclean.sif`:

| pass | first launched | A binary ms | 0015 binary ms | A evals/s | 0015 evals/s |
| --- | --- | ---: | ---: | ---: | ---: |
| sync | A | 162.18 | 150.25 (-7.4%) | 29.84 | 31.38 (+5.2%) |
| sync | 0015 | 161.48 | 150.65 (-6.7%) | 30.09 | 31.95 (+6.2%) |

Every chain file and log(Z) (0.09607 +/- 0.00114) were byte-identical
between the arms in each pass. Simulate is unchanged by the patch (33.9/38.5
and 31.8/30.8 ms, launch order and load), so the whole gain is the binary:
-11 to -12 ms per evaluation, ~6% of worker time at these settings.

**Learned.**
- The per-cycle load was about half weights work: removing the meta read,
  weight read, imaging-weight lookup and counter updates halves it, and
  the rest is data and model reads plus the subtract.
- The gain grows with major cycles times rows, which is the 44% of binary
  time round 61's cost model put on visibilities x cycles; on the
  smallest MS (one or two cycles) it is nil.
- The second pass spent ~9.5 minutes in `uv run` before Python started
  (the first took seconds), with the node's load average at 55-60 from
  other jobs. `benchmarks.jsonl`'s `wall_s` (245-260 s) does not include
  it, so give a raced pair's `timeout` room for that startup.

## Round 69: residual rows read in blocks from the mappings (patch 0016)

**Why.** After round 68, each residual inversion of the largest MS still
spent ~14 ms "loading data in memory" (341k rows x 8 channels here). That is
~6% of the binary over five such passes.

**First attempt (no gain).** A reader call that copied a whole chunk's data
and model at once, followed by one subtract-and-weight loop, left the load at
14-16 ms. So the time is not per-row call overhead. A `perf` profile of the
0015 build shows the pass is memory-bound: data and model are each copied
twice, the uvws once, and the freshly allocated chunk buffer (22 MB here)
page-faults on first touch every pass. The extra model buffer only added
more of the same.

**Change.** Patch 0016 (`docker/wsclean/patches/`, described in
[nested-sampling-wsclean-patches.md](nested-sampling-wsclean-patches.md))
adds `MSReader::MapDataAndModelRows()`. For a regular reordered set it
returns pointers into the data and model mappings for a whole chunk and
moves the cursors as the row reads would. `(data - model) * weight` is then
formed in one loop into a buffer kept in 0015's cache, so its pages stay
mapped, and the cached uvws go to the gridder directly. The operations are
the same, in the same order, so the result is the same.

**Phase timing.** On the largest MS with `-log-time`, the residual passes'
load falls from 13.8-14.2 ms to 5.6-6.0 ms. The PSF and dirty passes are
unchanged, at 36 and 14 ms. WSClean drops from 1091 to 1041 ms (two runs
per arm).

**Solo.** Fresh real-draw MSs, which are XX/YY since round 66, came from an
`NS_KEEP_MEASUREMENT_SETS=1` search. Four chains, each pinned to one CPU,
ran ABBA blocks at the same time (6 blocks per chain, 3 reps per MS, mean of
reps 2-3). A = 0001-0015 (round 68's B2), B = 0001-0016, both built by
`patch-wsclean-sif.sh`. The node's load average from other jobs was 31-35:

| MS | 0001-0015 | + 0016 | change | 0016 faster |
| --- | ---: | ---: | ---: | ---: |
| largest | 1065.1 ms | 1034.0 ms | -2.9% | 24/24 |
| 2nd largest | 1057.6 ms | 1008.6 ms | -4.6% | 24/24 |
| 3rd largest | 1012.1 ms | 967.3 ms | -4.4% | 24/24 |
| median-size | 132.9 ms | 129.3 ms | -2.8% | 20/24 |
| smallest | 41.6 ms | 41.5 ms | -0.3% | 11/24 |

All 1440 FITS data blocks are byte-identical. So are all 1440 headers, apart
from DATE and HISTORY.

**In run.** Synchronous raced 17-rank pairs (`NS_SEED=4242`, nlive 150,
ndead 300) image the same draws. The arms differ only in `SIF_DIR`'s
`wsclean.sif`:

| pass | first launched | A binary ms | 0016 binary ms | A evals/s | 0016 evals/s |
| --- | --- | ---: | ---: | ---: | ---: |
| sync | A | 148.06 | 144.61 (-2.3%) | 31.63 | 32.94 (+4.2%) |
| sync | 0016 | 147.39 | 143.60 (-2.6%) | 32.38 | 32.91 (+1.6%) |

In each pass, every chain file and log(Z) (0.09607 +/- 0.00114) were
byte-identical between the arms. The two evals/s gains differ because in-run
simulate depends on launch order (37.96/30.20 ms, then 29.79/30.66 ms). The
binary, at -3.5 to -3.8 ms per evaluation, is the measure that does not.

**Learned.**
- After 0015, a residual pass's load is memory bandwidth plus page faults,
  not row-loop overhead. Block copies through the reader save nothing.
  Avoiding the copies and reusing a warm buffer does.
- A regular reordered set's data and model files are flat arrays from the
  first row: data from `PartHeader::BINARY_SIZE`, model at
  `current_value_position_`. Callers can therefore read them in place, with
  no row API at all.
- Harness: `.probe/r69inner.sh`, `r69abba.sh`, `r69ab.sh` and
  `r69fitscmp.py` (git-ignored). FITS identity is checked as the data block
  after the END card, plus every header card except DATE and HISTORY.

## Round 70: a run warns when `wsclean.sif` lags the tree's patches

**Why.** Rounds 43 and 61 found that the production `images/wsclean.sif` is
still the 21 Sep build, which has patches 0001-0006 and nothing after. Every
WSClean round from 23 to 69 (0007-0016) therefore reaches a real run only
after someone runs `./ri images patch-wsclean`, and a run gave no sign that
it was missing them. Round 43 measured 0007-0011 alone at -16 to -27% of
binary time. Round 61 found the smallest evaluation 46% slower on the
shipped SIF than on a 0001-0014 build.

**Change.** `ns_warn_stale_wsclean_sif` (`scripts/lib/defaults.sh`), called
by `run-nested-sampling.sh` (and so by `./ri search wsclean` and `./ri
resume`), reads the SIF's labels once (~0.1 s). It warns on stderr, without
refusing, unless one of these holds:

- `ri.wsclean-patches` (written by `patch-wsclean-sif.sh`) equals the hash
  of this tree's `docker/wsclean/patches/*.patch` and `src/*.cpp`;
- or, with no such label, `ri.build-inputs` equals what `scripts/build.sh`
  would label a WSClean image built from this tree.

Both hashes now live in `defaults.sh` (`ns_wsclean_patches_hash`,
`ns_wsclean_build_inputs`, and `inputs_hash`, moved from `build.sh`), so the
builders and the check cannot drift apart. The moved `inputs_hash` gives
the same WSClean and R2D2 hashes as before. Against real SIFs, the shipped
`wsclean.sif` and round 69's 0001-0015 build both warn, and the 0001-0016
build stays quiet. `patch-wsclean-sif.sh` rebuilt into `/dev/shm` with the
shared hash and its output stays quiet. `scripts/test-defaults.sh`, which
CI runs, covers all four label cases with a stand-in `apptainer`. No search
timing changes, so this round has no A/B.

**Simulate today (diagnostic).** `probe_simulate_stage.py warm` with
`PROFILE=1` covered round 63's 48 real draws (`params15`), pinned to one
CPU on the current tree. On the 41 draws whose one-timestep template was
already cached, simulate took **17.8 ms** (mean; median 14.1 ms; mean
49.7k rows). Seven draws built a new template at ~134 ms each, a cost a long
run pays once per channel count and observation. cProfile's per-evaluation
split: XX/YY noise 5.0 ms, which round 63 showed is near its floor; `putcol`
2.9 ms; table opens and closes (~4.6 an evaluation) ~2 ms; the phase ramp
1.4 ms; the observation `.npz` load ~1 ms (1.9 under cProfile).

**Learned.**
- After rounds 56-66, simulate is ~18 ms solo, about 10% of a ~178 ms worker
  evaluation, and in-run simulate is 26-30 ms. MeqTrees itself is not called
  for WSClean (round 1's analytic predict), so the "MeqTrees ~30-50% of
  runtime" stage of the early runs is now this.
- Memoising the observation `.npz` per worker was considered and dropped.
  It would save at most ~1 ms (~0.5% of worker time). Entries are 160-270 KB
  and there are up to 910 keys, so it could hold ~200 MB per rank, and a
  small LRU's hit rate across 910 keys is unknown.
- Round 69's three biggest remaining shares are all inside WSClean: ducc0
  x2grid (30%), grid2x (21%) and the Wgridder constructor (5%). On the
  cluster, the largest single speed-up is still bringing
  `images/wsclean.sif` up to 0016, which this round only makes visible.

## Round 71: what the shipped `wsclean.sif` costs a cluster run today

**Why.** Round 70 made a run warn when `images/wsclean.sif` lags the tree,
but each patch since round 23 was measured against the build before it. No
round had measured the whole gap that `./ri images patch-wsclean` closes on
today's code.

**Measure.** `patch-wsclean-sif.sh` built 0001-0016 over the shipped SIF
into `/dev/shm` (1m58s, 36 jobs). Two `SIF_DIR`s of symlinks that differ
only in `wsclean.sif` raced two 17-rank searches on this half node
(cpu-q-278, load ~11 from other jobs): nlive 150, 15 repeats, 300 dead
points, synchronous, seed 4242. The two passes launched opposite arms first
(`.probe/r71ab.sh`).

| pass | arm | evals | evals/s | ms/eval | simulate | binary | peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A first | shipped (0001-0006) | 7821 | 26.35 | 223.1 | 29.4 | 192.6 | 182 |
| A first | tree (0001-0016) | 7821 | 37.24 | 156.4 | 28.9 | 126.3 | 278 |
| B first | shipped (0001-0006) | 7821 | 26.45 | 220.1 | 25.8 | 193.3 | 182 |
| B first | tree (0001-0016) | 7821 | 37.59 | 154.9 | 27.6 | 126.2 | 278 |

Binary time falls **34.6%** (193 -> 126 ms) and evals/s rises **41-42%**
in both orders. log(Z) is 0.09607 +/- 0.00114 in all four runs, and every
`chains/` file, `clusters/` included, is byte-identical between arms and
across passes. That checks 0007-0016 together, not one at a time. The
patched arm finished ~85 s early, so the shipped arm ran its tail alone, and
the gap is if anything understated. Peak memory per rank rises 96 MB, from
0015's per-MS weight cache and 0016's residual buffer. At ~0.28 GB per rank
that is still far inside a node.

**Fix: bench could not tell the two apart.** A `Bootstrap: localimage` SIF
inherits its base's labels, so `ns_image_id` gave both arms the same
`NS_IMAGER_IMAGE_ID`. `./ri bench` groups rows by image id, so it would have
pooled a 42% difference into one group. `ns_image_id`
(`scripts/lib/defaults.sh`) now appends `+p` plus the first 12 characters of
`ri.wsclean-patches` when that label is present. Unpatched images keep
their ids unchanged. `scripts/test-defaults.sh` gains three cases (plain,
patched, unlabelled). The two r71 tree-arm rows in `benchmarks.jsonl` were
recorded before the fix and had the suffix added by hand, as their notes
say.

**Learned.**
- On the cluster as shipped, WSClean runs at ~70% of the speed this tree
  can reach. That is the largest single speed-up available, and it costs
  one 2-minute compute-node job. It is still left for the user to run,
  because `images/` is shared with their checkout.
- Round 69's 0001-0016 arm took 144 ms of binary time on the same node,
  same settings and same draws. Here the same patches took 126 ms with the
  node's load average at ~11. Other jobs' load moves absolute times by
  ~12%, so only raced arms compare.

## Round 72 (diagnostic, one rejection): a production-shaped search on 0001-0016, and `-j N` for the lone last chain

**Question.** With every WSClean patch in (round 71's 0001-0016 SIF), what
does a production-shaped search spend its worker time on, and is any of the
idle worker time something a large run would also pay?

**Run.** 36 ranks, async, nlive 150, 15 repeats, 900 dead, seed 4242,
r71's patched `wsclean.sif`, cpu-q-278 at load ~25 (row `r72-prof`):
26,899 evaluations in 154 s, **174 evals/s** whole run.

| stage (`./ri profile`) | per eval | share of worker time |
| --- | ---: | ---: |
| simulate (MeqTrees) | 28 ms | 14.0% |
| `wsclean` binary | 123 ms | 61.0% |
| zygote, metrics, harness | 1.2 ms | 0.6% |
| PolyChord (no evaluation in flight) | | 0.4% |
| idle | | **23.9%** |

MeqTrees is 18% of evaluation time, down from the 30-50% that motivated
this work. Busy workers per 7.7 s window, from each evaluation's
`started_epoch`/`ended_epoch` (`.probe/r72busy.py`):

| window | busy | evals/s |
| --- | ---: | ---: |
| 0-8 s | 91% | 183 |
| 8-115 s | 97-98.5% | 188-266 (**233** mean over 10-110 s) |
| 115-123 s | 38% | 55 |
| 123-154 s | 3-9% | 3-9 |

So mid-run idle is ~2%. The 24% is one worker finishing the last
slice-sampling chain alone (round 39): 470 evaluations end after 116 s, the
last ~28 s on a single rank, back to back. The tail's evaluations sit in
the converged region and are heavy: 318 ms of binary time against 123 ms
for the run as a whole. A converged production search is ~1,400 dead points
(round 51), so ~4 minutes of steady state now carries a ~35 s tail, ~13%.

**Rejected: `-j N` for the evaluations a busy count says are alone.** Round
9 gives a lone R2D2 request the idle cores. The WSClean equivalent is `-j N`.
On four real-draw MSs from round 69 (`.probe/r72j.sh`, inside the 0001-0016
SIF, 3 reps each, `-niter 100000` as in a search):

| MS | `-j 1` | `-j 2` | `-j 4` | `-j 8` |
| --- | ---: | ---: | ---: | ---: |
| m1 (largest) | 1.157 s | 0.942 | 0.757 | 0.671 (-42%) |
| m2 | 1.013 s | 0.949 | 0.802 | 0.718 (-29%) |
| m4 (tail-sized) | 0.140 s | 0.136 | 0.124 | 0.112 (-20%) |

Every `-j N > 1` output differs from `-j 1`, the PSF included: up to
1.3e-4 of the PSF peak and 8e-5 of the image peak on m1. `-j 2` and `-j 8`
are not even reproducible run to run (2e-7 to 1.4e-6). An objective that
depends on how many other ranks were busy breaks same-seed reproducibility,
and at tail-sized evaluations the gain is ~20%, i.e. ~6 s off a ~30 s tail.
Not taken.

**Learned.**
- On a fully patched SIF, a WSClean search is saturated mid-run (97-98.5%
  of 35 workers busy). Short benchmarks' whole-run evals/s understates
  steady throughput by ~25% at 900 dead (174 against 233). Compare steady
  windows, or same-`NS_MAX_NDEAD` rows only.
- The remaining large-run cost outside the worker evaluation is the last
  chain's ~30 s. (Wrong, as round 81 found: PolyChord throws those
  chains away, so they now end as soon as it has finished.) Per-evaluation work, i.e. ducc0 gridding (round 69) and
  simulate's 28 ms, is still what scales.
- WSClean's `-j` is not a thread knob that leaves the output alone. Even
  the PSF changes, so `-j 1` has to stay fixed for every evaluation.

## Round 74: the w-range scan skipped when w-gridding is off (patch 0017)

**Profile.** `perf record` on five runs of the largest real-draw MS (round
69's m1) through a 0001-0016 build, with symbols taken from round 69's
unstripped binary bound over the SIF's (the SIF's own is stripped, so
`perf report` prints bare addresses; map them with `nm -C` against the
unstripped copy):

| symbol | share of samples |
| --- | ---: |
| ducc0 `x2grid_c_helper<6>` (gridding) | 32.3% |
| ducc0 `grid2x_c_helper<6>` (degridding) | 21.7% |
| ducc0 `Wgridder` constructor (inlined `scanData`) | 5.3% |
| `GridRegularMeasurementSet` | 2.9% |
| `ImageWeights::Grid` | 2.2% |
| ducc0 `countRanges` | 3.0% |
| everything else | under 1.4% each |

The constructor's hot loop is `scanData()`'s 0009 fast path: per visibility,
`norm(vis) != 0`, then `|w * f/c|` folded into a min and max. Those two values
are read only under `do_wgridding` (w-plane count, `wmin`, kernel cost
model), and 0006 turns w-gridding off.

**Change.** Patch 0017 skips the w arithmetic and the min/max in that fast
path when `do_wgridding` is false. The mask, the visibility count and the
zeroed degrid outputs are built exactly as before.

**Solo.** `.probe/r74/` harness (round 69's: five MSs, 3 reps a block, 6 ABBA
blocks on each of 4 pinned CPUs, `-niter 1000`), median per block:

| MS | 0001-0016 | 0001-0017 | change | 0017 faster |
| --- | ---: | ---: | ---: | ---: |
| largest | 1047.8 ms | 1027.7 ms | -1.9% | 21/24 |
| 2nd largest | 1011.2 ms | 988.7 ms | -2.2% | 24/24 |
| 3rd largest | 963.9 ms | 942.2 ms | -2.3% | 24/24 |
| median-size | 127.5 ms | 126.6 ms | -0.7% | 17/24 |
| smallest | 40.9 ms | 41.2 ms | +0.9% | 8/24 |

All 1440 FITS data blocks (image, model, psf, residual) are byte-identical.

**In run.** Raced synchronous 17-rank pairs (`.probe/r74ab.sh`: seed 4242,
nlive 150, 300 dead, same draws), arms differing only in `wsclean.sif`:

| pass | first launched | A binary ms | 0017 binary ms | A evals/s | 0017 evals/s |
| --- | --- | ---: | ---: | ---: | ---: |
| sync | A | 126.62 | 125.62 (-0.8%) | 37.34 | 37.91 (+1.5%) |
| sync | 0017 | 128.38 | 125.85 (-2.0%) | 37.15 | 37.73 (+1.6%) |

In both passes every chain file and log(Z) (0.09607 +/- 0.00114) are
byte-identical between the arms, and peak memory is unchanged (277-279 MB).
The in-run gain is smaller than the solo one on large MSs because most
evaluations are median-sized or smaller, where the scan is a smaller share.
Like 0007-0016, it reaches a cluster run only through `./ri images
patch-wsclean`.

**Learned.**
- With 0001-0016 in, the two ducc0 kernels are 54% of a large evaluation's
  binary samples; everything outside them is spread across items of 3% or
  less. 0017 was the largest of those that is provably dead work.
- `perf` cannot resolve symbols in a SIF's stripped `wsclean`. Bind an
  unstripped build over `/opt/wsclean/bin/wsclean` (it must be executable)
  and map addresses with `nm -C` yourself: `perf report` still printed
  bare addresses with the bound binary.
- When waiting on a background job by process name, `pgrep -f` matches the
  waiting shell's own command line. Wait on recorded PIDs or an output file.

## Round 75 (rejected): WSClean built with profile-guided optimisation

**Question.** Nothing in this log has tried the compiler itself. With the
ducc0 kernels at 54% of the binary (round 74), does GCC profile-guided
optimisation (PGO) of the 0001-0017 build buy anything?

**Build.** `.probe/r75/build.sh` and `use.sh`: the same bookworm sandbox
and flags as `patch-wsclean-sif.sh` (`-O3 -march=x86-64-v3`, GCC 12), with
`-fprofile-generate -fprofile-update=single` added. It was trained on
round 69's five real-draw MSs, twice each at a search's `-niter 100000`
(90 `.gcda` files), then rebuilt with `-fprofile-use -fprofile-correction`
and layered over round 74's 0001-0017 SIF. Radler is an ExternalProject
with its own `CMAKE_CXX_FLAGS`, so its deconvolution stayed unprofiled.
ducc0 and the rest of WSClean were profiled.

**Solo.** Round 74's harness: five MSs, 3 reps a block, 6 ABBA blocks on
each of 4 pinned CPUs, `-niter 1000`. A is the plain 0001-0017 build, B is
the PGO build. Median per block:

| MS | plain | PGO | change | PGO faster |
| --- | ---: | ---: | ---: | ---: |
| largest | 1014.8 ms | 1034.1 ms | +1.9% | 3/24 |
| 2nd largest | 988.6 ms | 997.0 ms | +0.9% | 5/24 |
| 3rd largest | 941.4 ms | 954.8 ms | +1.4% | 2/24 |
| median-size | 126.7 ms | 128.4 ms | +1.3% | 4/24 |
| smallest | 41.2 ms | 40.9 ms | -0.6% | 15/24 |

None of the 1440 FITS data blocks is byte-identical between the arms, PSF
included. The largest difference is 1.8e-7 of the image peak. Within one
arm, repeat runs differ only in the header's date bytes.

**Rejected** on both counts: PGO is ~1-2% slower where the time is spent,
and it changes every output. No in-run A/B was run.

**Learned.**
- PGO moves WSClean's floating-point results, most likely through FMA. GCC contracts `a*b+c` into
  FMA by default (`-ffp-contract=fast`, and FMA is in `x86-64-v3`), so a
  change in inlining changes which operations are fused. Any build-flag
  experiment on WSClean (PGO, LTO, another compiler) must check FITS bytes
  first; a speed gain alone is not enough.
- The ducc0 kernels are hand-vectorised, branch-light template code, so a
  profile gives GCC little to act on. The hot path is already what `-O3`
  emits.
- In a WSClean FITS file from a 32x32 search, the data block is bytes
  5760-9856 (two 2880-byte header blocks, then 4096 bytes of float32).
  Compare `tail -c +5761`, not the whole file: the header carries the date.

## Round 76 (diagnostic): why simulate is ~1.5x slower inside a run

**Question.** Round 72 measured simulate at 28 ms an evaluation in a
36-rank search, against ~18 ms solo (round 70). Is that gap a kernel or
tmpfs lock that fewer syscalls could fix, a drop in clock speed, or memory?

**Harness.** `.probe/r76/run.sh N REPS`: N `meqtrees.sif` processes, one
per pinned CPU of this job (the even CPUs, all of socket 0, no SMT). Each
warms round 63's 48 real draws (`params15`) and then times `simulate()` on
all of them together, from a shared start file. `getrusage` gives user and
system time per call. No imager runs, so the only load is simulate itself.

| copies | wall | user | system | IPC (user) | LLC demand loads per call | LLC miss rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.4-20.7 ms | 12.4-13.0 | 2.5-3.0 | 2.99 | ~7k | 24% |
| 17 | 24.4 ms | 15.4 | 4.0 | | | |
| 34 | 29.3-32.1 ms | 18.5-19.7 | 5.0-5.4 | 1.85 | ~100k | 72% |

(IPC and LLC counts are `perf stat -e cycles,instructions,LLC-loads,LLC-load-misses`
over 3 s of the timed loop.)

- **Not a lock.** User time grows as much as system time (+51% against
  +78%), voluntary context switches stay at ~3 a call, and an strace of the
  timed loop shows no slow syscalls. A call makes ~1,100 `lseek`, ~650
  `write`, ~390 `newfstatat` (casacore re-stats `table.dat` ~26 times a
  call), ~170 `fcntl` (table locks) and ~63 `unlinkat`. Even removing all
  of the ~3 ms of solo system time would save ~1.5% of an evaluation.
- **Not the clock.** A pure-Python integer loop takes 244 ms solo and 241
  ms across 34 copies.
- **Memory.** Demand L2 misses per call rise ~14x and three quarters of
  them miss the shared 58 MB L3 as well. The jump is superlinear in copies
  (+18% at 17, +55% at 34). That is what a hot set of a few MB a copy does
  once 34 copies no longer fit in L3 together (58/17 = 3.4 MB a copy fits,
  58/34 = 1.7 MB does not). Solo, the prefetchers hide almost all of it.
- The ~4.5 ms solo gap between wall time and user+system time is makems.
  The first pass over the draws builds 7 one-timestep templates in a child
  process, which `RUSAGE_SELF` does not count. It is not waiting.

**Learned.**
- At one rank per CPU, simulate's in-run cost is ~10 ms of L3 misses on top
  of its ~18 ms of work. Neither fewer syscalls nor locking changes reach
  it. The lever is a smaller working set per call: fewer full-size
  complex64 temporaries between the noise, the phase ramp and `putcol`.
  At ~14% of worker time that is worth at most ~5%.
- Benchmark simulate changes under a 34-copy load as well as solo. A
  change that shrinks the working set can be worth up to ~1.5x more
  in-run than solo, and a solo-only timing would miss that.
- `Cpus_allowed_list` on this allocation is the even CPUs 0-74, i.e. all of
  NUMA node 0. `taskset -c i` on an odd CPU fails, so pin with `2*i`.

## Round 78: the observation cache read with pickle, not `np.load`

**Change.** `simulate_point_source_ms.py` stored each cached makems
observation (round 36: times, per-antenna UVW, interval, exposure, field
directions) as an `.npz`, and every extended evaluation read one back with
`np.load`. The zip parse costs more than the ~100-270 kB of arrays it reads.
The cache is now a protocol-5 pickle of the same arrays (`.pkl`). This round
finishes gnhf 77's change, which was cut off before it was measured.

**Read alone** (`meqtrees.sif`, three real observations of 100-266 kB, 300
loads each, twice): `np.load` 0.76-0.95 ms, `pickle.load` 0.03-0.08 ms. Every
array came back equal, with the same dtype and shape.

**In-run** (`.probe/r78/run.sh`, round 76's harness: pinned
`meqtrees.sif` copies each warm round 63's 48 draws, then time `simulate()`
from a shared start file; ABBA, 2 passes of 48 draws a copy). This job's
CPU set had 29 usable even CPUs, so 29 copies:

| pair | A (`.npz`) wall / user ms | B (pickle) wall / user ms |
|---|---|---|
| 1 (AB) | 30.36 / 17.27 | 29.08 / 16.07 |
| 2 (BA) | 30.61 / 17.52 | 29.26 / 16.22 |
| 3 (AB) | 30.71 / 17.50 | 29.35 / 16.16 |
| 4 (BA) | 30.29 / 17.31 | 29.42 / 16.31 |
| mean | 30.49 / 17.40 | 29.28 / 16.19 |

Simulate is 1.21 ms (-4.0%) faster in-run, 4/4 pairs, all of it user time
(system time 4.7 ms in both). That is about 0.8% of a 154 ms WSClean
evaluation (round 74's 17-rank search), and is too small to see end to end
(AGENTS.md: a three-repeat control spans ~5%), so it was timed at the stage.

**Results.** Unchanged: the cached arrays load bit-identical, and
`./ri self-check simulate` passes (observation prefix, mat-only and
analytic predict included). No search ran, so there is no
`benchmarks.jsonl` row.

**Learned.**
- `np.load` of a small `.npz` is ~0.8 ms of fixed cost (zip directory,
  one header parse a member). For a hot cache of small arrays, pickle is
  ~15x cheaper and loads the same bytes.
- The new file name means the persistent cache
  (`NS_OBSERVATION_CACHE_DIR`, ~200 MB per meqtrees image id) starts cold
  once and is rebuilt as `.pkl`. The old `.npz` files are no longer read
  and can be deleted.
- A harness that leaves each copy's scratch in `/dev/shm` fills the 126 GB
  tmpfs after three 34-copy arms (1.4 GB a copy), and the failures look like
  makems errors. Delete each arm's scratch when it ends.
- CPU sets differ between jobs: this one was
  `0,2,...,20,22-34,36,...,52,64,...,72`, not round 76's even 0-74. Read
  `Cpus_allowed_list` before pinning.

## Round 79: FLAG left to casacore's zero-filled tiles

**Change.** `write_visibilities()` wrote FLAG with
`putcol("FLAG", np.zeros(data.shape, bool))` on every evaluation: a bool
array the size of DATA, which casacore then packs to bits. FLAG is a
`TiledColumnStMan` column in every MS simulate fills (the XX/YY template
and makems' own), and casacore zero-fills new tiles, so every row already
reads False. The putcol is gone. `self_check_observation_prefix()` now
asserts FLAG is all False in both the extended and the makems-built MS.

**Results.** Unchanged. On round 63's 48 real draws, every column of the
main table reads back identical with and without the putcol, and FLAG has
no True value anywhere. Every file of each MS is byte-identical except the
`IncrementalStMan` index `table.f0i`, which also differs between two runs
of the unchanged code (43 of 48 MSs), so it carries uninitialised bytes,
not values. `./ri self-check simulate` passes.

**In-run** (`.probe/r79/run.sh`, round 78's harness with pinning read from
`Cpus_allowed_list`: 34 pinned `meqtrees.sif` copies, each timing 4 passes
of the 48 draws from a shared start file; ABBA):

| pair | A (putcol) wall / user ms | B (no putcol) wall / user ms |
|---|---|---|
| 1 (AB) | 23.90 / 15.38 | 23.65 / 15.22 |
| 2 (BA) | 23.74 / 15.29 | 23.60 / 15.12 |
| 3 (AB) | 23.81 / 15.29 | 23.62 / 15.16 |
| 4 (BA) | 23.72 / 15.21 | 23.60 / 15.16 |
| mean | 23.79 / 15.29 | 23.62 / 15.17 |

Simulate is 0.17 ms (-0.7%) faster in-run, in 4 of 4 pairs. That is about
0.1% of an evaluation, so this is dead work removed rather than a speed-up
anyone will see end to end. No search ran, so there is no
`benchmarks.jsonl` row.

**Learned.**
- An MS column that is about to be all-default does not need a putcol
  when its storage manager zero-fills (tiled managers do). Check the
  values come back the same, not the files: casacore's ISM index file
  differs between identical runs.
- This job's CPU set, `0,2,...,20,22-34,36,...,52,64,...,72` (38 CPUs),
  gave 34-copy in-run simulate at 23.8 ms, against round 78's 30.5 ms
  across 29 copies on even CPUs only. The odd CPUs 23-33 are on socket 1
  (`topology/physical_package_id`), with its own L3, so six copies left
  socket 0's L3. In-run simulate timings are only comparable within one
  job's CPU set.
- A simulate harness must name its MS `sim.ms` in every call: makems'
  `MSName=` is part of the observation-cache key, so a different name
  misses the cache and silently times makems and the 4-correlation MS.

## Round 80: the phase ramp's output left unzeroed

**Change.** `_phase_ramp()` allocated its `(rows, chans, n_corr)` complex64
output with `np.zeros` and then had `fastnoise.c`'s `phase_ramp_c64()`
write correlations 0 and `n_corr - 1` of it. For the 2-correlation XX/YY
template (round 66) and `--mat-only`'s 1-correlation call, those are every
element, so the zeroing was a full extra write pass over the largest array
in simulate. It now uses `np.empty` when the C ramp runs and `n_corr <= 2`.
4-correlation data (makems' own MS, the first evaluation of a new
observation) and numpy's fallback keep `np.zeros`, since they leave XY and
YX to the zeros.

**Results.** Unchanged. `.probe/r80/ident.py` simulates round 63's 48 real
draws in each arm and hashes every main-table column: all 48 hashes match,
including the 4-correlation makems MSs. `./ri self-check simulate` passes.

**In-run** (`.probe/r80/run.sh`, round 79's harness on the same CPU set:
34 pinned `meqtrees.sif` copies, 4 passes of the 48 draws each, ABBA):

| pair | A (zeros) wall / user ms | B (empty) wall / user ms |
|---|---|---|
| 1 (AB) | 23.65 / 15.13 | 23.24 / 14.75 |
| 2 (BA) | 23.68 / 15.20 | 23.08 / 14.67 |
| 3 (AB) | 23.49 / 15.04 | 23.14 / 14.70 |
| 4 (BA) | 23.51 / 15.07 | 23.09 / 14.70 |
| mean | 23.58 / 15.11 | 23.14 / 14.71 |

Simulate is 0.44 ms (-1.9%) faster in-run, in 4 of 4 pairs, nearly all of
it user time. That is ~0.3% of a ~150 ms WSClean evaluation, too small to
see end to end, so no search ran and there is no `benchmarks.jsonl` row.

**Learned.**
- An `np.zeros` buffer that a C kernel then overwrites whole is a wasted
  pass that matters under L3 pressure (round 76): it costs 0.4 ms in-run
  here. Check what the kernel writes before zeroing.

## Round 81: in-flight chains end once PolyChord has finished

**Finding.** Rounds 39 and 72 put the last ~30 s of an async search, one
rank finishing its chain alone, down to the sampler and left it. PolyChord's
own source says otherwise (`nested_sampling.F90` at the pinned 370f6af).
The root writes every output (chains, `.resume`, stats, posteriors) and
makes its final `dump()` call. Only then does its cleanup loop
`catch_babies()` each worker's chain, and that loop keeps nothing but
`nlike`. Every point slice-sampled after the final write is thrown away, so
the tail was spent computing evaluations that change no output.

**Change.** `PolyChordFinish` (`common.py`), used by both drivers. Before
`run_polychord` every rank `Dup()`s `COMM_WORLD`. The root's dumper sees
the final dump, the only one with no live points, and `isend`s each worker
a message. A worker's likelihood `iprobe`s once a call (microseconds). Once
the message is there it answers without evaluating: low, low, high. A
step-out loop ends on a low, the shrink ends on a high, and every
slice_sample then ends within three calls (`self_check_polychord_finish`
is a port of `chordal_sampling.f90`'s loops). After `run_polychord`,
`close()` waits for the sends, and a worker that never probed (a
synchronous run has no chain in flight) receives its message there.

**In run.** `.probe/r81/ab.sh`: raced 17-rank pairs, shipped SIFs, seed
4242, nlive 150, 15 repeats. A is `a92b658` from a `git archive` tree, B
is this tree:

| pass | first launched | A PolyChord wall | B PolyChord wall | A evals | B evals | A log(Z) | B log(Z) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| async, 600 dead | A | 275.5 s | 237.4 s (**-13.8%**) | 17,050 | 17,196 | 0.09626 +/- 0.00054 | 0.09688 +/- 0.00056 |
| async, 600 dead | B | 242.6 s | 221.0 s (**-8.9%**) | 18,027 | 17,926 | 0.09561 +/- 0.00132 | 0.09649 +/- 0.00088 |
| sync, 300 dead | A | 287.1 s | 286.2 s | 7,821 | 7,821 | 0.09607 +/- 0.00114 | 0.09607 +/- 0.00114 |

The whole command took 361/251 s (A) against 256/231 s (B). A's first run
also spent ~85 s outside PolyChord starting up in its fresh tree, so
compare the PolyChord column. Async log(Z) differs run to run whatever the
code (the arms' log(Z)s overlap within their error bars). By construction
the change cannot alter an async run's outputs, because they are all
written before the dumper fires. In the synchronous pair, where both arms
see the same points, every chain file is byte-identical (`phys_live.txt`
once sorted), with the same evaluations and the same wall time. Workers
answered 3-42 calls each without evaluating, and none printed PolyChord's
"Non deterministic loglikelihood" warning. Mean binary time falls too
(179 to 158 ms in the first pair), because the dropped tail evaluations
were the run's heaviest (round 72: 318 ms). A 12-rank async R2D2 smoke
run (nlive 24, 24 dead) also ended cleanly, with 7 workers answering 3-9
calls each.

**Learned.**
- The tail was not the sampler's. PolyChord discards every chain that
  finishes after its final write, so a whole chain's worth of evaluations
  per in-flight rank (up to 470 in round 72's run) was waste. The saving is
  a fixed ~20-40 s per run segment. That is ~13% of a ~4 minute production
  WSClean search (round 72), and more for R2D2, whose evaluations are ~1 s.
  A resumed run saves it again on every segment.
- `evals_per_s` across the round-81 line is not comparable with earlier
  rows: B runs fewer of the heaviest evaluations, so its ms/eval and its
  rate both move. Compare PolyChord wall at a fixed `NS_MAX_NDEAD`.
- pypolychord's `dumper` runs on the root at every `update_files` and once
  at the very end, with an empty `live` array only on that last call. It is
  the one Python hook between the final write and the cleanup wait.

## Round 82: the phase ramp written with its noise

**Change.** Simulate built DATA in three passes over the array: the C phase
ramp wrote it, then `add_normal_f32_xxyy2()` read and rewrote every real
half, then again every imaginary half (the imaginary draws are the second
half of numpy's stream, so they cannot come first). New
`phase_ramp_c64_xxyy2()` in `fastnoise.c` shares `phase_ramp_c64()`'s body
and adds each element's XX and YY real noise while it is still in L1, so
the predict is written once with its noise. The imaginary pass is
unchanged. `point_source_visibilities()` takes the generator and sigma and
uses the fused kernel for 2-correlation data with a plain PCG64 stream;
anything else (4-correlation makems MSs, numpy fallback, an older
`fastnoise.so`) takes the old ramp-then-noise path. The kernel is checked
against numpy's ramp and numpy's 4-correlation fill once per process
(`_ramp_xxyy2_agrees()`, ~770 rejection-path draws a half) and in
`self_check_fast_noise()`.

**Results.** Unchanged. `.probe/r82/ident.py` (round 80's) hashes every
main-table column of round 63's 48 real draws per arm: all 48 match.
`self_check_fast_noise()` passes against the new library, and a 351k-row
fill is byte-identical to the separate passes for two seeds.

**Solo**, 351k rows x 1 channel, median of 30: 11.76 to 11.50 ms (-2%).

**In-run** (`.probe/r82/run.sh`, round 79's harness: 34 pinned
`meqtrees.sif` copies, 4 passes of the 48 draws each, ABBA; the arms differ
only in their bound `nested_sampling` tree and `fastnoise.so`):

| pair | A (separate) wall / user ms | B (fused) wall / user ms |
|---|---|---|
| 1 (AB) | 23.09 / 14.72 | 22.77 / 14.38 |
| 2 (BA) | 23.12 / 14.71 | 22.71 / 14.31 |
| 3 (AB) | 23.11 / 14.73 | 22.80 / 14.40 |
| 4 (BA) | 23.06 / 14.69 | 22.74 / 14.43 |
| mean | 23.10 / 14.71 | 22.76 / 14.38 |

Simulate is 0.34 ms (-1.5%) faster in-run, in 4 of 4 pairs, all of it user
time, against 0.26 ms solo: the saved pass is worth more under L3 pressure,
as round 76 predicted. That is ~0.2% of a ~150 ms WSClean evaluation, too
small to see end to end, so no search ran and there is no
`benchmarks.jsonl` row. A run needs a `fastnoise.so` built from this
source: `start-sidecars.sh` rebuilds its host copy when `fastnoise.c`
changes, and `meqtrees.sif` gets it at its next rebuild.

**Learned.**
- With round 80's zeroing gone, each remaining DATA-sized pass in simulate
  costs ~0.3-0.4 ms in-run. The imaginary noise pass and `putcol`'s copy
  into casacore's tiles are the two left; the first is fixed by the stream
  order, the second by casacore.

## Round 83 (diagnostic): the binary's profile over a search's mix of evaluations

**Question.** Rounds 69 and 74 profiled the *largest* real-draw MS, where
ducc0's kernels are 54% of the binary. Most evaluations are median-sized or
smaller (round 74's in-run gain was half its solo one). Over a search's own
mix of evaluations, is anything outside ducc0 big enough to patch?

**Run.** Round 63's 48 real draws (`.probe/params15.jsonl`), built through
the warm observation cache (`.probe/r83/gen.py`; XX/YY template MSs, TIME
in the SSM, as a run sees them), imaged once each by a fresh 0001-0017
build (`scripts/patch-wsclean-sif.sh`; its `make install` binary keeps its
symbols) with a search's flags, one pinned CPU, 3 passes under `perf record
-F 2000` (`.probe/r83/all.sh`). 48 images take 6.3 s, ~131 ms each with
process start. Addresses mapped with `nm -C` (`.probe/r83/map.py`).

| where | share of samples | share without process start |
| --- | ---: | ---: |
| `ld.so` (process start; the zygote forks past it in a run) | 14.3% | - |
| ducc0 `x2grid_c_helper<6>` (gridding) | 21.4% | 25% |
| ducc0 `grid2x_c_helper<6>` (degridding) | 13.2% | 15% |
| ducc0 scan, `countRanges`, `Baselines` | 5.0% | 6% |
| WSClean's own code, ~25 functions | 20.6% | 24% (largest: `GridRegularMeasurementSet` 2.0, `ImageWeights::Grid` 1.9, `CacheMeta` 1.8, `NextInputRow` 1.4) |
| libc (memcpy/malloc family) | 7.3% | 8.5% |
| casacore (tables + casa) | 6.1% | 7% (largest: `IPosition` 1.8) |
| FFTW | 4.1% | 4.8% |
| libz (squashfuse paging the SIF) | 2.6% | 3% |

**Learned.**
- Over a search's mix, ducc0 is ~46% of the binary and its two kernels
  40%. Outside them nothing is above ~2.3% of the binary: every remaining
  WSClean target is worth under ~1% of an evaluation, the size of rounds
  59 and 74. The next large WSClean win has to come from the kernels.
- The x2grid inner loop (cmav instantiation, `supp` 6, AVX2) is already
  tight: per visibility, two 8-wide Horner chains for the kernel
  (`eval2`) and 12 FMAs into the local buffer. A cheaper kernel would change
  image bits, and round 27 found AVX-512 changes them too.
- A simulate harness only gets template MSs if each draw's observation is
  first built at the 20-minute horizon (`make_ms` with `observation_minutes
  20`, as `.probe/r82/p.py` does). Calling `simulate()` on a cold cache, even
  twice, gives makems' 4-correlation MSs with TIME in the ISM, which make
  casacore look 2x more expensive in the binary (`ISMBucket::getInterval`
  1.7%).

## Round 84 (diagnostic): icelake against sapphire for a WSClean evaluation

**Question.** Production searches default to `icelake` (`slurm.sh`), and
every round so far was measured there. CSD3 also has `sapphire` (Xeon
Platinum 8480+, 112 cores, 4580MiB each). For this workload, is a sapphire
core faster than an icelake one, and is a half node of each worth more?

**Run.** `.probe/r84/bench.sh`, one script on both partitions: N pinned
copies at once, each imaging round 83's 48 real-draw template MSs three
times with the 0001-0017 build (one `wsclean` process per image, so
process start is included in both arms), then N pinned copies of round 82's
simulate harness (`meqtrees.sif` with this tree's `fastnoise.so`, 10 passes
of the 48 draws). Icelake ran in this job (cpu-q-576, 38 of its 76 CPUs,
other half someone else's). Sapphire ran as one 56-CPU job
(`.probe/r84/job.sh`, cpu-r-86), also half a node.

| arm | `wsclean` wall / user / sys ms | simulate wall ms | evaluation ms | evals/s for the allocation |
| --- | ---: | ---: | ---: | ---: |
| icelake, 1 copy | 113.2 / 89.5 / 16.7 | 15.3 | 128.5 | |
| sapphire, 1 copy | 98.2 / 73.6 / 18.9 | 12.9 | 111.1 | |
| icelake, 36 of 38 | 144.3 / 109.6 / 27.7 | 20.7 | 165.0 | 218 |
| sapphire, 36 of 56 | 128.0 / 92.8 / 28.2 | 18.1 | 146.1 | 246 |
| sapphire, 54 of 56 | 154.2 / 107.0 / 39.2 | 22.0 | 176.1 | **307** |

**Learned.**
- A sapphire core is 13-14% faster solo (user time -18% for `wsclean`), but
  at the same ~95% occupancy it slows more under load (+59% against +28%;
  kernel time doubles), so a fully loaded sapphire core does 5.7 evals/s
  against icelake's 6.1 (-6%).
- What a search buys is its allocation. Half a sapphire node runs 54 ranks
  at 307 evals/s against half an icelake node's 36 at 218 (+41%), and a
  whole node 112 against 76 cores. Per allocated core (56 vs 38) it is
  5.5 against 5.7 evals/s, so per core-hour icelake stays ~5% ahead; the
  public policy page gives no per-partition rate (`csd3.cam.ac.uk/charges`
  needs a Raven login), so whether sapphire is cheaper per evaluation
  depends on the project's charge. For a fixed wall-clock deadline,
  `SBATCH_PARTITION=sapphire` with ~54 ranks per half node is the faster
  run.
- The default stays `icelake`. Every per-evaluation round was measured
  there, and nothing in the code depends on the partition: WSClean's
  `target_clones` and `fastnoise.so` already pick their AVX2 paths on both.

## Round 85 (diagnostic): where a zygote request's kernel time goes

**Question.** Round 84 left ~17ms (solo) to ~28ms (36-way) of kernel time
per `wsclean` process. A run does not start a process per evaluation, it
forks one from `wsclean-zygote`, so how much kernel time and how many page
faults does a *zygote* request still pay, and what are they?

**Method.** `.probe/r85/zy.py` drives one pinned `wsclean-zygote` (round
83's 0001-0017 `D.sif`, production malloc tunables) through round 83's 48
real-draw template MSs, one warm-up pass plus N timed passes, and reads the
zygote's `cutime`/`cstime`/`cminflt` (its reaped children). `perf record -e
page-faults -c 1 -d` over a solo run, with each fault address classified
against the zygote parent's `/proc/PID/maps` (`cls.py`), splits the faults.

| arm | request ms | user ms | sys ms | minor faults |
| --- | ---: | ---: | ---: | ---: |
| 1 copy | 77.8 | 69 | 8.2 (11%) | 1.26k |
| 36 copies, one per CPU | 101.5 | 86.9 | 13.7 (13.5%) | 1.40k |

Faults per request, solo, by where they land (1.6k counted by perf):

| mapping | faults |
| --- | ---: |
| parent's `[heap]` (copy-on-write or untouched parent pages) | 683 |
| mappings the child creates (heap growth past the parent's top, reordered-file and model mmaps) | 449 |
| file-backed text and rodata (`r-xp` 247, `r--p` 150) | 397 |
| file-backed `rw-p` (GOT/data) and stack | 60 |

**Learned.**
- A forked request pays about half the kernel time of a fresh process
  (8.2 against 16.7ms solo, 13.7 against 27.7ms at 36-way), as expected.
  Kernel time is still ~13% of a loaded request, but page faults are only
  a part of it: ~1.4k faults at a few microseconds each is ~3-5ms, so the
  other ~9ms is syscalls, fork and exit. (Wrong, as round 87 found:
  at 36-way a fault costs ~6.6us, so faults are ~9.5ms of it.)
- ~400 faults per request (a quarter) re-map library text and rodata the
  parent already has mapped. Linux's `fork` does not copy page-table
  entries of file-backed mappings that have no anonymous pages, so every
  child faults them in again (16 pages per fault with fault-around). A
  workaround (giving each text mapping an anon page so `fork` copies its
  entries, plus pre-touching the code the parent never runs) could save at
  most ~400 faults, ~1ms of a 100ms request at 36-way, and adds per-fork
  page-table copying and refcount traffic on shared pages. Not attempted.
- The heap and new-mapping faults (~1.1k, ~4.5MB) are first touches of
  per-evaluation buffers, which round 41 already found a fork server
  cannot avoid.
- The syscall side cannot be counted on this host: `strace -f` (with or
  without `--seccomp-bpf`) hangs inside apptainer's starter before
  `wsclean` runs, and attaching `strace -f -p` to a running zygote
  traced only ~300 calls in 15s while requests ran normally (it reported
  the process switching between x32 and 64-bit mode). `perf_event_paranoid`
  is 2, so kernel profiling is out too. Splitting the remaining ~9ms needs
  an `LD_PRELOAD` syscall counter or a counter built into the zygote.
- `taskset -c N` must name a CPU in this job's cpuset
  (`grep Cpus_allowed_list /proc/self/status`); odd CPUs below 22 are not
  in it, and a wrong one makes `wsclean-zygote` exit with no reply.

## Round 86 (diagnostic, rejected): where the R2D2 U-Net's time goes on torch 2.13

**Question.** Round 11 put the U-Net (85% of an R2D2 request at 1 thread)
at its memory floor and blamed oneDNN's per-call weight reorders. Nothing
has touched R2D2 since round 21. With `r2d2.sif`'s torch 2.13, is any of
the forward a lever that keeps the output bit-identical?

**Profile** (`torch.profiler`, one checkpoint, 32x32, 1 thread, 50
forwards, `.probe/r86/pp.py`). The forward is not mostly oneDNN:

| op | share | calls / forward |
| --- | ---: | ---: |
| `_slow_conv2d_forward` (im2col + MKL `sgemm`; every conv at <= 8x8, and the 2-channel input conv) | 45.5% | 12 |
| `mkldnn_convolution` (the convs at 16x16 and 32x32) | 30.3% | 7 |
| `slow_conv_transpose2d` | 9.4% | 3 |
| instance norms (`native_batch_norm`) | ~10% | 22 |

The largest single op is the 1024 -> 1024 3x3 conv at 2x2: 3.5-3.9ms for
37.7MB of weights, ~10GB/s from one core, a weight stream through `sgemm`
with N = 4.

**Tried** (all in `.probe/r86/`, `r2d2.sif` unmodified):

| lever | solo | 36 copies | output |
| --- | ---: | ---: | --- |
| `torch.jit.trace` / `+ jit.freeze` | 29.0 / 29.0 ms against 28.2 | | identical |
| `jit.optimize_for_inference` (oneDNN prepack) | 55.0 ms | | 6.3e-6 off |
| MKL-packed weights (`mkl._mkl_reorder_linear_weight` + `_mkl_linear`) per slow conv | 1024->1024 at 2x2: 2.35 against 3.53ms; 512->1024 at 2x2: 1.18 against 1.76 | | differs where faster; identical only where slower (1024->512 at 4x4: 2.35 against 1.82) |
| `MKL_ENABLE_INSTRUCTIONS=AVX2` (MKL picks AVX-512 on icelake) | 26.4 against 27.2-29.0 ms (-3 to -9%) | 47.1 against 48.2-48.3 ms (-2.3%, 2 ABAB pairs) | 6.8e-6 of the output peak |

**Learned.**
- MKL's AVX-512 `sgemm` is ~1.5x slower than its AVX2 path on the deep
  1024-channel convs (3.50 against 2.35ms at 2x2), but under 36 copies the
  whole forward gains only 2.3%, since it is then bound on memory (48ms
  against 27ms solo, round 11's picture). Not worth changing every R2D2
  objective by ~7e-6.
- Freezing or tracing does nothing for a CPU forward: torch 2.13's frozen
  graph still calls the same `aten` convs, and the only prepacking pass
  (`optimize_for_inference`) is round 11's slower, non-identical oneDNN path.
- Any lever that is faster here changes the `sgemm` reduction order, so the
  R2D2 CPU forward is closed to bit-identical changes on this torch build.
  The open R2D2 avenues stay the ones round 11 named: batching forwards
  across ranks, or merging GPU contexts.

## Round 87 (diagnostic, two rejections): a zygote request's kernel time is page faults

**Question.** Round 85 left ~9ms of a loaded request's ~14ms kernel time
unexplained and guessed syscalls, fork and exit. `strace` cannot see
WSClean here, so what is it, and can any of it be cut?

**Method.** `.probe/r87/sc.c` is an `LD_PRELOAD` library passed to
`wsclean-zygote` through round 85's `zy.py` (`PRE=`). It times every
interposable file and memory call (open/read/lseek/stat/fopen/mmap/... ,
wall clock) and `fork` in the parent. At `_exit` it writes each child's
counts with its own `getrusage`, and the zygote's `wait4` writes the reaped
rusage, so exit teardown is the reaped `stime` minus the child's own.
Round 83's 0001-0017 `D.sif`, its 48 real-draw MSs, production malloc
tunables, 2 timed passes (96 requests per driver), `.probe/r87/agg.py`
sums them up.

| per request | 1 copy | 36 copies, one per CPU |
| --- | ---: | ---: |
| request | 76.7-77.8 ms | 101.9 ms |
| user | 68.8 ms | 88.5 ms |
| sys, up to `_exit` | 5.9-6.7 ms | 11.6 ms |
| sys, exit teardown (reaped - own) | 1.2 ms | 2.2 ms |
| minor faults | 1.26-1.61k | 1.43k |
| wrapped calls, wall | 1.65 ms | 2.09 ms |
| of which `read` (548 calls) | 1.02 ms | 1.25 ms |
| `lstat64` (210), `open64` (102), `fopen64` (44), `lseek64` (613) | 0.45 ms | 0.58 ms |
| `fork` in the parent | 0.23 ms | - |

So at 36-way ~9.5ms of the 13.8ms is page faults: ~1.4k faults at
~6.6us each, not round 85's few us. The syscalls a request makes cost ~2ms
(mostly `read`'s copy out of tmpfs), teardown ~2ms and `fork` ~0.2ms.

**Rejected: THP off** (`prctl(PR_SET_THP_DISABLE)` in the zygote, inherited
by every child). Faults go from ~1.3k to 5.28k a request, sys from 6.1 to
7.7-9.2ms, teardown from 1.2 to 1.7ms, and the request from 76.7-77.8 to
79.1-81.1ms solo (3 ABAB runs, +3%). CSD3's `always` THP is already saving
the heap ~4k small faults.

**Rejected: populate library text in the child.** Round 85 counted ~400
faults per request that re-map library text the parent already has.
`mlock` (populate) of each file-backed `r-xp` mapping right after `fork`,
with or without `munlock`, takes a request to 85.1-85.5ms (+11%), and all
`r-xp`/`r--p` mappings to 90.2-90.8ms (+17%), 2 runs each, interleaved.
Populating maps every text page; fault-around (16 pages a fault) maps only
what runs.

**Learned.**
- A loaded zygote request's kernel time is ~70% page faults (~9.5 of
  13.8ms), ~15% syscalls and ~15% exit. `fork` itself is 0.23ms. Round 85's
  "syscalls, fork and exit" guess was wrong.
- The faults left are first touches of per-request memory, the heap
  (~700) and the child's new mappings (~450). Fewer faults means touching
  less memory per request, not a kernel or THP setting: both levers that
  change how pages are mapped made it slower.
- At 36-way a fault costs ~6.6us against ~3.5us solo, so the fault cost,
  like simulate's (round 76), grows with the node's load.
- An `LD_PRELOAD` library is a working substitute for `strace` here:
  `apptainer exec ... env LD_PRELOAD=/dev/shm/...so` reaches the zygote
  and its forks. Wrap `_exit` to dump a forked child's counters, since
  `_exit` skips destructors, and don't wrap `clock_gettime` or `getenv`,
  which the wrapper itself calls.

## Round 88 (diagnostic, two rejections): the zygote parent's heap reshaped before `fork`

**Question.** Round 87 put ~700 of a request's ~1.4k faults in the parent's
`[heap]` range. Are those copy-on-write faults on heap the parent has
touched, and can reshaping that heap before `fork` turn them into fewer,
cheaper faults?

**What the heap looks like.** After three requests, the parent
`wsclean-zygote` under the production malloc tunables (`.probe/r88/peek.py`)
has a 65.5MB `[heap]` VMA (`top_pad` is 64MB), of which 3.8MB is resident:
one 2MB huge page and ~1.8MB of 4kB pages. The small pages sit in the head
below the heap's first 2MB boundary, because the program break starts
unaligned. CSD3's THP is `always` with `defrag=madvise`, and
`thp_fault_fallback` is ~2% of `thp_fault_alloc`, so the untouched 60MB
already gets huge pages on first touch.

**Rejected: `malloc_trim(0)` before each `fork`** (`.probe/r88/trim.c`, an
`LD_PRELOAD` wrapper of `fork` that trims once or on every call). The idea
was to give free heap pages back so the child zero-fills them instead of
copying them. Round 83's `D.sif` (0001-0017) was used with its 48 real
draws, pinned solo on CPU 24, 2 timed passes per run, interleaved:

| arm | run 1 | run 2 |
| --- | ---: | ---: |
| baseline | 77.24 ms | 76.78 ms |
| trim once | 77.08 | 76.69 |
| trim every `fork` | 77.02 | 77.23 |

**Rejected: a 2MB-aligned heap start** (`.probe/r88/align.c`, a
constructor that moves the break to the next 2MB boundary before glibc's
arena takes it). The resident small pages fell from ~1.8MB to ~1.2MB, but
requests did not get faster:

| arm | run 1 | run 2 | run 3 |
| --- | ---: | ---: | ---: |
| baseline | 77.48 ms | 77.15 | 77.27 |
| aligned heap | 77.31 | 77.01 | 77.40 |

Both levers are inside the ~0.5ms run-to-run spread. The zygote's
`cminflt`, i.e. the faults of all its children, varied by ±10% between
baseline runs alone, so it cannot resolve a fault-count change this small
either.

**Learned.**
- The parent's heap is small: ~3.5MB resident. Even if every 4kB page in
  it were copied once per request (~450-900 faults), the reshaping levers
  can only move the part THP could cover, and that did not show up in
  request time.
- The heap faults round 87 counted are first touches the child needs
  anyway. With rounds 85, 87 and 88 together, the per-request kernel-time
  avenues are closed: page-mapping settings (THP off, populate, trim,
  alignment) do not help, and a request's ~9.5ms of faults at 36-way can
  only come down by WSClean touching less memory.

## Round 89 (diagnostic): rounds 72-88 together in a production-shaped search

**Question.** Round 72 measured a production-shaped search on a 0001-0016
SIF: 174 evals/s whole run, 233 steady, and a ~30 s single-rank tail.
Since then patch 0017 (round 74), five simulate rounds (78-82) and the
tail cut (round 81) have each been measured alone. What do they give
together, and how much of it does a cluster run on the shipped SIF see?

**Run.** Round 72's shape: 36 ranks, async, nlive 150, 15 repeats, 900
dead, seed 4242, commit b63c209, cpu-q-576 (the other half of the node
busy, load ~45 at the end). Two searches run back to back, not raced: first
the shipped `wsclean.sif` (0001-0006), then a fresh 0001-0017 build
(`scripts/patch-wsclean-sif.sh /dev/shm/r89/wsclean.sif`, 2 minutes with
36 jobs). Both use the tree's simulate. Rows `r89-stock` and `r89-prof`.

| run | WSClean SIF | evals | wall | evals/s whole run | steady (10 s to end-10 s) | simulate | `wsclean` binary | log(Z) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r72-prof (round 72) | 0001-0016 | 26,899 | 154 s | 174.3 | 233 | 28.2 ms | 122.5 ms | |
| r89-stock | shipped 0001-0006 | 28,514 | 158 s | 180.0 | 184 | 23.9 ms | 162.0 ms | 0.09620 ± 0.00098 |
| r89-prof | 0001-0017 | 28,599 | 114 s | **250.5** | **258** | 23.4 ms | 108.6 ms | 0.09642 ± 0.00102 |

Workers are 97-98.5% busy in every 5.7 s window from 6 s to 108 s. The
last 5% of evaluations end within 6.5 s, against round 72's ~28 s on one
rank. Both log(Z)s agree within their error bars. The async runs evaluate
different points, so they are not identical.

**Learned.**
- Rounds 72-88 together take a production-shaped search from 174 to 250
  evals/s whole run (+44%) and from 233 to 258 steady (+11%). Most of the
  whole-run gain is round 81's tail cut, so a short benchmark's whole-run
  rate is now close to its steady rate (250 against 258, where round 72
  had 174 against 233).
- MeqTrees (simulate) is 17.5% of evaluation time (23.4 of 133 ms), about
  the same share as round 72's 18%. Simulate got 17% cheaper, but the
  binary got cheaper too. The binary is 82%, and ducc0's kernels are most
  of that (round 83).
- A cluster search on the shipped `images/wsclean.sif` runs at 184 evals/s
  steady against 258 on a patched build: 29% fewer evaluations an hour.
  This is still the largest single lever open to production.
  `./ri images patch-wsclean` in a compute-node job (2 minutes) closes it,
  and the run's stale-SIF warning flags it. The shipped SIF was left as it
  is here, because other worktrees use it.
- `WSCLEAN_SIF=...` in the environment is ignored: `defaults.sh` sets it
  from `SIF_DIR` unconditionally. To run a search on a different build
  without replacing the shared one, point `SIF_DIR` at a directory of
  symlinks with the new `wsclean.sif` in it. The row's
  `NS_IMAGER_IMAGE_ID` then gains a `+p<patch hash>` suffix, and its
  absence is how to spot a run that silently used the stock SIF.
- Each 36-rank, 900-dead run leaves ~58k files (106 MB `summary.json`), so
  delete it once its row is in `benchmarks.jsonl`.

## Round 90: the FIELD, OBSERVATION and SPECTRAL_WINDOW values spliced into the copy

**Why.** A cProfile of simulate on the current tree (round 63's 48 real
draws, one pinned CPU, 14.0 ms a call) put ~2 ms in casacore table
opens and closes, four a call. Three of them were `extend_template()`'s
one-row subtables: FIELD (four direction/time columns), OBSERVATION
(`TIME_RANGE`) and SPECTRAL_WINDOW (`patch_spectral_window()`), opened
writable to store ~10 float64 values. Alone, those three blocks timed
at 0.97 ms a call.

**Change.** `_splice_plan()` runs once per template per worker (~10 ms):
it copies the three subtables twice into scratch, writes a set of
unique random values into one copy and a second set into the other with
casacore, finds each value's bytes in the first copy's files (exactly
one hit, either byte order), and keeps the plan only if splicing the
second set into the first copy's bytes gives the second copy's files
byte for byte (all but `table.lock`). The first copy's bytes become the
base: casacore also rewrites a few header bytes in `table.f0` and
`table.f0i` on any write, so the template's own bytes are not the
right base. `_copy_template()` then writes those subtables from the
base with the evaluation's values spliced in. A template whose plan
fails, or a value whose shape is not the plan's, takes the old casacore
writes (`_put_subtables()`). `patch_spectral_window()`'s values moved to
`spectral_window_values()`, which both paths use.

**Results.** Unchanged. `.probe/r90/vals.py` hashes the row count,
keywords and every column of the main table and every subtable for each
of the 48 draws: all 48 match HEAD. Byte for byte, the three spliced
subtables match HEAD's in all 48. The files that differ (main `table.f0i`,
`POLARIZATION/table.f0i`, `simulation.json`) differ just as much
between two HEAD runs. `./ri self-check simulate` passes; its
observation-prefix check compares every subtable column with a makems
build, and now also asserts that every template it served got a splice
plan, so a silent fallback fails it. With the plan forced to None, the
fallback path's 48 MSs are column-identical to HEAD as well.

**Solo**, one pinned CPU, 48 draws x 5 reps, ABBAAB:

| arm | ms a call |
|---|---|
| A (casacore writes) | 14.04, 14.02, 13.99 |
| B (spliced) | 13.16, 13.12, 13.08 |

-0.89 ms (-6.4%).

**In-run** (`.probe/r90/run.sh`, round 82's harness: 34 pinned
`meqtrees.sif` copies, 4 passes of the 48 draws, ABBA, same
`fastnoise.so` for both arms):

| pair | A wall / user / sys ms | B wall / user / sys ms |
|---|---|---|
| 1 (AB) | 23.26 / 14.66 / 4.41 | 22.30 / 13.84 / 4.26 |
| 2 (BA) | 23.14 / 14.57 / 4.39 | 22.13 / 13.77 / 4.19 |
| 3 (AB) | 23.09 / 14.58 / 4.32 | 22.21 / 13.87 / 4.16 |
| 4 (BA) | 23.33 / 14.72 / 4.37 | 22.26 / 13.88 / 4.21 |
| mean | 23.21 / 14.63 / 4.37 | 22.23 / 13.84 / 4.21 |

Simulate is 0.98 ms (-4.2%) faster in-run, in 4 of 4 pairs: ~0.7% of a
~133 ms production evaluation (round 89), below what a search can
resolve end to end, so no search ran and there is no `benchmarks.jsonl`
row. The change is in the bound `nested_sampling` tree, so the next
cluster run gets it without a rebuild.

**Learned.**
- A casacore writable open-putcol-close of a one-row subtable costs
  ~0.3 ms, nearly all of it fixed cost, and that is the same order as one
  full DATA-sized pass in-run (round 82). Small table writes are worth
  as much as array passes.
- casacore rewrites a few header bytes of `table.f0`/`table.f0i` the
  first time a copy is written, and the value bytes alone after that.
  So a byte-level shortcut must start from a casacore-written copy, not
  from the template.
- A profile without the `fastnoise.so` bind falls back to numpy noise
  (36.6 ms a call, `Generator.normal` 18 ms). `meqtrees.sif` has no
  `/opt/ri-lib/fastnoise.so`, and only `start-sidecars.sh` supplies one,
  so a probe has to bind a host build (`gcc -O2 -ffp-contract=off
  -shared -fPIC docker/meqtrees/src/fastnoise.c -lm`).
- What is left of simulate solo: the C ramp and noise ~6 ms, the main
  table's 19 putcols ~2.5 ms, its one open/close ~0.5 ms, and
  `extend_template()`'s own numpy (uvw, tiles) ~0.7 ms.

## Round 91 (diagnostic): the file quota caps a run's size; simulate's table writes re-timed

**Question.** With rounds 72-90 a production-shaped WSClean search runs at
~250 evals/s. Round 89 noted each 900-dead run leaves ~58k files. What does
that make the largest run the 1M-file `hpc-work` quota can finish, and is
anything left in simulate's casacore writes after round 90?

**Files.** Counted with `lfs find` (57 s, no Slurm) in the last converged
production run, `r2d2-vlaa-20260922T174616Z` (24 ranks, nlive 150, 15
repeats, run to convergence): 149,976 files for 71,395 evaluations, 2.10 an
evaluation (74,713 directories). Its `.stats`: 1,373 dead points (9.2 x
nlive) and 69,225 likelihood calls (3.4 x num_repeats per dead point).
Round 89's 36-rank WSClean runs give the same ~2 files an evaluation. The
project quota (`lfs quota -p 110845`) had 398,660 of 1,048,576 used.

| run | evaluations | files | of an empty quota | of today's ~650k free |
| --- | ---: | ---: | ---: | ---: |
| nlive 150, converged | ~71k | ~150k | 14% | 23% |
| nlive 500 (`docs/cluster.md`'s example) | ~240k | ~490k | 47% | 75% |
| nlive 1000 | ~470k | ~970k | 93% | does not fit |

At 250 evals/s, 650k files is ~22 minutes of WSClean evaluations. A run
that hits the quota dies on `EDQUOT` with every rank failing `mkdir`
(round 16), and that is lost work no speed-up recovers.

**Change.** `ns_warn_file_quota()` (`scripts/lib/run-config.sh`): when a
search starts, both run scripts estimate its files as 7 x num_repeats x
dead points (`--max-ndead`, or 9 x nlive when it is -1) and warn if that is
more than the run directory's Lustre project has left (`lfs project -d`,
`lfs quota -p`: ~12 ms, no Slurm call). Off Lustre, or with no project
quota, it is silent. It warns and never refuses, because the estimate is
rough. The run-config self-check covers fit, overflow, a `--max-ndead`
bound, no project and no `lfs`.

**Simulate's main-table writes.** Round 90 left ~2.5 ms of putcols. Timed
per column (`.probe/r91/cols.py`, 48 real draws x 3, one pinned CPU, 12.9
ms a call): DATA 0.76, UVW 0.51, ANTENNA1 0.28, ANTENNA2 0.23, TIME 0.22,
TIME_CENTROID 0.22, close 0.32, addrows 0.29, INTERVAL 0.07 ms. Setting
`setmaxcachesize` on the StandardStMan to 64MB before the writes changed
nothing (12.93/12.92 against 12.99/12.98 ms, per-column times equal to 0.01
ms), so the 32KB buckets of round 33 are not cache-bound. The four scalar
columns are ~11 ns a row, about casacore's per-row put cost.

**Learned.**
- At ~2 files an evaluation the quota, not CPU time, bounds a run's size:
  about 970 files per unit of nlive for a converged 15-repeat run. Getting
  past it needs fewer inodes per evaluation (round 53's idea: no eval
  directory for the ~80% of records that hold only `metrics.json`, or
  per-rank record files), which every record reader would have to follow.
- `lfs project -d <dir>` and `lfs quota -p <id> <dir>` read the quota of
  the project that holds a directory in ~12 ms. `quota -w` lists it under
  `/rds-d6/user/<user>`, which does not prefix-match the `/rds/user/...`
  paths runs use.
- Simulate's casacore side is now near its floor: what is left is the C
  ramp and noise (~6 ms solo) and ~3.5 ms of table opens, puts and closes
  that no bucket setting moves.

## Round 92: one record file per evaluation, not a dir plus `metrics.json`

**Question.** Round 91 found the 1M-file quota, not CPU time, caps a run:
~2.1 files an evaluation, nearly all an `eval-*` directory holding nothing
but its `metrics.json`. Can a record be one inode?

**Change.** `write_evaluation_record()` writes `evaluations/eval-NNNN-key.json`
(atomic, through `.json.partial` and a rename). The `eval-NNNN-key/` dir now
exists only for an evaluation that keeps something (extremes, the 1 in 100
kept whole, failures), and `_strip_to_record()` removes it rather than
leaving it empty. Record writes no longer `mkdir` in `evaluations/` (round
54's shared directory lock). Every reader takes both layouts, so runs from
before resume and report as they did: `evaluation_record_paths()` (one
listing) for `common.py`'s record readers, plus `./ri runs`, `./ri health`
(in flight, disk per evaluation, wedge logs), the progress bar, `./ri
resume`'s count and `test_self_heal.sh`. Resume adoption sweeps a torn
`.json.partial` and an unreadable flat record, as it removed a torn record's
dir before. `ns_warn_file_quota` now estimates 4 files per repeat per dead
point, not 7.

**Measured.** Raced 17-rank WSClean arms, nlive 150, shipped SIFs, A = HEAD
(`dc27cd5`) from a `git archive` tree on Lustre, B = this tree
(`.probe/r92/ab.sh`):

| pass | arm | evals | files under `evaluations/` | files/eval | evals/s | log(Z) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sync, 300 dead | A | 7,821 | 16,225 | 2.07 | 27.15 | 0.09607 +/- 0.00114 |
| sync, 300 dead | B | 7,821 | 8,562 | 1.09 | 27.20 | 0.09607 +/- 0.00114 |
| async, 600 dead, A first | A | 17,641 | 36,099 | 2.05 | 76.47 | 0.09602 +/- 0.00111 |
| async, 600 dead, A first | B | 17,446 | 18,432 | 1.06 | 79.54 | 0.09682 +/- 0.00055 |
| async, 600 dead, B first | A | 16,727 | 34,256 | 2.05 | 89.39 | 0.09661 +/- 0.00055 |
| async, 600 dead, B first | B | 17,722 | 18,726 | 1.06 | 79.15 | 0.09578 +/- 0.00123 |

The synchronous seeded pass is identical: same evaluations, same log(Z).
Files per evaluation fall from ~2.06 to ~1.07 in every pass. Throughput does
not move visibly: the async pairs split +4% and -11%, the later-started arm
was faster both times, and A-b's mean binary time (153 ms against 173-181)
shows its evaluation mix was cheaper, not its records. The `harness` stage
is 0.26-0.30 ms in B against 0.29-0.32 ms in A. Post-run phases are equal.

**What it buys.** At round 91's 3.4 x num_repeats evaluations per dead point
and ~9 x nlive dead points, a converged 15-repeat run now needs ~510 files
per unit of nlive instead of ~970. The ~650k files free on the project fit
nlive ~1,250 instead of ~650, and 250 evals/s runs ~40 minutes before
filling them instead of ~22.

**Learned.**
- Raced searches from a compute node each call `squeue` once per named
  `--output-dir` (`ns_refuse_live_run` -> `_ns_slurm_job_live`), plus
  `rank-budget.sh`'s liveness check of other runs. It took ~1-3 minutes
  here and delayed arm B's start, so start-time offsets in raced A/Bs can be
  minutes, not the scripted 10 s. `resume-nested-sampling-run.sh
  --self-check` also calls the real `squeue` and blocks on it, so it is not
  safe to run inside the agent's job.
- Async arms that differ only in record layout still differ by up to 11% in
  evals/s, driven by the evaluation mix: compare their `image_binary` ms
  before reading a throughput gap as a code effect.

## Round 93: records packed into one file when a run finishes or resumes

**Question.** Round 92 left ~1.07 files an evaluation, nearly all its
`eval-*.json` record. They pile up across runs (the finished production
R2D2 run alone holds ~148k files) and across the segments of a resumed run.
Can records stop costing an inode each once nobody writes them?

**Change.** `pack_evaluation_records()` appends records to
`evaluations/records.jsonl`, one `name<TAB>record json` line each, fsyncs,
then unlinks their files (a flat record, or a pre-round-92 `metrics.json`
and then its dir if that held nothing else). It runs at the two moments no
rank writes records:
- at resume, inside `adopt_completed_evaluations()`, on rank 0 after a
  barrier, from the records every rank just read (no rank may still be
  reading a flat record: a missing one reads as torn, and its dir is
  swept);
- at finish, on rank 0 after `summary.json` is written, from the records
  `finished_run_records()` gathered and `prune_run_artefacts()` edited
  (`pack_finished_run()`, post-run phase `pack`).

`scripts/prune-run-artefacts.py` does the same for runs that finished
before this, from their `summary.json`. Readers: `common.py`'s record readers
(a flat record beats a packed one; the last line for a name wins; a line
without its newline is a torn append and is ignored), `./ri runs`, `./ri
health` (a packed record is timed by its `timing.ended_epoch`, and the dir
of a packed evaluation is not counted in flight), the progress bar's counts,
`./ri resume`'s count and `test_self_heal.sh`. Records keep the default
`json.dumps` separators so health's `"objective": 100.0` marker still
matches.

**Measured.**

| case | before | after |
| --- | ---: | ---: |
| 8-rank WSClean search, nlive 25, a rank `kill -9`ed at 900 evals, finished at 1,337: files under `evaluations/` | ~1,430 (1.07/eval) | 117 (22 kept dirs, their files, the pack) |
| 20k synthetic 4.2KB records on Lustre, read back (warm) | 2.2 s flat | 0.94 s packed |
| pack 20k records (unpinned) | - | 3.8 s |
| pack 5k records from a process pinned to one CPU | - | 31.3 s, then 0.89 s with the unlink threads unpinned |
| in-run resume pack, 8 ranks | 900 records in 5.6 s (pinned) | 1,023 in 0.3 s (unpinned) |
| `_ns_count_evals` / `_ns_completed_evals`, 20k records | 0.07 / 0.06 s | 0.18 / 0.18 s (80MB pack) |

Every eval_id, params and objective read back from the packed run equals
`summary.json`'s (1,337 of 1,337), and `./ri health` and `./ri runs` read it
as before (0 in flight, 1,337 evaluations). Records from an earlier segment
keep the `paths` they were written with, as their files did: `summary.json`
remains the authority for what a finished run kept. Evaluation throughput
is untouched: no record write changed. A run's own live segment still costs
~1.07 files an evaluation, so `ns_warn_file_quota` keeps that estimate, but
a run that resumes every job (CSD3's 12 h cap) now holds only its current
segment's records as files, and a finished run gives them all back.

**Learned.**
- Unlinks from threads of a process pinned to one CPU cost ~6 ms each on
  Lustre against ~0.2 ms unpinned: the same directory-lock stall round 54
  found for `mkdir`. Every thread pool rank 0 runs over `evaluations/` must
  run off the pinned CPU (`initializer=_unpin_thread`).
- `./ri self-check self-heal`'s mid-run kill can be reproduced cheaply by
  `kill -9` of one `polychord_wsclean.py` rank from a short 8-rank search
  (`--retries 2`): `run_with_retries` resumes it within seconds.

## Round 94: a search's job patches a stale `wsclean.sif` itself

**Question.** Round 89 found the largest lever left open to production
outside the code: the shared `images/wsclean.sif` stops at patch 0006, and a
cluster search on it runs 29% fewer evaluations an hour than on a
0001-0017 build. Runs only warned (round 70), and the SIF was left alone
because other worktrees share it. Can every search get the patches without
anyone replacing that file?

**Change.** `ns_use_patched_wsclean_sif` (`defaults.sh`), called by
`run-nested-sampling.sh` once it is inside its Slurm job (so also by `./ri
resume` and `./ri bench run`). When `WSCLEAN_SIF` lags the tree's patches
(`ns_wsclean_sif_current`, the old warning's test), it builds
`$RI_WORK_DIR/.cache/ri/wsclean-<base id>-p<patches hash>.sif` with
`patch-wsclean-sif.sh` and the job's CPUs, renames it into place, and points
the run at it. Later runs, on any node, find it and pay nothing. A failed
build warns and keeps the stock SIF; `NS_PATCH_WSCLEAN=0` skips it (for an
A/B that means to run a stale SIF). The shared SIF is never written. The
login node only warns, and says the job will patch. There is no lock: two
jobs racing build the same bytes and the rename is atomic.

`patch-wsclean-sif.sh` failed inside a job at first: `job-env.sh` puts the
container home (and the cwd) under `/rds`, and neither a `--writable`
sandbox nor the layer's `%post` can create `/rds` to mount it on
("destination /rds doesn't exist in container"). It now runs from its
`/tmp` work dir with `HOME` there and `APPTAINER_HOME` unset, and resolves
its paths first.

**Measured.** 16 ranks, nlive 50, 5 repeats, 150 dead, seed 4242,
synchronous, cpu-q-576 (this job's 38 CPUs), shipped SIFs; runs in this
order:

| run | WSClean SIF | wall | evals/s | `wsclean` binary | simulate | chains |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| r94-stock-1 | shipped (0001-0006) | 79.9 s | 20.14 | 179.2 ms | 27.3 ms | `ccfd575b` |
| r94-stock-buildfail | shipped (the first, failed build) | 79.1 s | 20.35 | 179.0 ms | 25.4 ms | `ccfd575b` |
| r94-auto-build | built this run, 0001-0017 | 52.5 s (+1m54 build) | 30.63 | 111.8 ms | 25.1 ms | `ccfd575b` |
| r94-auto-cached | cached | 52.6 s | 30.56 | 112.3 ms | 25.1 ms | `ccfd575b` |
| r94-stock-2 | shipped, `NS_PATCH_WSCLEAN=0` | 78.5 s | 20.49 | 177.6 ms | 25.0 ms | `ccfd575b` |

All five chains files (and `_dead-birth.txt`) are byte-identical, and log(Z)
is 0.09600 ± 0.00170 in each. The patched runs are +50% evals/s; the
benchmark rows carry `+p800c8ece3777` on the imager id, so bench keeps them
apart. A 12-hour job pays the ~2 minute build once at most (0.3%), and only
the first job after a patch change.

**Learned.**
- Anything that runs `apptainer --fakeroot --writable` or a `%post` from
  inside a run's job must keep both its cwd and `HOME`/`APPTAINER_HOME` off
  `/rds`; `job-env.sh` sets the home to `$RI_WORK_DIR/.ri-job/home`.
- At 16 ranks the shipped SIF costs a third of a search's throughput
  (20.3 against 30.6 evals/s), more than round 89's 29% at 36 ranks, since
  the binary is a larger share of a less loaded evaluation.


## Round 95: kept evaluations published by the record writer

**Question.** With round 94's patched SIF, a production-shaped search
(36 ranks, async, nlive 150, 15 repeats, 900 dead, seed 4242) runs at
~270 evals/s and `./ri profile` puts 5.2% of worker time in "idle". Some
of that is start-up and the tail. What is the rest?

**Method.** Each record carries `timing.started_epoch` and `ended_epoch`,
and `eval_id` counts up per rank. Matching each rank's evaluation k to
the nearest unclaimed k+1 start after its end gives the gap between one
evaluation and the next on a worker (`.probe/r95/gaps2.py`). The sum of
the gaps does not depend on the matching, only their distribution does.

**Found.** The median gap is 1.6 ms (PolyChord, the prior and the
harness), but ~280 gaps per run are 100-500 ms and are spread evenly over
the run, ~3 a second. The run kept 305 evaluation dirs. `ended_epoch` is
stamped before `publish_evaluation_scratch()`, and for the 1 in
`NS_KEEP_DETAIL_EVERY` (100) evaluations kept whole (for `./ri profile`)
that function moved ~7 files from tmpfs to Lustre on the rank's main
thread, ~0.2 s at 25-40 ms a file created. Round 47 had moved the
extremes off the main thread (`.held/`), but not these.

**Change.** `publish_evaluation_scratch(defer=...)`: the record path
rewrite stays on the rank's thread, and the move itself
(`_move_scratch()`) is queued on the rank's record writer, the unpinned
thread round 54 set up for record writes, ahead of the record's own
write. Drains, failures raised on the next record, and abort paths are
the record writer's. Without a writer (self-checks, `NS_SCRATCH_DIR`
unset) the move is inline as before. The streaming-retention self-check
waits on the writer before checking that scratch is gone; `./ri
self-check wsclean` passes.

**Measured.** ABAB, run one after another (cpu-q-576, this job's 38
CPUs; the node's other half was busy, load 51-53 from the second run on):

| run | code | evals/s | gaps, worker-s | gaps > 100 ms | log(Z) |
| --- | --- | ---: | ---: | ---: | --- |
| r95-a | HEAD | 273.0 | 117.5 | 281 | 0.09704 ± 0.00051 |
| r95-b | deferred | 266.2 | 70.2 | 69 | 0.09719 ± 0.00052 |
| r95-a2 | HEAD | 255.3 | 109.6 | 242 | 0.09681 ± 0.00050 |
| r95-b2 | deferred | 269.3 | 73.0 | 76 | 0.09687 ± 0.00051 |

Gap time falls by ~42 worker-s out of ~3,400 (35 workers x ~97 s), i.e.
~1.2% of a search's worker time. That is below what end-to-end evals/s
can resolve on a shared node (the rows above span 7%), so the gap sum is
the measurement. Almost all the gaps over 100 ms left in B (65 of 69 in
r95-b) are in the first 2 s, ranks waiting at the end of the prior
phase. Both arms keep the same layout (309/302 dirs, 57/56 with logs), and
log(Z) agrees within its error in all four. The change only moves files
after an evaluation is scored, so it cannot change what is sampled. It
is in the bound `nested_sampling` tree, so it needs no rebuild.

**Learned.**
- `ended_epoch` is stamped before the record's publish and retention
  work, so anything slow there is idle time in the profile, not
  evaluation time. Per-rank gaps from the records are the direct way to
  see it.
- `./ri profile --over-time` reads `wsclean.stdout.log` from kept
  evaluation dirs, and since rounds 92-93 only ~1 in 100 evaluations has
  one (55 of 26k here). The records' `timing` epochs would give it every
  evaluation.
- Startup accounts for ~22 worker-s per run (ranks waiting for the
  slowest prior evaluation) and the tail ~70. Together they are most of
  the idle that is left at 900 dead, and both get smaller as a run gets
  longer.

## Round 96 (tooling): `./ri profile --over-time` reads every evaluation

Round 95 found `--over-time` blind on current runs: it bucketed the
`-log-time` span of each kept `wsclean.stdout.log`, and since rounds
92-93 retention keeps a log for ~1 in 100 evaluations. It now reads the
run's records (packed or flat, through `load_evaluations_from_dir`) and
buckets every evaluation by its `timing.started_epoch`; `ms/eval` is the
whole evaluation (simulate + WSClean + metrics), `ended - started`. The
`vis/eval` column needs the visibility count, which only the logs had, so
each WSClean record now carries `metrics.gridded_visibilities`, parsed by
`clean_convergence()` from the log the evaluation already reads for its
CLEAN fields (one regex search; no new IO). A run from before this whose
logs cover at least as many evaluations as its records still gets the
old log view.

Checked on a 16-rank WSClean search (nlive 50, 300 dead, 1,816
evaluations): the old view had 47 logs to read; the new one times 1,807
records (the rest carry no epochs), every record has the count, and the
table shows the expected warm-up (97 evals/s in the first bucket, ~200
after). No sampled value changes, so there is no A/B.

## Round 97 (diagnostic, one rejection): an R2D2 evaluation on the current tree

**Where an R2D2 search's time goes now.** The user's 30-50% MeqTrees
figure is the 22 Sep production R2D2 run (job 36031480, before every gnhf
round): `./ri profile` gives simulate 34.7% (1.94 s an evaluation), the
R2D2 request 51.4% (2.87 s) and idle 11.5%. A 17-rank search on this tree
(nlive 50, 150 dead, seed 4242, 1,085 evaluations, 1m16s) gives simulate
**4.2%** (47 ms), the R2D2 request 94.0% (1.06 s), idle 1.6%.

**Inside a request.** `scripts/probe_r2d2_request.py` over 12 `.mat`
files re-simulated from that run's records (`--mat-only`, fastnoise
bound), 3 threads as the run's ranks had, solo: 0.33-0.36 s warm. Under
cProfile the U-Net is 65% (`conv2d` 43%, `conv_transpose2d` 9%, instance
norm 5%), `load_state_dict` 8%, the normal-operator kernel and PSF 4%,
`gen_imaging_weights` 4%.

**Rejected: one net per checkpoint.** `load_net` re-assigns each
iteration's checkpoint over the worker's one U-Net (~16 a request), 31 ms
of a request under cProfile. A per-worker cache of one deep-copied net per
checkpoint (same tensors, so the same arithmetic) turns that into a
lookup. Four interleaved ABAB probe pairs, warm mean: HEAD 0.331, 0.332,
0.330, 0.330 s; cache 0.336, 0.334, 0.335, 0.401 s, and the first request
paid +0.4 s for the copies. cProfile's per-call cost had inflated the
22,852 tiny `Module.load` calls; the real swap cost is the ~0.4 ms an
evaluation already priced in docs/nested-sampling-speed.md. Not kept.

**Tooling fix.** The probe wrote its own hand-copied config, which lacked
`target_dynamic_range: 0.0` (round 4). Without it every request computes
the operator norm by power iteration: 83% of a 2.0 s request, a cost runs
never pay. The probe now writes its config with the run's own
`write_r2d2_config` (`image_dim()` overridden by `DIM`, since r2d2.sif's
Python 3.10 has no `tomllib`), and `EXTRA` still appends to it.

**Learned.**
- MeqTrees is no longer a lever for R2D2 searches (4%). The request is
  the evaluation, and two thirds of it is the U-Net's convolutions, which
  round 86 closed for bit-identical changes.
- An R2D2 request at 3 threads is not bit-reproducible run to run: two
  runs of the same code differ in 10-15 of 48 output FITS by 1e-17 to
  8e-16. At 1 thread two runs are byte-identical (48 of 48). Identity
  checks of R2D2 changes must run at 1 thread, or compare to tolerance.

## Round 98 (negative): one checkpoint copy per NUMA node, workers pinned

**Tried.** The pool parent loaded one copy of the R2D2 checkpoints per NUMA
node holding the job's CPUs (`set_mempolicy(MPOL_PREFERRED)` through ctypes
syscall 238), and each worker was pinned to its node's CPUs.

**Probe** (38 pinned copies of `probe_unet_forward.py` at once, this
allocation's 32/6 split): weights node-local 51.4 ms per forward, interleaved
59.4 ms, weights on node 0 57.5 ms, on node 1 102 ms (1.9x worst case).

**Search** (38 x 1, seed 4242, ABBA; async, not synchronous as first
logged - the commands had no `--synchronous`, round 102): per request 1116/1117 ms
first touch against 1170/1188 ms per-node copies, **5-6% slower**. Rejected;
pinning takes away the scheduler's choice of idle CPU, which is worth more
than locality here.

**Learned.**
- `kernel.numa_balancing` is 0, so the parent's single load puts all
  ~3.1 GB of weights on the node it ran on, and nothing moves them.
- Upstream `set_common_args` caps torch threads at the affinity mask, so
  pinning a worker silently caps its request threads too.
- 38 x 1 R2D2 searches with seed 4242 gave four different chain md5s, even
  within one arm. Explained in round 102: they ran async (the default), so
  differing chains are expected; synchronous R2D2 runs reproduce.

## Round 99: R2D2 checkpoint pages interleaved across the job's sockets

**Why.** The R2D2 pool parent loads every checkpoint (~3.4 GB) once and
forks the workers, which all stream those pages. First touch puts them on
whichever NUMA node the parent ran on. A default `./ri search r2d2` job
takes a whole icelake node (`--exclusive`, 2 sockets x 38 cores), so half
the workers read every weight across the socket interconnect, and round 98's
pinned probe put a remote U-Net forward at ~1.9x a local one under load.

**Change.** `r2d2_serve.interleaved_pages()` wraps the parent's checkpoint
preload in `set_mempolicy(MPOL_INTERLEAVE)` over the NUMA nodes holding the
job's allowed CPUs, then restores `MPOL_DEFAULT`, so workers' own
allocations stay first-touch. One node, or `R2D2_INTERLEAVE_WEIGHTS=0`,
leaves it off. `/proc/self/numa_maps` after warm-up on this half node:
3.38 GB on node 0 before, 1.77 / 1.61 GB after. No worker is pinned
(round 98 showed pinning costs more than locality gains). The arithmetic is
unchanged; only where pages live moves.

**Whole node** (one `--exclusive --qos intr` icelake job, cpu-q-200;
76 x 1 async, nlive 200, 5 repeats, 1200 dead, seed 4242; warm = evaluations
starting 10 s after the first):

| run (order) | pages | evals | evals/s | warm R2D2 request | log(Z) |
| --- | --- | ---: | ---: | ---: | ---: |
| r99-long-1 (first in job) | interleaved | 19,570 | 46.6 | 1474 ms | 0.01795 ± 0.00034 |
| r99-long-2 | first touch | 19,234 | 41.8 | 1705 ms | 0.01805 ± 0.00032 |
| r99-long-3 | interleaved | 19,331 | 46.5 | 1533 ms | 0.01801 ± 0.00034 |

**+11% evals/s, -10 to -14% per request**, well past the ~4% that
sequential arms can fake; the interleaved arm also took the job's cold
first slot. The job was cut after three runs for time, so there is one
first-touch run; log(Z) agrees within its error (async 76-rank runs are not
bit-reproducible, round 97).

**Half node** (this 38-CPU allocation, 32 CPUs on node 0 and 6 on node 1;
38 x 1, nlive 50, 2 repeats, 300 dead): first touch 1306 ms (first run),
1093 ms; interleaved 1081, 1082 ms. Neutral on a lopsided split, where
most pages already sat on the majority node, though the 1306 ms first run
fits a parent that landed on the minority node.

**Learned.**
- On a whole icelake node first-touch weights cost R2D2 ~11% of its
  throughput. Interleaving is free on a half node and needs no rebuild:
  `r2d2_serve.py` is bound from the working tree.
- Short 76-rank runs (300 dead, ~300 evaluations, ~4 per rank) measure
  only the cold first request (1.7-1.8 s); a whole-node R2D2 A/B needs
  ~20k evaluations (~7.5 minutes a run), so an `intr` hour fits ~6 runs.

## Round 100 (negative): MKL-packed weights for the U-Net's 2x2 convs

**Tried.** The U-Net's 2x2 convs as `_mkl_linear` with prepacked weights,
round 86's isolated-conv winner, inside the real forward.

| | solo forward | 38 copies, one socket | max abs diff |
| --- | ---: | ---: | ---: |
| stock | 28.2 ms | 58.8 ms | - |
| MKL-packed | 29.5 ms | 65.4 ms | 4e-6 |

**5% slower solo, 11% slower loaded.** Rejected. Round 86's win came from an
isolated random-weight 1024->1024 conv that took 17.5 ms where the same conv
inside the network takes ~3.5 ms: time conv levers inside the real network.

**Also ruled out.**
- Huge pages: THP `always` already backs 4,598 of 5,926 MB of the anonymous
  memory of all 25 loaded checkpoints.
- A 1-thread request (0.70 s warm, ~18 forwards) is conv2d 59%,
  conv_transpose 7%, instance norm 5%, FINUFFT 5%, `gen_imaging_weights` 3%,
  `.mat` load 1.8%, FITS writes 1.8%. No non-U-Net target is left; the
  forward goes 28.2 -> 58.8 ms from solo to 38 copies (weight streaming), so
  the only CPU lever left is cross-rank batching.
- This allocation was all 38 CPUs of node 0 this time, not rounds 98-99's
  32/6: record the allocation shape with any NUMA-sensitive result.

## Round 101 (robustness): a lost worker pool costs 40s, not 5 minutes

**Why.** `./ri self-check self-heal`, owed since round 92, had not run. Its
host-side pass failed first: round 99 added `self_check_interleaved_pages`
without classifying it in `scripts/test_self_checks.py` (CI's check), now
listed as host-runnable (ctypes only). The fifth scenario, killing the
WSClean pool's process group, then failed its 300s budget.

**Found.** Timed alone (3 ranks, nlive 20), the pool died at 00:51:35 and
the ranks gave up at 00:56:36: each reconnect waited the full
`NS_POOL_CONNECT_SECONDS` (300s) for a worker no pool would start. The retry
then finished in 7s. On a production node that is five idle minutes of every
rank per lost pool. A second bug came with it: `sidecar_restore` ran on the
left of `| tee`, in a subshell, so the pool it started never reached
`SIDECAR_PIDS`. It outlived the run (the next search's leak check reaped
it), and a second retry would have started a duplicate pool on the same
FIFOs - for R2D2, another ~4 GB.

**Changed.**
- `_connect_shell_started_worker`: a reconnect (the rank has had a worker on
  these FIFOs) gives up once `<rank>.pid` has named a dead, unreplaced
  process for `NS_POOL_GONE_SECONDS` (30s), read once a second since the
  file is on Lustre. A first connect keeps the full 300s for slow warm-ups.
  `self_check_worker_pool_connect` covers it.
- `run_with_retries` runs `sidecar_restore` in its own shell, its stderr
  through a file and then into `run.log`. A process substitution was tried
  first: the restored pool inherits its pipe (fd 63), which kept a `tee`
  alive as long as the pool's FUSE daemons. The progress-bar self-check now
  asserts the hook runs in the launcher's shell, and fails on the old pipe.

| | pool killed -> run finished | `./ri self-check self-heal` |
| --- | ---: | --- |
| before | > 300s (ranks gave up at 301s) | FAIL at scenario five |
| after | 39s, 41s | passed, 2 min 37 s and 3 min 0 s |

No throughput change: evaluations never see this path, so there is no
`benchmarks.jsonl` row (the self-heal runs sit outside
`results/nested-sampling/` and were deleted).

**Learned.**
- `./ri self-check self-heal` is the only test of pool loss, and it was last
  run before round 92; host-side CI checks had broken in round 99 unnoticed.
  Run `python3.13 scripts/test_self_checks.py` (the host `python3` is 3.6)
  after adding any `self_check_*`.
- Apptainer's `squashfuse_ll` and `fuse-overlayfs` daemons are not in a
  pool's process group, so every pool killed as a group leaves two to four
  of them behind, reparented to init, until the job ends. 68 had piled up
  from this round's runs alone. Harmless inside a Slurm job; worth knowing
  when counting processes.

## Round 102 (diagnostic): synchronous R2D2 searches are bit-reproducible

**Question.** Round 98 logged that synchronous, seeded 38 x 1 R2D2 searches
gave different chains, which would rule out chain identity as an R2D2 A/B
gate. Is that so on the current tree?

**Runs** (commit 2c92357, this allocation, seed 4242, `--synchronous`,
nlive 20-50, 2 repeats; rows `r102-A`..`r102-F`):

| runs | shape | dead | evals | chains |
| --- | --- | ---: | ---: | --- |
| A, B (raced) | 8 x 1 | 30 | 170 | all 7 md5s identical |
| C, D (back to back) | 38 x 1 | 300 | 2,303 | all 7 md5s identical |
| E, F (back to back) | 8 x 4 threads | 60 | 394 | identical |

**Cause.** Round 98's commands had no `--synchronous`, and `defaults.toml`
sets `NS_SYNCHRONOUS = 0`, so those runs were async, which does not
reproduce by design. Round 98's section is corrected.

**Learned.**
- A synchronous, seeded R2D2 search reproduces byte for byte across runs,
  at 1 or 4 threads per request, raced or back to back. Chain md5 is a valid
  R2D2 A/B gate for anything meant to keep the arithmetic unchanged.
- The thread count changes the arithmetic. The same 170 draws scored at
  8 x 1 and at 8 x 4 differ in every objective, by a median relative 4.9e-8
  and at most 1.8e-5. A thread-count A/B therefore cannot be gated on
  chains, and an automatic `R2D2_OMP_THREADS` that differs between arms (it
  follows the memory-clamped rank count) splits chains too. Each record's
  `r2d2_threads` says which one it got.
- A synchronous R2D2 A/B has to pass `--synchronous`: the bare
  `./ri search r2d2` is async.

## Round 103 (diagnostic): R2D2 on `sapphire` against `icelake`

**Question.** Round 84 found that a loaded sapphire core is ~6% *slower*
than an icelake core for WSClean, so the default partition stayed
`icelake`. The R2D2 U-Net, though, is bound by each core's weight stream
(round 11), and sapphire has DDR5. Is R2D2 the other way round?

**Probe.** `.probe/r103/run.sh`: N pinned copies of the real 32x32 U-Net
(checkpoint N1, 124 MB of weights each, 1 thread), 4 x 30 forwards timed
from a shared start. Icelake ran on this job (38 CPUs, all NUMA node 0,
with the other half of the node busy). Sapphire ran as one 56-CPU job
(`job.sh`, cpu-r-101).

| copies | icelake ms/forward | sapphire ms/forward |
| ---: | ---: | ---: |
| 1 | 28.4 | 19.0 |
| 19 / 28 | 41.3 (19) | 28.6 (28) |
| 38 | 61.3 | 33.7 |
| 56 | - | 43.9 |

Per loaded core that is 16.3 forwards/s (icelake, 38) against 22.8
(sapphire, 56), +40%, and per half node it is 2.1x.

**Searches.** Async R2D2, nlive 150, 5 repeats, seed 4242, one rank per
CPU, commit bce22a8. Rows `r103-*`.

| run | partition, ranks | job CPUs per NUMA node | dead | evals/s | per core | request ms | log(Z) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| icelake38 | icelake, 38 | 38/0 | 600 | 30.1 | 0.79 | 1161 | 0.01789 ± 0.00050 |
| sapphire56 | sapphire, 56 | 54/2 | 600 | 45.3 | 0.81 | 1132 | 0.01780 ± 0.00054 |
| ice-1, ice-2 | icelake, 38 | 38/0 | 400 | 29.3, 30.8 | 0.77, 0.81 | 1200, 1148 | 0.01725, 0.01590 |
| sap-on1, on2 | sapphire, 56 | 20/36 | 400 | 54.2, 53.6 | 0.96 | 916, 921 | 0.01702, 0.01714 |
| sap-off1, off2 (`R2D2_INTERLEAVE_WEIGHTS=0`) | sapphire, 56 | 20/36 | 400 | 47.7 (cold), 53.6 | 0.96 | 1026, 930 | 0.01698, 0.01704 |

All log(Z)s agree within their errors. The runs are async, so their
chains differ.

**Learned.**
- For R2D2 the partitions trade places. Half a sapphire node does 1.8x
  the evaluations of half an icelake node (54 against 30 evals/s), and
  about +20% per allocated core. For WSClean (round 84) it is +41% and
  -5%. The request at one rank per CPU is ~0.92 s on sapphire against
  ~1.16 s on icelake. `SBATCH_PARTITION=sapphire` (`--partition
  sapphire`) is the faster R2D2 run whatever the deadline. Whether it is
  also the cheaper one depends on the project's per-partition charge,
  which round 84 could not read. The default stays `icelake` until that
  charge is known.
- The searches recover only half of the probe's per-core gap (+20%, where
  the probe gave +40%). In the probe every copy owns its weights, first
  touched locally. In a run, every rank streams the pool parent's one
  shared copy, and that is the realistic case.
- Round 99's interleave on a lopsided allocation: on 20/36 it is neutral
  (53.6-54.2 against 53.6 warm). The 54/2 job (`sapphire56`) came in at
  0.81 per core with a 1132 ms request, against 0.96 and ~920 ms on 20/36.
  That fits half the weights being placed on a node that holds 2 of the
  job's 56 CPUs, but that run also differs in dead count and node, so it
  is not shown. An interleave restricted to nodes holding a fair share of
  the job's CPUs is the follow-up, and it needs a 54/2-shaped allocation
  to test. Slurm decides the shape, so record it with every result.
- Every sapphire allocation in this round and round 84 shared its node
  with other jobs (load 47-139 on 112 cores), so the gains above are what
  a shared node gives.

## Round 105: one R2D2 rank per CPU on whole nodes (cap 56 -> 112)

`NS_R2D2_MAX_RANKS = 56` dated from E4 (`csd3-experiments.md`), a sapphire
scaling test at ~12 evals/s that predates every round here. It only binds on
a whole node, which is the default `./ri search r2d2` job: there it gave 56
ranks x ceil(CPUs/56) = 2 threads, i.e. 112 threads on icelake's 76 cores.
Round 99 measured an explicit 76 x 1 and never the shipped shape.

**Setup.** Two exclusive whole-node jobs (icelake 39285695, sapphire 39286524),
each running ABAB async searches back to back: `./ri search r2d2 --nlive 200
--num-repeats 5 --max-ndead 1000 --seed 4242`, shipped SIFs, commit 632611d.
Sapphire's B pair ran after the cap edit, but with explicit `--mpi-procs`, so
the cap played no part.

| node | arm | ranks x threads | evals/s | wall to 1000 dead | request ms | log(Z) x1e3 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| icelake 76 | A1 | 56 x 2 (shipped) | 38.4 | 427 s | 1369 | 18.15 +/- 0.34 |
| icelake 76 | B1 | 76 x 1 | 47.2 | 319 s | 1515 | 18.03 +/- 0.35 |
| icelake 76 | A2 | 56 x 2 (shipped) | 41.0 | 445 s | 1290 | 17.86 +/- 0.36 |
| icelake 76 | B2 | 76 x 1 | 47.1 | 353 s | 1519 | 17.83 +/- 0.38 |
| sapphire 112 | A1 | 56 x 2 (shipped) | 66.2 | 263 s | 781 | 17.94 +/- 0.34 |
| sapphire 112 | B1 | 112 x 1 | 83.2 | 170 s | 1229 | 17.84 +/- 0.37 |
| sapphire 112 | A2 | 56 x 2 (shipped) | 73.8 | 241 s | 705 | 17.97 +/- 0.35 |
| sapphire 112 | B2 | 112 x 1 | 86.0 | 197 s | 1199 | 17.74 +/- 0.40 |

(Wall is PolyChord's, from the benchmark row; request ms is the
`image_container` stage.)

**Result.** One rank per CPU wins on both partitions: +19% on icelake
(47.2 against 39.7 evals/s mean) and +21% on sapphire (84.6 against 70.0),
with the time to 1000 dead points down 21% and 24%. Every pair's log(Z)s
agree within their errors (async runs, so chains are not compared;
round 102). `NS_R2D2_MAX_RANKS` is now 112, sapphire's core count, so it no
longer binds on any CSD3 CPU node and the memory clamp (4.2 GB + 0.4 GB per
rank, ~49 GB at 112) decides, as it already did for half nodes.

**Learned.**
- A whole node with the shipped default ran ~20% under what its cores give.
  Every whole-node R2D2 search before this round paid that.
- Half the ranks at two threads each makes each request faster (705-781
  against ~1200 ms on sapphire) but not twice as fast, so the product
  favours more ranks. This matches round 18 on half nodes.
- An exclusive whole sapphire node (load 0.5 at start) gave 84.6 evals/s,
  0.76 per core. Round 103's shared half node gave 0.96 per core at 56 x 1,
  so the per-core rate falls off by a whole node, or the node's other
  tenants differ. Not separated here.
- `--qos=intr` allows one job per user; a second submit fails with
  `QOSMaxSubmitJobPerUserLimit`. The default QOS whole sapphire node
  started within ~5 minutes.

## Round 106: one R2D2 checkpoint copy per socket on whole nodes

**Question.** Round 105 found a whole node gives R2D2 ~20% less per core
than half a node (sapphire 0.76 against 0.96 evals/s a core). Round 99
interleaves the one shared checkpoint copy across both sockets, so every
worker still reads half its weights across the interconnect. Round 98's
per-node copies lost 5-6%, but only on a lopsided 32/6 half node. Does a
copy per socket win on a whole node?

**Probe** (`.probe/r106/forkunet.py`). Pool-shaped: the parent loads all 25
checkpoints and forks P workers, and each worker runs 200 1-thread U-Net
forwards cycling through the checkpoints. Exclusive whole nodes, ABAB,
ms per forward (mean of workers):

| node | P | interleaved (shipped) | first touch | per-node copies, workers pinned to node |
| --- | ---: | ---: | ---: | ---: |
| icelake cpu-q-481 (job 39287408) | 76 | 51.4 / 52.6 | 50.9 | **48.5 / 49.4** (-6%) |
| icelake | 38 | 36.0 / 34.8 | 36.1 | 36.3 / 35.3 |
| sapphire cpu-r-65 (job 39287529) | 112 | 38.9 / 38.9 | 38.4 | **35.7 / 34.8** (-9.5%) |
| sapphire | 56 | 29.4 / 29.2 | 29.7 | 27.0 / 26.9 |

At full load locality is worth 6-10% of a forward. The 76/112-copy forward
is still 35-45% slower than at half load: most of round 105's per-core
drop is memory bandwidth, not distance.

**Change.** `r2d2_serve.load_checkpoints()`: when the job's CPUs span
NUMA nodes and each node holds at least half its fair share of them, the
pool parent loads the checkpoint series once per node under
`MPOL_BIND`. `serve_pool()` assigns worker i to a node in proportion to
the node's CPU count (`worker_nodes()`; round 98 had split them evenly),
and each forked worker pins itself to that node's allowed CPUs and swaps
its node's copy into the checkpoint cache. `R2D2_WEIGHTS_PER_NODE`
(`auto`, `1` or `0`) controls it; a lopsided allocation such as 32/6 or
54/2, a single node, and `R2D2_DEVICE=cuda` keep round 99's one
interleaved copy. Weights and arithmetic are unchanged; only where pages
live and which CPUs a worker may use move.

**Measured.** One exclusive sapphire node (cpu-r-65, job 39287894), 112 x 1
async (the default shape), nlive 200, 5 repeats, 1000 dead, seed 4242,
arms in this order:

| run | weights | evals | evals/s | ms per eval | parent pages N0 / N1 | peak MB | log(Z) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| node-a (first in job) | per node | 13,804 | 96.6 | 1124 | 831k / 776k | 6,864 | 0.01788 ± 0.00037 |
| il-a | interleaved | 16,579 | 87.9 | 1243 | 446k / 402k | 3,767 | 0.01764 ± 0.00038 |
| node-b | per node | 19,116 | 110.3 | 989 | 790k / 816k | 6,786 | 0.01742 ± 0.00047 |
| il-b | interleaved | 14,535 | 85.6 | 1272 | 420k / 429k | 3,905 | 0.01780 ± 0.00037 |

**+19% evals/s** (103.5 against 86.8), and the request is 16% shorter.
log(Z)s agree within their errors (async runs are not reproducible, round
102). The interleaved arms match round 105's 112 x 1 (83.2/86.0), so the
gain is not a node effect. The second copy costs ~5 s more start-up and
~3 GB more peak in the pool parent, which still fits `NS_R2D2_SHARED_MB`
(8000). A half icelake node with all 38 CPUs on node 0 (this loop's
allocation) is unaffected: one node, one copy.

**Learned.**
- Round 98's per-node copies failed on placement, not on pinning as such:
  with workers split evenly, a 32/6 allocation puts 19 workers on 6 CPUs.
  On a balanced node, pinning a worker to its socket's 38 or 56 CPUs
  costs nothing measurable.
- Whole-node R2D2 is bound on memory bandwidth: 112 forwards at once run
  33% slower than 56, even with local weights. Fewer bytes per forward
  (smaller weights or reuse across ranks) is the lever left there.
- Not yet measured: a whole icelake node in a real search (the probe
  gives -6% per forward, so expect ~+5-10%), and a balanced half node
  such as 20/36 on sapphire.


## Round 107: per-socket checkpoint copies on a whole icelake node

This measures round 106 on the partition it had only probed. It ran as
one exclusive whole icelake node (job 39288491, `cpu-q-482`, 76 CPUs as
38/38), with the default shape: 76 x 1, async, nlive 200, 5 repeats,
1000 dead and seed 4242, arms in this order:

| run | weights | evals | evals/s | ms per eval | parent pages N0 / N1 | peak MB | log(Z) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| node-a (first in job) | per node | 16,468 | 57.1 | 1295 | 791k / 816k | 6,840 | 0.01795 ± 0.00035 |
| il-a | interleaved | 16,961 | 46.4 | 1599 | 436k / 413k | 3,971 | 0.01746 ± 0.00041 |
| node-b | per node | 15,226 | 56.3 | 1313 | 833k / 773k | 6,761 | 0.01813 ± 0.00035 |
| il-b | interleaved | 16,785 | 47.4 | 1564 | 437k / 412k | 3,791 | 0.01789 ± 0.00036 |

That is **+21% evals/s** (56.7 against 46.9), and the request is 18%
shorter (1304 against 1582 ms). Both pairs agree, and the log(Z)s agree
within their errors. The interleaved arms match round 105's icelake 76 x 1
(47.2). Round 106's forked probe predicted only -6% per forward here, so
the probe under-reads the search, as it also did on sapphire (-9.5% in
the probe, -16% in the search). The default (`auto`) needs no change:
it was already on for whole nodes.

## Round 109 (diagnostic): round 11's batching lever, re-measured at 38 copies

Round 106 left "fewer weight bytes per forward" as the whole-node R2D2
lever. Batching is the only way to get there that keeps the fp32 weights
(118.3 MB a checkpoint), and round 11 measured it: -22% at B=4 under 24
copies, not bit-identical. This round re-measured it at today's shape:
round 106's forked pool, 25 checkpoints cycled, 1 thread, first touch,
38 workers on this loop's half icelake node (all on node 0, other users'
jobs on the other half). The probe was a pool-shaped variant of
`scripts/probe_unet_forward.py`'s `B` knob.

| workers | B | ms per image |
| ---: | ---: | ---: |
| 1 | 1 / 4 / 8 | 31.2 / 22.0 / 17.5 |
| 38 | 1 | 45.7, 48.2 |
| 38 | 8 | 30.2, 31.5 |
| 38 | 4 | 45.0 |

Under full load, B=8 cuts the time per image by **35%** (ABAB). B=4
gained nothing this time, unlike round 11's -22% at 24 copies, so the
benefit is not monotone in B. Rows still differ from the batch-1 forward
of the same input by up to 3.0e-5 on outputs that peak at 18-28 (relative
~1e-6), the same order as round 11's figure. The verdict is unchanged:
it is not result-preserving, and it needs a pool that gathers concurrent
requests at the same R2D2 iteration into one forward. Not attempted.

## Round 110 (diagnostic): per-socket copies on half a sapphire node

Round 106's `auto` turns per-socket checkpoint copies on for any balanced
split, but only whole nodes had been measured. Two half-node sapphire jobs
(56 CPUs, async 56 x 1, nlive 200, 5 repeats, 600 dead, seed 4242,
commit 05eab88), arms in this order. Rows `r110-*`.

| job shape | run | weights | evals | evals/s | ms per eval | peak MB | log(Z) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `--cpus-per-task=56` (cpu-r-112, 0/56) | node-a (first) | auto (off: one node) | 7,862 | 45.9 | 1174 | 4,070 | 0.01774 ± 0.00065 |
| | il-a | interleaved | 8,808 | 57.9 | 930 | 3,876 | 0.01758 ± 0.00063 |
| | node-b | auto (off) | 8,447 | 55.4 | 973 | 4,043 | 0.01766 ± 0.00062 |
| | il-b | interleaved | 8,425 | 55.8 | 964 | 3,807 | 0.01768 ± 0.00060 |
| `--ntasks=2 --ntasks-per-socket=1 --cpus-per-task=28` (cpu-r-6, 28/28) | node-a (first) | per node | 8,319 | 59.9 | 901 | 6,783 | 0.01777 ± 0.00064 |
| | il-a | interleaved | 8,327 | 63.5 | 846 | 3,888 | 0.01754 ± 0.00063 |
| | node-b | per node | 8,758 | 74.3 | 727 | 6,960 | 0.01768 ± 0.00062 |
| | il-b | interleaved | 8,080 | 63.2 | 855 | 3,856 | 0.01769 ± 0.00061 |

All log(Z)s agree within their errors.

**Learned.**
- Slurm's default half sapphire node can be one socket (0/56 here; round
  103 got 20/36 and 54/2). There `auto` is off and round 106 does not
  apply, so the shape decides whether a half-node R2D2 run gets it.
- The first arm of each job is the slow one in both jobs, including the
  one-socket job whose four arms run identical code (45.9 against
  55.4-57.9). A fresh node's first search pays a cold start of ~20%, so
  the next node A/B needs a throwaway warm-up arm, or ABBA with the first
  pair dropped.
- With that, the 28/28 A/B is inconclusive: the warm pair gives per node
  +17.6% (74.3 against 63.2) and the cold pair -5.7%. Not enough to call.
- The 28/28 interleaved arms (63.2-63.5) beat the warm one-socket arms
  (55.4-57.9) by ~12%, but on different nodes (load ~55-75 on cpu-r-6
  against ~100-125 on cpu-r-112), so this is a lead only. Asking for two
  sockets (`--ntasks-per-socket`) may be a half-node lever by itself,
  since it doubles the memory channels behind the U-Net's weight stream.
  It needs one job holding both shapes, for example a whole node split
  into two simultaneous 56-CPU searches pinned 0/56 and 28/28, before
  `scripts/lib/slurm.sh` asks for it.

## Round 111 (diagnostic): one socket against two, same sapphire node

Round 110's lead, on one node: an exclusive sapphire node (job 39291533,
cpu-r-6) ran 56-CPU R2D2 searches pinned with `taskset` either to 56 CPUs
of socket 0 (S) or to 28 + 28 across both sockets, with per-socket copies
(P, `auto`) or one shared copy (Z, `R2D2_WEIGHTS_PER_NODE=0`). Async 56 x 1,
nlive 200, 5 repeats, 600 dead, seed 4242, commit e0d9672; a throwaway
warm-up arm, then S P Z Z P S. Rows `r111-pin-*`.

| run | CPUs | weights | evals | evals/s | ms per eval | peak MB | log(Z) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| warm-S (discarded) | 56/0 | auto (off) | 9,608 | 55.7 | 974 | 4,044 | 0.01689 ± 0.00086 |
| S1 | 56/0 | auto (off) | 8,373 | 55.7 | 972 | 3,900 | 0.01767 ± 0.00061 |
| P1 | 28/28 | per socket | 9,800 | 74.4 | 727 | 6,986 | 0.01649 ± 0.00091 |
| Z1 | 28/28 | one copy | 9,396 | 63.4 | 852 | 3,888 | 0.01699 ± 0.00092 |
| Z2 | 28/28 | one copy | 8,373 | 65.2 | 825 | 3,982 | 0.01770 ± 0.00061 |
| P2 | 28/28 | per socket | 8,468 | 75.3 | 716 | 6,777 | 0.01756 ± 0.00064 |
| S2 | 56/0 | auto (off) | 8,434 | 53.3 | 1013 | 3,925 | 0.01767 ± 0.00059 |

All log(Z)s agree within their errors.

**Learned.**
- Spreading over both sockets is worth +18% by itself (Z 64.3 against S
  54.5, both pairs agree) and per-socket copies another +16% on top (P 74.9
  against Z), +37% in all. Every pair is consistent, so placement, not
  noise. The warm-up arm was not slow this time (55.7), unlike round 110.
- Caveat: here the other 56 CPUs were idle, so each 28/28 socket had half
  its cores (and their share of L3 and memory bandwidth) free. On a shared
  node the neighbour uses them; round 110's real `--ntasks-per-socket` job
  with neighbours still ran 63.2-63.5 one-copy and 74.3 per-socket, close to
  this table, so most of the gain should survive.
- Production default is a whole node (`--exclusive`), which already gets
  both sockets and per-socket copies. Only a sized (`--mpi-procs`) R2D2 job
  is exposed: `scripts/lib/slurm.sh` asks for `--ntasks 1 --cpus-per-task N`,
  which Slurm may place on one socket. Asking for
  `--ntasks=2 --ntasks-per-socket=1 --cpus-per-task=N/2` fixes the shape but
  queued ~30 minutes against ~3 in round 110, so it is documented in
  `docs/cluster.md` as the by-hand half-node shape rather than made the
  default.

## Round 112: `NS_SLURM_SOCKETS=2` for sized jobs

Round 111's placement, as a knob instead of a by-hand `sbatch`.
`scripts/lib/slurm.sh` now reads `NS_SLURM_SOCKETS` (default 1, unchanged):
with 2 and an explicit `NS_MPI_PROCS=N`, a submitted run asks for
`--ntasks 2 --ntasks-per-socket 1 --cpus-per-task ceil(N/2)` and counts the
R2D2 pool's shared memory twice in `--mem`, since `auto` then loads one
checkpoint copy per socket (peak ~6.9 GB against ~3.9 GB in rounds 110-111).
Whole-node (`--exclusive`) and GPU jobs are unchanged; mpirun ignores the
two tasks (`OMPI_MCA_ras/plm=^slurm`, round 110), and the batch step sees
all the job's cores, which is how round 110's 28/28 job ran.

**Verified.** `bash scripts/lib/slurm.sh --self-check` has a case for the
split (`--ntasks 2 --ntasks-per-socket 1 --cpus-per-task 3 --mem 23096` for
5 ranks at 8000+600 MB); shellcheck 0.11 (`uvx --from shellcheck-py`) is
clean; `sbatch --test-only` on sapphire accepts the 2 x 28 shape. No search
was rerun: the throughput is round 111's (74.9 against 54.5 evals/s on one
socket), and the cost is round 110's longer queue (~30 against ~3 minutes),
which is why it stays opt-in.

## Round 113 (diagnostic): `NS_SLURM_SOCKETS=2` end to end, and simulate re-profiled

**Socket knob, end to end.** Round 112 checked the request shape alone.
This round submitted a real search from a compute node with
`SBATCH_PARTITION=sapphire NS_SLURM_SOCKETS=2 ./ri search r2d2
--mpi-procs 56 --nlive 200 --num-repeats 5 --max-ndead 600 --seed 4242`
(job 39292394). The job started within seconds, not round 110's ~30
minutes, on cpu-r-6 (a shared node this time), and finished with no
memory trouble:

| run | evals | evals/s | ms per eval | peak memory | log(Z) |
| --- | ---: | ---: | ---: | ---: | --- |
| r113-sock | 8,444 | 59.4 | 908 | 6,855 MB | 0.01784 ± 0.00063 |
| r111-pin-P2 (exclusive node, warm) | 8,468 | 75.3 | 716 | 6,777 MB | 0.01756 ± 0.00064 |
| r111-pin-S2 (one socket) | 8,434 | 53.3 | 1,013 | 3,925 MB | 0.01767 ± 0.00059 |

The peak memory shows two checkpoint copies, so `auto` saw a balanced
split and took per-socket copies, as intended; log(Z) agrees. The rate
is a single, uncontrolled arm: it was the job's first (cold) arm, which
round 110 found ~20% slow (75 x 0.8 = 60), on a node shared with other
jobs. Read it as "the knob works", not as a new throughput figure.

**Simulate, solo.** A cProfile of the current tree (round 90's harness,
48 real draws x 3, one pinned CPU, host-built `fastnoise.so` bound in)
gives 13.0 ms a call, against round 90's 13.1:

| part | ms a call |
| --- | ---: |
| C phase ramp + XX/YY noise (`_ramp_xxyy2`) | 6.0 |
| DATA putcol | 0.76 |
| main-table open + close | 0.87 |
| UVW, ANTENNA1/2, TIME, TIME_CENTROID putcols | 1.50 |
| addrows | 0.29 |
| rest (template copy, argparse, json, pathlib) | ~3.5 |

The ramp+noise pass is at round 70's floor. The four small integer/time
putcols are ~1.5 ms, about 1% of a ~133 ms production WSClean evaluation
(round 89) even before in-run inflation, so this round did not try
splicing them as round 90 did with the subtables. Without the
`fastnoise.so` bind, the probe drops to numpy's normal draws and reads
35.9 ms a call. Any simulate probe that skips the bind measures the
fallback, not production.

**Learned.**
- `NS_SLURM_SOCKETS=2` works end to end through `./ri search`: the
  2 x 28 sapphire shape was accepted and placed immediately, the
  `--mem` request covered both checkpoint copies, and auto engaged.
- MeqTrees simulate has no lever left above ~1% of a WSClean evaluation.
  The remaining WSClean time is the binary (ducc0, rounds 83-88).

## Round 114 (diagnostic): socket placement does not matter for WSClean

Rounds 110-111 found R2D2 on half a sapphire node 18-37% faster split
28/28 across sockets than on one, because the U-Net is bound on memory
bandwidth. This round asked whether WSClean is too, which would make
`NS_SLURM_SOCKETS=2` worth setting for sized WSClean jobs. Same method as
round 111: one exclusive sapphire node (cpu-r-6, other 56 CPUs idle),
async 56-rank `./ri search wsclean --nlive 200 --num-repeats 5 --seed 4242`
pinned by `taskset` to one socket (S, 0/56) or both (P, 28/28), on the
patched SIF cache (0001-0017), commit d2c6035, after a discarded warm-up.
Rows `r114-*`; arms converge by themselves at ~21k evaluations (~50 s), so
`--max-ndead` above ~3000 changes nothing.

| job | order | S evals/s | P evals/s |
| --- | --- | --- | --- |
| 39292656 | S P P S | 476, 564 | 517, 532 |
| 39292822 | S P P S S P P S | 426, 463, 439, 407 | 439, 445, 426, 489 |
| mean of 6 | | 462 | 475 (+2.6%) |

All 13 log(Z)s lie in 0.0962-0.0972 (± 0.0004-0.0010) and agree. Per
evaluation, simulate and the WSClean binary moved together in every arm
(e.g. 22.0 / 69.8 ms in the fastest arm, 28.2 / 98.5 ms in the slowest),
so the spread is node-wide drift, not placement.

**Learned.**
- WSClean has no socket effect worth a longer queue: +2.6% against an
  arm-to-arm spread of ~±10% (ABBA pair means: S 445 / P 442, then S 423 /
  P 457). `NS_SLURM_SOCKETS=2` stays R2D2-only advice.
- 56 WSClean ranks on half a sapphire node now run ~400-560 evals/s
  (93-130 ms an evaluation), against round 84's 307 for 54 ranks before
  patches 0007-0017 and rounds 90-95.
- Even on an exclusive node, 50-second WSClean arms drift by ±10% from one
  to the next (the first job climbed 476 -> 564 over five arms). Below
  ~10%, a WSClean A/B on a node needs many interleaved arms or a direct
  stage timing, as AGENTS.md already says for the 20-CPU host.
- Reusing an `--output-dir` that holds a finished run resumes it and
  records a bogus row (here 69,607 evals/s in 0.3 s); that row was dropped
  from `benchmarks.jsonl`. Give every arm of a resubmitted job a new label.

## Round 115 (diagnostic): the 22 Sep production search, rerun on today's tree

**Question.** Every round so far timed one lever, or a short search.
What do rounds 1-114 together do to the run that started this work?
That run was the 22 Sep production R2D2 search (job 36031480,
`cb59233`): 6h20m to convergence, with simulate 39% of worker time.

**Run.** The same search, rerun on `5ced3d0` with the same SIFs (same
imager, meqtrees and polychord image ids), the same seed (190182240),
nlive 150, 15 repeats, 24 ranks x 1 thread, async, run to convergence.
It was a sized sapphire job, as before (`./ri search r2d2 --mpi-procs 24`
from the compute node, job 39293372 on cpu-r-73, started at once). Row
`r115-bigrun`.

| | 22 Sep (`cb59233`) | round 115 (`5ced3d0`) | change |
| --- | ---: | ---: | ---: |
| wall to convergence | 17,331 s (6h 19m) | **2,459 s (41 min)** | **7.0x faster** |
| evaluations | 71,395 | 65,420 | |
| evals/s | 4.12 | **26.6** | 6.5x |
| worker seconds per evaluation | 4.94 s | 0.858 s | 5.8x less |
| simulate (MeqTrees) per evaluation | 1,936 ms (39.2%) | **32 ms (3.8%)** | 60x less |
| R2D2 request per evaluation | 2,868 ms (58.0%) | 825 ms (96.1%) | 3.5x less |
| convert | 97 ms (2.0%) | gone: simulate writes the `.mat` (round 13) | |
| idle fraction | 11.5% | 0.05% | |
| after PolyChord returns | ~88 min | 34.5 s | |
| files left | ~225k | 6,647 | |
| dead points | 1,373 | 1,368 | |
| log(Z) | 0.01722 ± 0.00042 | 0.01683 ± 0.00041 | 0.7 sigma apart |

The two log(Z)s agree: their difference, 0.0004, is 0.7 of the combined
error. They are not expected to be identical, because async runs draw
different points. The post-run phase was 34.5 s in total: records 2.0 s,
flush 1.3 s, prune 8.2 s, `summary.json` 4.5 s and packing 18.4 s.

**Learned.**
- The 6h20m production search now takes 41 minutes on the same 24
  sapphire CPUs and the same images. Every gain is in the tree, plus the
  host-built `fastnoise.so` (round 42). A whole sapphire node (112 x 1,
  per-socket copies, rounds 105-106) ran 103.5 evals/s, ~4x this run's
  rate.
- MeqTrees went from 39% of worker time to 3.8%, so the user's 30-50%
  figure is gone on CSD3 too. The R2D2 U-Net is now the evaluation.
- In this run nothing else costs much: workers were idle 0.05% of the
  time, the post-run phase was 35 s, and the run left 6.6k files instead
  of 225k.
- `./ri health` on this live run (~20k flat records, running on the job's
  node) did not answer within 120 s from this loop's node, so it was
  killed. It was not looked into further.
