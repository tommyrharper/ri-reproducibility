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
