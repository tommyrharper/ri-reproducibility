# CSD3 experiments

Jobs of increasing size on CSD3, to catch bugs before a long run and to set
`NS_R2D2_MAX_RANKS` and `NS_R2D2_CUDA_MAX_RANKS`. Newest last.

## Setup shared by every experiment

- Branch `csd3-experiments` off `cluster` at `fe94f1f`, run from a worktree
  with the main checkout's `images/` and `checkpoints/`.
- Images: the `./ri build` SIFs, `r2d2-cuda.sif` label `0d659c83`.
- `defaults.toml` as committed except the local experiment config:
  `NS_WSCLEAN_NITER = 100000` and four more enabled parameters
  (`source_l_pixels`, `source_m_pixels`, `declination_deg`,
  `integration_seconds`), so 9 searched parameters in all.
- Submitted from a login node (`./ri search`, so `sbatch`) unless noted.
- Accounts `WBARKER-SL3-CPU` / `WBARKER-SL3-GPU`. `intr` QoS for jobs of an
  hour or less.

## Experiments

| id | what | config | result |
| --- | --- | --- | --- |
| E1 | WSClean smoke, login-node submit, 9 params | icelake, 4 ranks, nlive 16, repeats 2, ndead 12, intr, 10 min | OK, 50 evals, 1.0 evals/s. 2.1s per eval (simulate 1.2s, image 0.9s), ~10x the 5-param space: `integration_seconds` 1-10s makes many more time samples. CLEAN always stopped at threshold (median 155 iterations), so `NITER=100000` costs nothing |
| E2 | R2D2 GPU smoke, login-node submit | ampere 1 GPU + 32 cores, 4 ranks, nlive 16, repeats 2, ndead 12, intr, 10 min | OK, 54 evals, 0 failed, 0.6 evals/s (0.78 steady). Image 0.65s, simulate 1.27s: on the GPU, MeqTrees simulation is now the larger stage, so more ranks should help |
| bug | whole-node CPU job refused | `sbatch --exclusive --mem=0` | CSD3's sbatch rejects `--mem=0` ("Please do not use --mem=0"), so any CPU search without `--mpi-procs` could not submit. Fixed: `ns_submit_run` asks for `--exclusive` alone, which brings the node's memory |
| bug | 1 of 826 evaluations failed in E3 (8 ranks) | `source_l_pixels` enabled | `simulation failed with exit 2`, scored as `FAILURE_OBJECTIVE`: an offset of `-7.4e-06` arcsec was passed as a separate argument, and argparse reads a negative in scientific notation as an option. Any near-zero negative source offset failed. Fixed: `common.py` passes the signed arguments as `--flag=value` |
| E3 | R2D2 GPU rank scaling | ampere 1 A100 + 32 cores, intr; one search per rank count, 6 min each, nlive 64, repeats 2, seed 4242; steady rate over the last 75% of each arm | 8 ranks 2.48/s, 16 ranks 4.63/s, 0 failed. 24 ranks: 72 of 1811 failed; 32 ranks: 356 of 1094. Failures are CUDA OOM: each worker holds all 25 checkpoints, 3.56GiB, so 24 x 3.56 > 80GB. The overloaded arms also had 3 and 13 `meqserver reported 2 error(s) during the predict`, never seen at 8 or 16 ranks |
| fix | GPU ranks bounded by GPU memory | `NS_R2D2_CUDA_MB_PER_RANK = 4096` | the run clamps ranks to 95% of the card (19 on an A100) and refuses a larger explicit `--mpi-procs`, rather than letting OOM failures score as fake worst cases |
| E4 | R2D2 CPU rank scaling | sapphire whole node (112 cores), normal QoS; as E3, 7 min per arm | 8 ranks 2.94/s, 16 5.59/s, 32 9.40/s, 56 11.97/s: still rising. At 32-pixel images a whole sapphire node beats one A100 (4.6/s). Failures: 1 of 3923 at 32 ranks, 4 of 4918 at 56, all `meqserver reported 2 error(s) during the predict` |
| diag | the predict failures | replayed 4 failed evaluations' exact commands twice each, alone (icelake 4 cores, intr) | all 8 replays succeeded in 2.5-8s, so load-dependent, not the parameters. Common factor: 3-7 channels, long observations, many at run start. The meqserver's error text was lost with the MS scratch; `common.py` now copies the MeqTrees logs into the evaluation on failure |
| set | rank caps | `NS_R2D2_MAX_RANKS` 8 -> 56, `NS_R2D2_CUDA_MAX_RANKS` 8 -> 16 | best measured without OOM failures |
| E5 | WSClean medium, whole node, no `--mpi-procs` | sapphire `--exclusive` (112 ranks), nlive 150, repeats 15, ndead -1, intr, 25 min limit | Whole-node submit worked (the `--mem` fix). But from minute 3 to 7 almost every evaluation failed: 4189 x `wsclean failed with exit 255` (`std::bad_alloc`) and 51 predict failures. Then a worker was missing, MPI aborted, and `run_with_retries` resumed from the checkpoint cleanly |
| bug | memory exhaustion in `/dev/shm` | 9-param space | The MS skeleton cache in `/dev/shm` had no bound (designed for <100 shapes in the 5-param space). With `integration_seconds` and `declination_deg` searched, 9159 shapes in 9260 evaluations: 138GB, and `/dev/shm` is capped at half the node's RAM. Fixed: `NS_MS_SKELETON_CACHE_MAX` (512) stops publishing past that many shapes |
| bug | the meqserver's error text was always lost | any failed simulate | `simulate()` builds in a `TemporaryDirectory` that deletes its logs as the error propagates. Fixed: on failure it copies `*.log` out first, and the error path already keeps them in the evaluation |
| E5 end | how the storm ended the run | | Not the time limit: PolyChord itself stopped ("unable to proceed after 151: failed spawn events") with a finished `summary.json` and log(Z) 97.8. `FAILURE_OBJECTIVE` (100) is the likelihood PolyChord maximises, so thousands of infrastructure failures took over the live points and the run "converged" on them. A failure storm yields a finished-looking, meaningless run, not just a slow one |
| E6 | E5 again with the skeleton cap | as E5 | Cap held (512 shapes, 7.5GB), but failures climbed again: 12 at 2 min, 151 at 4, 1974 at 6 (WSClean `bad_alloc`). `df` showed 96GB of `/dev/shm` used against 9GB of files. Cancelled |
| bug | meqserver leaks every MS it predicts into | 9-param space, tmpfs scratch | 113 meqservers held 38,304 deleted `sim.ms` table files open, so tmpfs could not free them: ~13MB per evaluation, invisible with the 5-param space's small MSs. Fixed: each simulate worker replaces its meqserver every `NS_MEQSERVER_RECYCLE` (20) predicts, with the existing wedge-restart path (~0.2s). Checked in the image: held deleted files drop to 0 at each replacement |
| E6b | E6 again with the meqserver recycling | as E5 | Healthy on the leak front (10 failed of 1292 at 4 min), but ranks began dying with `Disk quota exceeded`: `hpc-work` was at its 1,048,576-file quota. Cancelled |
| bug | failed evaluations filled the file quota | any failure storm | Every failed evaluation kept its MS and images (hundreds of files) for debugging, so E5 and E6's ~6000 failures wrote ~1M files. Fixed: `NS_KEEP_FAILED_ARTEFACTS` (20) failures per run keep everything, claimed atomically; the rest keep logs and metrics. E5, E6, E6b and the scaling runs deleted (numbers above) |
| E8 | E6b again, all fixes so far | as E5 | No storm: memory stable, failures 11 at 2 min, 87 at 4, 166 of 7426 at 6 (2.2%). Files grew ~10 per evaluation |
| bug | fresh-forest predict failures | E8's preserved logs | Both kinds follow a fresh compile in a new meqserver: `node 'VisDataMux' not found` (2 errors: E3/E4's "2 error(s)", then only after a wedge restart), and `Table /dev/shm/tmpXXX/sim.ms does not exist`, where the compiled selector still named the warm-up's MS. Recycling every 20 predicts took this path far more often: 2.2% of evaluations vs 0.08% in E4. Fixed: point the forest at the current MS after every compile, and retry a predict that reports errors once on a fresh meqserver (a real parameter failure fails again). Simulator self-checks pass |
| limit | files per evaluation | 1,048,576-file quota | WSClean kept 10 files per evaluation and R2D2 15 until the run ended. At the measured rates that is 2.5-6.5M files over 10 hours, so no long run fits. Fixed: as a run goes, each rank strips a successful evaluation to `metrics.json` unless it is among that rank's 20 lowest or highest objectives (the run's are among those, so the end-of-run image policy loses nothing) or one of 1 in `NS_KEEP_DETAIL_EVERY` (100) kept whole for `./ri profile`. `self_check_streaming_retention` covers it |
| E8 end | two worker deaths, 112 ranks | | The failure count stopped at 241 once the first restart loaded the retry fix (15,036 evaluations by the 25-min limit). But the run died twice with `simulate worker died`: a worker missed its reply deadline repeatedly, so MPI aborted and `run_with_retries` resumed from the checkpoint, using both retries. WSClean runs `-j 1`, so it is load on a saturated node (load average 366 on 112 cores in E6), not threads. 112 WSClean ranks leave no headroom; R2D2's 56-rank cap does |
| E7 | cancelled | | Queued 2h+ on the normal GPU queue, and it would have run the old code; replaced by E9 |
| bug | runs die of worker startup contention at 112 ranks | E8 FAILED at 18.5 min, retries used up | Each worker death came right after a start or restart (evaluation 80, then 10632 and 13032 right after restarts at 10631 and 13031). A worker opens its FIFO only after warming up, and on a restart all 112 warm up at once: the rank's 60s connect wait (E5's `no simulate worker for rank 78`) and 30s-plus reply bound ran out, and killing a slow worker only restarts a cold one. Fixed: connect wait 300s (`NS_POOL_CONNECT_SECONDS`) and 120s extra on a newly connected worker's first reply (`NS_WORKER_FIRST_REPLY_SLACK`). A dead worker is still caught, only later |
| E9 | R2D2 GPU medium, all fixes | ampere 1 A100 + 32 cores, 16 ranks (new default), nlive 150, repeats 15, ndead -1, intr, 40 min | Clean so far: 1715 evaluations, 0 failed, ~4.5 evals/s once live points were generated. Files ~2.07 per evaluation after each rank's first 40 |
| BIG | the ~10-hour run | worktree `bigrun` pinned at `cb59233` (so later edits cannot reach it mid-run) plus the 9-param config; R2D2 `--device cuda`, 16 ranks, nlive 150, repeats 15, ndead -1, normal QoS, 10:30:00; job 36025320, run `r2d2-vlaa-20260922T165238Z` | queued |
| E9 end | | | Cut at the 40-min limit with 10,211 evaluations, 0 failures, ndead 180, and no `summary.json`, so resumable |
| bug | resume brought a GPU run back on the CPU | E10 | `run.env` never recorded `R2D2_DEVICE`, so `./ri resume` of a GPU run defaulted to `cpu`: with a `-GPU` account sbatch refused it (partition icelake), and with a `-CPU` account it would have carried on quietly on the CPU. Fixed: `write_run_config` records it. Runs recorded before the fix resume with `R2D2_DEVICE=cuda ./ri resume <run>` |
| E10 | resume E9 in a new job | ampere, intr, 15 min, `R2D2_DEVICE=cuda ./ri resume` | job 36029471, resumed from 10,211 evaluations |
| E10 result | | | Resumed from the checkpoint at 10,211 evaluations and added ~2,800 more (13,014) with 0 failures before its own 15-min limit. Startup costs ~5 min: 16 workers read 25 checkpoints each (~56GB) off Lustre before the GPU sees anything |

## Where the rank caps landed

| setting | was | now | measured |
| --- | ---: | ---: | --- |
| `NS_R2D2_MAX_RANKS` (CPU) | 8 | 56 | sapphire, 112 cores: 8 ranks 2.9 evals/s, 16 5.6, 32 9.4, 56 12.0 |
| `NS_R2D2_CUDA_MAX_RANKS` (GPU) | 8 | 16 | one A100: 8 ranks 2.5 evals/s, 16 4.6; 24+ exceed the card's memory |

Both were measured on a 20-thread rig where 8 ranks saturated the cores. The
CPU figure was still rising at 56, so it is the best measured rather than a
plateau; the GPU one is bounded by 3.56GiB of checkpoints per worker.

## Notes for the next run

- A GPU run recorded before `R2D2_DEVICE` reached `run.env` resumes with
  `R2D2_DEVICE=cuda ./ri resume <run>`.
- WSClean at 112 ranks (a whole sapphire node) is where worker startup
  contention bites; R2D2's 56-rank cap leaves headroom on the same node.
- A 10-hour WSClean run does not fit the file quota even with the new bounds
  (~60k evaluations an hour); R2D2 does.
