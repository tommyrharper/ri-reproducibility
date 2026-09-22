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
