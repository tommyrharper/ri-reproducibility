# Running on CSD3

This branch (`cluster`) runs the searches on Cambridge's CSD3
(<https://docs.hpc.cam.ac.uk/>) under Slurm, without Docker. It is the branch
to use on the cluster; `main` stays the Docker one.

## What changes and what does not

The four environments (WSClean with its patches and fork server, MeqTrees,
R2D2, PolyChord) are still defined by the Dockerfiles under `docker/`. Nothing
in them is rebuilt natively on the cluster: MeqTrees only exists as KERN
Ubuntu packages, and WSClean carries six local patches. They run under
[Apptainer](https://apptainer.org/), which CSD3 provides on every node with no
module to load, as `apptainer` (older docs say `singularity`; the scripts
accept either).

Verified on a Docker host before any of the runtime was ported: a SIF built
from a `docker save` archive keeps the image's ENTRYPOINT, ENV and labels;
`mpirun -np 4` with mpi4py and pypolychord works inside `polychord.sif`, with
and without Slurm's environment present; the MeqTrees image passes the whole
`simulate_point_source_ms.py --self-check` from its SIF, with the repo bound in
from `$HOME`. Apptainer shares the host PID namespace by default, so a rank
inside one SIF can `pkill` a wedged worker running under another - the one
thing the Docker design needed `docker exec` for.

## Images: build on a Docker host, carry as SIF

```bash
# on any machine with Docker (a laptop, the lab box):
./ri build                       # the four images, as on main
./ri images export               # docker save -> images/archives/<name>.tar
rsync -avz images/archives/ login-cpu.hpc.cam.ac.uk:<repo>/images/archives/

# on CSD3, in the same checkout:
./ri images import               # apptainer build images/<name>.sif
```

`images/` is gitignored. The archives total ~2.1GB (r2d2 is 1.4GB of it) and
the SIFs the same again; importing all four took 80s on a 20-core box. Import
unpacks into `images/.tmp` and caches under `images/.cache`, both beside the
output, so the login node's `/tmp` and the 50GB home quota are not part of
the story. An archive newer than its SIF is rebuilt; anything else is skipped.

## Where things live on CSD3

| what | where | why |
| --- | --- | --- |
| this checkout, `images/`, `results/` | `/rds/user/<crsid>/hpc-work/ri-reproducibility` | Lustre, 1TB, shared by every node; home is 50GB NFS |
| in-flight Measurement Sets (`NS_SCRATCH_DIR`) | `/dev/shm` on the node | same as on main; nodes have 256GB+ |
| worker FIFOs | inside the run directory, as on main | one node per run, so a FIFO on Lustre is local to its readers |

`hpc-work` has a one-million-file quota. Evaluation directories are pruned as
on main (`prune_evaluation_artefacts`), and `NS_KEEP_MEASUREMENT_SETS` stays 0:
a Measurement Set is a directory of hundreds of files.

## Runtime design (the port, in order)

1. **Sidecars become processes.** `scripts/lib/start-sidecars.sh` starts each
   worker pool with `apptainer exec <sif> ...` in the background instead of
   `docker run --detach`; the ranks talk to them over the same FIFOs. There is
   no container to `exec` into, so the R2D2 and simulate fallbacks in
   `common.py` that spawned a worker through `docker exec` go away, and a
   pooled worker that dies is restarted by the shell loop that started it.
2. **The WSClean fork server joins the FIFO pools.** On main each rank spawns
   its zygote through `docker exec`; here the run script starts one per rank
   over a FIFO pair, the way the simulate workers already are, because a rank
   running inside `polychord.sif` cannot start another SIF.
3. **PolyChord runs inside its SIF**, `apptainer exec polychord.sif mpirun -np
   N python3 /opt/ri-nested-sampling/polychord_*.py ...`, on one node.
   Memory, not cores, still sets `NS_MPI_PROCS`; `rank-budget.sh` reads the
   Slurm allocation (`SLURM_MEM_PER_NODE`, else `MemAvailable`) instead of
   `docker info`.
4. **`./ri search` submits.** Outside an allocation it writes and `sbatch`es a
   job that runs the same run script; inside one (`SLURM_JOB_ID` set, or
   `sintr`) it runs in place, which is also how it runs on a plain Linux box
   with Apptainer. `--account`, `--partition` and `--time` map to `#SBATCH`
   lines; the run directory, `run.log` and `./ri resume` are unchanged.
5. **Everything that only read Docker** - `record-environment.sh` (image id
   from the SIF's labels), `nested-sampling-health.py` (`ps` only, no `docker
   top`), `generate-report.sh`, `plot-fits.sh`, `smoke-test-*.sh`, `shell.sh`,
   `self-check.sh`, `clean.sh`, `./ri disk-usage` - runs the same command under
   `apptainer exec`/`apptainer run`.

## Slurm facts the scripts depend on

- Submit with `-A <PROJECT>-CPU` (`mybalance` lists yours); SL2/SL3 jobs are
  capped at 36 hours, so a long search is `./ri resume` across jobs.
- One node per job. `icelake` is 76 cores x 3.4GB (256GB), `icelake-himem`
  6.8GB per core (512GB); `sapphire` 112 cores, 4.6GB per core. R2D2 needs
  ~3.4GB per rank, so `--mem` sets the rank count, not `-c`.
- `nproc` inside the job reports the allocated cores, so `HOST_CPUS` in the
  run scripts needs no change.
- Load nothing: Apptainer, Slurm and the SIFs are the whole toolchain. `uv`
  is still needed on the login node for the host-side scripts (`./ri profile`,
  `./ri merge`, the defaults loader); install it into `~/.local/bin`.
