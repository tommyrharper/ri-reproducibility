# Running on CSD3

This branch (`cluster`) runs the searches on Cambridge's CSD3
(<https://docs.hpc.cam.ac.uk/>) under Slurm, without Docker. It is the branch
to use on the cluster; `main` stays the Docker one.

## What changes and what does not

The four environments (WSClean with its patches and fork server, MeqTrees,
R2D2, PolyChord) are still defined by the Dockerfiles under `docker/`. Nothing
in them is rebuilt natively on the cluster: MeqTrees only exists as KERN
Ubuntu packages, and WSClean carries six local patches. They run under
[Apptainer](https://apptainer.org/), which CSD3's Rocky 8 nodes provide as
`apptainer` with no module to load (its own docs run `apptainer pull` bare);
the scripts accept `singularity` too, and if neither is on PATH the CSD3
module is `singularity/current`.

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

`hpc-work` is reached through symlinks (`~/rds/hpc-work` ->
`/rds/user/<crsid>/hpc-work` -> an `/rds-dN` mount); a checkout addressed by
either spelling works, since apptainer creates a bind destination the
container lacks and resolves one it reaches through a bound symlink (checked
here with `--no-mount tmp` against a symlinked path, FIFO and writes included).

`hpc-work` has a one-million-file quota. Evaluation directories are pruned as
on main (`prune_evaluation_artefacts`), and `NS_KEEP_MEASUREMENT_SETS` stays 0:
a Measurement Set is a directory of hundreds of files.

## Starting a search

```bash
# on a login node:
export SBATCH_ACCOUNT=MYPROJECT-CPU        # or --account; `mybalance` lists yours
./ri search r2d2 --nlive 500               # Submitted batch job 12345
./ri runs                                  # the run, listed as running
squeue -u $USER                            # the job, named after the run
./ri resume r2d2-vlaa-20260913T175735Z     # after the time limit: another job
```

`./ri search` and `./ri resume` submit a job when they are run outside a
Slurm allocation on a host that has `sbatch` (`scripts/lib/slurm.sh`); inside
`sintr` or a batch script, and on a host without Slurm, they run in place as
they always did. The run directory is claimed on the login node, so the job is
named after it, its stdout is `slurm-<jobid>.out` beside `run.log`, and
`./ri runs`, `./ri resume` and `./ri search --output-dir` treat the run as
live while a job of that name is queued or running (`squeue`), since the login
node cannot see the compute node's processes. The job inherits the whole
environment, so every flag and `NS_*` variable given on the login node, and
the seed, reach it unchanged.

Sizing: without `--mpi-procs` the job takes a whole node (`--exclusive
--mem 0`) and the run script sizes the ranks from the allocation; with it the
job asks for that many cores and the matching memory. `--partition` defaults
to `icelake` and `--time` to `12:00:00`, the SL3 cap, which every service
level accepts; SL1/SL2 accounts can pass `--time 36:00:00`, their own cap,
and a limit above your level is refused by sbatch. Both, and anything
else sbatch accepts, can also be set through sbatch's own `SBATCH_*` variables
(`SBATCH_QOS`, `SBATCH_RESERVATION`, ...). `NS_SBATCH=0` forces a run in
place. A failed submission (a bad account, say) removes the claimed directory
again.

`./ri bench run` submits the same way, as one job named `bench-<imager>-<stamp>`
(its Slurm log under `results/bench/`) that runs the warm-up and every repeat
in place on one node, so an interleave's arms share a machine. While it runs,
`./ri runs` and `./ri health` on the login node do not know its searches are
alive - they have no job of their own - so read the job's `slurm-<id>.out`.

## How a run works here

The run scripts (`scripts/run-nested-sampling*.sh`) are the Docker ones with
the containers replaced by processes:

- **Worker pools are host processes.** `scripts/lib/start-sidecars.sh` starts
  each pool as `apptainer exec <sif> ...` in the background, in its own
  process group (`setsid`), so the pool - the container's shell, its workers,
  their meqservers - is one `kill -- -pgid` at the end. The ranks talk to the
  workers over the same per-rank FIFO pairs as on `main`. Each pool logs to
  `workers-<image>.log` in the run directory.
- **Every worker is kept alive.** A simulate worker or WSClean fork server
  that dies, or exits because its rank's end of the FIFO closed, is started
  again by the shell loop around it; the R2D2 pool forks a replacement from
  its warm parent (`serve_pool`). The rank reconnects to the same FIFOs
  (`_connect_shell_started_worker` in `common.py`, sixty seconds' grace). A
  rank kills a wedged worker itself: every worker writes `<rank>.pid` beside
  its FIFOs, Apptainer shares the host's pid namespace, and
  `FifoWorker.kill()` walks `/proc` for its children, since `polychord.sif`
  has no procps. There is no rank-started fallback worker any more - the
  ranks cannot start a SIF from inside one - so a rank with no pool is a
  `WorkerDied`, which `run_with_retries` answers by restarting the attempt
  after `sidecar_restore` has brought a dead pool back.
- **The WSClean fork server is a pool too** (`.wsclean-workers/`), one
  `wsclean-zygote` per rank reading its FIFO, where the Docker branch spawned
  it per rank through `docker exec`.
- **PolyChord runs inside `polychord.sif`** on one node:
  `apptainer exec polychord.sif mpirun -np N python3 /opt/ri-nested-sampling/
  polychord_*.py`. The working tree's `scripts/lib/nested_sampling` is bound
  over the copy baked into `polychord.sif` and `meqtrees.sif`, so a run
  executes the code in this checkout - there is no `docker build` here to
  keep the two in step, and no rebuild after editing them. A change to a
  Dockerfile, a patch, or the `[[parameter_space]]` the MeqTrees image bakes
  its MS skeleton cache from still needs the image rebuilt on a Docker host
  and carried over again.
- **Memory sets the rank count**, as on `main`; `rank-budget.sh` reads the
  job's limit inside one (`SLURM_MEM_PER_NODE`, or `SLURM_MEM_PER_CPU` times
  the cores, which is how a partition default is spelled; `--mem 0` means the
  node, so `MemAvailable`) and `MemAvailable` outside one. A pool a
  SIGKILLed run left behind is reaped by the next run's budget (the pool's
  shell names its FIFO directory, whose run has no ranks and whose launcher
  pid in `.launcher.pid` is gone).
- **Open MPI is told not to read the Slurm allocation** (`OMPI_MCA_ras=^slurm`,
  `OMPI_MCA_plm=^slurm`): mpirun forks every rank itself on the one node, and
  a job that asks for cores rather than tasks would otherwise report one slot.
  Not yet exercised against a real `slurmd`; the first job on CSD3 is the
  test.
- `run.env` and the manifests record each image as the `ri.build-inputs`
  label the Docker build stamped it with (`ns_image_id`), which the SIF
  keeps.

### Reading a run from the login node

`./ri health` and `./ri runs` ask `squeue` as well as `ps`: a run whose job is
pending headlines `QUEUED`, and one whose job is running on a compute node this
host cannot see is read from what it has written to the shared filesystem
(`docs/run-health.md`). Everything that only ever read Docker (`./ri shell`,
`./ri smoke`, `./ri report`, `./ri plot fits`, `./ri clean`, `./ri disk-usage`,
`./ri self-check self-heal`) runs its SIF with `apptainer exec` and the same
binds the Docker mounts were; the self-heal check kills a pool's process
group where it used to remove a container.

## Slurm facts the scripts depend on

- Submit with `-A <PROJECT>-CPU` (`mybalance` lists yours), which is
  `SBATCH_ACCOUNT` or `--account` here. Wallclock is capped per service
  level - 36 hours on SL1/SL2, 12 hours on SL3 (`policies.html`) - so a long
  search is `./ri resume` across jobs.
- One node per job. `icelake` is 76 cores x 3370MiB (256GB), `icelake-himem`
  6760MiB per core (512GB); `sapphire` 112 cores x 4580MiB (512GB). A
  `--mem` above the cores' share is granted by allocating (and charging)
  more cores. R2D2 needs
  ~3.4GB per rank, so `--mem` sets the rank count, not `-c`.
- `nproc` inside the job reports the allocated cores, so `HOST_CPUS` in the
  run scripts needs no change. `NS_R2D2_MAX_RANKS` (8, from the 20-core
  host) caps an R2D2 job well below a 76-core node; the first whole-node R2D2
  run is where to re-measure it (`docs/nested-sampling-throughput.md`).
- Load nothing: Apptainer, Slurm and the SIFs are the whole toolchain. `uv`
  is still needed for the host-side scripts (`./ri profile`, `./ri merge`, the
  defaults loader, and the `bench.py record` step at the end of every job);
  install it into `~/.local/bin` and `uv sync` once on a login node, since
  the system `python3` is 3.6 and `./ri` itself re-runs under uv's Python.
