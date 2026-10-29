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
rsync -avz images/archives/ login.hpc.cam.ac.uk:<repo>/images/archives/

# on CSD3, in the same checkout:
./ri images import               # apptainer build images/<name>.sif
```

The images must be `linux/amd64`: CSD3's CPU and GPU nodes are all x86_64.
A Linux x86_64 Docker host builds that by default; on Apple Silicon run
`DOCKER_DEFAULT_PLATFORM=linux/amd64 ./ri build` (emulated, and slow for the
WSClean and casacore compiles), and `./ri images export` refuses anything else.

`images/` is gitignored. The archives total ~2.1GB (r2d2 is 1.4GB of it) and
the SIFs the same again; importing all four took 80s on a 20-core box. Import
unpacks into `images/.tmp` and caches under `images/.cache`, both beside the
output, so the login node's `/tmp` and the 50GB home quota are not part of
the story. An archive newer than its SIF is rebuilt; anything else is skipped.

A SIF carries only the WSClean patches that existed when its Docker image was
built. To bring `images/wsclean.sif` up to this tree's
`docker/wsclean/patches/` without a Docker host, run
`./ri images patch-wsclean` inside a compute-node job (~2 minutes with 30
cores). It compiles WSClean in an `apptainer --fakeroot` sandbox against the
SIF's own casacore, layers the result over the SIF, and keeps the old one as
`wsclean.sif.orig`. The derived SIF keeps the old `ri.build-inputs` label and
adds `ri.wsclean-patches` (a hash of the patches), and its image id in
`run.env` and `benchmarks.jsonl` is the old one plus `+p` and the first 12
characters of that hash, so bench never pools the two. When `wsclean.sif` matches neither label to this tree, a WSClean search warns on
the login node, and its job builds the same layer once into
`$RI_WORK_DIR/.cache/ri/wsclean-<base id>-p<patches hash>.sif` (~2 minutes
with the job's CPUs), uses it, and every later run reuses it; the shared SIF
is not touched, a failed build falls back to it, and `NS_PATCH_WSCLEAN=0`
skips the step (round 94: +50% evals/s, chains byte-identical). On today's code the
upgrade from the shipped 0001-0006 SIF to 0001-0016 is worth +42% evals/s in a
17-rank search (binary 193 -> 126 ms, chains byte-identical). See
[csd3-speed](csd3-speed.md), rounds 43, 70 and 71.

## Where things live on CSD3

| what | where | why |
| --- | --- | --- |
| this checkout, `images/`, `results/` | `/rds/user/<crsid>/hpc-work/ri-reproducibility` | Lustre, 1TB, shared by every node; home is 50GB NFS, and CSD3 asks that jobs do no I/O there - a run writes `results/`, its FIFOs and `benchmarks.jsonl` under the checkout, so a checkout in `~` is the wrong place |
| in-flight Measurement Sets (`NS_SCRATCH_DIR`) | `/dev/shm` on the node | same as on main; nodes have 256GB+ |
| makems observations kept across runs (`NS_OBSERVATION_CACHE_DIR`) | `hpc-work/.cache/ri/ms-observations/<meqtrees image id>` | up to 183MB per image; saves ~0.33s/eval ([csd3-speed.md](csd3-speed.md), round 6) |
| a host-built `fastnoise.so` for a `meqtrees.sif` without one (`NS_FASTNOISE_DIR`) | `hpc-work/.cache/ri/fastnoise/<hash of source + gcc>` | ~20ms/eval of simulate ([csd3-speed.md](csd3-speed.md), round 42); like the row above, under `RI_WORK_DIR` (else the checkout's parent) even with `NS_JOB_ENV=0` |
| a job's `uv`, uv's Pythons (which `.venv` points into), `XDG_CACHE_HOME`, `HOME` | `hpc-work/.local/bin`, `hpc-work/.local/share/uv/python`, `hpc-work/.cache`, `hpc-work/.ri-job/home` | a job runs nothing from `/home` (below) |
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
they always did - inside an allocation, in the job's own environment rather
than the calling shell's (`ns_enter_job_env`; next section). The run directory is claimed on the login node, so the job is
named after it, its stdout is `slurm-<jobid>.out` beside `run.log`, and
`./ri runs`, `./ri resume` and `./ri search --output-dir` treat the run as
live while a job of that name is queued or running (`squeue`), since the login
node cannot see the compute node's processes. Every flag and `NS_*`
variable given on the login node, and the seed, reach the job unchanged -
but nothing else of the login node's environment does (next section).

## A job's environment

A job does no I/O on `/home` and runs nothing from a Nix store, although the
interactive tooling on a CSD3 account may well live there (a nix-portable
store, uv's own install, dotfiles). `ns_submit_run` submits with
`--export=NIL` - not `NONE`, which on this Slurm rebuilds the environment by
running the login shell, `~/.bashrc` and all, on the compute node - and the
job starts in `scripts/lib/job-env.sh`, which:

- empties the environment down to `SLURM_*`, however the job was submitted -
  and a run started by hand inside `sintr` or a batch script of your own
  re-execs itself through it too, once (`NS_JOB_ENV=0` keeps the shell's);
- sources the run's settings, saved by the login node as
  `.job-settings.env` in the run directory: `NS_*`, `R2D2_*`, `RI_*`, the
  image and output directories, as physical paths (`~/rds/hpc-work` is
  spelled through `/home`);
- builds the rest from nothing: `module purge`, then `module load
  rhel8/slurm` (`RI_JOB_MODULES` to change it); the base OS's `bash`,
  `python3`, `gcc` and `apptainer`; `uv` from `hpc-work/.local/bin`, with only
  uv's own Pythons under `hpc-work/.local/share/uv/python`; `HOME`,
  `XDG_CACHE_HOME`, `UV_CACHE_DIR` and `APPTAINER_CACHEDIR` under hpc-work;
- sets `APPTAINER_HOME` too: apptainer mounts the passwd home (`/home/<crsid>`)
  into every container whatever `$HOME` says, so without it every worker
  could read and write NFS home;
- refuses to start (`FATAL`, in the job's `slurm-<id>.out`) if any tool, the
  `.venv` interpreter, or a path any variable names resolves into `/home` or
  contains `/nix/store/`, and otherwise prints what each tool resolved to.

Once per account, on a login node, from the checkout:

```bash
W=/rds/user/$USER/hpc-work
install -D -m 755 "$(command -v uv)" "$W/.local/bin/uv"   # a standalone uv, not a Nix-built one
export UV_PYTHON_INSTALL_DIR=$W/.local/share/uv/python    # also in ~/.bashrc, so ./ri agrees
"$W/.local/bin/uv" python install "$(cat .python-version)"
rm -rf .venv && "$W/.local/bin/uv" sync
bash scripts/lib/job-env.sh --check                       # the job's environment, checked here
```

`--check` builds exactly what a job would get, with the settings this shell
would hand it, and says `nothing resolves into /home or a Nix store` or what
does. `bash scripts/lib/job-env.sh --self-check` tests the check itself.

Sizing: without `--mpi-procs` the job takes a whole node (`--exclusive`;
CSD3 refuses `--mem 0`) and the run script sizes the ranks from the allocation; with it the
job asks for that many cores and the matching memory. A CPU R2D2 job then
runs one rank more than its cores, for PolyChord's root, which never
evaluates (round 253; `NS_R2D2_ROOT_RANK_EXTRA=0` runs exactly `--mpi-procs`). `--partition` defaults
to `icelake` and `--time` to `12:00:00`, the SL3 cap, which every service
level accepts; SL1/SL2 accounts can pass `--time 36:00:00`, their own cap,
and a limit above your level is refused by sbatch. Both, and anything
else sbatch accepts, can also be set through sbatch's own `SBATCH_*` variables
(`SBATCH_QOS`, `SBATCH_RESERVATION`, ...). `NS_SBATCH=0` forces a run in
place. A failed submission (a bad account, say) removes the claimed directory
again.

`./ri search r2d2 --nodes N` (CPU only, `NS_NODES`) spreads one search over N
whole nodes (`--nodes N --exclusive`), each with `--mpi-procs` ranks of its own
(sized per node as above). The launcher's node starts the others' pools with
an `srun` step that runs the same script as `NS_NODE_ROLE=pools`, and the ranks
are a second `srun --mpi=pmix` step rather than the image's `mpirun`, which
cannot reach another node. A FIFO on Lustre is a pipe on each node by itself,
so each host's ranks and pools meet in `.r2d2-workers/<host>/` and
`.simulate-workers/<host>/`; pool logs are in `.node-<host>/`. Only rank 0's node
holds the chains tmpfs and its Lustre mirror. Before a retry each other node
resets its own workers when the launcher drops a `.node-<host>/reset` file, and
`run.env` keeps `NS_NODES`, so `./ri resume` comes back on as many nodes. Two
icelake half-nodes ran a converged search in half the wall time of one, at 94%
of the single-node rate per node (docs/csd3-speed.md, rounds 287-288). PolyChord
looks for its `.resume` on every rank, so a resumed attempt copies it into each
other node's chains tmpfs first (round 289: without that, every multi-node
retry and resume deadlocked). Each node appends its records to its own
`evaluations/records.<host>.jsonl`, which the finish or a resume folds into
`records.jsonl`. Each append writes every queued line at once
(`NS_RECORD_BATCH=1`), so a kill -9 on 2 nodes at 3,000 scored evaluations lost
no record, and a same-job `./ri resume` clears the killed attempt's `/dev/shm`
scratch on every node first (rounds 297-298). More nodes buy evaluations/s (4 nodes:
3.57x one), but PolyChord spends ~0.4 x workers/nlive more of them to
converge, so raise `--nlive` with `--nodes`: at least 4x the total ranks keeps
the overhead under ~10%, and the launcher warns below that (round 293).

`--qos intr` (`SBATCH_QOS`) is CSD3's interactive quality of service: the job
skips the queue but is capped at one hour, so without `--time` the limit
defaults to `01:00:00` under it - enough for a smoke-sized search or to try a
change on a real node.

`--then`, `--plot` and `--report` are refused on a login node: they wait for
the search to finish, and there the search only submits its job and returns.
Run them after the job, or chain them inside an allocation.

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
  (`_connect_shell_started_worker` in `common.py`, `NS_POOL_CONNECT_SECONDS`' grace, 300s; 30s once the pool itself is gone). A
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
  the cores, which is how a partition default and a whole-node `--exclusive` job are
  spelled) and `MemAvailable` outside one. A pool a
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
  keeps, plus a `+p<patches hash>` suffix on a `patch-wsclean-sif.sh` build.

### Being a good citizen on CSD3

The administrators have asked that login-node work stays at a few CPUs for a
few seconds and that nothing calls a Slurm client command more often than every
two minutes. What here would otherwise break that:

- `./ri health` and `./ri runs` - and so `./ri tui`, which reruns them every
  five seconds - share one cached `squeue` answer per user
  (`scripts/slurm_queue.py`, `~/.cache/ri/`), refreshed at most every 120s
  (`RI_SQUEUE_TTL`). A submission drops the cache, so a new run shows as queued
  at once; a job that has just ended can read as live for up to two minutes.
  `ns_run_is_live` (the guard in front of `./ri resume` and `--output-dir`)
  still asks directly, once per command.
- `./ri report` draws with two processes per pool on a login node rather than
  one per core (`RI_REPORT_WORKERS`).
- `./ri images import` caps `mksquashfs` at four processors outside a job
  (`RI_IMPORT_PROCESSORS`); inside `sintr` it uses the allocation.
- Never poll `squeue` in a loop of your own; `./ri runs` already caches it.

`OMP_NUM_THREADS=1` is exported by CSD3's default login environment
(`rhel8/global`), which a job no longer inherits or loads - but `sintr` and a
hand-written batch script still see it. GNU `nproc` honours it, so the run scripts count
CPUs with it unset (`env -u OMP_NUM_THREADS nproc`); a bare `nproc` there reads
1 on a 76-core node, which sized a whole-node job to a single rank.

### What a long run needs here

`hpc-work` allows a million files, and an evaluation used to keep 10-15 until
the run ended, so a run at CSD3 rates reached the quota within hours and every
rank then died on `mkdir`. Four bounds keep a long run inside it, all in
`defaults.toml` (`docs/csd3-experiments.md` has the measurements):

- `NS_KEEP_DETAIL_EVERY` (100): a successful evaluation is stripped to its
  record as the run goes, unless it is among its rank's 20 lowest or
  highest objectives - which is what the end-of-run image policy needs - or one
  of the 1 in 100 kept whole, which `./ri profile` reads.
- `NS_KEEP_FAILED_ARTEFACTS` (20): failures per run that keep their Measurement
  Set and images; the rest keep logs and metrics.
- `NS_MS_SKELETON_CACHE_MAX` (512): MS skeletons cached in `/dev/shm`. A
  parameter space that varies `integration_seconds` or `declination_deg` makes
  nearly every evaluation a new shape, and the cache had no bound.
- `NS_MEQSERVER_RECYCLE` (20): predicts before a simulate worker replaces its
  meqserver, which holds every MS it has written open and so pins it in tmpfs.

Those bounds still leave ~1.1 files per evaluation (its `eval-*.json` record;
a dir only for what it keeps), ~510 per unit of nlive for a converged
15-repeat run, so an nlive of ~2000 cannot finish even in an empty quota. A
search warns at the start when its estimate passes what its project has left
(`ns_warn_file_quota`, `docs/csd3-speed.md` rounds 91-92). That is the live
segment only: a resume packs the records before it into
`evaluations/records.jsonl` and a finished run packs them all, and
`scripts/prune-run-artefacts.py results/nested-sampling/*/` packs runs that
finished before round 93.

A failure storm is worse here than slow: `FAILURE_OBJECTIVE` is the likelihood
PolyChord maximises, so thousands of infrastructure failures took over the live
points and the run "converged" on them.

### Reading a run from the login node

`./ri health` and `./ri runs` ask `squeue` as well as `ps`: a run whose job is
pending headlines `QUEUED`, and one whose job is running on a compute node this
host cannot see is read from what it has written to the shared filesystem
(`docs/run-health.md`). Everything that only ever read Docker (`./ri shell`,
`./ri smoke`, `./ri report`, `./ri plot fits`, `./ri clean`, `./ri disk-usage`,
`./ri self-check self-heal`) runs its SIF with `apptainer exec` and the same
binds the Docker mounts were; the self-heal check kills a pool's process
group where it used to remove a container.

## R2D2 on a GPU

`./ri search r2d2 --device cuda --account WBARKER-SL3-GPU` runs R2D2's U-Net
on one A100: from a login node the job goes to the `ampere` partition with
`--gres gpu:1` and the 32 cores CSD3 allows per GPU (charged in GPU hours, so
the cores come free), and the R2D2 worker pool starts from
`images/r2d2-cuda.sif` with `apptainer exec --nv`. A `-CPU` account is refused
before anything is submitted, and inside a job without a GPU the run stops
before it starts rather than scoring every evaluation as a failure. WSClean has
no GPU path; `--device` only means something to `r2d2`.

Only the network moves (`patch_cuda_network` in `r2d2_serve.py`). Upstream's
own switch, `meas_op_on_gpu`, would also move the measurement operator - the
NUFFTs, the operator norm and this repo's FINUFFT plan cache - which gains
nothing at these image sizes and needs cufinufft. Instead the worker keeps each
checkpoint's weights on the GPU once, and upstream's single U-Net call gets its
three image tensors on the GPU and hands the result back as the CPU tensor the
rest of upstream expects. TF32 is off, so the GPU does the CPU's float32
arithmetic. The pool still forks its workers from one warm parent: the parent
never touches CUDA, and each worker makes its own context on its first request
(a context made before a fork is unusable in the child). The run starts a
CUDA MPS daemon on the host first (`sidecar_cuda_mps_flags` in
`start-sidecars.sh`; no root needed, pipe directory in the run's scratch) so
those contexts share the card instead of time-slicing it: 16 ranks on an A100
went from ~16 to ~41 evals/s. `NS_CUDA_MPS=0` turns it off
(`docs/csd3-speed.md`, rounds 117 and 119). A worker waits for the card on a
blocking CUDA event, not by spinning in the copy back, so its CPU goes to the
other ranks while it waits (+6%; `R2D2_CUDA_BLOCKING_SYNC=0` spins, round 123).
That freed CPU is why the default is 19 ranks a card, the most its memory
holds (`NS_R2D2_CUDA_MAX_RANKS`; 43.9 against 41.1 evals/s at 16, round 124).
`R2D2_RUNTIME=trt` runs the U-Net as TensorRT fp32 engines instead: requests
are 45% faster, but only 12 ranks fit, so a search gains ~4% (plans are built
on first use and cached, ~5 minutes; round 171). Since round 215 the GPU
default is TensorRT with `R2D2_TRT_SHARED=1`: one forward server holds the 25
engines for every worker, the workers stay on the CPU and no MPS daemon is
started, so the 19 ranks fit again. It converged the 22 Sep search in 785 s
against 1,165 s for eager torch (90.8 against 64.2 evals/s, log(Z) 0.06 sigma
apart). `R2D2_RUNTIME=torch` brings back eager torch with MPS, and
`R2D2_TRT_SHARED=0` brings back per-worker engines.

Measured on `gpu-q-23`, one search each, same seed (`--nlive 16 --max-ndead
48`), CPU and GPU on the same 32-core allocation:

| arm | evaluations/s | R2D2 imaging, median | simulate, median |
| --- | ---: | ---: | ---: |
| CPU, 8 ranks | 1.1 | 2.47s | 0.64s |
| GPU, 8 ranks | 1.8 | 1.19s | 0.78s |
| GPU, 16 ranks | 1.5 | 1.56s | 0.37s |
| GPU, 8 ranks, `./ri build r2d2-cuda` image | 3.1 | 0.85s | 0.61s |

The first three rows ran on an equivalent image made on the cluster from
`r2d2.sif` with pip (same torch, same wheel index); the last on the image the
Dockerfile builds, whose other packages resolved afresh. Its objective agreed
with the CPU run to the same 4.9e-6 over 34 shared evaluations. The 1.8 -> 3.1
gap between the two GPU images is one run each on a shared node and is not
explained; roughly 2-3x the CPU rate is what these support.
Those CPU rates predate csd3-speed.md rounds 6-10. Re-measured after them
(round 11), one A100 does 12.5-13.2 evals/s against 15.4-16.8 for 24 icelake
cores, so production should stay on CPUs.

Over the 30 evaluations the CPU and GPU runs shared exact parameters for, the
objective (`total_rms_jy`) agreed to 4.9e-6 relative and every metric to
1.3e-4 (off-source RMS, the smallest quantity) - float32 rounding between two
convolution implementations. The GPU sat at 0-50% utilisation with ~29GB in use
for 8 workers (~3.6GB each, the 25 checkpoints' weights), so what is left of the
imaging stage is its CPU side; `NS_R2D2_CUDA_MAX_RANKS` (8) is set from the
table and is worth re-measuring on a longer run.

The image: `./ri build r2d2-cuda` on the Docker host builds
`ri-reproducibility/r2d2:cuda` from the same Dockerfile with
`torch==2.13.0+cu130` (CSD3's A100 driver is 595, CUDA 13.2), `./ri images
export` carries it when it exists (~4GB) and `./ri images import` makes
`images/r2d2-cuda.sif`. It is not part of `./ri build all`.

## Slurm facts the scripts depend on

- Submit with `-A <PROJECT>-CPU` (`mybalance` lists yours), which is
  `SBATCH_ACCOUNT` or `--account` here. Wallclock is capped per service
  level - 36 hours on SL1/SL2, 12 hours on SL3 (`policies.html`) - so a long
  search is `./ri resume` across jobs.
- One node per job. `icelake` is 76 cores x 3370MiB (256GB), `icelake-himem`
  6760MiB per core (512GB); `sapphire` 112 cores x 4580MiB (512GB). A
  `--mem` above the cores' share is granted by allocating (and charging)
  more cores. For WSClean, half a sapphire node does ~41% more evals/s
  than half an icelake node but ~5% fewer per core (`docs/csd3-speed.md`,
  round 84), so pick by deadline or by charge. For R2D2, sapphire wins on
  both: half a node does ~1.8x the evals/s of half an icelake node and ~20%
  more per core (round 103), so `--partition sapphire` is the better R2D2 run
  unless sapphire is charged >20% more per core. R2D2 needs ~4.2GB once plus
  ~0.4GB per rank (the workers share the checkpoints after the fork;
  `docs/csd3-speed.md`, round 18), so on CPU nodes the cores set the rank count.
- `nproc` inside the job reports the allocated cores, so `HOST_CPUS` in the
  run scripts needs no change. `NS_R2D2_MAX_RANKS` (112) caps an R2D2 job
  at a whole sapphire node's cores: one rank per CPU beat 56 x 2 by ~20% on
  whole icelake and sapphire nodes (`docs/csd3-speed.md`, round 105).
  A CPU run then takes one rank more than its cores, because PolyChord's
  root rank never evaluates (+3.7% on half an icelake node, round 186).
  On such a node the R2D2 pool loads one checkpoint copy per socket and
  pins each worker to its socket (+19% on a whole sapphire node, round 106; +21% on a whole icelake
  node, round 107), and builds the ONNX Runtime sessions once per socket the
  same way (+36% on a whole icelake node, round 149; +40% and 117 evals/s on a
  whole sapphire node, round 150);
  `R2D2_WEIGHTS_PER_NODE=0` keeps one shared copy. A sized (`--mpi-procs`)
  R2D2 job can land on one socket, which turns this off; on one sapphire
  node, 56 CPUs split 28/28 ran 75 evals/s against 54 on one socket (+37%,
  round 111; +26% on icelake, round 250), so a sized CPU R2D2 search
  (`--mpi-procs` >= 4) submits with `NS_SLURM_SOCKETS=2` by default: the job
  asks for `--ntasks 2 --ntasks-per-socket 1 --cpus-per-task N/2` and twice
  the shared memory (`sbatch --test-only` gives it the same start estimate as
  one socket, round 250; `NS_SLURM_SOCKETS=1` keeps one). It is for R2D2 only: 56-rank
  WSClean ran the same either way (round 114), so a WSClean job need not wait
  for it.
- Load `rhel8/slurm` and nothing else: Apptainer, Slurm and the SIFs are the
  whole toolchain, and a job purges the rest (above). `uv` is still needed for
  the host-side scripts (`./ri profile`, `./ri merge`, the defaults loader,
  and the `bench.py record` step at the end of every job); a job runs the copy
  in `hpc-work/.local/bin` and uv's Python under hpc-work (the setup above),
  since the system `python3` is 3.6 and `./ri` itself re-runs under uv's
  Python.
