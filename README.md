# ri-reproducibility

**PolyChord searches this repo's R2D2-RI and WSClean parameter spaces for failure modes**; container images, smoke tests, and pinned revisions keep searches runnable and results trustworthy.

**This is the `cluster` branch**: it runs on Cambridge's CSD3 under Slurm and
Apptainer, without Docker. [`docs/cluster.md`](docs/cluster.md) is the
authority for everything cluster-specific; `main` is the Docker branch.

**Which R2D2:** the U-Net incarnation (`series: R2D2`, `layers: 1`), from R2D2-RI
v2.0 - checkpoints `R2D2_A1_T2_Realisation1`, 25 terms. `A1` = U-Net (`A2` =
U-WDSR), `T2` = the generalized training set. That model is described in
[arXiv:2503.02554](https://arxiv.org/abs/2503.02554), *not* in the older
[arXiv:2403.05452](https://arxiv.org/abs/2403.05452) vendored under
`r2d2-paper/`, which is where `claims/` and `latex/` take their metric
definitions and published numbers from. Do not compare search output against
`claims/r2d2_claims.md` as if it were the same model.

`./ri` is the front door - one command for every part of that:

```bash
./ri --help            # the whole surface, and --help on every subcommand
./ri images import     # the four images, from archives built on a Docker host
./ri search wsclean    # run a search
```

It is a thin dispatcher over `scripts/`, so anything it does can also be
run by hand, and `./ri --dry-run <command>` prints exactly what it would run.

Science write-up: `latex/notes.tex`. Operational detail:
[`docs/nested-sampling.md`](docs/nested-sampling.md). Command index:
[`docs/cheatsheet.md`](docs/cheatsheet.md).

Read-only reference material: `r2d2-paper/` defines reused R2D2 metrics and
`claims/` records published WSClean and R2D2 numbers. Neither is a
reproduction target.

## 1. Running a search

```bash
export SBATCH_ACCOUNT=MYPROJECT-CPU   # or --account; `mybalance` lists yours
./ri search wsclean     # WSClean search, submitted as a Slurm job
./ri search r2d2        # R2D2 search (needs checkpoints, section 5)

./ri search wsclean --nlive 20 --num-repeats 5 --max-ndead 20
./ri search r2d2 --metric sigma_res
NS_NLIVE=20 ./ri search wsclean    # same thing; every flag has a variable
```

On a login node `./ri search` and `./ri resume` submit themselves as a job
named after the run (`squeue -u $USER`); inside an allocation, or on a host
without Slurm, they run in place. `--account`, `--partition` and `--time` (or
any `SBATCH_*` variable) shape the job; see
[`docs/cluster.md`](docs/cluster.md).

Output lands in `results/nested-sampling/<tool>-vlaa-<UTC>/`: one
directory per likelihood evaluation (the MS, the reconstruction FITS, and
`metrics.json`), plus a run-level `summary.json`.

Two runs of the same shape can be combined, and the profiler breaks a
finished run down per stage:

```bash
./ri merge results/nested-sampling/A results/nested-sampling/B
./ri profile results/nested-sampling/<run>
```

Every finished search also adds a row to `benchmarks.jsonl`, so what this
commit costs on this machine can be compared with what the last one did:

```bash
./ri bench                          # the table, per commit and machine
./ri bench run wsclean --repeat 3   # the controlled measurement
```

## 2. Reading the results

```bash
./ri tui                # runs, live health, the profile, the benchmark table and a new-run form (needs Go)
./ri report             # all runs
./ri report --last 1    # newest run only
./ri serve              # read the report from a browser on another machine
./ri plot gui           # interactive corner plots (needs a display)
./ri plot likelihood    # R2D2 vs WSClean, overlaid both ways and side by side
./ri plot likelihood --last   # ...for the last two directly comparable runs
```

`./ri plot likelihood` (`scripts/plot-merged-likelihood-compare.py`) writes
merged-failure-score figures into `reports/`, included by `latex/notes.tex`.
`--last` compares the newest R2D2 and WSClean run that agree on VLA config,
metric, parameter space and sampler effort, merged or not, and leaves those
included figures alone. Either way the pair is also kept under its own name in
`reports/likelihood-comparisons/`, which `./ri report` collects onto a page
linked from the top of the index.

## 3. Host prerequisites

On the cluster:

- Apptainer (`apptainer` or `singularity`) and Slurm (`sbatch`, `squeue`);
  CSD3 has both on every node, no module to load.
- `git`; also there already.
- [`uv`](https://docs.astral.sh/uv/) for every host-side Python: install it
  into your home (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then
  run `uv sync` once in the checkout on a login node so a job never has to
  download an interpreter or a package. CSD3's system `python3` is 3.6, too
  old for the stdlib-only `./ri` dispatcher (3.9+), so `./ri` re-runs itself
  under uv's Python when it finds an old one; no `module load` needed.

Plus, somewhere else, one machine with Docker to build the images on (section 4).

## 4. Building the images

The four environments are still defined by the Dockerfiles under `docker/`,
built on a Docker host and carried to the cluster as SIF files:

```bash
# on a Docker host (x86-64, to match the cluster):
./ri build              # all four; unchanged inputs skip `docker build`
./ri images export      # docker save -> images/archives/<name>.tar
rsync -avz images/archives/ login-cpu.hpc.cam.ac.uk:<repo>/images/archives/

# on the cluster, in the same checkout:
./ri images import      # apptainer build images/<name>.sif
```

`./ri search` builds nothing and refuses to start without its SIFs. Editing
`scripts/lib/nested_sampling/` needs no rebuild (a run binds the working tree
over the baked copy); a Dockerfile, patch or `[[parameter_space]]` change
means build, export and import again.

WSClean defaults to portable `x86-64-v3`; `--native` tunes it for the build
host's CPU, which is only right if that is the compute node's CPU. See the
[throughput guide](docs/nested-sampling-throughput.md) for CPU targets.

### Does the imager under test actually run?

```bash
./ri smoke              # both
./ri smoke wsclean      # wsclean --version + a real tiny imaging run
./ri smoke r2d2         # imports -> app modules -> bundled data load ->
                        # config validation -> (real inference if
                        # checkpoints are present)
./ri smoke ms-to-mat    # the MS -> R2D2 .mat bridge, before an R2D2 search
```

These verify that images can run their workloads. Run them after an import,
before searches. `./ri plot fits` renders their FITS output (or supplied
paths) to PNG using the r2d2 image's astropy + matplotlib.

## 5. Fetching R2D2 checkpoints

```bash
./ri fetch-checkpoints R2D2_A1_T2_Realisation1.zip
```

**This cannot be fully automated.** The checkpoint host
(`researchportal.hw.ac.uk`) serves files behind a Cloudflare bot
challenge that rejects `curl`/`wget` (HTTP 403, verified 2026-08-03). The
script attempts the download, detects that failure precisely, and prints
the direct URL plus exact placement instructions instead of a stack
trace. See `checkpoints/README.md`. The R2D2 search needs
`checkpoints/R2D2_A1/R2D2_UNet_N<k>.ckpt`.

## 6. Binds

`scripts/*.sh` bind these into the containers (`apptainer --bind`); the host
side is set in `defaults.toml` and overridable in the environment:

| Host path (default) | Container path | Purpose |
|---|---|---|
| `./data` | `/data` | Measurement Sets, `.mat` files, ground-truth FITS |
| `./checkpoints` (`CHECKPOINTS_DIR`) | `/checkpoints` | R2D2 pretrained DNN checkpoints |
| `./results` (`RESULTS_DIR`) | `/results` | Nested-sampling runs, smoke-test output |
| `./reports` | (host-side only) | Run manifests and the generated HTML report |

Put the checkout, with `images/` and `results/`, on `hpc-work` (Lustre, shared
by every node), not in the 50GB home; `docs/cluster.md` has the layout.

None of these are baked into an image layer or committed to Git (see
`.gitignore`) - **with one documented exception**: R2D2-RI's own ~100 MB
bundled example (`data/data_3c353.mat`, `data/3c353_gdth.fits`) ships
inside the upstream repository, so cloning it at build time unavoidably
bakes those two files into the `r2d2` image layer. That is upstream's
packaging decision. See `data/README.md`.

## 7. Architecture and CPU-only notes

- Images are built for the Docker host's architecture; the cluster is x86-64,
  so build there or set `DOCKER_DEFAULT_PLATFORM=linux/amd64` to cross-build
  (slow, under QEMU).
- Imagers are CPU-only. No GPU is required or used.

## 8. How upstream revisions are pinned

`versions.env` is the source of truth for upstream URLs, revisions, packages,
and deliberately unpinned casacore data. Update matching Dockerfile defaults,
rebuild, and commit both when changing it.

## 9. Reproducibility limitations

Pinned revisions do not make images bit-exact: casacore fetches unversioned
IERS/leap-second/ephemeris data at build time; its image SHA-256 is recorded at
`/opt/casacore-data/WSRT_Measures.ztar.sha256`. The bundled 3c353 example is
also baked in (section 6). Compare `wall_seconds` within runs, not across
machines or partitions.

## 10. Reclaiming disk space

```bash
./ri clean            # the SIFs, their archives + generated smoke-test outputs
./ri disk-usage       # du over images/, results/, reports/, checkpoints/, data/
```

`./ri clean` leaves `data/`, `checkpoints/`, `results/` and `reports/` alone.
`hpc-work` has a one-million-file quota as well as 1TB;
`docs/cleaning-up-old-run-output.md` covers pruning old runs.

## Troubleshooting

- **`FATAL: images/<name>.sif is missing`:** `./ri images import` after rsyncing the archives (section 4).
- **Stale code:** `scripts/lib/nested_sampling/` is bound from the checkout, so a run is always the code checked out; a Dockerfile or patch change needs build, export, import.
- **WSClean `Illegal instruction`:** the image was built `--native` on a different CPU; rebuild with the default `x86-64-v3` (`docker/wsclean/Dockerfile`).
- **Job never starts / refused:** `squeue -u $USER` and the run's `slurm-<jobid>.out`; a bad `--account` fails at submission and removes the claimed run directory.
- **Build OOM / disk (Docker host):** `BUILD_JOBS=1 ./ri build wsclean` (it otherwise compiles at `nproc`); the archives total ~2.1GB and the SIFs the same again.
