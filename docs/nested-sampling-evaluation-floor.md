# Where the floor is, and the one lever above it

Thirty rounds of profiling have taken a WSClean evaluation from ~2.3s to ~143ms
at production concurrency. This page is the thirty-first round's answer to "what
is left": a refreshed budget measured on the current tree, six more avenues
closed with numbers, and the two levers that remain - both of which cost
something other than engineering.

The headline is that **`-mgain` is now a flag** (`./ri search --mgain 0.9`,
`NS_WSCLEAN_MGAIN`). It is the only measured, double-digit, harness-side lever
left, and until this iteration taking it meant editing `common.py`.

Everything below was measured on 29 August 2026 on the tree at HEAD.

## The budget, by worker count

Same seed, same parameter space, `--nlive 60 --max-ndead -1`, from
`./ri profile <run>`:

| | 1 worker | 9 workers | 19 workers |
|---|---:|---:|---:|
| simulate (MeqTrees) | 5 ms | 11 ms | 12 ms |
| `wsclean` binary | 40 ms | 91 ms | 128 ms |
| zygote round trip | 1 ms | 0.4 ms | 2 ms |
| metrics | 1 ms | 1 ms | 1 ms |
| **accounted** | **46 ms** | **104 ms** | **143 ms** |
| unaccounted (PolyChord + idle) | 1.9% | 6.6% | 5.3% |
| evaluations/second | 21.7 | 81 | **126** |

Two things to read off it:

- **The unaccounted share is 5.3% at production concurrency, and 1.9% of it is
  the harness's own Python.** The premise this whole line of work started from -
  "almost half the time is idling or PolyChord" - was true before iteration 1's
  switch to asynchronous MPI and has not been true since. There is no idle time
  left to find; a rank spends 95% of the run inside one of the three stages.
- **126 evaluations/second at 19 workers**, against the 62 the
  [power-limit doc](nested-sampling-power-limit.md) measured before iterations
  18-30. Any table in this repo quoting evals/s from before iteration 18 is
  half of what the tree does now.

## Inside the binary

`./ri profile i31-r20 --phases`, 2956 evaluations at 19 workers, 123.2 ms of
logged work per evaluation:

| share | n/eval | phase |
|---:|---:|---|
| 36.1% | 8.59 | gridding |
| 23.1% | 6.59 | degridding (predict) |
| 7.7% | 0.87 | fitting the restoring beam to the PSF |
| ~10% | 6.4 | Radler's minor loops |
| ~8% | - | reorder, reordered-part opens, "loading data in memory" |
| 3.7% | 1.00 | process start to `=== IMAGING TABLE ===` |
| 1.9% | 0.88 | rendering the restored image |

59% is ducc0. [The gridder floor](nested-sampling-gridder-floor.md) prices that
as arithmetic (57% gridding proper, 23% FFT, 11% corrections, 3% index) and the
pass counts as the only handle on it - which is what `-mgain` moves.

## `-mgain`, measured again, as a flag

Two 10-rank searches at the same seed started **simultaneously**, one arm per
`-mgain`:

| | passes/eval (grid + predict) | logged work/eval | evaluations/second |
|---|---:|---:|---:|
| `--mgain 0.8` (default) | 8.52 + 6.52 | 125.3 ms | 60.0 |
| `--mgain 0.9` | 6.73 + 4.73 | 107.2 ms | 68.2 |

The pass counts are the clean signal - **6.5 major cycles down to 4.7**,
exactly what [the clean-loop doc](nested-sampling-clean-loop.md) predicted - and
they are what makes the effect trustworthy at a run length where evals/s still
carries the faster arm finishing first and getting the host to itself.

The default stays 0.8. It is part of the experiment definition every archived
run was scored under, and while it is result-preserving for the default
`total_rms_jy` objective (1e-7 median), it is not for `peak_flux_abs_error_jy`
or `sigma_res`. `./ri search --mgain 0.9` is now how a run that wants the
throughput asks for it, `run.env` carries it through a resume, and
`summary.json`'s `wsclean_fixed_hyperparameters` records what was used.

## Six avenues closed this round

Each of these looked worth a patch and priced out below the noise floor. They
are recorded so the next round does not re-measure them.

### The allocator is not the problem

A whole cold `wsclean` process takes **4511 minor page faults** and 23 major
ones (`/usr/bin/time -v`), i.e. ~2 ms, most of it the dynamic linker. Raising
`MALLOC_MMAP_THRESHOLD_` so the ~1.1 MB per-pass grids come off the heap
instead of fresh `mmap`s cannot be worth more than that, and the forked zygote
child pays only the part after `main()`.

### Syscalls are not the problem either

`strace -c -f` over the same run: **3618 calls, 12 ms of system time under
ptrace**, of which 504 `openat` (228 failing - the linker's search path) and
399 `mmap` are start-up. `System time` off `/usr/bin/time -v` is 20 ms of a
110 ms cold process and almost all of it is `execve` plus 73 shared objects,
which is exactly what [the zygote](nested-sampling-wsclean-zygote.md) already
pays once per rank.

### `wsclean -j 1` still starts 19 threads, and they are idle

A `wsclean` process clones **19 threads in one burst**, immediately after
`== Constructing PSF ==` - `GriddingTaskManager::RunDirect` sizes
`MSGridderManagerScheduler` from `resources.NCpus()` rather than from `-j`.
They are created once (the scheduler is cached), they do no work (the process
runs at 84% of one CPU), and the burst is ~1.1 ms *under strace*. Neither
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `DUCC0_NUM_THREADS=1`,
`--cpuset-cpus`, nor an `LD_PRELOAD` that makes `sysconf(_SC_NPROCESSORS_ONLN)`
return 1 changes the count, so removing it means a sixth WSClean patch for
under 0.5% of an evaluation. Not taken.

### Half the DATA column is never read

`sim.ms` carries **four correlations** (`CORR_TYPE` 9-12, RR/RL/LR/LL) because
that is what makems' bundled VLA-A example builds, and `-pol I` only ever uses
RR and LL. Dropping the cross-hands would be bit-identical *if* the noise were
still drawn over the full 4-correlation shape and then sliced (the RNG stream
is what fixes the column - see `simulate_point_source_ms.py`). It is not worth
the change: [`patches/0005`](nested-sampling-row-blocks.md) already reads DATA
in blocks at 0.053 ms per 2106 rows, so halving it saves ~0.03 ms of a 143 ms
evaluation, and the skeleton's DATA column would have to be rebuilt with a new
shape at cache-build time.

### The power limit is genuinely out of reach

[The power-limit doc](nested-sampling-power-limit.md) says the 65W PL1 costs
~26% of the evaluations per second and needs root. Docker looks like a way
around that - `docker run --privileged -v /sys:/hostsys` - and it is not:
**Docker on this host is rootless**, so the container's `root` maps to the
unprivileged host user. Reading
`/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` from a `--privileged`
container returns `Permission denied`. The limit is still the box owner's
decision and still cannot be taken from this account.

### The harness's own Python is 0.9 ms

A `--mpi-procs 1` search accounts for **98.1%** of its wall clock in the three
stages, so everything the likelihood wrapper does outside them - the parameter
hash, the evaluation directory, `simulation.json`, `write_json_atomic` (no
`fsync`), the pruning `unlink`s, the JSON line to the progress bar - is 0.9 ms
an evaluation together. There is nothing here to optimise, and it is the
control that says the 5.3% unaccounted at 19 workers is MPI and scheduling
rather than harness work.

## What is actually left

In descending order of size, with what each costs:

| lever | worth | cost |
|---|---:|---|
| raise the host's 65W PL1 to its rated 117W | +26% | root on the box; not reachable from this account |
| drop w-gridding | +29% on the binary | not result-preserving: 5.4e-3 rad of ignored w-phase at the image corner ([gridder floor](nested-sampling-gridder-floor.md)) |
| `--mgain 0.9` | +14% to +20% | `peak_flux_abs_error_jy` and `sigma_res` move; changes the experiment definition |
| `-wgridder-accuracy 1e-2` | +13.8% | 10 mJy of gridding error against a `log10_dynamic_range` that reaches 1e6 |
| fix schaapcommon's unsigned beam-fit bug | ~4% | changes the restoring beam ([process warm-up](nested-sampling-process-warm-up.md)) |

Every one of them is a decision about the experiment or the machine, not an
engineering task. **The harness-side mining is finished**: what is left inside a
143 ms evaluation is 59% ducc0 arithmetic, 8% a Gaussian fit that has to happen,
10% the minor loop, and ~20% of metadata and I/O that five patches and two
zygote warm-ups have already been through.

## Reproducing any of this

```sh
# the budget table
./ri search wsclean --nlive 60 --max-ndead -1 --mpi-procs 20 --seed 11 \
  --output-dir results/nested-sampling/floor-r20
./ri profile floor-r20              # stage budget
./ri profile floor-r20 --phases     # inside the binary

# the mgain pair, simultaneously (sequential arms lie by ~4% on this host)
./ri search wsclean --nlive 60 --max-ndead -1 --mpi-procs 10 --seed 11 \
  --mgain 0.8 --output-dir results/nested-sampling/mg08 &
./ri search wsclean --nlive 60 --max-ndead -1 --mpi-procs 10 --seed 11 \
  --mgain 0.9 --output-dir results/nested-sampling/mg09 &
wait

# one evaluation's page faults, syscalls and threads (needs a kept sim.ms:
# ./ri search wsclean --keep-measurement-sets ...)
docker run --rm --cap-add SYS_PTRACE --entrypoint sh \
  -v /dev/shm/probe:/dev/shm/probe -w /dev/shm/probe \
  ri-reproducibility/wsclean:v3.7 -c 'apt-get update -qq >/dev/null &&
    apt-get install -y -qq strace >/dev/null &&
    strace -c -f wsclean <the argv out of metrics.json "commands"."wsclean">'
```
