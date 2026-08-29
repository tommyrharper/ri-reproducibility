# Making a nested-sampling search faster: the index

Start here. Thirty-two rounds of profiling took a WSClean likelihood
evaluation from ~2.3 s to ~143 ms at production concurrency, and the host from
~62 to **126 evaluations/second at 19 workers**. Each round left a page with
its own measurements; this page says what shipped, what it was worth, and
which page holds the evidence.

Two things to know before reading any number below:

- **The percentages do not add up, and are not meant to.** Each was measured
  against the tree as it stood at that iteration, so a later 3% is 3% of an
  already smaller evaluation. The only end-to-end figures are the two in the
  paragraph above.
- **Any evaluations/second figure written before iteration 18 is about half of
  what the tree does now.** Where an older page still quotes one, it is the
  measurement that page made at the time, not a current rate.

## What shipped

Roughly in the order it was found, which is also roughly largest-first.

| change | where it lives | measured worth |
|---|---|---|
| PolyChord's asynchronous MPI mode by default | `polychord_wsclean.py` | -27-33% wall clock per dead point ([throughput](nested-sampling-throughput.md)) |
| WSClean built for the host CPU, `x86-64-v3` default | `WSCLEAN_TARGET_CPU` | +8.6% evals/s |
| MeqTrees no longer predicts a constant | `point_source_forest.py` | +16-20% evals/s |
| `WEIGHT`/`SIGMA` columns no longer written | `simulate_point_source_ms.py` | +3.8% evals/s |
| Measurement Set built in a shared host tmpfs | `NS_SCRATCH_DIR` | -1.2 ms/eval ([I/O placement](nested-sampling-io-placement.md)) |
| Cross-device Measurement Set copy removed | `scratch_root_for()` | -6.4 ms/eval, ~+2.2% evals/s |
| Six unread MS subtables dropped | `simulate_point_source_ms.py` | +14.9% evals/s ([MS open](nested-sampling-ms-open.md)) |
| `FEED` dropped too | same | -3.2% on the `wsclean` binary |
| Antenna names cached in-process | `patches/0001` | +10.5% evals/s ([patches](nested-sampling-wsclean-patches.md)) |
| `wsclean-zygote` fork server | `docker/wsclean/src/zygote.cpp` | +8.4% evals/s ([zygote](nested-sampling-wsclean-zygote.md)) |
| `-data-column DATA` | `polychord_wsclean.py` | -1.0% on the binary ([cost model](nested-sampling-cost-model.md)) |
| Dead work cut from the simulate stage | `simulate_point_source_ms.py` | -20% of that stage ([simulate](nested-sampling-simulate-stage.md)) |
| One parent-MS open shared five ways | `patches/0003` | -3.7% on the binary ([shared open](nested-sampling-shared-ms-open.md)) |
| FFTW planner warmed in the zygote parent | `zygote.cpp` | -6.9% on the binary ([FFTW](nested-sampling-fftw-planner.md)) |
| `schaapcommon` FFTW plan cache | `patches/0004` | -2.7% on the binary ([gridder floor](nested-sampling-gridder-floor.md)) |
| Measurement Set read in row blocks | `patches/0005` | -2.2% on the binary ([row blocks](nested-sampling-row-blocks.md)) |
| cfitsio and casacore init moved to the zygote parent | `zygote.cpp` | -2.39 ms/eval ([warm-up](nested-sampling-process-warm-up.md)) |

And two that bought run *size* rather than speed, which is what the speed was
for - see [disk footprint](nested-sampling-disk-footprint.md) and
[run scaling](nested-sampling-run-scaling.md):

| change | measured worth |
|---|---|
| `sim.ms` deleted as the evaluation is scored | evaluation directory 3.4x smaller |
| WSClean's four unread FITS images pruned | another 3.94x - 393.6 KB to 99.9 KB, so this host's ceiling went from 477k evaluations to 1.88M |
| A resume keeps each adopted objective, not each record | 62 GB down to 5.4 GB, which is the difference between the target run finishing and being OOM-killed |
| Progress-bar redraw backed off to 9x its own cost | 44% of a core down to ~12%, and no longer growing with the run |

## What is priced but deliberately not taken

| lever | worth | why not |
|---|---|---|
| `-mgain 0.9` | +20% evals/s | Not result-preserving for `peak_flux_abs_error_jy` or `sigma_res`, and it is the experiment definition every archived run was scored under. Now `./ri search --mgain 0.9`, default still 0.8. [clean loop](nested-sampling-clean-loop.md) |
| Dropping the w-gridding cube | -29% on the binary | **Closed.** The ignored-`w` phase error exceeds ducc0's own 1e-4 epsilon on 5962 of 5962 evaluations, so no lossless per-evaluation rule can skip it. [run scaling](nested-sampling-run-scaling.md) |
| Raising the 65W RAPL package limit | ~+26% evals/s | **Closed.** Docker here is rootless, so `--privileged` still maps to an unprivileged user and cannot write it. [power limit](nested-sampling-power-limit.md) |

## Where the remaining time goes

126 evals/s at 19 workers, 143 ms an evaluation, 5.3% unaccounted - of which a
`--mpi-procs 1` control says 1.9% is the harness's own Python. **There is no
idle time left to find.** The premise this work started from, that almost half
a run is idling or running PolyChord, was true before iteration 1 and has not
been true since; PolyChord itself is 3-7 us per likelihood call at every
`nlive`.

59% of an evaluation is ducc0's gridding and degridding, and
[the gridder floor](nested-sampling-gridder-floor.md) prices that as
arithmetic. The pass count is the only handle on it, which is what `-mgain`
moves. See [the evaluation floor](nested-sampling-evaluation-floor.md) for the
current budget and the six further avenues closed with numbers.

## Reading your own run

No rig, no patched tree, no replay - every run already carries this:

```bash
./ri profile <run>              # per-stage budget and worker utilisation
./ri profile <run> --phases     # inside the wsclean binary, from its own -log-time output
./ri profile <run> --over-time  # why evaluations/second falls as a run goes deeper
```

That last one is the answer to "inconsistent throughput": an evaluation costs
`70.7 ms + 4.58 us x visibilities` and nested sampling compresses towards the
long-observation, many-channel corner of the parameter space, so a run's rate
falls monotonically with nothing degrading. See
[the cost model](nested-sampling-cost-model.md).

## The rest of the pages

Chronological, because each one starts where the last stopped.

| page | what it settles |
|---|---|
| [throughput](nested-sampling-throughput.md) | The running log of iterations 1-14: idle ranks, the asynchronous-MPI switch, rank 0, the AVX2 build, the constant predict, the two constant columns, the scratch tmpfs. Long, and the older figures in it are superseded. |
| [run scaling](nested-sampling-run-scaling.md) | What a bigger run costs. A 10x `--nlive` costs nothing per evaluation; the ceiling on this host is ~850k evaluations, set by rank 0's memory while it builds `summary.json`. |
| [power limit](nested-sampling-power-limit.md) | The host's 65W RAPL cap, why it is not thermal, and the 8-second averaging window behind the burst-clock measurement trap. |
| [evaluation budget](nested-sampling-evaluation-budget.md) | The first production-concurrency decomposition, and the flag screen that closed WSClean's own knobs. |
| [I/O placement](nested-sampling-io-placement.md) | Where an evaluation's 2.4 MB goes, and why A/B arms must run *simultaneously* - sequential ones give consistent ~4% false positives on this host. |
| [clean loop](nested-sampling-clean-loop.md) | `-mgain`, and its exact science cost across all eight metrics on 600 real evaluations. |
| [MS open](nested-sampling-ms-open.md) | casacore re-attaching every subtable on each of the ~16 parent-MS opens, and why the avenue is closed from the simulator's side. |
| [WSClean patches](nested-sampling-wsclean-patches.md) | How `docker/wsclean/patches/` works - `git apply` in the Dockerfile, hashed into `ri.build-inputs`, and a `WSCLEAN_GIT_TAG` bump that breaks a patch fails the build on purpose. |
| [zygote](nested-sampling-wsclean-zygote.md) | The fork server: 27 ms of every 163 ms process ran before `main()`. Also why an archived run's `image_binary` column is ~5 ms low against a run made since. |
| [phase profile](nested-sampling-phase-profile.md) | `-log-time` by default, and `./ri profile --phases`. Read it on *medians*. |
| [cost model](nested-sampling-cost-model.md) | Why throughput falls through a run, and `--over-time`. |
| [simulate stage](nested-sampling-simulate-stage.md) | The MeqTrees half, and the confirmation that nothing measurable is left in it. |
| [shared MS open](nested-sampling-shared-ms-open.md) | `patches/0003`, counted with a 30-line `LD_PRELOAD` shim over `open64` - no rebuild, no profiler. |
| [FFTW planner](nested-sampling-fftw-planner.md) | 63 transform plans built and thrown away per process. Note the four warmed sizes derive from `DEFAULT_IMAGE_DIM = 128`; a stale list costs the speedup, not a result. |
| [gridder floor](nested-sampling-gridder-floor.md) | ducc0's own timer tree, and `patches/0004`. Two arms' FITS files never byte-compare equal - the header carries the `-name` path, so compare data blocks. |
| [row blocks](nested-sampling-row-blocks.md) | `patches/0005`: 29484 casacore column reads an evaluation down to 27. |
| [process warm-up](nested-sampling-process-warm-up.md) | The 5 ms before the first visibility - and the unsigned-arithmetic bug that makes WSClean fit its restoring beam on a quarter of the box it means to use. A real failure mode, not result-preserving to fix. |
| [disk footprint](nested-sampling-disk-footprint.md) | Bytes rather than clock. A run is `~17 x nlive x num_repeats` evaluations, and `evaluations/` needs no sharding. |
| [evaluation floor](nested-sampling-evaluation-floor.md) | The current budget, `--mgain` as a flag, and the last six avenues closed. |

## If you are picking this up

Read [the evaluation floor](nested-sampling-evaluation-floor.md) first - it is
the most recent budget - then this page's "priced but not taken" table. The
measurement discipline that made these numbers trustworthy is worth copying:
run A/B arms **simultaneously**, compare FITS **data blocks** rather than
files, read `--phases` on **medians**, and discard the first ~8 seconds of any
burst (the power limit's averaging window).
