# Where an evaluation's bytes go

An evaluation writes about 2.4MB and deletes most of it again: a ~1MB
Measurement Set, five ~74KB FITS images, WSClean's reordered visibility file,
and the record. Three of those writes were candidates for being in the wrong
place. Exactly one of them was.

The one that was is worth **-6.5ms per evaluation** (28% of the simulate
stage): the simulator assembled its Measurement Set on the meqtrees container's
*private* `/dev/shm` and then moved it to the run's *shared* scratch tmpfs -
which is a different mount, so the move was a full cross-device copy rather than
a rename. The other two - WSClean's `-temp-dir` and its FITS output - are free
where they are, and this document records the measurements that say so, because
the rig that first said otherwise was wrong in a way worth remembering.

Everything below was measured on the current tree, 29 August 2026, on the host
described in
[docs/nested-sampling-power-limit.md](nested-sampling-power-limit.md).

## The Measurement Set was copied between two tmpfs mounts

Iteration 12 stopped the Measurement Set from ever reaching the disk
(`NS_SCRATCH_DIR`, a host tmpfs bind-mounted into all three containers - see
the section on it in
[docs/nested-sampling-throughput.md](nested-sampling-throughput.md)). What it
left behind was a copy between two pieces of RAM.

`simulate()` assembles the MS in a `tempfile.TemporaryDirectory(dir=SCRATCH_ROOT)`
and moves the finished tables to their destination at the end. `SCRATCH_ROOT` is
`/dev/shm`, which inside a container is the private tmpfs Docker gives it
(`--shm-size 512m` here). The shared scratch is a *bind mount* at
`/dev/shm/ri-ns-scratch-<pid>`. Those are different filesystems:

```
$ docker exec <meqtrees sidecar> python3 -c \
    "import os; print(os.stat('/dev/shm').st_dev, os.stat('/dev/shm/ri-ns-scratch-1234').st_dev)"
58 26
```

`shutil.move` tries `os.rename` first, gets `EXDEV`, and falls back to
`copytree` + `rmtree`. So every evaluation copied its whole MS across that
boundary. Timed in the meqtrees image against a real 984,114-byte, 81-entry MS,
30 repetitions each:

| move | median | min | max |
| --- | --- | --- | --- |
| to the shared scratch (cross-device copy) | 1.60 ms | 1.56 | 2.31 |
| within one tmpfs (rename) | **0.01 ms** | 0.01 | 0.01 |

The fix is to assemble the MS in the destination directory when the destination
is already the shared tmpfs - it is as fast to write (both are RAM) and the
closing move becomes the rename. `scratch_root_for()` in
`simulate_point_source_ms.py` makes that choice; `scripts/lib/start-sidecars.sh`
now passes `NS_SCRATCH_DIR` into the containers' environment as well as
bind-mounting it, which is how the simulator knows. With no shared scratch (a
self-check, a host with no writable `/dev/shm`) nothing changes.

### What it is worth in a real search

Three interleaved pairs, `./ri search wsclean --nlive 100 --num-repeats 10
--max-ndead -1 --mpi-procs 20` (19 workers), each arm ~110s, the first 20
seconds of each discarded (the power limit's burst window - see
[docs/nested-sampling-power-limit.md](nested-sampling-power-limit.md)). The two
arms are the same working tree built into two image tags, alternated with
`MEQTREES_IMAGE` and `--no-build`:

| pair | simulate, before | simulate, after | change | evaluations/s |
| --- | --- | --- | --- | --- |
| 1 | 23.35 ms (n=5742) | 16.76 ms (n=5917) | **-6.59 ms** | - |
| 2 | 23.38 ms (n=5764) | 16.95 ms (n=5800) | **-6.43 ms** | 65.19 → 65.56 |
| 3 | 23.15 ms (n=5907) | 16.95 ms (n=5819) | **-6.20 ms** | 65.42 → 65.92 |

Medians move the same way (21.81 → 15.87, 21.77 → 16.06, 21.59 → 16.05).
Against a ~294ms evaluation at 19 workers -6.4ms is **+2.2% evaluations per
second**; the end-to-end column reads +0.6% and +0.8%, which is what a ~4%-noise
measurement does with a 2% effect. The stage column is the measurement here -
see the last section.

The 6.5ms is four times the 1.60ms measured serially, which is the direction
iteration 10 established: an isolated rig understates a simulate-side cost
because in a real search that stage contends with 19 concurrent `wsclean`
processes rather than with itself.

### It is the same Measurement Set

The MS is byte-identical, which is the whole claim - only the directory it is
assembled in changed. Running the simulator from both images over the same argv
(a real evaluation's, `--seed 1973691575`):

```
$ diff -r -x simulation.json <old>/e1 <new>/e1 && echo IDENTICAL
IDENTICAL
```

and `simulation.json` matches field for field once the `measurement_set` path is
dropped. Both arms' searches also show the same failure population (20 of 7102
and 14 of 7160, all `wsclean failed with exit 255` bar a couple of simulate
failures) - no new failure class.

## WSClean's temporary files are already free

WSClean reorders the MS into `-temp-dir` and reopens that file on every
inversion and every predict - 15 opens and 7 write-backs for the 6 major cycles
a default evaluation runs. `-temp-dir` is the evaluation directory, on the ext4
bind mount. Moving it to the shared tmpfs is worth nothing:

| | run 1 | run 2 |
| --- | --- | --- |
| temp-dir on tmpfs / temp-dir on ext4 | 0.998 | 1.004 |

(Paired design, described below; 20 concurrent `wsclean` processes, 60s,
ratio of per-worker evaluation counts.)

## Nor are the FITS images, if you keep them

A scored evaluation writes five FITS files - image, dirty, residual, model and
psf, ~74KB each - and keeps three of them (the record's `paths` name image,
dirty and residual; `./ri render` turns them into PNGs). Building them in the
tmpfs instead is worth nothing either:

| | run 1 | run 2 |
| --- | --- | --- |
| all five on tmpfs / all five on ext4 | 0.999 | 0.991 |
| built on tmpfs, three copied back / all five on ext4 | 0.984 | 0.986 |

Writing them once to ext4 beats writing them to RAM and copying three back. So
there is no throughput argument for dropping the images; if they are ever
dropped it will be for the disk (five FITS is ~370KB an evaluation, i.e. ~72GB
of the ~78GB a `--nlive 500 --num-repeats 25` run is projected to occupy).

### The rig that said otherwise

A first pass at this measured `+1.75%` for moving the FITS to tmpfs, in four
paired runs out of four. It was wrong: that rig deleted the output directory
after each iteration, and a real scored evaluation never does. What it measured
was ext4's `rm -rf` of a six-file directory, not the writes:

| rig | run 1 | run 2 | run 3 | run 4 |
| --- | --- | --- | --- | --- |
| output deleted each iteration (wrong) | 1.012 | 1.012 | 1.021 | 1.024 |
| output kept, as a real run keeps it | 0.999 | 0.991 | | |

The rule from iterations 10 and 12 - *match the rig's mix of work to the run's*
- extends to the run's cleanup, not just its concurrency.

## More ranks than threads buys nothing

Rank 0 is not a worker and PolyChord's own sampling costs microseconds a call,
so `--mpi-procs` = `nproc` + 1 would put one worker on every hardware thread and
leave rank 0 to scavenge. It does not pay:

| ranks | workers | evaluations/s |
| --- | --- | --- |
| 20 | 19 | 64.55, 65.09, 65.41, 64.54, 66.62 |
| 21 | 20 | 66.19, 66.09, 64.00 |
| 22 | 21 | 65.25 |

Interleaved 20/21 pairs read +1.0% and -0.8%. That is the power limit doing what
it does: the package is pinned at 65W, so an extra busy thread lowers the
all-core clock by as much as it adds. The `--mpi-procs 20` default stands.

Worth knowing if you try this: Open MPI counts slots and refuses to launch more
ranks than the host has hardware threads -

```
There are not enough slots available in the system to satisfy the 24 slots
that were requested by the application
```

- and the run dies immediately with `run_with_retries: not retrying`. Adding
`--oversubscribe` to the `mpirun` line in `scripts/run-nested-sampling.sh` is
what lets a bigger `--mpi-procs` start at all; it is deliberately *not* in the
tree, because nothing above 19 workers was faster and the refusal is a useful
guard against a typo turning into a thrashing host.

## WSClean is already using its fast gridder

`-gridder` is worth one measurement and then never again. Ten serial runs each,
same MS, same `-scale`:

| gridder | ms per run |
| --- | --- |
| default (no flag) | 98 |
| `-gridder wgridder` | 93 |
| `-gridder wstacking` | 111 |
| `-gridder tuned-wgridder` | fails: "no appropriate kernel found" |

WSClean 3.7 already defaults to the ducc0 wgridder; naming `wstacking`
explicitly is 13% slower. The three reported phase timers (`Inversion:`,
`prediction:`, `deconvolution:`) are identical to three decimal places between
the default and `wgridder`, which is the tell.

## How to A/B on this host without fooling yourself

The first version of every measurement above ran the two arms **sequentially**,
45 seconds each, alternating. That design produced a clean, consistent,
four-out-of-four false positive: `+2.9 / +4.3 / +0.8 / +3.9%` for a change later
shown to be worth zero. Re-running the same arms in a different order put the
*baseline* ahead by the same margin. Within-arm spread at 19 workers is ~4%
peak-to-peak over minutes, which is the same size as anything worth finding.

What works is to run both arms **at the same time**:

- half the workers on arm A, half on arm B, on one host, for a fixed wall-clock
  duration; the score is each worker's completed-evaluation count;
- pair the inputs *within* the split - adjacent workers `2k` and `2k+1` get the
  same Measurement Set - or the arms differ by which inputs they happened to
  draw, which is a bigger effect than the change (an early version assigned
  input `w % 30` and read a 10% "difference" that was purely even-numbered
  Measurement Sets being cheaper);
- run it twice with the arm assignment swapped and take the geometric mean, so a
  worker landing on a P-core rather than an E-core cancels.

That design resolved 1.2% cleanly and repeatably. `/dev/shm/it14/arm5.sh` in
this iteration's scratch is the shape of it; it is not committed because the
useful part is the design, not the script.

For a change on the simulate side, though, prefer the **per-stage** column over
evaluations per second entirely: `timing.simulate_seconds` in every
`metrics.json` gave a 6.5ms effect at n≈5800 with two interleaved sequential
pairs agreeing to 0.16ms, while the end-to-end `evals/s` from the very same runs
could not separate the arms at all. This is the same conclusion iteration 11
reached about the `wsclean` binary column.

## A killed search leaks its scratch, and the host cannot clean it up

`_sidecar_remove` in `scripts/lib/start-sidecars.sh` removes `NS_SCRATCH_DIR` on
`EXIT`/`INT`/`TERM`, and for a run that finishes normally that directory is
already empty - every evaluation deleted its own contents as it scored. A run
killed mid-flight is different: it leaves the Measurement Sets of everything
that was in flight, ~25MB a time, and **`rm -rf` from the host account cannot
remove them**, because the containers create them as root. The trap's `rm -rf`
fails silently and the run's directory stays in `/dev/shm` for good; ten killed
searches in one profiling session cost 244MB of RAM here.

A host-side reaper cannot fix this - it is a permissions problem, not a
bookkeeping one - and doing it properly means a container run as root at every
search start, which is more machinery than a 25MB leak on a 32GB tmpfs is worth.
Clean up by hand instead:

```
docker run --rm -v /dev/shm:/hostshm --entrypoint sh ri-reproducibility/meqtrees:kern-10 \
  -c 'for d in /hostshm/ri-ns-scratch-*; do rm -rf "$d"; done'
```

## Still on the disk, not measured

`polychord_r2d2.py` writes `r2d2_data.mat` into the evaluation directory on ext4
and `prune_evaluation_artefacts` deletes it again when the evaluation scores -
the same write-then-delete shape that turned out to cost something in the FITS
rig above. The R2D2 PoC is not the default search and no A/B of it was run here,
so it is left alone; `evaluation_scratch_dir()` is already the one-line home for
it if anyone measures it.
