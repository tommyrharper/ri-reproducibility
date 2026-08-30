# Tracking throughput across commits

[The speed index](nested-sampling-speed.md) records what each change was worth
when it was made. This is the other half: a table that says what this commit
costs *now*, on *this* machine, so the next change can be shown to have helped
and the one after that cannot quietly give it back.

```bash
./ri bench                       # the table
./ri bench run wsclean --repeat 3   # synchronous, fixed-seed comparison
./ri bench run wsclean --preset throughput --repeat 3  # production async mode
./ri bench run wsclean --preset production --repeat 15 # target-scale workload
./ri bench run wsclean --preset production --timeout 300 # bounded probe
./ri bench run wsclean --native --repeat 3       # host-specific WSClean build
./ri bench run wsclean --preset throughput --mpi-procs 16 --repeat 3  # rank-scaling probe
./ri bench run wsclean --preset throughput --interleave-mpi-procs 19 20 --repeat 3  # paired rank A/B
./ri bench run wsclean --preset throughput --interleave-mpi-procs 20 21 --allow-oversubscription --repeat 3  # explicit oversubscription probe
./ri bench run r2d2 --preset throughput --omp-threads 4 --repeat 3  # thread-count probe
```

`b` in `./ri tui` shows the same table.

## The loop

1. Commit the change. A row carries the commit it was measured at, and a row
   measured on a dirty tree is marked with a trailing `+` - useful while
   iterating, worthless as a reference.
2. `./ri bench run <imager> --repeat 3`. That is one unrecorded warm-up search
   and three recorded ones: for WSClean, about 90 seconds and 270 MB of
   `results/`.
3. `./ri bench`. The new commit is the leftmost column, with `Δ evals/s`
   against the one behind it.
4. Keep it or revert it. Either way the rows stay, and the next change is
   measured against what is now there.

Every search that finishes adds a row, not only `./ri bench run` - an ad-hoc
`./ri search` lands in a `custom` group beside the controlled one, which is
where a run at settings you were exploring rather than benchmarking belongs.

`--native` records `WSCLEAN_TARGET_CPU=native` in `run.env` and the row's
settings, so host-specific binaries cannot be mixed with portable-build rows.
On this host, three native default-preset repeats measured 70.2 eval/s median
versus 70.6 eval/s for the portable baseline: no measurable speedup.

`--mpi-procs N` overrides host-derived worker count for rank-scaling probes;
the count is recorded in row settings, keeping results in separate groups.
Explicit values above the process affinity are rejected before starting a
search, avoiding Open MPI slot errors and wasted probes.

`--omp-threads N` overrides R2D2's automatic per-rank thread count and is
recorded in row settings, keeping thread-count probes in separate groups.

`--interleave-omp-threads A B` alternates two R2D2 thread counts after one
unrecorded warm-up; `--repeat N` runs N measured searches per arm. This keeps
small thread-count differences paired against host drift.

`--interleave-mpi-procs A B` does the same for MPI worker counts. It rejects
counts above available CPU affinity before launching the warm-up. Add
`--allow-oversubscription` only for an explicit probe of counts above that
limit; it does not change search defaults or rank budgeting.

## What a row is

One JSON object per line in `benchmarks.jsonl`, appended by
`scripts/bench.py record` at the end of every run that finished (a resumed or
self-healed run is skipped: its wall clock covers one segment and its
evaluation count covers all of them, so the throughput it implies is fiction).

A row carries evaluations/second, peak worker memory, and the per-evaluation cost of each stage the
profiler measures - the same numbers `./ri profile` prints, so
[the profiling reference](nested-sampling-profiling.md) is what each stage
means - the commit, the machine, and the settings the run used.

Rows are grouped by **machine, imager and settings**, and only rows in one
group are ever compared. The machine id is derived from `/etc/machine-id` (the
`IOPlatformUUID` on macOS), so it survives a reboot, a rename and a fresh
checkout, and a laptop's rows can never pool with a server's. `NS_SEED` is not
part of the grouping - it changes which points are drawn, not the
configuration being measured, and it is random per run, so in the key every
ad-hoc search would be a group of one.

The `throughput` preset keeps the workload but sets `NS_SYNCHRONOUS = 0`, so it
measures the asynchronous scheduler used by production searches. Use it for
throughput work; the `default` preset remains synchronous because its fixed seed
makes repeated evaluation counts directly comparable.

The `production` preset matches target searches: `NS_NLIVE = 150`,
`NS_NUM_REPEATS = 15`, `NS_MAX_NDEAD = -1`, fixed seed, and asynchronous
scheduling. It is intentionally expensive and should be run only for final
confirmation after a cheaper preset shows a candidate.

Use `--timeout` for expensive probes. It stops the benchmark's entire process
group and returns 124, without recording a partial run, so a failed or
interrupted measurement cannot leave workers behind.

## What makes two commits comparable

`[benchmark.default.<imager>]` in `defaults.toml` pins the settings
`./ri bench run` uses. Three of them are pinned against measurement traps
rather than for realism, all measured on the 20-CPU Hetzner host:

- **A fixed seed, and `NS_SYNCHRONOUS = 1`.** A seed alone is not enough. In
  the default asynchronous mode PolyChord adopts whichever evaluation arrives
  first, so one seed gave 605, 736, 779 and 878 evaluations across four runs,
  of visibly different sizes: evaluations/second moved 112-140 without a line
  of code changing. Synchronous, the same seed evaluates the same 608 points
  every time. It costs ~45% of the throughput, which does not matter for a
  number that is only ever compared with itself - but it does mean this table
  cannot see a change to the asynchronous scheduler. That question is
  [throughput](nested-sampling-throughput.md), and it needs its own A/B.
- **A warm-up search that is not recorded.** The first search after an idle
  spell measured 76.5 evaluations/second against 70.2, 72.1 and 69.3 for the
  three behind it - the package spends a power budget it then has to pay back
  ([the power limit](nested-sampling-power-limit.md)). A cold first run is
  effectively a faster machine, so `./ri bench run` throws one away before it
  records anything.
- **`NS_MAX_NDEAD = 150`**, which is ~608 evaluations and ~8 seconds of
  measured wall clock. Not longer: at 300 the three repeats spread 2.4%
  against 2.0% at 150, because what is left is drift *between* runs, not noise
  *within* one. Longer runs would only cost disk.

Pinned in `defaults.toml` rather than inherited from the defaults above it, so
a tuning change to the search cannot silently redefine what every archived row
was measured under. Change a pinned value and the rows part company into two
groups, which is the intended behaviour: they are not comparable any more.

## Reading the table

Each cell is a median over repeats in that column, `±` an IQR-based robust
standard-error estimate. This prevents one long-tail timing from dominating
the result while the error estimate narrows as repeats accumulate. A single
row shows no error bar.

`peak memory MB` is the largest recorded imaging-worker peak in the run. It is
reported alongside speed because R2D2's per-rank memory budget limits useful
parallelism; older rows without this field simply leave the cell blank.

`Δ evals/s` is the change against the column to its right. It is starred when
the two medians are more than two combined robust standard errors apart. One repeat
each can never earn a star; three tight ones can. With the ~2% run-to-run
spread above, three repeats a side resolve a change of about 4%.

The `evals` row is a diagnostic, not a result: in a group where it moves, the
columns above it are comparing different work, and nothing there means what it
appears to.

## What this does not replace

A sequential comparison on this host carries a consistent ~4% false-positive
rate ([I/O placement](nested-sampling-io-placement.md) has the measurement),
and this table is sequential by construction - it compares a commit measured
today with one measured last week. It is a regression net at the few-percent
scale, not a way to price a 1% change. For that, two arms still have to run
*simultaneously*, which is what `WSCLEAN_IMAGE` exists for.

## Cost

A WSClean benchmark run is ~23 seconds and ~67 MB of `results/`. R2D2 images
in 14.2 seconds where WSClean takes 0.13, so its preset is sized down to 41
evaluations and a repeat still costs ~110 seconds. Nothing reads those
evaluations again once the row is written, so they are the first thing to
delete when disk gets tight.
