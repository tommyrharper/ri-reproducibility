# The WSClean fork server

**A quarter of a second is not what a WSClean evaluation costs; 27 ms of every
163 ms `wsclean` process runs before `main()` does, initialising the same 73
shared objects the last one initialised. `wsclean-zygote` pays that once per
rank and forks an already-initialised child per evaluation: +8.4% evaluations per second end to end over eight simultaneous swapped
pairs of real searches, with bit-identical images.****

Host: the same 20-thread i5-13500 every other measurement in `docs/` was taken
on, at the 65W package limit
[the power-limit doc](nested-sampling-power-limit.md) describes. 29 August 2026.

## Where the 27 ms was found

The previous four iterations of this work all measured *inside* the `wsclean`
binary, and [the evaluation budget](nested-sampling-evaluation-budget.md) is a
decomposition of what WSClean itself reports. That decomposition can only ever
add up to what the process does after it starts.

`wsclean -log-time` timestamps every output line, so the gap between the
process being `exec`ed and its first log line is measurable directly - print
`date +%s%N` on either side of the command and compare against the log's own
wall-clock stamps. Over the 160-Measurement-Set corpus below, replayed
19-way-concurrent for twelve passes so the sample lands past the 8-second RAPL
averaging window:

| span | ms |
| --- | ---: |
| `exec` -> first log line | 27.1 |
| first log line -> last log line | 137.1 |
| last log line -> exit | 4.6 |
| **total** | **168.7** |

That 27.1 ms is not the loader: `LD_DEBUG=statistics` puts `ld.so` at 0.9 ms.
It is C++ static initialisation, and `LD_PRELOAD`ing one library into
`/bin/true` prices each subtree of it (serial, so ~2.2x cheaper than the same
work at production concurrency):

| `LD_PRELOAD` | us |
| --- | ---: |
| (nothing, `/bin/true`) | 209 |
| `libcasa_casa.so.9` | 4404 |
| `libcasa_tables.so.9` | 5886 |
| `libcasa_scimath.so.9` | 6235 |
| `libcasa_measures.so.9` | 7736 |
| `libcasa_fits.so.9` | 8964 |
| `libcasa_ms.so.9` | 11404 |
| `libhdf5_serial_cpp.so.103` | 3051 |
| `libcurl-gnutls.so.4` | 1993 |
| `libcfitsio.so.10` | 2127 |
| `libpython3.11.so.1.0` | 598 |
| `libmpi.so.40` | 529 |
| `libgomp.so.1` | 180 |
| `wsclean --version`, whole binary | 14802 |

Casacore's own chain is 11.4 ms of the 14.8 ms a whole `wsclean --version`
costs, and each library's number includes its dependencies. There is nothing
surgical to remove: iteration 15 already measured dropping MPI, Python and
HDF5 from the build at ~0, ~0 and ~1.5% respectively, and casacore is not
optional. The cost is real, it is identical every time, and the search starts
~70 of these processes a second.

So stop paying it every time.

## What the zygote is

`docker/wsclean/src/zygote.cpp` is ~150 lines that link the same `wsclean-lib`
the `wsclean` binary links. After *its* static initialisers have run it reads
one request per line from stdin and forks a child per request; the child
inherits the initialised address space and calls the same
`CommandLine::Parse`/`CommandLine::Run` pair `main/main.cpp` calls, so it
images exactly what `wsclean` would have imaged.

```
request: <cwd> \t <stdout path> \t <stderr path> \t <arg> ...
reply:   <exit status> \t <wall seconds> \t <peak rss bytes>
```

Three details are load-bearing:

* **The child must be able to fork safely.** `fork()` copies only the calling
  thread, so a second thread holding a lock at fork time deadlocks the child.
  Nothing wsclean links starts a thread from a static initialiser today - the
  gridder's and OpenMP's pools are created on first use, inside the child - but
  the zygote reads `/proc/self/status` at startup and refuses to run if it is
  not single-threaded, because the failure mode is a search that hangs rather
  than one that stops.
* **The child scopes its `WSClean` object and then `_exit()`s.** The
  destructor is what removes the reordered temp files (`Cleaning up temporary
  files...` is the last line of every log), so it has to run; the global
  destructors and the unmapping of 73 shared objects afterwards do not, and are
  most of the 4.6 ms above.
* **The reply carries `wait4()`'s rusage.** That is the wall clock and peak RSS
  the harness used to fork `/usr/bin/time -v` per evaluation to get, so the
  fork+exec of GNU `time` goes away with it - and `image_binary_seconds` stops
  being quantised to GNU `time`'s centisecond clock. The profiler's
  `image_container_overhead` line, which
  [the throughput doc](nested-sampling-throughput.md) had to explain away as
  mostly that rounding, now reads ~1.9 ms an evaluation and means what it says.

It is built by `docker/wsclean/patches/0002-build-wsclean-zygote.patch`, which
adds nothing but an `add_executable`/`install` pair to WSClean's
`CMakeLists.txt`; the source itself is copied into the tree by the Dockerfile
rather than expressed as a diff, because a whole new file is easier to read and
to keep building than the same file as a patch. That makes 0002 the one patch
in `docker/wsclean/patches` that is *not* something upstream would take as-is -
what it enables is this repo's, not WSClean's - but it changes nothing WSClean
computes, which is the property that matters for comparing archived runs.

## How the harness uses it

`zygote_run()` in `scripts/lib/nested_sampling/common.py` replaces
`sidecar_run()` on the WSClean path and keeps its retry shape exactly: the same
`worker_send`/`worker_reply` bounds, the same drop-and-restart on a worker that
dies between evaluations, the same `WORKER_DIED` after the last attempt. The
one long-lived `sh` per rank that `sidecar_shell()` used to be is now
`sidecar_worker()`, which starts either `sh` or `wsclean-zygote` inside the
rank's sidecar container and caches it per `(image, command)`.

`./ri self-check zygote` runs `scripts/test_zygote.py` inside the WSClean image
against the real binary: exit statuses plumbed through, a failing request not
ending the server, an unparseable request answered rather than desynchronising
the stream, `cwd` honoured, and the rusage fields non-zero.

## The images are identical

40 Measurement Sets from a `--keep-measurement-sets` corpus were imaged twice
with the same argv - once as `wsclean`, once through the zygote - and all 200
output FITS **data blocks** (image, dirty, residual, model, psf) compare equal.
The comparison is on the data block rather than the file because WSClean writes
its own command line into the header, so two identical images from two
directories have different checksums.

That is the expected result and it is worth saying why: the child is a `fork()`
of a process that has run nothing but static initialisers, so it starts from
the same state a fresh `wsclean` starts from and then runs the same two
functions `main.cpp` runs. Nothing is reused between evaluations - the point is
that the *initialisation* is, not the state.

## What it is worth

Eight pairs of real searches, both arms running at the same instant - 10 ranks
each on this 20-thread host - one against the images built from `HEAD` before
this change and one against the images built with it. Launch order was swapped
between pairs. Each arm is scored over the intersection of the two runs'
evaluation windows, because they hit `--max-ndead` at different times.

| pair | first | base evals/s | zygote evals/s | ratio | base `image_binary` | zygote `image_binary` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | base | 41.82 | 44.34 | 1.060 | 182.40 ms | 163.65 ms |
| 2 | base | 41.34 | 42.14 | 1.019 | 181.84 | 178.88 |
| 3 | base | 37.29 | 42.02 | 1.127 | 205.29 | 183.21 |
| 4 | base | 38.03 | 43.12 | 1.134 | 202.62 | 178.76 |
| 5 | zygote | 39.38 | 42.50 | 1.079 | 187.32 | 180.96 |
| 6 | base | 38.15 | 42.83 | 1.123 | 202.43 | 183.42 |
| 7 | zygote | 39.16 | 41.19 | 1.052 | 199.25 | 185.06 |
| 8 | base | 39.58 | 42.84 | 1.083 | 193.38 | 179.95 |
| **geometric mean** | | | | **1.084** | | **0.923** |

`--nlive 25 --num-repeats 10 --max-ndead 250` (pair 1: `--max-ndead 80`, ~40 s,
which is why it is the shortest and the least trustworthy). The spread is
1.019-1.134; pair 2 is the low outlier and its *baseline* arm is the fast one
(181.84 ms against 193-205 ms in the other seven), so it reads as a quiet
minute for the baseline rather than a slow one for the zygote.

The `image_binary` column understates the change and cannot be read as-is
across these two arms: the baseline's number comes from GNU `time -v`'s
truncating centisecond clock and is ~5 ms low on a ~200 ms call
([why](nested-sampling-throughput.md#image_container_overhead-is-mostly-gnu-times-clock-resolution)),
while the zygote's comes from `CLOCK_MONOTONIC`. Adding that 5 ms back to every
baseline entry puts the geometric mean at 0.900, i.e. -10% on the binary, which
is the same 27 ms measured at the top of this page arriving as ~20 ms of
recovered wall clock. The remainder is what `fork()` of a process with 73
mapped shared objects costs, plus the page faults the child takes on the pages
it writes.

### What is left

The zygote forks a process that has done static initialisation and nothing
else, so every child still pays casacore's *lazy* initialisation on first use.
This page originally read the ~11 ms between a child's first log line and
`=== IMAGING TABLE ===` as that cost and put a parent-side warm-up at ~7% of an
evaluation. Measured since, the process-global part of it is **0.94 ms** - most
of those 11 ms is per-Measurement-Set work a child would pay anyway, and the
expensive lazy initialisation casacore does have (34 ms for a process's first
`MDirection` conversion) is never triggered in this configuration. A warm-up is
therefore worth ~1% of an evaluation, below what any rig here resolves, and it
would still trade away the property that makes this change safe - the parent has
run no WSClean code. The avenue is closed for *casacore*;
[the phase profile](nested-sampling-phase-profile.md) has the numbers.

The parent is not idle any more, though. Process-global state that is worth
pre-paying does exist - it is FFTW's, not casacore's: 4.4 ms an evaluation of
transform-plan building that a warmed parent hands to every child, which is what
`WarmFftwPlanner()` does and
[the FFTW planner doc](nested-sampling-fftw-planner.md) measures. It keeps the
property above: building an FFT plan runs no WSClean code.

casacore's ~1% has since been taken as well, because a second process-global
item turned up beside it - cfitsio's one-time initialisation, 0.47 ms - and the
pair reads straight off the phase table. `WarmCasacore()` opens and closes the
Measurement Set the *first request* names, which is the only way the parent can
reach a real set without carrying one; it still runs no WSClean code, and no
handle or lock file crosses the fork. See
[the process warm-up doc](nested-sampling-process-warm-up.md).

## Reproducing it

The start/shutdown split, from a corpus of Measurement Sets kept by
`./ri search wsclean --keep-measurement-sets`:

```sh
# one request per corpus entry, each wrapped in its own timestamps
date +%s%N > $out/pre; wsclean -log-time <argv> > $out/log 2>&1; date +%s%N > $out/post
```

then, per log, `first log timestamp - pre` and `post - last log timestamp`.
The log's stamps are local time and `date +%s%N` is epoch, so subtract the zone
offset. Run the whole corpus at least twelve times at 19-way concurrency: a
single pass of 160 Measurement Sets takes 1.2 s and lands entirely inside the
burst-clock window, which reads ~25% fast.

The per-library attribution:

```sh
LD_LIBRARY_PATH=/opt/casacore/lib
LD_PRELOAD=/opt/casacore/lib/libcasa_ms.so.9 /bin/true   # x40, timed
```

The end-to-end A/B, two searches at once against two builds - the only honest
shape for a change that is half in the WSClean image and half in the sampler's
(see [the patches doc](nested-sampling-wsclean-patches.md)):

```sh
git archive HEAD | tar -x -C /tmp/base
docker build -f docker/wsclean/Dockerfile   -t ri-reproducibility/wsclean:base   /tmp/base
docker build -f docker/polychord/Dockerfile -t ri-reproducibility/polychord:base /tmp/base
WSCLEAN_IMAGE=ri-reproducibility/wsclean:base POLYCHORD_IMAGE=ri-reproducibility/polychord:base \
  ./ri search wsclean --mpi-procs 10 ... &
./ri search wsclean --mpi-procs 10 ... &
```

Score each arm over the intersection of the two runs' evaluation windows
(`metrics.json` mtimes), not over its own wall clock: the arms hit
`--max-ndead` at different times and whoever outlives its partner gets a
quieter machine.
