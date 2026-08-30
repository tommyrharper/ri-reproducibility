# Nested sampling: profiling

## What is instrumented

Both `polychord_wsclean.py` and `polychord_r2d2.py` time each likelihood stage
with `time.perf_counter()` around the existing invocations. Each evaluation's
`metrics.json` (and aggregated `summary.json`) gets a `timing` block:

| Field | Meaning |
|---|---|
| `simulate_seconds` | Wall time for the MeqTrees `docker run` that produces `sim.ms` (container start + RIME simulation, not split further) |
| `convert_seconds` | R2D2 only: wall time for the MS -> `.mat` conversion in the rank's simulate worker |
| `image_container_seconds` | Wall time for the imaging round trip: a `docker run` for WSClean, one request to this rank's R2D2 worker for R2D2 |
| `image_binary_seconds` | WSClean only: the imaging child's own elapsed time, from `wait4()` in the rank's `wsclean-zygote` (before 29 August 2026, from `/usr/bin/time -v`, quantised to 10ms - see [the zygote doc](nested-sampling-wsclean-zygote.md)) |
| `metrics_seconds` | Wall time for `compute_image_metrics()` (FITS read + numpy) |
| `started_epoch`, `ended_epoch` | Wall-clock epochs the evaluation began and finished at, stamped by `mark_evaluation_start()` and `write_evaluation_record()`. Not a stage: they are the interval the stages sat inside, and they are what separates the time spent in PolyChord from the time spent idle below. A run that predates them has them reconstructed at read time from the mtime of this file - see `backfill_busy_seconds()` |

`image_container_overhead_seconds` (container round trip minus binary time) is
only available for WSClean, because only its path reports the two separately;
R2D2 and MeqTrees report only the round-trip time as one blob.

`summary.json` also gets a run-level `profiling` block: each field above
summed across every evaluation, plus:

- `accounted_worker_seconds` - sum of every stage total across all evaluated
  points.
- `accounted_seconds` - same value as `accounted_worker_seconds`, but emitted
  only for serial runs where `NS_MPI_PROCS=1`.
- `polychord_overhead_seconds` = `total_wall_seconds - accounted_seconds`.
  This is whatever PolyChord itself is doing outside likelihood calls (its own
  slice-sampling bookkeeping, live-point management, I/O to `chains/`). It is
  emitted only for serial runs where `NS_MPI_PROCS=1`; at higher MPI process
  counts, ranks run likelihood evaluations concurrently, so summed
  worker-seconds cannot be subtracted from rank-0 elapsed wall time.
- `busy_worker_seconds` and `busy_wall_seconds` - the evaluation intervals
  summed, and the same intervals unioned: worker-seconds spent inside a
  likelihood evaluation, and the wall clock over which *any* evaluation was in
  flight. `evaluation_busy_seconds()` computes both, clamping to the segment
  this wall clock covers so a resumed run's adopted records cannot count twice.

## Running the profiler

The instrumentation runs automatically. `./ri tui` shows this table for the
selected run (`enter`, then `l` loops health, log, and profile), and the HTML
report carries it per run page. Read a completed run from the shell:

```bash
./ri profile results/nested-sampling/wsclean-vlaa-<UTC timestamp>
./ri profile results/nested-sampling/wsclean-vlaa-<UTC timestamp> --json
./ri profile results/nested-sampling/r2d2-vlaa-<UTC timestamp> --r2d2-phases
```

The profiler only reads `summary.json`; older runs without a `profiling` block
must be re-run.

### How the printed shares are computed

The JSON fields above are raw sums. Turning them into something readable is
`profiling_breakdown()` in `scripts/lib/nested_sampling/common.py`, shared
by the CLI and by the HTML report's "Profiling (where the run's time went)"
section so the two cannot drift apart. It adds three things the raw block does
not carry:

- **A per-evaluation column** - each stage total divided by the number of
  evaluations that recorded it, so "39m 15s of imaging" also reads as "53.5s an
  image".
- **A single share denominator that works at any MPI process count** - the run's
  *worker-time budget*, `total_wall_seconds x mpi_procs`. Every top-level row is
  a share of that, so the breakdown always comes to 100% of what the whole
  process spent. At `NS_MPI_PROCS=1` the budget is just the wall clock.
- **The run in two halves, split at `evaluating (sum of the above)`.** Above
  that line is what happened inside a likelihood evaluation: the timed stages,
  and the harness Python around them. Below it is the time no evaluation was
  running in - which used to print as one row, "unaccounted (PolyChord sampling
  + idle)", named after everything that could be in it, because it was a
  subtraction rather than a measurement. The evaluation intervals measure it:

  | Row | What it is | What moves it |
  |---|---|---|
  | harness (Python around the stages) | Worker-seconds inside an evaluation but outside its timed stages: this repo's own Python between the subprocess calls. A stage in all but name, so it sits with them | Harness code only |
  | PolyChord (no evaluation in flight) | Wall clock during which no rank was inside an evaluation, charged to every worker because not one of them could spend it. PolyChord's own sampling and `chains/` I/O, plus the run's start-up and shutdown | Nothing on the harness side. [PolyChord costs microseconds a call at every `nlive`](nested-sampling-throughput.md#polychord-itself-costs-microseconds-a-call-at-every-nlive), so on a run of any length this is the per-run constant, not the sampler |
  | idle (waiting on other workers) | What is left once PolyChord is taken out: workers waiting while other workers were still evaluating, i.e. load imbalance, since an evaluation's cost varies with the point drawn | `--mpi-procs` against `--nlive` - a bigger run keeps more evaluations in flight, which is why [utilisation rises with run size](nested-sampling-throughput.md#a-bigger-run-is-a-more-efficient-run) |

  A run archived before the epochs were stamped is split anyway, from its own
  file mtimes - see below - and keeps the single combined row only when even
  that is not available.
- **The intervals of a run that never stamped any**, from
  `backfill_busy_seconds()`. Each evaluation's `metrics.json` is written once,
  at the end of that evaluation, so its mtime is the interval's end and the
  stage totals on the record are its length. Against a run carrying both, the
  reconstruction is within 0.5% of the stamped numbers. It is refused - leaving
  the combined row, and a note saying why - in the two cases where it would
  invent a timeline rather than read one:

  - **A record whose evaluation directory has gone.** The time of an evaluation
    nobody can see would be charged to PolyChord, which is worse than not
    splitting at all.
  - **mtimes that cannot be a timeline.** Directories restored from a backup,
    or rewritten in a batch, all carry the time of that copy. The giveaway is
    arithmetic: worker-seconds of imaging that will not fit in the wall clock
    the mtimes claim they happened in (`busy_wall x workers < busy_worker`).
    Five of this repo's earliest runs are refused on exactly that.

  A reconstructed interval is the timed stages and nothing else, so those runs
  carry no `harness` row - what the subtraction would leave is float noise, and
  it goes to idle with the rest of what was never measured.
- **Stage labels naming the actual imager** - "wsclean container" or "r2d2
  container", taken from the summary's `algorithm`.

### How the report ties those shares back to the wall clock

Worker-seconds exceed page wall time by `mpi_procs`. Dividing stage totals by
`mpi_procs` gives the report's `wall clock` column (omitted for serial runs),
and `evaluating + PolyChord + idle = worker-time / workers = wall clock`.
`render_profiling()` in `scripts/lib/generate_report.py` charts these proportions
as one average-worker lane per rank; imaging is coloured, other rows grey.

R2D2's aggregate image stage can be split with `--r2d2-phases`. It reads the
worker's model-update and residual timings, reporting median milliseconds per
evaluation and call counts. The latest 41-evaluation run measured 7219.7
ms/evaluation for model updates (25 calls) versus 80.1 ms/evaluation for
residuals, making model inference the next optimization target.

### What the two halves show

A 4-rank (3 workers) `--nlive 10 --max-ndead 40` run, 370 evaluations over
7.14s of wall clock - 21.4s of worker-time - profiles as:

| Row | Total | Per eval | Share |
|---|---:|---:|---:|
| WSClean container (total) | 17.7s | 48ms | 82.5% |
| MeqTrees simulate | 2.10s | 6ms | 9.8% |
| Metrics computation | 305ms | 1ms | 1.4% |
| harness (Python around the stages) | 53ms | 143us | 0.2% |
| **evaluating (sum of the above)** | **20.1s** | **54ms** | **94.0%** |
| PolyChord (no evaluation in flight) | 58ms | | 0.3% |
| idle (waiting on other workers) | 1.22s | | 5.7% |

Which is the answer the split exists to give: the time outside an evaluation
is not the sampler. PolyChord and this harness together are 0.5% of the run,
and 92% of what is not imaging is three workers waiting on each other - the
one part of it that `--mpi-procs` and `--nlive` move.

`wsclean` itself is at its floor for this problem size: it self-reports
0.035s inversion + 0.023s prediction + 0.008s deconvolution per evaluation
against ~0.018s of process startup, and moving `-temp-dir` to `/dev/shm`
changes nothing measurable (the reordered scratch files never reach the ext4
journal). `-j 4` buys ~5ms but would multiply threads by the MPI rank count
and make the gridding sum order - and so the image - non-deterministic.

### The PolyChord ranks run with one BLAS thread each

`numpy`'s bundled OpenBLAS spawns one busy-waiting worker thread per host CPU,
in every process that touches it. Each MPI rank is one such process, so on this
20-CPU host the default 8 ranks carried ~160 spinning threads between them and
burnt ~10 cores waiting for work that never arrives - nothing in this pipeline
has a BLAS call big enough to want them, the largest being a norm over a
128x128 image. The starvation showed up as every stage inflating under MPI:
`metrics_seconds` averaged 0.28s per evaluation on 8 ranks against 0.005s on
one, for 1.5ms of actual arithmetic.

Both run scripts therefore pass `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` to the PolyChord container. Measured over 3 runs each,
the default 8-rank run went 8.25s -> 5.26s (-36%) and its summed metrics
computation 11.3s -> 1.3s. This is not worth pushing down into the sidecars:
the MeqTrees image's system `numpy` spawns no BLAS threads at all, and `wsclean`
already runs `-j 1`.

Single-rank is ~3% *slower* with the pin (10.95s -> 11.28s, consistent across
interleaved A/B runs) - with no contention to remove, the spinning threads were
keeping cores out of idle states. The default configuration is 8 ranks, so the
pin stays.

`log(Z)` and every evaluation's objective are unchanged. `sigma_res` moves in
its last bit, because a single-threaded `np.linalg.norm` reduces in a different
order than a threaded one; that also makes it reproducible across hosts with
different CPU counts, which it previously was not.

### The compiled TDL forest is reused across evaluations

`Compile.compile_file()` was ~0.034s of every evaluation, and the forest it
builds does not depend on the Measurement Set's shape: the antenna layout and
phase centre come from the fixed antenna table and the hardcoded
`RightAscension`/`Declination` in `write_makems_config()`, and the time and
frequency axes are runtime data the `VisDataMux` reads per request. So
`run_meqtrees_predict()` keys a process-level cache on the generated `.tdlconf`
text with the `ms_sel.msname` line removed, and on a hit calls
`point_to_measurement_set()` - which re-points the `MSSelector` at the new MS -
instead of recompiling. Over a 62-evaluation run that is one compile and 61
reuses.

`MSSelector._select_new_ms()` re-lists the MS's data columns, which resets the
output-column option that `_define_forest()` set to `DATA`; the reuse path has
to re-assert it or the sinks quietly write `CORRECTED_DATA` and `DATA` comes
back all zeros with no error anywhere. `simulate_point_source_ms.py
--self-check` guards exactly that: it predicts three MS shapes off one cached
forest and off three fresh compiles and asserts the `DATA` columns are equal.

Measured over three runs each of the default single-rank configuration: 16.9s
before, 13.2s after (-22%), with the simulate stage down 39% (9.0s -> 5.5s -
more than the compile alone, because a reuse also skips the MS-metadata reads
the compile did). The 8-rank default went 9.1s to 8.4s (-8%; with ~5
evaluations per rank there is much less to amortise). All 62 images were
pixel-identical, every science metric matched, the evaluation directories held
the same file tree and `log(Z)` was unchanged.

### The `makems` skeleton is cached per MS shape

`makems` is ~0.05s of every simulate and its output depends on the whole
`makems.cfg` except `StartFreq`/`StepFreq`, which move exactly six
`SPECTRAL_WINDOW` columns (`CHAN_FREQ`, `CHAN_WIDTH`, `EFFECTIVE_BW`,
`RESOLUTION`, `REF_FREQUENCY`, `TOTAL_BANDWIDTH`) and nothing else - verified
by comparing every column of every subtable across two frequency settings.
`make_ms_skeleton()` in `simulate_point_source_ms.py` therefore keys a cache on
the config text with those two lines removed, and on a hit copies the cached
skeleton inside `/dev/shm` (~0.002s) and rewrites those six columns instead of
running `makems`. Only `observation_minutes` and `channel_count` reach the key,
so the parameter space has 20 distinct shapes.

The cache is a directory in `/dev/shm`, not a dict in the worker process, and
that matters on the default 8 ranks: all eight `--serve` workers `docker exec`
into the same meqtrees sidecar, so they share `/dev/shm`, and a shape any one of
them has built is a `copytree` away for the other seven. A default run makes 41
evaluations over only ~12 distinct shapes, so per-process caches missed on most
of them - each rank sees ~5 evaluations and almost every one was a fresh
`makems`. Entries are staged in a scratch directory and `rename`d into place
under `sha256(key)`, so a concurrent worker either does not see an entry or sees
a complete one; losing that race is normal and the loser just drops its copy.

Measured over six interleaved A/B pairs of the default 8-rank run (rebuilding
the meqtrees image between arms): summed simulate worker-seconds 5.38s -> 4.68s
(-13%, 6/6 pairs) and end to end 3.65s -> 3.38s (-7%, 5/6 pairs). All 41
evaluations matched on every science metric and `log(Z)` was bit-identical.

#### The image ships every skeleton, so no run calls `makems`

Waiting for an evaluation to miss puts the ~0.11s of a fresh `makems` in the
middle of the sampler's critical path. Building the shapes in the workers'
background threads once the run had started only half-fixed that: a default
8-rank run still took ~7 misses (0.06-0.10s each, in-worker timings) because
the sampler asks for its first evaluations well before ~3 shapes per worker
have been built, and all eight workers additionally raced on the same fresh
`makems` inside `warm_forest()`.

The parameter space only has 20 shapes and they cost ~1.2s and ~18MB to build,
so the meqtrees image builds all of them at `docker build` time -
`simulate_point_source_ms.py --prebuild-skeletons` into `/opt/ms-skeletons` -
and `skeleton_dir()` prefers that directory when it exists. It is an ordinary
writable container path, so a shape the image was not built with is still built
and published there at runtime: the baked set is a head start, not a fixed set.
`common.py` and `defaults.toml` are `--mount=type=bind`ed for that one build
step rather than copied, so the shapes come from the single authoritative
`[[parameter_space]]` in `defaults.toml` and the runtime image still carries
only the three simulate-side scripts.

The MS name is part of the cache key, so a prebuilt entry is only useful if it
is built under the name a real evaluation uses (`sim.ms`).
`self_check_skeleton_prebuild()` is the guard: it prebuilds a two-by-two slice
of shapes and then asserts a real `make_ms_skeleton()` call for one of them
reports a cache hit. It fails if the prebuild builds under any other name.

Measured over 40 interleaved A/B pairs of the default 8-rank run against
pre-tagged `:ab-old`/`:ab-new` images (which is also the TaQL `WEIGHT`/`SIGMA`
change above): end to end 3.148s -> 2.955s (-6.1%, -0.193s +/- 0.029s, t = 6.7,
33/40 pairs), summed simulate worker-seconds 2.73s -> 1.89s (-31%), and zero
skeleton cache misses against seven before. All 41 evaluations' objectives were
bit-identical.

Because the workers still build into the cache directory when they miss, and
because `/dev/shm` is where an unbaked image's cache lives, the sidecars keep
`--shm-size 512m`; docker's 64MB default is only about 3x what 20 skeletons
need.

`simulate_point_source_ms.py --self-check` is the guard on the rewrite formula
(and on the forest reuse below): it builds each shape both ways and asserts a
patched cache hit matches a fresh `makems` run column for column. Run it in the
meqtrees image:

```bash
docker run --rm --network none ri-reproducibility/meqtrees:kern-10 --self-check
```

Measured over three runs each of the default single-rank configuration above:
19.8s before, 16.9s after (-15%), with the simulate stage down 26% (12.2s ->
9.0s). All 62 images were pixel-identical, every science metric matched, the
evaluation directories held the same file tree, and log(Z) was unchanged.
`copytree(..., symlinks=True)` matters here: `makems` leaves `vis.DATA`,
`vis.uvw` and `vis.flg` as symlinks into the tiled storage manager files, and
copying them as regular files leaves stale duplicates of the visibilities in
every evaluation directory.

### Sidecar commands go through one long-lived `sh` per rank

`docker exec` costs ~0.033s on this host, a third of the `wsclean` binary's own
~0.107s, and every evaluation paid it again. `sidecar_shell()` in
`common.py` therefore `docker exec -i`s a single `sh` into the rank's
sidecar on first use, and `sidecar_run()` sends each later evaluation one
command line - `cd <eval_dir> && <cmd> >stdout.log 2>stderr.log; echo $?` - and
reads the exit code back. Arguments are `shlex.quote`d, the command's own output
goes to the log files, so nothing a sidecar prints can be mistaken for a reply,
and a shell that dies without answering is dropped from the cache the same way
the simulate worker is.

A round trip costs ~0.0003s against ~0.033s for `docker exec`, taking WSClean
container overhead from 0.78s to 0.18s over 19 evaluations and the profiled run
from 7.88s to 7.04s (medians of three runs each, -10.6%). A 4-rank
54-evaluation run went 14.0s to 13.4s. Metrics, `log(Z)` and the reconstructed
FITS images are pixel-identical; only the recorded `commands.wsclean` changes,
from the `docker exec` argv to the in-container command it wrapped.

### The simulate sidecar is a long-lived worker process

Even inside a reused sidecar container, a per-evaluation `docker exec` of
`simulate_point_source_ms.py` spent ~0.45s of its ~0.7s on startup the next
evaluation would immediately repeat: 0.10s of Python plus numpy/casacore
imports, 0.14s of Timba imports, 0.04s starting a meqserver and ~0.10s reaping
it again, against ~0.14s of actual RIME predict and ~0.05s of `makems`.

`simulate_point_source_ms.py --serve` therefore reads one JSON request per
stdin line - `{"argv": [...], "stdout": path, "stderr": path}` - and replies
with `{"returncode": int}` on its original stdout, with fds 1 and 2 pointed at
the request's log files for the duration so `makems` and the meqserver still log
per-evaluation exactly as they did when each was its own process.
`simulate_worker()` in `common.py` starts one such process per rank on first
use and writes to its stdin from then on; a worker that dies without replying is
dropped from the cache so the next evaluation starts a fresh one instead of
inheriting the corpse.

The predict itself moved in-process with it: `run_meqtrees_predict()` now runs
the same `TDLOptions`/`Compile.compile_file`/job sequence `meqtree-pipeliner.py`
runs, against a meqserver that survives between requests. `mqs.get_error_log()`
flushes, so each request only ever sees its own errors. Those errors are
printed with `!r`, not `str()`: Timba's DMI record `__str__` is still py2
(`string.join`) and raises `AttributeError`, which used to replace the
meqserver's actual error with a traceback from the error-reporting path itself.
`__repr__` on the same class is py3-clean.

Measured cost per simulate dropped from 0.62s one-shot to ~0.18s served. On the
profiled single-rank run that is 16.8s to 7.7s total (-54%), and on a 4-rank
54-evaluation run 25.2s to 13.7s (-45%), with identical science metrics,
identical `log(Z)` and identical per-evaluation artifact file sets. Only
`wall_seconds` and `peak_memory_bytes` - the WSClean timing metrics - differ,
as they do between any two runs.

The worker starts its meqserver before it reads its first request. It used to
be started lazily, inside `meqserver_session()`, which meant the ~0.3s of Timba
imports plus server startup was paid *inside* evaluation one on every rank -
and since PolyChord asks all ranks for their initial live points at once, all of
it landed on the wall clock. Nothing has been asked of a freshly spawned worker,
so `serve()` pays it up front instead, concurrently with the rank's own
PolyChord import and setup. Interleaved A/B over 9 pairs (rebuilding the
`meqtrees` image between arms) put the default 8-rank run at 4.52s before and
4.11s after, -9%, with B faster in all 9 pairs and bit-identical `log(Z)` and
per-evaluation objectives.

That warm-up runs under `redirect_fds(os.devnull)`: Timba prints to fd 1 while
it starts (`Qt not available, substituting proxy types for QObject` and
friends), and fd 1 is the reply pipe, so without the redirect the first
`{"returncode": ...}` line arrives behind three lines of chatter and
`json.loads` fails. `--self-check` covers exactly that: it runs a worker,
sends one deliberately invalid request and asserts its stdout is a single JSON
line - verified to fail when the redirect is removed.

#### The workers are started by the container, not by the ranks

Starting the meqserver eagerly still leaves the worker unable to answer for
~0.5s after it is launched, measured across eight launched at once: ~0.11s of
`docker exec`, ~0.07s of interpreter and imports, ~0.11s of meqserver, and the
rest the first TDL compile and first predict (a fresh worker's first simulate
is ~0.17s against ~0.03s for its second, even against a warm skeleton cache).
No amount of eager work *inside* the worker can hide that, because the rank
that launched it asks for its first evaluation ~0.2s later.

So the ranks no longer launch them. `run-nested-sampling.sh` creates one
`<rank>.in`/`<rank>.out` FIFO pair per rank under
`<output-dir>/.simulate-workers`, and the meqtrees sidecar's *container
command* - `sidecar_launch ... -- sh -c ...`, in place of the default `sleep
infinity` - spawns one `simulate_point_source_ms.py --serve --fifo <base>` per
pair. `common._connect_shell_started_worker()` opens that pair instead of
spawning anything, and `FifoWorker` presents the same
`.stdin`/`.stdout`/`.terminate()` surface as the `subprocess.Popen` it replaces,
so nothing downstream changed. The FIFOs reach across containers because the
PolyChord container and the meqtrees sidecar both bind-mount `REPO_ROOT`: a FIFO
on a bind mount is one host inode both of them open.

It has to be the container's command rather than a `docker exec` into it. A
`docker exec` cannot be issued until `docker run` has returned, which is ~0.02s
after the container's own command has already started and ~0.1s before the
exec's process does; an earlier version that retried `docker exec --detach`
into the container as soon as it would accept one measured -0.06s end to end
over 14 pairs on a ~2.95s baseline, against -0.15s for this one. Under `--fifo` the worker also
compiles the forest and runs one throwaway predict before it opens its request
pipe (`warm_forest()`); on the stdin path it deliberately does not, because
there the rank that started it is already waiting.

The price is that `HOST_CPUS` - which sets how many FIFO pairs to make - moves
back in front of the sidecar launches, because the FIFOs have to exist before
the container's command globs for them. It is read from `nproc` rather than
`docker info --format '{{.NCPU}}'` for exactly that reason; see "`HOST_CPUS`
comes from `nproc`, not from the daemon" below.

Measured on the default 8-rank run: the eight `eval_id == 1` `simulate_seconds`
records go from 0.18-0.41s (median ~0.33s) to 0.05-0.11s against a ~0.05s
steady state, `total_wall_seconds` from ~2.4s to ~1.8s, and end to end 3.40s to
3.25s - -4.4%, 20 of 24 interleaved pairs, sd of the paired difference 0.18s -
with bit-identical `.stats`, `.txt` and `_dead-birth.txt` chains and identical
per-evaluation params and metrics.

Two sharp edges. Opening a FIFO blocks until the other end opens, so both sides
must open the request pipe first and the reply pipe second; reverse either and
the run hangs with no error at all. `--self-check`'s `self_check_serve_fifo()`
is the guard, verified to fail (rather than hang) when `serve()`'s two opens are
swapped. And the rank's side opens with `O_NONBLOCK`, which is how a FIFO
write-open reports "no reader yet" (`ENXIO`) instead of blocking forever: it
retries for 10s and then falls back to starting its own worker, so a missing or
broken pool costs latency, not the run. That fallback is what an `OUTPUT_DIR`
outside `REPO_ROOT` gets, since the FIFOs are then not visible in both
containers.

One sharp edge: Timba registers `stop_default_mqs()` with `atexit`, but CPython
joins non-daemon threads - including octopussy's event thread, which only exits
once the server is stopped - *before* it runs `atexit` handlers, so a process
that leaves meqserver teardown to `atexit` finishes all its work and then hangs
at exit forever. `meqtree-pipeliner.py` avoids that by calling
`stop_default_mqs()` explicitly, and so does `stop_meqserver_session()` here.

### The Measurement Set is built in tmpfs

`makems` and casacore `fsync` on nearly every table write, so on the
bind-mounted repo the simulate sidecar spends most of its wall time parked in
`jbd2_log_wait_commit` waiting for ext4 journal commits - sampling
`/proc/<pid>/wchan` during a `makems` run put it in journal wait for ~75% of
the samples. The cost is entirely fixed overhead, not data volume: `makems`
takes 0.54s for a 1-time-sample, 1-channel MS and 0.55s for the largest MS this
parameter space produces, but only 0.046s when the same run happens on tmpfs.

`simulate_point_source_ms.py`'s `main()` therefore builds everything -
`makems.cfg`, the unpacked `VLAA_ANT` table, the MS, the MeqTrees predict and
the noise fill - inside a `tempfile.TemporaryDirectory(dir="/dev/shm")`, then
moves the finished directory contents to the real output path in one go. The
whole MS is ~1MB, so the copy out is ~2ms, and every artifact a run used to
leave in the evaluation directory (including `makems.log`,
`meqtree-pipeliner.log` and `point_source_forest.tdlconf`) still lands there -
verified by `find`-diffing evaluation directories before and after. The two
MeqTrees files are absent from an evaluation of a source at the phase centre,
which no longer runs a predict at all (see
[nested-sampling-throughput.md](nested-sampling-throughput.md)).

Measured per-simulate cost dropped from 1.12s to 0.55s standalone, and on the
profiled run from 23.7s to 13.3s of simulate (27.5s to 16.9s total, -38%) with
identical per-evaluation metrics and identical `log(Z)`. Under 8-way MPI the
win is larger still - the ranks were contending for the same journal - taking
an 8-rank 41-evaluation run from 19.9s to 13.8s.

Docker gives a container 64MB of `/dev/shm` by default, which is ~30x the
largest MS this parameter space produces; a bigger parameter space needs
`--shm-size` on the sidecar. `SCRATCH_ROOT` falls back to the `tempfile`
default when `/dev/shm` is not writable.

### Each rank warms its sidecar attachments before the sampler starts

The first evaluation on a rank used to cost ~0.7s that later ones did not, and
every rank paid it at the same moment - PolyChord asks all `nlive` initial live
points at once - so the whole thing landed on the wall clock in front of
evaluation one. It was four independent startups run one after the other inside
`evaluate()`:

| Startup | Cost |
|---|---:|
| `simulate_point_source_ms.py --serve` worker (`docker exec`, Python, Timba, meqserver) | ~0.45s, now started eagerly by the worker itself |
| `astropy.io.fits` import, on the first metrics call | ~0.45s on 8 ranks, now gone - see below |
| `docker inspect` of each image's `ENTRYPOINT`, twice | ~0.05s |
| the WSClean sidecar's `sh` (`docker exec`) | ~0.03s |

`common.prewarm()` starts all of them in threads, so a rank pays the slowest
instead of the sum. `main()` calls it before `import pypolychord` and joins it
immediately before `run_polychord()`, so the remainder also overlaps the
sampler's own import and setup. Nothing may touch a sidecar between the call and
the join: `_SIMULATE_WORKERS`, `_SIDECAR_SHELLS` and `_IMAGE_ENTRYPOINTS` are
plain dicts with no lock, and a lazy start racing the prewarm thread would leave
a second, orphaned worker.

**What is still left in evaluation one.** Per-evaluation `simulate_seconds`
from `summary.json` used to show the eight `eval_id == 1` records (one per
rank, all issued at the same moment) at 0.18-0.41s against a ~0.05s median for
the rest of the run - the `--serve` worker's startup, finishing inside the first
request because the rank that started it had nothing else to do meanwhile. The
run script now starts those workers as the meqtrees container's own command
(see above) and the same records read 0.05-0.11s, so this is no longer the
largest item on the critical path.

What replaced it is the prewarm join itself, and two-sided timestamps have since
shown that the join is the whole critical path - see "The critical path is the
simulate worker's warm-up, not the rank" below. Closing it needs the worker to
be *ready* sooner, not started sooner: the remaining ~0.4s is ~0.07s of
interpreter and imports, ~0.10s of meqserver and ~0.25s of first compile and
first predict. Baking a ready-made Measurement Set into the image to skip the
warm-up's `makems` was measured and moves worker-ready time by nothing.

### FITS images are read without astropy

`from astropy.io import fits` was the single largest per-rank startup left, and
it dominated the prewarm join: ~0.45s when the 8 default ranks import it at
once, against ~0.07s for both sidecar startups put together. Instrumenting the
prewarm threads is what showed it - the other two only `Popen`, so they return
in milliseconds and the join was pure astropy.

All it was doing is reading a single-HDU, uncompressed, `BITPIX = -32` image and
two header cards (`CRPIX1`/`CRPIX2`). `common.load_fits_2d()` now does that
directly: 2880-byte header blocks of 80-column cards, then big-endian samples in
C order. Anything outside that shape - an integer or `BSCALE`/`BZERO`-scaled
image, a short data block - raises instead of being guessed at. astropy is still
installed in the image and still used by the self-check.

The trap is card parsing, not the data block: a quoted value may contain the `/`
that otherwise starts the comment (`BUNIT = 'JY/BEAM '`), so `_fits_card_value()`
closes the quote before cutting the comment. `self_check_fits_reader()` (run by
`POLYCHORD_WSCLEAN_SELF_CHECK=1`) writes exactly that card with astropy and
asserts the reader agrees; it fails if the quote handling is removed.

Verified against astropy on all 16833 FITS files this repo's results tree
contains - identical pixels and identical values for every non-comment header
card. Eight interleaved A/B pairs (rebuild between arms) gave 3.91s -> 3.61s end
to end on the default 8 ranks, -7.8%, 8/8 pairs, with a bit-identical
`chains/wsclean_vlaa.stats` and bit-identical metrics for all 41
evaluations.

### Sidecar teardown does not block the run

The EXIT trap's `docker rm --force` of the three containers costs ~0.4s, spent
after every result is already on disk. It is now backgrounded (`... &`); the
orphaned `docker rm` outlives the shell and finishes. On a `SIGKILL` of the run
script the containers survive as before, and
`docker rm -f $(docker ps -q --filter name=ri-ns-sidecar-)` still clears them.

Measured together with four interleaved A/B runs of the default 8-rank
configuration, end-to-end script wall time went 5.15s -> 4.42s (-14%), split
roughly 0.5s to the prewarm and 0.2s to the backgrounded teardown.
`total_wall_seconds` went 3.6s -> 2.8s, but only part of that is real: the
prewarm happens *before* `run_start`, so it moves cost out of that window as
well as shrinking it. All eight runs produced identical `log(Z)` and identical
objectives for all 41 evaluations.

### The PolyChord container is a sidecar too

The run scripts do not `docker run` the PolyChord container either: they start
it detached alongside the two data-plane sidecars and enter it with `docker
exec mpirun ...`. A `docker run` of this image costs ~0.7s of create, start,
wait and `--rm` teardown; the `docker exec` costs ~0.03s, and the container's
own startup now happens concurrently with the MeqTrees and WSClean ones.

One more thing rides on that: **the manifest write moved into the gap.**
`scripts/record-environment.sh` is ~0.4s of `git` and `docker image inspect`,
and now runs between `sidecar_launch` and `sidecar_wait` instead of after the
containers are up. `NS_SIDECARS` is exported by `sidecar_launch` rather than
`sidecar_wait` because the container names are known as soon as the launches
are issued.

The `docker info` that resolves `HOST_CPUS` (and doubles as the
daemon-availability check) moved *after* the launches for the same reason -
nothing between the launches and `sidecar_wait` touches a sidecar - which put
`launches-issued` at ~0.005s after script start instead of ~0.075s. Both
scripts then had to move `HOST_CPUS` back in front of the launches, because the
FIFO pairs the containers' commands glob for have to exist first (see "The
workers are started by the container, not by the ranks" above); only the
daemon check is still below them, and `HOST_CPUS` no longer costs a daemon
round trip (see "`HOST_CPUS` comes from `nproc`, not from the daemon" below).

Measured with four interleaved A/B runs of the default 8-rank configuration,
end-to-end script wall time went 6.82s -> 5.29s (-22%); single-rank went 13.1s
-> 12.2s (-7%). All eight runs produced identical `log(Z)` and byte-identical
objectives for all 41 evaluations.

Note which clock that is. `summary.json`'s `total_wall_seconds` - the
number the profile table above totals - is measured around
`run_polychord()` *inside* the container, so it does not see container startup
or teardown at all: it is 3.65s before and after this change. Only
`time scripts/run-nested-sampling.sh` shows it. Anything that moves fixed
setup cost has to be measured end to end.

`run-nested-sampling-r2d2.sh` now does the same - it was the last
`docker run` in either PoC. Measured over six interleaved A/B runs of the
default 8-rank configuration (38 evaluations, identical evaluation multiset and
objectives), end-to-end script wall time went from a 5.93s median (mean 5.96s)
to 5.68s (mean 5.54s), while `total_wall_seconds` was unchanged at ~2.9s - the
whole saving is fixed setup, exactly as the note above predicts.

The R2D2 saving is smaller than the WSClean one because a different cost now
sets the floor. Stage timestamps printed from inside `polychord_r2d2_poc.py`
put the R2D2 PoC's in-container time at ~3.9s, of which:

| From script start | Stage |
|---:|---|
| ~0.1s | the three `sidecar_launch`es are issued |
| ~0.9s | the ranks are up, through `import pypolychord`, and call `warm()` |
| ~2.3s | `warm()` returns: the R2D2 worker has finished importing |
| ~4.7s | `run_polychord()` returns and the summary is written |

`warm()` blocks for ~1.4s, over a third of the in-container time, because
opening a FIFO blocks until the other end is opened and `serve()` opens its
pipes only after its imports. The worker cannot be ready sooner than ~1.9s
after the script starts: ~0.5s for the R2D2 container itself plus ~1.3s of
imports (`python3 -X importtime` in that image: torch 0.89s, `utils` 0.31s of
which lightning 0.07s, torchmetrics 0.05s and scipy 0.06s). So every remaining
front-end saving is capped by that number - shaving setup only makes the ranks
wait longer - and the next real lever is the import itself, or overlapping the
wait with work the sampler could be doing.

### The images ship byte-compiled, so no container needs a warm-up

`sidecar_launch` used to run a throwaway `python3 -c "import numpy,
pypolychord, common, argparse; argparse.ArgumentParser()"` in the fresh
PolyChord container before `sidecar_wait` returned, because the *first* Python
process in a container cost far more than the next one: the real 8-rank
`mpirun python3` exec measured 0.99s cold against 0.22s warm.

That was byte compilation, not the overlay mount. `python:3.11-slim` ships no
`.pyc` for most of the stdlib, and pip left `/opt/venv` with `.pyc` whose
recorded source mtimes no longer match after the image's `COPY`. `docker diff`
on a container that has done nothing but one import shows 60 freshly written
`.pyc` files; eight ranks starting at once each compile the same modules and
race to write them. Reading every file in `/opt/venv` first (`cat` over the
`.so` and `.pyc` files, 0.12s) leaves the 8-rank exec at 0.91s, so it is not
page cache; a `python3 -c pass` first does not help either.

So both images now run `python3 -m compileall` at build time
(`docker/polychord/Dockerfile`, `docker/meqtrees/Dockerfile`). A fresh
container's 8-rank cold import is then 0.22-0.26s with no warm-up at all and
writes zero `.pyc`, and `sidecar_launch` lost its warm-up hook entirely -
`sidecar_wait` now only waits for `docker run`, which the manifest write
already hides. Shell markers put `sidecars-ready` at ~0.47s after script start
instead of ~0.70s.

The MeqTrees image only wrote two `.pyc` (Ubuntu ships them for its own
`python3` packages, and one of the two is the `Timba` module the Dockerfile
patches), so `compileall` there is cheap insurance rather than a measured win.

Ten interleaved A/B pairs of the default 8-rank run, rebuilding both images
between arms: 3.31s -> 3.11s end to end (-6%), 9/10 pairs in the right
direction. The 41 evaluations' parameter sets, science metrics and objectives
and the PolyChord `.stats` output are identical; only `eval_id` ordering and
the per-evaluation `wall_seconds`/`peak_memory_bytes` differ, as they do
between any two runs.

Also measured and rejected: warming the MeqTrees and WSClean sidecars is not
worth it even before `compileall` (cold-vs-warm first exec 0.17s vs 0.13s and
0.09s vs 0.08s, and warming the Timba/casacore imports moved eight concurrent
`--serve` worker startups only 0.34s -> 0.32s for 0.22s of warm-up); warming
with `mpirun -np 8` cost 1.0s instead of 0.28s with no gain in the real exec.

Also measured and rejected: **`wsclean -j 2` and above.** The imaging binary is
the largest per-evaluation stage (~0.11s of a ~0.16s warm evaluation) and
`-j 4` runs it in ~0.09s on an idle host, but multi-threaded gridding changes
the summation order: the image, dirty, residual and PSF pixels differ from the
`-j 1` run at the float32 rounding level (~1e-7 on a peak of 1.0). The
objective is `off_source_rms_jy` ~ 8e-6, so a 1e-7 pixel shift moves the
likelihood and takes the whole sampler down a different path. This repository
is about reproducibility; `-j 1` stays.

#### The R2D2 image was left out, and it was the expensive one

`compileall` landed in `docker/polychord/Dockerfile` and
`docker/meqtrees/Dockerfile` when those were the images a run started fresh.
`docker/r2d2/Dockerfile` never got it, and by the time the imaging pool became
the thing evaluation one waits on it was the one place where it mattered most:
`python:3.10-slim-bookworm` ships 1483 stdlib `.py` and **zero** `.pyc`, and
the R2D2 checkout ships 48 `.py` and zero `.pyc`. Only the venv was compiled,
because pip does that itself (16942 of 16943).

So the first Python process in a fresh R2D2 container compiled every stdlib
module `import torch` reaches - `asyncio`, `email`, `http`, `inspect` and the
rest - and wrote it into the container layer:

| | |
|---|---:|
| `import torch`, first process in a fresh container | 0.887s |
| `import torch`, second process in the same container | 0.615s |
| `docker diff` after one `import torch` | 222 entries |

Same three negative controls as the section above, all on a fresh container:
`cat` over every file under `torch/` (0.20s) leaves it at 0.886s, so it is not
page cache; `python3 -c pass` first leaves it at 0.893s and a `ctypes.CDLL` of
`libtorch_cpu.so` at 0.889s, so it is not the loader; allocating and freeing
400MB leaves it at 0.888s, so it is not first-touch of anonymous memory. Only a
previous full `import torch` in the same container helps, which is what `docker
diff` explains.

With `RUN python3 -m compileall -q "/usr/local/lib/python${PYTHON_VERSION}"
/opt/r2d2/R2D2-RI/src || true` in the runtime stage, a fresh container imports
torch in 0.64s and writes 3 diff entries. Ten interleaved pairs of the default
8-rank run against the same image built without that line (`R2D2_IMAGE` picks
the arm, so no rebuild between runs):

| | with `compileall` | without | wins |
|---|---:|---:|---:|
| end to end | 2.627s | 2.905s | 10/10 |
| `run_polychord` wall | 1.408s | 1.695s | 10/10 |
| pool ready, from script start | 1.619s | 1.908s | 10/10 |
| pool warm-up alone | 1.023s | 1.186s | 10/10 |

The warm-up's own 0.163s is the stdlib torch reaches; the rest of the 0.289s is
`r2d2_serve.py`'s own imports before it, which pay the same stdlib. log(Z) is
identical (0.999287799533384E+002) and so are every evaluation's parameters,
objective and error; only the timings and `peak_memory_bytes` move, and the
latter *down* (283.0MB -> 280.6MB), because the worker no longer compiles
bytecode in-process.

`compileall` there is not insurance the way it is in the MeqTrees image, which
gets Debian's own `.pyc` for its `/usr/lib/python3` stdlib (3242 of 3242).
Check a new image with `docker run --rm --entrypoint sh <image> -c 'python3 -c
"import sysconfig;print(sysconfig.get_paths()[\"stdlib\"])"'` and count `.pyc`
under what it prints.

### `mpi_rank()` reads the launcher's environment first

`from mpi4py import MPI` initialises MPI, and eight ranks doing that at once
costs 0.24s each. `prewarm()` needs the rank before anything else has touched
MPI - it is how a rank finds its own FIFO pair - so calling
`mpi_rank()` there added that 0.24s to every rank's pre-sampler startup, where
it hid from `total_wall_seconds` (measured around `run_polychord()`) and showed
up only end to end. `mpi_rank()` therefore reads `OMPI_COMM_WORLD_RANK`, which
OpenMPI's launcher exports, and only falls back to `mpi4py`.

### MPI picks its transport by search unless it is told not to

`from mpi4py import MPI` runs `MPI_Init`, and on this host that cost 0.25s on
every one of the eight ranks at the same moment. Nothing in this repo imports it
on the sampler path any more (see above), so where it lands is inside PolyChord:
`pypolychord` imports `mpi4py` lazily, and the whole 0.25s therefore shows up as
the first `run_polychord()` call taking that long to reach its first likelihood,
with a second call in the same process reaching it in ~0.3ms. `strace` on the
first call is what identifies it - 2796 `openat`s and 903 `clock_nanosleep`s
worth 0.17s, none of them PolyChord's. It is not the transport itself:
`OMPI_MCA_pml=ob1` takes the slowest rank's `MPI_Init` to 0.05s, and
`OMPI_MCA_pml=^ucx,cm` - the image's `/etc/openmpi/openmpi-mca-params.conf`
already excludes `ucx` - to 0.046s, so the 0.19s is Open MPI opening the `cm`
PML, which opens the MTL framework, which has libfabric probe every provider it
can find. This job never leaves one container, and `ob1` over shared memory is
what the search settles on anyway, so both run scripts name it:

```
docker exec ... -e OMPI_MCA_pml=ob1 ...
```

Adding `-e OMPI_MCA_btl=self,vader` on top buys a further ~0.007s and was left
out. Output is bit-identical: the same 41 evaluations with the same metrics, and
`.stats`, `.txt`, `_dead-birth.txt`, `_phys_live.txt` and `_equal_weights.txt`
all compare equal against a run without it.

Forty interleaved A/B pairs: 3.205s -> 3.068s median, paired difference
-0.136s +/- 0.029s, 29/40 pairs. Instrumented runs put the rank's
`run_polychord()`-to-first-likelihood gap at 0.234-0.266s without the setting
and 0.022-0.053s with it, 6/6 pairs; end to end recovers rather less than that
because the ranks reach the join spread over ~0.2s.

An earlier 24-pair A/B of this same change read -0.007s and nearly sent it to
the bin. Both arms had been run against a meqtrees image left over from a
reverted experiment that moved `warm_forest()` to the other side of the worker's
FIFO open - which is precisely where the rank's wait comes from, so it masked
the effect. **Rebuild every image the arms depend on before an A/B, not just the
one being changed**. That used to cost ~2s per image against runs that are
~3s each; since builds skip on an unchanged input hash it is ~0.08s, so there
is no longer a reason not to.

### The critical path is the simulate worker's warm-up, not the rank

Absolute timestamps taken on both sides of the FIFO at once - in the rank's
`prewarm()` and inside the `--serve --fifo` worker - line up like this on a
default 8-rank run (seconds from the run script starting, worker running from
the bind mount so its imports are ~0.15s slower than the baked-in copy):

| Marker | When |
|---|---:|
| worker process enters `serve()` | 0.58-0.66 |
| worker's `meqserver_session()` returns | 0.69-0.88 |
| rank reaches `prewarm()`'s join | 0.76-0.93 |
| worker's `warm_forest()` returns, worker opens its FIFOs | **1.00-1.20** |
| rank's `simulate_worker()` returns | 1.00-1.20, to the millisecond |
| worker reads request one | 1.00-1.20, to the millisecond |

Every rank-side marker lands *before* the worker is ready, and the rank's
connect and the worker's FIFO open are the same instant, so the rank spends
0.2-0.3s blocked in the join. Anything a rank does *before* the join is
therefore free, and anything after it is not: `OMPI_MCA_pml=ob1` (above) sits
after the join and is worth 0.136s end to end, while paying PolyChord's one-time
setup with a throwaway `run_polychord()` before the join was measured at
-0.007s +/- 0.026s over 20 pairs and dropped, as was opening the worker's FIFOs
before its warm-up instead of after (0.35s off the join, 0.00s end to end - it
only moves the same wait into evaluation one).

It also explains an apparent regression: `OMPI_MCA_pml=ob1` takes the eight
`eval_id == 1` `simulate_seconds` records from ~0.06s to ~0.14s and summed
simulate worker-seconds from 2.25 to 2.88. Nothing got slower. The rank starts
timing when it writes the request and the worker reads it at the same absolute
moment either way, so a rank that reaches evaluation one earlier simply measures
more of the wait. Steady-state `simulate_seconds` is ~0.05-0.06s in both.

The next lever is therefore inside the worker: ~0.32s of `docker run` before its
command starts, then ~0.07s of interpreter and imports (with the baked-in,
byte-compiled copy), ~0.10s of meqserver, and ~0.25s of `warm_forest()`. Only
the last two are ours.

#### How much slack each branch of the startup has

Two independent branches converge before the first likelihood, and only the
slower one is on the clock. Measured with `PS4='+[${EPOCHREALTIME}] ' bash -x`
on the run script plus in-process markers, seconds from the script starting:

| Branch | Steps | Ready at |
|---|---|---:|
| worker | `docker info` 0.045 -> `docker run -d` issued -> container command starts +0.26-0.35 -> worker ready +0.51 | **0.85-0.95** |
| rank | ... -> all three `docker run -d` return +0.30-0.53 (the manifest's 0.24-0.42s hides inside it) -> `docker exec` + `mpirun` + interpreter + imports +0.20-0.27 | 0.55-0.80 |

So the rank branch has 0.25-0.40s of slack: work moved onto it is free until it
becomes the binding branch, and work taken off it buys nothing. The worker
branch is what to attack, and inside the worker's 0.51s the split is ~0.13s of
interpreter and imports, ~0.15s of `meqserver_session()`, ~0.12s of `makems` and
~0.20s of TDL compile plus first predict.

The tail is 0.085s: that is what elapses between the last rank's `atexit` and
`docker exec` returning - CPython finalisation, `MPI_Finalize`, `mpirun` reaping
and the exec stream closing. Nothing in the repo runs during it.

#### Measured and rejected inside the worker's warm-up

- **Building the warm-up MS while the meqserver starts.** `makems` is a
  subprocess this process only waits on and `meqserver_session()` is Timba
  imports plus another child, so running them in two threads should have taken
  ~0.12s off the serial path. It does - for the *fastest* worker. Time to ready
  across eight concurrent workers went min 0.489 -> 0.413 but max 0.504 ->
  0.508, and 30 interleaved end-to-end pairs read +0.020s +/- 0.030s. The eight
  workers contend (one worker alone is ready in 0.35-0.40s against 0.50-0.54s
  for eight), so reordering within a worker just concentrates the contention
  instead of removing work. This is the same null result as baking a ready-made
  MS into the image, for the same reason.
- **Importing `mpi4py` eagerly, before `prewarm()`'s join.** `mpirun -np 8
  python3 -c "from mpi4py import MPI"` costs 0.20s against 0.07s for `-c pass`,
  so `MPI_Init` looked like ~0.13s sitting after the join. It is not: markers
  show only ~0.02s between the last rank leaving the join and the first
  likelihood call, and forcing the import early made first-likelihood 0.417 ->
  0.447. Whatever the standalone `mpirun` measurement is paying for, the real
  run does not pay it there.
- **Further Open MPI MCA tuning.** On top of `OMPI_MCA_pml=ob1`, none of
  `btl=self,vader`, `osc=sm`, `coll=basic,libnbc,self`,
  `hwloc_base_binding_policy=none` or `rmaps_base_mapping_policy=slot` moves
  `mpirun -np 8 python3 -c "from mpi4py import MPI"` outside 0.19-0.22s.
- **Container `docker run` options.** Best-of-four time from `docker run
  --detach` to the container's command running is 0.26s minimal, 0.26s with
  `--shm-size 512m`, 0.26s with the repo bind mount added and 0.34s on the
  default bridge network. `--network none` is the only one that matters and is
  already in use; 0.26s is this host's rootless-Docker floor.

### Long-lived sidecar containers, one per image

`docker run` costs ~0.40s of create/start/teardown on this host regardless of
image, mounts or `--platform`, while `docker exec` into an already-running
container costs ~0.03s. The MeqTrees simulate and WSClean imaging sidecars are
both short work against bind-mounted paths, so each runs in a detached `sleep
infinity` container that lives for the whole run - through the long-lived `sh`
above for WSClean, through the `--serve` worker for simulate. Both run scripts
source `scripts/lib/start-sidecars.sh` and start those containers before the
PolyChord container, handing the names to every rank in `NS_SIDECARS`;
`sidecar_container()` in `common.py` still starts one itself for any image
that is not in there.

That container mounts `REPO_ROOT` at its own host path (the same trick the
PolyChord container already uses), so sidecar arguments are plain absolute
paths instead of the old per-evaluation `-v {eval_dir}:/work` plus `/work/...`.
`sidecar_command()` reads the image's `ENTRYPOINT` back with `docker inspect`
rather than restating the Dockerfile, since neither `docker exec` nor the
sidecar shell applies it, and each evaluation runs in its own working directory
so anything a sidecar writes relative to the cwd still stays per-evaluation -
except the simulate worker, which outlives any one evaluation and so runs in
`REPO_ROOT` and writes only absolute paths.
Pre-started containers are removed by the run script's `EXIT` trap, and any a
rank started itself via `atexit`; a `SIGKILL`ed run leaks sleeping containers,
cleaned up with `docker rm -f $(docker ps -q --filter name=ri-ns-sidecar-)`.

One container per image, rather than one per rank, is what makes the start
cheap. A single `docker run` of these images costs ~0.36s here, but 16 of them
at once - which is what 8 ranks x 2 images did the moment the ranks came up -
costs 1.3s, and all of it lands in front of the first evaluation. Separate
`docker exec` processes in one container are already isolated: 8 concurrent
`--serve` workers in a single MeqTrees container, each with its own meqserver,
run without interfering. Measured over 3 runs each, the default 8-rank run went
5.44s -> 3.64s (-33%) with identical `log(Z)`, the same 41 evaluations and
byte-identical metrics for every one of them.

The WSClean call also dropped `run_docker_monitored()`'s `docker stats`
sampler. GNU `time -v` already runs inside that container and reports an exact
peak RSS, where the 0.2s-interval sampler both missed short peaks and delayed
noticing the process had exited. R2D2 later stopped using
`run_docker_monitored()` too - see "R2D2 imaging runs in a long-lived worker"
below - and with no callers left the helper and its `docker stats` parsing are
gone from `poc_common.py`.

Together these took the profiled run from 43.5s to 27.5s (-37%) with identical
per-evaluation metrics, identical `log(Z)`, and the same set of files in every
evaluation directory. The R2D2 run picks up the simulate half of this through
the shared `simulate_measurement_set()`; its MS-to-`.mat` convert and imaging
step now go through sidecars of their own (see below).

### The R2D2 MS-to-`.mat` convert runs in the MeqTrees sidecar

`polychord_r2d2.py` used to convert `sim.ms` to `r2d2_data.mat` with its own
`docker run` of the MeqTrees image, once per evaluation - even though the same
run already keeps a MeqTrees sidecar warm for the simulate worker. Measured on
this host, a fresh `docker run --network none` of that image reaching a bare
`python3 -c pass` costs 0.54-0.68s against 0.11-0.20s for a `docker exec` into
the running sidecar.

It now calls `sidecar_run()`, the same helper the WSClean run uses, so the
request is a line written to this rank's already-attached `sh` rather than a new
container. Because the sidecar bind-mounts `REPO_ROOT` at its host path, the
converter takes the evaluation's absolute `sim.ms` / `r2d2_data.mat` paths
directly instead of the old per-evaluation `/work` mount. Measured
`convert_seconds` over a five-evaluation run: 0.245-0.296s, against ~0.8s for
the `docker run` path (~0.55s of container start plus the converter's own
~0.25s of casacore and scipy imports). The `.mat` files are unchanged.

#### And then into the simulate worker itself

A `docker exec` still bought a fresh interpreter and a fresh set of imports.
Measured inside the warm sidecar against a four-channel `sim.ms`, the whole
`docker exec python3 ms_to_r2d2_mat.py` round trip is 0.13-0.16s, of which the
conversion itself - `table.getcol` through `savemat` - is 0.009-0.027s. The rest
is the exec, the interpreter and ~0.10s of numpy, casacore and scipy imports
that the rank's simulate worker already holds live, having just written the MS.

`simulate_point_source_ms.py`'s serve loop therefore takes a second kind of
request, `{"action": "convert", "argv": [...], ...}`, dispatched by
`handle_request()` to `ms_to_r2d2_mat.main(argv)`; `poc_common.convert_ms_to_mat()`
is the caller-side half, over the same worker pipe `simulate_measurement_set()`
uses. `ms_to_r2d2_mat` is imported on first convert rather than at module scope,
so the WSClean PoC's worker never pays for scipy. Measured over four requests to
one worker: 0.49s for the first (cold container filesystem; the import itself is
0.032s), then 0.011s, 0.019s, 0.011s - against 0.13-0.16s each for the
`docker exec` path. The `.mat` contents are identical field for field; the files
differ only in the creation timestamp `savemat` writes into the header.

That leaves nothing fixed in this stage: at ~0.015s a request the convert is now
the cheapest of the three per-evaluation stages.

### R2D2 imaging runs in a long-lived worker, not a container per evaluation

`polychord_r2d2_poc.py` used to image with one `docker run` of the R2D2 image
per evaluation. Warm (the 5.6GB image already in page cache - a cold cache makes
the first few evaluations take 84s, 29s, 31s, 18s, and any timing taken there is
meaningless) that round trip costs ~2.1s on this host, of which almost none is
science: ~0.5s of container create/start and ~1.3s of `import torch` plus the
R2D2 module imports, repeated every time.

`scripts/lib/nested_sampling/r2d2_serve.py` is the R2D2 equivalent of the
simulate side's `--serve` worker: one process per rank, inside a shared R2D2
sidecar, running one imaging job per JSON request line
(`{"argv": [...], "stdout": path, "stderr": path}` in, `{"returncode": int,
"peak_memory_bytes": int}` out). Upstream's `src/imager.py` has no importable
entry point - its whole body sits under `if __name__ == "__main__"` - so each
request re-runs that body with `runpy.run_path(..., run_name="__main__")`, and
its imports come back from `sys.modules`. `poc_common.run_r2d2_imaging()` is the
caller-side half, shaped like `sidecar_run()`.

Measured against the same `.mat`, `--ckpt_path` pointing at an empty
`checkpoints/R2D2_A1` so both paths stop at the same place (`get_DNNs`, "Checkpoint
for N1 not found"), `R2D2_OMP_THREADS=1`:

| Path | Per evaluation |
|---|---|
| `docker run` per evaluation | 2.08s, 2.15s, 2.15s |
| worker, first request on a rank | 1.49s |
| worker, steady state | 0.10s, 0.15s, 0.15s |

So ~1.95s per evaluation, against a total ~2.3s of measured fixed overhead
before it. What is left in the 0.10-0.15s is the whole of the actual imaging
work - data load, imaging weights, measurement operator, operator norm - plus,
once checkpoints are present, the 25 UNet forward passes this measurement could
not run. That part is unchanged; only the fixed cost in front of it is gone.

Two consequences of a worker holding a whole rank rather than one evaluation:

- `peak_memory_bytes` for R2D2 is now the worker's own high-water RSS
  (`ru_maxrss`), not a `docker stats` sample of a per-evaluation container. It is
  a running maximum across a rank's evaluations, so only the first evaluation on
  a worker reports exactly what the container did. `docker stats` was a poor
  source anyway: each `--no-stream` call takes ~2.0s on this host because the CLI
  waits for two samples.
- The checkpoints mount point stays `/checkpoints` rather than becoming the host
  path, because `ckpt_path` is recorded in every `poc-summary.json` and compared
  across runs by `merge-nested-sampling-runs.py`; a host path there would stop
  runs from different checkouts merging. The evaluation's own `data_file` and
  `output_path` do become absolute host paths, the same way the convert step's
  did.

`r2d2_serve.py` runs from the repository bind mount, not from a copy baked into
the R2D2 image, so editing it needs no 5.6GB rebuild.

Not done here, and worth a look next: the worker is started on this rank's first
evaluation, so that ~1.5s lands on the wall clock in front of evaluation one on
every rank. The WSClean PoC hides the same cost behind PolyChord's own startup
with `prewarm()`; the R2D2 PoC calls neither that nor anything like it.

### R2D2 sizes its own torch thread pool, and ignored the env that limited it

Every measurement above was taken one worker at a time. Under the default 8
ranks the same imaging step took **29.1s per evaluation**, not 0.15s, and the
run was 3m 06s of wall clock for 38 evaluations.

The cause is upstream's `set_common_args()` in `src/utils/args.py`: it calls
`torch.set_num_threads(len(psutil.Process().cpu_affinity()))` on every imaging
run unless `ncpus` is set. `torch.set_num_threads()` overrides `OMP_NUM_THREADS`
- which is the only lever the worker's `docker exec` had - so each rank asked
torch for all 20 host CPUs regardless, and the 8 ranks ran ~160 compute threads
on 20 cores. Iteration-1's measurement of `get_op_norm()` at 3.67s on 20 threads
against 0.09s on one is the same effect inside a single process.

`polychord_r2d2_poc.py` therefore writes `ncpus: <R2D2_OMP_THREADS>` into every
per-evaluation `r2d2_config.yaml`; `ImagingArgs` takes it straight from the
YAML, so no new flag is needed, and the imaging log now says `avaiable cpus 20,
request cpus 2` instead of `avaiable cpus 20`. Measured end to end
(`NS_MPI_PROCS=8`, 38 evaluations, identical parameters and objectives before
and after):

| | Wall clock | r2d2 per evaluation |
|---|---:|---:|
| `OMP_NUM_THREADS` only | 3m 06s, 3m 55s | 29.1s |
| plus `ncpus` | 8.4s, 8.4s | 0.56s |

That 0.56s still carries each rank's ~1.5s first request; steady state is ~0.3s.

The oversubscription cliff is sharp, and it is about the product not the
per-rank count. Sweeping `R2D2_OMP_THREADS` at 8 ranks on this 20-CPU host:
1 thread 7.7s, 2 threads (the old `host CPUs / NS_MPI_PROCS` default) 8.4s, 4
threads 24.2s. A later three-repeat search measurement found 3 threads at
7.43s/evaluation versus 8.12s at 2 threads, so automatic allocation now rounds
the per-rank CPU share up. A current three-repeat explicit four-thread probe
measured 7.39s/evaluation versus 7.45s at 3 threads, but four threads still
oversubscribes this host and remains an explicit candidate rather than the
automatic setting. The
seven-thread follow-up measured 7.62s/evaluation (0.7956 eval/s median) versus
7.45s at 3 threads and 7.39s at 4 threads, so it is not a candidate either.
The eight-thread follow-up measured 7.72s/evaluation (0.7493 eval/s median),
also slower than four threads, so it is rejected.
1-thread column of that sweep is
misleading, though - most of what it was measuring was OpenMP spin-waiting, not
the thread count; see "The imaging workers' OpenMP threads sleep between
requests" below.

### The R2D2 PoC warms its two workers before the sampler starts

`polychord_wsclean_poc.py` had called `prewarm()` since the section above;
`polychord_r2d2_poc.py` never had, so both of its workers were started lazily by
the first request that needed one, and every rank hit that moment together.
`poc-summary.json` showed it plainly: across the eight `eval_id == 1` records,
`simulate_seconds` was ~0.58s against a ~0.045s median for the rest of the run
and `image_container_seconds` was ~1.78s against ~0.19s - ~2.3s of pure startup
sitting in front of evaluation one on every rank at once.

Neither worker needs to be *ready* for the prewarm to pay: spawning the process
is enough, because the expensive part happens inside it. `r2d2_serve.py` imports
torch and the R2D2 modules before it reads its first request line, and the
simulate worker starts its meqserver the same way, so both run while the rank is
still doing `import pypolychord` and PolyChord's own setup.

`prewarm()` therefore now takes callables rather than the two image names it was
hard-coded to - the R2D2 PoC warms `simulate_worker()` and `r2d2_worker()`, the
WSClean PoC warms `simulate_worker()` and its wsclean `sh`, and the same
no-touching-a-sidecar-between-call-and-join rule applies to both. Measured
(`NS_MPI_PROCS=8`, 38 evaluations, byte-identical evaluation set and log(Z)
before and after, three runs each):

| | Sampler wall clock | End to end |
|---|---:|---:|
| lazy worker start | 6.9s, 7.5s | 9.2s, 9.7s, 10.3s |
| prewarmed | 5.0s, 4.9s, 4.5s | 6.8s, 7.2s, 6.6s |

What is left in evaluation one is what the overlap window could not cover: the
imaging worker's `import torch` is ~1.3s and eight of them run at once, so
`image_container_seconds` on the first evaluation is still ~1.2-1.7s. The next
section closes that.

### Both R2D2 worker pools are started by their container, not by the ranks

`run-nested-sampling-poc.sh` had started the simulate workers as the meqtrees
container's own command over a FIFO pair per rank since "The workers are started
by the container, not by the ranks" above; `run-nested-sampling-r2d2-poc.sh` had
neither pool, so even after `prewarm()` its ranks still started both workers
themselves, inside a PolyChord container that does not exist until both sidecars
are up.

It now creates `<output-dir>/.simulate-workers` and `<output-dir>/.r2d2-workers`
with one `<rank>.in`/`<rank>.out` pair each, and gives both sidecars a container
command that spawns one worker per pair - the same
`simulate_point_source_ms.py --serve --fifo <base>` line the WSClean script
uses, and a new `r2d2_serve.py` (`--fifo <base>` at the time these numbers were
taken; the subsection below replaced that with one `--fifo-dir` process that
forks the pool). The head start is everything the
run still has to do afterwards: the manifest, the PolyChord `docker run`,
`mpirun`, `import pypolychord` and PolyChord's own setup. Measured
(`NS_MPI_PROCS=8`, 38 evaluations, identical evaluation set and log(Z) to the
digit, three runs each):

| | eval 1 `image_container_seconds` | eval 1 `simulate_seconds` | Sampler wall clock | End to end |
|---|---:|---:|---:|---:|
| ranks start both workers | ~2.0-2.7s | ~0.5-0.7s | 5.49s, 4.88s, 5.55s | 7.37s, 6.58s, 7.63s |
| containers start both pools | ~0.24-0.28s | ~0.05-0.07s | 2.58s, 2.49s, 2.28s | 5.65s, 5.51s, 5.14s |

Evaluation one is now indistinguishable from the steady state (imaging median
0.242s, first-evaluation maximum 0.279s), so no R2D2 startup cost reaches the
sampler's wall clock at all - it is 2.2x faster for that reason alone, and the
~2.9s that remains end to end is image-build checks, `docker info`, the three
container starts and the manifest, none of which is per-evaluation.

Two details differ from the WSClean pool. `r2d2_serve.py` takes its thread caps
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) from the
container's `docker run -e` rather than from a per-rank `docker exec -e`, which
loses nothing: torch and finufft read them at import time and every rank was
given the same value anyway. And `serve()`'s FIFO branch does no extra warm-up
before opening its pipes the way the simulate worker's `warm_forest()` does -
the imports it already does eagerly are the whole cost, and the remaining
per-request work needs the request's parameters.

`_connect_shell_started_worker()` now takes the name of the environment variable
holding its pool directory, so `simulate_worker()` and `r2d2_worker()` share it;
both keep the "spawn my own if there is no pool" fallback unchanged.

#### The R2D2 pool forks from one warm-up instead of importing eight times

Starting one `r2d2_serve.py --fifo <base>` interpreter per rank means eight
`import torch` at the same moment, and they contend: measured inside the R2D2
sidecar, `import optimiser, utils` is 0.885-0.899s on its own but 1.05-1.61s
across eight concurrent interpreters. The sampler waits for the slowest, so the
run paid the 1.61s, not the 0.89s.

`r2d2_serve.py --fifo-dir <dir>` replaces the shell loop: it imports once, then
forks one child per `<rank>.in`/`<rank>.out` pair in the directory and each child
serves its pair exactly as `--fifo` did. Interleaved A/B, alternating the two
run scripts (`NS_MPI_PROCS=8`, 38 evaluations, identical evaluation set in every
run):

| | End to end |
|---|---:|
| one interpreter per rank | 5.37s, 5.12s, 5.53s, 5.44s, 5.18s |
| forked from one warm-up | 4.22s, 4.83s, 4.72s, 4.61s, 4.42s |

Median pairwise saving ~0.8s, ~15% of the run. The children also share the
~300MB of imports copy-on-write rather than holding a copy each, which is what
makes the rank count cheap to raise.

One thing the fork breaks and has to put back: a child starts a fresh
`ru_maxrss` counter even though it starts holding all of the parent's resident
pages, so the same imaging request reported 196MB from a pool worker against
303MB from a worker that had imported for itself. `serve_pool()` records the
warm-up's high-water mark before forking and `peak_memory_bytes()` returns the
larger of that and the worker's own, which brings it back to 302MB - the metric
`poc-summary.json` records is unchanged in meaning. `--self-check`'s
`self_check_serve_pool()` guards both halves: it asserts the warm-up ran exactly
once for two workers, and that a 64MB allocation made only during that warm-up
still shows up in a child's reply.

### The three no-op image builds run concurrently

`make nested-sampling-r2d2-poc` builds its three images before the run. On an
unchanged tree every one of them is a no-op, and a no-op build is not free:
buildkit still resolves the `docker/dockerfile:1` frontend and the base image's
metadata, walks ~20 `CACHED` steps and re-exports and re-tags the manifest.
Measured on this host, `scripts/build.sh` costs 1.36-1.59s per image with
nothing to do, so the three in series were ~4s in front of a ~4.5s run - the
largest single cost left in the command a user actually types.

They have no dependency on each other, so the two PoC targets now run them as
`+$(MAKE) -j3 ...` rather than as prerequisites; the sub-make gets the
parallelism without the caller having to remember `-j`. Interleaved A/B of
`make nested-sampling-r2d2-poc`, alternating the two Makefiles:

| | End to end |
|---|---:|
| builds in series | 8.10s, 8.13s, 8.76s, 8.23s |
| builds concurrent | 6.26s, 6.28s, 5.95s, 6.43s |

~1.9-2.8s in every pair, ~23% of the command. The build step itself goes
3.3-4.2s to ~2.05s; the rest of the spread is the run.

#### And then they do not run at all unless an input changed

Concurrency only shrinks three no-op builds down to the slowest one, ~2.05s,
which was still ~37% of the command. `scripts/build.sh` now hashes the files a
build actually reads - the Dockerfile plus whatever it `COPY`s or bind-mounts
from the context, with the platform and build args mixed in - records that hash
in an `ri.build-inputs` label on the image it produces, and skips `docker build`
entirely when the tag already names an image carrying that hash. One `docker
image inspect` answers both questions that matter, "built from these inputs?"
and "does it still exist?", so a `docker rmi` cannot leave a stale skip behind.
`FORCE_BUILD=1` builds regardless.

| `scripts/build.sh <image>` | build | skip |
|---|---:|---:|
| r2d2 | 2.04s | 0.09s |
| meqtrees | 1.80s | 0.08s |
| polychord | 1.57s | 0.09s |
| wsclean | 2.46s | 0.08s |

`make -j3 build-r2d2 build-meqtrees build-polychord` goes 2.07s to 0.08s, and
`make build` - all four images - to 0.28s. Interleaved A/B of the whole command,
with `FORCE_BUILD=1` as the always-build arm:

| | End to end |
|---|---:|
| always build | 5.80s, 5.39s, 5.40s |
| skip unchanged | 3.38s, 3.46s, 3.40s |

~2.0s in 3 of 3 pairs, ~37% of `make nested-sampling-r2d2-poc`. `log(Z)` and the
38-evaluation parameter set are bit-identical to a run from before the change,
which they have to be - the images are the same images.

Two caveats worth keeping in mind. The input list is written out per image in
`build.sh` and has to stay in step with the Dockerfiles; `grep -n 'COPY\|--mount=type=bind' docker/*/Dockerfile`
is the check, and file *names* go into the hash as well as contents so that
renaming a file inside a `COPY`ed directory is a rebuild rather than a false
skip. And the hash deliberately does not cover what Docker's own layer cache
does not cover either - `apt-get`/`pip` output drifting under a pinned base
image, or a moved upstream git ref - so those still need `FORCE_BUILD=1` or a
`docker rmi`, exactly as they previously needed `--no-cache`.

#### Where the remaining R2D2 wall clock is, and why it is a floor

Stage timestamps from inside `polychord_r2d2_poc.py` on a 4.55s end-to-end run
(`NS_MPI_PROCS=8`, 38 evaluations), relative to the run script starting:

| | |
|---|---:|
| ranks reach `main()` | 0.78s |
| `warm()` returns - the R2D2 worker is answering | 2.00s |
| `run_polychord` returns | 3.66s |
| script exits | 4.55s |

Both halves are accounted for and neither has slack left:

- **Worker readiness, ~2.0s.** Measured directly against a bare `docker run
  --detach` of the R2D2 image: 0.36s until the container's own `python3` starts
  and a further 1.24s of `import optimiser, utils`, of which `python3 -X
  importtime` attributes 0.877s to `torch` alone and 0.309s to `utils`. The
  ~0.15s left is the run script's own preamble (`defaults.sh` 0.02s, the
  daemon-check `docker info` 0.08s - since replaced by `nproc`, see below).
  Everything else the script does before the
  first evaluation - the manifest write, the other two containers, `mpirun`,
  `import pypolychord` - already fits inside that window. Two things measured
  and rejected: `OMP_WAIT_POLICY=PASSIVE` on the R2D2 sidecar (2 wins, 3 losses
  over 5 pairs) and launching the R2D2 sidecar before the other two (worker
  ready 2.03s either way over 6 pairs, because `sidecar_launch` already
  backgrounds each `docker run`). The 1.24s of R2D2 imports is now 1.10s - see
  "The imaging worker warms what `imager.py` imports, and no more" below.
- **Sampler, ~1.7s, which is one rank's evaluations.** The `eval_id` histogram
  of a run is seven ranks with 5 evaluations and one with 7, and 7 x ~0.3s is
  the whole sampler wall - PolyChord's own overhead is in the noise. Of that
  ~0.3s, imaging is ~0.25s, simulate ~0.05s and the convert ~0.015s. The FIFO
  round trip is not part of it: the worker's own measurement of each request
  sums to within 0.1% of what the rank measures around it.

The imaging request is ~0.10s on an idle worker and ~0.25s at the real
concurrency; the gap is contention, not overhead. Eight forked workers running
the same request in one container measure 0.104s solo against 0.352s eight-way
at two threads each, and 0.129s against 0.181s at one thread - so one thread is
27% faster per evaluation and ~38% higher aggregate throughput at eight ranks,
which is the same direction the end-to-end A/B shows (1 thread won 4 of 5
pairs). The default is still `HOST_CPUS / NS_MPI_PROCS`, unchanged: this is all
checkpoint-less evidence, where the whole imaging run is 0.096s of
`meas_op.get_op_norm()` - a ~47-step power iteration over 95 small finufft
transforms that parallelises badly - and the 25 UNet forward passes that a real
run adds are the part that might actually want the threads. What the
measurement does settle is that this is a shared-host contention question and
not another thread-pool bug like the `ncpus` one: an imaging worker holds 5
threads, and torch's inter-op pool is never created.

**The same floor, re-measured after the changes below.** Two temporary probes
make it directly observable: `date +%s.%N` as the first thing the R2D2
sidecar's container command runs, and a `time.time()` pair around
`warm_imports()` in `serve_pool` written to a file under the run's output
directory. On a 2.63s end-to-end run:
the R2D2 container's own command starts 0.52s after the script does, its FIFOs
are open 0.09s later, and `warm_imports` returns at 1.62s; the ranks reach
their first imaging request at ~0.95s and block on the difference, which is why
`eval_id == 1` `image_container_seconds` is ~0.83s against a ~0.07s steady
state. Everything else has ~0.8s of slack behind that, and three attempts to
spend it came to nothing:

- **Launching the R2D2 sidecar first** (8 pairs): container command start 0.60s
  -> 0.56s, pool ready 2.02s -> 1.98s, end to end 3.85s -> 3.84s. Same result
  as the 6-pair test above, one image later.
- **Launching it first *and* waiting for its `docker run` before issuing the
  other two** (10 pairs): `run_polychord` wall 1.630s -> 1.480s (10/10), pool
  readiness unchanged (1.824s -> 1.827s), end to end unchanged (2.821s ->
  2.829s, 4 wins/5 losses). It moves the rank's wait out of the sampler and
  into the script's preamble and nothing else - a warning about reading the
  sampler wall clock on its own.
- **The transforms themselves.** At the eight-way concurrency a real run has,
  `finufft` `nthreads=1` and `nthreads=2` are a tie (0.058-0.067s per request
  either way over 3 trials of 15 requests x 8 workers), and `upsampfac=1.25` -
  28% faster per transform solo, and not bit-identical, so out of bounds anyway
  - is *slower* at eight (0.065s against 0.060s). A solo request is 0.031s of
  which 0.023s is 37 `finufft` `execute` calls.

#### Pool readiness is 1:1 on the run's wall clock

"Judge a startup-side change on pool readiness" (above) is now calibrated.
Inject a `time.sleep(0.25)` in front of `warm_imports()` in `serve_pool` -
`r2d2_serve.py` runs off the repo bind mount, so this is a file swap and no
rebuild - and interleave it against the stock file: end to end goes 2.54s ->
2.80s, **+0.26s from +0.25s of delay, 6 pairs out of 6**. The exchange rate is
1:1, so a second taken off the R2D2 pool's readiness is a second off the run,
and a second taken off anything else is worth whatever slack that branch has -
which, for the ranks, is the ~0.4s they already spend blocked.

The timeline behind that, measured by wrapping `prewarm`'s two targets in
`polychord_r2d2_poc.py` with `time.time()` probes (a temporary patch; the
polychord image has to be rebuilt for it) on a 2.35s run:

| | |
|---|---:|
| script starts | 0.00s |
| R2D2 sidecar's `docker run` issued | 0.17s |
| `sidecar_wait` returns, `docker exec` issued | 0.55s |
| ranks reach `main()` | 0.71-0.74s |
| ranks attached to the imaging pool | +0.001-0.009s |
| ranks attached to the simulate pool, `warm()` returns | 0.89-1.03s |
| R2D2 pool answers | ~1.55s |
| script exits | 2.35s |

The imaging attach is free because of the pre-opened FIFO pairs; the simulate
attach is not, because that worker still opens its pair after `meqserver` and
`warm_forest()`. Neither is the binding branch: both are done by ~1.0s and the
pool does not answer until ~1.55s.

Two more numbers for scale. The sampler is linear in evaluations - 38
evaluations in 1.35s, 55 in 1.80s, i.e. ~0.32s + ~0.027s per evaluation - and
`NS_MAX_NDEAD=120` still only produces 55, because the precision criterion
stops an `NS_NLIVE=8` run first. Steady-state per evaluation is simulate
0.054s, imaging 0.069s, convert 0.014s. And `make nested-sampling-r2d2-poc` is
now only ~0.06s more than calling the run script directly, so the build checks
are no longer worth measuring separately.

#### `HOST_CPUS` comes from `nproc`, not from the daemon

`HOST_CPUS` has to be resolved before the sidecar launches, because it sets how
many FIFO pairs to create and the containers' commands glob for them. Reading
it with `docker info --format '{{.NCPU}}'` is 0.038-0.064s of CLI-plus-daemon
round trip (`nproc` is 0.001-0.004s) sitting in front of the R2D2 sidecar's
`docker run`, which the exchange rate above prices at 1:1. The two answers
cannot differ for any daemon these scripts can use: every sidecar bind-mounts
host paths, so the daemon is always this host. Only the daemon-availability
check the `docker info` doubled as still needs the daemon, and it sits below
the launches now, where it overlaps the containers coming up.

Measured with `PS4='+ $(date +%s.%N) ' bash -x` on the run script, 4 runs per
arm, the R2D2 sidecar's `docker run` is issued at 0.129-0.212s (median 0.164s)
before and 0.096-0.113s (median 0.099s) after - **0.065s earlier, with the two
ranges not overlapping**. `docker exec` is issued at the same time either way,
because that is gated by the manifest write and the container starts, not by
the preamble.

End to end, over 37 valid interleaved pairs of
`scripts/run-nested-sampling-r2d2-poc.sh` (one 38th pair dropped: arm A hit the
MeqTrees predict hang below and its `timeout 300` fired):

| | before | after |
|---|---:|---:|
| mean | 2.640s | 2.575s |
| median | 2.633s | 2.589s |

**-0.066s +-0.022s (t=2.98), 28 wins of 37.** The observed saving is the 0.065s
of earlier launch, as the 1:1 exchange rate predicts. Note how much data that
took: the first 14 pairs read +0.009s +-0.027s, which on its own is
indistinguishable from zero. Per-pair spread here is ~0.10s, so anything under
~0.1s needs 30+ pairs before the sign of the result means anything - and a
directly measured proxy for what the change actually moves (here, when the
`docker run` is issued) is worth having before the A/B is started, not after.

### The operator norm is solved with Lanczos, not a power iteration

With `get_op_norm()` at 85-96% of an R2D2 imaging request, it is the whole
per-evaluation cost worth attacking. Upstream's
`MeasOp.get_op_norm` is a plain power iteration: start from `torch.randn`,
apply the forward/adjoint NUFFT pair, stop when the norm's relative change
drops below 1e-5. The measurement operator's spectrum is tightly clustered, so
that converges badly - over this parameter space it takes **39 to 305**
operator applications, and which end of that range an evaluation lands on is
decided by the unseeded random start, not by the parameters. Measured against a
tight-tolerance reference, the answer it stops at is only accurate to ~1e-4,
and it differs from run to run for the same evaluation.

That tail is exactly what the sampler pays for. A PolyChord round costs the
*slowest* rank's evaluation: on a baseline 8-rank run the per-round maxima of
`image_container_seconds` summed to 2.47s while the medians summed to 1.48s, so
the straggler was ~40% of the sampler wall clock.

`r2d2_serve.py`'s `patch_op_norm()` replaces the method with an ARPACK Lanczos
solve (`scipy.sparse.linalg.eigsh`, `k=1`, `ncv=8`, `tol=1e-3`) over the same
`adjoint_op(forward_op(.))`, started from a deterministic vector - `ones` for
the worker's first operator and that operator's converged eigenvector for every
later one, see below. It
computes the same quantity under the same caching contract - `_op_norm`,
`compute_flag`, and therefore `get_op_norm_prime` too - and falls back to the
original power iteration if ARPACK ever fails to converge within 100 restarts.
The patch is applied in `warm_imports()`, so every request on every pool worker
gets it, and `r2d2_serve.py` runs from the repository bind mount, so it needs no
image rebuild.

Measured over 12 evaluations of this parameter space, per operator:

| | Operator applications | Time | Relative error |
|---|---:|---:|---:|
| power iteration (upstream) | 39-305 | 0.065-0.316s | ~1e-4 |
| Lanczos, `ncv=8`, `tol=1e-5` | 25-40 | 0.044-0.069s | ~1e-10 |

An `ncv` sweep (6, 8, 10, 12, 20 at `tol` 1e-5 and 1e-4) is flat to within
noise; 8 has the lowest maximum, which is the number that matters here. `tol`
is not flat, and 1e-5 was not the right end of it - see below.

Across a whole 8-rank, 38-evaluation run, `image_container_seconds`:

| | sum | median | max |
|---|---:|---:|---:|
| power iteration | 9.40s | 0.235s | 0.694s |
| Lanczos | 5.70s | 0.142s | 0.237s |

The maximum falls by 66% - more than the median's 40% - because the lottery is
what the tail was. Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh`
end to end, `NS_MPI_PROCS=8`, alternating the two `r2d2_serve.py`:

| | End to end |
|---|---:|
| power iteration | 3.92s, 4.28s, 4.63s, 4.76s, 4.64s |
| Lanczos | 3.62s, 3.64s, 3.71s, 3.52s, 3.59s |

5 pairs of 5, ~1.0s at the median (4.63s -> 3.62s, -22%). The evaluation set is
identical; `target_dynamic_range` moves by a median 2.0e-4 and at most 2.0e-3,
which is the power iteration's own error being removed, and it is now the same
number on every run.

`python3 scripts/lib/nested_sampling/r2d2_serve.py --self-check` guards it on a
synthetic 400x400 spectrum with the same 0.999 eigenvalue ratio: the 1e-5
relative-change test the power iteration uses must stop more than 1e-4 out, and
Lanczos must land at least 10x closer than it did. The assertion is against the
power iteration rather than against an absolute number because `tol` is a knob
that trades applications for accuracy, and what has to hold across a change to
it is the comparison the patch exists to win. It needs scipy, so run it in the R2D2 image
(`docker run --rm -v "$PWD:$PWD" --entrypoint python3 ri-reproducibility/r2d2:cpu
"$PWD/scripts/lib/nested_sampling/r2d2_serve.py" --self-check`); on a host
without scipy that one check prints that it was skipped and the other three
still run.

#### `tol` is 1e-3, not the 1e-5 this started at

ARPACK's `tol` is a straight trade of operator applications against accuracy,
and the accuracy that has to be met is not machine precision - the norm only
scales the operator, and upstream's power iteration delivers ~1e-4. The first
version of the patch asked for 1e-5 out of caution and got ~1e-10, i.e. it was
paying six orders of magnitude it had no use for.

Measured over 24 real operators from a PoC run (each solved again at `ncv=40`,
`tol=0` for the reference):

| `ncv`, `tol` | Applications (mean) | Applications (max) | Median relative error |
|---|---:|---:|---:|
| 8, 1e-5 | 28.7 | 33 | 9.1e-11 |
| 8, 1e-4 | 24.5 | 29 | 2.6e-08 |
| **8, 1e-3** | **21.2** | **25** | **2.9e-06** |
| 8, 1e-2 | 16.8 | 21 | 4.9e-04 |
| 6, 1e-2 | 16.8 | 22 | 3.0e-04 |
| 12, 1e-3 | 22.0 | 25 | 1.5e-06 |

1e-3 is the last step that is free: it is 26% fewer applications than 1e-5 and
still ~30x more accurate than the power iteration. 1e-2 is another 21% cheaper
but its median error, 4.9e-4, is *worse* than the power iteration this patch
replaced, so it is the wrong side of the trade. Widening `ncv` instead of
loosening `tol` buys nothing - the mean moves by under one application.

Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh` end to end,
`NS_MPI_PROCS=8`, alternating the two `r2d2_serve.py` (no `docker build` in
either arm, and no foreign container on the host for any of the 24 runs):

| | Median | 12 pairs |
|---|---:|---|
| `tol=1e-5` | 3.125s | - |
| `tol=1e-3` | 2.952s | 12 wins, 0 losses |

-0.173s, -5.6% end to end. Inside the run, medians over the 13 runs of each arm:

| | sampler wall | imaging worker-seconds | per-evaluation median | per-evaluation max |
|---|---:|---:|---:|---:|
| `tol=1e-5` | 1.125s | 3.67s | 0.095s | 0.132s |
| `tol=1e-3` | 0.950s | 2.87s | 0.075s | 0.112s |

-22% of the imaging stage for -16% of the sampler, and the simulate stage does
not move (2.03s -> 1.97s). The evaluation set is identical across both arms;
the only imaging number the checkpoint-less path prints, the estimated target
dynamic range, moves by a median 2.2e-7 and at most 4.6e-6 over the 38
evaluations - two orders of magnitude inside the power iteration's own error.

#### The start vector is the last eigenvector this worker converged on

`ones` is a bad guess for a matrix whose top eigenvector is a smooth,
strongly peaked image. A *converged* eigenvector from a neighbouring operator
is a very good one: one imaging worker sees a whole sequence of operators drawn
from the same parameter space, and although their individual top eigenvectors
are nearly orthogonal (cos ~0.01) their dominant subspaces coincide. So
`get_op_norm` keeps the first eigenvector it converges on and feeds it back as
ARPACK's `v0` for every later operator in that worker's life.

Measured over 24 real operators from a PoC run, at the shipped `ncv=8`,
`tol=1e-3` and `OP_NORM_UPSAMPFAC=1.25`:

| start vector | Applications (mean) | Applications (max) | Solve (median) | 24-operator total |
|---|---:|---:|---:|---:|
| `ones` | 21.2 | 25 | 19.6ms | 491ms |
| **first converged eigenvector** | **14.3** | **17** | **14.0ms** | **298ms** |
| rolling (previous operator's) | 14.3 | 25 | - | 291ms |

Frozen and rolling tie on the mean, so the frozen one wins on its worst case:
17 applications against 25. A rolling vector chases each operator's own
eigenvector and occasionally lands somewhere the next operator's Lanczos has to
climb out of.

The price is that the answer now depends on which operator the worker saw
first, and that dependence is far below the tolerance the answer is already
specified to. Against a solve from `ones` the eigenvalue moves 6.3e-07 median /
3.4e-06 worst, and the spread across five *different* frozen start vectors is
the same size (6.2e-07 median, 3.9e-06 worst) - against ARPACK's own 1e-3
`tol`, and upstream's ~1e-4 power iteration. Run-to-run reproducibility is
untouched: which rank runs which evaluation is PolyChord's seeded, rank-indexed
business, and worker N always serves rank N.

Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh` end to end,
`NS_MPI_PROCS=8`, alternating the two `r2d2_serve.py` with no `docker build` in
either arm. 60 pairs run, 57 valid - two dropped for the MeqTrees predict hang
and one for straddling a foreign session's image rebuild, which the harness
catches by recording `docker images -q` per run:

| | `ones` | reused | delta | wins |
|---|---:|---:|---:|---|
| end-to-end wall | 2.652s | 2.608s | **-0.044s (-1.7%)**, t=-2.6 | 39/57 |
| steady-state imaging worker-seconds | 1.759s | 1.378s | **-0.380s (-21.6%)**, t=-13.2 | 55/57 |
| steady-state simulate worker-seconds | 1.673s | 1.662s | -0.011s, t=-0.5 | 32/57 |

The simulate column is the control: it is not supposed to move, and it does
not. Every one of the 114 runs produced 38 evaluations with an identical
objective hash. -0.380 imaging worker-seconds for -0.044s of wall clock is the
low end of the 0.15-0.33 exchange rate below, which is what a host running
another agent's containers for most of the first block should give.

#### Warm and deterministic start vectors, measured and rejected

Before the Lanczos solve, three cheaper ideas were measured on the same
operators and all failed:

- **Warm start from the previous evaluation's eigenvector.** ~1.9x fewer
  iterations on average (472 to 252 over 8 evaluations) but wildly unreliable -
  2 iterations on one operator, 90 on the next. Note that this is a verdict on
  the *power iteration*: once the solve is Lanczos the same idea is reliable
  and is what ships, see above. The instability was the power iteration's, not
  the warm start's.
- **Deterministic seeds:** `ones`, the PSF, and a cosine at the argmax of
  `fft2(psf)` (the circulant approximation's top eigenvector). None beat
  `randn` consistently, and the cosine seed was off by 5e-2 on two operators. Under Lanczos the same family fails again for a different
  reason - none of them is a converged eigenvector: over 24 operators a seeded
  `randn` costs 21.3 applications, a low-passed one 20.7-25.5, Gaussians
  18.7-22.2 and one or two power iterations from `ones` 21.2-21.3, against
  `ones`' own 21.2 and a reused eigenvector's 14.3.
- The reason all of them disappoint is the same one Lanczos fixes: the spectral
  gap is so small that a power iteration started anywhere crawls. A tight
  reference run needs >2000 applications to reach 1e-9.

### Each measurement operator keeps its FINUFFT plans

With the operator norm down to ~29 Lanczos matvecs, what is left of an imaging
request is 61 NUFFT transforms - and `finufft`'s *simple* interface, which
`pytorch_finufft` calls, builds a plan, sets the sampling points on it and
destroys it again on every one of them. The trajectory does not change between
transforms, so all of that is repetition. `cProfile` of one warm request:

| | calls | tottime |
|---|---:|---:|
| `finufft` `Plan.__init__` (`makeplan`) | 61 | 0.014s |
| `finufft` `Plan.setpts` | 61 | 0.004s |
| `finufft` `Plan.execute` (the actual transform) | 61 | 0.033s |

`r2d2_serve.py`'s `patch_nufft_plans()` replaces `MeasOpPytorchFinufft._GA` and
`._AtGt` with versions that build one plan per (operator, transform type) on
first use and call `execute` on it thereafter, with the same
`eps`/`isign`/`upsampfac`/`modeord` `pytorch_finufft` would have passed. It also
drops the `torch.autograd.Function.apply` dispatch and the input checks around
each transform, which eager inference has no use for. A non-CPU device or a
batch of more than one image is not what the cached plan was built for and falls
back to upstream. Like `patch_op_norm()` it is applied in `warm_imports()`, so
every request on every pool worker gets it with no image rebuild.

| | forward+adjoint pair | warm imaging request |
|---|---:|---:|
| plan per transform (upstream) | 1.81ms | 0.069s |
| plan per operator | 1.25ms | 0.045s |

Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh` end to end,
`NS_MPI_PROCS=8`, alternating the two `r2d2_serve.py`, 10 pairs:

| | median | min | `image_container_seconds` (380 evaluations) |
|---|---:|---:|---|
| plan per transform | 3.352s | 3.162s | sum 53.5s, median 0.135s, max 0.359s |
| plan per operator | 3.063s | 2.983s | sum 37.7s, median 0.094s, max 0.324s |

9 of 10 pairs to the cached plans, median pair difference 0.324s (-8.6% end to
end, -29% of the imaging worker-seconds). All 20 runs report the same
`log(Z) = 99.92878 +/- 0.06674`.

The win is larger than the 0.018s of `makeplan`/`setpts` on its own because the
imaging pool is CPU-bound with 8 workers on a 20-CPU host, so work removed from
a worker is worth more than its solo cost.

#### The operator norm runs on a coarser upsampling grid

Once the plans are cached and the checkpoints are absent, an imaging request is
*almost entirely* `get_op_norm`. `cProfile` of one warm request, 0.038s solo:
53 `finufft` `execute` calls at 0.047s of `tottime` under the profiler, ARPACK's
`iterate` at 0.063s cumulative of the 0.085s the whole request takes. Everything
else - the YAML parse, the `.mat` read, `gen_imaging_weights`, building the
operator - is single-digit milliseconds.

FINUFFT's `upsampfac` sets how far the nonuniform points are spread before the
FFT: 2.0 puts a 128x128 image on a 256x256 grid, 1.25 on a 160x160 one, 2.56x
fewer FFT points for a wider spreading kernel. With ~3000 visibilities against
128x128 modes this transform is FFT-bound, so that trade is one-sided. Measured
on a real operator, one forward/adjoint pair:

| `dtype` | `eps` | `upsampfac` | per pair | max relative error |
|---|---:|---:|---:|---:|
| double | 1e-6 | 2.0 | 0.893ms | - |
| double | 1e-4 | 2.0 | 0.870ms | 9.2e-05 |
| double | 1e-3 | 2.0 | 0.829ms | 7.8e-04 |
| single | 1e-5 | 2.0 | 0.822ms | 7.8e-06 |
| double | 1e-6 | 1.25 | 0.598ms | 5.1e-06 |
| single | 1e-5 | 1.25 | 0.306ms | 1.2e-04 |

Loosening `eps` buys almost nothing, which is the same statement: `eps` sets the
spreading width, and the spreading is not where the time is.

`r2d2_serve.py`'s `OP_NORM_UPSAMPFAC` is 1.25 and applies to the operator-norm
matvecs only - `get_op_norm` sets `self._ri_upsampfac` around the Lanczos solve
and restores 2.0 in a `finally`, and the plan cache is keyed on
`(transform type, upsampfac)` so the imaging transforms keep their own 2.0
plans. Single precision is *not* used: it is the bigger win of the two but its
1.2e-04 is close enough to the solve's own `tol=1e-3` to be worth avoiding for
0.3ms.

Over 12 real operators from this parameter space, solving each three ways:

| `upsampfac` | applications | solve | eigenvalue vs 2.0 (median / max) |
|---|---:|---:|---|
| 2.0 | 19.7 (max 25) | 25.99ms | - |
| 1.5 | 19.7 (max 25) | 21.35ms | 1.1e-07 / 1.2e-07 |
| 1.25 | 19.7 (max 25) | 18.06ms | 5.2e-08 / 7.3e-08 |

The eigenvalue moves 100x less than a single transform does, because the answer
is an average over a 128x128 eigenvector, and the application count does not
move at all - so this is 30% off `get_op_norm` for nothing. It is also two
orders of magnitude inside the `tol=1e-3` the solve already stops at, and four
inside the ~1e-4 the upstream power iteration it replaced delivered.

Under load it is worth more than solo, because the FFT it removes is what
saturates memory bandwidth. Eight forked workers each imaging 8 real
evaluations, alternating the two settings:

| `upsampfac` | median request | mean |
|---|---:|---:|
| 2.0 | 68.6ms, 69.9ms | 68.5ms, 69.4ms |
| 1.25 | 51.2ms, 52.7ms | 51.2ms, 52.9ms |

Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh` end to end,
`NS_MPI_PROCS=8`, alternating the two `r2d2_serve.py`, 24 pairs:

| | end to end | sampler wall | `image_container_seconds` (912 evaluations) |
|---|---:|---:|---:|
| `upsampfac` 2.0 | 2.671s | 1.560s | sum 6.497s |
| `upsampfac` 1.25 | 2.578s | 1.456s | sum 5.783s |

-0.094s end to end (t = -5.7, 20 of 24 pairs), -0.104s of sampler wall, -0.713s
of imaging worker-seconds. Every run reports the same objectives.

An earlier block of 10 pairs, run while another agent session was loading the
host (per-pair spread ~0.25s against the usual ~0.10s), read the same change as
-0.007s +/- 0.072 - noise - while still showing the full -0.518s of imaging
worker-seconds. Pooled over all 34 pairs it is -0.068s +/- 0.024 (t = -2.8).
Check `uptime` before believing an end-to-end A/B on this host; the worker-second
proxy survives load that the wall clock does not.

#### What a second of imaging worker time is worth end to end

Iteration-scale changes to a *stage* are worth measuring against a calibration
rather than a guess. `time.sleep(0.020)` in front of the reply in `answer()` -
a file swap, no rebuild, and no CPU consumed - costs the run

| | end to end | sampler wall |
|---|---:|---:|
| unchanged | 2.668s | 1.565s |
| +0.020s per imaging request | 2.916s | 1.749s |

over 6 pairs: +0.248s end to end for +0.76 worker-seconds, a ratio of 0.33. The
`upsampfac` change above removed 0.713 worker-seconds for 0.104s of sampler
wall, a ratio of 0.15 - the difference being that the sleep also lands on
evaluation one, where a rank is already blocked on the pool and a delay is 1:1.
Either way the imaging stage *is* on the critical path: with 7 of the 8 ranks
evaluating, ~5-6 rounds each, a second of imaging worker time is worth 0.15-0.33
seconds of run.

The rounds are visible directly. Logging each request's arrival and reply time
in `answer()` for one run gives 38 requests over a 0.800s span, one rank taking
1 (the PolyChord administrator) and the rest 5 or 6:

```
rank 1: 6 reqs busy=0.424 first=0.000 last=0.800
rank 2: 5 reqs busy=0.331 first=0.001 last=0.712
...
```

Requests arrive in waves 0.13-0.15s apart, each wave 7 requests wide, and a
rank's own gap between one reply and its next request is 0.043-0.083s - the
simulate and convert stages of the next evaluation. So a rank's round is roughly
half imaging and half simulate, the ranks are not barriered against each other,
and the run ends when the rank that drew 6 evaluations finishes its sixth.

#### The transforms are the same bits, but FINUFFT's type 1 is not always

Compared against upstream on a real evaluation's operator - 5616 sampling points
on 128x128 - the patched forward and adjoint are bitwise equal and `get_op_norm`
returns the identical double. That is not a property of the patch alone:
FINUFFT's type 1 spreads onto the grid, and its thread partitioning makes the
summation order vary between two *identical* calls once the points are sparse
enough. Ten identical `nufft2d1` calls at 200 points on 16x16 spread 4e-16
relative; at the PoC's own size they are exactly equal, ten times out of ten.
Type 2 interpolates one output per point and is bitwise reproducible either way.
So `--self-check`, whose operator is deliberately tiny, compares the forward
exactly and the adjoint at `rtol=1e-12`.

### The imaging workers' OpenMP threads sleep between requests, they do not spin

With the operator norm down to ~30 NUFFT pairs, an imaging request that takes
0.068s solo still took 0.158s inside a run - 2.3x, on a 20-CPU host where the 8
ranks at `R2D2_OMP_THREADS=2` only ever ask for 16 threads. The gap is not
oversubscription: it is libgomp's default `OMP_WAIT_POLICY=ACTIVE`, which spins
the team's second thread for the rest of its timeslice after every parallel
region rather than sleeping. A 128x128 NUFFT *is* the parallel region here, so
each worker's second thread spends most of its life spinning on a core the
other seven workers want.

Measured with 8 forked pool workers each running 5 imaging requests off the
same warm-up - the harness the run actually uses - on the same evaluation
directories:

| Threads per worker | Wait policy | Median request | Aggregate |
|---|---|---:|---:|
| 2 | ACTIVE (default) | 0.215s | 27.7 req/s |
| 2 | PASSIVE | 0.118s | 50.4 req/s |
| 2 | `GOMP_SPINCOUNT=0` | 0.123s | 47.8 req/s |
| 1 | ACTIVE | 0.092s | 50.0 req/s |
| 1 | PASSIVE | 0.097s | 43.0 req/s |
| 3 | PASSIVE | 0.221s | 27.0 req/s |
| 4 | PASSIVE | 0.429s | 13.2 req/s |

`GOMP_SPINCOUNT=0` is the same fix by the libgomp-specific lever; `PASSIVE` is
the portable spelling and is what the run script passes. The R2D2 sidecar
therefore gets `-e OMP_WAIT_POLICY=PASSIVE` alongside its thread caps, and
`r2d2_docker_thread_env_flags()` passes it on the fallback path where a rank
starts its own worker.

Note what the table does *not* say. Spinning at 1 thread is as fast as passive
at 2 because a 1-thread team has nothing to spin; that is iteration 9's "1
thread is 27% faster per evaluation" result, and it was measuring this, not the
thread count. Once the spinning is gone, 2 threads is the better setting again,
so `R2D2_OMP_THREADS = host CPUs / NS_MPI_PROCS` stays as it is - which matters
because the 25 UNet forward passes a checkpointed run adds are exactly the
large parallel regions that want the second thread. Above 2 the product really
does exceed the host (8x3 = 24 threads on 20 CPUs) and passive does not save
it, so the existing "do not raise it" rule is unchanged. Solo, passive is not a
regression either: 0.072s against 0.078s for one worker at 2 threads.

Interleaved A/B of `scripts/run-nested-sampling-r2d2-poc.sh`, `NS_MPI_PROCS=8`,
38 evaluations, alternating the two run scripts, ten pairs over two batches:

| | Sampler wall clock (median of 10) | Imaging worker-seconds (median of 5) | End to end (median of 5) |
|---|---:|---:|---:|
| ACTIVE | 1.634s | 6.58s | 3.70s |
| PASSIVE | 1.318s | 4.93s | 3.52s |

10 of 10 pairs favour PASSIVE on sampler wall clock (-19%) and 5 of 5 on
imaging worker-seconds (-25%); end to end is 4 of 5 and only -5%, because at
this problem size ~2.0s of the run is the fixed setup the section below
describes and the sampler is now the smaller half. The evaluation set and
objectives are identical across all ten runs.

Iteration 9 tested `OMP_WAIT_POLICY=PASSIVE` end to end and called it a wash (2
wins, 3 losses over 5 pairs). It was measuring the same real effect through the
power iteration's 39-305-application lottery, whose run-to-run spread was
larger than the ~0.3s the policy is worth. A per-worker microbenchmark of the
imaging request would have shown it either way; end-to-end wall clock at this
size will not resolve a change this small until the noise source above it is
gone.

### The imaging worker warms what `imager.py` imports, and no more

With the no-op builds gone and the sampler down to ~1.3s, the binding wait in
`make nested-sampling-r2d2-poc` is the R2D2 worker pool's readiness. It is
directly observable from the host without instrumenting anything: a pool worker
opens its `<rank>.in` FIFO for reading before it answers, so polling
`os.open(fifo, O_WRONLY | O_NONBLOCK)` until it stops raising `ENXIO` dates the
moment the rank stops waiting. On a ~3.5s run that lands at ~2.0s, and
`run_polychord` starts within ~0.05s of it - every rank is sitting in `warm()`
until then.

`warm_imports()` used to pay for that window with a hand-copied `import
optimiser, utils`, which was both more and less than `imager.py` needs.

- **Less**: `create_meas_op` imports its NUFFT backend *inside the function*, so
  `finufft`, `pytorch_finufft` and
  `ri_measurement_operator...meas_op_nufft_pytorch_finufft` were not warmed at
  all and every worker imported them on its own request one - 0.165s against a
  0.072s steady state. Pre-importing the backend in the warm-up removes that.
- **More**: `utils/__init__.py` re-exports one name from each of its nine
  submodules, so importing any part of `utils` imports all of them. The imaging
  path uses seven. The other two are the expensive ones: `utils.util_training`
  pulls `lightning` and `utils.noise` pulls `scipy.optimize`.

| after `import torch`, in the R2D2 image | |
|---|---:|
| `import utils` (the package `__init__`) | 0.335s |
| the seven submodules the imaging path uses, `__init__` bypassed | 0.209s |
| the two it does not (`noise`, `util_training`) | 0.127s |

`install_lazy_utils()` puts a module with a PEP 562 `__getattr__` in
`sys.modules["utils"]`: a name is resolved by walking the submodules in order
until one defines it, and `noise` and `util_training` are last in that order, so
nothing on the imaging path reaches them. Same names, same values - only the
unused imports go unpaid. The warm-up then runs `imager.py` itself through
`runpy.run_path(..., run_name="__warmup__")` rather than naming its imports by
hand: upstream puts its whole body behind `if __name__ == "__main__"`, so what
executes under any other name is exactly its import block, and there is no copy
of that block here to drift.

Interleaved A/B, 8 pairs, alternating the two `r2d2_serve.py` versions in place
(it runs off the bind mount, so no rebuild):

| | worker pool ready | end to end |
|---|---:|---:|
| before | 1.95 1.99 1.96 1.99 2.14 2.05 2.08 2.09 | 3.35 3.38 3.34 3.51 3.64 3.58 3.61 3.65 |
| after | 1.88 1.81 1.87 1.84 1.93 1.86 1.95 1.89 | 3.26 3.19 3.47 3.42 3.47 3.49 3.61 3.43 |

Readiness is 8 of 8, median -0.145s. End to end is 6 wins, 1 tie, 1 loss, median
-0.10s (~3%) - the wait is what shrank, and only some of it shows up outside.
`log(Z)` is 99.92878 +/- 0.06674 in every run of both arms, and both arms
produce the same 38 evaluations.

Two things worth recording about how this was measured. Pre-importing the NUFFT
backend on its own is worth nothing end to end on a warm host (5 interleaved
pairs, flat to +/-0.01s): the first profiled run that motivated it showed eight
evaluations at 0.73-0.87s against a 0.13s steady state, one per rank, but that
was the page cache being cold on `libtorch` and the finufft shared objects, not
eight workers importing at once - the same run repeated warm has no such spike.
And the R2D2 image reports `torch` at 0.85-0.92s by `-X importtime`, of which
`torch._C` (0.19s), `_meta_registrations` (0.17s), `torch.functional`/`torch.nn`
(0.16s), `torch.export` (0.08s) and `torch.nested` (0.06s) are stock and not
disableable - that part of the readiness window is a hard floor.

#### The ranks attach to the pool before the warm-up, not after

The readiness wait above is the pool's, not the ranks'. A rank attaches to its
worker by write-opening `<rank>.in` - `ENXIO` until someone is reading it - and
then read-opening `<rank>.out`, which blocks until someone is writing it. Both
ends were opened by the forked children, and the children do not exist until
`warm_imports()` has finished, so the rank could not attach until the whole
~1.2s of `import torch` and the R2D2 modules was paid.

`serve_pool()` now opens both ends of every pair itself, before the warm-up, and
holds them until that pair's child exits. `.out` is `O_RDWR` because a plain
write-open would block waiting for the rank, which is the wait being removed; on
a FIFO `O_RDWR` never blocks. Measured from the host with a stand-in for the
ranks - eight attaches against a pool started exactly as the run script starts
it, `docker run` issued at t=0:

| | rank 7 attached at |
|---|---:|
| children open the FIFOs | 1.206 1.226 1.216 1.195 |
| the pool opens them before warming | 0.298 0.296 0.284 0.286 |

-0.92s, 4 of 4. What the rank does with that window is the rest of PolyChord's
startup and evaluation one's simulate and convert, and then it blocks on the
imaging reply anyway - so only the ~0.05s it would otherwise have spent on those
after the wait can come off the run. Interleaved A/B, 12 pairs alternating the
two `r2d2_serve.py` versions in place (one pair dropped: its
second arm hit the meqserver hang described below and was killed by the
harness' 60s timeout):

| | End to end |
|---|---:|
| children open the FIFOs | 2.647 2.744 2.641 2.826 2.785 2.945 2.742 2.759 2.783 2.808 2.848 |
| the pool opens them first | 2.588 2.643 2.764 2.759 2.749 2.784 2.792 2.711 2.781 2.807 2.725 |

9 wins, 2 losses, median -0.048s (-1.7%), mean -0.039s. `log(Z)` is 99.92878 +/-
0.06674 in all 23 completed runs and both arms produce the same 38 evaluations.
This is the same shape of result as the simulate worker's version of this change
(above), which was rejected at 0.35s off the join and 0.00s end to end; the
difference is that a rank has simulate and convert to run before it needs an
imaging reply, where on the simulate side it had nothing.

It does move where the wait is *counted*. The rank starts timing an imaging
request when it writes it, so a rank that reaches evaluation one while the pool
is still importing measures the rest of the import as imaging: in a profiled
pair, `eval_id == 1` `image_container_seconds` goes from 0.102s to 0.587s and
summed imaging worker-seconds from 2.73s to 6.14s, while the steady-state median
is 0.068s against 0.073s and the run is faster. Read those two numbers as the
readiness wait, not as an imaging regression - the same trap `OMPI_MCA_pml=ob1`
set for `simulate_seconds`.

Three FIFO details this depends on, each of which deadlocks or corrupts the run
if it is got wrong:

- The children close every inherited end, their own included, before `answer()`
  re-opens the pair. Two processes holding a request pipe's read end means a
  request can be delivered to the one that will not answer it.
- The parent keeps holding both ends until the pair's child exits, rather than
  dropping them after the fork. Dropping the `.out` write end leaves a window
  with no writer at all, and a rank sitting in `readline()` reads that as EOF -
  an empty reply, which the run reports as a dead worker. Measured: it happens
  on the first request every time.
- The parent drops a pair's ends *as* its child exits, not at the end of the
  run. While it holds them, a rank whose worker died would write into a pipe
  nobody reads and then wait forever for a reply; closing them gives it the same
  broken pipe and empty reply it gets when there is no pool at all.

`self_check_serve_pool()` guards the property: its stand-in for `import torch`
sleeps 0.5s, and rank 0 must have opened *both* of its FIFOs before that sleep
has finished.

#### An evaluation's MeqTrees predict can hang forever

Seen twice in ~25 runs on a host shared with other work, in both arms of the A/B
above and therefore not caused by it: a rank stops in `pipe_read` waiting for its
simulate worker's reply, and that worker is in `futex_wait_queue_me` inside
`meqserver`'s `await_()` - a `threading.Event.wait()` with no timeout - for a
predict that never completes. All eight imaging workers are idle. Nothing in the
path has a timeout (`simulate_worker_request()` blocks in `readline()`), so the
run hangs until it is killed. Worth knowing when timing anything here: an A/B
harness needs `timeout` around each run or one hang eats the whole measurement.

Sidecars use `--network none`, saving ~0.2s per container with identical
results. `docker/meqtrees/Dockerfile` also replaces Timba's 10s shutdown sleep
with 0.1s polling while preserving its ~200s SIGKILL ceiling; MS outputs stay
identical.
