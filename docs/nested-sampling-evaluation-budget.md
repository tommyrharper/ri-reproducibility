# Where a WSClean evaluation's 280 ms goes

A profile of one evaluation at the concurrency a real search runs at, taken
apart far enough to say what is left. The short answer: **69% of it is
WSClean's clean loop, 11% is starting a process, 20% is the path to the first
inversion, and 6% is everything this repo writes** - so there is nothing left
in the harness, and the remaining lines are WSClean's or the host's.

This is the per-evaluation companion to
[the throughput doc](nested-sampling-throughput.md) (which is about how the
ranks are kept busy) and to
[the power-limit doc](nested-sampling-power-limit.md) (which is about how fast
the host is allowed to run while they are).

## The measurement

Every evaluation already leaves the numbers behind. WSClean prints its own
phase totals as its last line of `wsclean.stdout.log`:

```
Inversion: 00:00:00.030745, prediction: 00:00:00.019075, deconvolution: 00:00:00.008083
```

and `metrics.json` records the wall clock of the binary that printed them. So
a whole run decomposes with no instrumentation and no extra run:

```sh
python3 - <<'EOF'
import json, re, statistics
from pathlib import Path
root = Path("results/nested-sampling/<run>/evaluations")
pat = re.compile(r"Inversion: (\S+), prediction: (\S+), deconvolution: (\S+)")
def secs(t):
    h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + float(s)
rows = []
for d in root.iterdir():
    m = pat.search((d / "wsclean.stdout.log").read_text())
    t = json.loads((d / "metrics.json").read_text())["timing"]
    rows.append((t["image_binary_seconds"], *(secs(x) for x in m.groups())))
n = len(rows)
for i, name in enumerate(["binary", "inversion", "prediction", "deconvolution"]):
    print(f"{name:14s} {1000 * sum(r[i] for r in rows) / n:6.1f} ms")
EOF
```

Two searches, `--nlive 25 --num-repeats 10 --max-ndead -1 --mpi-procs 20`
(19 workers), seeds 4242 and 7, on the tree at iteration 14. They agree to
better than 0.5% on every line, which is what makes the split below worth
quoting to three figures:

| line | seed 4242 | seed 7 | share of the evaluation |
|---|---:|---:|---:|
| simulate | 16.55 ms | 16.98 ms | 5.9% |
| wsclean container | 266.16 ms | 266.44 ms | 94% |
| - of which the `wsclean` binary | 257.83 ms | 258.24 ms | 91% |
| - - inversion | 106.3 ms | 106.8 ms | 38% |
| - - prediction | 64.2 ms | 64.3 ms | 23% |
| - - deconvolution | 19.7 ms | 19.4 ms | 7% |
| - - everything else | 67.6 ms | 67.8 ms | 24% |
| metrics | 1.59 ms | 1.61 ms | 0.6% |
| evaluations (n) | 6557 | 6458 | |
| evaluations/s | 62.8 | 62.1 | |

WSClean's own three timers account for **73.7-73.8%** of the binary. A default
evaluation does 8.38 gridding passes (one PSF plus one per major cycle) and
6.38 degridding passes, so a gridding pass over ~4000 visibilities into a
128x128 image costs ~12.7 ms and a degridding pass ~10.1 ms - fixed per-pass
overhead, not arithmetic (see the throughput doc's decomposition of the
binary, and iteration 14's gridder A/B).

## Splitting the 68 ms that WSClean does not time

Replay 190 real Measurement Sets - kept with `./ri search
--keep-measurement-sets` - through 19 concurrent `wsclean` processes, four
passes over the corpus per arm, each arm's argv taken verbatim from the
`commands.wsclean` its evaluation recorded. Per-call latency below is the
rig's wall clock times 19 workers over the call count.

| arm | per-call latency | what it includes |
|---|---:|---|
| `wsclean --version` | 25-30 ms | process start and nothing else |
| the real argv with `-niter 0` | 87 ms | the above, plus MS open, reorder, model init, weight precalc, one inversion, the dirty FITS |
| the real argv | 242 / 277 / 282 ms | the above, plus the PSF, 6.4 major cycles and the four other FITS |

(The 242 ms is the first arm of the first pair: the burst window the power-limit
doc warns about, arriving exactly on cue. Read the second and third pairs.)

So of ~280 ms at this concurrency:

- **~30 ms (11%) is process start**, before WSClean opens anything;
- **~57 ms (20%) is the path to the first inversion** - casacore opening the
  MS, the reorder, initialising the model, precalculating the weights, one
  gridding pass and the dirty image;
- **~193 ms (69%) is the clean loop**, which is `-niter 100 -mgain 0.8` and
  therefore the science being searched.

## The 30 ms of process start is casacore's, and it does not come off

It is not the dynamic loader: `LD_DEBUG=statistics wsclean` reports 2.66M
cycles in the loader, under a millisecond. It is not file I/O:
`strace -T wsclean --version` opens exactly three non-library files
(`/sys/devices/system/cpu/possible`, `/proc/cpuinfo`, `/etc/localtime`), none
of them slow. It is C++ static initialisation spread across the 73 shared
objects the binary pulls in, and the gaps in the trace are spread evenly over
their `mprotect`/RELRO points.

Measured by `dlopen` cost above a bare interpreter, in-image:

| object | added |
|---|---:|
| `libcasa_casa.so.9` | 7.9 ms |
| `libhdf5_serial.so.103` | 2.2 ms |
| `libcurl.so.4` and the ~20-object TLS/HTTP stack behind it | 2.0 ms |

`chgcentre`, which links casacore but none of WSClean's own libraries, starts
in 9.4 ms against `wsclean`'s 14-17 ms serial. Two build-side attacks were
considered against these numbers and both were rejected without a rebuild:

- **Dropping MPI and Python.** `libmpi`, `libopen-pal`, `libopen-rte`,
  `libhwloc`, `libevent*` and `libpython3.11` are all linked in and none of
  them does anything at load time - Open MPI's plugin scan happens in
  `MPI_Init`, which `wsclean` never calls, and pybind11 does not start an
  interpreter until a Python deconvolution asks for one. Worth ~0.
- **Dropping HDF5.** Debian's `libhdf5` is what drags the whole curl/TLS stack
  in, so `-DUSE_HDF5=OFF` on casacore would remove ~20 of the 73 objects -
  and about 4 ms serial, ~1.5% end to end, for a 40-minute casacore rebuild
  and the loss of HDF5 image support. Not worth it.

That leaves only one route: not starting a process per evaluation. WSClean has
no server mode, so this would mean a pre-initialised zygote that forks after
static init - a patch to somebody else's C++ for 10% of an evaluation. Left
undone deliberately.

## `/proc/cpuinfo` costs 20 ms to open, and the search hides it

On this kernel (5.15), `openat("/proc/cpuinfo")` blocks for **17-21 ms**
whenever the per-CPU APERF/MPERF sample is stale: `arch_freq_get_on_cpu()`
sleeps out its refresh delay so that the `cpu MHz` field is not a lie. Warm,
the same open is 0.04 ms.

```
cold (200 ms apart):  open  16.561  19.469   0.049   0.051  ms
warm (back to back):  open   0.008   0.005   0.006  17.872  ms
```

`wsclean` opens it once per process. In a real search that is ~62 opens a
second across the host, so the sample is never stale - sampled in situ during
a live search, 40 opens gave p50 0.040 ms and max 0.056 ms, none over 5 ms.
**It costs the search nothing.** But it does mean that:

- a single cold `time wsclean ...` reads ~20 ms too slow, which is 7% of an
  evaluation and enough to invent or hide any result on this page. Warm the
  cache (`cat /proc/cpuinfo > /dev/null`) at the top of any rig - the one in
  this doc does;
- the clock-sampling recipe in
  [the power-limit doc](nested-sampling-power-limit.md) is reading a file
  whose first read after a quiet second takes 20 ms and whose subsequent reads
  are cached for about a second. That is fine for that doc's purpose (it
  samples during a loaded run, where the value is fresh) but it is not a
  free or an instantaneous read.

## What is not measured here

`results/nested-sampling/` holds 45 runs and every one of them is a WSClean
run. The R2D2 side cannot be profiled on this host at all: `checkpoints/`
contains only its README, and `./ri fetch-checkpoints` cannot complete without
a browser (the upstream host serves the ~5 GB realisation archives behind a
Cloudflare challenge - see `checkpoints/README.md`). Every per-evaluation
number this repo has ever published is therefore a WSClean number, and the
R2D2 evaluation - simulate, `ms_to_r2d2_mat`, then 25 network iterations in a
long-lived sidecar - remains entirely unprofiled.
