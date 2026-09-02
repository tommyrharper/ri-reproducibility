# Nested sampling

PolyChord searches for failure modes by maximizing a configurable objective
(default `total_rms_jy`), not by fitting a Bayesian posterior.

Every run uses one unpolarized 1 Jy phase-centre point source; complex Gaussian
visibility noise controls dynamic range.

## Images

`./ri build` also builds MeqTrees and PolyChord images. MeqTrees uses KERN 10 on
Ubuntu 24.04, makems' bundled VLA.A table, and a Meow point-source RIME; thermal
noise is added to its clean prediction.

Phase-centre predict is constant, so it bypasses meqserver. The simulate
self-check (`./ri self-check simulate`) verifies exact agreement;
`source_offset_fraction` restores the MeqTrees path. See
[nested-sampling-throughput.md](nested-sampling-throughput.md).

## Run it

Both algorithms share `NS_*` and `OUTPUT_DIR` overrides (see "Environment
overrides"). Each target builds required images, starts one long-lived sidecar
per image, and runs PolyChord through `docker exec` (see
[nested-sampling-profiling.md](nested-sampling-profiling.md)).

Once the containers are up, a status line tracks the search against its
`--max-ndead` budget: elapsed time, dead points done (from PolyChord's own
`chains/*_dead-birth.txt`, one line per dead point - not the raw evaluation
count, which is always higher since PolyChord's slice sampler makes several
evaluations per accepted dead point), a percent, and an ETA extrapolated from
the rate so far (`scripts/lib/progress-bar.sh`). With `--max-ndead <= 0` (run
until the evidence tolerance is met, no fixed cap) there is no dead-point cap
to measure a percent against, so `_ns_evidence_total` estimates one from the
run's own evidence, and every figure derived from it is marked `~`:
`~ 53%  ~241/~452 dead points ... eta ~4h35m`.

PolyChord rewrites `chains/` only every `nlive` dead points, so the dead-point
count is frozen between writes by construction - two hours at a time on a
16-rank R2D2 search, which is a bar that sits still for two hours and then
jumps fifty. `_ns_dead_now` carries it across that interval: the evaluation
directories appear every few seconds, the slice sampler spends a near-constant
number of them per dead point, so the evaluations that landed after the
checkpoint convert back into dead points at the run's own measured ratio. The
carried count takes a `~` of its own even when the denominator is an exact
`--max-ndead`, and `./ri health`'s **forecast** line carries the count the same
way from the same ratio.

That estimate is the same model `./ri health`'s **forecast** line uses, from
the same two files (`chains/*.stats` for the accumulated `log(Z)`,
`chains/*_phys_live.txt` for the live points' current log-likelihoods) and the
same measured stopping fraction - see the **forecast** field below for the
calibration and its evidence. The bar's copy of that constant is
`_NS_TERMINATION_EVIDENCE_RATIO`, and `progress-bar.sh --self-check` fails if
it and the health script's `TERMINATION_EVIDENCE_RATIO` drift apart: a status
line and a report disagreeing about the same run is what this replaced. The
earlier bar divided PolyChord's documented `precision_criterion` by the
current evidence ratio, which is both uncalibrated and exponential in the
quantity being reported - it read 3% on a live 16-rank R2D2 search that this
model puts at 38%.

Like the health forecast, it approximates the remaining prior volume as a
single global `-ndead/nlive` rather than PolyChord's own per-cluster tracking,
so a run whose live points split across several clusters (`chains/*.stats`'
`ncluster` line) will diverge further from PolyChord's exact figure than a
single-cluster run does. Before the first e-fold (`ndead < nlive`, where the
live set is still the prior and the estimate would report its own constant),
and before `.stats` or `phys_live.txt` exist at all, the line falls back to a
bouncing bar and the raw dead-point rate. On a real terminal it is pinned to the bottom via
a scroll region, so PolyChord's own feedback scrolling past above it doesn't
bury it; only drawn
on a TTY, so piped or logged runs are unaffected.

### WSClean

```bash
./ri search wsclean
```

Output: `results/nested-sampling/wsclean-vlaa-<UTC timestamp>/`.

The timestamp is claimed with a bare `mkdir`, not assumed: two searches started
in the same second - two sessions sharing this host, or one script launching a
pair - would otherwise resolve to the same directory and write each other's
evaluations, FIFOs and `summary.json`, with the first to finish deleting the
FIFO directory the other was reading. The loser of the race waits for the next
second rather than decorating its name, so a run directory always ends in a
stamp. An `--output-dir` you name yourself is yours and may already exist -
unless a job is still in it, which is refused for the same reason
`./ri resume` refuses a live run: measured, a second search into a live run
directory deleted its FIFOs, recreated them with its own rank count, and
wrote its own `chains/*.resume` over the live checkpoint while the first run
was still imaging. Liveness is the host's process list (`ns_run_is_live` in
`scripts/lib/progress-bar.sh`), which is the only thing `mkdir -p` cannot
see.

It also has to be inside the repository. Every container the run starts is
given one bind mount, `-v $REPO_ROOT:$REPO_ROOT`, so a run directory outside
it exists twice over: on the host, holding `run.env` and the FIFOs, and
emptily inside each container, where PolyChord's chains and the evaluation
directories are actually written. Measured on a real `--output-dir /tmp/...`
search, that cost two minutes of container startup and then died on
evaluation 1 with `FileNotFoundError: .../eval-0001-*/simulate.stdout.log`;
`ns_refuse_unmounted_run` in `scripts/lib/run-config.sh` now refuses it in
0.1s instead. The path is resolved to an absolute one first, so a relative
`--output-dir` still works and the run names itself the same way everywhere.

Useful overrides:

```bash
./ri search wsclean --nlive 8 --num-repeats 2 --max-ndead 12
./ri search wsclean --mpi-procs 4
./ri search wsclean --metric badness
./ri search wsclean --metric snr
./ri search r2d2 --metric off_source_rms_jy
./ri search r2d2 --metric sigma_res
./ri search wsclean --output-dir results/nested-sampling/manual
./ri search wsclean --then r2d2 --nlive 125 --num-repeats 25 --max-ndead=-1
```

### Running both imagers

`--then <imager>` runs a second search with the same settings once the first
finishes, and only if it finishes: a failed first search means something is
broken - a build, an image, the parameter space - and the second would meet the
same fault. `./ri tui` offers the same pairs from its `n` form.

One after the other rather than both at once, deliberately. This host is
work-conserving: measured with a WSClean and an R2D2 search overlapping for
280 seconds, WSClean kept 0.66 of its solo throughput and R2D2 0.36, summing to
1.015 - so running the pair concurrently finishes at the same wall clock as
running them in turn, and leaves every per-evaluation timing in both runs
measured against the other one. See
[the throughput doc](nested-sampling-throughput.md).

The chained search claims its own output directory, so `--output-dir` names the
first one only.

`--plot` and `--report` finish the job off, so a pair left running overnight has
its comparison waiting in the morning:

```bash
./ri search wsclean --then r2d2 --nlive 125 --num-repeats 25 --plot --report
```

`--plot` is `./ri plot likelihood --last` and `--report` is `./ri report`, run in
that order once every search above them has finished - the report's comparisons
page collects the figures on disk, so plotting has to come first. After a
`--then` pair the plotted pair is the two runs just finished; used on a single
search it is whichever comparable pair is newest, and the plot names the two
runs it used.

PolyChord likelihood evaluations run in parallel across MPI ranks inside the
PolyChord container. `NS_MPI_PROCS` sets the rank count (default
`min(NS_NLIVE, host CPUs)`). Set `NS_MPI_PROCS=1` to disable parallel
evaluations for debugging.

The target builds any missing WSClean, MeqTrees, and PolyChord images first.
Each likelihood evaluation runs one MeqTrees simulate and one WSClean imaging
step in this rank's already-running sidecar containers.

### R2D2

```bash
./ri search r2d2
```

Output: `results/nested-sampling/r2d2-vlaa-<UTC timestamp>/`.

The target builds R2D2, MeqTrees, and PolyChord images first. Each likelihood
evaluation runs one MeqTrees simulate, one MeqTrees-hosted MS-to-`.mat`
conversion and one R2D2 imaging job, each a request to a long-lived process
inside one of the run's two sidecar containers.

R2D2 requires pretrained checkpoints at `checkpoints/R2D2_A1/R2D2_UNet_N*.ckpt`
(see `./ri fetch-checkpoints` and `./ri smoke r2d2`).

Before a full end-to-end run, validate the MS-to-`.mat` bridge:

```bash
./ri smoke ms-to-mat                 # or: scripts/check-ms-to-r2d2-mat.sh
```

`run-nested-sampling-r2d2.sh` runs `NS_MPI_PROCS` PolyChord ranks
concurrently, each with its own `r2d2_serve.py` imaging worker inside the
shared R2D2 sidecar and its own simulate worker inside the MeqTrees one. Both
pools are started by their container's own command, over one FIFO pair per rank,
before the PolyChord container exists - the imaging pool as one `--fifo-dir`
process that imports torch once, forks a worker per pair, and opens every pair
before it starts importing so the ranks do not wait for it (see "R2D2 imaging runs in a long-lived
worker", "The workers are started by the container, not by the ranks" and "The
ranks attach to the pool before the warm-up" in
[nested-sampling-profiling.md](nested-sampling-profiling.md)). That process also
patches R2D2's `MeasOp.get_op_norm` to solve the operator norm with Lanczos
rather than upstream's power iteration - the same quantity, ~3.5x fewer NUFFT
pairs and ~3e-6 relative accuracy instead of ~1e-4, and no longer a different
answer on every run (see "The operator norm is solved with Lanczos" there) - and
it gives each measurement operator one FINUFFT plan per transform type instead
of the one-plan-per-transform `pytorch_finufft` builds, worth ~30% of a warm
imaging request (see "Each measurement operator keeps its FINUFFT plans"). Its
warm-up runs `imager.py`'s own import block - the file under a run name that is
not `__main__` - plus the NUFFT backend `create_meas_op` imports lazily, and
makes `utils` resolve its submodules on demand so the imaging path never pays
for `lightning` or `scipy.optimize` (see "The imaging worker warms what
`imager.py` imports, and no more"). The workers get
OpenMP/BLAS thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`) set from the host's available CPU count, overridable via
`R2D2_OMP_THREADS`. The previous image default of
`OMP_NUM_THREADS=4` capped finufft/OpenMP work when the Docker VM exposed more
CPUs than four. To avoid CPU oversubscription, the script defaults
`R2D2_OMP_THREADS` to the rounded-up per-rank CPU share (minimum `1`) when not set
explicitly, so each rank's imaging worker gets a fair share of the host's cores
instead of all of them. Set `R2D2_OMP_THREADS` explicitly to override this
per-rank default. The same count is written into every per-evaluation
`r2d2_config.yaml` as `ncpus`, because those env vars alone do not reach
torch - see "R2D2 sizes its own torch thread pool" in
[nested-sampling-profiling.md](nested-sampling-profiling.md).

### Environment overrides

Both run scripts read the same variables and forward the sampler ones to
`polychord_wsclean.py` / `polychord_r2d2.py` as command-line flags.
Sampler defaults live in `defaults.toml` at the repository root, loaded by
`scripts/lib/defaults.sh`. Setting a variable yourself still wins over
`defaults.toml`; a flag wins over both.

#### Tweak these

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--nlive` | `NS_NLIVE` | Number of PolyChord live points | `8` |
| `--num-repeats` | `NS_NUM_REPEATS` | How much PolyChord explores inside the likelihood constraint before generating a replacement live point | `2` |
| `--max-ndead` | `NS_MAX_NDEAD` | Dead-point budget that terminates the run | `12` |
| `--seed` | `NS_SEED` | PolyChord random seed. Randomised per run, so two searches explore different points; set it to repeat one exactly. The value used is in the run's `run.env` and `summary.json`, and `./ri resume` reuses it | random |
| `--metric` | `NS_METRIC` | Objective: `badness`, a bare metric name, or an expression over metric names - see "Choosing the objective" below | `total_rms_jy` |
| `--retries` | `NS_RETRIES` | Times a run that dies after scoring evaluations restarts itself from its checkpoint, counting only failures that come straight back (`NS_RETRY_RESET_SECONDS`, 1800, hands the budget back); `0` disables - see [robustness.md](robustness.md) | `2` |
| `--stall-timeout` | `NS_STALL_TIMEOUT` | Seconds with no evaluation finishing before a run is killed as hung, so `--retries` can restart it; `0` disables - see "A run that hangs instead of dying" | `7200` |

#### Leave these alone

Flags exist, but the defaults are derived. Leave them unset unless you want
serial debugging (`--mpi-procs 1`), a different rank/thread split, or a
pinned run directory.

| Flag | Variable | Meaning | Default |
|---|---|---|---|
| `--mpi-procs` | `NS_MPI_PROCS` | PolyChord rank count (`mpirun -np`); `1` is serial | `min(NS_NLIVE, host CPUs)`, host CPUs from `nproc` (`sysctl -n hw.ncpu` on macOS, which has no `nproc`), then clamped to what free memory holds - see "Rank count is the memory budget" |
| `--omp-threads` | `R2D2_OMP_THREADS` | Per-rank R2D2 OpenMP/BLAS/torch threads | rounded-up `host CPUs / NS_MPI_PROCS`, min 1, from the rank count before the memory clamp |
| `--output-dir` | `OUTPUT_DIR` | Run directory; must be inside the repository, and not one a job is still in | `results/nested-sampling/<algo>-vlaa-<UTC>` |

### Rank count is the memory budget

Rank count, not `NS_NLIVE`, is what costs memory: each rank keeps one warm
worker holding its own copy of the imaging stack. Measured on a 20-CPU, 62GB
host with `NS_NLIVE` held at 12 and only the rank count varied:

| Ranks | R2D2 peak memory |
|---:|---:|
| 4 | 13.5GB |
| 8 | 27.0GB |
| 12 | 40.6GB |

That is 3.4GB per R2D2 rank, linear, against ~0.2GB per WSClean rank. Two
consequences:

- **`NS_NLIVE` is free.** It sets search quality, not memory. `--nlive 40
  --mpi-procs 12` evaluates 40 live points 12 at a time and costs 12 ranks of
  memory. Raise `--nlive` for a better search without paying for it in RAM.
- **`--mpi-procs` is what has to fit.** On a 62GB host R2D2 tops out around 16
  ranks; the CPU would allow 20.

Both run scripts clamp an auto-derived rank count to what free memory can
hold, and reserve it so that runs started at the same moment size themselves
around each other rather than both assuming an empty host
(`scripts/lib/rank-budget.sh`). A clamp prints a `NOTE:`; a host with no room
for even one rank fails before starting any container. An explicit
`--mpi-procs` is honoured, with a `WARNING:` if it will not fit.

### When a run breaks

A failed evaluation scores `FAILURE_OBJECTIVE` (`100.0`), which PolyChord
maximizes - deliberately, because failure modes are what these runs look for.
That is only correct when the *algorithm* failed, so a dead worker, a wedged
meqserver and an OOM kill are kept apart from it and never scored.

A run that dies restarts itself from its own checkpoint, up to `--retries`
times, and one that hangs is turned into a crash the same machinery recovers
from by a stall watchdog. A run that stops for good is listed by `./ri runs`
and continued by `./ri resume <run>`.

**[robustness.md](robustness.md)** covers all of it: what is scored and what is
retried, the three bounds on a wedged meqserver, the restart loop and its
budget, torn checkpoints and half-written summaries, and
`./ri self-check self-heal`.

### Is the run healthy?

`./ri runs` answers "did it finish?"; `./ri health` reports live progress,
forecast, resource use, and warnings. It is read-only; exit status is 1 when
something needs attention.

**[run-health.md](run-health.md)** has the full report, what every line reads
and why, and the thresholds that warn.

### Running both algorithms

WSClean and R2D2 can run concurrently. Size combined R2D2 ranks to fit free
memory, and give R2D2 available cores first. The scripts export the sidecar
and worker paths internally; details live in
[nested-sampling-profiling.md](nested-sampling-profiling.md).

## Parameter space

VLA configuration is an outer-loop dimension. The runs here only use `VLA.A`.

PolyChord dimensions for both algorithms:

| Dimension | Range | Meaning |
|---|---:|---|
| `dynamic_range` | `1e2` to `1e3` | One-Jy source divided by thermal-noise sigma |
| `observation_minutes` | `4` to `10` | Total requested observing time |
| `channel_count` | `2` to `6` | Number of frequency channels |
| `start_frequency_hz` | a receiver band (see below) | First channel frequency |
| `channel_width_hz` | `0.5e6` to `2.0e6` | Uniform spacing between channels |
| `source_offset_fraction` | `0.0` to `0.35` | Source offset from the phase centre, as a fraction of the image half-width |
| `source_l_pixels` | `+/-(NS_IMAGE_DIM / 2) * 0.5` | Source position along l, in pixels from the central pixel |
| `source_m_pixels` | `+/-(NS_IMAGE_DIM / 2) * 0.5` | Source position along m, in pixels from the central pixel |
| `declination_deg` | `-30` to `80`, whole degrees | Declination of the phase centre; sets how foreshortened the array and how elliptical the PSF is |
| `integration_seconds` | one of `10, 30, 60, 120, 300` | Correlator dump time; sets sampling density against track length, and is the only dimension time smearing depends on |

Channel frequencies are represented as a contiguous uniform
`start_frequency_hz` plus `channel_width_hz` grid. Arbitrary per-channel
frequency sets are a follow-up ceiling.

The current box for every dimension is in `defaults.toml`, which is the one
authoritative copy - this table names them, not their exact ranges.

### Toggling dimensions on and off

Every `[[parameter_space]]` entry in `defaults.toml` takes `enabled` (default
true). Setting `enabled = false` pins that dimension out of the search
instead of deleting it: `cube_to_params()` fixes it at its `default` (falling
back to `min` when no `default` is given) rather than drawing it from the
cube. `source_offset_fraction`, for example, disables back to the old
hard-coded centred source, because its `min` already is `0.0`.
`source_offset_fraction`, `source_l_pixels`, `source_m_pixels`,
`declination_deg` and `integration_seconds` all ship disabled. The last two
sit inside the MS skeleton cache key, so enabling either misses the skeletons
baked into the MeqTrees image and rebuilds them lazily during the run.

A dimension can be an explicit list rather than a box: `kind = "choice"` gives
each of its `values` an equal share of the cube, the way `band_start` splits
it between receiver bands. `min`/`max` are filled in from the list, so every
other reader treats it like any other dimension.

Two ways to see and change this without editing the file:

```
./ri params                                     # what is searched, what is pinned
./ri search wsclean --disable-param source_offset_fraction --enable-param channel_count
```

`--enable-param` / `--disable-param` are repeatable and set `NS_ENABLE_PARAMS`
/ `NS_DISABLE_PARAMS` (comma-separated names), which override `enabled` in
defaults.toml for that one invocation - an env-var edit, not a file edit, for
a one-off search. `--enable-param` wins if a name is passed to both.

Toggling a dimension changes PolyChord's dimension count, so - like
reordering `[[parameter_space]]` - it invalidates existing chains, and
`merge-nested-sampling-runs.py` refuses to merge runs whose `parameter_space`
differs.

### Receiver bands

A telescope only receives inside its bands, so `start_frequency_hz` is not
scaled onto a plain box: the `[[receiver_band]]` list in `defaults.toml` names
the bands, each gets an equal share of that unit-cube dimension, and the start
frequency is uniform inside the band its share picks. Equal share per band
rather than uniform across the union of them, or the 32 MHz-wide 4-band would
come up about once in 1500 draws and never actually be searched.

The committed list is the VLA's:

| Band | Range | Band | Range |
|---|---:|---|---:|
| 4 | 54-86 MHz (off) | X | 8-12 GHz |
| P | 224-480 MHz (off) | Ku | 12-18 GHz |
| L | 1-2 GHz | K | 18-26.5 GHz |
| S | 2-4 GHz | Ka | 26.5-40 GHz |
| C | 4-8 GHz | Q | 40-50 GHz |

A band takes `enabled` (default true), the same toggle `[[parameter_space]]`
has: `enabled = false` drops it from the search without deleting it, and the
searched frequency range is the union of the enabled bands only. Bands 4 and
P ship disabled. Changing which bands are enabled changes the prior on
`start_frequency_hz`, so - like toggling a dimension - it invalidates existing
chains.

Nothing in the code is VLA-specific: another telescope is a matter of
replacing the list (any number of bands, in any order, gaps allowed).

### Fitting the window into the band

`channel_count` and `channel_width_hz` are drawn from their own boxes, knowing
nothing about frequency, so a window can easily be wider than the room left
above the start frequency that came up. `fit_spectral_window()` in
`common.py` fits it to that room, giving up as little as possible at each
step:

1. the window fits - keep the draw;
2. it does not - narrow the channels until it does, if that stays at or above
   `channel_width_hz`'s min;
3. it would go below that - hold the width at the min and drop channels
   instead, if that stays at or above `channel_count`'s min;
4. even the smallest window does not fit, so the start frequency is too close
   to the top of its band to hold anything - draw another start frequency and
   start over.

The redraw steps a fixed distance (the golden ratio conjugate) around the unit
interval rather than drawing from an RNG, which spreads successive tries
across every band and keeps the prior transform a pure function of the cube -
what PolyChord requires, and what makes the `theta -> cube -> params` round
trip a fixed point: a fitted window is one that already fits, so re-deriving it
changes nothing.

Two consequences worth knowing. The fitting only ever gives ground, so the
mins are hard floors and the maxes are never exceeded - but a run does measure
narrower channels than it drew, and the prior on `channel_width_hz` is no
longer flat: it is deformed towards the narrow end near the top of each band.
And a box whose *smallest* window - `channel_count`'s min at
`channel_width_hz`'s min - fits no band at all cannot be fitted from any start
frequency, so `check_channel_box_against_bands()` refuses it at load.

### What the fitting costs

Every draw that is fitted rather than kept is a draw whose parameters are not
the ones the sampler asked for, and every redraw is work thrown away, so both
are counted. `WINDOW_FIT_STATS` tallies them and each run's `summary.json`
carries the result under `spectral_window_fitting`:

```json
"spectral_window_fitting": {
  "draws": 2000, "as_sampled": 1972, "width_reduced": 25, "count_reduced": 3,
  "redrawn_draws": 5, "redraws": 5, "seconds": 0.0017,
  "seconds_per_draw": 8.6e-07
}
```

The run also prints that line when it finishes. With the committed box the
fitting is cheap and rare - the window tops out at 12 MHz against a 32 MHz
narrowest band, so it only bites near the top of a band - and the cost is
microseconds a draw against seconds an evaluation. Widen `channel_count` or
`channel_width_hz` and the reduced counts climb, which is the number to watch:
it is how much of the box the run is not really searching.

`self_check_spectral_window()` is the guard - it samples the cube, asserts
every fitted window lands inside a band and inside the configured channel
boxes, forces each rung of the ladder above, and asserts the round trip comes
back to the same parameters.

The band guarantee is the sampler's. `simulate_point_source_ms.py` takes
`--start-frequency-hz` and `--channel-width-hz` as given, so a hand-run
simulate (or a smoke test) can still ask for whatever it likes.

### Cell size

The cell size is derived per evaluation, not fixed. R2D2 sizes its own pixels
from the sampling pattern it is handed - upstream `src/utils/io.py` takes the
longest projected baseline in wavelengths and sets

    image_pixel_size = 206265 arcsec / (super_resolution * 2 * max_proj_baseline)

so the WSClean runner applies the same formula (`image_pixel_size_arcsec()` in
`common.py`) to the `observation.max_proj_baseline_lambda` the simulator
records, and passes the result as `-scale`. Both imagers then reconstruct the
same sky at the same resolution, and each WSClean evaluation records the value
it used as `image_pixel_size_arcsec`.

A fixed `-scale` cannot work here, because `start_frequency_hz` spans 54 MHz to
50 GHz while the VLA-A maximum baseline does not move: against the 1 arcsec
cell this used to pass, the synthesized beam is ~31 arcsec at the bottom of
that range and ~0.04 arcsec at the top. The search would have been measuring
how badly WSClean's grid was mismatched to the sampled frequency - and only
WSClean's, since R2D2 rescaled either way.

`super_resolution` is 1.5, R2D2's own default, now written into the R2D2
config explicitly rather than left implicit, because WSClean's `-scale` is
derived from it.

### Source offset

`source_offset_fraction` moves the source off the phase centre, at a fixed 30
degree position angle (non-axis-aligned, to avoid the symmetries a purely
horizontal or vertical offset would have). At `0.0` it reproduces the old
hard-coded behaviour exactly: no bandwidth smearing, no time smearing, no
w-term, no pixel-interpolation error, and `point_source_forest.py` skips
K-Jones outright.

`source_offset_to_lm()` in `common.py` converts the fraction to an (l, m)
offset in arcsec using a *nominal* image half-width - `image_pixel_size_arcsec()`
against VLA-A's ~36 km maximum baseline and the sampled frequency, not the
`max_proj_baseline_lambda` the simulator will actually record - because the
source position has to reach `simulate_point_source_ms.py` before the MS (and
its real baselines) exist. `compute_image_metrics()` places the truth pixel at
that same offset (`source_pixel()`), so an off-centre evaluation is not scored
against a source that is not there.

`source_l_pixels` and `source_m_pixels` say the same thing in cartesian: a
signed offset from the central pixel, against that same nominal pixel size, so
the pair reaches any pixel rather than the points along one ray. They add to
`source_offset_fraction`'s offset, each being zero at its own default.
Enabling one alone leaves the other axis unexercised and gives up the symmetry
the 30 degree ray was chosen for, so enable both or neither.

Enabling the polar dimension and either cartesian one together is refused:
the offsets add, so one sky position comes from many draws and PolyChord
spends a dimension on a direction the likelihood is flat along.

`kind = "image_pixels"` in `defaults.toml` takes a `fraction`, not a min/max:
`load_all_parameter_specs()` resolves the box to `+/-(NS_IMAGE_DIM / 2) *
fraction`, `+/-8` pixels of the default 32-pixel image. A box in pixels would
mean a different fraction of the sky each time the image size moved.

Caveat: `ms_to_r2d2_mat.py` writes only `u` and `v` (see the bridge table
below) - `w` is dropped, so R2D2 sees a coplanar 2-D array while WSClean does
not. `source_offset_fraction`'s box tops out at 0.35 to stay inside the
small-field regime that keeps this an acceptable approximation rather than
comparing the two imagers on different physics. The cartesian boxes ship at
`fraction = 0.5`, and both axes at maximum sit `0.71` of the half-width out -
past that limit. Lower `fraction` to `0.25` to stay inside it.

Fixed hyperparameters (not searched) on every evaluation:

**WSClean:** `-niter 10,000`, `-auto-threshold 3.0`, and an `NS_IMAGE_DIM`-square
image, recorded in `summary.json` under `wsclean_fixed_hyperparameters`.
**R2D2:** the same `NS_IMAGE_DIM` image size as the WSClean run,
`num_iter 25`, `architecture unet`, `num_chans 64`, `ckpt_path
/checkpoints/R2D2_A1`, and `ckpt_realisations 1`, recorded in `summary.json`
under `r2d2_fixed_hyperparameters`.

### What the image size costs

`defaults.toml` is where the size is written down, and it is the only place:
`image_dim()` in `common.py` reads it and stops the run if the key is gone
rather than carrying a default of its own, so changing it there changes it
everywhere and nothing else needs editing. `NS_IMAGE_DIM` overrides it the
way the environment overrides every other key in that file, and the size is
recorded in `run.env` and grouped by in the benchmark ledger, so
`./ri bench run <imager> --interleave NS_IMAGE_DIM 128 32` measures a change to
it like any other setting. It is not a search dimension: it also sets the field
of view (`source_offset_to_lm()` scales the half-width by it), so two sizes are
two different skies, and the R2D2 checkpoints were trained at one resolution.
Runs archived before the default moved were scored at 128 - `NS_IMAGE_DIM=128`
reproduces them.

Median per-evaluation imaging time over the `default` preset on the 20-CPU
host, four interleaved repeats per arm (`wsclean`) and three (`r2d2`):

| dim | R2D2 image | speed-up | WSClean image | speed-up |
| --: | ---------: | -------: | ------------: | -------: |
| 128 |     7451ms |     1.0x |         150ms |     1.0x |
|  64 |     2584ms |     2.9x |         123ms |     1.2x |
|  32 |     1411ms |     5.3x |         100ms |     1.5x |

R2D2 fits `0.96s + 0.39ms/pixel` across all three sizes, so the fixed second -
`docker exec`, the MS-to-`.mat` bridge, the operator norm, and orchestrating 25
iterations - caps the whole knob at about 7.5x however small the image gets.
WSClean does not fit a pixel count at all: 16x fewer pixels buys 1.5x, because
its evaluation is CLEAN iterations and gridding ~3000 visibilities, and neither
shrinks with the image. Peak memory does not move either (3.46 GB at every
size, it is the checkpoint), so a smaller image buys no extra R2D2 ranks.

None of that is the wall-clock of a search. The sampler's path changes with the
likelihood surface: the WSClean arms above took 980 evaluations at 128, 710 at
64 and 1053 at 32, so the 32 arm finished *slower* (24.8s against 17.7s) while
each evaluation was 1.5x cheaper. The R2D2 arms happened to stay level (44, 43,
43), which is what makes its 3.2x and 5.8x whole-run speed-ups real.

## MS to R2D2 `.mat` bridge

R2D2-RI reads visibilities from a MATLAB `.mat` file via `load_data_to_tensor()`
in the upstream `src/utils.py`. The nested-sampling simulator produces a CASA
Measurement Set (`sim.ms`) that WSClean consumes directly. The R2D2 run adds
`scripts/lib/nested_sampling/ms_to_r2d2_mat.py`, which the rank's simulate
worker runs in-process inside the MeqTrees sidecar (python3-casacore plus
scipy), and which writes the minimal field set R2D2 loads without flag
metadata:

| Field | Meaning |
|---|---|
| `u`, `v` | UV coordinates in wavelengths, flattened across rows and channels |
| `y` | Complex visibilities for correlation index 0 (parallel-hand Stokes I) |
| `nW` | `sqrt(WEIGHT)` from the MS, divided by `--noise-sigma-jy` |

The simulator leaves `WEIGHT` at makems' 1.0 rather than writing `1/sigma^2` into every row - it is one number for the whole MS, and writing it was a third of the simulate stage. `polychord_r2d2.py` passes the sigma from the evaluation's `simulation.json` (`noise.complex_sigma_jy`) as `--noise-sigma-jy`, so `nW` comes out exactly where the column used to put it. The default of 1.0 takes `WEIGHT` at face value, which is what a Measurement Set carrying real weights wants.

Imaging weights are generated inside R2D2 when `data_weighting: True` in the
per-evaluation YAML config. The converter does not replicate the bundled
`data_3c353.mat` pruning or tau-compressed weight fields.

## Metrics and objective

For each sample, the pipeline records:

| Metric | Source |
|---|---|
| `snr` | Reconstructed image peak divided by off-source RMS |
| `log_snr` | `log10(snr)` |
| `off_source_rms_jy` | Off-source RMS in Jy/beam |
| `total_rms_jy` | RMS of (reconstructed image − one-pixel truth) over all pixels |
| `peak_jy_per_beam` | Peak absolute flux in the reconstructed image |
| `relative_l2_error` | Image residual versus the one-pixel point-source truth |
| `peak_flux_abs_error_jy` | Absolute centre-pixel flux error |
| `sigma_res` | Paper data-fidelity \(\overline{\sigma}_{\textrm{res.}}=\|\widehat{\mathbf{r}}\|_2/\|\mathbf{x}_{\textrm{d}}\|_2\) (final residual dirty over dirty) |
| `wall_seconds` | Imaging container runtime |
| `peak_memory_bytes` | Peak imaging memory: GNU `time -v` for WSClean; for R2D2 the imaging worker's own high-water RSS, which is a running maximum across that rank's evaluations |

PolyChord maximizes whatever value the run returns as its log-likelihood. The
default objective is `total_rms_jy` (RMS of the reconstructed image minus
the one-pixel truth, over all pixels).

An optional composite `badness` score is also available (higher means worse
reconstruction or a more expensive run):

```text
max(0, 3 - log_snr)
+ min(relative_l2_error, 10)
+ 0.05 * min(wall_seconds / 60, 5)
+ 0.02 * min(peak_memory_bytes / 2 GiB, 5)
```

### Choosing the objective (`--metric` / `NS_METRIC`)

Both `polychord_wsclean.py` and `polychord_r2d2.py` accept
`--metric <value>` (default `total_rms_jy`). The shell wrappers forward
`NS_METRIC`, whose default lives in `defaults.toml`, with the same value.
Resolution order:

1. `badness` - the composite formula above.
2. Any bare metric name from the table - use that raw value directly as the
   objective (including the default `total_rms_jy`).
3. Any other string - treat it as an arithmetic expression over the same metric
   names (for example `log_snr + 0.1 * wall_seconds`, or the composite formula
   rewritten by hand).

Expressions are compiled once at startup (before any Docker evaluations) and
evaluated in a restricted namespace: no Python builtins, metric names as locals,
and `math` module functions available by name. A typo or unsafe expression fails
immediately at startup.

PolyChord always maximizes the returned value with no automatic sign flip. The
`badness` composite is oriented so higher is worse. Raw metrics keep their
natural orientation: the default `total_rms_jy` search prefers higher
whole-image RMS error, `--metric snr` searches for the highest-SNR corner, and
a worst-SNR search must negate explicitly (`--metric "-snr"` or
`--metric "1/snr"`). `off_source_rms_jy` and `sigma_res` are also
higher-is-worse (noisier reconstruction / worse data fidelity); search for the
best corner with `--metric "-total_rms_jy"`, `--metric "-off_source_rms_jy"`
or `--metric "-sigma_res"`. Failed simulations or imaging runs still receive
objective `100.0`.

Each evaluation record and `summary.json` store the chosen value in an
`objective` field. `summary.json` also records the `--metric` string and a
`likelihood_framing` sentence describing what was optimized.

## Profiling

Every run records stage timings; see [nested-sampling-profiling.md](nested-sampling-profiling.md).

## Output files

### WSClean

Each likelihood evaluation:

```text
evaluations/eval-*/simulation.json
evaluations/eval-*/wsclean/recon-image.fits
evaluations/eval-*/wsclean/recon-dirty.fits
evaluations/eval-*/wsclean/recon-residual.fits
evaluations/eval-*/metrics.json
```

### R2D2

Each likelihood evaluation:

```text
evaluations/eval-*/simulation.json
evaluations/eval-*/r2d2_data.mat   (deleted once the evaluation is scored)
evaluations/eval-*/r2d2_config.yaml
evaluations/eval-*/r2d2/r2d2_data/R2D2_model_image.fits
evaluations/eval-*/r2d2/r2d2_data/dirty_normalised.fits
evaluations/eval-*/r2d2/r2d2_data/R2D2_residual_dirty_image.fits
evaluations/eval-*/metrics.json
```

Scored evaluations build `sim.ms` in shared tmpfs and delete it after scoring;
R2D2 also deletes its derived `.mat`. Failed evaluations retain all artefacts,
and `./ri search --keep-measurement-sets` retains them for every evaluation.
Once the run finishes, it keeps the three images above, and the imager's logs,
only for the 20 worst and 20 best evaluations by objective and one in every 100
between them, so a finished run holds a few hundred of each rather than one set
per evaluation; `NS_KEEP_ALL_IMAGES=1` keeps every one. That pruning runs when
the run ends, so a run still going still has every image it has written - the
report's gallery applies the same 20/20/one-in-100 policy itself rather than
showing whatever is on disk, which is what keeps `./ri report --live` from
drawing a card per evaluation. Before a WSClean log is
dropped, `clean_stop_reason` (`threshold` or `max-iterations`),
`clean_iterations` and `clean_major_iterations` are read out of it into the
evaluation's `metrics`, so whether CLEAN converged outlives the log.
See [nested-sampling-disk-footprint.md](nested-sampling-disk-footprint.md) for
pruning details. The record's `params` (`noise_seed` included) reproduce the MS,
and the images with it.

### Run summary and reports

Run-level summary, written only once PolyChord returns:

```text
summary.json
```

Everything the run printed, written as it goes and appended to by `./ri
resume`, so it survives a run that never reaches `summary.json`:

```text
run.log
```

This is the only artifact that records *why* a run stopped - a traceback out
of the PolyChord container reaches nowhere else. `./ri health` quotes its last
line for a stopped run.

One line per time the run stopped and started again, when that happened at all
- `<stamp> exit N after M evaluations` for a restart it made itself (see "A run
that dies restarts itself"), `<stamp> resumed at N evaluations` for a `./ri
resume` someone typed. Both because `./ri health` reads this file to know which
gaps between evaluations were the run not running, and takes them out of every
rate it prints:

```text
restarts.log
```

View completed runs (settings, evidence, per-evaluation metrics and
reconstructions) in the nested-sampling HTML report:

```bash
./ri report
# open reports/nested-sampling-report/index.html

./ri report --last 1
./ri report --run results/nested-sampling/r2d2-vlaa-merged-20260818T125604Z
./ri report --upgrade
./ri report --force
```

Each run gets its own page, `reports/nested-sampling-report/<run>.html`,
plus an `index.html` that lists every run on disk and links into them; each
run page links back to the index. Rendering a run means reading its FITS
output, so **run pages that are already up to date are skipped** - a re-run
only builds pages for new runs.

The index has a toolbar above the run cards: filter by algorithm (R2D2 /
WSClean) or by merged/unmerged, and sort newest/oldest or by eval count. It is
plain client-side JavaScript over the cards already on the page - no rebuild
or server needed, and it works the same off a `file://` open as it does
through `./ri serve`.

The index always lists every run on disk, whatever `--last` or `--run`
narrowed the page build to, so the values on its cards are cached in
`reports/nested-sampling-report/.index-facts.json`, keyed by each summary's
size and mtime. Without it a `./ri report --last 1` over a thousand runs
reparses every `summary.json` under `results/nested-sampling/` - gigabytes of
JSON, nearly all of it the `evaluations` array that a card shows as two
counts. `--force` rebuilds the cache from scratch; a change to what
`index_entry_facts` returns requires incrementing `INDEX_FACTS_VERSION` in
`scripts/lib/generate_report.py`.

Plots are cached PNGs under
`reports/nested-sampling-report/images/`; unchanged inputs are skipped. Delete
the report directory to force a full rebuild. A drawing-code change requires
incrementing `IMAGE_RENDER_VERSION` in `scripts/lib/generate_report.py`.
Performance measurements and implementation details live in
[nested-sampling-throughput.md](nested-sampling-throughput.md).

Every page carries the version of the report generator that wrote it (the
hash of `scripts/lib/generate_report.py`, in a
`<meta name="report-version">` tag), so changing the card design, the CSS or
anything else in that file makes existing pages **outdated** rather than
silently stale. Outdated pages are still skipped by a plain run - it says how
many it saw - and the index flags them with an `outdated page` badge.
`UPGRADE=1` rebuilds exactly those, bringing every page up to the current
design.

`LAST=N` only considers the newest N runs (timestamp sort). `RUN=` targets
one named run (directory, repo-relative path, or directory name) and always
rebuilds its page. `UPGRADE=1` rebuilds the pages an older report version
wrote. `FORCE=1` rebuilds every page in scope, up to date or not. `LAST=` and
`RUN=` cannot be combined. Make cannot take `--last`; use `LAST=1`.

The report globs `results/nested-sampling/*/summary.json` directly
(no manifest join), so a merged run directory (see **Merge runs** below) shows
up as its own card automatically. Evidence prefers a `log_z` /
`log_z_err` pair already in the summary (written for merged runs); otherwise
it parses PolyChord `chains/*.stats` for log(Z). It shows
each run's total wall-clock duration (from `total_wall_seconds`, when present)
top-right in the card header. The run page itself carries only the
best-effort `anesthetic` KDE contour corner plot, so it loads without
decoding one raster per evaluation; per-run images - the shared synthesized
ground-truth image and a per-evaluation card gallery (reconstruction,
objective, and searched parameters) - live on their own
`<run>-images.html` page, linked from the run page and written alongside it. Corner plots are weighted by the raw log-likelihood (the
failure score), not by nested-sampling posterior mass. Runs are ordered newest-first by the UTC timestamp in the
run directory name.

### Read the report from another machine

On a headless host, `./ri serve` serves the report:

```bash
./ri serve                  # loopback:8000
./ri serve --port 9000      # [REPORT_PORT]
./ri serve --bind 0.0.0.0   # [REPORT_BIND]
```

The default loopback bind prints this SSH tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<host>
# then open http://localhost:8000/
```

Loopback limits traffic to the host and tunnel, but provides no authentication;
`--bind 0.0.0.0` exposes the unauthenticated report to anything reaching it.

The tunnel follows the selected bind. Set `REPORT_SSH_HOST` when its guessed
address (`hostname -I`) fails behind NAT.

For a copied report, `python3 -m http.server` needs no dependencies and serves
until Ctrl-C. Copy the whole directory (`rsync -a`), since pages link to
`images/`.

### Replay a run in anesthetic's GUI

For an interactive nested-sampling replay (live points vs \(\ln X\), \(\beta\)
tempering) with human-readable parameter labels, run on the **host** (needs a
display; not inside Docker/Colima):

```bash
./ri plot gui
./ri plot gui results/nested-sampling/wsclean-vlaa-<UTC timestamp>
```

With no `RUN=`, the latest *completed* run under `results/nested-sampling/*/`
is used - either a plain run (`summary.json` and `chains/`) or a merged
run (`summary.json` with `merged_from`, no local `chains/`). For a plain
run the script writes/refreshes `chains/<root>.paramnames` from that run's
`summary.json` / `parameter-space.json`; either way it passes only the
searched Fourier parameter names into `samples.gui(params=...)` (not `logL` /
`logL_birth` / `nlive`). Close the GUI window to return to the shell.
Requires the host `uv` project dependency `anesthetic`
(`uv add anesthetic` if missing).

The run also writes a standard environment manifest through
`scripts/record-environment.sh`.

## Merge runs

Independent PolyChord runs of the **same likelihood and prior** can be fused
after the fact into one run directory, without re-running PolyChord. This is
post-processing only: it concatenates nested-sampling dead points with
`anesthetic.samples.merge_nested_samples` and recomputes live-point weights.
Evaluation directories, FITS images, and PolyChord chain files are never
copied; the merged summary just points back at the absolute evaluation paths
and source run directories already on disk.

Sampler effort may differ between sources; the search itself must not:

| May differ | Must match |
|---|---|
| `NS_NLIVE` / `polychord.nlive` | `algorithm` |
| `NS_NUM_REPEATS` / `polychord.num_repeats` | `vla_config` |
| `NS_MAX_NDEAD` / `polychord.max_ndead` | `metric` |
| `seed`, `mpi_procs` | `parameter_space` (name/min/max/kind) |
| | `r2d2_fixed_hyperparameters` **or** `wsclean_fixed_hyperparameters` |

WSClean and R2D2 runs never merge with each other, nor do runs with a
different `--metric` / `NS_METRIC` or a different prior box (the prior box is
`[[parameter_space]]` in `defaults.toml`, copied into
every `summary.json` as `parameter_space`).

With no directories, every completed source run under
`results/nested-sampling/` is grouped by the must-match fields above
and one merged directory is written per group of 2+. Incomplete dirs,
previous merges (`merged_from`), and singleton groups are skipped.
Zero groups of 2+ exits non-zero. `--out` is only valid with an explicit
directory list.

```bash
uv run scripts/merge-nested-sampling-runs.py
./ri merge

uv run scripts/merge-nested-sampling-runs.py \
  results/nested-sampling/r2d2-vlaa-AAA \
  results/nested-sampling/r2d2-vlaa-BBB

./ri merge results/nested-sampling/r2d2-vlaa-AAA results/nested-sampling/r2d2-vlaa-BBB
```

Writes `results/nested-sampling/<algorithm>-vlaa-merged-<UTC>/summary.json`
(pass `--out DIR` on the explicit form to pick a different output directory).
The explicit form refuses with a non-zero exit on fewer than two runs, a run
missing `summary.json` or `chains/`, or any must-match field above differing.
`polychord.nlive` in the merged summary is the sum of source nlives;
`num_repeats` / `max_ndead` / `seed` stay a single value when all sources
agree, else become a list. Pooled `evaluations` keep source argument order,
are renumbered `eval_id` `1..N` (originals kept as `source_eval_id` /
`source_run`), and keep their original absolute `paths`.

`./ri report` and `./ri plot gui <merged-dir>` both
treat the merged directory as a completed run - see the sections above.
