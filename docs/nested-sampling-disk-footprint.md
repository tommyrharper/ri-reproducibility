# Run disk footprint

Removing four post-metric FITS files cut WSClean footprint **3.94x**, from
393.6 KB to 99.9 KB per evaluation; `evaluations/` needs no sharding here.

## What an evaluation directory holds

Measured over `results/nested-sampling/i30-foot`, a real `--nlive 30
--num-repeats 2 --mpi-procs 10` WSClean search at HEAD before the change
(1737 scored evaluations, `du -sb` over `evaluations/`):

| File | Bytes each | Total | Share | Who reads it |
| --- | --- | --- | --- | --- |
| `wsclean/recon-image.fits` | 74880 | 130.1 MB | 19.0% | the analyst; `generate_report.py`'s gallery |
| `wsclean/recon-residual.fits` | 74880 | 130.1 MB | 19.0% | `compute_image_metrics()`, for `sigma_res`, then nothing |
| `wsclean/recon-model.fits` | 74880 | 130.1 MB | 19.0% | **nothing, ever** |
| `wsclean/recon-psf.fits` | 72000 | 125.1 MB | 18.3% | **nothing, ever** |
| `wsclean/recon-dirty.fits` | 72000 | 125.1 MB | 18.3% | `compute_image_metrics()`, for `sigma_res`, then nothing |
| `wsclean.stdout.log` | 12255 | 19.1 MB | 2.8% | `./ri profile <run> --phases` |
| `metrics.json` | 3740 | 6.5 MB | 0.9% | everything |
| `simulation.json` | 987 | 1.7 MB | 0.3% | the evaluation itself |
| `simulate.stdout.log` | 987 | 1.7 MB | 0.3% | a failure investigation |
| `*.stderr.log` | 0 | 0 | 0% | a failure investigation |
| **total** | | **683.7 MB** | | **393.6 KB an evaluation** |

The Measurement Set is already gone by this point - `prune_evaluation_artefacts()`
has deleted it, which took a 1.44 MB evaluation directory to 0.43 MB back in
iteration 8. What that left is 94% FITS, and four of the five images have no
reader after the evaluation that produced them has been scored.

`recon-model.fits` and `recon-psf.fits` have never had one at all: the model is
the clean-component list rendered as an image and the psf is a function of the
uv coverage, and no script in this repo opens either. `recon-dirty.fits` and
`recon-residual.fits` are opened exactly once, by `compute_image_metrics()` in
`common.py`, which reduces them to the single number `sigma_res` in
`metrics.json` - and that happens *before* `write_evaluation_record()` runs, so
by the time pruning happens their only consumer is finished with them.

## The change

`PRUNED_ARTEFACTS` in `scripts/lib/nested_sampling/common.py` owns this list and
the shared pruning path. Failed evaluations and
`NS_KEEP_MEASUREMENT_SETS=1` retain all artefacts; successful evaluations retain
the three images the retention policy below may keep, logs, JSON records, and
replayable inputs. The self-check covers these contracts; implementation
rationale and throughput measurements are in
[nested-sampling-throughput.md](nested-sampling-throughput.md).

## Which evaluations keep images

An evaluation writes a dirty image, a reconstruction and a residual dirty image;
`recon-model.fits`, `recon-psf.fits` and R2D2's `PSF.fits` are dropped at
scoring, having never had a reader. Of the three that survive, a finished run
keeps all three for the `IMAGE_KEEP_ENDS` (20) worst and best evaluations by
objective, then one evaluation in every `IMAGE_KEEP_STRIDE` (100) across the
ordered middle. The extremes are the failure modes the search exists to find,
plus the contrast that makes them readable; the stride keeps the ground between
them legible without an image per evaluation. `NS_IMAGE_KEEP_ENDS` and
`NS_IMAGE_KEEP_STRIDE` override the two, and `NS_KEEP_ALL_IMAGES=1` keeps
everything.

This is necessarily a whole-run decision. When an evaluation is scored, whether
it belongs in the worst 20 depends on evaluations that have not run yet, so
`prune_run_artefacts()` cannot hook the per-evaluation path and instead runs once,
from each algorithm's summary writer, just before `summary.json` is written. It
mutates the records the summary embeds, so a summary never names a file it
deleted and the report falls back to its placeholder. A consequence worth
knowing: a run holds every image until it finishes, so peak usage during a run
is unchanged and only the finished run is small.

Evaluations are identified by directory name, not `eval_id`: PolyChord reuses
the number across parameter vectors, so one run holds several `eval-0083-*`
directories and ranking on the number alone would spare or delete whole groups
together.

The imager's stdout goes the same way, and for the same reason: `./ri profile
--phases` reads per-phase timings out of it, and a few hundred evaluations a run
is plenty for the medians it reports. WSClean's stdout was 9.3 GB across 828,825
files. The part of it worth outliving the log - why CLEAN stopped and how far it
got - is read into each record at scoring time by `clean_convergence()`, so
`clean_stop_reason`, `clean_iterations` and `clean_major_iterations` survive in
`summary.json` after the log is gone. `PRUNED_EVALUATION_LOGS` owns the list.

`scripts/prune-run-artefacts.py` applies the same policy to runs that finished
before it existed, from the records `summary.json` already embeds. It only
touches finished runs: a run without a `summary.json` is incomplete or
resumable, and `./ri resume` rebuilds its cache by walking `evaluations/`.

## What it measures

Two real searches, both `--nlive 30 --num-repeats 2 --mpi-procs 10` on the same
seed (`881843876`), one either side of the change:

| Run | Evaluations | `du -sb evaluations/` | Per evaluation |
| --- | --- | --- | --- |
| `i30-foot`, before | 1737 | 683.7 MB | 393.6 KB |
| `i30-pruned`, after | 1260 | 125.9 MB | **99.9 KB** |

3.94x. The two runs score different numbers of evaluations from the same seed
because asynchronous MPI incorporates chains in completion order (see
`docs/nested-sampling-throughput.md`), which is also why their wall clocks are
not comparable; the per-evaluation figure is, and it is confirmed exactly by
deleting the same four files out of the *before* run afterwards, which takes it
to 99.8 KB an evaluation.

What is left per evaluation is 74.9 KB of restored image (75%), 12.3 KB of
`wsclean.stdout.log` (12%) and 5.7 KB of json and simulate log.

### It costs 48.5 us an evaluation

Timed over the 1737 evaluation directories of the *before* run, unlinking the
four files:

```
6948 unlinks over 1737 evaluations: 84.2 ms total, 48.5 us/evaluation
```

Against a ~200 ms evaluation that is 0.02%, which no end-to-end A/B on this
host can resolve (see the null-run discussion in
`docs/nested-sampling-io-placement.md`). It is recorded as a direct
measurement rather than an A/B for that reason.

## The `evaluations/` directory does not need sharding

The other suspected scaling wall was the directory itself: a run of the size
this repo is aiming at puts hundreds of thousands of subdirectories in one
`evaluations/`, and ext4's htree is often blamed for degrading there. It does
not, on this host's filesystem: creating `eval-%07d` directories with two small
files in each, in 20000-directory batches, up to 400000:

| Directories in the parent | Cost of one more evaluation directory |
| --- | --- |
| 20000 | 45 us |
| 100000 | 42 us |
| 200000 | 114 us |
| 300000 | 46 us |
| 400000 | 54 us |

Flat at ~45 us with excursions from concurrent load, i.e. 0.02% of an
evaluation and no trend. `os.listdir()` of the full 400000 entries costs
81 ms. There is nothing to fix, and the obvious "fix" - sharding evaluation
directories into `000/eval-...` buckets - would break every path recorded in
every archived `metrics.json` for no gain.

## What this buys

The four archived runs of the `--nlive` scan on disk give the scaling, all at
`--num-repeats 10`:

| Run | `nlive` | Evaluations | Evaluations per `nlive` |
| --- | --- | --- | --- |
| `curve-n25` | 25 | 5925 | 237 |
| `curve-n50` | 50 | 10757 | 215 |
| `curve-n100` | 100 | 18844 | 188 |
| `big-n200` | 200 | 34682 | 173 |

Evaluations are linear in `nlive` (slightly sub-linear over this range) and
linear in `num_repeats` - the sampler spends 2.1 likelihood calls per slice
step and `num_repeats` slice steps per replacement, so `evaluations ~= 17 x
nlive x num_repeats` at the high end of that table. Against 188 GB free on
this host:

| Run | Evaluations | Before | After |
| --- | --- | --- | --- |
| `--nlive 200 --num-repeats 10` (the largest run archived here) | 35k | 13.7 GB | 3.5 GB |
| `--nlive 500 --num-repeats 25` | 210k | 83 GB | 21 GB |
| `--nlive 1000 --num-repeats 50` | 850k | 335 GB - **does not fit** | 85 GB |
| the disk ceiling itself | | 477k evaluations | **1.88M evaluations** |

At ~50 evaluations a second, 1.88M evaluations is ~10 hours, so on this host
the binding constraint on a single search is now its wall clock rather than
its disk. Note that the ceiling is per *host*, not per run: several archived
runs share the 188 GB, and `./ri search` does not delete anything.

## Reproducing

```bash
# the footprint of any run, by file
find <run>/evaluations -type f -printf "%f\t%s\n" \
  | awk -F'\t' '{s[$1]+=$2;c[$1]++} END {for (k in s) printf "%8.1f MB %6d %s\n", s[k]/1e6, c[k], k}' \
  | sort -rn
du -sb <run>/evaluations   # divide by the evaluation count

# the pruning itself, with no images and no containers
python3 -c 'import sys; sys.path.insert(0, "scripts/lib/nested_sampling"); \
  import common; common.self_check_evaluation_pruning()'

# keep everything for a debugging run
./ri search wsclean --keep-measurement-sets ...
```

## Remaining footprint

- Restored image: 74.9 KB/evaluation (75%), the minimum measured FITS evidence;
  reducing it requires an explicit retention-policy change.
- `wsclean.stdout.log`: 12.3 KB/evaluation (12%), required by `./ri profile`.
- R2D2 remains unmeasured and unpruned; this repo has no R2D2 search archive.
