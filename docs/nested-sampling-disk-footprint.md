# What a run costs on disk, and the ceiling that puts on its size

Every other page under `docs/` about nested-sampling throughput asks how long
one evaluation takes. This one asks how many bytes it leaves behind, because
that - not CPU - is what says how big a search can be. Five iterations of
per-evaluation optimisation have taken a WSClean evaluation to ~200 ms; at
~50 evaluations a second on this host a search that runs for ten hours is
1.8 million evaluations, and at the footprint this repo had before this page
that is 700 GB against 188 GB free.

The change this page records prunes four of the five FITS images WSClean
writes per evaluation, once the metrics have been read out of them:
**393.6 KB an evaluation down to 99.9 KB, a factor of 3.94**, for 48.5 us of
`unlink()` an evaluation. It also records the measurement that says the
`evaluations/` directory itself does *not* need sharding, which was the other
suspected scaling wall.

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

`PRUNED_ARTEFACTS` in `scripts/lib/nested_sampling/common.py` gains the four
images:

```python
PRUNED_ARTEFACTS = (
    ("sim.ms", "measurement_set"),
    ("VLAA_ANT", None),
    ("r2d2_data.mat", "mat"),
    ("wsclean/recon-dirty.fits", "dirty"),
    ("wsclean/recon-residual.fits", "residual"),
    ("wsclean/recon-model.fits", None),
    ("wsclean/recon-psf.fits", None),
)
```

Everything about the surrounding machinery is unchanged, which is the point of
putting them there rather than writing new code:

- a **failed** evaluation keeps all of them, because a failure is what this
  project exists to find and its artefacts are the first thing anyone will want;
- `NS_KEEP_MEASUREMENT_SETS=1` (`./ri search --keep-measurement-sets`) keeps
  all of them, so the replay benchmarks in
  `docs/nested-sampling-throughput.md` are unaffected;
- the record's `paths` block drops the `dirty` and `residual` keys with the
  files, so a `metrics.json` never names a file that is not there;
- `recon-image.fits`, the restored image, is never pruned. That is the evidence
  a failure-mode search exists to produce, and
  `self_check_evaluation_pruning()` has asserted it survives since iteration 8.

WSClean is not asked to stop *writing* the four images, only to have them
deleted. There is no flag for the model and the psf, the dirty image is needed
for the metrics, and iteration 14 already measured that the five FITS writes
are free (`-no-dirty` changed nothing). Deleting instead of not-writing also
means the pages usually never reach the disk at all: 293 KB an evaluation of
dirty page cache is unlinked ~200 ms after it is written, so at 50 evaluations
a second this is ~15 MB/s of writeback that no longer happens.

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

## What is left

- **74.9 KB of restored image an evaluation, 75% of what remains.** It is a
  128x128 float32 in a 74880-byte FITS. Nothing smaller is possible without
  either dropping the evidence the search exists to produce or storing it in
  something that is not a FITS file; gzip on float32 noise is not worth the
  complexity. The lever that would matter is a policy one - keep the image for
  a bounded sample of evaluations plus every failure - and that reverses an
  explicit project decision, so it is the owner's to take, not this page's.
- **12.3 KB of `wsclean.stdout.log`, 12%.** That is the `-log-time` phase
  timeline `./ri profile <run> --phases` reads, and it is the only reason that
  command needs no rig. Worth keeping.
- The R2D2 side is untouched. `polychord_r2d2.py` records one image path per
  evaluation and this repo has never run an R2D2 search (`./ri
  fetch-checkpoints` needs a browser), so there is no measured footprint to
  prune against.
