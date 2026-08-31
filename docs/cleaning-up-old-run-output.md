# Cleaning up old run output

Bringing a machine's `results/` in line with what a run produces today. Every
path here is gitignored, so none of it arrives by pulling: a checkout that has
been running searches for a while holds run output written under older
retention rules, and only re-running the pruners brings it forward.

Safe to run more than once, and safe to run on a machine that is already clean:
everything below is idempotent and skips what it does not recognise.

## What changed, and why old runs differ

Three rules have tightened, in this order. A run is stuck at whichever rule was
in force when it finished.

| Rule | Since | Old runs therefore hold |
| --- | --- | --- |
| The Measurement Set is deleted once an evaluation is scored | PR #80 | a `sim.ms` per evaluation, ~1.2 MB each |
| Images kept only for the 20 worst, 20 best and one in 100 between | PR #92 | a reconstruction per evaluation, ~75 KB each |
| Imager logs kept alongside those images; `-niter` raised to 1000 | this change | a `wsclean.stdout.log` per evaluation, ~11 KB each |

`recon-model.fits`, `recon-psf.fits`, R2D2's `PSF.fits`, `VLAA_ANT` and
`r2d2_data.mat` have no reader at all and go from every scored evaluation.

## Do it

`scripts/prune-run-artefacts.py` applies all three rules, from the records
`summary.json` already embeds. Point it at every run directory on the machine -
including the ones inside worktrees, which is where output accumulates
unnoticed:

```bash
cd /path/to/ri-reproducibility

ls -d results/nested-sampling/*/ > /tmp/runs.txt
ls -d .claude/worktrees/*/results/nested-sampling/*/ >> /tmp/runs.txt 2>/dev/null
ls -d ../ri-reproducibility-*-worktrees/*/results/nested-sampling/*/ >> /tmp/runs.txt 2>/dev/null
sort -u /tmp/runs.txt -o /tmp/runs.txt
wc -l /tmp/runs.txt

xargs -a /tmp/runs.txt -d '\n' python3 scripts/prune-run-artefacts.py
```

`xargs` may split a long list across several invocations, so the output holds
one `removed N ...` total per batch. Sum them:

```bash
grep '^removed [0-9]* ' out.log | awk '{s+=$2} END{print s}'
```

Expect it to take minutes and to delete a lot of files. On the machine this was
first run on it removed 855,317 images from 1,065 runs and freed 102 GB of
Measurement Sets before that.

## What it will not touch, on purpose

- **Runs with no `summary.json`.** They are incomplete or resumable, and
  `./ri resume` rebuilds its cache by walking `evaluations/`. Pruning one
  destroys the ability to continue it. `./ri runs` lists which those are.
- **Failed evaluations.** A record with an `error` keeps every artefact: the
  evidence a failure-mode search exists to produce is never pruned.
- **`checkpoints/`.** The R2D2 weights. The imager will not run without them.
- **`chains/` and `summary.json`.** The posterior and the complete record of
  every evaluation. These are the run; they are never pruned.

## Checking it worked

```bash
# Only failed evaluations should still hold these.
find results .claude/worktrees/*/results -name 'sim.ms' -o -name 'PSF.fits' \
     -o -name 'recon-model.fits' | wc -l

# A finished run should hold a few hundred images, not one per evaluation.
python3 - <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
summary = json.loads((run / "summary.json").read_text())
records = summary["evaluations"]
named = sum(1 for r in records for k in ("image", "dirty", "residual")
            if (r.get("paths") or {}).get(k))
on_disk = sum(1 for _ in (run / "evaluations").rglob("*.fits"))
print(f"{len(records)} evaluations, {named} images named, {on_disk} on disk")
assert named == on_disk, "summary and disk disagree"
PY
```

`named == on_disk` is the invariant: a summary must never name a file that is
not there, which is what lets the report fall back to its placeholder.

Then rebuild the report and confirm it renders, with a dash where an image was
not retained:

```bash
./ri report --force
```

## Things outside the repo worth reclaiming

Not run output, and not covered by the pruner, but the same machine usually has
them:

```bash
docker builder prune -f          # build cache; the first run of this freed 48 GB
docker image prune -f            # dangling images only - `-a` would force a slow ./ri build

# Stale /tmp, only what nothing has touched for two days. Check first.
find /tmp -maxdepth 1 -mindepth 1 -mtime +1
```

Before deleting anything in `/tmp`, check what is holding it open - long-lived
agent daemons put their install directories there, and killing them is fine but
should be deliberate:

```bash
lsof +D /tmp 2>/dev/null | awk 'NR>1 {print $1, $2}' | sort -u
```

## If the machine is short of space mid-run

Nothing here helps: a run holds every image and log until it finishes, because
retention ranks evaluations by objective and that ordering only exists once the
last one is scored. Prune *other* runs, or let it finish. `./ri health` reports
GB/hour against free space and warns before ENOSPC.
