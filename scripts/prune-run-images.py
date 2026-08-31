#!/usr/bin/env python3
"""Apply the image retention policy to runs that finished before it existed.

A run scored before prune_run_images() was wired into the summary writers kept
an image for every evaluation. This applies the same policy after the fact,
from the records summary.json already embeds, and rewrites the summary so it
never names a file this deleted.

Only finished runs are touched: summary.json is written once PolyChord returns,
so a run without one is incomplete or resumable, and `./ri resume` rebuilds its
cache by walking evaluations/.

    scripts/prune-run-images.py results/nested-sampling/*/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib" / "nested_sampling"))

from common import (  # noqa: E402
    PRUNED_ARTEFACTS,
    evaluations_keeping_images,
    prune_run_images,
    write_json_atomic,
)


def drop_always_pruned(evaluations_dir: Path, records: list[dict]) -> int:
    """Remove artefacts a run scored today would never have kept.

    Runs from before an entry joined PRUNED_ARTEFACTS still hold it - R2D2's
    PSF.fits most of all. Failed evaluations keep everything, as they do live.
    """
    import shutil

    removed = 0
    for record in records:
        if "error" in record:
            continue
        eval_dir = Path((record.get("paths") or {}).get("eval_dir") or "")
        local = evaluations_dir / eval_dir.name
        if not local.is_dir():
            continue
        for name, path_key in PRUNED_ARTEFACTS:
            target = local / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
            elif target.is_file():
                target.unlink()
                removed += 1
            if path_key:
                (record.get("paths") or {}).pop(path_key, None)
    return removed


def main(argv: list[str]) -> int:
    total_removed = 0
    for arg in argv:
        run = Path(arg)
        summary_path = run / "summary.json"
        if not (run / "evaluations").is_dir() or not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            print(f"skipped {run.name}: unreadable summary.json")
            continue
        records = summary.get("evaluations")
        if not isinstance(records, list) or not records:
            continue
        removed = prune_run_images(run / "evaluations", records)
        removed += drop_always_pruned(run / "evaluations", records)
        if removed:
            write_json_atomic(summary_path, summary)
            kept = len(evaluations_keeping_images(records))
            print(f"{run.name}: removed {removed} images, kept {kept} of {len(records)} evaluations")
        total_removed += removed
    print(f"removed {total_removed} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
