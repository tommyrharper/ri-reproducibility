#!/usr/bin/env python3
"""Regression check for merging sources with mismatched on-disk .paramnames.

Run: uv run python3 scripts/lib/nested_sampling/test_anesthetic_io.py

Before the fix, a merge source whose chains/ already had a .paramnames file
on disk (e.g. because anesthetic-gui.py had been run against it) kept its
real tex label, while a source without one got a blank tex label. concat
then saw two same-named-but-differently-labelled columns as duplicates, and
plot_2d raised ValueError on the ambiguous WeightedLabelledSeries. This
builds exactly that mismatched pair and merges them through the real
load_nested_samples() path.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PARAM_NAMES = [
    "log10_dynamic_range",
    "observation_minutes",
    "channel_count",
    "start_frequency_hz",
    "channel_width_hz",
]

DEAD_BIRTH_ROWS = [
    "0.1 0.2 0.3 0.4 0.5 -1.0 -2.0",
    "0.2 0.3 0.4 0.5 0.6 -0.5 -1.5",
    "0.3 0.4 0.5 0.6 0.7 -0.2 -1.0",
    "0.4 0.5 0.6 0.7 0.8 -0.1 -0.9",
]


def _write_source(nested_dir: Path, run_id: str, with_paramnames: bool) -> None:
    chains = nested_dir / run_id / "chains"
    chains.mkdir(parents=True)
    (chains / f"{run_id}_dead-birth.txt").write_text("\n".join(DEAD_BIRTH_ROWS) + "\n")
    if with_paramnames:
        lines = [f"{name}\t{name}" for name in PARAM_NAMES]
        (chains / f"{run_id}.paramnames").write_text("\n".join(lines) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        nested_dir = repo_root / "results" / "nested-sampling-poc"
        nested_dir.mkdir(parents=True)

        _write_source(nested_dir, "source-a", with_paramnames=False)
        _write_source(nested_dir, "source-b", with_paramnames=True)

        merged_dir = nested_dir / "merged-run"
        merged_dir.mkdir()
        (merged_dir / "poc-summary.json").write_text(
            json.dumps(
                {
                    "merged_from": [
                        {"path": "results/nested-sampling-poc/source-a"},
                        {"path": "results/nested-sampling-poc/source-b"},
                    ],
                    "parameter_space": [{"name": name} for name in PARAM_NAMES],
                }
            )
        )

        from anesthetic_io import load_nested_samples

        samples = load_nested_samples(merged_dir)

        top_level_names = [c[0] if isinstance(c, tuple) else c for c in samples.columns]
        tex_labels = [c[1] for c in samples.columns if isinstance(c, tuple) and c[0] in PARAM_NAMES]
        for name in PARAM_NAMES:
            count = top_level_names.count(name)
            assert count == 1, f"expected exactly one {name!r} column, found {count}"
        for tex in tex_labels:
            assert str(tex).startswith("$") and str(tex).endswith("$"), tex

        import matplotlib

        matplotlib.use("Agg")
        axes = samples.plot_2d(PARAM_NAMES[:2])
        assert axes is not None

        print("OK: merged mismatched-.paramnames sources without duplicate columns")


if __name__ == "__main__":
    main()
