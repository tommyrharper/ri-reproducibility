#!/usr/bin/env python3
"""Shared anesthetic chain loading for the nested-sampling GUI and report.

Understands both a plain PolyChord run directory and a merged run directory
(``poc-summary.json`` with a ``merged_from`` list) written by
``scripts/merge-nested-sampling-runs.py``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling-poc"

# GetDist / anesthetic axis labels (wrapped in $...$ by anesthetic). Mirrors
# poc_common.PARAMETER_TEX_LABELS; duplicated (not imported) because this
# module is also used host-side, where poc_common's astropy import isn't
# installed.
PARAMETER_TEX_LABELS = {
    "log10_dynamic_range": r"\mathrm{log}_{10}(\rho_{DR})",
    "observation_minutes": r"t_{\mathrm{obs}}\,[\mathrm{min}]",
    "channel_count": r"n_{\mathrm{freq}}",
    "start_frequency_hz": r"\nu_{\mathrm{start}}\,[\mathrm{Hz}]",
    "channel_width_hz": r"\Delta\nu\,[\mathrm{Hz}]",
    "wsclean_niter": r"N_{\mathrm{iter}}",
    "wsclean_auto_threshold": r"\sigma_{\mathrm{thresh}}",
}


def _similar_run_dirs(name: str) -> list[str]:
    """Suggest nearby run directory names when RUN= looks like a typo."""
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    runs = [p.name for p in NESTED_SAMPLING_DIR.iterdir() if p.is_dir()]
    scored = []
    for run in runs:
        common = 0
        for a, b in zip(name, run):
            if a != b:
                break
            common += 1
        if common >= 8 or name.rstrip("0123456789") == run or run.startswith(name[:16]):
            scored.append((common, run))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [f"results/nested-sampling-poc/{run}" for _, run in scored[:5]]


def find_chain_root(target: Path) -> Path:
    """Return PolyChord file root (path without _dead-birth.txt suffix)."""
    target = target.expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()

    if not target.exists() and not Path(str(target) + "_dead-birth.txt").is_file():
        suggestions = _similar_run_dirs(target.name)
        hint = ""
        if suggestions:
            hint = " Did you mean:\n  " + "\n  ".join(suggestions)
        raise SystemExit(f"Path does not exist: {target}{hint}")

    if target.is_file():
        name = target.name
        for suffix in (
            "_dead-birth.txt",
            "_phys_live-birth.txt",
            "_dead.txt",
            "_equal_weights.txt",
            ".stats",
            ".paramnames",
            ".txt",
        ):
            if name.endswith(suffix):
                return target.with_name(name[: -len(suffix)])
        raise SystemExit(f"Unrecognized chain file: {target}")

    if target.is_dir():
        chains_dir = target / "chains" if (target / "chains").is_dir() else target
        dead = sorted(chains_dir.glob("*_dead-birth.txt"))
        if len(dead) == 1:
            return dead[0].with_name(dead[0].name[: -len("_dead-birth.txt")])
        if len(dead) > 1:
            names = ", ".join(p.name for p in dead)
            raise SystemExit(f"Multiple chain roots in {chains_dir}: {names}")
        raise SystemExit(f"No *_dead-birth.txt under {chains_dir}")

    dead = Path(str(target) + "_dead-birth.txt")
    if dead.is_file():
        return target
    raise SystemExit(f"No nested-sampling chains at {target}")


@contextmanager
def hide_empty_phys_live_birth(chain_root: Path) -> Iterator[None]:
    """Aside empty *_phys_live-birth.txt so anesthetic's np.loadtxt doesn't warn.

    Best-effort: the report runs against a read-only bind mount, where the
    rename raises OSError. Fall through to letting anesthetic warn instead.
    """
    path = Path(str(chain_root) + "_phys_live-birth.txt")
    aside = path.with_name(path.name + ".empty")
    moved = False
    if path.is_file() and path.stat().st_size == 0:
        try:
            path.rename(aside)
            moved = True
        except OSError:
            moved = False
    try:
        yield
    finally:
        if moved and aside.is_file() and not path.exists():
            aside.rename(path)


def _repo_root_from_run_dir(run_dir: Path) -> Path:
    """run_dir is <repo_root>/results/nested-sampling-poc/<id>."""
    return run_dir.parents[2]


def _resolve_source_run_dir(run_dir: Path, entry: Any) -> Path:
    """Resolve one merged_from entry as a repo-relative path or a sibling name."""
    repo_root = _repo_root_from_run_dir(run_dir)
    siblings_dir = run_dir.parent
    name = entry.get("name") if isinstance(entry, dict) else str(entry)
    path = entry.get("path") if isinstance(entry, dict) else None

    candidates = []
    if path:
        candidates.append(repo_root / path)
    if name:
        candidates.append(siblings_dir / name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(
        f"Cannot resolve merged_from source run {entry!r} under {repo_root} or {siblings_dir}"
    )


def read_chains_at(chain_root: Path):
    import warnings

    from anesthetic import read_chains

    with hide_empty_phys_live_birth(chain_root):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"loadtxt: input contained no data:.*_phys_live-birth\.txt",
                category=UserWarning,
            )
            return read_chains(str(chain_root))


def _mathtext_label(tex: str) -> str:
    """Wrap a GetDist-style tex fragment in $...$ for matplotlib / anesthetic."""
    tex = str(tex).strip()
    if tex.startswith("$") and tex.endswith("$") and len(tex) >= 2:
        return tex
    return f"${tex}$"


def label_chain_samples(samples, param_names: list[str]):
    """Stamp (name, tex_label) for every column matching param_names.

    Always overwrites both the name and the tex label, whether the column
    started as a raw positional PolyChord index or was already named from
    an on-disk .paramnames file, so every merge source presents identical
    column tuples before merge_nested_samples runs (mismatched tex labels
    otherwise make pandas.concat treat same-named columns as duplicates).
    """
    import numpy as np
    import pandas as pd

    new_tuples = []
    for col in samples.columns:
        name = None
        if isinstance(col, tuple):
            if isinstance(col[0], (int, np.integer)):
                idx = int(col[0])
                if idx < len(param_names):
                    name = param_names[idx]
            elif col[0] in param_names:
                name = col[0]
        if name is not None:
            new_tuples.append((name, _mathtext_label(PARAMETER_TEX_LABELS.get(name, name))))
        else:
            new_tuples.append(col)
    labelled = samples.copy()
    labelled.columns = pd.MultiIndex.from_tuples(new_tuples, names=samples.columns.names)
    return labelled


def load_nested_samples(run_dir: Path):
    """Load NestedSamples for a run directory, merging sources when needed."""
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "poc-summary.json"
    merged_from = None
    param_names: list[str] = []
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        merged_from = summary.get("merged_from")
        param_names = [
            str(spec["name"]) for spec in (summary.get("parameter_space") or []) if "name" in spec
        ]

    if merged_from:
        from anesthetic.samples import merge_nested_samples

        sub_samples = []
        for entry in merged_from:
            source_dir = _resolve_source_run_dir(run_dir, entry)
            chain_root = find_chain_root(source_dir)
            sub = read_chains_at(chain_root)
            if param_names:
                sub = label_chain_samples(sub, param_names)
            sub_samples.append(sub)
        samples = merge_nested_samples(sub_samples)
    else:
        chain_root = find_chain_root(run_dir)
        samples = read_chains_at(chain_root)
        if param_names:
            samples = label_chain_samples(samples, param_names)
    return samples
