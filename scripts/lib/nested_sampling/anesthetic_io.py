#!/usr/bin/env python3
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from common import PARAMETER_TEX_LABELS

REPO_ROOT = Path(__file__).resolve().parents[3]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling"


def _similar_run_dirs(name: str) -> list[str]:
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    scored = []
    for run in (p.name for p in NESTED_SAMPLING_DIR.iterdir() if p.is_dir()):
        common = 0
        for a, b in zip(name, run):
            if a != b:
                break
            common += 1
        if common >= 8 or name.rstrip("0123456789") == run or run.startswith(name[:16]):
            scored.append((common, run))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [f"results/nested-sampling/{run}" for _, run in scored[:5]]


def find_chain_root(target: Path) -> Path:
    target = target.expanduser().resolve()

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


def _resolve_source_run_dir(run_dir: Path, entry: Any) -> Path:
    repo_root = run_dir.parents[2]
    siblings_dir = run_dir.parent
    name = entry.get("name") if isinstance(entry, dict) else str(entry)
    path = entry.get("path") if isinstance(entry, dict) else None

    for base, value in ((repo_root, path), (siblings_dir, name)):
        if value and (candidate := base / value).is_dir():
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
    tex = str(tex).strip()
    if tex.startswith("$") and tex.endswith("$") and len(tex) >= 2:
        return tex
    return f"${tex}$"


def label_chain_samples(samples, param_names: list[str]):
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


def weight_by_likelihood(samples):
    import numpy as np

    logL = np.squeeze(np.asarray(samples["logL"], dtype=float)).reshape(-1)
    weights = np.zeros(len(logL), dtype=float)
    finite = np.isfinite(logL)
    if not finite.any():
        weights[:] = 1.0
    else:
        shifted = logL[finite] - np.min(logL[finite])
        if np.max(shifted) > 0:
            weights[finite] = shifted
        else:
            weights[finite] = 1.0
    return samples.set_weights(weights)


def load_nested_samples(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
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

        source_dirs = [_resolve_source_run_dir(run_dir, entry) for entry in merged_from]
    else:
        source_dirs = [run_dir]
    samples = [read_chains_at(find_chain_root(source_dir)) for source_dir in source_dirs]
    if param_names:
        samples = [label_chain_samples(sample, param_names) for sample in samples]
    return merge_nested_samples(samples) if merged_from else samples[0]
