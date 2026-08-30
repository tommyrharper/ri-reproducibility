#!/usr/bin/env python3
"""Launch anesthetic's GUI for a labelled PolyChord chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from anesthetic_io import PARAMETER_TEX_LABELS, find_chain_root, load_nested_samples  # noqa: E402

FALLBACK_PARAMETER_SPACE = [
    {"name": "log10_dynamic_range"},
    {"name": "observation_minutes"},
    {"name": "channel_count"},
    {"name": "start_frequency_hz"},
    {"name": "channel_width_hz"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        help="Run directory, run name, chains/ directory, or PolyChord file root. "
        "Default: most recent completed results/nested-sampling/*/ ",
    )
    return parser.parse_args()


def is_completed_run(run_dir: Path) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False
    if (run_dir / "chains").is_dir():
        return True
    try:
        return bool(json.loads(summary_path.read_text()).get("merged_from"))
    except (OSError, ValueError):
        return False


def latest_run_dir() -> Path:
    runs = [p for p in NESTED_SAMPLING_DIR.glob("*") if is_completed_run(p)]
    if not runs:
        raise SystemExit(f"No completed nested-sampling runs found under {NESTED_SAMPLING_DIR}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def resolve_target(target: Path) -> Path:
    target = target.expanduser()
    if not target.exists() and (NESTED_SAMPLING_DIR / target).is_dir():
        return NESTED_SAMPLING_DIR / target
    return target.resolve()


def load_parameter_space(run_dir: Path) -> list[dict[str, Any]]:
    summary = run_dir / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text())
        space = data.get("parameter_space")
        if isinstance(space, list) and space:
            return space
    space_path = run_dir / "parameter-space.json"
    if space_path.is_file():
        space = json.loads(space_path.read_text())
        if isinstance(space, list) and space:
            return space
    return list(FALLBACK_PARAMETER_SPACE)


def write_paramnames(chain_root: Path, parameter_space: list[dict[str, Any]]) -> Path:
    path = chain_root.parent / f"{chain_root.name}.paramnames"
    path.write_text("".join(
        f"{name}   {PARAMETER_TEX_LABELS.get(name, name)}\n"
        for name in (str(spec["name"]) for spec in parameter_space)
    ))
    return path


def main() -> None:
    args = parse_args()
    target = resolve_target(Path(args.target)) if args.target else latest_run_dir()

    summary_path = target / "summary.json"
    run_dir = None
    if target.is_dir() and summary_path.is_file():
        try:
            if json.loads(summary_path.read_text()).get("merged_from"):
                run_dir = target
        except (OSError, ValueError):
            pass
    chain_root = None
    if run_dir is None:
        chain_root = find_chain_root(target)
        run_dir = chain_root.parent.parent if chain_root.parent.name == "chains" else chain_root.parent

    space = load_parameter_space(run_dir)
    params = [str(spec["name"]) for spec in space if "name" in spec]
    parts = [run_dir.name]
    if summary_path.is_file():
        data = json.loads(summary_path.read_text())
        meta = [str(data.get(key)) for key in ("algorithm", "vla_config", "metric") if data.get(key)]
        if data.get("merged_from"):
            meta.append("merged")
        if meta:
            parts.append(" · ".join(meta))
    else:
        parts.append(chain_root.name if chain_root else run_dir.name)
    title = " — ".join(parts)

    if chain_root is not None:
        paramnames = write_paramnames(chain_root, space)
        print(f"chain root: {chain_root}")
        print(f"paramnames: {paramnames}")
    else:
        print(f"merged run dir: {run_dir}")
    print(f"gui params: {params}")
    print(f"title: {title}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "anesthetic (and matplotlib) required on the host. Install with: uv add anesthetic"
        ) from exc

    samples = load_nested_samples(run_dir)
    plotter = samples.gui(params=params)
    plotter.fig.suptitle(title, fontsize=12)
    plotter.fig.subplots_adjust(top=0.92)
    manager = getattr(plotter.fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)
    print("Opening GUI — close the window to return to the shell.", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
