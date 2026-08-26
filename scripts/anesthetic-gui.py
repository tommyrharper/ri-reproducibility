#!/usr/bin/env python3
"""Launch anesthetic's nested-sampling GUI with labelled PolyChord chains.

Run on the host (needs a display), e.g.:

  uv run scripts/anesthetic-gui.py
  uv run scripts/anesthetic-gui.py results/nested-sampling-poc/wsclean-vlaa-...
  ./ri plot gui results/nested-sampling-poc/wsclean-vlaa-...

Also opens a merged run directory (poc-summary.json with merged_from),
re-merging the source runs' chains on the fly via anesthetic_io.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling-poc"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from anesthetic_io import find_chain_root, load_nested_samples  # noqa: E402

# GetDist / anesthetic axis labels (wrapped in $...$ by anesthetic).
PARAMETER_TEX_LABELS = {
    "log10_dynamic_range": r"\mathrm{log}_{10}(\rho_{DR})",
    "observation_minutes": r"t_{\mathrm{obs}}\,[\mathrm{min}]",
    "channel_count": r"n_{\mathrm{freq}}",
    "start_frequency_hz": r"\nu_{\mathrm{start}}\,[\mathrm{Hz}]",
    "channel_width_hz": r"\Delta\nu\,[\mathrm{Hz}]",
    "wsclean_niter": r"N_{\mathrm{iter}}",
    "wsclean_auto_threshold": r"\sigma_{\mathrm{thresh}}",
}

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
        default=None,
        help="Run directory, chains/ directory, or PolyChord file root. "
        "Default: most recent completed results/nested-sampling-poc/*/ ",
    )
    return parser.parse_args()


def is_completed_run(run_dir: Path) -> bool:
    """Completed: poc-summary.json + chains/, or a merged run (poc-summary.json only)."""
    if not run_dir.is_dir():
        return False
    summary_path = run_dir / "poc-summary.json"
    if not summary_path.is_file():
        return False
    if (run_dir / "chains").is_dir():
        return True
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return False
    return bool(summary.get("merged_from"))


def latest_run_dir() -> Path:
    runs = [p for p in NESTED_SAMPLING_DIR.glob("*") if is_completed_run(p)]
    if not runs:
        raise SystemExit(f"No completed nested-sampling runs found under {NESTED_SAMPLING_DIR}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def resolve_target(target: Path) -> Path:
    target = target.expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()
    return target


def merged_run_dir(target: Path) -> Path | None:
    """target itself if it's a merged run directory, else None."""
    summary_path = target / "poc-summary.json"
    if not target.is_dir() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return None
    return target if summary.get("merged_from") else None


def run_dir_for_chain_root(chain_root: Path) -> Path:
    return chain_root.parent.parent if chain_root.parent.name == "chains" else chain_root.parent


def load_parameter_space(run_dir: Path) -> list[dict[str, Any]]:
    summary = run_dir / "poc-summary.json"
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


def searched_param_names(parameter_space: list[dict[str, Any]]) -> list[str]:
    """Fourier / search params only — never logL, logL_birth, nlive."""
    return [str(spec["name"]) for spec in parameter_space if "name" in spec]


def write_paramnames(chain_root: Path, parameter_space: list[dict[str, Any]]) -> Path:
    path = chain_root.parent / f"{chain_root.name}.paramnames"
    with path.open("w") as handle:
        for spec in parameter_space:
            name = str(spec["name"])
            tex = PARAMETER_TEX_LABELS.get(name, name)
            handle.write(f"{name}   {tex}\n")
    return path


def run_title(run_dir: Path, fallback_label: str) -> str:
    """Human-readable window title for the anesthetic GUI."""
    parts: list[str] = [run_dir.name]
    summary_path = run_dir / "poc-summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text())
        algorithm = data.get("algorithm")
        metric = data.get("metric")
        vla = data.get("vla_config")
        meta = [str(x) for x in (algorithm, vla, metric) if x]
        if data.get("merged_from"):
            meta.append("merged")
        if meta:
            parts.append(" · ".join(meta))
    else:
        parts.append(fallback_label)
    return " — ".join(parts)


def main() -> None:
    args = parse_args()
    target = resolve_target(Path(args.target)) if args.target else latest_run_dir()

    run_dir = merged_run_dir(target)
    chain_root = None
    if run_dir is None:
        chain_root = find_chain_root(target)
        run_dir = run_dir_for_chain_root(chain_root)

    space = load_parameter_space(run_dir)
    params = searched_param_names(space)
    title = run_title(run_dir, chain_root.name if chain_root else run_dir.name)

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
