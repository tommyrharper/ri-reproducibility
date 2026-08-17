#!/usr/bin/env python3
"""Launch anesthetic's nested-sampling GUI with labelled PolyChord chains.

Run on the host (needs a display), e.g.:

  uv run scripts/anesthetic-gui.py
  uv run scripts/anesthetic-gui.py results/nested-sampling-poc/wsclean-vlaa-...
  make anesthetic-gui RUN=results/nested-sampling-poc/wsclean-vlaa-...
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling-poc"

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


def latest_run_dir() -> Path:
    """Newest completed run: requires poc-summary.json and chains/."""
    runs = [
        p
        for p in NESTED_SAMPLING_DIR.glob("*")
        if p.is_dir() and (p / "chains").is_dir() and (p / "poc-summary.json").is_file()
    ]
    if not runs:
        raise SystemExit(f"No completed nested-sampling runs found under {NESTED_SAMPLING_DIR}")
    return max(runs, key=lambda p: p.stat().st_mtime)


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


def _similar_run_dirs(name: str) -> list[str]:
    """Suggest nearby run directory names when RUN= looks like a typo."""
    if not NESTED_SAMPLING_DIR.is_dir():
        return []
    runs = [p.name for p in NESTED_SAMPLING_DIR.iterdir() if p.is_dir()]
    # Prefer names that share a long common prefix with the typo.
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


@contextmanager
def hide_empty_phys_live_birth(chain_root: Path) -> Iterator[None]:
    """Aside empty *_phys_live-birth.txt so anesthetic's np.loadtxt doesn't warn."""
    path = Path(str(chain_root) + "_phys_live-birth.txt")
    aside = path.with_name(path.name + ".empty")
    moved = False
    if path.is_file() and path.stat().st_size == 0:
        path.rename(aside)
        moved = True
    try:
        yield
    finally:
        if moved and aside.is_file() and not path.exists():
            aside.rename(path)


def run_title(run_dir: Path, chain_root: Path) -> str:
    """Human-readable window title for the anesthetic GUI."""
    parts: list[str] = [run_dir.name]
    summary_path = run_dir / "poc-summary.json"
    if summary_path.is_file():
        data = json.loads(summary_path.read_text())
        algorithm = data.get("algorithm")
        metric = data.get("metric")
        vla = data.get("vla_config")
        meta = [str(x) for x in (algorithm, vla, metric) if x]
        if meta:
            parts.append(" · ".join(meta))
    else:
        parts.append(chain_root.name)
    return " — ".join(parts)


def main() -> None:
    args = parse_args()
    target = Path(args.target) if args.target else latest_run_dir()
    chain_root = find_chain_root(target)
    run_dir = run_dir_for_chain_root(chain_root)
    space = load_parameter_space(run_dir)
    params = searched_param_names(space)
    paramnames = write_paramnames(chain_root, space)
    title = run_title(run_dir, chain_root)
    print(f"chain root: {chain_root}")
    print(f"paramnames: {paramnames}")
    print(f"gui params: {params}")
    print(f"title: {title}")

    try:
        from anesthetic import read_chains
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "anesthetic (and matplotlib) required on the host. Install with: uv add anesthetic"
        ) from exc

    # Let anesthetic read <root>.paramnames so TeX labels get wrapped in $...$.
    with hide_empty_phys_live_birth(chain_root):
        samples = read_chains(str(chain_root))
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
