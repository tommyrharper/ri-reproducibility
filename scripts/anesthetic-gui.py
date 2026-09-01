#!/usr/bin/env python3
"""Launch anesthetic's GUI for a labelled PolyChord chain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from anesthetic_io import (  # noqa: E402
    PARAMETER_TEX_LABELS,
    find_chain_root,
    load_nested_samples,
    snapshot_chains,
    whole_rows,
)
from live_runs import latest_live_run  # noqa: E402

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
    parser.add_argument(
        "--live",
        action="store_true",
        help="The run in progress, rather than the most recent completed one. "
        "Its chains are a snapshot taken now; close and reopen for a later one.",
    )
    args = parser.parse_args()
    if args.live and args.target:
        parser.error("--live picks the run in progress, so it takes no run argument")
    return args


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
    if args.live:
        live_run = latest_live_run()
        target = snapshot_chains(live_run)
        print(f"live run: {live_run}")
    elif args.target:
        target = resolve_target(Path(args.target))
    else:
        target = latest_run_dir()

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
    if args.live:
        # The window is the only place this is said, and a corner plot of a
        # run that is still moving looks exactly like a finished one.
        parts.append("live snapshot")
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


def self_check() -> None:
    import tempfile

    # A row caught mid-write is short, and one short row is enough for numpy to
    # refuse the file - which is the whole reason a live run is read from a
    # copy rather than in place.
    assert whole_rows(b"1 2 3\n4 5 6\n") == b"1 2 3\n4 5 6\n"
    assert whole_rows(b"1 2 3\n4 5 6\n7 8") == b"1 2 3\n4 5 6\n"
    assert whole_rows(b"1 2 3\n4 5\n6 7 8\n") == b"1 2 3\n6 7 8\n"
    assert whole_rows(b"") == b""
    assert whole_rows(b"\n") == b""

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "wsclean-vlaa-20260101T000000Z"
        (run_dir / "chains").mkdir(parents=True)
        (run_dir / "chains" / "root_dead-birth.txt").write_bytes(b"1 2\n3 4\n5")
        (run_dir / "chains" / "root.stats").write_bytes(b"log(Z) = -1\n")
        (run_dir / "parameter-space.json").write_text('[{"name": "channel_count"}]')

        snapshot = snapshot_chains(run_dir)
        assert snapshot != run_dir and tmp not in str(snapshot), snapshot
        assert (snapshot / "chains" / "root_dead-birth.txt").read_bytes() == b"1 2\n3 4\n"
        # Anything that is not a table of rows is copied as it stands.
        assert (snapshot / "chains" / "root.stats").read_bytes() == b"log(Z) = -1\n"
        assert load_parameter_space(snapshot) == [{"name": "channel_count"}]

        # The run itself is never written to: the GUI's .paramnames lands in
        # the copy, beside chains PolyChord is not appending to.
        write_paramnames(snapshot / "chains" / "root", load_parameter_space(snapshot))
        assert (snapshot / "chains" / "root.paramnames").is_file()
        assert not (run_dir / "chains" / "root.paramnames").exists()
        assert sorted(p.name for p in (run_dir / "chains").iterdir()) == [
            "root.stats", "root_dead-birth.txt"]
    print("anesthetic-gui self-check passed")


if __name__ == "__main__":
    if os.environ.get("ANESTHETIC_GUI_SELF_CHECK") == "1":
        self_check()
    else:
        main()
