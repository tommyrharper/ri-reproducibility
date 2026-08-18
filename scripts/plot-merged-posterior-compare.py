#!/usr/bin/env python3
"""One-off: compare merged R2D2 vs WSClean nested-sampling posteriors.

  uv run scripts/plot-merged-posterior-compare.py

Writes an overlay and a side-by-side pair of corner plots.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from anesthetic_io import NESTED_SAMPLING_DIR, load_nested_samples  # noqa: E402

PARAMS = [
    "log10_dynamic_range",
    "observation_minutes",
    "channel_count",
    "start_frequency_hz",
    "channel_width_hz",
]
OUT_OVERLAY = REPO_ROOT / "benchmarks" / "merged-r2d2-wsclean-posterior.png"
OUT_SIDE = REPO_ROOT / "benchmarks" / "merged-r2d2-wsclean-posterior-side-by-side.png"
PLOT_KW = {"kind": "kde", "ncompress": False}


def latest_merged(algorithm: str) -> Path:
    runs = sorted(
        p
        for p in NESTED_SAMPLING_DIR.glob(f"{algorithm}-vlaa-merged-*")
        if p.is_dir() and (p / "poc-summary.json").is_file()
    )
    if not runs:
        raise SystemExit(f"No merged {algorithm} run under {NESTED_SAMPLING_DIR}")
    return runs[-1]


def corner(samples, color, title):
    axes = samples.plot_2d(PARAMS, label=title, color=color, **PLOT_KW)
    fig = axes.iloc[0, 0].figure
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def main() -> None:
    r2d2_dir = latest_merged("r2d2")
    wsclean_dir = latest_merged("wsclean")
    print(f"R2D2:    {r2d2_dir.name}")
    print(f"WSClean: {wsclean_dir.name}")

    r2d2 = load_nested_samples(r2d2_dir)
    wsclean = load_nested_samples(wsclean_dir)
    r2d2.label = "R2D2"
    wsclean.label = "WSClean"

    OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)

    axes = wsclean.plot_2d(PARAMS, label="WSClean", color="C0", **PLOT_KW)
    r2d2.plot_2d(axes, label="R2D2", color="C1", **PLOT_KW)
    fig = axes.iloc[0, 0].figure
    handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_OVERLAY}")

    img_w = fig_to_array(corner(wsclean, "C0", "WSClean"))
    img_r = fig_to_array(corner(r2d2, "C1", "R2D2"))
    pair, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img in zip(axs, (img_w, img_r)):
        ax.imshow(img)
        ax.axis("off")
    pair.tight_layout(w_pad=0.4)
    pair.savefig(OUT_SIDE, dpi=150, bbox_inches="tight")
    plt.close(pair)
    print(f"wrote {OUT_SIDE}")


if __name__ == "__main__":
    main()
