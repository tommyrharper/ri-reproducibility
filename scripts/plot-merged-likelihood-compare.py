#!/usr/bin/env python3
"""Compare the latest comparable merged R2D2 and WSClean likelihoods.

Picks the newest merged R2D2 run and newest merged WSClean run that share
vla_config, metric, and parameter_space (imager hyperparameters may differ).

  uv run scripts/plot-merged-likelihood-compare.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from anesthetic_io import (  # noqa: E402
    NESTED_SAMPLING_DIR,
    load_nested_samples,
    weight_by_likelihood,
)

FALLBACK_PARAMS = [
    "log10_dynamic_range",
    "observation_minutes",
    "channel_count",
    "start_frequency_hz",
    "channel_width_hz",
]
OUT_OVERLAY = REPO_ROOT / "benchmarks" / "merged-r2d2-wsclean-likelihood.png"
OUT_SIDE = REPO_ROOT / "benchmarks" / "merged-r2d2-wsclean-likelihood-side-by-side.png"
PLOT_KW = {"kind": "kde", "ncompress": False}


def compare_key(summary: dict) -> str:
    return json.dumps(
        {
            "vla_config": summary.get("vla_config"),
            "metric": summary.get("metric"),
            "parameter_space": summary.get("parameter_space"),
        },
        sort_keys=True,
    )


def merged_runs() -> list[tuple[Path, dict]]:
    found = []
    for path in NESTED_SAMPLING_DIR.glob("*-vlaa-merged-*"):
        summary_path = path / "poc-summary.json"
        if not path.is_dir() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("merged_from") and summary.get("algorithm") in {"r2d2", "wsclean"}:
            found.append((path, summary))
    return found


def latest_comparable_pair() -> tuple[Path, dict, Path, dict]:
    """Newest R2D2 + WSClean merged dirs that share metric / VLA / prior box."""
    groups: dict[str, dict[str, list[tuple[Path, dict]]]] = defaultdict(
        lambda: {"r2d2": [], "wsclean": []}
    )
    for path, summary in merged_runs():
        groups[compare_key(summary)][summary["algorithm"]].append((path, summary))

    pairs = []
    for sides in groups.values():
        if not sides["r2d2"] or not sides["wsclean"]:
            continue
        r2d2 = max(sides["r2d2"], key=lambda item: item[0].name)
        wsclean = max(sides["wsclean"], key=lambda item: item[0].name)
        pairs.append((max(r2d2[0].name, wsclean[0].name), r2d2, wsclean))
    if not pairs:
        raise SystemExit(
            "No comparable merged R2D2 + WSClean pair "
            "(need matching vla_config, metric, and parameter_space)"
        )
    _, r2d2, wsclean = max(pairs, key=lambda item: item[0])
    return r2d2[0], r2d2[1], wsclean[0], wsclean[1]


def searched_params(summary: dict) -> list[str]:
    names = [spec["name"] for spec in (summary.get("parameter_space") or []) if "name" in spec]
    return names or list(FALLBACK_PARAMS)


def corner(samples, params, color, title):
    axes = samples.plot_2d(params, label=title, color=color, **PLOT_KW)
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
    r2d2_dir, r2d2_summary, wsclean_dir, wsclean_summary = latest_comparable_pair()
    params = searched_params(r2d2_summary)
    print(f"R2D2:    {r2d2_dir.name}")
    print(f"WSClean: {wsclean_dir.name}")
    print(f"metric:  {r2d2_summary.get('metric')}  vla: {r2d2_summary.get('vla_config')}")

    r2d2 = weight_by_likelihood(load_nested_samples(r2d2_dir))
    wsclean = weight_by_likelihood(load_nested_samples(wsclean_dir))
    r2d2.label = "R2D2"
    wsclean.label = "WSClean"
    params = [name for name in params if name in r2d2.columns and name in wsclean.columns]
    if len(params) < 2:
        raise SystemExit("Need at least two shared searched parameters to plot")

    OUT_OVERLAY.parent.mkdir(parents=True, exist_ok=True)

    axes = wsclean.plot_2d(params, label="WSClean", color="C0", **PLOT_KW)
    r2d2.plot_2d(axes, label="R2D2", color="C1", **PLOT_KW)
    fig = axes.iloc[0, 0].figure
    handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_OVERLAY}")

    img_w = fig_to_array(corner(wsclean, params, "C0", "WSClean"))
    img_r = fig_to_array(corner(r2d2, params, "C1", "R2D2"))
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
