#!/usr/bin/env python3
"""Compare latest comparable R2D2 and WSClean likelihoods."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

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
OUT_OVERLAY = REPO_ROOT / "reports" / "merged-r2d2-wsclean-likelihood.png"
OUT_SIDE = REPO_ROOT / "reports" / "merged-r2d2-wsclean-likelihood-side-by-side.png"
# Every pair ever plotted, each under its own name, so the report can show the
# lot. The two fixed names above stay the merged pair latex/notes.tex includes.
GALLERY_DIR = REPO_ROOT / "reports" / "likelihood-comparisons"
PLOT_KW = {"kind": "kde", "ncompress": False}
# How a run was executed, not what it searched: two runs differing only in
# seed, rank count or synchronicity are still directly comparable, so the
# effort part of the key is these three and not the whole polychord block.
EFFORT_FIELDS = ("nlive", "num_repeats", "max_ndead")
# A summary carries every evaluation and runs to hundreds of MB, but the
# run-level fields all sit ahead of them, so only the head is ever read.
SUMMARY_HEAD = 1 << 16


def read_summary_head(summary_path: Path) -> dict:
    """A summary's run-level fields, without reading its evaluations."""
    with open(summary_path, encoding="utf-8", errors="replace") as f:
        head = f.read(SUMMARY_HEAD)
    # Past the start of `evaluations` a field belongs to one evaluation rather
    # than to the run, so the head stops there and is closed back up.
    cut = head.find('"evaluations"')
    text = head if cut < 0 else head[:cut].rstrip().rstrip(",") + "}"
    try:
        parsed = json.loads(text)
    except ValueError:  # caught mid-write, or a run-level value the head cut short
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compare_key(summary: dict, effort: bool = False) -> str:
    polychord = summary.get("polychord") or {}
    return json.dumps(
        {
            "vla_config": summary.get("vla_config"),
            "metric": summary.get("metric"),
            "parameter_space": summary.get("parameter_space"),
            **({field: polychord.get(field) for field in EFFORT_FIELDS} if effort else {}),
        },
        sort_keys=True,
    )


def has_chains(path: Path) -> bool:
    """Same two layouts find_chain_root accepts: a chains/ directory, or loose
    PolyChord files in the run directory itself."""
    return (path / "chains").is_dir() or any(path.glob("*_dead-birth.txt"))


def candidate_runs(merged_only: bool = True) -> list[tuple[Path, dict]]:
    found = []
    for path in NESTED_SAMPLING_DIR.glob("*-vlaa-merged-*" if merged_only else "*-vlaa-*"):
        summary_path = path / "summary.json"
        if not path.is_dir() or not summary_path.is_file() or not has_chains(path):
            continue
        summary = read_summary_head(summary_path)
        if summary.get("algorithm") not in {"r2d2", "wsclean"}:
            continue
        if merged_only and not summary.get("merged_from"):
            continue
        found.append((path, summary))
    return found


def latest_comparable_pair(merged_only: bool = True) -> tuple[Path, dict, Path, dict]:
    """The newest R2D2 and WSClean run that searched the same space the same way.

    Run directory names carry a UTC timestamp, so newest is the largest name.
    """
    groups: dict[str, dict[str, list[tuple[Path, dict]]]] = defaultdict(
        lambda: {"r2d2": [], "wsclean": []}
    )
    for path, summary in candidate_runs(merged_only):
        key = compare_key(summary, effort=not merged_only)
        groups[key][summary["algorithm"]].append((path, summary))

    pairs = []
    for sides in groups.values():
        if not sides["r2d2"] or not sides["wsclean"]:
            continue
        r2d2 = max(sides["r2d2"], key=lambda item: item[0].name)
        wsclean = max(sides["wsclean"], key=lambda item: item[0].name)
        pairs.append((max(r2d2[0].name, wsclean[0].name), r2d2, wsclean))
    if not pairs:
        raise SystemExit(
            "No comparable {}R2D2 + WSClean pair (need matching vla_config, "
            "metric and parameter_space{})".format(
                "merged " if merged_only else "",
                "" if merged_only else ", and matching nlive, num_repeats and max_ndead",
            )
        )
    _, r2d2, wsclean = max(pairs, key=lambda item: item[0])
    return r2d2[0], r2d2[1], wsclean[0], wsclean[1]


def gallery_paths(wsclean_dir: Path, r2d2_dir: Path) -> tuple[Path, Path]:
    """This pair's own names, so plotting a later pair does not overwrite it."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", f"{wsclean_dir.name}-vs-{r2d2_dir.name}")
    return GALLERY_DIR / f"{stem}.png", GALLERY_DIR / f"{stem}-side-by-side.png"


def pyplot():
    """Imported on use, so choosing the runs - and its self-check - needs only
    the standard library."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def corner(samples, params, color, title):
    axes = samples.plot_2d(params, label=title, color=color, **PLOT_KW)
    fig = axes.iloc[0, 0].figure
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def fig_to_array(fig):
    plt = pyplot()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--last", action="store_true",
        help="the last two directly comparable runs, merged or not, instead of "
             "the merged ones only. Their sampler effort must match too, and "
             "the figures latex/notes.tex includes by name are left alone.",
    )
    return parser.parse_args(argv)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


def main(argv=None) -> None:
    plt = pyplot()
    args = parse_args(argv)
    r2d2_dir, r2d2_summary, wsclean_dir, wsclean_summary = latest_comparable_pair(
        merged_only=not args.last
    )
    gallery_overlay, gallery_side = gallery_paths(wsclean_dir, r2d2_dir)
    # --last plots runs the paper does not cite, so it writes only its own
    # copies and leaves the fixed names notes.tex includes as they were.
    overlay_targets = [gallery_overlay] + ([] if args.last else [OUT_OVERLAY])
    side_targets = [gallery_side] + ([] if args.last else [OUT_SIDE])
    params = [spec["name"] for spec in (r2d2_summary.get("parameter_space") or []) if "name" in spec] or list(FALLBACK_PARAMS)
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

    axes = wsclean.plot_2d(params, label="WSClean", color="C0", **PLOT_KW)
    r2d2.plot_2d(axes, label="R2D2", color="C1", **PLOT_KW)
    fig = axes.iloc[0, 0].figure
    handles, labels = axes.iloc[-1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    for target in overlay_targets:
        save(fig, target)
    plt.close(fig)

    img_w = fig_to_array(corner(wsclean, params, "C0", "WSClean"))
    img_r = fig_to_array(corner(r2d2, params, "C1", "R2D2"))
    pair, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img in zip(axs, (img_w, img_r)):
        ax.imshow(img)
        ax.axis("off")
    pair.tight_layout(w_pad=0.4)
    for target in side_targets:
        save(pair, target)
    plt.close(pair)


def _self_check_head_and_pairing():
    import tempfile

    global NESTED_SAMPLING_DIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        def run(name, algorithm, nlive, merged=False):
            path = root / name
            (path / "chains").mkdir(parents=True)
            summary = {"algorithm": algorithm, "vla_config": "VLA.A",
                       "metric": "total_rms_jy", "parameter_space": [{"name": "a"}],
                       "polychord": {"nlive": nlive, "num_repeats": 2, "max_ndead": -1,
                                     "seed": nlive},
                       "evaluations": [{"eval_id": 1}]}
            if merged:
                summary["merged_from"] = ["x"]
            (path / "summary.json").write_text(json.dumps(summary))

        run("wsclean-vlaa-20260101T000000Z", "wsclean", 8)
        run("r2d2-vlaa-20260102T000000Z", "r2d2", 8)
        # Newer, but no WSClean run shares its effort, so it is not the pair.
        run("r2d2-vlaa-20260103T000000Z", "r2d2", 125)

        head = read_summary_head(root / "r2d2-vlaa-20260102T000000Z" / "summary.json")
        assert head["polychord"]["nlive"] == 8, head
        assert "evaluations" not in head, head

        original, NESTED_SAMPLING_DIR = NESTED_SAMPLING_DIR, root
        try:
            r2d2_dir, _, wsclean_dir, _ = latest_comparable_pair(merged_only=False)
        finally:
            NESTED_SAMPLING_DIR = original
        assert r2d2_dir.name == "r2d2-vlaa-20260102T000000Z", r2d2_dir
        assert wsclean_dir.name == "wsclean-vlaa-20260101T000000Z", wsclean_dir
        overlay, side = gallery_paths(wsclean_dir, r2d2_dir)
        assert overlay.name == (
            "wsclean-vlaa-20260101T000000Z-vs-r2d2-vlaa-20260102T000000Z.png"
        ), overlay
        assert side.name.endswith("-side-by-side.png"), side


if __name__ == "__main__":
    if "--self-check" in sys.argv[1:]:
        _self_check_head_and_pairing()
        print("ok   plot-merged-likelihood-compare self-check")
    else:
        main()
