#!/usr/bin/env python3
"""Merge two or more compatible nested-sampling PoC runs into one run directory.

Post-processing only: concatenates PolyChord dead points via
anesthetic.samples.merge_nested_samples and writes a new
results/nested-sampling-poc/<algorithm>-vlaa-merged-<UTC>/poc-summary.json
that points back at the source runs. Does not copy evaluations/ or chains/.

Run on the host (needs anesthetic), e.g.:

  uv run scripts/merge-nested-sampling-runs.py \\
      results/nested-sampling-poc/r2d2-vlaa-AAA \\
      results/nested-sampling-poc/r2d2-vlaa-BBB
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NESTED_SAMPLING_DIR = REPO_ROOT / "results" / "nested-sampling-poc"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

from anesthetic_io import find_chain_root, read_chains_at  # noqa: E402

MUST_MATCH_TOP_LEVEL = ("algorithm", "vla_config", "metric")
FIXED_HYPERPARAMETER_KEYS = ("r2d2_fixed_hyperparameters", "wsclean_fixed_hyperparameters")
LOGZ_NSAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run directories to merge (>= 2)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: results/nested-sampling-poc/<algorithm>-vlaa-merged-<UTC>",
    )
    return parser.parse_args()


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def relative_to_repo_root(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(REPO_ROOT))
    except ValueError:
        raise SystemExit(f"refuse: {run_dir} is outside the repo root ({REPO_ROOT})")


def load_summary(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise SystemExit(f"refuse: not a directory: {run_dir}")
    summary_path = run_dir / "poc-summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"refuse: {run_dir} is missing poc-summary.json")
    if not (run_dir / "chains").is_dir():
        raise SystemExit(f"refuse: {run_dir} is missing chains/")
    return json.loads(summary_path.read_text())


def fixed_hyperparameters_field(summary: dict[str, Any]) -> tuple[str | None, Any]:
    for key in FIXED_HYPERPARAMETER_KEYS:
        if summary.get(key) is not None:
            return key, summary[key]
    return None, None


def check_compatible(run_dirs: list[Path], summaries: list[dict[str, Any]]) -> None:
    first_dir, first = run_dirs[0], summaries[0]
    first_hp_key, first_hp = fixed_hyperparameters_field(first)
    for run_dir, summary in zip(run_dirs[1:], summaries[1:]):
        for field in MUST_MATCH_TOP_LEVEL:
            if summary.get(field) != first.get(field):
                raise SystemExit(
                    f"refuse: {run_dir.name} has {field}={summary.get(field)!r}, "
                    f"expected {first.get(field)!r} (from {first_dir.name})"
                )
        if summary.get("parameter_space") != first.get("parameter_space"):
            raise SystemExit(
                f"refuse: {run_dir.name} has a different parameter_space than {first_dir.name}"
            )
        hp_key, hp = fixed_hyperparameters_field(summary)
        if hp_key != first_hp_key or hp != first_hp:
            raise SystemExit(
                f"refuse: {run_dir.name} fixed hyperparameters ({hp_key}={hp}) "
                f"do not match {first_dir.name} ({first_hp_key}={first_hp})"
            )


def merged_polychord(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    polychords = [s.get("polychord", {}) for s in summaries]
    nlive_values = [p.get("nlive") for p in polychords]
    if any(v is None for v in nlive_values):
        raise SystemExit("refuse: a source run is missing polychord.nlive")
    result: dict[str, Any] = {"nlive": sum(nlive_values)}
    for field in ("num_repeats", "max_ndead", "seed"):
        values = [p.get(field) for p in polychords]
        unique = list(dict.fromkeys(values))
        result[field] = unique[0] if len(unique) == 1 else values
    return result


def pooled_evaluations(run_dirs: list[Path], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pooled = []
    next_id = 1
    for run_dir, summary in zip(run_dirs, summaries):
        for ev in summary.get("evaluations", []):
            new_ev = dict(ev)
            new_ev["source_eval_id"] = ev.get("eval_id")
            new_ev["source_run"] = run_dir.name
            new_ev["eval_id"] = next_id
            next_id += 1
            pooled.append(new_ev)
    return pooled


def merged_nested_samples(run_dirs: list[Path]):
    from anesthetic.samples import merge_nested_samples

    sub_samples = [read_chains_at(find_chain_root(run_dir)) for run_dir in run_dirs]
    return merge_nested_samples(sub_samples)


def main() -> None:
    args = parse_args()
    if len(args.runs) < 2:
        raise SystemExit("refuse: need at least two run directories to merge")

    run_dirs = [resolve_run_dir(raw) for raw in args.runs]
    summaries = [load_summary(run_dir) for run_dir in run_dirs]
    check_compatible(run_dirs, summaries)

    first = summaries[0]
    algorithm = first["algorithm"]
    utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out).resolve() if args.out else NESTED_SAMPLING_DIR / f"{algorithm}-vlaa-merged-{utc_stamp}"
    if out_dir.exists():
        raise SystemExit(f"refuse: output directory already exists: {out_dir}")

    samples = merged_nested_samples(run_dirs)
    logz_samples = samples.logZ(LOGZ_NSAMPLES)
    log_z = float(logz_samples.mean())
    log_z_err = float(logz_samples.std())

    evaluations = pooled_evaluations(run_dirs, summaries)
    worst_evaluation = max(evaluations, key=lambda ev: ev.get("objective", float("-inf"))) if evaluations else None

    wall_values = [s.get("total_wall_seconds") for s in summaries if s.get("total_wall_seconds") is not None]
    total_wall_seconds = sum(wall_values) if wall_values else None

    hp_key, hp = fixed_hyperparameters_field(first)

    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "vla_config": first.get("vla_config"),
        "run_type": "merged nested-sampling run",
        "metric": first.get("metric"),
        "likelihood_framing": first.get("likelihood_framing"),
        "polychord": merged_polychord(summaries),
        "parameter_space": first.get("parameter_space"),
        "merged_from": [
            {"name": run_dir.name, "path": relative_to_repo_root(run_dir)} for run_dir in run_dirs
        ],
        "evaluations": evaluations,
        "worst_evaluation": worst_evaluation,
        "total_wall_seconds": total_wall_seconds,
        "log_z": log_z,
        "log_z_err": log_z_err,
    }
    if hp_key:
        summary[hp_key] = hp

    out_dir.mkdir(parents=True)
    summary_path = out_dir / "poc-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
