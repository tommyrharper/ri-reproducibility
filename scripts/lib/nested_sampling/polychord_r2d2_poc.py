#!/usr/bin/env python3
"""Cheap PolyChord search over R2D2 failure-prone point-source runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings

from poc_common import (
    FAILURE_OBJECTIVE,
    PARAMETER_SPACE,
    cube_like_from_theta,
    cube_to_params,
    compute_image_metrics,
    params_key,
    prior_vector,
    resolve_metric,
    run_checked,
    run_docker_monitored,
    self_check_metric_resolution,
    simulate_measurement_set,
    stable_seed,
)

DEFAULT_R2D2_IM_DIM = 128
DEFAULT_R2D2_NUM_ITER = 25
DEFAULT_R2D2_NUM_CHANS = 64
DEFAULT_R2D2_ARCHITECTURE = "unet"
DEFAULT_R2D2_SUPER_RESOLUTION = 1.52
DEFAULT_R2D2_CKPT_REALISATIONS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", os.getcwd()))
    parser.add_argument("--meqtrees-image", default=os.environ.get("MEQTREES_IMAGE", "ri-reproducibility/meqtrees:kern-10"))
    parser.add_argument("--r2d2-image", default=os.environ.get("R2D2_IMAGE", "ri-reproducibility/r2d2:cpu"))
    parser.add_argument("--checkpoints-dir", default=os.environ.get("CHECKPOINTS_DIR", ""))
    parser.add_argument("--nlive", type=int, default=8)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--max-ndead", type=int, default=12)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--metric", default="off_source_rms_jy", help="Objective metric: badness, a raw metric name, or an expression over metric names")
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    return parser.parse_args()


def write_r2d2_config(config_path: Path, data_file: str, output_path: str) -> None:
    lines = [
        f"data_file: {data_file}",
        f"output_path: {output_path}",
        "save_all_outputs: False",
        "nufft_pkg: finufft",
        "meas_op_on_gpu: False",
        "meas_dtype: double",
        f"im_dim_x: {DEFAULT_R2D2_IM_DIM}",
        f"im_dim_y: {DEFAULT_R2D2_IM_DIM}",
        "data_weighting: True",
        "natural_weight: True",
        "weight_type: briggs",
        f"num_iter: {DEFAULT_R2D2_NUM_ITER}",
        f"num_chans: {DEFAULT_R2D2_NUM_CHANS}",
        "series: R2D2",
        "layers: 1",
        f"architecture: {DEFAULT_R2D2_ARCHITECTURE}",
        "prune: True",
        "sigma_res_tol: 1e-4",
        "ckpt_path: /checkpoints/R2D2_A1",
        f"ckpt_realisations: {DEFAULT_R2D2_CKPT_REALISATIONS}",
        "",
    ]
    config_path.write_text("\n".join(lines))


def evaluate(
    params: dict[str, Any],
    args: argparse.Namespace,
    eval_dir: Path,
    eval_id: int,
    objective_from_metrics: Callable[[dict[str, float]], float],
) -> dict[str, Any]:
    ms_path, sim_cmd, sim_error = simulate_measurement_set(params, eval_dir, args.meqtrees_image, args.platform)
    if sim_error is not None:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"simulation failed with exit {sim_error.returncode}",
            "paths": {"eval_dir": str(eval_dir)},
        }

    mat_path = eval_dir / "r2d2_data.mat"
    convert_stdout = eval_dir / "convert.stdout.log"
    convert_stderr = eval_dir / "convert.stderr.log"
    convert_cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        args.platform,
        "-v",
        f"{eval_dir}:/work",
        "--entrypoint",
        "python3",
        args.meqtrees_image,
        "/opt/ri-nested-sampling/ms_to_r2d2_mat.py",
        "--ms-path",
        "/work/sim.ms",
        "--mat-path",
        "/work/r2d2_data.mat",
    ]
    try:
        run_checked(convert_cmd, convert_stdout, convert_stderr)
    except subprocess.CalledProcessError as exc:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"ms_to_r2d2_mat failed with exit {exc.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path)},
        }

    r2d2_dir = eval_dir / "r2d2"
    r2d2_dir.mkdir()
    config_path = eval_dir / "r2d2_config.yaml"
    write_r2d2_config(config_path, "/work/r2d2_data.mat", "/work/r2d2")

    container_name = f"ri-ns-r2d2-{uuid.uuid4().hex[:12]}"
    r2d2_stdout = eval_dir / "r2d2.stdout.log"
    r2d2_stderr = eval_dir / "r2d2.stderr.log"
    r2d2_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--platform",
        args.platform,
        "-v",
        f"{eval_dir}:/work",
        "-v",
        f"{args.checkpoints_dir}:/checkpoints:ro",
        args.r2d2_image,
        "./src/imager.py",
        "--config",
        "/work/r2d2_config.yaml",
        "--ckpt_path",
        "/checkpoints/R2D2_A1",
    ]
    run_result = run_docker_monitored(r2d2_cmd, container_name, r2d2_stdout, r2d2_stderr)
    peak_memory_bytes = run_result.peak_memory_bytes
    if run_result.returncode != 0:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"r2d2 failed with exit {run_result.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path), "mat": str(mat_path)},
            "wall_seconds": run_result.wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        }

    image_path = r2d2_dir / "R2D2_model_image.fits"
    try:
        metrics = compute_image_metrics(image_path, params["source_flux_jy"], run_result.wall_seconds, peak_memory_bytes)
        objective = objective_from_metrics(metrics)
    except Exception as exc:
        return {
            "eval_id": eval_id,
            "params": params,
            "objective": FAILURE_OBJECTIVE,
            "error": f"metric computation failed: {exc}",
            "paths": {
                "eval_dir": str(eval_dir),
                "measurement_set": str(ms_path),
                "mat": str(mat_path),
                "image": str(image_path),
            },
        }

    record = {
        "eval_id": eval_id,
        "params": params,
        "metrics": metrics,
        "objective": objective,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(eval_dir / "simulation.json"),
            "mat": str(mat_path),
            "image": str(image_path),
        },
        "commands": {
            "simulate": sim_cmd,
            "ms_to_r2d2_mat": convert_cmd,
            "r2d2": r2d2_cmd,
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    args = parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    if not args.checkpoints_dir:
        args.checkpoints_dir = str(Path(args.repo_root) / "checkpoints")
    args.checkpoints_dir = str(Path(args.checkpoints_dir).resolve())

    objective_from_metrics, likelihood_framing = resolve_metric(args.metric)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    (output_dir / "parameter-space.json").write_text(json.dumps(PARAMETER_SPACE, indent=2) + "\n")

    cache: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube)
        return prior_vector(cube, params)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
        params = cube_to_params(cube_like_from_theta(theta))
        key = params_key(params)
        params["noise_seed"] = stable_seed(args.seed, key)
        key = params_key(params)
        if key not in cache:
            eval_id = len(evaluations) + 1
            eval_dir = evaluations_dir / f"eval-{eval_id:04d}-{key}"
            record = evaluate(params, args, eval_dir, eval_id, objective_from_metrics)
            cache[key] = record
            evaluations.append(record)
            print(json.dumps({"eval_id": eval_id, "objective": record["objective"], "params": params}), flush=True)
        return float(cache[key]["objective"]), []

    settings = PolyChordSettings(len(PARAMETER_SPACE), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "r2d2_vlaa_poc"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.read_resume = False
    settings.write_resume = False
    settings.feedback = 1

    pypolychord.run_polychord(likelihood, len(PARAMETER_SPACE), 0, settings, prior)

    best = max(evaluations, key=lambda item: item["objective"]) if evaluations else None
    summary = {
        "algorithm": "r2d2",
        "vla_config": "VLA.A",
        "run_type": "cheap infrastructure PoC",
        "metric": args.metric,
        "likelihood_framing": likelihood_framing,
        "polychord": {
            "nlive": args.nlive,
            "num_repeats": args.num_repeats,
            "max_ndead": args.max_ndead,
            "seed": args.seed,
        },
        "r2d2_fixed_hyperparameters": {
            "im_dim_x": DEFAULT_R2D2_IM_DIM,
            "im_dim_y": DEFAULT_R2D2_IM_DIM,
            "num_iter": DEFAULT_R2D2_NUM_ITER,
            "num_chans": DEFAULT_R2D2_NUM_CHANS,
            "architecture": DEFAULT_R2D2_ARCHITECTURE,
            "super_resolution": DEFAULT_R2D2_SUPER_RESOLUTION,
            "ckpt_path": "/checkpoints/R2D2_A1",
            "ckpt_realisations": DEFAULT_R2D2_CKPT_REALISATIONS,
        },
        "parameter_space": PARAMETER_SPACE,
        "evaluations": evaluations,
        "worst_evaluation": best,
    }
    summary_path = output_dir / "poc-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    if os.environ.get("POLYCHORD_R2D2_POC_SELF_CHECK") == "1":
        self_check_metric_resolution()
        print("metric resolution self-check passed")
    else:
        main()
