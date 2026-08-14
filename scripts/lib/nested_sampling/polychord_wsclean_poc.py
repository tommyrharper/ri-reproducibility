#!/usr/bin/env python3
"""Cheap PolyChord search over WSClean failure-prone point-source runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pypolychord
from astropy.io import fits
from pypolychord.settings import PolyChordSettings


PARAMETER_SPACE = [
    {"name": "log10_dynamic_range", "min": 2.0, "max": 3.0},
    {"name": "observation_minutes", "min": 4.0, "max": 10.0},
    {"name": "channel_count", "min": 2, "max": 6, "kind": "integer"},
    {"name": "start_frequency_hz", "min": 1.0e9, "max": 1.1e9},
    {"name": "channel_width_hz", "min": 0.5e6, "max": 2.0e6},
    {"name": "wsclean_niter", "min": 25, "max": 150, "kind": "integer"},
    {"name": "wsclean_auto_threshold", "min": 1.5, "max": 5.0},
]


@dataclass
class DockerRunResult:
    returncode: int
    wall_seconds: float
    peak_memory_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", os.getcwd()))
    parser.add_argument("--meqtrees-image", default=os.environ.get("MEQTREES_IMAGE", "ri-reproducibility/meqtrees:kern-10"))
    parser.add_argument("--wsclean-image", default=os.environ.get("WSCLEAN_IMAGE", "ri-reproducibility/wsclean:v3.7"))
    parser.add_argument("--nlive", type=int, default=8)
    parser.add_argument("--num-repeats", type=int, default=2)
    parser.add_argument("--max-ndead", type=int, default=12)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--platform", default=os.environ.get("DOCKER_DEFAULT_PLATFORM", "linux/arm64"))
    return parser.parse_args()


def scale(cube_value: float, lower: float, upper: float) -> float:
    return lower + cube_value * (upper - lower)


def cube_to_params(cube: np.ndarray) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    for i, spec in enumerate(PARAMETER_SPACE):
        value = scale(float(cube[i]), float(spec["min"]), float(spec["max"]))
        if spec.get("kind") == "integer":
            value = int(round(value))
        raw[spec["name"]] = value
    raw["dynamic_range"] = 10.0 ** raw.pop("log10_dynamic_range")
    raw["vla_config"] = "VLA.A"
    raw["source_flux_jy"] = 1.0
    raw["source_l_arcsec"] = 0.0
    raw["source_m_arcsec"] = 0.0
    return raw


def params_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def stable_seed(global_seed: int, key: str) -> int:
    return (global_seed + int(key[:8], 16)) % (2**31 - 1)


def run_checked(cmd: list[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


MEM_RE = re.compile(r"(?P<value>[0-9.]+)\s*(?P<unit>[KMGT]?i?B)")


def memory_to_bytes(text: str) -> int:
    first = text.split("/", 1)[0].strip()
    match = MEM_RE.search(first)
    if not match:
        return 0
    value = float(match.group("value"))
    unit = match.group("unit")
    factors = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    return int(value * factors.get(unit, 1))


def run_wsclean_monitored(cmd: list[str], container_name: str, stdout_path: Path, stderr_path: Path) -> DockerRunResult:
    started = time.perf_counter()
    peak_memory = 0
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        while proc.poll() is None:
            stats = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if stats.returncode == 0 and stats.stdout.strip():
                peak_memory = max(peak_memory, memory_to_bytes(stats.stdout.strip()))
            time.sleep(0.2)
        returncode = proc.wait()
    wall = time.perf_counter() - started
    return DockerRunResult(returncode=returncode, wall_seconds=wall, peak_memory_bytes=peak_memory)


def compute_image_metrics(image_path: Path, source_flux_jy: float, wall_seconds: float, peak_memory_bytes: int) -> dict[str, float]:
    data, header = fits.getdata(image_path, header=True)
    image = np.squeeze(np.asarray(data, dtype=np.float64))
    if image.ndim != 2:
        raise ValueError(f"{image_path} is not 2-D after squeezing; shape={image.shape}")

    y_size, x_size = image.shape
    cx = int(round(float(header.get("CRPIX1", x_size / 2.0)) - 1.0))
    cy = int(round(float(header.get("CRPIX2", y_size / 2.0)) - 1.0))
    cx = max(0, min(x_size - 1, cx))
    cy = max(0, min(y_size - 1, cy))

    truth = np.zeros_like(image)
    truth[cy, cx] = source_flux_jy
    residual = image - truth

    yy, xx = np.ogrid[:y_size, :x_size]
    off_source = (yy - cy) ** 2 + (xx - cx) ** 2 > 25
    off_values = image[off_source]
    rms = float(np.sqrt(np.nanmean(off_values * off_values))) if off_values.size else 0.0
    peak = float(np.nanmax(np.abs(image)))
    snr = peak / rms if rms > 0 else float("inf")
    log_snr = math.log10(snr) if math.isfinite(snr) and snr > 0 else 99.0
    relative_l2_error = float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12))
    peak_flux_error = abs(float(image[cy, cx]) - source_flux_jy)

    return {
        "snr": float(snr),
        "log_snr": float(log_snr),
        "off_source_rms_jy": rms,
        "peak_jy_per_beam": peak,
        "relative_l2_error": relative_l2_error,
        "peak_flux_abs_error_jy": peak_flux_error,
        "wall_seconds": float(wall_seconds),
        "peak_memory_bytes": float(peak_memory_bytes),
    }


def read_gnu_time_peak_memory(time_path: Path) -> int:
    if not time_path.is_file():
        return 0
    for line in time_path.read_text(errors="replace").splitlines():
        if line.strip().startswith("Maximum resident set size (kbytes):"):
            _, value = line.rsplit(":", 1)
            return int(value.strip()) * 1024
    return 0


def badness_from_metrics(metrics: dict[str, float]) -> float:
    log_snr_loss = max(0.0, 3.0 - metrics["log_snr"])
    fidelity_loss = min(metrics["relative_l2_error"], 10.0)
    time_loss = min(metrics["wall_seconds"] / 60.0, 5.0)
    memory_loss = min(metrics["peak_memory_bytes"] / (2.0 * 1024.0 * 1024.0 * 1024.0), 5.0)
    return float(log_snr_loss + fidelity_loss + 0.05 * time_loss + 0.02 * memory_loss)


def evaluate(params: dict[str, Any], args: argparse.Namespace, eval_dir: Path, eval_id: int) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=False)
    ms_path = eval_dir / "sim.ms"
    sim_stdout = eval_dir / "simulate.stdout.log"
    sim_stderr = eval_dir / "simulate.stderr.log"
    sim_cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        args.platform,
        "-v",
        f"{eval_dir}:/work",
        args.meqtrees_image,
        "--output-ms",
        "/work/sim.ms",
        "--metadata-json",
        "/work/simulation.json",
        "--vla-config",
        params["vla_config"],
        "--observation-minutes",
        str(params["observation_minutes"]),
        "--channel-count",
        str(params["channel_count"]),
        "--start-frequency-hz",
        str(params["start_frequency_hz"]),
        "--channel-width-hz",
        str(params["channel_width_hz"]),
        "--source-flux-jy",
        str(params["source_flux_jy"]),
        "--source-l-arcsec",
        str(params["source_l_arcsec"]),
        "--source-m-arcsec",
        str(params["source_m_arcsec"]),
        "--dynamic-range",
        str(params["dynamic_range"]),
        "--seed",
        str(params["noise_seed"]),
    ]

    try:
        run_checked(sim_cmd, sim_stdout, sim_stderr)
    except subprocess.CalledProcessError as exc:
        return {
            "eval_id": eval_id,
            "params": params,
            "badness": 100.0,
            "error": f"simulation failed with exit {exc.returncode}",
            "paths": {"eval_dir": str(eval_dir)},
        }

    wsclean_dir = eval_dir / "wsclean"
    wsclean_dir.mkdir()
    container_name = f"ri-ns-wsclean-{uuid.uuid4().hex[:12]}"
    wsclean_stdout = eval_dir / "wsclean.stdout.log"
    wsclean_stderr = eval_dir / "wsclean.stderr.log"
    wsclean_time = wsclean_dir / "time.txt"
    wsclean_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--platform",
        args.platform,
        "-v",
        f"{eval_dir}:/work",
        "--entrypoint",
        "/usr/bin/time",
        args.wsclean_image,
        "-v",
        "-o",
        "/work/wsclean/time.txt",
        "wsclean",
        "-name",
        "/work/wsclean/recon",
        "-temp-dir",
        "/work/wsclean",
        "-size",
        "128",
        "128",
        "-scale",
        "1asec",
        "-niter",
        str(params["wsclean_niter"]),
        "-mgain",
        "0.8",
        "-auto-threshold",
        f"{params['wsclean_auto_threshold']:.6f}",
        "-weight",
        "natural",
        "-pol",
        "I",
        "-j",
        "1",
        "-no-update-model-required",
        "/work/sim.ms",
    ]
    run_result = run_wsclean_monitored(wsclean_cmd, container_name, wsclean_stdout, wsclean_stderr)
    peak_memory_bytes = max(run_result.peak_memory_bytes, read_gnu_time_peak_memory(wsclean_time))
    if run_result.returncode != 0:
        return {
            "eval_id": eval_id,
            "params": params,
            "badness": 100.0,
            "error": f"wsclean failed with exit {run_result.returncode}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path)},
            "wall_seconds": run_result.wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        }

    image_path = wsclean_dir / "recon-image.fits"
    try:
        metrics = compute_image_metrics(image_path, params["source_flux_jy"], run_result.wall_seconds, peak_memory_bytes)
        badness = badness_from_metrics(metrics)
    except Exception as exc:
        return {
            "eval_id": eval_id,
            "params": params,
            "badness": 100.0,
            "error": f"metric computation failed: {exc}",
            "paths": {"eval_dir": str(eval_dir), "measurement_set": str(ms_path), "image": str(image_path)},
        }

    record = {
        "eval_id": eval_id,
        "params": params,
        "metrics": metrics,
        "badness": badness,
        "paths": {
            "eval_dir": str(eval_dir),
            "measurement_set": str(ms_path),
            "simulation_metadata": str(eval_dir / "simulation.json"),
            "image": str(image_path),
            "residual": str(wsclean_dir / "recon-residual.fits"),
            "time": str(wsclean_time),
        },
        "commands": {
            "simulate": sim_cmd,
            "wsclean": wsclean_cmd,
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    (output_dir / "parameter-space.json").write_text(json.dumps(PARAMETER_SPACE, indent=2) + "\n")

    cache: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []

    def prior(cube: np.ndarray) -> np.ndarray:
        params = cube_to_params(cube)
        return np.asarray([params[spec["name"]] if spec["name"] in params else math.log10(params["dynamic_range"]) for spec in PARAMETER_SPACE], dtype=np.float64)

    def likelihood(theta: np.ndarray) -> tuple[float, list[float]]:
        # ponytail: theta values are rounded back to the documented parameter
        # space here; a later science run should keep integer/discrete handling
        # in one sampler-aware transform instead of this PoC bridge.
        cube_like = np.zeros(len(PARAMETER_SPACE), dtype=np.float64)
        for i, spec in enumerate(PARAMETER_SPACE):
            lower = float(spec["min"])
            upper = float(spec["max"])
            cube_like[i] = (float(theta[i]) - lower) / (upper - lower)
            cube_like[i] = min(1.0, max(0.0, cube_like[i]))
        params = cube_to_params(cube_like)
        key = params_key(params)
        params["noise_seed"] = stable_seed(args.seed, key)
        key = params_key(params)
        if key not in cache:
            eval_id = len(evaluations) + 1
            eval_dir = evaluations_dir / f"eval-{eval_id:04d}-{key}"
            record = evaluate(params, args, eval_dir, eval_id)
            cache[key] = record
            evaluations.append(record)
            print(json.dumps({"eval_id": eval_id, "badness": record["badness"], "params": params}), flush=True)
        return float(cache[key]["badness"]), []

    settings = PolyChordSettings(len(PARAMETER_SPACE), 0)
    settings.base_dir = str(output_dir / "chains")
    settings.file_root = "wsclean_vlaa_poc"
    settings.nlive = args.nlive
    settings.num_repeats = args.num_repeats
    settings.max_ndead = args.max_ndead
    settings.seed = args.seed
    settings.read_resume = False
    settings.write_resume = False
    settings.feedback = 1

    pypolychord.run_polychord(likelihood, len(PARAMETER_SPACE), 0, settings, prior)

    best = max(evaluations, key=lambda item: item["badness"]) if evaluations else None
    summary = {
        "algorithm": "wsclean",
        "vla_config": "VLA.A",
        "run_type": "cheap infrastructure PoC",
        "likelihood_framing": "PolyChord log-likelihood is the badness score; higher means worse reconstruction.",
        "polychord": {
            "nlive": args.nlive,
            "num_repeats": args.num_repeats,
            "max_ndead": args.max_ndead,
            "seed": args.seed,
        },
        "parameter_space": PARAMETER_SPACE,
        "evaluations": evaluations,
        "worst_evaluation": best,
    }
    summary_path = output_dir / "poc-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
