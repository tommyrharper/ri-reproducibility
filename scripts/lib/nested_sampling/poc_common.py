#!/usr/bin/env python3
"""Algorithm-agnostic nested-sampling PoC helpers."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


METRIC_NAMES = (
    "snr",
    "log_snr",
    "off_source_rms_jy",
    "peak_jy_per_beam",
    "relative_l2_error",
    "peak_flux_abs_error_jy",
    "wall_seconds",
    "peak_memory_bytes",
)

FAILURE_OBJECTIVE = 100.0

DEFAULT_WSCLEAN_NITER = 100
DEFAULT_WSCLEAN_AUTO_THRESHOLD = 3.0

PARAMETER_SPACE = [
    {"name": "log10_dynamic_range", "min": 2.0, "max": 3.0},
    {"name": "observation_minutes", "min": 4.0, "max": 10.0},
    {"name": "channel_count", "min": 2, "max": 6, "kind": "integer"},
    {"name": "start_frequency_hz", "min": 1.0e9, "max": 1.1e9},
    {"name": "channel_width_hz", "min": 0.5e6, "max": 2.0e6},
]


def write_polychord_paramnames(base_dir: Path, file_root: str) -> Path:
    """Write PolyChord's <file_root>.paramnames beside future chain output."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{file_root}.paramnames"
    with path.open("w") as handle:
        for spec in PARAMETER_SPACE:
            name = spec["name"]
            handle.write(f"{name}   {name}\n")
    return path


@dataclass
class DockerRunResult:
    returncode: int
    wall_seconds: float
    peak_memory_bytes: int


def r2d2_thread_count() -> int:
    override = os.environ.get("R2D2_OMP_THREADS")
    if override:
        return max(1, int(override))
    return os.cpu_count() or 1


def r2d2_docker_thread_env_flags() -> list[str]:
    threads = str(r2d2_thread_count())
    return [
        "-e",
        f"OMP_NUM_THREADS={threads}",
        "-e",
        f"MKL_NUM_THREADS={threads}",
        "-e",
        f"OPENBLAS_NUM_THREADS={threads}",
    ]


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
    return hashlib_sha256(json.dumps(params, sort_keys=True))


def hashlib_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


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


def run_docker_monitored(cmd: list[str], container_name: str, stdout_path: Path, stderr_path: Path) -> DockerRunResult:
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


def _math_namespace() -> dict[str, Any]:
    return {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}


def resolve_metric(metric_spec: str) -> tuple[Callable[[dict[str, float]], float], str]:
    if metric_spec == "badness":
        return badness_from_metrics, (
            "PolyChord log-likelihood is the composite badness score; higher means worse reconstruction."
        )

    if metric_spec in METRIC_NAMES:
        key = metric_spec

        def raw_metric(metrics: dict[str, float]) -> float:
            return float(metrics[key])

        return raw_metric, (
            f"PolyChord log-likelihood is the raw metric `{key}` with no sign flip; "
            "higher returned values are preferred by PolyChord."
        )

    try:
        code = compile(metric_spec, "<metric>", "eval")
    except SyntaxError as exc:
        raise SystemExit(f"invalid --metric expression: {exc}") from exc

    globals_ns = _math_namespace()
    globals_ns["__builtins__"] = {}
    probe_metrics = {name: 1.0 for name in METRIC_NAMES}
    try:
        eval(code, globals_ns, probe_metrics)
    except Exception as exc:
        raise SystemExit(f"invalid --metric expression: {exc}") from exc

    def expression_metric(metrics: dict[str, float]) -> float:
        return float(eval(code, globals_ns, metrics))

    return expression_metric, (
        f"PolyChord log-likelihood is the expression `{metric_spec}` with no sign flip; "
        "higher returned values are preferred by PolyChord."
    )


def simulate_measurement_set(
    params: dict[str, Any],
    eval_dir: Path,
    meqtrees_image: str,
    platform: str,
) -> tuple[Path, list[str], subprocess.CalledProcessError | None]:
    eval_dir.mkdir(parents=True, exist_ok=False)
    ms_path = eval_dir / "sim.ms"
    sim_stdout = eval_dir / "simulate.stdout.log"
    sim_stderr = eval_dir / "simulate.stderr.log"
    sim_cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        platform,
        "-v",
        f"{eval_dir}:/work",
        meqtrees_image,
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
        return ms_path, sim_cmd, exc
    return ms_path, sim_cmd, None


def cube_like_from_theta(theta: np.ndarray) -> np.ndarray:
    cube_like = np.zeros(len(PARAMETER_SPACE), dtype=np.float64)
    for i, spec in enumerate(PARAMETER_SPACE):
        lower = float(spec["min"])
        upper = float(spec["max"])
        cube_like[i] = (float(theta[i]) - lower) / (upper - lower)
        cube_like[i] = min(1.0, max(0.0, cube_like[i]))
    return cube_like


def prior_vector(cube: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [params[spec["name"]] if spec["name"] in params else math.log10(params["dynamic_range"]) for spec in PARAMETER_SPACE],
        dtype=np.float64,
    )


def self_check_r2d2_thread_env() -> None:
    saved = os.environ.get("R2D2_OMP_THREADS")
    try:
        os.environ["R2D2_OMP_THREADS"] = "6"
        flags = r2d2_docker_thread_env_flags()
        assert flags == [
            "-e",
            "OMP_NUM_THREADS=6",
            "-e",
            "MKL_NUM_THREADS=6",
            "-e",
            "OPENBLAS_NUM_THREADS=6",
        ]
        del os.environ["R2D2_OMP_THREADS"]
        count = r2d2_thread_count()
        assert count >= 1
        auto_flags = r2d2_docker_thread_env_flags()
        assert auto_flags[1] == f"OMP_NUM_THREADS={count}"
    finally:
        if saved is None:
            os.environ.pop("R2D2_OMP_THREADS", None)
        else:
            os.environ["R2D2_OMP_THREADS"] = saved


def self_check_metric_resolution() -> None:
    sample = {name: float(index + 1) for index, name in enumerate(METRIC_NAMES)}
    sample["log_snr"] = 2.5
    sample["relative_l2_error"] = 0.5
    sample["wall_seconds"] = 120.0
    sample["peak_memory_bytes"] = 1024.0**3

    badness_fn, _ = resolve_metric("badness")
    assert badness_fn(sample) == badness_from_metrics(sample)

    snr_fn, _ = resolve_metric("snr")
    assert snr_fn(sample) == sample["snr"]

    expr_fn, _ = resolve_metric("log_snr + 0.1 * wall_seconds")
    assert expr_fn(sample) == sample["log_snr"] + 0.1 * sample["wall_seconds"]

    for invalid in ("not_a_metric", "snr + unknown", "snr ++", "__import__('os').system('id')"):
        try:
            resolve_metric(invalid)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected SystemExit for invalid metric {invalid!r}")
