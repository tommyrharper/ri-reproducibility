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


METRIC_NAMES = (
    "snr",
    "log_snr",
    "off_source_rms_jy",
    "total_rms_jy",
    "peak_jy_per_beam",
    "relative_l2_error",
    "peak_flux_abs_error_jy",
    "sigma_res",
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


def write_polychord_paramnames(
    base_dir: Path,
    file_root: str,
    parameter_space: list[dict[str, Any]] | None = None,
) -> Path:
    """Write PolyChord's <file_root>.paramnames beside future chain output."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{file_root}.paramnames"
    specs = parameter_space if parameter_space is not None else PARAMETER_SPACE
    with path.open("w") as handle:
        for spec in specs:
            name = str(spec["name"])
            tex = PARAMETER_TEX_LABELS.get(name, name)
            handle.write(f"{name}   {tex}\n")
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


def load_fits_2d(path: Path) -> tuple[np.ndarray, Any]:
    from astropy.io import fits

    data, header = fits.getdata(path, header=True)
    image = np.squeeze(np.asarray(data, dtype=np.float64))
    if image.ndim != 2:
        raise ValueError(f"{path} is not 2-D after squeezing; shape={image.shape}")
    return image, header


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(values * values))) if values.size else 0.0


def sigma_res(residual: np.ndarray, dirty: np.ndarray) -> float:
    """R2D2-paper data-fidelity: ||residual_dirty||_2 / ||dirty||_2."""
    return float(np.linalg.norm(residual) / max(np.linalg.norm(dirty), 1e-12))


def compute_image_metrics(
    image_path: Path,
    source_flux_jy: float,
    wall_seconds: float,
    peak_memory_bytes: int,
    dirty_path: Path | None = None,
    residual_dirty_path: Path | None = None,
) -> dict[str, float]:
    image, header = load_fits_2d(image_path)

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
    off_rms = rms(image[off_source])
    total_rms = rms(residual)
    peak = float(np.nanmax(np.abs(image)))
    snr = peak / off_rms if off_rms > 0 else float("inf")
    log_snr = math.log10(snr) if math.isfinite(snr) and snr > 0 else 99.0
    relative_l2_error = float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12))
    peak_flux_error = abs(float(image[cy, cx]) - source_flux_jy)

    metrics = {
        "snr": float(snr),
        "log_snr": float(log_snr),
        "off_source_rms_jy": off_rms,
        "total_rms_jy": total_rms,
        "peak_jy_per_beam": peak,
        "relative_l2_error": relative_l2_error,
        "peak_flux_abs_error_jy": peak_flux_error,
        "wall_seconds": float(wall_seconds),
        "peak_memory_bytes": float(peak_memory_bytes),
    }
    if dirty_path is not None and residual_dirty_path is not None:
        dirty, _ = load_fits_2d(dirty_path)
        residual_dirty, _ = load_fits_2d(residual_dirty_path)
        metrics["sigma_res"] = sigma_res(residual_dirty, dirty)
    return metrics


def read_gnu_time_peak_memory(time_path: Path) -> int:
    if not time_path.is_file():
        return 0
    for line in time_path.read_text(errors="replace").splitlines():
        if line.strip().startswith("Maximum resident set size (kbytes):"):
            _, value = line.rsplit(":", 1)
            return int(value.strip()) * 1024
    return 0


def read_gnu_time_wall_seconds(time_path: Path) -> float | None:
    """Parse GNU time -v's own elapsed wall clock, in [h:]mm:ss(.cc) form.

    This is the binary's actual run time inside the container, separate from
    docker create/start/teardown overhead measured around the whole `docker
    run` invocation.
    """
    if not time_path.is_file():
        return None
    for line in time_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("Elapsed (wall clock) time"):
            continue
        # Label itself contains colons (e.g. "(h:mm:ss or m:ss): 1:02.50"), so
        # split on the literal "): " marker instead of the last colon.
        _, _, value = line.partition("): ")
        parts = value.strip().split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return None
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + part
        return seconds
    return None


PROFILING_STAGE_FIELDS = (
    "simulate_seconds",
    "convert_seconds",
    "image_container_seconds",
    "image_binary_seconds",
    "metrics_seconds",
)


def summarize_profiling(
    evaluations: list[dict[str, Any]],
    total_wall_seconds: float,
    mpi_procs: int,
) -> dict[str, Any]:
    """Aggregate per-evaluation stage timing into a run-level breakdown.

    Sums each `timing.*` field across every evaluation that has one (failed
    evaluations that errored out before a stage ran simply omit that field).
    `image_container_overhead_seconds` is the docker round-trip minus the
    binary's own GNU-time-reported elapsed time, i.e. container create/start/
    teardown plus the `docker stats` polling loop. `polychord_overhead_seconds`
    is whatever's left of total_wall_seconds once every accounted evaluation
    stage is subtracted out - PolyChord's own sampling/bookkeeping plus (at
    mpi_procs > 1) any cross-rank idle time, so it is only a clean serial
    figure when mpi_procs == 1.
    """
    totals: dict[str, float] = {field: 0.0 for field in PROFILING_STAGE_FIELDS}
    counts: dict[str, int] = {field: 0 for field in PROFILING_STAGE_FIELDS}
    image_container_overhead = 0.0
    image_container_overhead_count = 0
    for record in evaluations:
        timing = record.get("timing") or {}
        for field in PROFILING_STAGE_FIELDS:
            value = timing.get(field)
            if value is not None:
                totals[field] += float(value)
                counts[field] += 1
        image_container_value = timing.get("image_container_seconds")
        image_binary_value = timing.get("image_binary_seconds")
        if image_container_value is not None and image_binary_value is not None:
            image_container_overhead += float(image_container_value) - float(image_binary_value)
            image_container_overhead_count += 1

    image_binary_total = totals["image_binary_seconds"]
    image_container_total = totals["image_container_seconds"]
    counts["image_container_overhead_seconds"] = image_container_overhead_count

    accounted = (
        totals["simulate_seconds"]
        + totals["convert_seconds"]
        + totals["image_container_seconds"]
        + totals["metrics_seconds"]
    )
    polychord_overhead = total_wall_seconds - accounted

    return {
        "mpi_procs": mpi_procs,
        "total_wall_seconds": total_wall_seconds,
        "stage_totals_seconds": {
            "simulate": totals["simulate_seconds"],
            "convert": totals["convert_seconds"] if counts["convert_seconds"] else None,
            "image_container": image_container_total,
            "image_binary": image_binary_total if counts["image_binary_seconds"] else None,
            "image_container_overhead": image_container_overhead if image_container_overhead_count else None,
            "metrics": totals["metrics_seconds"],
        },
        "stage_eval_counts": counts,
        "accounted_seconds": accounted,
        "polychord_overhead_seconds": polychord_overhead,
        "note": (
            "polychord_overhead_seconds is a clean serial figure only when "
            "mpi_procs == 1; at higher mpi_procs it also folds in cross-rank "
            "idle/imbalance time."
        ),
    }


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


def mpi_rank() -> int:
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        return 0


def load_evaluations_from_dir(evaluations_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(evaluations_dir.glob("eval-*/metrics.json")):
        records.append(json.loads(metrics_path.read_text()))
    return records


def write_evaluation_record(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


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
    total_fn, _ = resolve_metric("total_rms_jy")
    assert total_fn(sample) == sample["total_rms_jy"]
    sigma_fn, _ = resolve_metric("sigma_res")
    assert sigma_fn(sample) == sample["sigma_res"]
    assert abs(rms(np.array([3.0, 4.0])) - 5.0 / math.sqrt(2.0)) < 1e-12
    assert abs(sigma_res(np.array([3.0, 4.0]), np.array([0.0, 2.0])) - 2.5) < 1e-12

    expr_fn, _ = resolve_metric("log_snr + 0.1 * wall_seconds")
    assert expr_fn(sample) == sample["log_snr"] + 0.1 * sample["wall_seconds"]

    for invalid in ("not_a_metric", "snr + unknown", "snr ++", "__import__('os').system('id')"):
        try:
            resolve_metric(invalid)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected SystemExit for invalid metric {invalid!r}")


def self_check_profiling() -> None:
    import tempfile

    gnu_time_text = (
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n"
        "\tMaximum resident set size (kbytes): 4096\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        time_path = Path(tmp) / "time.txt"
        time_path.write_text(gnu_time_text)
        assert abs(read_gnu_time_wall_seconds(time_path) - 62.5) < 1e-9
        assert read_gnu_time_peak_memory(time_path) == 4096 * 1024
        assert read_gnu_time_wall_seconds(Path(tmp) / "missing.txt") is None

    evaluations = [
        {"timing": {"simulate_seconds": 1.0, "image_container_seconds": 5.0, "image_binary_seconds": 3.0, "metrics_seconds": 0.5}},
        {"timing": {"simulate_seconds": 1.0, "image_container_seconds": 5.0, "image_binary_seconds": 3.0, "metrics_seconds": 0.5}},
        {"timing": {"simulate_seconds": 1.0, "image_container_seconds": 5.0, "metrics_seconds": 0.5}},
        {"error": "simulation failed", "paths": {}},
    ]
    profiling = summarize_profiling(evaluations, total_wall_seconds=25.0, mpi_procs=1)
    assert profiling["stage_totals_seconds"]["simulate"] == 3.0
    assert profiling["stage_totals_seconds"]["image_container"] == 15.0
    assert profiling["stage_totals_seconds"]["image_binary"] == 6.0
    assert profiling["stage_totals_seconds"]["image_container_overhead"] == 4.0
    assert profiling["accounted_seconds"] == 19.5
    assert abs(profiling["polychord_overhead_seconds"] - 5.5) < 1e-9
    assert profiling["stage_eval_counts"]["simulate_seconds"] == 3
    assert profiling["stage_eval_counts"]["image_container_overhead_seconds"] == 2

    empty_profiling = summarize_profiling([], total_wall_seconds=5.0, mpi_procs=1)
    assert empty_profiling["stage_totals_seconds"]["image_binary"] is None
    assert empty_profiling["stage_totals_seconds"]["image_container_overhead"] is None
    assert empty_profiling["accounted_seconds"] == 0.0
    assert empty_profiling["polychord_overhead_seconds"] == 5.0
