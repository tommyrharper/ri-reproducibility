#!/usr/bin/env python3
from __future__ import annotations

import atexit
import json
import math
import os
import re
import select
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any


class _LazyNumpy:
    def __getattr__(self, name: str) -> Any:
        global np
        import numpy as np

        return getattr(np, name)


np: Any = _LazyNumpy()


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

# PolyChord maximizes this; infrastructure failures use WORKER_DIED (see docs/robustness.md).
FAILURE_OBJECTIVE = 100.0

# Worker died mid-request; negative keeps it distinct from real exit statuses.
WORKER_DIED = -1

# The zygote reports a signal-killed child as 128 + signal (docker/wsclean/src/
# zygote.cpp), and SIGKILL is the one signal nothing in this pipeline sends an
# imager: it is the kernel's OOM killer. A killed imager says nothing about the
# parameters it was given, so it is classified with WORKER_DIED rather than
# scored - see docs/robustness.md. A crash the imager chose (SIGSEGV, SIGABRT)
# stays scored: that is a failure mode, which is what these runs look for.
OOM_KILLED = 128 + 9


def is_infrastructure_failure(returncode: int) -> bool:
    """True when the host failed rather than the imager. Never scored."""
    return returncode in (WORKER_DIED, OOM_KILLED)


# Fresh worker per attempt; increasing delays allow transient OOM pressure to
# clear. See docs/robustness.md for retry semantics.
WORKER_RETRY_DELAYS = (0.0, 1.0, 5.0, 15.0, 30.0)


def worker_attempts() -> Any:
    for attempt, delay in enumerate(WORKER_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        yield attempt


# Reply bounds turn silent-worker deadlocks into recoverable failures. Keep
# SIMULATE_REPLY_TIMEOUT above normal simulation/convert latency and below the
# longer imaging bound; rationale and measurements live in docs/robustness.md.
SIMULATE_REPLY_TIMEOUT = 10.0
SHELL_REPLY_TIMEOUT = 300.0
IMAGING_REPLY_TIMEOUT = 3600.0


def worker_reply(stream: Any, timeout: float) -> str | None:
    if not select.select([stream], [], [], timeout)[0]:
        return None
    return stream.readline()


def worker_send(stream: Any, request: str) -> bool:
    try:
        stream.write(request)
        stream.flush()
    except OSError:
        # Close too, or the buffered request can emit a late BrokenPipeError.
        try:
            stream.close()
        except OSError:
            pass
        return False
    return True


class WorkerDied(RuntimeError):
    pass


def abort_run(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr, flush=True)
    try:
        from mpi4py import MPI

        MPI.COMM_WORLD.Abort(1)
    except Exception:
        # No MPI, or it is already too broken to abort through: fall through
        # to the hard exit, which the launcher turns into a failed run anyway.
        pass
    os._exit(1)


# CLEAN's minor-iteration budget. At the old 100 roughly four evaluations in
# five stopped on `maximum number of iterations` rather than on the threshold,
# so the objective was scoring the residual after 100 components rather than
# what CLEAN converges to; `clean_stop_reason` in each record now says which.
DEFAULT_WSCLEAN_NITER = int(os.environ.get("NS_WSCLEAN_NITER") or 1000)
DEFAULT_WSCLEAN_AUTO_THRESHOLD = 3.0

# Keep archived runs reproducible; override with `--mgain` when throughput matters.
DEFAULT_WSCLEAN_MGAIN = float(os.environ.get("NS_WSCLEAN_MGAIN") or 0.8)

# Shared image geometry; detailed rationale: docs/nested-sampling.md.
DEFAULT_IMAGE_DIM = 128
DEFAULT_SUPER_RESOLUTION = 1.5

# `source_offset_fraction` geometry: docs/parameter-space-proposal.md.
VLA_A_MAX_BASELINE_M = 36_000.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
SOURCE_OFFSET_POSITION_ANGLE_DEG = 30.0


def image_pixel_size_arcsec(
    max_proj_baseline_lambda: float,
    super_resolution: float = DEFAULT_SUPER_RESOLUTION,
) -> float:
    if not max_proj_baseline_lambda > 0.0:
        raise SystemExit(f"FATAL: non-positive max projected baseline: {max_proj_baseline_lambda!r}")
    return (180.0 / math.pi) * 3600.0 / (super_resolution * 2.0 * max_proj_baseline_lambda)


def source_offset_to_lm(fraction: float, start_frequency_hz: float) -> tuple[float, float]:
    max_proj_baseline_lambda = VLA_A_MAX_BASELINE_M * start_frequency_hz / SPEED_OF_LIGHT_M_S
    half_width_arcsec = image_pixel_size_arcsec(max_proj_baseline_lambda) * (DEFAULT_IMAGE_DIM / 2.0)
    radius_arcsec = fraction * half_width_arcsec
    angle_rad = math.radians(SOURCE_OFFSET_POSITION_ANGLE_DEG)
    return radius_arcsec * math.sin(angle_rad), radius_arcsec * math.cos(angle_rad)


@cache
def load_defaults() -> dict[str, Any]:
    import tomllib

    here = Path(__file__).resolve()
    for root in (os.environ.get("REPO_ROOT"), here.parent.parent.parent.parent):
        if not root:
            continue
        path = Path(root) / "defaults.toml"
        if path.is_file():
            with path.open("rb") as handle:
                return tomllib.load(handle)
    raise SystemExit("no defaults.toml found - set REPO_ROOT to the repository root")


@cache
def load_receiver_bands() -> list[dict[str, Any]]:
    bands = [band for band in load_defaults()["receiver_band"] if band.get("enabled", True)]
    if not bands:
        raise SystemExit("defaults.toml: no enabled [[receiver_band]]")
    return sorted(bands, key=lambda band: float(band["min"]))


@cache
def load_all_parameter_specs() -> list[dict[str, Any]]:
    return load_defaults()["parameter_space"]


def _param_name_set(env_var: str) -> set[str]:
    return {name.strip() for name in os.environ.get(env_var, "").split(",") if name.strip()}


@cache
def load_parameter_space() -> list[dict[str, Any]]:
    force_off = _param_name_set("NS_DISABLE_PARAMS")
    force_on = _param_name_set("NS_ENABLE_PARAMS")
    specs = [
        spec
        for spec in load_all_parameter_specs()
        if str(spec["name"]) in force_on
        or (spec.get("enabled", True) and str(spec["name"]) not in force_off)
    ]
    for spec in specs:
        if spec.get("kind") == "band_start":
            bands = load_receiver_bands()
            spec.setdefault("min", min(float(band["min"]) for band in bands))
            spec.setdefault("max", max(float(band["max"]) for band in bands))
            check_channel_box_against_bands(specs)
    return specs


def check_channel_box_against_bands(specs: list[dict[str, Any]]) -> None:
    by_name = {str(spec["name"]): spec for spec in specs}
    count, width = by_name.get("channel_count"), by_name.get("channel_width_hz")
    if not count or not width:
        return
    widest = max(float(band["max"]) - float(band["min"]) for band in load_receiver_bands())
    smallest = float(count["min"]) * float(width["min"])
    if smallest > widest:
        raise SystemExit(
            f"defaults.toml: the smallest window the parameter space allows - {count['min']} "
            f"channels of {float(width['min']) / 1e6:g} MHz, {smallest / 1e6:g} MHz - is wider "
            f"than the widest receiver band ({widest / 1e6:g} MHz), so no start frequency can "
            f"hold it. Lower channel_count's min or channel_width_hz's min."
        )


# Redraw invalid starts by golden-ratio steps: they spread tries across bands
# while keeping the prior transform a pure function of the PolyChord cube.
START_REDRAW_STEP = 0.6180339887498949
MAX_START_REDRAWS = 64


@dataclass
class WindowFitStats:
    draws: int = 0
    as_sampled: int = 0
    width_reduced: int = 0
    count_reduced: int = 0
    redrawn_draws: int = 0
    redraws: int = 0
    seconds: float = 0.0

    def record(self, outcome: str, redraws: int, seconds: float) -> None:
        self.draws += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        self.redrawn_draws += 1 if redraws else 0
        self.redraws += redraws
        self.seconds += seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "as_sampled": self.as_sampled,
            "width_reduced": self.width_reduced,
            "count_reduced": self.count_reduced,
            "redrawn_draws": self.redrawn_draws,
            "redraws": self.redraws,
            "seconds": self.seconds,
            "seconds_per_draw": self.seconds / self.draws if self.draws else 0.0,
        }


WINDOW_FIT_STATS = WindowFitStats()


def gathered_window_fit_stats() -> dict[str, Any]:
    counters = ("draws", "as_sampled", "width_reduced", "count_reduced", "redrawn_draws", "redraws", "seconds")
    mine = WINDOW_FIT_STATS.as_dict()
    try:
        from mpi4py import MPI

        parts = MPI.COMM_WORLD.gather(mine, root=0) or [mine]
    except Exception:
        # No MPI (a host run, or the self-checks): this process is the run.
        parts = [mine]
    total: dict[str, Any] = {key: sum(part[key] for part in parts) for key in counters}
    total["ranks"] = len(parts)
    total["seconds_per_draw"] = total["seconds"] / total["draws"] if total["draws"] else 0.0
    return total


def window_fit_summary_line(stats: dict[str, Any]) -> str:
    return (
        f"spectral window fitting: {stats['draws']} draws over {stats['ranks']} rank(s), "
        f"{stats['as_sampled']} as sampled, {stats['width_reduced']} narrowed, "
        f"{stats['count_reduced']} with channels dropped, {stats['redrawn_draws']} restarted "
        f"({stats['redraws']} start frequencies discarded), {stats['seconds'] * 1e3:.1f} ms total"
    )


def start_frequency_from_cube(cube_value: float) -> tuple[dict[str, Any], float]:
    bands = load_receiver_bands()
    position = min(1.0, max(0.0, cube_value)) * len(bands)
    index = min(int(position), len(bands) - 1)
    band = bands[index]
    lower, upper = float(band["min"]), float(band["max"])
    return band, lower + (position - index) * (upper - lower)


def start_frequency_cube_value(start_frequency_hz: float) -> float:
    bands = load_receiver_bands()
    for index, band in enumerate(bands):
        lower, upper = float(band["min"]), float(band["max"])
        if start_frequency_hz <= upper or index == len(bands) - 1:
            within = (start_frequency_hz - lower) / (upper - lower)
            return (index + min(1.0, max(0.0, within))) / len(bands)
    return 1.0


@cache
def channel_floors() -> tuple[int, float]:
    by_name = {str(spec["name"]): spec for spec in load_parameter_space()}
    return int(by_name["channel_count"]["min"]), float(by_name["channel_width_hz"]["min"])


def fit_spectral_window(
    cube_value: float, channel_count: int, channel_width_hz: float, track: bool = False
) -> tuple[float, int, float]:
    started = time.perf_counter() if track else 0.0
    count_min, width_min = channel_floors()
    cube_value = min(1.0, max(0.0, cube_value))

    def finish(start: float, count: int, width: float, outcome: str, redraws: int) -> tuple[float, int, float]:
        if track:
            WINDOW_FIT_STATS.record(outcome, redraws, time.perf_counter() - started)
        return start, count, width

    for redraws in range(MAX_START_REDRAWS + 1):
        band, start = start_frequency_from_cube(cube_value)
        room = float(band["max"]) - start
        count, width = int(channel_count), float(channel_width_hz)
        if count * width <= room:
            return finish(start, count, width, "as_sampled", redraws)
        width = room / count
        if width >= width_min:
            return finish(start, count, width, "width_reduced", redraws)
        count = int(room // width_min)
        if count >= count_min:
            return finish(start, count, width_min, "count_reduced", redraws)
        cube_value = (cube_value + START_REDRAW_STEP) % 1.0

    # Unreachable for any parameter space check_channel_box_against_bands()
    # lets through unless the bands are so tightly packed that most of every
    # one of them is unusable, in which case the box, not the draw, is wrong.
    abort_run(
        f"no start frequency in {MAX_START_REDRAWS} tries left room for {count_min} channels of "
        f"{width_min / 1e6:g} MHz - lower channel_count's min or channel_width_hz's min"
    )
    raise AssertionError("abort_run does not return")


# GetDist / anesthetic axis labels (wrapped in $...$ by anesthetic).
PARAMETER_TEX_LABELS = {
    "log10_dynamic_range": r"\mathrm{log}_{10}(\rho_{DR})",
    "observation_minutes": r"t_{\mathrm{obs}}\,[\mathrm{min}]",
    "channel_count": r"n_{\mathrm{freq}}",
    "start_frequency_hz": r"\nu_{\mathrm{start}}\,[\mathrm{Hz}]",
    "channel_width_hz": r"\Delta\nu\,[\mathrm{Hz}]",
    "source_offset_fraction": r"f_{\mathrm{offset}}",
    "wsclean_niter": r"N_{\mathrm{iter}}",
    "wsclean_auto_threshold": r"\sigma_{\mathrm{thresh}}",
}


def write_polychord_paramnames(
    base_dir: Path,
    file_root: str,
    parameter_space: list[dict[str, Any]] | None = None,
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{file_root}.paramnames"
    specs = parameter_space if parameter_space is not None else load_parameter_space()
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
    # The command's own wall clock, as distinct from what this rank waited.
    # Only zygote_run() can tell them apart.
    binary_seconds: float = 0.0


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
        "-e",
        "OMP_WAIT_POLICY=PASSIVE",
    ]


def fill_disabled_parameters(raw: dict[str, Any]) -> None:
    enabled_names = {spec["name"] for spec in load_parameter_space()}
    for spec in load_all_parameter_specs():
        name = spec["name"]
        if name in enabled_names or name in raw:
            continue
        if "default" not in spec and spec.get("kind") == "band_start":
            raise SystemExit(
                f"defaults.toml: parameter_space '{name}' is disabled but has no "
                "`default` - a band_start dimension has no min/max to fall back "
                "on, so pin it explicitly, e.g. default = 1.4e9"
            )
        raw[name] = spec.get("default", spec.get("min", 0.0))


def cube_to_params(cube: np.ndarray, track: bool = False) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    specs = load_parameter_space()
    for i, spec in enumerate(specs):
        lower, upper = float(spec["min"]), float(spec["max"])
        value = lower + float(cube[i]) * (upper - lower)
        if spec.get("kind") == "integer":
            value = int(round(value))
        raw[spec["name"]] = value
    fill_disabled_parameters(raw)
    # The band-start dimension is resolved last: it fits the window to the band
    # it lands in, which can narrow the channels or drop some of them.
    for i, spec in enumerate(specs):
        if spec.get("kind") == "band_start":
            raw[spec["name"]], raw["channel_count"], raw["channel_width_hz"] = fit_spectral_window(
                float(cube[i]), raw["channel_count"], raw["channel_width_hz"], track=track
            )
    raw["dynamic_range"] = 10.0 ** raw.pop("log10_dynamic_range")
    raw["vla_config"] = "VLA.A"
    raw["source_flux_jy"] = 1.0
    # Keep source_offset_fraction for prior_vector()'s round-trip check.
    raw["source_l_arcsec"], raw["source_m_arcsec"] = source_offset_to_lm(
        raw["source_offset_fraction"], raw["start_frequency_hz"]
    )
    return raw


def params_key(params: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def stable_seed(global_seed: int, key: str) -> int:
    return (global_seed + int(key[:8], 16)) % (2**31 - 1)


# Usually pre-started by start-sidecars.sh; missing images start on first use.
_SIDECAR_CONTAINERS: dict[str, str] = json.loads(os.environ.get("NS_SIDECARS", "{}"))
_IMAGE_ENTRYPOINTS: dict[str, list[str]] = {}


def sidecar_container(image: str, platform: str, extra_args: list[str] | None = None) -> str:
    if image not in _SIDECAR_CONTAINERS:
        repo_root = os.environ.get("REPO_ROOT", os.getcwd())
        # The shared MS scratch tmpfs, when the run script made one; see
        # evaluation_scratch_dir().
        scratch = os.environ.get("NS_SCRATCH_DIR", "")
        scratch_mount = ["-v", f"{scratch}:{scratch}"] if scratch else []
        name = f"ri-ns-sidecar-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            [
                "docker", "run", "--detach", "--rm", "--name", name,
                # No network needed; "none" keeps loopback for meqserver and avoids
                # rootless Docker's bridge setup.
                "--network", "none",
                # MS and makems caches live in /dev/shm.
                "--shm-size", "512m",
                "--platform", platform,
                "-v", f"{repo_root}:{repo_root}",
                *scratch_mount,
                *(extra_args or []),
                "--entrypoint", "sleep", image, "infinity",
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        # ponytail: a SIGKILLed rank can leak this container; reap labelled sidecars.
        atexit.register(
            subprocess.run,
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _SIDECAR_CONTAINERS[image] = name
    return _SIDECAR_CONTAINERS[image]


def sidecar_exec(
    image: str,
    platform: str,
    workdir: Path,
    prefix: list[str] | None = None,
    interactive: bool = False,
) -> list[str]:
    return [
        "docker", "exec",
        *(["--interactive"] if interactive else []),
        "--workdir", str(workdir),
        sidecar_container(image, platform),
        *sidecar_command(image, prefix),
    ]


def sidecar_command(image: str, prefix: list[str] | None = None) -> list[str]:
    if image not in _IMAGE_ENTRYPOINTS:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Entrypoint}}", image],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()
        entrypoint = json.loads(inspected)
        if not entrypoint:
            raise SystemExit(f"FATAL: {image} has no ENTRYPOINT to run inside a sidecar")
        _IMAGE_ENTRYPOINTS[image] = entrypoint
    return [*(prefix or []), *_IMAGE_ENTRYPOINTS[image]]


_SIDECAR_WORKERS: dict[tuple[str, str], subprocess.Popen] = {}


def sidecar_worker(image: str, platform: str, argv: list[str]) -> subprocess.Popen:
    key = (image, argv[0])
    if key not in _SIDECAR_WORKERS:
        worker = subprocess.Popen(
            # Bypass image ENTRYPOINT; each request names its evaluation directory.
            ["docker", "exec", "--interactive", sidecar_container(image, platform), *argv],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        # Container teardown is registered by sidecar_container().
        atexit.register(worker.terminate)
        _SIDECAR_WORKERS[key] = worker
    return _SIDECAR_WORKERS[key]


ZYGOTE_COMMAND = "wsclean-zygote"


def zygote_run(
    image: str,
    platform: str,
    workdir: Path,
    argv: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> DockerRunResult:
    fields = [str(workdir), str(stdout_path), str(stderr_path), *argv]
    assert not any("\t" in field or "\n" in field for field in fields), fields
    request = "\t".join(fields) + "\n"
    started = time.perf_counter()
    for attempt in worker_attempts():
        zygote = sidecar_worker(image, platform, [ZYGOTE_COMMAND])
        if not worker_send(zygote.stdin, request):
            _SIDECAR_WORKERS.pop((image, ZYGOTE_COMMAND), None)
            continue
        reply = worker_reply(zygote.stdout, SHELL_REPLY_TIMEOUT)
        if reply:
            code, binary_seconds, peak_memory_bytes = reply.split("\t")
            return DockerRunResult(
                returncode=int(code),
                wall_seconds=time.perf_counter() - started,
                peak_memory_bytes=int(peak_memory_bytes),
                binary_seconds=float(binary_seconds),
            )
        if reply is None:
            zygote.kill()
        _SIDECAR_WORKERS.pop((image, ZYGOTE_COMMAND), None)
    stderr_path.write_text(
        f"FATAL: {image} {ZYGOTE_COMMAND} gave no reply, {len(WORKER_RETRY_DELAYS)} times\n"
    )
    return DockerRunResult(
        returncode=WORKER_DIED,
        wall_seconds=time.perf_counter() - started,
        peak_memory_bytes=0,
    )


def _fits_card_value(field: str) -> Any:
    field = field.lstrip()
    if field.startswith("'"):
        end = field.index("'", 1)
        return field[1:end].strip()
    value = field.split("/")[0].strip()
    return value == "T" if value in ("T", "F") else float(value)


def load_fits_2d(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an unscaled float primary HDU as a 2-D float64 array; reject others."""
    header: dict[str, Any] = {}
    with path.open("rb") as handle:
        while True:
            block = handle.read(2880)
            if len(block) < 2880:
                raise ValueError(f"{path}: FITS header ended mid-block")
            for start in range(0, 2880, 80):
                card = block[start:start + 80].decode("ascii")
                key = card[:8].strip()
                if key == "END":
                    block = None
                    break
                if card[8:10] != "= ":
                    continue  # COMMENT, HISTORY and blank cards carry no value
                header[key] = _fits_card_value(card[10:])
            if block is None:
                break
        bitpix = int(header["BITPIX"])
        if bitpix not in (-32, -64):
            raise ValueError(f"{path}: BITPIX {bitpix} is not a float image")
        if float(header.get("BSCALE", 1.0)) != 1.0 or float(header.get("BZERO", 0.0)) != 0.0:
            raise ValueError(f"{path}: scaled FITS data (BSCALE/BZERO) is not supported")
        shape = [int(header[f"NAXIS{axis}"]) for axis in range(int(header["NAXIS"]), 0, -1)]
        count = int(np.prod(shape))
        raw = handle.read(count * (abs(bitpix) // 8))
    if len(raw) != count * (abs(bitpix) // 8):
        raise ValueError(f"{path}: FITS data block is short")
    data = np.frombuffer(raw, dtype=">f4" if bitpix == -32 else ">f8").reshape(shape)
    # astype, not asarray: frombuffer gives a read-only view, and asarray would
    # hand it straight back on a host where the FITS byte order is native.
    # compute_image_metrics writes one pixel of what it is given.
    image = np.squeeze(data.astype(np.float64))
    if image.ndim != 2:
        raise ValueError(f"{path} is not 2-D after squeezing; shape={image.shape}")
    return image, header


def rms(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    if not np.isnan(values).any():
        return float(np.sqrt(np.dot(values.ravel(), values.ravel()) / values.size))
    return float(np.sqrt(np.nanmean(values * values)))


def sigma_res(residual: np.ndarray, dirty: np.ndarray) -> float:
    """R2D2-paper data-fidelity: ||residual_dirty||_2 / ||dirty||_2."""
    residual_flat = residual.ravel()
    dirty_flat = dirty.ravel()
    return float(
        np.sqrt(np.dot(residual_flat, residual_flat))
        / max(np.sqrt(np.dot(dirty_flat, dirty_flat)), 1e-12)
    )


def source_pixel(
    header: dict[str, Any],
    cx: int,
    cy: int,
    source_l_arcsec: float,
    source_m_arcsec: float,
    x_size: int,
    y_size: int,
) -> tuple[int, int]:
    if source_l_arcsec == 0.0 and source_m_arcsec == 0.0:
        return cx, cy
    cdelt1_arcsec = float(header["CDELT1"]) * 3600.0
    cdelt2_arcsec = float(header["CDELT2"]) * 3600.0
    sx = cx + int(round(source_l_arcsec / cdelt1_arcsec))
    sy = cy + int(round(source_m_arcsec / cdelt2_arcsec))
    return max(0, min(x_size - 1, sx)), max(0, min(y_size - 1, sy))


@lru_cache(maxsize=64)
def off_source_mask(shape: tuple[int, int], sx: int, sy: int) -> np.ndarray:
    """Pixels more than 5 px from the source, cached: one mask serves a run.

    Every evaluation of a search asks for the same handful of (shape, source)
    combinations, and building one costs more than the reduction it feeds.
    """
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask = (yy - sy) ** 2 + (xx - sx) ** 2 > 25
    mask.setflags(write=False)  # callers share this array; none may edit it
    return mask


def compute_image_metrics(
    image_path: Path,
    source_flux_jy: float,
    wall_seconds: float,
    peak_memory_bytes: int,
    dirty_path: Path | None = None,
    residual_dirty_path: Path | None = None,
    source_l_arcsec: float = 0.0,
    source_m_arcsec: float = 0.0,
    pixel_size_arcsec: float | None = None,
) -> dict[str, float]:
    image, header = load_fits_2d(image_path)

    y_size, x_size = image.shape
    # R2D2 headers lack WCS, so callers must provide CDELT; guessing it can
    # centre a good image incorrectly. WSClean headers already provide it.
    if "CDELT1" not in header or "CDELT2" not in header:
        if pixel_size_arcsec is None:
            raise ValueError(
                f"{image_path}: image has no CDELT1/CDELT2 and no pixel_size_arcsec was given"
            )
        header = {
            **header,
            "CDELT1": -pixel_size_arcsec / 3600.0,
            "CDELT2": pixel_size_arcsec / 3600.0,
        }
    # FITS CRPIX is 1-based; `size / 2 + 1` is the centred default for even axes.
    cx = int(round(float(header.get("CRPIX1", x_size / 2.0 + 1.0)) - 1.0))
    cy = int(round(float(header.get("CRPIX2", y_size / 2.0 + 1.0)) - 1.0))
    cx = max(0, min(x_size - 1, cx))
    cy = max(0, min(y_size - 1, cy))
    sx, sy = source_pixel(header, cx, cy, source_l_arcsec, source_m_arcsec, x_size, y_size)

    off_rms = rms(image[off_source_mask(image.shape, sx, sy)])
    peak = float(np.nanmax(np.abs(image)))
    snr = peak / off_rms if off_rms > 0 else float("inf")
    log_snr = math.log10(snr) if math.isfinite(snr) and snr > 0 else 99.0
    peak_flux_error = abs(float(image[sy, sx]) - source_flux_jy)

    # The residual differs at one pixel only. Mutate the already-owned float64
    # array for the two residual reductions, avoiding another full-image copy.
    source_value = image[sy, sx]
    try:
        image[sy, sx] -= source_flux_jy
        if np.isnan(image).any():
            total_rms = rms(image)
            residual_norm = float(np.linalg.norm(image))
        else:
            residual_norm = float(np.sqrt(np.dot(image.ravel(), image.ravel())))
            total_rms = residual_norm / math.sqrt(image.size)
        relative_l2_error = residual_norm / max(abs(source_flux_jy), 1e-12)
    finally:
        image[sy, sx] = source_value

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


PROFILING_STAGE_FIELDS = (
    "simulate_seconds",
    "convert_seconds",
    "image_container_seconds",
    "image_binary_seconds",
    "metrics_seconds",
)

# The stages an evaluation is made of, which is not every field above:
# image_binary_seconds is the imaging *inside* image_container_seconds, and
# adding it would count that time twice.
ACCOUNTED_STAGE_FIELDS = (
    "simulate_seconds",
    "convert_seconds",
    "image_container_seconds",
    "metrics_seconds",
)


def evaluation_busy_seconds(
    evaluations: list[dict[str, Any]],
    window: tuple[float, float],
) -> tuple[float, float]:
    start, end = window
    spans = []
    for record in evaluations:
        timing = record.get("timing") or {}
        began, finished = timing.get("started_epoch"), timing.get("ended_epoch")
        if began is None or finished is None:
            continue
        began, finished = max(float(began), start), min(float(finished), end)
        if finished > began:
            spans.append((began, finished))
    if not spans:
        return 0.0, 0.0
    spans.sort()
    union = 0.0
    merged_start, merged_end = spans[0]
    for began, finished in spans:
        if began > merged_end:
            union += merged_end - merged_start
            merged_start, merged_end = began, finished
        else:
            merged_end = max(merged_end, finished)
    union += merged_end - merged_start
    return sum(finish - begin for begin, finish in spans), union


def backfill_busy_seconds(summary: dict[str, Any]) -> dict[str, Any]:
    profiling = summary.get("profiling") or {}
    total_wall = profiling.get("total_wall_seconds")
    if profiling.get("busy_worker_seconds") is not None or not total_wall:
        return profiling

    spans = []
    for record in summary.get("evaluations") or []:
        timing = record.get("timing") or {}
        length = sum(float(timing.get(field) or 0.0) for field in ACCOUNTED_STAGE_FIELDS)
        if not length:
            continue
        eval_dir = (record.get("paths") or {}).get("eval_dir")
        try:
            ended = (Path(eval_dir) / "metrics.json").stat().st_mtime
        except (OSError, TypeError):
            return profiling
        spans.append({"timing": {"started_epoch": ended - length, "ended_epoch": ended}})
    if not spans:
        return profiling

    end = max(span["timing"]["ended_epoch"] for span in spans)
    busy_worker, busy_wall = evaluation_busy_seconds(
        spans, (end - float(total_wall), end))
    if busy_wall * worker_procs(int(profiling.get("mpi_procs") or 1)) < busy_worker:
        return profiling
    return {**profiling, "busy_worker_seconds": busy_worker,
            "busy_wall_seconds": busy_wall, "busy_seconds_reconstructed": True}


def summarize_profiling(
    evaluations: list[dict[str, Any]],
    total_wall_seconds: float,
    mpi_procs: int,
    run_started_epoch: float | None = None,
) -> dict[str, Any]:
    """Aggregate evaluation timings into a run-level breakdown."""
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

    accounted = sum(totals[field] for field in ACCOUNTED_STAGE_FIELDS)
    polychord_overhead = total_wall_seconds - accounted if mpi_procs == 1 else None

    # Anchor missing starts to the last evaluation's end.
    ends = [
        float(timing["ended_epoch"]) for timing in
        ((record.get("timing") or {}) for record in evaluations)
        if timing.get("ended_epoch") is not None
    ]
    window_start = run_started_epoch if run_started_epoch is not None else (
        max(ends) - total_wall_seconds if ends else None)
    busy_worker, busy_wall = evaluation_busy_seconds(
        evaluations, (window_start, window_start + total_wall_seconds)
    ) if window_start is not None else (None, None)

    return {
        "mpi_procs": mpi_procs,
        "total_wall_seconds": total_wall_seconds,
        "busy_worker_seconds": busy_worker,
        "busy_wall_seconds": busy_wall,
        "stage_totals_seconds": {
            "simulate": totals["simulate_seconds"],
            "convert": totals["convert_seconds"] if counts["convert_seconds"] else None,
            "image_container": image_container_total,
            "image_binary": image_binary_total if counts["image_binary_seconds"] else None,
            "image_container_overhead": image_container_overhead if image_container_overhead_count else None,
            "metrics": totals["metrics_seconds"],
        },
        "stage_eval_counts": counts,
        "accounted_worker_seconds": accounted,
        "accounted_seconds": accounted if mpi_procs == 1 else None,
        "polychord_overhead_seconds": polychord_overhead,
        "note": (
            "stage totals are summed worker-seconds; accounted_seconds and "
            "polychord_overhead_seconds are only emitted for mpi_procs == 1."
        ),
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    value = max(0.0, float(seconds))
    if value < 1.0:
        milliseconds = round(value * 1000)
        # Preserve nonzero sub-millisecond measurements.
        if milliseconds == 0 and value > 0.0:
            return f"{value * 1e6:.0f}us"
        # Rounded 1000ms belongs in the seconds branch.
        if milliseconds < 1000:
            return f"{milliseconds}ms"
    if value < 10.0:
        return f"{value:.2f}s"
    if value < 60.0:
        return f"{value:.1f}s"
    total = int(round(value))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_share(fraction: float | None) -> str:
    if fraction is None:
        return ""
    pct = 100.0 * fraction
    if 0.0 < pct < 0.05:
        return "<0.1%"
    return f"{pct:.1f}%"


# (stage key, eval-count key, label, indented under the row above it).
PROFILING_VIEW_STAGES = (
    ("simulate", "simulate_seconds", "simulate (MeqTrees)", False),
    ("convert", "convert_seconds", "convert (MS -> .mat)", False),
    ("image_container", "image_container_seconds", "{imager} container (total)", False),
    ("image_binary", "image_binary_seconds", "of which: {imager} itself", True),
    ("image_container_overhead", "image_container_overhead_seconds", "of which: container overhead", True),
    ("metrics", "metrics_seconds", "metrics computation", False),
)

UNACCOUNTED_LABEL = "unaccounted (PolyChord sampling + idle)"


HARNESS_LABEL = "harness (Python around the stages)"
POLYCHORD_LABEL = "PolyChord (no evaluation in flight)"
IDLE_LABEL = "idle (waiting on other workers)"

PROFILING_VIEW_NOTE = (
    "stage totals are worker-seconds across evaluations; shares use worker-time "
    "(wall clock x workers, excluding rank 0), so top-level rows total 100%. "
    "Divide totals by workers for wall-clock stage costs; they sum to end-to-end time."
)

# Appended to the note above once the records carry evaluation intervals, and
# omitted for the runs written before they did, whose one combined row it would
# be describing a split of that is not on the page.
PROFILING_SPLIT_NOTE = (
    " Everything above the sum happened inside a likelihood evaluation; below "
    "it, PolyChord is wall clock during which no rank was inside one - the "
    "sampler's own work, plus the run's start-up and shutdown, charged to every "
    "worker because not one of them could spend it - and idle is what is left, "
    "workers waiting while other workers were still evaluating."
)


PROFILING_UNSPLIT_NOTE = (
    " The remainder is one row here because this run's records carry no "
    "evaluation intervals - it finished before those were stamped, and the "
    "mtimes of its own metrics.json files are not a usable timeline either - so "
    "what of it was PolyChord and what was idle cannot be recovered."
)


def profiling_breakdown(profiling: dict[str, Any], algorithm: str | None = None) -> dict[str, Any]:
    imager = algorithm or "image"
    mpi_procs = int(profiling.get("mpi_procs") or 1)
    total_wall = profiling.get("total_wall_seconds")
    total_wall = None if total_wall is None else float(total_wall)
    stages = profiling.get("stage_totals_seconds") or {}
    counts = profiling.get("stage_eval_counts") or {}
    accounted = profiling.get("accounted_worker_seconds")
    if accounted is None:
        accounted = profiling.get("accounted_seconds")
    accounted = float(accounted or 0.0)

    workers = worker_procs(mpi_procs)
    budget = None if total_wall is None else total_wall * workers
    # A budget below what the stages already accounted for would push shares
    # over 100%; an oversubscribed host or a clock jump can produce one, so fall
    # back to the accounted total and keep the breakdown adding up.
    denominator = accounted if budget is None else max(budget, accounted)
    denominator = denominator or None

    def share(value: float | None) -> float | None:
        if value is None or not denominator:
            return None
        return value / denominator

    rows = []
    for key, count_key, label, is_sub in PROFILING_VIEW_STAGES:
        value = stages.get(key)
        if value is None:
            continue
        value = float(value)
        evals = int(counts.get(count_key) or 0)
        rows.append({
            "key": key,
            "label": label.format(imager=imager),
            "seconds": value,
            "evals": evals,
            "per_eval_seconds": value / evals if evals else None,
            "share": share(value),
            "is_sub": is_sub,
        })

    evals = max((row["evals"] for row in rows), default=0)
    unaccounted = None if denominator is None else max(0.0, denominator - accounted)
    busy_worker = profiling.get("busy_worker_seconds")
    busy_wall = profiling.get("busy_wall_seconds")

    if unaccounted and busy_worker is not None and busy_wall is not None and total_wall is not None:
        # Worker-seconds inside a likelihood call but outside every timed stage:
        # this repo's own Python between the subprocess calls. A stage in all
        # but name, so it joins them above the sum rather than below it.
        harness = max(0.0, min(float(busy_worker), denominator) - accounted)
        if profiling.get("busy_seconds_reconstructed"):
            # A reconstructed interval is the stages themselves, so there is no
            # harness time in it to show. What the subtraction leaves is the
            # float noise of having gone round the addition twice, and it
            # belongs in idle with everything else that was never measured.
            harness = 0.0
        if harness > 0.0:
            rows.append({
                "key": "harness",
                "label": HARNESS_LABEL,
                "seconds": harness,
                "evals": evals,
                "per_eval_seconds": harness / evals if evals else None,
                "share": share(harness),
                "is_sub": False,
            })
        evaluating = accounted + harness
        # Wall clock during which no rank was inside an evaluation, charged to
        # every worker because not one of them could spend it. Capped by what
        # is left, so a clock that moved cannot push the rows past 100%.
        polychord = min(denominator - evaluating,
                        max(0.0, total_wall - float(busy_wall)) * workers)
        idle = max(0.0, denominator - evaluating - polychord)
        remainder = [
            {"key": "polychord", "label": POLYCHORD_LABEL, "seconds": polychord},
            {"key": "idle", "label": IDLE_LABEL, "seconds": idle},
        ]
        subtotal_label, subtotal = "evaluating (sum of the above)", evaluating
        note = PROFILING_VIEW_NOTE + PROFILING_SPLIT_NOTE
        total_label = "end-to-end (evaluating + PolyChord + idle)"
        terms = [f"{format_duration(evaluating)} evaluating",
                 f"+ {format_duration(polychord)} PolyChord",
                 f"+ {format_duration(idle)} idle"]
    else:
        # No evaluation intervals on the records - a run written before they
        # were stamped - so the remainder stays the one bucket named after
        # everything that could be in it.
        remainder = [] if unaccounted is None else [
            {"key": "unaccounted", "label": UNACCOUNTED_LABEL, "seconds": unaccounted}]
        subtotal_label, subtotal = "accounted (sum of stages above)", accounted
        note = PROFILING_VIEW_NOTE + PROFILING_UNSPLIT_NOTE
        total_label = "end-to-end (accounted + unaccounted)"
        terms = [f"{format_duration(accounted)} accounted",
                 f"+ {format_duration(unaccounted)} unaccounted"]
    for row in remainder:
        row["share"] = share(row["seconds"])

    return {
        "imager": imager,
        "mpi_procs": mpi_procs,
        "worker_procs": workers,
        "evals": evals,
        "total_wall_seconds": total_wall,
        "busy_worker_seconds": busy_worker,
        "busy_wall_seconds": busy_wall,
        "worker_seconds_budget": denominator,
        "accounted_seconds": accounted,
        "subtotal_label": subtotal_label,
        "subtotal_seconds": subtotal,
        "subtotal_share": share(subtotal),
        "subtotal_per_eval_seconds": subtotal / evals if evals else None,
        "remainder_rows": remainder,
        "total_label": total_label,
        "note": note,
        "equation_terms": terms,
        "rows": rows,
    }


def badness_from_metrics(metrics: dict[str, float]) -> float:
    log_snr_loss = max(0.0, 3.0 - metrics["log_snr"])
    fidelity_loss = min(metrics["relative_l2_error"], 10.0)
    time_loss = min(metrics["wall_seconds"] / 60.0, 5.0)
    memory_loss = min(metrics["peak_memory_bytes"] / (2.0 * 1024.0 * 1024.0 * 1024.0), 5.0)
    return float(log_snr_loss + fidelity_loss + 0.05 * time_loss + 0.02 * memory_loss)


def resolve_metric(metric_spec: str) -> tuple[Callable[[dict[str, float]], float], str]:
    if metric_spec == "badness":
        return badness_from_metrics, (
            "PolyChord log-likelihood is the composite badness score; higher means worse reconstruction."
        )

    if metric_spec in METRIC_NAMES:
        key = metric_spec
        return lambda metrics: float(metrics[key]), (
            f"PolyChord log-likelihood is the raw metric `{key}` with no sign flip; "
            "higher returned values are preferred by PolyChord."
        )

    try:
        code = compile(metric_spec, "<metric>", "eval")
    except SyntaxError as exc:
        raise SystemExit(f"invalid --metric expression: {exc}") from exc

    globals_ns = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    globals_ns["__builtins__"] = {}
    probe_metrics = {name: 1.0 for name in METRIC_NAMES}
    try:
        eval(code, globals_ns, probe_metrics)
    except Exception as exc:
        raise SystemExit(f"invalid --metric expression: {exc}") from exc

    return lambda metrics: float(eval(code, globals_ns, metrics)), (
        f"PolyChord log-likelihood is the expression `{metric_spec}` with no sign flip; "
        "higher returned values are preferred by PolyChord."
    )


def mpi_rank() -> int:
    # The launcher's environment first: `import mpi4py` initialises MPI, which
    # costs 0.24s per rank when eight of them do it at once, and prewarm() asks
    # for the rank before anything else here has touched MPI.
    launcher_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    if launcher_rank is not None:
        return int(launcher_rank)
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_rank())
    except ImportError:
        return 0


def worker_procs(mpi_procs: int) -> int:
    return mpi_procs - 1 if mpi_procs > 1 else 1


def read_evaluation_record(metrics_path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(metrics_path.read_text())
    except FileNotFoundError:
        # An evaluation that was still in flight. The ordinary case on every
        # resume, and not worth a word.
        return None
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        print(f"WARNING: ignoring unreadable {metrics_path}", file=sys.stderr, flush=True)
        return None
    return record


def load_evaluations_from_dir(evaluations_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(evaluations_dir.glob("eval-*/metrics.json")):
        record = read_evaluation_record(metrics_path)
        if record is not None:
            records.append(record)
    return records


def adopt_completed_evaluations(
    evaluations_dir: Path,
    cache: dict[str, float],
) -> int:
    """Cache finished evaluations and remove incomplete directories."""
    import shutil

    adopted = 0
    for eval_dir in sorted(evaluations_dir.glob("eval-*")):
        if not eval_dir.is_dir():
            continue
        record = read_evaluation_record(eval_dir / "metrics.json")
        if record is None:
            # ignore_errors because every rank runs this, and they are all
            # removing the same directories at the same moment.
            shutil.rmtree(eval_dir, ignore_errors=True)
            continue
        cache[params_key(record["params"])] = float(record["objective"])
        adopted += 1
    return adopted


def self_check_resume_adoption() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        evaluations_dir = Path(tmp)
        params = {"channel_count": 2, "noise_seed": 7}
        eval_dir = evaluations_dir / "eval-0001-abc"
        eval_dir.mkdir()
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": params, "objective": 0.5})

        cache: dict[str, float] = {}
        adopted = adopt_completed_evaluations(evaluations_dir, cache)
        assert adopted == 1
        # Keyed the way the likelihood keys it, or the resumed run would
        # recompute the point and collide with its own directory.
        assert params_key(params) in cache
        # The objective alone, not the record: the likelihood returns this
        # value directly and nothing else reads an adopted evaluation.
        assert cache[params_key(params)] == 0.5
        # The next eval id continues rather than restarting at 1.
        assert adopted + 1 == 2

    with tempfile.TemporaryDirectory() as tmp:
        # An evaluation that was in flight when the run stopped has a
        # directory but no metrics.json. It has to go: it holds nothing, and
        # simulate_measurement_set() creates directories with exist_ok=False,
        # so leaving it would crash the resumed run when the sampler proposed
        # that point again.
        evaluations_dir = Path(tmp)
        finished = evaluations_dir / "eval-0001-abc"
        finished.mkdir()
        write_evaluation_record(finished, {"eval_id": 1, "params": {"a": 1}, "objective": 0.5})
        in_flight = evaluations_dir / "eval-0002-def"
        in_flight.mkdir()
        (in_flight / "sim.ms").write_text("half a measurement set")

        cache = {}
        assert adopt_completed_evaluations(evaluations_dir, cache) == 1
        assert finished.exists()
        assert not in_flight.exists()

    with tempfile.TemporaryDirectory() as tmp:
        # A metrics.json the run was killed in the middle of writing. Before
        # this was tolerated, one zero-byte file like this ended a search for
        # good: json.loads raised on every restart and on every ./ri resume,
        # before either scored anything, so the retry loop gave up too.
        evaluations_dir = Path(tmp)
        good = evaluations_dir / "eval-0001-abc"
        good.mkdir()
        write_evaluation_record(good, {"eval_id": 1, "params": {"a": 1}, "objective": 0.5})
        killed = evaluations_dir / "eval-0002-def"
        killed.mkdir()
        (killed / "metrics.json").write_text("")
        half = evaluations_dir / "eval-0003-ghi"
        half.mkdir()
        (half / "metrics.json").write_text('{\n  "eval_id": 3,\n  "para')

        cache = {}
        assert load_evaluations_from_dir(evaluations_dir) == [
            {"eval_id": 1, "params": {"a": 1}, "objective": 0.5}
        ]
        assert adopt_completed_evaluations(evaluations_dir, cache) == 1
        assert good.exists()
        # Removed, not merely skipped: simulate_measurement_set() creates the
        # directory with exist_ok=False, so a kept one crashes the run this is
        # rescuing the moment the sampler proposes that point again.
        assert not killed.exists()
        assert not half.exists()

    with tempfile.TemporaryDirectory() as tmp:
        # The record is renamed into place rather than truncated and rewritten,
        # so a rank killed mid-write leaves no metrics.json at all instead of
        # half of one. A rename gives a new inode; an in-place write does not.
        eval_dir = Path(tmp) / "eval-0001-abc"
        eval_dir.mkdir()
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": {"a": 1}, "objective": 0.5})
        first_inode = (eval_dir / "metrics.json").stat().st_ino
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": {"a": 1}, "objective": 0.7})
        assert (eval_dir / "metrics.json").stat().st_ino != first_inode
        assert list(eval_dir.iterdir()) == [eval_dir / "metrics.json"]

    with tempfile.TemporaryDirectory() as tmp:
        # A fresh run adopts nothing and starts at id 1.
        cache = {}
        assert adopt_completed_evaluations(Path(tmp), cache) == 0


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file, then replace `path` atomically."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    partial.replace(path)


# Scored evaluations keep only replayable inputs, metrics and the three images
# the retention policy may later keep; see docs/nested-sampling-disk-footprint.md.
# Set NS_KEEP_MEASUREMENT_SETS=1 to retain intermediate artefacts for replay
# benchmarks. The dirty and residual images are NOT here: which evaluations keep
# them is a whole-run decision, taken by prune_run_artefacts() once every
# evaluation has been scored and can be ranked.
PRUNED_ARTEFACTS = (
    ("sim.ms", "measurement_set"),
    ("VLAA_ANT", None),
    ("r2d2_data.mat", "mat"),
    ("wsclean/recon-model.fits", None),
    ("wsclean/recon-psf.fits", None),
    ("r2d2/r2d2_data/PSF.fits", None),
)


def evaluation_scratch_dir(eval_dir: Path) -> Path | None:
    root = os.environ.get("NS_SCRATCH_DIR", "")
    return Path(root) / eval_dir.name if root else None


def prune_evaluation_artefacts(eval_dir: Path, record: dict[str, Any]) -> None:
    import shutil

    keeping = "error" in record or os.environ.get("NS_KEEP_MEASUREMENT_SETS", "0") != "0"
    scratch = evaluation_scratch_dir(eval_dir)
    if scratch is not None and scratch.is_dir():
        if keeping:
            for produced in scratch.iterdir():
                destination = eval_dir / produced.name
                if destination.is_dir():
                    shutil.rmtree(destination)
                shutil.move(str(produced), destination)
            if (eval_dir / "sim.ms").exists() and "measurement_set" in record.get("paths", {}):
                record["paths"]["measurement_set"] = str(eval_dir / "sim.ms")
            if (eval_dir / "r2d2_data.mat").exists() and "mat" in record.get("paths", {}):
                record["paths"]["mat"] = str(eval_dir / "r2d2_data.mat")
        shutil.rmtree(scratch, ignore_errors=True)
    if keeping:
        return
    for name, path_key in PRUNED_ARTEFACTS:
        target = eval_dir / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        if path_key:
            record.get("paths", {}).pop(path_key, None)


# A finished run keeps every image for the IMAGE_KEEP_ENDS worst and best
# evaluations, then one in IMAGE_KEEP_STRIDE across the ordered middle. The
# extremes are the failure modes the search exists to find and the contrast
# that makes them readable; the stride keeps the ground between them legible
# without storing an image per evaluation. Ordering is by objective, so this is
# necessarily a whole-run decision: when an evaluation is scored, whether it
# lands in the worst 20 depends on evaluations that have not run yet.
IMAGE_KEEP_ENDS = int(os.environ.get("NS_IMAGE_KEEP_ENDS", "20"))
IMAGE_KEEP_STRIDE = int(os.environ.get("NS_IMAGE_KEEP_STRIDE", "100"))

# Both algorithms record their dirty image, reconstruction and residual dirty
# image under these keys, so the policy needs no per-algorithm filenames.
RETAINED_IMAGE_KEYS = ("image", "dirty", "residual")


# WSClean prints why CLEAN stopped and how far it got, and nothing else records
# it: an evaluation that exhausted `-niter` scored where CLEAN had got to, not
# what it converges to, which is a different thing for a failure-mode search to
# have found. Read out of the log while it is still beside the evaluation, so
# the log itself does not have to be kept.
_CLEAN_STOPPED = re.compile(r"Stopped on peak [^,]+, because ([^\n]+)")
_CLEAN_ITERATIONS = re.compile(r"Performed (\d+) iterations in total")
_CLEAN_MAJOR = re.compile(r"(\d+) major iterations were performed")


def clean_convergence(log_text: str) -> dict[str, Any]:
    """WSClean's stopping condition and iteration counts, from its stdout.

    The last `Stopped on peak` line is the terminal one; the earlier ones are
    per-major-cycle minor-loop exits. `Performed N iterations in total` is
    cumulative, so the last is the run's total.
    """
    out: dict[str, Any] = {}
    reasons = _CLEAN_STOPPED.findall(log_text)
    if reasons:
        last = reasons[-1].strip()
        if "maximum number" in last:
            out["clean_stop_reason"] = "max-iterations"
        elif "minor-loop threshold" in last:
            out["clean_stop_reason"] = "minor-loop-threshold"
        elif "threshold" in last:
            out["clean_stop_reason"] = "threshold"
        else:
            out["clean_stop_reason"] = last.rstrip(".")
    iterations = _CLEAN_ITERATIONS.findall(log_text)
    if iterations:
        out["clean_iterations"] = int(iterations[-1])
    major = _CLEAN_MAJOR.search(log_text)
    if major:
        out["clean_major_iterations"] = int(major.group(1))
    return out


def clean_convergence_from(log_path: Path) -> dict[str, Any]:
    """clean_convergence() for a log that may not be readable."""
    try:
        return clean_convergence(log_path.read_text(errors="replace"))
    except OSError:
        return {}



def evaluation_key(record: dict[str, Any]) -> str:
    """Identify one evaluation.

    Not eval_id: PolyChord reuses it across parameter vectors, so a run holds
    several eval-0083-* directories and keying retention on the number alone
    would spare or delete whole groups of evaluations together. The directory
    name carries the parameter hash and is unique.
    """
    eval_dir = (record.get("paths") or {}).get("eval_dir")
    return Path(eval_dir).name if eval_dir else f"eval_id:{record.get('eval_id')}"


def evaluations_keeping_images(
    records: list[dict[str, Any]],
    ends: int | None = None,
    stride: int | None = None,
) -> set[str]:
    """The evaluation keys whose images a finished run keeps."""
    ends = IMAGE_KEEP_ENDS if ends is None else ends
    stride = IMAGE_KEEP_STRIDE if stride is None else stride
    # A failed evaluation keeps everything: it cannot be ranked on an objective
    # it never produced, and it is the case worth looking at.
    keep = {evaluation_key(r) for r in records if "error" in r}
    scored = [r for r in records if "error" not in r and r.get("objective") is not None]
    ordered = sorted(scored, key=lambda r: (r["objective"], evaluation_key(r)))
    if len(ordered) <= 2 * ends:
        return keep | {evaluation_key(r) for r in ordered}
    keep |= {evaluation_key(r) for r in ordered[:ends]}
    keep |= {evaluation_key(r) for r in ordered[len(ordered) - ends:]}
    keep |= {evaluation_key(r) for r in ordered[ends:len(ordered) - ends:stride]}
    return keep


def _recorded_image_path(evaluations_dir: Path, recorded: str) -> Path | None:
    """Resolve a recorded image path against the run being pruned.

    Records carry absolute paths from wherever the run was scored, so a run
    that was copied or moved still names the original files. Resolving inside
    evaluations_dir first, and refusing anything that lands outside it, keeps
    pruning one run from deleting another one's images.
    """
    root = evaluations_dir.resolve()
    text = str(recorded)
    candidates = []
    if "evaluations" in text:
        candidates.append(evaluations_dir / text.split("evaluations", 1)[1].lstrip("/\\"))
    candidates.append(Path(recorded))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved == root or root in resolved.parents:
            return resolved
    return None


# Logs a scored evaluation keeps only for as long as its images: `./ri profile
# --phases` reads per-phase timings out of the imager's stdout, and the few
# hundred evaluations a run retains are plenty for the medians it reports.
# WSClean's stdout was 9.3 GB across 828,825 files before this, and the part of
# it worth outliving the log - why CLEAN stopped, how far it got - is read into
# the record by clean_convergence() at scoring time. A failed evaluation keeps
# every log, as it keeps every other artefact.
PRUNED_EVALUATION_LOGS = (
    "wsclean.stdout.log",
    "wsclean.stderr.log",
    "r2d2.stdout.log",
    "r2d2.stderr.log",
    "simulate.stdout.log",
    "simulate.stderr.log",
    "convert.stdout.log",
    "convert.stderr.log",
    "makems.log",
    "meqtree-pipeliner.log",
)


def prune_run_artefacts(evaluations_dir: Path, records: list[dict[str, Any]]) -> int:
    """Delete images and logs of evaluations the retention policy does not keep.

    Mutates `records` so a summary never names a file this just deleted, which
    is what lets the report fall back to its placeholder. Set
    NS_KEEP_ALL_IMAGES=1 to keep everything.
    """
    if os.environ.get("NS_KEEP_ALL_IMAGES", "0") != "0":
        return 0
    keep = evaluations_keeping_images(records)
    removed = 0
    for record in records:
        paths = record.get("paths") or {}
        retained = evaluation_key(record) in keep
        for key in RETAINED_IMAGE_KEYS:
            recorded = paths.get(key)
            if not recorded:
                continue
            target = _recorded_image_path(evaluations_dir, recorded)
            if target is None:
                # Already gone; the record must not keep pointing at it.
                paths.pop(key, None)
                continue
            if retained:
                continue
            target.unlink(missing_ok=True)
            paths.pop(key, None)
            removed += 1
        if retained or "error" in record:
            continue
        eval_dir = Path(paths.get("eval_dir") or "")
        local = evaluations_dir / eval_dir.name if eval_dir.name else None
        if local is None or not local.is_dir():
            continue
        for name in PRUNED_EVALUATION_LOGS:
            log = local / name
            if log.is_file():
                log.unlink()
                removed += 1
    return removed



def self_check_evaluation_pruning() -> None:
    import tempfile

    def evaluation(root: Path, name: str) -> Path:
        eval_dir = root / name
        (eval_dir / "sim.ms" / "table.f0").parent.mkdir(parents=True)
        (eval_dir / "sim.ms" / "table.f0").write_text("data")
        (eval_dir / "VLAA_ANT").mkdir()
        (eval_dir / "r2d2_data.mat").write_text("mat")
        (eval_dir / "wsclean").mkdir()
        for image in ("image", "dirty", "residual", "model", "psf"):
            (eval_dir / "wsclean" / f"recon-{image}.fits").write_text(image)
        return eval_dir

    def record_for(eval_dir: Path, **extra: Any) -> dict[str, Any]:
        paths = {"eval_dir": str(eval_dir), "measurement_set": str(eval_dir / "sim.ms"),
                 "mat": str(eval_dir / "r2d2_data.mat"), "image": str(eval_dir / "wsclean" / "recon-image.fits"),
                 "dirty": str(eval_dir / "wsclean" / "recon-dirty.fits"),
                 "residual": str(eval_dir / "wsclean" / "recon-residual.fits")}
        return {"eval_id": 1, "params": {"a": 1}, "objective": 0.5, "paths": paths, **extra}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scored = evaluation(root, "eval-0001-scored")
        written = write_evaluation_record(scored, record_for(scored))
        assert not (scored / "sim.ms").exists() and not (scored / "VLAA_ANT").exists()
        assert not (scored / "r2d2_data.mat").exists()
        # The evidence a failure-mode search exists to produce is never pruned.
        # These three survive scoring because which evaluations keep them is a
        # whole-run decision, taken later by prune_run_artefacts().
        for image in ("image", "dirty", "residual"):
            assert (scored / "wsclean" / f"recon-{image}.fits").exists(), image
        # The two nothing ever reads go straight away.
        for image in ("model", "psf"):
            assert not (scored / "wsclean" / f"recon-{image}.fits").exists(), image
        # A record must not name a file this just deleted.
        assert "measurement_set" not in written["paths"] and "mat" not in written["paths"]
        assert written["paths"]["image"].endswith("recon-image.fits")
        assert written["paths"]["dirty"].endswith("recon-dirty.fits")
        assert json.loads((scored / "metrics.json").read_text())["paths"] == written["paths"]

        # A failed evaluation keeps its inputs: that is the case worth looking at.
        failed = evaluation(root, "eval-0002-failed")
        write_evaluation_record(failed, record_for(failed, error="wsclean failed with exit 1"))
        assert (failed / "sim.ms").exists() and (failed / "r2d2_data.mat").exists()
        assert (failed / "wsclean" / "recon-residual.fits").exists()

        saved = os.environ.get("NS_KEEP_MEASUREMENT_SETS")
        os.environ["NS_KEEP_MEASUREMENT_SETS"] = "1"
        try:
            kept = evaluation(root, "eval-0003-kept")
            written = write_evaluation_record(kept, record_for(kept))
            assert (kept / "sim.ms").exists() and (kept / "VLAA_ANT").exists()
            assert (kept / "wsclean" / "recon-psf.fits").exists()
            assert written["paths"]["measurement_set"].endswith("sim.ms")
        finally:
            if saved is None:
                del os.environ["NS_KEEP_MEASUREMENT_SETS"]
            else:
                os.environ["NS_KEEP_MEASUREMENT_SETS"] = saved

        # With a scratch directory the MS is built outside the evaluation, so a
        # scored one has nothing to delete and a failed one has to have its
        # inputs moved back beside its record before the scratch goes away.
        os.environ["NS_SCRATCH_DIR"] = str(root / "scratch")
        try:
            for name, extra in (("eval-0004-scored", {}), ("eval-0005-failed", {"error": "wsclean failed"})):
                eval_dir = root / name
                eval_dir.mkdir()
                scratch = evaluation_scratch_dir(eval_dir)
                assert scratch == root / "scratch" / name, scratch
                (scratch / "sim.ms").mkdir(parents=True)
                (scratch / "sim.ms" / "table.f0").write_text("data")
                (scratch / "VLAA_ANT").mkdir()
                (scratch / "r2d2_data.mat").write_text("mat")
                record = {"eval_id": 1, "params": {"a": 1}, "objective": 0.5,
                          "paths": {"eval_dir": str(eval_dir), "measurement_set": str(scratch / "sim.ms"),
                                    "mat": str(scratch / "r2d2_data.mat")}, **extra}
                written = write_evaluation_record(eval_dir, record)
                assert not scratch.exists(), f"{name} left its scratch behind"
                if extra:
                    assert (eval_dir / "sim.ms" / "table.f0").read_text() == "data"
                    assert (eval_dir / "VLAA_ANT").is_dir()
                    assert written["paths"]["measurement_set"] == str(eval_dir / "sim.ms")
                    assert written["paths"]["mat"] == str(eval_dir / "r2d2_data.mat")
                else:
                    assert not (eval_dir / "sim.ms").exists()
                    assert "measurement_set" not in written["paths"]
        finally:
            del os.environ["NS_SCRATCH_DIR"]
    print("evaluation artefact pruning self-check passed")


# Wall-clock epoch of the evaluation this rank is inside, or None between
# them. Per rank, because each rank is its own process, and single-valued
# because PolyChord calls the likelihood one evaluation at a time. It is what
# turns the profiler's unaccounted remainder from one bucket into three: with a
# start and an end on every record, the intervals say when *some* worker was
# evaluating and when none was. See summarize_profiling().
_EVALUATION_STARTED_EPOCH: float | None = None


def self_check_image_retention() -> None:
    import tempfile

    def record(i, objective, **extra):
        return {"eval_id": i, "objective": objective, "paths": {}, **extra}

    # 20 worst + 20 best + every 100th of the 960 between them.
    def keys(ids):
        return {f"eval_id:{i}" for i in ids}

    records = [record(i, i / 1000) for i in range(1000)]
    keep = evaluations_keeping_images(records, ends=20, stride=100)
    assert keep == keys(range(20)) | keys(range(980, 1000)) | keys(range(20, 980, 100)), keep
    assert len(keep) == 50, len(keep)

    # A run no bigger than both ends keeps everything.
    assert evaluations_keeping_images(records[:40], ends=20, stride=100) == keys(range(40))

    # A failure is kept whatever its rank, and is not ranked on an objective
    # it never produced.
    failed = records[:5] + [record(999, None, error="wsclean failed")]
    assert "eval_id:999" in evaluations_keeping_images(failed, ends=1, stride=100)

    # PolyChord reuses eval_id across parameter vectors, so two evaluations can
    # share one and must still be ranked apart.
    shared = [{"eval_id": 7, "objective": o,
               "paths": {"eval_dir": f"/run/evaluations/eval-0007-{h}"}}
              for o, h in ((0.1, "aaa"), (0.9, "bbb"))]
    assert evaluations_keeping_images(shared, ends=1, stride=100) == {
        "eval-0007-aaa", "eval-0007-bbb"}
    assert len({evaluation_key(r) for r in shared}) == 2

    with tempfile.TemporaryDirectory() as tmp:
        evaluations = Path(tmp) / "evaluations"
        made = []
        for i in range(60):
            eval_dir = evaluations / f"eval-{i:04d}-x"
            (eval_dir / "wsclean").mkdir(parents=True)
            paths = {"eval_dir": str(eval_dir)}
            for key, name in (("image", "recon-image.fits"), ("dirty", "recon-dirty.fits"),
                              ("residual", "recon-residual.fits")):
                (eval_dir / "wsclean" / name).write_text(name)
                paths[key] = str(eval_dir / "wsclean" / name)
            for name in PRUNED_EVALUATION_LOGS:
                (eval_dir / name).write_text(name)
            made.append({"eval_id": i, "objective": i / 60, "paths": paths})
        # A failure keeps every artefact whatever its rank.
        failed_dir = evaluations / "eval-0099-x"
        (failed_dir / "wsclean").mkdir(parents=True)
        for name in PRUNED_EVALUATION_LOGS:
            (failed_dir / name).write_text(name)
        made.append({"eval_id": 99, "objective": FAILURE_OBJECTIVE, "error": "wsclean failed",
                     "paths": {"eval_dir": str(failed_dir)}})

        removed = prune_run_artefacts(evaluations, made)
        keep = evaluations_keeping_images(made, ends=20, stride=100)
        # 60 scored evaluations, ends of 20: 40 at the ends plus every 100th of
        # the 20 between, which is just the first of them - and the failure,
        # which is kept whatever its rank.
        assert len(keep) == 42, len(keep)
        dropped = 60 - 41
        assert removed == dropped * (3 + len(PRUNED_EVALUATION_LOGS)), removed
        for rec in made[:60]:
            kept = evaluation_key(rec) in keep
            for key in RETAINED_IMAGE_KEYS:
                exists = Path(rec["paths"][key]).is_file() if key in rec["paths"] else False
                assert exists == kept, (rec["eval_id"], key, exists, kept)
                # A summary must never name a file this deleted.
                assert (key in rec["paths"]) == kept, (rec["eval_id"], key)

        # Logs go with the images, and a failure keeps all of them.
        for rec in made:
            eval_dir = Path(rec["paths"]["eval_dir"])
            kept = evaluation_key(rec) in keep or "error" in rec
            for name in PRUNED_EVALUATION_LOGS:
                assert (eval_dir / name).is_file() == kept, (rec["eval_id"], name)

    print("image retention self-check passed")


def self_check_clean_convergence() -> None:
    capped = ("Stopped on peak 1 mJy, because the minor-loop threshold was reached.\n"
              "Performed 40 iterations in total, 40 in this major iteration\n"
              "Stopped on peak -326 uJy, because maximum number of iterations was reached.\n"
              "Performed 100 iterations in total, 60 in this major iteration\n"
              "7 major iterations were performed.\n")
    got = clean_convergence(capped)
    # The last Stopped-on-peak line is the terminal one; the earlier minor-loop
    # exits are per-major-cycle, and `in total` is cumulative.
    assert got == {"clean_stop_reason": "max-iterations", "clean_iterations": 100,
                   "clean_major_iterations": 7}, got

    converged = ("Stopped on peak -326.54 uJy, because the threshold was reached.\n"
                 "Performed 84 iterations in total, 20 in this major iteration\n"
                 "5 major iterations were performed.\n")
    assert clean_convergence(converged) == {
        "clean_stop_reason": "threshold", "clean_iterations": 84,
        "clean_major_iterations": 5}, clean_convergence(converged)

    # A log that never got as far as cleaning yields nothing, not a crash.
    assert clean_convergence("WSClean version 3.7\n") == {}
    assert clean_convergence("") == {}
    assert clean_convergence_from(Path("/nonexistent/wsclean.stdout.log")) == {}

    print("clean convergence self-check passed")


def mark_evaluation_start() -> None:
    global _EVALUATION_STARTED_EPOCH
    _EVALUATION_STARTED_EPOCH = time.time()


def write_evaluation_record(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    global _EVALUATION_STARTED_EPOCH
    if _EVALUATION_STARTED_EPOCH is not None:
        timing = record.setdefault("timing", {})
        timing["started_epoch"] = _EVALUATION_STARTED_EPOCH
        timing["ended_epoch"] = time.time()
        _EVALUATION_STARTED_EPOCH = None
    prune_evaluation_artefacts(eval_dir, record)
    write_json_atomic(eval_dir / "metrics.json", record)
    return record


_SIMULATE_WORKERS: dict[str, "subprocess.Popen | FifoWorker"] = {}


# Set when a pooled worker had to be killed. Its FIFO pair died with it, so
# reconnecting can only find a corpse, and the ENXIO wait below would be pure
# delay in front of a retry that has to fall back to a rank-started worker
# anyway. Per rank, because each rank is its own process.
_FIFO_POOL_ABANDONED = False


def fifo_worker_pgrep_pattern(base: Path) -> str:
    return f"serve --fif[o] {base}$"


class FifoWorker:
    def __init__(self, write_fd: int, reply_path: Path, container: str, base: Path) -> None:
        self.stdin = os.fdopen(write_fd, "w")
        # Opening a FIFO blocks until the other end is open, so this must be the
        # same order serve() uses - request pipe first, reply pipe second.
        self.stdout = reply_path.open("r")
        self.container = container
        self.base = base

    def terminate(self) -> None:
        self.stdin.close()

    def kill(self) -> None:
        """Kill wedged worker and its meqserver inside the sidecar."""
        global _FIFO_POOL_ABANDONED
        _FIFO_POOL_ABANDONED = True
        pattern = fifo_worker_pgrep_pattern(self.base)
        subprocess.run(
            [
                "docker", "exec", self.container, "sh", "-c",
                f"p=$(pgrep -f {shlex.quote(pattern)}) || exit 0;"
                " kill -9 $(pgrep -P $p) $p 2>/dev/null || true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for stream in (self.stdin, self.stdout):
            try:
                stream.close()
            except OSError:
                pass


def _connect_shell_started_worker(fifo_dir_var: str, container: str) -> FifoWorker | None:
    """Attach to this rank's pre-warmed worker, or return None if unavailable."""
    fifo_dir = os.environ.get(fifo_dir_var)
    if not fifo_dir or _FIFO_POOL_ABANDONED:
        return None
    base = Path(fifo_dir) / str(mpi_rank())
    # Nonblocking open returns ENXIO until worker starts; timeout falls back.
    deadline = time.monotonic() + 10.0
    while True:
        try:
            write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            if time.monotonic() > deadline:
                return None
            time.sleep(0.002)
            continue
        os.set_blocking(write_fd, True)
        return FifoWorker(write_fd, Path(f"{base}.out"), container, base)


def simulate_worker(meqtrees_image: str, platform: str) -> subprocess.Popen | FifoWorker:
    """Keep one warm simulator worker per rank to avoid repeated startup cost."""
    if meqtrees_image not in _SIMULATE_WORKERS:
        worker = _connect_shell_started_worker(
            "NS_SIMULATE_FIFO_DIR", sidecar_container(meqtrees_image, platform)
        )
        if worker is None:
            repo_root = Path(os.environ.get("REPO_ROOT", os.getcwd()))
            worker = subprocess.Popen(
                [*sidecar_exec(meqtrees_image, platform, repo_root, interactive=True), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
        # The container itself is torn down by sidecar_container()'s own atexit
        # hook, which is registered first and so runs last.
        atexit.register(worker.terminate)
        _SIMULATE_WORKERS[meqtrees_image] = worker
    return _SIMULATE_WORKERS[meqtrees_image]


_R2D2_WORKERS: dict[str, "subprocess.Popen | FifoWorker"] = {}


def r2d2_worker(r2d2_image: str, platform: str, checkpoints_dir: str) -> "subprocess.Popen | FifoWorker":
    """Return this rank's long-lived R2D2 worker, creating it once per image."""
    if r2d2_image not in _R2D2_WORKERS:
        container = sidecar_container(r2d2_image, platform, ["-v", f"{checkpoints_dir}:/checkpoints:ro"])
        worker = _connect_shell_started_worker("NS_R2D2_FIFO_DIR", container)
        if worker is None:
            repo_root = Path(os.environ.get("REPO_ROOT", os.getcwd()))
            worker = subprocess.Popen(
                [
                    "docker", "exec", "--interactive",
                    *r2d2_docker_thread_env_flags(),
                    container,
                    "python3",
                    # Read live off the repo bind mount: the R2D2 image bakes in
                    # no copy of this repo's scripts, so nothing to rebuild.
                    str(repo_root / "scripts" / "lib" / "nested_sampling" / "r2d2_serve.py"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
        # The container itself is torn down by sidecar_container()'s own atexit
        # hook, which is registered first and so runs last.
        atexit.register(worker.terminate)
        _R2D2_WORKERS[r2d2_image] = worker
    return _R2D2_WORKERS[r2d2_image]


def run_r2d2_imaging(
    r2d2_image: str,
    platform: str,
    checkpoints_dir: str,
    argv: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> DockerRunResult:
    """Run one `imager.py` request, retrying worker deaths as `WORKER_DIED`."""
    request = {"argv": argv, "stdout": str(stdout_path), "stderr": str(stderr_path)}
    started = time.perf_counter()
    for attempt in worker_attempts():
        worker = r2d2_worker(r2d2_image, platform, checkpoints_dir)
        if not worker_send(worker.stdin, json.dumps(request) + "\n"):
            _R2D2_WORKERS.pop(r2d2_image, None)
            continue
        reply = worker_reply(worker.stdout, IMAGING_REPLY_TIMEOUT)
        if reply:
            answer = json.loads(reply)
            return DockerRunResult(
                returncode=answer["returncode"],
                wall_seconds=time.perf_counter() - started,
                peak_memory_bytes=answer["peak_memory_bytes"],
            )
        if reply is None:
            # ponytail: this kills the `docker exec`
            # client and leaves the worker wedged in the sidecar.
            worker.kill()
        # The worker died or went silent mid-request; drop it so the next
        # attempt starts a fresh one instead of inheriting the corpse.
        _R2D2_WORKERS.pop(r2d2_image, None)
    stderr_path.write_text(
        f"FATAL: r2d2 worker gave no reply, {len(WORKER_RETRY_DELAYS)} times\n"
    )
    return DockerRunResult(
        returncode=WORKER_DIED,
        wall_seconds=time.perf_counter() - started,
        peak_memory_bytes=0,
    )


def prewarm(*targets: Callable[[], None]) -> Callable[[], None]:
    """Start sidecar attachments concurrently and return a joiner."""
    threads = [threading.Thread(target=target, daemon=True) for target in targets]
    for thread in threads:
        thread.start()

    def join() -> None:
        for thread in threads:
            thread.join()

    return join


def simulate_worker_request(
    meqtrees_image: str,
    platform: str,
    request: dict[str, Any],
    stderr_path: Path,
) -> int:
    for attempt in worker_attempts():
        worker = simulate_worker(meqtrees_image, platform)
        if not worker_send(worker.stdin, json.dumps(request) + "\n"):
            _SIMULATE_WORKERS.pop(meqtrees_image, None)
            continue
        reply = worker_reply(worker.stdout, SIMULATE_REPLY_TIMEOUT)
        if reply:
            return int(json.loads(reply)["returncode"])
        if reply is None:
            worker.kill()
        _SIMULATE_WORKERS.pop(meqtrees_image, None)
    stderr_path.write_text(
        f"FATAL: simulate worker gave no reply, {len(WORKER_RETRY_DELAYS)} times\n"
    )
    return WORKER_DIED


def simulate_measurement_set(
    params: dict[str, Any],
    eval_dir: Path,
    meqtrees_image: str,
    platform: str,
) -> tuple[Path, list[str], subprocess.CalledProcessError | None]:
    eval_dir.mkdir(parents=True, exist_ok=False)
    scratch = evaluation_scratch_dir(eval_dir)
    if scratch is not None:
        # A restart that re-runs an evaluation whose scratch survived would hit
        # require_clean_output() in the simulator; the run owns this directory,
        # so clearing it is safe and normally costs an ENOENT.
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
    ms_path = (scratch or eval_dir) / "sim.ms"
    sim_stdout = eval_dir / "simulate.stdout.log"
    sim_stderr = eval_dir / "simulate.stderr.log"
    sim_cmd = [
        "--output-ms",
        str(ms_path),
        "--metadata-json",
        str(eval_dir / "simulation.json"),
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
    returncode = simulate_worker_request(
        meqtrees_image,
        platform,
        {"argv": sim_cmd, "stdout": str(sim_stdout), "stderr": str(sim_stderr)},
        sim_stderr,
    )
    if returncode != 0:
        return ms_path, sim_cmd, subprocess.CalledProcessError(returncode, sim_cmd)
    return ms_path, sim_cmd, None


def convert_ms_to_mat(
    argv: list[str],
    eval_dir: Path,
    meqtrees_image: str,
    platform: str,
) -> int:
    return simulate_worker_request(
        meqtrees_image,
        platform,
        {
            "action": "convert",
            "argv": argv,
            "stdout": str(eval_dir / "convert.stdout.log"),
            "stderr": str(eval_dir / "convert.stderr.log"),
        },
        eval_dir / "convert.stderr.log",
    )


def cube_like_from_theta(theta: np.ndarray) -> np.ndarray:
    specs = load_parameter_space()
    values = {str(spec["name"]): float(theta[i]) for i, spec in enumerate(specs)}
    cube_like = np.zeros(len(specs), dtype=np.float64)
    for i, spec in enumerate(specs):
        if spec.get("kind") == "band_start":
            cube_like[i] = start_frequency_cube_value(values[str(spec["name"])])
            continue
        lower = float(spec["min"])
        upper = float(spec["max"])
        cube_like[i] = (float(theta[i]) - lower) / (upper - lower)
        cube_like[i] = min(1.0, max(0.0, cube_like[i]))
    return cube_like


def prior_vector(cube: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [params[spec["name"]] if spec["name"] in params else math.log10(params["dynamic_range"]) for spec in load_parameter_space()],
        dtype=np.float64,
    )


def self_check_worker_timeout() -> None:
    """Silent workers are retried, then reported as WORKER_DIED."""
    import tempfile

    class Worker:
        def __init__(self, reply: str | None, broken_stdin: bool = False) -> None:
            read_fd, self._write_fd = os.pipe()
            self.stdout = os.fdopen(read_fd, "r")
            if broken_stdin:
                # A worker that died while nothing was talking to it: the pipe
                # is closed at the far end, so the next write raises rather
                # than the next read returning "".
                request_read, request_write = os.pipe()
                os.close(request_read)
                self.stdin = os.fdopen(request_write, "w")
            else:
                self.stdin = open(os.devnull, "w")
            self.killed = False
            if reply is not None:
                os.write(self._write_fd, reply.encode())

        def kill(self) -> None:
            self.killed = True

    # An empty read means the worker died; no read at all means it went silent.
    died = Worker("")
    os.close(died._write_fd)
    assert worker_reply(died.stdout, 5.0) == ""
    answered = Worker('{"returncode": 3}\n')
    assert worker_reply(answered.stdout, 5.0) == '{"returncode": 3}\n'
    assert worker_reply(Worker(None).stdout, 0.05) is None
    # And the death worker_reply cannot see: a worker that went before the
    # request was written to it at all.
    assert worker_send(Worker(None).stdin, "x\n") is True
    assert worker_send(Worker(None, broken_stdin=True).stdin, "x\n") is False

    original = {name: globals()[name] for name in ("simulate_worker", "WORKER_RETRY_DELAYS", "SIMULATE_REPLY_TIMEOUT")}
    workers: list[Worker] = []

    def spawn(reply: str | None) -> Any:
        def worker(*_args: Any, **_kwargs: Any) -> Worker:
            workers.append(Worker(reply))
            return workers[-1]

        return worker

    try:
        globals()["WORKER_RETRY_DELAYS"] = (0.0, 0.0)
        globals()["SIMULATE_REPLY_TIMEOUT"] = 0.05

        globals()["simulate_worker"] = spawn(None)
        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / "simulate.stderr.log"
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path) == WORKER_DIED
            assert "gave no reply" in stderr_path.read_text()
        # Every silent worker is killed rather than left holding its meqserver,
        # and each attempt gets a fresh one instead of the corpse.
        assert len(workers) == 2, len(workers)
        assert all(worker.killed for worker in workers)

        # The bound must not cost a worker that does answer its reply.
        workers.clear()
        globals()["simulate_worker"] = spawn('{"returncode": 7}\n')
        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / "simulate.stderr.log"
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path) == 7
            assert not stderr_path.exists()
        assert len(workers) == 1 and not workers[0].killed

        # A worker killed between two evaluations - the OOM killer taking an
        # idle 3.4GB R2D2 worker is the real shape of this - leaves a broken
        # pipe, so the write fails before the request is sent. That has to be
        # the same retry as a death mid-request: unhandled, the BrokenPipeError
        # unwound out of the likelihood and MPI_Abort ended the whole run on
        # the first attempt, with no retry made and no WORKER_DIED reported.
        workers.clear()

        def spawn_broken_then_answering(*_args: Any, **_kwargs: Any) -> Worker:
            first = not workers
            workers.append(Worker(None if first else '{"returncode": 7}\n',
                                  broken_stdin=first))
            return workers[-1]

        globals()["simulate_worker"] = spawn_broken_then_answering
        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / "simulate.stderr.log"
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path) == 7
            assert not stderr_path.exists()
        assert len(workers) == 2, len(workers)
        # Nothing to kill: it was already gone, which is why the write failed.
        assert not workers[0].killed
    finally:
        globals().update(original)

    print("worker timeout self-check passed")


def self_check_worker_pool_connect() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / str(mpi_rank())
        os.mkfifo(f"{base}.in")
        os.mkfifo(f"{base}.out")
        # O_RDWR is the one open mode a FIFO never blocks on, and it puts the
        # reader on .in and the writer on .out that a live worker would.
        ends = [os.open(f"{base}.in", os.O_RDWR), os.open(f"{base}.out", os.O_RDWR)]
        original = dict(os.environ)
        try:
            os.environ["NS_SIMULATE_FIFO_DIR"] = tmp
            os.environ["NS_R2D2_FIFO_DIR"] = tmp
            # The run script already started these; naming them here keeps the
            # check off docker entirely.
            _SIDECAR_CONTAINERS["meqtrees-self-check"] = "self-check"
            _SIDECAR_CONTAINERS["r2d2-self-check"] = "self-check"
            for worker in (
                simulate_worker("meqtrees-self-check", "linux/amd64"),
                r2d2_worker("r2d2-self-check", "linux/amd64", "/checkpoints"),
            ):
                assert isinstance(worker, FifoWorker), type(worker)
                atexit.unregister(worker.terminate)
        finally:
            for image in ("meqtrees-self-check", "r2d2-self-check"):
                _SIDECAR_CONTAINERS.pop(image, None)
            _SIMULATE_WORKERS.pop("meqtrees-self-check", None)
            _R2D2_WORKERS.pop("r2d2-self-check", None)
            os.environ.clear()
            os.environ.update(original)
            for fd in ends:
                os.close(fd)

    print("worker pool connect self-check passed")


def self_check_parameter_space() -> None:
    specs = load_parameter_space()
    assert len(specs) == 5, specs
    for spec in specs:
        assert spec["name"] in PARAMETER_TEX_LABELS, spec
        assert float(spec["min"]) < float(spec["max"]), spec
    # TOML keeps int and float apart, and cube_to_params rounds this one. Asserted
    # by shape rather than by value: the ranges in defaults.toml are meant to be
    # retuned, and pinning them here only means editing two files instead of one.
    channel_count = next(spec for spec in specs if spec["name"] == "channel_count")
    assert channel_count["kind"] == "integer", channel_count
    assert isinstance(channel_count["min"], int) and isinstance(channel_count["max"], int), channel_count
    assert channel_count["min"] >= 1, channel_count
    print("parameter space self-check passed")


def self_check_parameter_toggle() -> None:
    saved_off = os.environ.get("NS_DISABLE_PARAMS")
    saved_on = os.environ.get("NS_ENABLE_PARAMS")
    try:
        os.environ["NS_DISABLE_PARAMS"] = "source_offset_fraction"
        os.environ.pop("NS_ENABLE_PARAMS", None)
        load_parameter_space.cache_clear()
        specs = load_parameter_space()
        assert "source_offset_fraction" not in {spec["name"] for spec in specs}, specs
        assert len(specs) == len(load_all_parameter_specs()) - 1, specs

        # A disabled dimension is still one fewer cube dimension for
        # cube_to_params() to draw, and still a params key: pinned at its
        # `default`/`min`, here 0.0, so it round-trips to a centred source.
        raw: dict[str, Any] = {spec["name"]: float(spec.get("min", 0.0)) for spec in specs}
        fill_disabled_parameters(raw)
        assert raw["source_offset_fraction"] == 0.0, raw

        os.environ["NS_DISABLE_PARAMS"] = "channel_count"
        os.environ["NS_ENABLE_PARAMS"] = "channel_count"
        load_parameter_space.cache_clear()
        assert "channel_count" in {spec["name"] for spec in load_parameter_space()}, "enable wins over disable"
    finally:
        for var, saved in (("NS_DISABLE_PARAMS", saved_off), ("NS_ENABLE_PARAMS", saved_on)):
            if saved is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = saved
        load_parameter_space.cache_clear()
    print("parameter toggle self-check passed")


def self_check_source_offset() -> None:
    l_arcsec, m_arcsec = source_offset_to_lm(0.0, 1.4e9)
    assert l_arcsec == 0.0 and m_arcsec == 0.0, (l_arcsec, m_arcsec)

    l_arcsec, m_arcsec = source_offset_to_lm(0.35, 1.4e9)
    assert l_arcsec > 0.0 and m_arcsec > 0.0, (l_arcsec, m_arcsec)
    l_double_freq, _ = source_offset_to_lm(0.35, 2.8e9)
    # Doubling frequency doubles the baseline in wavelengths, which halves the
    # cell size and so the offset it corresponds to in arcsec.
    assert math.isclose(l_double_freq, l_arcsec / 2.0, rel_tol=1e-9), (l_double_freq, l_arcsec)
    l_half_fraction, _ = source_offset_to_lm(0.175, 1.4e9)
    assert math.isclose(l_half_fraction, l_arcsec / 2.0, rel_tol=1e-9), (l_half_fraction, l_arcsec)

    # A pixel scale big enough that a 1-pixel round trip is exact.
    header = {"CRPIX1": 65.0, "CRPIX2": 65.0, "CDELT1": -1.0 / 3600.0, "CDELT2": 1.0 / 3600.0}
    cx, cy = 64, 64
    assert source_pixel(header, cx, cy, 0.0, 0.0, 128, 128) == (cx, cy)
    sx, sy = source_pixel(header, cx, cy, 3.0, 5.0, 128, 128)
    assert (sx, sy) == (cx - 3, cy + 5), (sx, sy)

    # Pinned against an actual WSClean image (a real header, a real 1 Jy
    # point source at this exact offset), not just the FITS/WCS formula: the
    # `l` sign was wrong here once - source_pixel() agreed with itself but not
    # with where WSClean actually put the source - and only running a real
    # search caught it, not this self-check. See the PR that added this line.
    real_header = {
        "CRPIX1": 65.0, "CRPIX2": 65.0,
        "CDELT1": -0.000376683333333333, "CDELT2": 0.000376683333333333,
    }
    sx, sy = source_pixel(real_header, 64, 64, 6.87503877002526, 11.907916453689596, 128, 128)
    assert (sx, sy) == (59, 73), (sx, sy)

    # R2D2 writes no WCS at all, so compute_image_metrics() has to supply one.
    # Both cases below are real R2D2 reconstructions from a search with
    # `source_offset_fraction` enabled: the (l, m) and max projected baseline
    # the simulator recorded, against the pixel the brightest pixel of the
    # reconstruction was actually measured at. This is the check that would
    # have caught the missing header, and the one that pins the l-axis sign
    # and the centre pixel for R2D2 the way `real_header` above pins WSClean.
    import tempfile

    from astropy.io import fits

    for l_as, m_as, max_proj_baseline_lambda, expected in (
        (20.529971838368628, 35.5589543020127, 32152.12622557544, (54, 81)),
        (12.833132485007466, 22.22763748429558, 57746.84392542726, (53, 83)),
    ):
        scale = image_pixel_size_arcsec(max_proj_baseline_lambda)
        image = np.zeros((DEFAULT_IMAGE_DIM, DEFAULT_IMAGE_DIM))
        image[expected[1], expected[0]] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            # PrimaryHDU alone writes what R2D2 writes: SIMPLE/BITPIX/NAXIS
            # and nothing else. No CRPIX either, so this also pins the centre.
            path = Path(tmp) / "r2d2_model_image.fits"
            fits.PrimaryHDU(image).writeto(path)
            metrics = compute_image_metrics(
                path, 1.0, 0.0, 0,
                source_l_arcsec=l_as,
                source_m_arcsec=m_as,
                pixel_size_arcsec=scale,
            )
            # The 1 Jy spike sits exactly where the source was, so a metric
            # that located it sees a perfect image; one pixel out and the
            # residual carries the whole 1 Jy twice over.
            assert metrics["peak_flux_abs_error_jy"] == 0.0, (l_as, m_as, metrics["peak_flux_abs_error_jy"])
            assert metrics["total_rms_jy"] == 0.0, (l_as, m_as, metrics["total_rms_jy"])
            assert metrics["relative_l2_error"] == 0.0, metrics["relative_l2_error"]

            # A header with no WCS and no scale to stand in for it must say so
            # rather than quietly centring the source and scoring a good image
            # as a catastrophic one, which is what makes such a bug invisible.
            try:
                compute_image_metrics(path, 1.0, 0.0, 0, source_l_arcsec=l_as, source_m_arcsec=m_as)
            except ValueError:
                pass
            else:
                raise AssertionError("a header-less image with no pixel_size_arcsec must raise")

    # A cube with everything at 0.5 fixes the offset fraction at its box's
    # midpoint; band_start's own resolution can move start_frequency_hz, so
    # only the sign and non-zero-ness of the offset are pinned here. Forced on
    # rather than read from defaults.toml: this dimension is toggled off and on
    # between runs, and the arithmetic above it still has to be right on the
    # runs that enable it.
    saved_on = os.environ.get("NS_ENABLE_PARAMS")
    try:
        os.environ["NS_ENABLE_PARAMS"] = "source_offset_fraction"
        load_parameter_space.cache_clear()
        n = len(load_parameter_space())
        params = cube_to_params(np.full(n, 0.5))
        assert params["source_l_arcsec"] != 0.0 or params["source_m_arcsec"] != 0.0, params
    finally:
        if saved_on is None:
            os.environ.pop("NS_ENABLE_PARAMS", None)
        else:
            os.environ["NS_ENABLE_PARAMS"] = saved_on
        load_parameter_space.cache_clear()
    print("source offset self-check passed")


def self_check_spectral_window() -> None:
    bands = load_receiver_bands()
    assert bands and all(float(b["min"]) < float(b["max"]) for b in bands), bands

    specs = load_parameter_space()
    index = next(i for i, spec in enumerate(specs) if spec.get("kind") == "band_start")
    count_spec = next(spec for spec in specs if spec["name"] == "channel_count")
    width_spec = next(spec for spec in specs if spec["name"] == "channel_width_hz")
    count_min, width_min = channel_floors()

    before = WINDOW_FIT_STATS.as_dict()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        cube = rng.random(len(specs))
        params = cube_to_params(cube, track=True)
        start = params["start_frequency_hz"]
        count, width = params["channel_count"], params["channel_width_hz"]
        band = next(
            (b for b in bands if float(b["min"]) <= start and start + count * width <= float(b["max"])),
            None,
        )
        assert band is not None, params
        seen.add(band["name"])
        # Fitting only ever gives ground: it narrows channels or drops them,
        # and never past the floors the parameter space set.
        assert width_min <= width <= float(width_spec["max"]), params
        assert count_min <= count <= int(count_spec["max"]), params
        # theta -> cube -> params is the sampler's own round trip; a fitted
        # window has to be a fixed point of the fitting, or the likelihood
        # would see different parameters from the ones PolyChord stored.
        theta = prior_vector(cube, params)
        again = cube_to_params(cube_like_from_theta(theta))
        assert again.keys() == params.keys()
        for name, value in params.items():
            other = again[name]
            assert isinstance(value, float) and math.isclose(value, other, rel_tol=1e-9) or value == other, (name, value, other)
    assert len(seen) == len(bands), sorted(seen)

    # The tally adds up, and only the tracked draws above were counted.
    after = WINDOW_FIT_STATS.as_dict()
    counted = {key: after[key] - before[key] for key in after}
    assert counted["draws"] == 2000, counted
    assert counted["as_sampled"] + counted["width_reduced"] + counted["count_reduced"] == 2000, counted
    assert counted["redraws"] >= counted["redrawn_draws"], counted
    assert counted["seconds"] > 0.0, counted

    # The bottom of the cube is the bottom of the lowest band, where a window
    # always fits as sampled; the top is the very top of the highest band,
    # where nothing does, so it is redrawn into a band that has room.
    edges = np.zeros(len(specs))
    assert cube_to_params(edges)["start_frequency_hz"] == float(bands[0]["min"])
    edges[index] = 1.0
    top = cube_to_params(edges)
    top_stop = top["start_frequency_hz"] + top["channel_count"] * top["channel_width_hz"]
    assert any(float(b["min"]) <= top["start_frequency_hz"] and top_stop <= float(b["max"]) for b in bands)

    # Each rung of the ladder, forced directly: a start frequency with room for
    # the window, for a narrowed one, for the minimum width with fewer
    # channels, and for nothing at all.
    band = bands[-1]
    top_of_band = float(band["max"])
    for room, count, width, expect in (
        (1e9, 4, 2.0e6, (4, 2.0e6)),
        (4 * width_min, 4, 2.0e6, (4, width_min)),
        (count_min * width_min, 4, 2.0e6, (count_min, width_min)),
    ):
        cube_value = start_frequency_cube_value(top_of_band - room)
        start, got_count, got_width = fit_spectral_window(cube_value, count, width)
        assert (got_count, got_width) == expect or math.isclose(got_width, expect[1], rel_tol=1e-9), (
            room, got_count, got_width, expect
        )
        assert start + got_count * got_width <= top_of_band + 1e-3, (start, got_count, got_width)

    # No room at all: the start frequency is thrown away and another drawn.
    start, count, width = fit_spectral_window(start_frequency_cube_value(top_of_band), 4, 2.0e6)
    assert any(float(b["min"]) <= start and start + count * width <= float(b["max"]) for b in bands)
    assert count >= count_min and width >= width_min

    # A box whose smallest window fits no band cannot be fitted at all, and is
    # fatal at load rather than mid-run.
    try:
        check_channel_box_against_bands(
            [{"name": "channel_count", "min": 32, "max": 64}, {"name": "channel_width_hz", "min": 1.0e9, "max": 2.0e9}]
        )
    except SystemExit as error:
        assert "no start frequency can hold it" in str(error), error
    else:
        raise AssertionError("a box whose smallest window fits no band should not load")


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
            "-e",
            "OMP_WAIT_POLICY=PASSIVE",
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


def self_check_fits_reader() -> None:
    import tempfile

    from astropy.io import fits

    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 1, 8, 6)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = ("JY/BEAM", "Units are in Jansky per beam")
    hdu.header["CRPIX1"] = 4.0
    hdu.header["CRPIX2"] = 5.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.fits"
        hdu.writeto(path)
        image, header = load_fits_2d(path)
        expected, expected_header = fits.getdata(path, header=True)
    assert image.shape == (8, 6), image.shape
    assert np.array_equal(image, np.squeeze(np.asarray(expected, dtype=np.float64)))
    # compute_image_metrics writes one pixel of this array to form the residual.
    assert image.flags.writeable
    assert header["BUNIT"] == "JY/BEAM", header["BUNIT"]
    assert header["CRPIX1"] == 4.0 and header["CRPIX2"] == 5.0
    assert header["SIMPLE"] is True
    assert int(header["NAXIS"]) == int(expected_header["NAXIS"])


def self_check_lazy_numpy() -> None:
    """Importing this module must defer numpy until `np` is used."""
    probe = (
        "import sys, common;"
        "assert 'numpy' not in sys.modules, 'common imported numpy eagerly';"
        "assert common.np.array([1.0, 2.0]).sum() == 3.0;"
        "assert 'numpy' in sys.modules;"
        "assert common.np is sys.modules['numpy'], 'np was not rebound'"
    )
    subprocess.run(
        [sys.executable, "-c", probe], cwd=str(Path(__file__).parent), check=True
    )


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
    residual = np.array([1.0, -2.0, 3.0])
    dirty = np.array([4.0, 5.0, -6.0])
    assert sigma_res(residual, dirty) == np.linalg.norm(residual) / np.linalg.norm(dirty)
    assert off_source_mask((8, 8), 4, 4) is off_source_mask((8, 8), 4, 4)
    # One cached mask is handed to every caller, so none of them may write it.
    assert not off_source_mask((8, 8), 4, 4).flags.writeable

    expr_fn, _ = resolve_metric("log_snr + 0.1 * wall_seconds")
    assert expr_fn(sample) == sample["log_snr"] + 0.1 * sample["wall_seconds"]

    for invalid in ("not_a_metric", "snr + unknown", "snr ++", "__import__('os').system('id')"):
        try:
            resolve_metric(invalid)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected SystemExit for invalid metric {invalid!r}")


def self_check_image_pixel_size() -> None:
    """Check derived pixel size against R2D2's upstream expression."""
    speed_of_light = 299792458.0
    baseline_m = np.array([120.0, 36400.0, 4800.0])
    freqs_hz = np.array([1.0e9, 1.4e9])

    u = np.concatenate([baseline_m / (speed_of_light / f) for f in freqs_hz])
    v = np.zeros_like(u)
    max_proj_baseline = float(np.max(np.sqrt(u**2 + v**2)))
    upstream = (180.0 / math.pi) * 3600.0 / (DEFAULT_SUPER_RESOLUTION * 2 * max_proj_baseline)

    # What the simulator records: the longest baseline scaled to the top channel.
    recorded = float(np.max(baseline_m)) * float(freqs_hz.max()) / speed_of_light
    assert abs(image_pixel_size_arcsec(recorded) - upstream) < 1e-12, (
        f"{image_pixel_size_arcsec(recorded)} != {upstream}"
    )

    # A finer cell for a longer baseline, and inversely proportional to it.
    assert image_pixel_size_arcsec(2.0 * recorded) < image_pixel_size_arcsec(recorded)
    assert abs(image_pixel_size_arcsec(2.0 * recorded) * 2.0 - image_pixel_size_arcsec(recorded)) < 1e-12

    for bad in (0.0, -1.0):
        try:
            image_pixel_size_arcsec(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected SystemExit for max projected baseline {bad!r}")
    print("image pixel size self-check passed")


def self_check_backfilled_intervals() -> None:
    import os
    import tempfile

    def run_dir(root: Path, ends: list[float], length: float = 1.0) -> dict[str, Any]:
        evaluations = []
        for index, end in enumerate(ends):
            eval_dir = root / f"eval-{index:04d}-abc"
            eval_dir.mkdir()
            record = eval_dir / "metrics.json"
            record.write_text("{}")
            os.utime(record, (end, end))
            evaluations.append({
                "paths": {"eval_dir": str(eval_dir)},
                "timing": {"image_container_seconds": length},
            })
        return {
            "evaluations": evaluations,
            "profiling": {
                "mpi_procs": 3, "total_wall_seconds": 10.0,
                "stage_totals_seconds": {"image_container": length * len(ends)},
                "stage_eval_counts": {"image_container_seconds": len(ends)},
                "accounted_worker_seconds": length * len(ends),
            },
        }

    # Two workers, four one-second evaluations in the last four seconds of a
    # ten-second run: 4s of the 20s budget inside an evaluation, 6s of wall
    # clock with nothing in flight (12s across the two workers), and the 4s
    # left over is one worker waiting for the other.
    with tempfile.TemporaryDirectory() as raw:
        summary = run_dir(Path(raw), [1006.0, 1007.0, 1009.0, 1010.0])
        filled = backfill_busy_seconds(summary)
        assert abs(filled["busy_worker_seconds"] - 4.0) < 1e-6, filled
        assert abs(filled["busy_wall_seconds"] - 4.0) < 1e-6, filled
        split = profiling_breakdown(filled)
        assert [(row["key"], round(row["seconds"], 3)) for row in split["remainder_rows"]] == [
            ("polychord", 12.0), ("idle", 4.0)], split["remainder_rows"]
        # Reconstructed intervals are the stages themselves, so the harness row
        # - which would be nothing but the float noise of the subtraction -
        # is not among them.
        assert "harness" not in [row["key"] for row in split["rows"]], split["rows"]
        assert PROFILING_UNSPLIT_NOTE not in split["note"]

    # Directories copied or restored all carry the time of the copy, and no
    # timeline can be read out of them: four worker-seconds of imaging cannot
    # fit in the 0.01s of wall clock these mtimes claim. Left unsplit.
    with tempfile.TemporaryDirectory() as raw:
        summary = run_dir(Path(raw), [1010.0, 1010.0, 1010.005, 1010.01])
        filled = backfill_busy_seconds(summary)
        assert "busy_worker_seconds" not in filled, filled
        unsplit = profiling_breakdown(filled)
        assert [row["label"] for row in unsplit["remainder_rows"]] == [UNACCOUNTED_LABEL]
        assert PROFILING_UNSPLIT_NOTE in unsplit["note"]

    # An evaluation whose directory has gone: its time would be charged to
    # PolyChord, which is worse than not splitting at all.
    with tempfile.TemporaryDirectory() as raw:
        summary = run_dir(Path(raw), [1006.0, 1007.0, 1009.0, 1010.0])
        (Path(summary["evaluations"][0]["paths"]["eval_dir"]) / "metrics.json").unlink()
        assert "busy_worker_seconds" not in backfill_busy_seconds(summary)

    # A run that stamped its own intervals is never second-guessed.
    with tempfile.TemporaryDirectory() as raw:
        summary = run_dir(Path(raw), [1010.0, 1010.0, 1010.0, 1010.0])
        summary["profiling"]["busy_worker_seconds"] = 4.0
        summary["profiling"]["busy_wall_seconds"] = 2.0
        assert backfill_busy_seconds(summary)["busy_wall_seconds"] == 2.0
    print("backfilled evaluation intervals self-check passed")


def self_check_profiling() -> None:
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
    assert empty_profiling["accounted_worker_seconds"] == 0.0
    assert empty_profiling["polychord_overhead_seconds"] == 5.0

    mpi_profiling = summarize_profiling(evaluations, total_wall_seconds=10.0, mpi_procs=4)
    assert mpi_profiling["accounted_worker_seconds"] == 19.5
    assert mpi_profiling["accounted_seconds"] is None
    assert mpi_profiling["polychord_overhead_seconds"] is None

    assert format_duration(None) == "n/a"
    assert format_duration(0.0) == "0ms"
    assert format_duration(0.000143) == "143us"
    assert format_duration(0.0005) == "500us"
    assert format_duration(0.0474) == "47ms"
    assert format_duration(0.9996) == "1.00s"
    assert format_duration(1.4351) == "1.44s"
    assert format_duration(33.03) == "33.0s"
    assert format_duration(455.58) == "7m 36s"
    assert format_duration(3600 + 5 * 60 + 9) == "1h 05m 09s"
    assert format_share(None) == ""
    assert format_share(0.6462) == "64.6%"
    assert format_share(0.0001) == "<0.1%"
    assert format_share(0.0) == "0.0%"

    serial = profiling_breakdown(profiling, algorithm="wsclean")
    assert serial["worker_procs"] == 1
    assert serial["worker_seconds_budget"] == 25.0
    assert serial["evals"] == 3
    labels = [row["label"] for row in serial["rows"]]
    assert "wsclean container (total)" in labels and "of which: wsclean itself" in labels
    assert "convert (MS -> .mat)" not in labels
    container = next(row for row in serial["rows"] if row["key"] == "image_container")
    assert container["seconds"] == 15.0 and container["evals"] == 3
    assert abs(container["per_eval_seconds"] - 5.0) < 1e-9
    assert abs(container["share"] - 0.6) < 1e-9
    assert abs(serial["remainder_rows"][0]["seconds"] - 5.5) < 1e-9
    top_level = sum(row["share"] for row in serial["rows"] if not row["is_sub"])
    assert abs(top_level + serial["remainder_rows"][0]["share"] - 1.0) < 1e-9

    assert (worker_procs(1), worker_procs(2), worker_procs(15)) == (1, 1, 14)
    mpi = profiling_breakdown(mpi_profiling, algorithm="r2d2")
    assert mpi["worker_procs"] == 3
    assert mpi["worker_seconds_budget"] == 30.0  # not 40.0: rank 0 is not a worker
    assert abs(mpi["subtotal_share"] - 0.65) < 1e-9
    assert abs(mpi["remainder_rows"][0]["seconds"] - 10.5) < 1e-9
    assert mpi["rows"][0]["label"] == "simulate (MeqTrees)"

    oversubscribed = profiling_breakdown(summarize_profiling(evaluations, total_wall_seconds=1.0, mpi_procs=2))
    assert oversubscribed["worker_seconds_budget"] == 19.5
    assert oversubscribed["remainder_rows"][0]["seconds"] == 0.0
    assert oversubscribed["rows"][0]["label"] == "simulate (MeqTrees)"

    for unsplit in (serial, mpi):
        assert [row["label"] for row in unsplit["remainder_rows"]] == [UNACCOUNTED_LABEL]
        assert unsplit["subtotal_label"] == "accounted (sum of stages above)"
        assert "harness" not in [row["key"] for row in unsplit["rows"]]
    assert serial["equation_terms"] == ["19.5s accounted", "+ 5.50s unaccounted"]

    timed = [
        {"timing": {"simulate_seconds": 1.0, "image_container_seconds": 2.0, "metrics_seconds": 0.5,
                    "started_epoch": 1001.0, "ended_epoch": 1005.0}},
        {"timing": {"simulate_seconds": 0.5, "image_container_seconds": 1.0, "metrics_seconds": 0.5,
                    "started_epoch": 1001.0, "ended_epoch": 1003.0}},
        {"timing": {"started_epoch": 900.0, "ended_epoch": 950.0}},
    ]
    split = profiling_breakdown(summarize_profiling(
        timed, total_wall_seconds=10.0, mpi_procs=3, run_started_epoch=1000.0))
    assert split["worker_seconds_budget"] == 20.0
    assert abs(split["busy_worker_seconds"] - 6.0) < 1e-9, split["busy_worker_seconds"]
    assert split["rows"][-1]["key"] == "harness" and split["rows"][-1]["seconds"] == 0.5
    assert abs(split["subtotal_seconds"] - 6.0) < 1e-9
    assert [(row["key"], round(row["seconds"], 6)) for row in split["remainder_rows"]] == [
        ("polychord", 12.0), ("idle", 2.0)], split["remainder_rows"]
    assert abs(sum(row["share"] for row in split["rows"] if not row["is_sub"])
               + sum(row["share"] for row in split["remainder_rows"]) - 1.0) < 1e-9
    assert split["equation_terms"] == ["6.00s evaluating", "+ 12.0s PolyChord", "+ 2.00s idle"]
    assert split["total_label"] == "end-to-end (evaluating + PolyChord + idle)"

    anchored = profiling_breakdown(summarize_profiling(timed[:2], total_wall_seconds=4.0, mpi_procs=3))
    assert abs(anchored["busy_wall_seconds"] - 4.0) < 1e-9, anchored["busy_wall_seconds"]
    assert anchored["remainder_rows"][0]["seconds"] == 0.0

    self_check_backfilled_intervals()

    degenerate = profiling_breakdown({"mpi_procs": 1, "total_wall_seconds": 0.0, "stage_totals_seconds": {}})
    assert degenerate["worker_seconds_budget"] is None
    assert degenerate["remainder_rows"] == []
    assert degenerate["rows"] == []
