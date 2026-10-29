#!/usr/bin/env python3
from __future__ import annotations

import atexit
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
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


# Reply bounds turn silent-worker deadlocks into recoverable failures. The two
# simulate bounds scale with the Measurement Set the request asks for, because
# the MeqTrees predict they cover is linear in makems' NTimes - measured at
# 0.0015s per time sample and flat in channels - so a scalar that suits the 9
# time samples of a default run is two orders of magnitude too tight for the
# 1200 an `integration_seconds` search can ask for, and turns a merely slow
# predict into a wedge, a dead worker, and a run that cannot restart past that
# one point. Rationale and measurements live in docs/robustness.md.
PREDICT_BASE_SECONDS = 3.0
PREDICT_SECONDS_PER_TIME_SAMPLE = 0.0015
# The predict shares its host with 19 other ranks and an imager, so it gets an
# order of magnitude over the idle measurement before it is called wedged.
# Lower this and a loaded host starts reporting healthy predicts as deaths.
PREDICT_CONTENTION_MARGIN = 10.0
# What a request costs the worker either side of the predict. The makems
# skeleton is the part that scales - measured at 0.0022s per time sample, and it
# gets the same contention margin as the predict - and the base covers the rest:
# the FIFO round trip, the noise, and writing the DATA column.
#
# 30s rather than the 10s this replaces because a search that moves
# `declination_deg` or `integration_seconds` changes the skeleton cache key on
# every draw - both are in the makems config the key is taken from - so every
# evaluation now runs a real makems inside this bound instead of copying a
# cached one. Being generous here is nearly free: the bound is only ever spent
# in full by a worker that has genuinely died.
SIMULATE_BASE_SECONDS = 30.0
SKELETON_SECONDS_PER_TIME_SAMPLE = 0.0022
# The worker replaces its meqserver once and retries, so the rank has to sit
# through two of its predict bounds before deciding the worker is gone.
WORKER_PREDICT_ATTEMPTS = 2
SHELL_REPLY_TIMEOUT = 300.0
IMAGING_REPLY_TIMEOUT = 3600.0


def evaluation_time_samples(params: dict[str, Any]) -> int:
    """makems' `NTimes` for these parameters - what both simulate bounds scale on."""
    return max(1, math.ceil(float(params["observation_minutes"]) * 60.0 / float(params["integration_seconds"])))


def predict_wait_seconds(n_times: int = 0) -> float:
    """How long the worker waits on its meqserver before calling it wedged."""
    return PREDICT_BASE_SECONDS + PREDICT_CONTENTION_MARGIN * PREDICT_SECONDS_PER_TIME_SAMPLE * n_times


@cache
def observation_horizon_minutes() -> float:
    """The longest observation the search can draw, 0.0 when that is pinned:
    the length the simulator caches an unseen observation at."""
    spec = next((spec for spec in load_parameter_space() if spec["name"] == "observation_minutes"), None)
    return float(spec["max"]) if spec else 0.0


def simulate_reply_timeout(n_times: int = 0) -> float:
    """How long the rank waits on the worker: its two predict attempts, and the rest.

    Derived from what the worker actually does with a request rather than from a
    multiplier, so the ladder cannot drift: whatever the worker is allowed to
    spend on its two predicts, the rank is already waiting longer than that.

    `n_times` defaults to the smallest Measurement Set, which is also the right
    answer for the R2D2 convert requests that share this bound and have no
    predict in them at all.
    """
    return (
        SIMULATE_BASE_SECONDS
        + WORKER_PREDICT_ATTEMPTS * predict_wait_seconds(n_times)
        + PREDICT_CONTENTION_MARGIN * SKELETON_SECONDS_PER_TIME_SAMPLE * n_times
    )


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
    drain_record_writes(timeout=30.0)
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


def nominal_pixel_size_arcsec(start_frequency_hz: float) -> float:
    """Image scale the source position is placed against.

    Nominal: VLA-A's maximum baseline, not the one the simulator records - the
    position has to reach simulate_point_source_ms.py before the MS exists.
    """
    return image_pixel_size_arcsec(VLA_A_MAX_BASELINE_M * start_frequency_hz / SPEED_OF_LIGHT_M_S)


def source_offset_to_lm(fraction: float, start_frequency_hz: float) -> tuple[float, float]:
    half_width_arcsec = nominal_pixel_size_arcsec(start_frequency_hz) * (image_dim() / 2.0)
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
def image_dim() -> int:
    """Pixels on a side of the image both imagers make.

    defaults.toml is the only place the number is written down - a default
    here as well would be a second one, and the two would disagree the first
    time either moved. `NS_IMAGE_DIM` still wins, because that is the contract
    every other key in defaults.toml is read under (scripts/lib/defaults.sh),
    and it is what `--interleave NS_IMAGE_DIM 128 32` measures with.
    """
    defaults = load_defaults()
    if "NS_IMAGE_DIM" not in defaults:
        raise SystemExit("FATAL: defaults.toml defines no NS_IMAGE_DIM, so there is no image size")
    return int(os.environ.get("NS_IMAGE_DIM") or defaults["NS_IMAGE_DIM"])


@cache
def load_receiver_bands() -> list[dict[str, Any]]:
    bands = [band for band in load_defaults()["receiver_band"] if band.get("enabled", True)]
    if not bands:
        raise SystemExit("defaults.toml: no enabled [[receiver_band]]")
    return sorted(bands, key=lambda band: float(band["min"]))


@cache
def load_all_parameter_specs() -> list[dict[str, Any]]:
    specs = load_defaults()["parameter_space"]
    for spec in specs:
        if "default" not in spec:
            raise SystemExit(
                f"defaults.toml: parameter_space '{spec.get('name', '?')}' has no default"
            )
        # A pixel box would mean a different fraction of the sky each time
        # NS_IMAGE_DIM moved. Here, not in load_parameter_space(), so a
        # disabled dimension still reports its box.
        if spec.get("kind") == "image_pixels":
            reach = (image_dim() / 2.0) * float(spec["fraction"])
            spec["min"], spec["max"] = -reach, reach
    return specs


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
    check_source_position_dimensions(specs)
    return specs


def check_source_position_dimensions(specs: list[dict[str, Any]]) -> None:
    names = {str(spec["name"]) for spec in specs}
    pixels = sorted(names & {"source_l_pixels", "source_m_pixels"})
    if "source_offset_fraction" in names and pixels:
        raise SystemExit(
            f"source_offset_fraction and {', '.join(pixels)} are enabled together, and "
            "their offsets add - so one sky position is reachable from many draws and "
            "PolyChord would spend a dimension on a direction the likelihood is flat "
            "along. Enable the cartesian pair or the polar dimension, not both: "
            "`--disable-param source_offset_fraction`, or `--disable-param "
            "source_l_pixels --disable-param source_m_pixels`."
        )


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
    "source_l_pixels": r"l\,[\mathrm{px}]",
    "source_m_pixels": r"m\,[\mathrm{px}]",
    "declination_deg": r"\delta\,[\mathrm{deg}]",
    "integration_seconds": r"\tau_{\mathrm{int}}\,[\mathrm{s}]",
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
    base = max(1, int(override)) if override else os.cpu_count() or 1
    # Spread the job's CPUs over the evaluations in flight. A run's final
    # chain otherwise images alone on one thread while every other core idles:
    # the last ~30 of the 71k-evaluation run's 289 minutes (docs/csd3-speed.md, round 9).
    ceiling = int(os.environ.get("R2D2_MAX_THREADS") or 0)
    busy_dir = _busy_dir()
    if ceiling <= base or busy_dir is None:
        return base
    try:
        busy = len(os.listdir(busy_dir))
    except FileNotFoundError:
        busy = 0
    return max(base, ceiling // max(1, busy))


def fill_disabled_parameters(raw: dict[str, Any]) -> None:
    enabled_names = {spec["name"] for spec in load_parameter_space()}
    for spec in load_all_parameter_specs():
        name = spec["name"]
        if name in enabled_names or name in raw:
            continue
        raw[name] = spec["default"]


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
    # Keep the offset dimensions for prior_vector()'s round-trip check. Polar
    # and cartesian offsets add; each is zero at its own default.
    pixel_arcsec = nominal_pixel_size_arcsec(raw["start_frequency_hz"])
    offset_l, offset_m = source_offset_to_lm(
        raw["source_offset_fraction"], raw["start_frequency_hz"]
    )
    raw["source_l_arcsec"] = offset_l + raw["source_l_pixels"] * pixel_arcsec
    raw["source_m_arcsec"] = offset_m + raw["source_m_pixels"] * pixel_arcsec
    return raw


def params_key(params: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def stable_seed(global_seed: int, key: str) -> int:
    return (global_seed + int(key[:8], 16)) % (2**31 - 1)


def sidecar_command(image: str, prefix: list[str] | None = None) -> list[str]:
    """The WSClean image's entrypoint, as the zygote wants argv[0] spelled."""
    return [*(prefix or []), "wsclean"]


ZYGOTE_COMMAND = "wsclean-zygote"

_ZYGOTE_WORKERS: dict[str, "FifoWorker"] = {}


def zygote_worker(image: str) -> "FifoWorker":
    """This rank's WSClean fork server, one of the pool the run script started."""
    if image not in _ZYGOTE_WORKERS:
        worker = _connect_shell_started_worker("NS_WSCLEAN_FIFO_DIR")
        if worker is None:
            raise WorkerDied(f"no {ZYGOTE_COMMAND} pool for rank {mpi_rank()} under NS_WSCLEAN_FIFO_DIR")
        atexit.register(worker.terminate)
        _ZYGOTE_WORKERS[image] = worker
    return _ZYGOTE_WORKERS[image]


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
        zygote = zygote_worker(image)
        if not worker_send(zygote.stdin, request):
            _forget(_ZYGOTE_WORKERS, image)
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
        _forget(_ZYGOTE_WORKERS, image)
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
            ended = evaluation_record_file(Path(eval_dir)).stat().st_mtime
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
    ("simulate", "simulate_seconds", "simulate (visibilities + noise)", False),
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


# An evaluation's record is one file, eval-NNNN-xxx.json, beside the eval dir
# that holds only what it keeps: a dir plus its metrics.json was two inodes an
# evaluation, and the file quota, not CPU time, capped a run (round 91). Runs
# from before round 92 have eval-NNNN-xxx/metrics.json, which is still read.
RECORD_SUFFIX = ".json"


def flat_record_path(eval_dir: Path) -> Path:
    return eval_dir.with_name(eval_dir.name + RECORD_SUFFIX)


def evaluation_record_paths(evaluations_dir: Path) -> dict[str, Path]:
    """{eval dir name: its record path} for every evaluation an entry names,
    from one listing: a flat record if there is one, else the dir's metrics.json
    (which may not exist: an evaluation in flight, or one a killed rank left)."""
    try:
        names = os.listdir(evaluations_dir)
    except FileNotFoundError:
        return {}
    paths: dict[str, Path] = {}
    for name in names:
        if not name.startswith("eval-") or name.endswith(".partial"):
            continue
        if name.endswith(RECORD_SUFFIX):
            paths[name[:-len(RECORD_SUFFIX)]] = evaluations_dir / name
        else:
            paths.setdefault(name, evaluations_dir / name / "metrics.json")
    return paths


# Records packed into one file, a `name<TAB>record json` line each, so that a
# finished run is one inode of records rather than one an evaluation, and a
# resumed run carries only its last segment's as files (round 93). Packed when
# a run finishes and when it resumes, the two moments no rank writes records;
# a flat record beside a packed one wins. A line without its newline is a torn
# append, and is ignored.
RECORD_PACK = "records.jsonl"
# Across nodes each node's ranks append to a pack of their own,
# records.<host>.jsonl (NS_RECORD_PACK, set by the multi-node launcher): on
# one shared file 2 nodes spent ~37x one node's writer-seconds and a kill lost
# ~20% of the printed records (round 289). Readers take the node packs and
# then the run's pack, which wins; a finish or a resume folds them into it.
NODE_PACK_GLOB = "records.*.jsonl"


def node_packs(evaluations_dir: Path) -> list[Path]:
    try:
        return sorted(evaluations_dir.glob(NODE_PACK_GLOB))
    except OSError:
        return []


def _read_pack_file(path: Path, texts: dict[str, str]) -> int:
    """Add {name: record json} from one pack to `texts`; its whole lines' length."""
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return 0
    whole = data.rfind(b"\n") + 1
    for line in data[:whole].decode(errors="replace").splitlines():
        name, tab, text = line.partition("\t")
        if tab:
            texts[name] = text
    return whole


def _read_record_pack(evaluations_dir: Path, nodes: bool = True) -> tuple[dict[str, str], int, set[str]]:
    """({name: record json} from the packs, the length of the run pack's
    whole lines, the names only a node pack holds)."""
    texts: dict[str, str] = {}
    for path in node_packs(evaluations_dir) if nodes else []:
        _read_pack_file(path, texts)
    on_nodes = set(texts)
    own: dict[str, str] = {}
    whole = _read_pack_file(evaluations_dir / RECORD_PACK, own)
    texts.update(own)
    return texts, whole, on_nodes - own.keys()


def packed_record_texts(evaluations_dir: Path) -> dict[str, str]:
    return _read_record_pack(evaluations_dir)[0]


def _record_reader(evaluations_dir: Path) -> tuple[list[str], Any, dict[str, Path]]:
    """(every evaluation name, a function reading one's record or None,
    evaluation_record_paths())."""
    paths = evaluation_record_paths(evaluations_dir)
    packed = packed_record_texts(evaluations_dir)

    def read(name: str) -> dict[str, Any] | None:
        path = paths.get(name)
        if name in packed and (path is None or path.parent != evaluations_dir
                               or not path.exists()):  # the run's packer just moved it
            try:
                record = json.loads(packed[name])
            except ValueError:
                record = None
            if isinstance(record, dict):
                return record
            print(f"WARNING: ignoring unreadable {name} in {evaluations_dir / RECORD_PACK}",
                  file=sys.stderr, flush=True)
            return None
        return read_evaluation_record(path)

    return sorted(paths.keys() | packed.keys()), read, paths


def _read_records(names: list[str], read: Any, files: dict[str, Path]) -> list[dict[str, Any] | None]:
    """read() of each name. Only names with a record file go through the
    pool: a packed one is a json.loads, which holds the GIL, so 32 threads
    parsed the pack slower than this one does meanwhile (round 268)."""
    from concurrent.futures import ThreadPoolExecutor

    on_disk = [name for name in names if name in files]
    with ThreadPoolExecutor(RECORD_READ_THREADS) as pool:
        pending = pool.map(read, on_disk)  # submitted now, run while this thread parses
        records = {name: read(name) for name in names if name not in files}
        records.update(zip(on_disk, pending))
    return [records[name] for name in names]


def pack_evaluation_records(evaluations_dir: Path, known: dict[str, dict[str, Any]],
                            fold_nodes: bool = True) -> int:
    """Move the records named in `known` ({name: record}) into the pack and
    unlink their files - a flat record, or a pre-round-92 metrics.json and
    then its dir if that held nothing else - and append again any packed one
    `known` has edited since; how many lines it appended. Writers of the pack
    (a resume, the run's packer, the finish) take its lock in turn.
    `fold_nodes` also moves the node packs in, which only a caller no rank
    is appending beside may do: the run's own packer passes False."""
    import fcntl
    from concurrent.futures import ThreadPoolExecutor

    clock = [time.monotonic()]

    def lap(phase: str) -> None:
        clock.append(time.monotonic())
        LAST_PACK_SECONDS[phase] = clock[-1] - clock[-2]

    LAST_PACK_SECONDS.clear()
    with open(evaluations_dir / RECORD_PACK, "ab") as pack:
        fcntl.flock(pack, fcntl.LOCK_EX)
        lap("lock")
        paths = evaluation_record_paths(evaluations_dir)  # one listing, not a stat a record
        lap("list")
        texts, whole, on_nodes = _read_record_pack(evaluations_dir, fold_nodes)
        folding = node_packs(evaluations_dir) if fold_nodes else []
        lap("read")
        lines: dict[str, str] = {name: texts[name] for name in sorted(on_nodes)}
        moving = []
        for name in sorted(known):
            if name in paths and (paths[name].parent == evaluations_dir or name not in texts):
                moving.append(name)  # a packed name's dir has no record file: only a flat one is newer
            text = json.dumps(known[name])
            # Not on disk at all: a record write the finish cancelled.
            if name in moving or name in on_nodes or text != texts.get(name):
                lines[name] = text  # the last line of a name is its record
        lap("encode")
        # A torn last line from a killed packer, or a killed rank's append,
        # which the next append would otherwise run on into.
        if whole < os.fstat(pack.fileno()).st_size:
            pack.truncate(whole)
        if lines:
            pack.write("".join(f"{name}\t{text}\n" for name, text in lines.items()).encode())
            pack.flush()
            os.fsync(pack.fileno())
        for path in folding:  # every line of them is in the run's pack now
            path.unlink(missing_ok=True)
        lap("write")
        if not lines:
            return 0

    def unlink(name: str) -> None:
        paths[name].unlink(missing_ok=True)
        if paths[name].parent != evaluations_dir:
            try:
                paths[name].parent.rmdir()
            except OSError:
                pass  # it keeps something

    # Off rank 0's pinned CPU, as the record writer is (round 54): pinned, a
    # thread holding the directory's lock waits for the core, and an unlink
    # cost ~6ms rather than ~0.2ms (round 93).
    with ThreadPoolExecutor(RECORD_READ_THREADS, initializer=_unpin_thread) as pool:
        list(pool.map(unlink, moving))
    lap("unlink")
    LAST_PACK_SECONDS["lines"], LAST_PACK_SECONDS["files"] = len(lines), len(moving)
    return len(lines)


# Where the last pack_evaluation_records() call's time went, for the post-run
# line: a whole sapphire node's final pack took 18.2s against 4.3s on 38
# icelake CPUs for 12% more records, and nothing said why (round 256).
LAST_PACK_SECONDS: dict[str, float] = {}


# A Lustre unlink holds the evaluations dir's lock, ~0.13ms each whatever the
# thread count, so packing 67k records after the sampler cost 17.8s (round
# 196). A run's rank 0 packs the flat records as they land instead, in a
# child process: pypolychord's root holds the GIL, so a thread would never
# run (round 199).
RECORD_PACK_SECONDS = 15.0


def pack_flat_records(evaluations_dir: Path) -> int:
    """Pack every flat record on disk now; how many lines it appended."""
    from concurrent.futures import ThreadPoolExecutor

    flat = [path for path in evaluation_record_paths(evaluations_dir).values()
            if path.parent == evaluations_dir]
    if not flat:
        # A run appending its records (round 240): no pass may hold the pack's
        # lock, which every rank's next append then waits on, for nothing.
        return 0
    with ThreadPoolExecutor(RECORD_READ_THREADS) as pool:
        records = dict(zip((path.name.removesuffix(RECORD_SUFFIX) for path in flat),
                           pool.map(read_evaluation_record, flat)))
    return pack_evaluation_records(
        evaluations_dir, {name: record for name, record in records.items() if record is not None},
        fold_nodes=False)


def start_record_packer(evaluations_dir: Path) -> subprocess.Popen | None:
    """Rank 0: pack records every RECORD_PACK_SECONDS until stop_record_packer()."""
    if os.environ.get("NS_PACK_DURING_RUN", "1") == "0":
        return None
    # Until rank 0 is gone too: a packer orphaned by a crash ends at its next pass.
    code = ("import os, sys, time; sys.path.insert(0, sys.argv[1]); import common\n"
            "if sys.argv[4]: os.sched_setaffinity(0, map(int, sys.argv[4].split(',')))\n"
            "while os.getppid() == int(sys.argv[3]):\n"
            "    time.sleep(common.RECORD_PACK_SECONDS)\n"
            "    try:\n"
            "        common.pack_flat_records(common.Path(sys.argv[2]))\n"
            "    except OSError as error:\n"
            "        print(f'record packer: {error}', file=sys.stderr, flush=True)\n")
    return subprocess.Popen(
        [sys.executable, "-c", code, str(Path(__file__).resolve().parent), str(evaluations_dir),
         str(os.getpid()), ",".join(map(str, _UNPINNED_CPUS))],
        stdin=subprocess.DEVNULL)


def stop_record_packer(packer: subprocess.Popen | None) -> None:
    """Before rank 0 reads the run's records back: none may move under it."""
    if packer is not None:
        packer.kill()  # a pack it cut short is whole lines and files still on disk
        packer.wait()


def evaluation_record_file(eval_dir: Path) -> Path:
    """The record of one evaluation, in either layout (two stats at most)."""
    flat = flat_record_path(eval_dir)
    return flat if flat.exists() else eval_dir / "metrics.json"


def read_evaluation_record(metrics_path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(metrics_path.read_text())
    except (FileNotFoundError, NotADirectoryError):
        # An evaluation that was still in flight (or a stray eval-* file).
        # The ordinary case on every resume, and not worth a word.
        return None
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        print(f"WARNING: ignoring unreadable {metrics_path}", file=sys.stderr, flush=True)
        return None
    return record


# Concurrent record reads. On CSD3's Lustre a cold metrics.json read is ~86ms
# and globbing eval-*/metrics.json another ~19ms per directory, so reading a
# 71k-evaluation run back serially took ~2h; 32 threads read one every ~2ms.
RECORD_READ_THREADS = 32


def read_evaluation_records(evaluations_dir: Path) -> list[tuple[Path, dict[str, Any] | None]]:
    """(eval dir, record or None) for every eval-* entry, in name order."""
    names, read, files = _record_reader(evaluations_dir)
    records = _read_records(names, read, files)
    return [(evaluations_dir / name, record) for name, record in zip(names, records)]


def load_evaluations_from_dir(evaluations_dir: Path) -> list[dict[str, Any]]:
    return [record for _, record in read_evaluation_records(evaluations_dir) if record is not None]


def gathered_and_earlier_records(evaluations_dir: Path, written: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The run's records in name order, as load_evaluations_from_dir() gives
    them, reading from disk only those not already in `written`."""
    listed, read_one, files = _record_reader(evaluations_dir)
    names = set(listed)
    earlier = sorted(names - written.keys())
    read = dict(zip(earlier, _read_records(earlier, read_one, files)))
    _FINISHED_RECORDS.clear()
    for name in sorted(names | written.keys()):
        record = written.get(name) or read.get(name)
        if record is not None:
            _FINISHED_RECORDS[name] = record
    return list(_FINISHED_RECORDS.values())


# The records finished_run_records() gave rank 0, by name, which
# pack_finished_run() packs once prune_run_artefacts() has edited them.
_FINISHED_RECORDS: dict[str, dict[str, Any]] = {}


def pack_finished_run(evaluations_dir: Path) -> int:
    """Rank 0, once the summary is written: no rank writes records now."""
    return pack_evaluation_records(evaluations_dir, _FINISHED_RECORDS)


class PolyChordFinish:
    """Ends the chains still in flight once PolyChord has finished (round 81).

    An async run's root writes every output (chains, .resume, stats) and
    makes its final dump() call, then waits for each worker's chain only to
    throw its points away (NestedSampling's cleanup keeps just nlike). That
    wait was the last ~30s of a run, one rank slicing alone. On the final
    dump - the only one with no live points - the root tells every worker,
    which then answers PolyChord without evaluating: low, low, high, so a
    step-out ends on a low and the shrink on a high within three calls.
    Build it on every rank before run_polychord (Dup() is collective) and
    call close() on every rank after it.
    """

    TAG = 81

    def __init__(self) -> None:
        try:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD.Dup()
        except Exception:
            self.comm = None  # no MPI: one process, nothing in flight to end
        self.rank = self.comm.Get_rank() if self.comm is not None else 0
        self.done = False
        self.skipped = 0
        self._sent: list | None = None

    def dumper(self, live, dead, logweights, logZ, logZerr) -> None:
        if self.rank == 0 and len(live) == 0:
            self._send()

    def _send(self) -> None:
        if self._sent is None and self.comm is not None:
            self._sent = [self.comm.isend(None, dest=rank, tag=self.TAG) for rank in range(1, self.comm.Get_size())]

    def finished(self) -> bool:
        """On a worker, True once the root's final dump has happened. One
        Iprobe a call, which costs microseconds."""
        if not self.done and self.rank and self.comm.iprobe(source=0, tag=self.TAG):
            self.comm.recv(source=0, tag=self.TAG)
            self.done = True
        return self.done

    def discarded(self) -> float:
        self.skipped += 1
        return 1e300 if self.skipped % 3 == 0 else -1e300

    def close(self) -> None:
        if self.comm is None:
            return
        if self.rank == 0:
            self._send()  # a run whose final dump was missed must not hang its workers below
            from mpi4py import MPI

            MPI.Request.Waitall(self._sent)
        elif not self.done:
            self.comm.recv(source=0, tag=self.TAG)  # synchronous runs end with no chain in flight
            self.done = True
        if self.skipped:
            print(f"rank {self.rank}: {self.skipped} likelihood calls answered without evaluating "
                  "after PolyChord finished", flush=True)


CHAINS_MIRROR_SECONDS = 2.0


def chains_base_dir(output_dir: Path) -> Path:
    """Where PolyChord writes: node tmpfs, mirrored to <run>/chains (round 227).

    The root creates <root>_temp.resume at every checkpoint, and on CSD3's
    Lustre about one such create in 60-80 blocks 15-45s; no worker gets a seed
    meanwhile, so the whole search stops (round 225). mpirun forks every
    rank on one node, so they all see the same tmpfs.
    """
    chains = output_dir / "chains"
    if os.environ.get("NS_CHAINS_TMPFS", "1") == "0" or not os.path.isdir("/dev/shm"):
        return chains
    import hashlib

    tag = hashlib.sha1(str(chains.resolve()).encode()).hexdigest()[:16]
    return Path("/dev/shm") / f"ri-chains-{tag}"


def sync_chains(src: Path, dst: Path, seen: dict[Path, tuple[int, int]]) -> int:
    """Copy each file changed since `seen` to dst. A .resume goes through a
    temp name and a rename, so <run>/chains never holds half a checkpoint;
    the text files PolyChord only writes are rewritten in place, because a
    Lustre create blocks for up to ~40 s (12 of 1,211 under load) while an
    in-place rewrite never passed 1 s (round 229). PolyChord renames its
    .resume into place, so the copy reads a whole one; a text file caught
    mid-write is copied again on the next pass, its stamp having moved."""
    import shutil

    copied = 0
    for path in sorted(src.rglob("*")):
        if "_temp." in path.name:
            continue
        try:
            stat = path.stat()
            if not path.is_file() or seen.get(path) == (stat.st_mtime_ns, stat.st_size):
                continue
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix != ".resume" and target.is_file():
                with open(path, "rb") as source, open(target, "r+b") as out:
                    shutil.copyfileobj(source, out)
                    out.truncate()
            else:
                partial = target.with_name(f".{target.name}.mirror")
                shutil.copyfile(path, partial)
                partial.replace(target)
        except FileNotFoundError:
            continue  # renamed away under us; the next pass sees its successor
        seen[path] = (stat.st_mtime_ns, stat.st_size)
        copied += 1
    return copied


class ChainsMirror:
    """Rank 0: seed tmpfs from <run>/chains, then mirror it back every
    CHAINS_MIRROR_SECONDS from a child process, which Lustre may block without
    stopping the root. Not a thread: run_polychord holds the GIL in Fortran,
    so a thread never runs. stop() is the blocking final sync. A crash loses
    the mirror's lag: one pass, up to ~40s while Lustre blocks its copy
    (round 227). `./ri resume` and run_with_retries read <run>/chains."""

    def __init__(self, output_dir: Path, base_dir: Path) -> None:
        import shutil

        self.lustre, self.tmpfs = output_dir / "chains", base_dir
        self.child: subprocess.Popen | None = None
        if self.tmpfs == self.lustre:
            return
        shutil.rmtree(self.tmpfs, ignore_errors=True)  # a killed run's leftovers: Lustre is the authority
        if self.lustre.is_dir():
            shutil.copytree(self.lustre, self.tmpfs, ignore=shutil.ignore_patterns(".*.mirror"))
        self.tmpfs.mkdir(parents=True, exist_ok=True)
        # Until rank 0 is gone too, so a crash orphans it for one pass at most.
        code = ("import os, sys, time; sys.path.insert(0, sys.argv[1]); import common\n"
                "if sys.argv[5]: os.sched_setaffinity(0, map(int, sys.argv[5].split(',')))\n"
                "src, dst = common.Path(sys.argv[2]), common.Path(sys.argv[3])\n"
                "seen = {}  # one full pass first: whatever changed before it started\n"
                "while os.getppid() == int(sys.argv[4]):\n"
                "    time.sleep(common.CHAINS_MIRROR_SECONDS)\n"
                "    try:\n"
                "        common.sync_chains(src, dst, seen)\n"
                "    except OSError as error:\n"
                "        print(f'chains mirror: {error}', file=sys.stderr, flush=True)\n")
        self.child = subprocess.Popen(
            [sys.executable, "-c", code, str(Path(__file__).resolve().parent), str(self.tmpfs),
             str(self.lustre), str(os.getpid()), ",".join(map(str, _UNPINNED_CPUS))],
            stdin=subprocess.DEVNULL)
        atexit.register(self.stop)

    def stop(self) -> None:
        import shutil

        if self.child is None:
            return
        self.child.kill()  # a copy it cut short is a .mirror file the sync below replaces
        self.child.wait()
        self.child = None
        sync_chains(self.tmpfs, self.lustre, {})
        shutil.rmtree(self.tmpfs, ignore_errors=True)


def start_chains_mirror(output_dir: Path, base_dir: Path) -> ChainsMirror | None:
    return ChainsMirror(output_dir, base_dir) if mpi_rank() == 0 else None


def self_check_chains_mirror() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run, shm = Path(tmp) / "run", Path(tmp) / "shm"
        (run / "chains").mkdir(parents=True)
        (run / "chains" / "r.resume").write_text("old")
        (run / "chains" / "r.paramnames").write_text("p")
        mirror = ChainsMirror(run, shm)
        assert (shm / "r.resume").read_text() == "old", "seeded from <run>/chains"
        (shm / "r.resume").write_text("new")
        (shm / "r_temp.resume").write_text("half")
        (shm / "clusters").mkdir()
        (shm / "clusters" / "r_1.txt").write_text("c")
        time.sleep(CHAINS_MIRROR_SECONDS * 2.5)
        assert (run / "chains" / "r.resume").read_text() == "new", "mirrored while running"
        (shm / "r.stats").write_text("s")
        paramnames, resume = ((run / "chains" / name).stat().st_ino for name in ("r.paramnames", "r.resume"))
        (shm / "r.paramnames").write_text("longer than before")
        (shm / "r.resume").write_text("x")
        mirror.stop()
        assert (run / "chains" / "r.stats").read_text() == "s", "final sync"
        assert (run / "chains" / "r.paramnames").read_text() == "longer than before"
        assert (run / "chains" / "r.paramnames").stat().st_ino == paramnames, "text rewritten in place: no create"
        assert (run / "chains" / "r.resume").read_text() == "x"
        assert (run / "chains" / "r.resume").stat().st_ino != resume, ".resume renamed into place"
        assert (run / "chains" / "clusters" / "r_1.txt").read_text() == "c"
        assert not (run / "chains" / "r_temp.resume").exists(), "temp files stay behind"
        assert not list((run / "chains").glob(".*.mirror")), "no partial left"
        assert not shm.exists(), "tmpfs removed"
        same = ChainsMirror(run, run / "chains")
        assert same.child is None, "NS_CHAINS_TMPFS=0 mirrors nothing"
        os.environ["NS_CHAINS_TMPFS"] = "0"
        try:
            assert chains_base_dir(run) == run / "chains"
        finally:
            del os.environ["NS_CHAINS_TMPFS"]


def evidence_record(output: Any) -> dict[str, Any] | None:
    """log(Z) and its error from the PolyChordOutput run_polychord() returns,
    which has already parsed chains/<root>.stats, so it costs nothing. In the
    summary because a validation run in tmpfs loses its chains, and run.log
    gives log(Z) to 2 decimals at a checkpoint before the end (round 184)."""
    try:
        return {"log_z": output.logZ, "log_z_err": output.logZerr, "ndead": output.ndead, "nlike": output.nlike}
    except AttributeError:
        return None


def finished_run_records(evaluations_dir: Path) -> list[dict[str, Any]] | None:
    """Drain every rank's records and settle its held extremes; rank 0 gets
    every record, the others None. Collective: every rank calls it.

    Each rank holds its own 20 lowest and highest so far, so a run ends with
    ~40 held per rank of which only the run's 40 survive prune_run_artefacts().
    Flushing all of them cost ~20s of copies to Lustre and ~10s of unlinks
    after a 36-rank run (csd3-speed.md, round 51). So rank 0 reads the
    records first, decides what the run keeps - the same decision prune
    makes - and the ranks publish only that and drop the rest in tmpfs.
    """
    POST_RUN_SECONDS.clear()
    POST_RUN_BACKLOG.clear()
    started = time.monotonic()
    # This rank's own backlog and drain, apart from the barrier's wait for
    # the slowest: round 233's 60s drain was one of the two (round 234).
    queued = [future.kind for future in _RECORD_WRITES if not future.done()]
    # A record still queued has its text in _WRITTEN_RECORDS, which rank 0
    # gathers and pack_finished_run() writes into the pack: writing the file
    # first only for the pack to unlink it was most of a backlog of up to
    # 400 jobs a rank behind a Lustre slower than ~0.5s a create (round 234).
    for future in _RECORD_WRITES:
        if future.kind == "record":
            future.cancel()
    # And a queued publish of a held extreme goes back to being held, so that
    # the keep-filtered flush below drops it in tmpfs unless the run keeps it.
    for name in list(_PUBLISHING):
        cancelled = _cancel_held_publish(name)
        if cancelled is not None:
            _HELD[name] = cancelled
    _PUBLISHING.clear()
    _RECORD_WRITES[:] = [future for future in _RECORD_WRITES if not future.cancelled()]
    drain_record_writes(flush_held=False)
    _PENDING_RECORDS.clear()  # their texts are in _WRITTEN_RECORDS too
    backlog = (queued, time.monotonic() - started, dict(_WRITER_SECONDS))
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except Exception:
        comm = None  # no MPI (a host run, or the self-checks): this process is the run
    if comm is not None:
        comm.Barrier()  # apart from the gather, so post_run_line() tells the two apart
    started = post_run_phase("drain", started)
    backlogs = comm.gather(backlog, root=0) if comm is not None else [backlog]
    if backlogs:
        kinds = [kind for queued, _, _ in backlogs for kind in queued]
        POST_RUN_BACKLOG.update({kind: kinds.count(kind) for kind in sorted(set(kinds))},
                                queued_max=max(len(q) for q, _, _ in backlogs),
                                writer_max_s=max(s for _, s, _ in backlogs))
        busy = [spent for _, _, spent in backlogs]
        POST_RUN_BACKLOG.update({f"{kind}_s": sum(b.get(kind, 0.0) for b in busy)
                                 for kind in sorted({k for b in busy for k in b})})
    # Each rank parses its own records, in parallel, and hands them to rank 0,
    # which reads from Lustre only what an earlier segment of a resumed run
    # wrote: re-reading 43k metrics.json files took it 7.5s (round 52). The
    # texts held for this cost ~4KB a record, ~5MB a rank in that run.
    here = os.path.abspath(evaluations_dir)
    mine = [(os.path.basename(path), json.loads(text)) for path, text in _WRITTEN_RECORDS.items()
            if os.path.dirname(path) == here]
    gathered = comm.gather(mine, root=0) if comm is not None else [mine]
    records = None
    if mpi_rank() == 0:
        written = {name: record for rank_records in gathered for name, record in rank_records}
        records = gathered_and_earlier_records(evaluations_dir, written)
    started = post_run_phase("records", started)
    keep = (None if records is None or os.environ.get("NS_KEEP_ALL_IMAGES", "0") != "0"
            else evaluations_keeping_images(records))
    if comm is not None:
        keep = comm.bcast(keep, root=0)
    flush_held_evaluations(force=True, keep=keep)
    if comm is not None:
        comm.Barrier()  # the kept extremes are on Lustre before rank 0 prunes
    post_run_phase("flush", started)
    return records


# Seconds per step between run_polychord() returning and the summary landing,
# which `./ri bench` never sees (its wall is the evaluation span) and which
# grows with the evaluation count: 13.5% of a production run in round 51.
POST_RUN_SECONDS: dict[str, float] = {}
# Rank 0's view of every rank's record-writer backlog when the run ended.
POST_RUN_BACKLOG: dict[str, float] = {}


def post_run_phase(name: str, started: float) -> float:
    now = time.monotonic()
    POST_RUN_SECONDS[name] = now - started
    return now


def post_run_line() -> str:
    line = "post-run seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in POST_RUN_SECONDS.items())
    if "pack" in POST_RUN_SECONDS and LAST_PACK_SECONDS:
        pack = dict(LAST_PACK_SECONDS)
        counts = f"{pack.pop('lines', 0):.0f} lines, {pack.pop('files', 0):.0f} files"
        line += " (" + ", ".join(f"{k} {v:.1f}" for k, v in pack.items()) + f"; {counts})"
    if POST_RUN_BACKLOG:
        backlog = dict(POST_RUN_BACKLOG)
        line += "; slowest rank's drain {:.1f}, most queued on one rank {:.0f}".format(
            backlog.pop("writer_max_s"), backlog.pop("queued_max"))
        line += "".join(f", {kind} {count:.0f}" for kind, count in backlog.items()
                        if not kind.endswith("_s"))
        cancelled = backlog.pop("cancelled_held_s", None)
        line += "; writer seconds, all ranks: " + ", ".join(
            f"{kind[:-2]} {spent:.0f}" for kind, spent in backlog.items() if kind.endswith("_s"))
        if cancelled is not None:
            line += f"; held publishes cancelled {cancelled:.0f}"
    return line


def adopt_completed_evaluations(
    evaluations_dir: Path,
    cache: dict[str, float],
) -> int:
    """Cache finished evaluations and remove incomplete directories.

    Rank 0 alone reads the records and broadcasts {key: objective}: every
    rank parsing every record cost 1 GB and 6 s a rank at 67k evaluations,
    and 6.4 GB a rank at 400k, which a 113-rank resume cannot hold (round 267).
    """
    import shutil

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            comm = None
    except Exception:
        comm = None  # no MPI (a host run, or the self-checks): this process is the run
    if comm is not None and comm.Get_rank() != 0:
        adopted, keys = comm.bcast(None, root=0)
        cache.update(keys)
        return adopted
    adopted = 0
    keys: dict[str, float] = {}
    good: dict[str, dict[str, Any]] = {}
    # A record a killed writer left half-written: the record it was replacing,
    # if any, is still there beside it.
    for partial in evaluations_dir.glob("eval-*" + RECORD_SUFFIX + ".partial"):
        partial.unlink(missing_ok=True)
    # Only these can need packing: a record read from the pack is in it.
    listed = evaluation_record_paths(evaluations_dir)
    for eval_dir, record in read_evaluation_records(evaluations_dir):
        if record is not None and eval_dir.name in listed:
            good[eval_dir.name] = record
        if record is None:
            # A half-written record goes too, or every resume warns about it.
            flat_record_path(eval_dir).unlink(missing_ok=True)
            if not eval_dir.is_dir():
                continue
            # ignore_errors: best effort, the next resume sweeps what is left.
            shutil.rmtree(eval_dir, ignore_errors=True)
            continue
        keys[params_key(record["params"])] = float(record["objective"])
        adopted += 1
    cache.update(keys)
    # Collective, and before the pack: no rank reads a flat record after it.
    if comm is not None:
        comm.bcast((adopted, keys), root=0)
    if good or node_packs(evaluations_dir):
        started = time.monotonic()
        packed = pack_evaluation_records(evaluations_dir, good)
        if packed:
            print(f"packed {packed} earlier records into {RECORD_PACK} in "
                  f"{time.monotonic() - started:.1f}s", flush=True)
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

        # A resumed run's records: rank 0 is handed this segment's and reads
        # only the earlier segment's back, and the result is what reading
        # them all from disk gives, in the same order.
        later = evaluations_dir / "eval-0002-def"
        later.mkdir()
        write_evaluation_record(later, {"eval_id": 2, "params": {"a": 2}, "objective": 0.25})
        (evaluations_dir / "eval-0003-ghi").mkdir()  # in flight: no record
        written = {"eval-0002-def": json.loads(flat_record_path(later).read_text())}
        assert gathered_and_earlier_records(evaluations_dir, written) == load_evaluations_from_dir(evaluations_dir)
        assert [r["eval_id"] for r in finished_run_records(evaluations_dir)] == [1, 2]

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
        # so a rank killed mid-write leaves no record at all instead of half
        # of one. A rename gives a new inode; an in-place write does not.
        eval_dir = Path(tmp) / "eval-0001-abc"
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": {"a": 1}, "objective": 0.5})
        first_inode = flat_record_path(eval_dir).stat().st_ino
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": {"a": 1}, "objective": 0.7})
        assert flat_record_path(eval_dir).stat().st_ino != first_inode
        # One inode an evaluation that keeps nothing: no dir beside it.
        assert list(Path(tmp).iterdir()) == [flat_record_path(eval_dir)]
        # A half-written record a killed writer left is swept on resume, and
        # the record it was replacing still counts.
        flat_record_path(eval_dir).with_name("eval-0001-abc.json.partial").write_text('{"ev')
        assert adopt_completed_evaluations(Path(tmp), {}) == 1
        # ...and the record, adopted, is packed (round 93).
        assert list(Path(tmp).iterdir()) == [Path(tmp) / RECORD_PACK]
        # ...and one that was renamed into place half-written (a full disk).
        torn = Path(tmp) / "eval-0009-zzz.json"
        torn.write_text('{"ev')
        assert adopt_completed_evaluations(Path(tmp), {}) == 1
        assert not torn.exists()
        # Both layouts in one run (a run resumed across round 92) read as one.
        legacy = Path(tmp) / "eval-0002-def"
        legacy.mkdir()
        (legacy / "metrics.json").write_text(json.dumps({"eval_id": 2, "params": {"a": 2}, "objective": 0.1}))
        kept = Path(tmp) / "eval-0001-abc" / "wsclean"
        kept.mkdir(parents=True)  # a kept evaluation's artefacts, beside its record
        assert [r["eval_id"] for _, r in read_evaluation_records(Path(tmp))] == [1, 2]
        assert evaluation_record_file(legacy) == legacy / "metrics.json"

    with tempfile.TemporaryDirectory() as tmp:
        # The pack (round 93): a resume packs every earlier flat record, keeps
        # a packed evaluation's dir, reads a flat record over a packed one and
        # drops a torn last line; a finished run packs what prune left.
        evaluations_dir = Path(tmp)
        kept = evaluations_dir / "eval-0001-abc"
        (kept / "wsclean").mkdir(parents=True)
        write_evaluation_record(kept, {"eval_id": 1, "params": {"a": 1}, "objective": 0.5})
        write_evaluation_record(evaluations_dir / "eval-0002-def",
                                {"eval_id": 2, "params": {"a": 2}, "objective": 0.25})
        cache = {}
        assert adopt_completed_evaluations(evaluations_dir, cache) == 2 and len(cache) == 2
        assert sorted(p.name for p in evaluations_dir.iterdir()) == ["eval-0001-abc", RECORD_PACK]
        assert adopt_completed_evaluations(evaluations_dir, {}) == 2  # nothing left to pack
        assert (kept / "wsclean").is_dir(), "a packed evaluation's kept dir was swept"
        with open(evaluations_dir / RECORD_PACK, "a") as pack:
            pack.write('eval-0003-ghi\t{"eval_id": 3, "par')  # a packer killed mid-line
        assert [r["eval_id"] for r in load_evaluations_from_dir(evaluations_dir)] == [1, 2]
        write_evaluation_record(evaluations_dir / "eval-0002-def",
                                {"eval_id": 2, "params": {"a": 2}, "objective": 0.75})
        assert [r["objective"] for r in load_evaluations_from_dir(evaluations_dir)] == [0.5, 0.75]
        write_evaluation_record(evaluations_dir / "eval-0004-jkl",
                                {"eval_id": 4, "params": {"a": 4}, "objective": 0.1})
        records = finished_run_records(evaluations_dir)
        assert [r["eval_id"] for r in records] == [1, 2, 4]
        records[2]["paths"] = {}  # as prune_run_artefacts() edits a record
        assert pack_finished_run(evaluations_dir) == 2
        assert sorted(p.name for p in evaluations_dir.iterdir()) == ["eval-0001-abc", RECORD_PACK]
        assert (evaluations_dir / RECORD_PACK).read_text().count("\n") == 4  # 1, 2, 2 again, 4
        assert load_evaluations_from_dir(evaluations_dir) == records
        # The run's packer (round 199): flat records into the pack as they
        # land, then the finish appends only what prune edited since.
        write_evaluation_record(evaluations_dir / "eval-0007-stu",
                                {"eval_id": 7, "params": {"a": 7}, "objective": 0.2,
                                 "paths": {"image": "x"}})
        assert pack_flat_records(evaluations_dir) == 1 and pack_flat_records(evaluations_dir) == 0
        assert not flat_record_path(evaluations_dir / "eval-0007-stu").exists()
        _WRITTEN_RECORDS.clear()  # a second finish, from records on disk alone
        records = finished_run_records(evaluations_dir)
        assert [r["eval_id"] for r in records] == [1, 2, 4, 7]
        assert pack_finished_run(evaluations_dir) == 0  # nothing edited
        records[3]["paths"] = {}
        assert pack_finished_run(evaluations_dir) == 1
        assert load_evaluations_from_dir(evaluations_dir) == records
        packer = start_record_packer(evaluations_dir)
        assert packer is not None and packer.poll() is None
        stop_record_packer(packer)
        assert packer.returncode is not None
        # A pre-round-92 run: metrics.json goes, and its dir unless it keeps something.
        for name, keeps in (("eval-0005-mno", False), ("eval-0006-pqr", True)):
            (evaluations_dir / name).mkdir()
            (evaluations_dir / name / "metrics.json").write_text(json.dumps(
                {"eval_id": 5 + keeps, "params": {"a": 5 + keeps}, "objective": 0.2}))
            if keeps:
                (evaluations_dir / name / "wsclean").mkdir()
        assert adopt_completed_evaluations(evaluations_dir, {}) == 6
        assert sorted(p.name for p in evaluations_dir.iterdir()) == [
            "eval-0001-abc", "eval-0006-pqr", RECORD_PACK]
        assert [r["eval_id"] for r in load_evaluations_from_dir(evaluations_dir)] == [1, 2, 4, 5, 6, 7]
        # A record still queued when the run ends goes into the pack, never
        # to its own file (round 234), with the flat writer too.
        import threading

        _WRITTEN_RECORDS.clear()
        os.environ["NS_RECORD_APPEND"] = "0"
        write_records_in_background()
        gate = threading.Event()
        _submit_record_job(gate.wait, "held")
        queued = evaluations_dir / "eval-0008-vwx"
        write_evaluation_record(queued, {"eval_id": 8, "params": {"a": 8}, "objective": 0.3})
        threading.Timer(0.5, gate.set).start()
        records = finished_run_records(evaluations_dir)
        assert [r["eval_id"] for r in records] == [1, 2, 4, 5, 6, 7, 8]
        assert not flat_record_path(queued).exists(), "a queued record was written"
        assert pack_finished_run(evaluations_dir) == 1
        assert load_evaluations_from_dir(evaluations_dir) == records
        # A writer that fell behind appends its backlog to the pack in one
        # write; one that keeps up writes a flat file (round 247).
        write_records_in_background()
        gate = threading.Event()
        _submit_record_job(gate.wait, "held")
        backlog = [evaluations_dir / f"eval-002{i}-bkl" for i in range(3)]
        for i, eval_dir in enumerate(backlog):
            write_evaluation_record(eval_dir, {"eval_id": 20 + i, "params": {"a": 20 + i}, "objective": 0.5})
        gate.set()
        drain_record_writes()
        assert not any(flat_record_path(eval_dir).exists() for eval_dir in backlog), "a backlog went flat"
        assert [json.loads(packed_record_texts(evaluations_dir)[d.name])["eval_id"] for d in backlog] == [20, 21, 22]
        write_records_in_background()
        lone = evaluations_dir / "eval-0023-bkl"
        write_evaluation_record(lone, {"eval_id": 23, "params": {"a": 23}, "objective": 0.5})
        drain_record_writes()
        assert flat_record_path(lone).exists(), "a lone record was not written flat"
        records = finished_run_records(evaluations_dir)
        assert [r["eval_id"] for r in records] == [1, 2, 4, 5, 6, 7, 8, 20, 21, 22, 23]
        assert pack_finished_run(evaluations_dir) == 1  # the flat one; the appended are packed
        # A run's writer (the default since round 264) appends its records
        # to the pack, never a flat file, and the finish then has nothing to
        # add (round 240). A torn last
        # line - a rank killed mid-append - is cut by the next pack writer,
        # so the next append does not run on into it.
        _WRITTEN_RECORDS.clear()
        os.environ["NS_RECORD_APPEND"] = "1"
        write_records_in_background()
        appended = evaluations_dir / "eval-0010-yza"
        write_evaluation_record(appended, {"eval_id": 10, "params": {"a": 10}, "objective": 0.4})
        drain_record_writes()
        assert not flat_record_path(appended).exists(), "an appended record was written flat"
        assert json.loads(packed_record_texts(evaluations_dir)[appended.name])["eval_id"] == 10
        records = finished_run_records(evaluations_dir)
        assert [r["eval_id"] for r in records] == [1, 2, 4, 5, 6, 7, 8, 10, 20, 21, 22, 23]
        assert pack_finished_run(evaluations_dir) == 0, "the finish re-appended an appended record"
        with open(evaluations_dir / RECORD_PACK, "a") as pack:
            pack.write('eval-0011-bcd\t{"eval_id": 11, "par')
        assert pack_evaluation_records(evaluations_dir, {}) == 0
        write_records_in_background()
        write_evaluation_record(evaluations_dir / "eval-0012-efg",
                                {"eval_id": 12, "params": {"a": 12}, "objective": 0.6})
        drain_record_writes()
        os.environ.pop("NS_RECORD_APPEND")
        assert 12 in [r["eval_id"] for r in load_evaluations_from_dir(evaluations_dir)]
        # Across nodes a rank appends to its node's pack (round 290), which
        # every reader takes too; the run's packer leaves it alone, and a
        # finish or a resume folds it into the run's pack, which wins.
        os.environ["NS_RECORD_PACK"] = "records.node-b.jsonl"
        write_records_in_background()
        remote = evaluations_dir / "eval-0013-hij"
        write_evaluation_record(remote, {"eval_id": 13, "params": {"a": 13}, "objective": 0.6})
        drain_record_writes()
        os.environ.pop("NS_RECORD_PACK")
        node_pack = evaluations_dir / "records.node-b.jsonl"
        assert node_pack.read_text().startswith("eval-0013-hij\t"), "a node's record missed its pack"
        assert json.loads(packed_record_texts(evaluations_dir)[remote.name])["eval_id"] == 13
        write_evaluation_record(evaluations_dir / "eval-0014-klm",
                                {"eval_id": 14, "params": {"a": 14}, "objective": 0.6})  # flat
        assert pack_flat_records(evaluations_dir) == 1 and node_pack.exists()
        with open(node_pack, "a") as pack:
            pack.write('eval-0015-nop\t{"eval_id": 15, "par')  # a rank killed mid-append
        assert adopt_completed_evaluations(evaluations_dir, {}) == 15
        assert not node_pack.exists(), "a resume left a node pack unfolded"
        texts = packed_record_texts(evaluations_dir)
        assert json.loads(texts[remote.name])["eval_id"] == 13 and "eval-0015-nop" not in texts
        node_pack.write_text('eval-0013-hij\t{"eval_id": 13, "params": {"a": 13}, "objective": 9}\n'
                             'eval-0016-qrs\t{"eval_id": 16, "params": {"a": 16}, "objective": 0.1}\n')
        _WRITTEN_RECORDS.clear()
        records = finished_run_records(evaluations_dir)
        assert {r["eval_id"]: r["objective"] for r in records}[13] == 0.6, "a node pack beat the run's"
        assert pack_finished_run(evaluations_dir) == 1 and not node_pack.exists()
        assert load_evaluations_from_dir(evaluations_dir) == records

    with tempfile.TemporaryDirectory() as tmp:
        # A fresh run adopts nothing and starts at id 1.
        cache = {}
        assert adopt_completed_evaluations(Path(tmp), cache) == 0


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file, then replace `path` atomically."""
    _write_text_atomic(path, _json_text(payload))


def _json_text(payload: Any) -> str:
    # A summary's evaluations go one compact line each: indent=2 takes the
    # pure-Python encoder, 5.1s of a 67k-evaluation summary against ~1.6s
    # this way (round 197). The same JSON values; the head stays indented.
    evaluations = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(evaluations, list) or not evaluations:
        return json.dumps(payload, indent=2) + "\n"
    # Two spaces and a newline: only the top-level key (a string holds no raw newline).
    slot = '\n  "evaluations": []'
    head = json.dumps({**payload, "evaluations": []}, indent=2)
    body = ",\n    ".join(map(json.dumps, evaluations))
    return head.replace(slot, slot[:-1] + "\n    " + body + "\n  ]", 1) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)


def self_check_summary_json() -> None:
    summary = {"evaluation_count": 2, "note": '"evaluations": []', "nested": {"evaluations": []},
               "evaluations": [{"eval_id": 1, "objective": float("nan")}, {"eval_id": 2, "params": {}}],
               "after": [1, 2]}
    text = _json_text(summary)
    assert text.startswith('{\n  "evaluation_count": 2,'), text  # the head readers' layout
    assert '\n    {"eval_id": 2, "params": {}}\n  ],\n  "after"' in text, text
    assert repr(json.loads(text)) == repr(summary)
    for payload in ({}, {"evaluations": []}, [1], {"eval_id": 1}):
        assert _json_text(payload) == json.dumps(payload, indent=2) + "\n"
    print("OK: summary.json writer")


# Scored evaluations keep only replayable inputs, metrics and the three images
# the retention policy may later keep; see docs/nested-sampling-disk-footprint.md.
# Set NS_KEEP_MEASUREMENT_SETS=1 to retain intermediate artefacts for replay
# benchmarks. The dirty and residual images are NOT here: which evaluations keep
# them is a whole-run decision, taken by prune_run_artefacts() once every
# evaluation has been scored and can be ranked.
PRUNED_ARTEFACTS = (
    ("sim.ms", "measurement_set"),
    ("VLAA_ANT", None),
    ("makems.cfg", None),
    ("r2d2_data.mat", "mat"),
    ("wsclean/recon-model.fits", None),
    ("wsclean/recon-psf.fits", None),
    ("r2d2/r2d2_data/PSF.fits", None),
)


def evaluation_scratch_dir(eval_dir: Path) -> Path | None:
    root = os.environ.get("NS_SCRATCH_DIR", "")
    return Path(root) / eval_dir.name if root else None


_FAILED_ARTEFACT_SLOTS_FULL: set[Path] = set()


def claim_failed_artefacts_slot(eval_dir: Path) -> bool:
    """One of the run's NS_KEEP_FAILED_ARTEFACTS slots for a failure's MS and
    images. A failure storm otherwise keeps hundreds of files per failure: one
    in docs/csd3-experiments.md (E5, E6) filled the one-million-file quota.
    Slots are never freed, so once a rank finds them full it stops asking:
    otherwise every later failure costs ~21 contended Lustre mkdirs on the
    rank's main thread (~36ms with 36 ranks; docs/csd3-speed.md, round 38)."""
    slots = eval_dir.parent / ".failed-artefacts-kept"
    if slots in _FAILED_ARTEFACT_SLOTS_FULL:
        return False
    slots.mkdir(exist_ok=True)
    for index in range(int(os.environ.get("NS_KEEP_FAILED_ARTEFACTS", "20"))):
        try:
            (slots / str(index)).mkdir()
            return True
        except FileExistsError:
            continue
    _FAILED_ARTEFACT_SLOTS_FULL.add(slots)
    return False


def prune_evaluation_artefacts(eval_dir: Path, record: dict[str, Any]) -> bool:
    """Delete the artefacts no reader needs; True if this evaluation keeps them."""
    import shutil

    failed = "error" in record
    keeping = os.environ.get("NS_KEEP_MEASUREMENT_SETS", "0") != "0" or (
        failed and claim_failed_artefacts_slot(eval_dir))
    if keeping:
        return True
    work = evaluation_scratch_dir(eval_dir) or eval_dir
    for name, path_key in PRUNED_ARTEFACTS:
        target = work / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        if path_key:
            record.get("paths", {}).pop(path_key, None)
    return False


def publish_evaluation_scratch(eval_dir: Path, record: dict[str, Any], kept: bool,
                               files_only: bool = False, defer: Any = None) -> None:
    """Move what an evaluation wrote to scratch beside its record, if it is kept.

    Everything an evaluation writes goes to tmpfs scratch because on CSD3's
    Lustre each file created costs 25-40ms and each unlink ~10ms, and most
    evaluations are stripped to metrics.json straight after scoring
    (docs/csd3-speed.md, round 5). Records name the final paths either way.
    `defer`, if given, runs the move to Lustre instead (the record writer).
    """
    import shutil

    scratch = evaluation_scratch_dir(eval_dir)
    if scratch is None or not scratch.is_dir():
        return
    if kept:
        (defer or (lambda job: job()))(
            lambda: _move_scratch(scratch, eval_dir, files_only))
    else:
        shutil.rmtree(scratch, ignore_errors=True)
    paths = record.get("paths") or {}
    for key, value in paths.items():
        if isinstance(value, str) and Path(value).is_relative_to(scratch):
            paths[key] = str(eval_dir / Path(value).relative_to(scratch))


def _move_scratch(scratch: Path, eval_dir: Path, files_only: bool) -> None:
    import shutil

    eval_dir.mkdir(parents=True, exist_ok=True)
    for produced in scratch.iterdir():
        if files_only and produced.is_dir():
            continue
        destination = eval_dir / produced.name
        if destination.is_dir():
            shutil.rmtree(destination)
        shutil.move(str(produced), destination, copy_function=shutil.copyfile)
    shutil.rmtree(scratch, ignore_errors=True)


def salvage_evaluation_logs(eval_dir: Path) -> None:
    """Keep a dead evaluation's logs: its scratch goes with the job, and a dead
    worker's stderr is the only record of why it died."""
    try:
        publish_evaluation_scratch(eval_dir, {}, kept=True, files_only=True)
    except OSError:
        pass


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
_GRIDDED = re.compile(r"Gridded visibility count: (\d+)")


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
    # What drives an evaluation's cost (docs/nested-sampling-cost-model.md),
    # kept in the record so `./ri profile --over-time` need not read logs the
    # retention policy deletes from all but ~1 in 100 evaluations.
    gridded = _GRIDDED.search(log_text)
    if gridded:
        out["gridded_visibilities"] = int(gridded.group(1))
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
    if not eval_dir:
        return f"eval_id:{record.get('eval_id')}"
    # Path().name, without building a Path: ~4 a record at the end of a run
    # (round 269). Only a trailing "/" or "." needs Path's normalising.
    name = eval_dir.rpartition("/")[2] if isinstance(eval_dir, str) else ""
    return name if name not in ("", ".") else Path(eval_dir).name


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
    from concurrent.futures import ThreadPoolExecutor

    if os.environ.get("NS_KEEP_ALL_IMAGES", "0") != "0":
        return 0
    keep = evaluations_keeping_images(records)
    # One listing. An extreme a rank later evicted still names the images its
    # stripped dir lost, and was over half a run's records: resolving those
    # paths on Lustre, ~3 stats and a realpath each, was ~95% of prune
    # (docs/csd3-speed.md, round 198). A dir not listed holds no image.
    try:
        listed = set(os.listdir(evaluations_dir))
    except FileNotFoundError:
        listed = set()

    # Each record's own paths only, so records can go concurrently (see
    # RECORD_READ_THREADS: ~26ms of Lustre stats per record serially).
    def prune(record: dict[str, Any]) -> int:
        removed = 0
        paths = record.get("paths") or {}
        retained = evaluation_key(record) in keep
        dir_name = Path(paths.get("eval_dir") or "").name
        gone = bool(dir_name) and dir_name not in listed
        # A scored record that names no image was stripped to its record as
        # it was scored (retain_evaluation_detail() drops the paths), so its
        # dir holds nothing but metrics.json: no Lustre listing for it. That
        # was 80% of a production run's records (round 52).
        named = any(paths.get(key) for key in RETAINED_IMAGE_KEYS)
        for key in RETAINED_IMAGE_KEYS:
            recorded = paths.get(key)
            if not recorded:
                continue
            target = None if gone else _recorded_image_path(evaluations_dir, recorded)
            if target is None:
                # Already gone; the record must not keep pointing at it.
                paths.pop(key, None)
                continue
            if retained:
                continue
            target.unlink(missing_ok=True)
            paths.pop(key, None)
            removed += 1
        if retained or "error" in record or not named or gone:
            return removed
        eval_dir = Path(paths.get("eval_dir") or "")
        if not eval_dir.name:
            return removed
        # Stripped like an evaluation evicted as it was scored: its logs,
        # simulation.json and the emptied wsclean/ and r2d2/ go, and so does
        # the dir. Left behind they were ~3 inodes for each extreme a flush
        # published and the run did not keep, ~1,000 per 4,000-evaluation
        # WSClean run against the file quota (round 236).
        local = evaluations_dir / eval_dir.name
        try:
            entries = os.listdir(local)
        except (FileNotFoundError, NotADirectoryError):
            return removed
        removed += sum(name in PRUNED_EVALUATION_LOGS for name in entries)
        _strip_to_record(local)
        return removed

    # Only a record naming an image in a dir still listed touches Lustre: ~1%
    # of them. The rest go inline, since a future each cost ~1s of the pool's
    # own overhead at 78k records (round 201).
    def touches_lustre(record: dict[str, Any]) -> bool:
        paths = record.get("paths") or {}
        dir_name = Path(paths.get("eval_dir") or "").name
        return (not dir_name or dir_name in listed) and any(
            paths.get(key) for key in RETAINED_IMAGE_KEYS)

    slow, removed = [], 0
    for record in records:
        paths = record.get("paths") or {}
        if not any(paths.get(key) for key in RETAINED_IMAGE_KEYS):
            continue  # prune() of a record naming no image does nothing: ~90% of them (round 269)
        if touches_lustre(record):
            slow.append(record)
        else:
            removed += prune(record)
    with ThreadPoolExecutor(RECORD_READ_THREADS) as pool:
        return removed + sum(pool.map(prune, slow))



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
        assert json.loads(flat_record_path(scored).read_text())["paths"] == written["paths"]

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
    # Path().name, whatever the spelling (round 269 skips building the Path).
    for spelled in ("/run/evaluations/eval-0007-aaa", "eval-0007-aaa", "/run//eval-1/", "/run/eval-1/.",
                    "/run/..", "/", "a/./b", Path("/run/evaluations/eval-0007-aaa")):
        assert evaluation_key({"paths": {"eval_dir": spelled}}) == Path(spelled).name, spelled

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
            for name in PRUNED_EVALUATION_LOGS + ("simulation.json",):
                (eval_dir / name).write_text(name)
            made.append({"eval_id": i, "objective": i / 60, "paths": paths})
        # A failure keeps every artefact whatever its rank.
        failed_dir = evaluations / "eval-0099-x"
        (failed_dir / "wsclean").mkdir(parents=True)
        for name in PRUNED_EVALUATION_LOGS:
            (failed_dir / name).write_text(name)
        made.append({"eval_id": 99, "objective": FAILURE_OBJECTIVE, "error": "wsclean failed",
                     "paths": {"eval_dir": str(failed_dir)}})
        # A rank's extreme it later evicted: stripped, dir and all, but its
        # record still names images. Mid-ranked, so the keep set is unchanged.
        evicted = {"eval_id": 98, "objective": 0.505, "paths": {
            "eval_dir": str(evaluations / "eval-0098-x"),
            "image": str(evaluations / "eval-0098-x" / "wsclean" / "recon-image.fits")}}

        removed = prune_run_artefacts(evaluations, made + [evicted])
        # Gone already: no file to count, and no path left to name it.
        assert evicted["paths"] == {"eval_dir": str(evaluations / "eval-0098-x")}, evicted
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

        # Logs go with the images, and a failure keeps all of them. A dropped
        # one is stripped dir and all, as if evicted when it was scored.
        for rec in made:
            eval_dir = Path(rec["paths"]["eval_dir"])
            kept = evaluation_key(rec) in keep or "error" in rec
            assert eval_dir.is_dir() == kept, (rec["eval_id"], "dir")
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
    assert clean_convergence("2026-Aug-29 10:19:02.105500 Gridded visibility count: 4140\n"
                             "Gridded visibility count: 4140\n") == {"gridded_visibilities": 4140}
    assert clean_convergence("") == {}
    assert clean_convergence_from(Path("/nonexistent/wsclean.stdout.log")) == {}

    print("clean convergence self-check passed")


def _busy_dir() -> Path | None:
    root = os.environ.get("NS_SCRATCH_DIR", "")
    return Path(root) / ".busy" if root else None


def _busy_marker() -> Path | None:
    busy_dir = _busy_dir()
    return busy_dir / str(os.getpid()) if busy_dir else None


def mark_evaluation_start() -> None:
    global _EVALUATION_STARTED_EPOCH
    _EVALUATION_STARTED_EPOCH = time.time()
    marker = _busy_marker()
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


def write_evaluation_record(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    global _EVALUATION_STARTED_EPOCH
    if _EVALUATION_STARTED_EPOCH is not None:
        timing = record.setdefault("timing", {})
        timing["started_epoch"] = _EVALUATION_STARTED_EPOCH
        timing["ended_epoch"] = time.time()
        _EVALUATION_STARTED_EPOCH = None
        marker = _busy_marker()
        if marker is not None:
            marker.unlink(missing_ok=True)
    keeping = prune_evaluation_artefacts(eval_dir, record)
    kept = retain_evaluation_detail(eval_dir, record) or keeping
    if kept and not keeping and not _always_retained(eval_dir, record):
        hold_evaluation_scratch(eval_dir, record)
    else:
        # The 1 in NS_KEEP_DETAIL_EVERY kept whole is ~7 files to Lustre,
        # ~0.2s, and the rank sat idle through it: ~2% of a 36-rank search
        # (docs/csd3-speed.md, round 95).
        publish_evaluation_scratch(eval_dir, record, kept, defer=(
            (lambda job: _submit_record_job(job, "publish"))
            if _RECORD_WRITER is not None else None))
    flush_held_evaluations()
    text = _json_text(record)
    _WRITTEN_RECORDS[os.path.abspath(eval_dir)] = text
    if _RECORD_WRITER is None:
        _write_record(flat_record_path(eval_dir), text)
    elif os.environ.get("NS_RECORD_APPEND", "1") == "1":
        line = f"{eval_dir.name}\t{json.dumps(record)}\n".encode()
        _submit_record_job(lambda: _append_record(eval_dir.parent, line))
    elif os.environ.get("NS_RECORD_COALESCE", "1") == "0":
        _submit_record_job(lambda: _write_record(flat_record_path(eval_dir), text))
    else:
        with _PENDING_LOCK:
            _PENDING_RECORDS.append((eval_dir, text, record))
        _submit_record_job(_write_pending_records)
    return record


# A flat writer that falls behind - a create in evaluations/ costs 2ms on a
# quiet RDS and up to 1.2s on a loaded one - held 58-67% of a run's records
# only in memory (round 244), and a kill lost them from the archive (161 of
# 4,933, round 245). So the writer takes every record queued when it gets to
# one: a lone record is a flat file, as before; a backlog is one append to
# the pack, NS_RECORD_APPEND's single write, so it never grows (round 247).
_PENDING_RECORDS: list[tuple[Path, str, dict[str, Any]]] = []
_PENDING_LOCK = threading.Lock()


def _write_pending_records() -> None:
    with _PENDING_LOCK:
        pending = _PENDING_RECORDS[:]
        _PENDING_RECORDS.clear()
    if len(pending) == 1:
        eval_dir, text, _ = pending[0]
        _write_record(flat_record_path(eval_dir), text)
    elif pending:
        _append_record(pending[0][0].parent, b"".join(
            f"{eval_dir.name}\t{json.dumps(record)}\n".encode() for eval_dir, _, record in pending))
        _WRITER_SECONDS["coalesced"] = _WRITER_SECONDS.get("coalesced", 0) + len(pending)


def _submit_record_job(job: Any, kind: str = "record") -> Any:
    for done in [future for future in _RECORD_WRITES if future.done()]:
        _RECORD_WRITES.remove(done)
        done.result()  # a failed write raises here, on the rank's own thread
    future = _RECORD_WRITER.submit(_timed_record_job, job, kind)
    future.kind = kind
    _RECORD_WRITES.append(future)
    return future


def _timed_record_job(job: Any, kind: str) -> None:
    started = time.monotonic()
    try:
        job()
    finally:
        _WRITER_SECONDS[kind] = _WRITER_SECONDS.get(kind, 0.0) + time.monotonic() - started


def _write_record(path: Path, text: str) -> None:
    _write_text_atomic(path, text)


# A run's writer appends each record to the pack instead of creating a flat
# file - one write on a descriptor held open, against a create and a rename in
# the evaluations dir (on RDS, 38 processes: 0.15-0.24 against 1.9-2.1 ms,
# worst 0.01 against 1.2 s). A kill loses none (rounds 245, 261), and at 113
# ranks on whole sapphire flat creates convoyed in the MDS (~40x the
# writer-seconds, load ~200) for +1.9% evals/s, so it is the default (round
# 264; NS_RECORD_APPEND=0 for the coalescing flat writer). O_APPEND keeps
# concurrent lines whole; the shared lock keeps them out of a pack writer's
# truncate of a torn last line, which takes it exclusively.
_PACK_FDS: dict[str, int] = {}


def _append_record(evaluations_dir: Path, line: bytes) -> None:
    import fcntl

    path = evaluations_dir / os.environ.get("NS_RECORD_PACK", RECORD_PACK)
    fd = _PACK_FDS.get(str(path))
    if fd is None:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        _PACK_FDS[str(path)] = fd
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise OSError(f"short append to {path}: {written} of {len(line)} bytes")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


# A record write (create, write, rename) costs ~50ms on CSD3's Lustre, a fifth
# of a WSClean evaluation, and the rank sat idle through it. A run hands it to
# one thread per rank instead; whatever reads the records back drains it first.
_RECORD_WRITER: Any = None
_RECORD_WRITES: list[Any] = []
# Every record this process wrote, by eval dir, for finished_run_records().
_WRITTEN_RECORDS: dict[str, str] = {}
# Seconds this rank's writer spent on each kind of job, for post_run_line(),
# and as cancelled_held a count: the held publishes it was spared (round 235).
_WRITER_SECONDS: dict[str, float] = {}


# The writer must not share the rank's pinned CPU. Every mkdir in evaluations/
# holds the directory's lock across its Lustre round trip, so a writer that then
# waits behind its rank's busy worker for the core stalls all 36 ranks' writers:
# ~570ms a mkdir there while scoring, against 0.13ms in a subdirectory
# (docs/csd3-speed.md, round 54).
def _unpin_thread() -> None:
    if _UNPINNED_CPUS:
        os.sched_setaffinity(0, _UNPINNED_CPUS)  # this thread only, on Linux


def write_records_in_background() -> None:
    global _RECORD_WRITER
    from concurrent.futures import ThreadPoolExecutor

    if _RECORD_WRITER is None:
        _RECORD_WRITER = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="record-writer", initializer=_unpin_thread)
        atexit.register(drain_record_writes)


def drain_record_writes(timeout: float | None = None, flush_held: bool = True) -> None:
    """Wait for queued record writes; back to writing inline. Raises the first
    failed write, unless `timeout` is set (abort paths, which must not hang).
    `flush_held=False` leaves the held extremes to finished_run_records()."""
    global _RECORD_WRITER
    import concurrent.futures

    if timeout is None and flush_held:
        flush_held_evaluations(force=True)
    writer, _RECORD_WRITER = _RECORD_WRITER, None
    if writer is None:
        return
    pending = list(_RECORD_WRITES)
    _RECORD_WRITES.clear()
    concurrent.futures.wait(pending, timeout=timeout)
    writer.shutdown(wait=timeout is None)
    if timeout is None:
        for future in pending:
            future.result()


# Keeping every evaluation's logs and images until a run ends costs 10-15 files
# an evaluation, and CSD3's hpc-work allows a million: a 112-rank run reaches it
# within hours (docs/csd3-experiments.md). So as it goes, each rank strips a
# successful evaluation to its metrics.json unless it is one of that rank's
# IMAGE_KEEP_ENDS lowest or highest objectives so far - the run's own are always
# among those, so prune_run_artefacts() still finds them - or one of the 1 in
# NS_KEEP_DETAIL_EVERY kept whole, which `./ri profile` reads its logs from.
_LOWEST: list[tuple[float, str]] = []
_HIGHEST: list[tuple[float, str]] = []


def _keeps_detail(key: str) -> bool:
    import hashlib

    every = int(os.environ.get("NS_KEEP_DETAIL_EVERY", "100"))
    return every <= 1 or int(hashlib.sha256(key.encode()).hexdigest(), 16) % every == 0


def _strip_to_record(eval_dir: Path) -> None:
    import shutil

    try:
        entries = list(eval_dir.iterdir())
    except FileNotFoundError:
        return  # stripped already, or never published
    for entry in entries:
        # A pre-round-92 record, or its .partial while a writer is on it.
        if entry.name.startswith("metrics.json"):
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    try:
        eval_dir.rmdir()  # its record is beside it, not in it
    except OSError:
        pass


def _always_retained(eval_dir: Path, record: dict[str, Any]) -> bool:
    return (os.environ.get("NS_KEEP_ALL_IMAGES", "0") != "0"
            or os.environ.get("NS_KEEP_MEASUREMENT_SETS", "0") != "0"
            or "error" in record or record.get("objective") is None
            or _keeps_detail(eval_dir.name))


def retain_evaluation_detail(eval_dir: Path, record: dict[str, Any]) -> bool:
    """False if this evaluation is stripped to its record."""
    import heapq
    import shutil

    if _always_retained(eval_dir, record):
        return True
    key, objective = eval_dir.name, float(record["objective"])
    evicted = set()
    heapq.heappush(_LOWEST, (-objective, key))
    heapq.heappush(_HIGHEST, (objective, key))
    if len(_LOWEST) > IMAGE_KEEP_ENDS:
        evicted.add(heapq.heappop(_LOWEST)[1])
    if len(_HIGHEST) > IMAGE_KEEP_ENDS:
        evicted.add(heapq.heappop(_HIGHEST)[1])
    evicted -= {k for _, k in _LOWEST} | {k for _, k in _HIGHEST}
    for name in evicted:
        if name in _HELD:
            shutil.rmtree(_HELD.pop(name)[0], ignore_errors=True)
        elif (cancelled := _cancel_held_publish(name)) is not None:
            # Its copy to Lustre had not started, so nothing of it is there:
            # dropped in tmpfs, with no publish or strip for the writer.
            shutil.rmtree(cancelled[0], ignore_errors=True)
        # This evaluation's own dir is empty, or not made yet, while its
        # output is still in scratch: no Lustre walk for it (round 14).
        elif name != key or evaluation_scratch_dir(eval_dir) is None:
            # One already on Lustre: its walk, unlinks and rmdir go to the
            # record writer, after its own move there. On the rank's thread a
            # Lustre metadata stall froze every rank at once for 15-48s,
            # ~19% of a 1,500-dead search (docs/csd3-speed.md, round 232).
            _defer_lustre(lambda path=eval_dir.parent / name: _strip_to_record(path), "strip")
    if key in evicted:
        for image_key in RETAINED_IMAGE_KEYS:
            (record.get("paths") or {}).pop(image_key, None)
    return key not in evicted



# A rank's extremes so far are mostly evicted again: replaying the 71k-evaluation
# R2D2 run's objectives, 9-13% of all evaluations at 24-36 ranks enter one, and
# publishing each to Lustre and stripping it later cost ~0.3s of the rank's
# main thread (docs/csd3-speed.md, round 47). So with scratch they wait in
# tmpfs under .held/, an eviction there is free, and the survivors go to Lustre
# every NS_RETAIN_FLUSH_SECONDS and, at the end, only those the run keeps
# (finished_run_records()). A job that dies loses at most that much of its
# extremes' images; the records survive and prune_run_artefacts() drops the
# paths that no longer exist.
_HELD: dict[str, tuple[Path, Path]] = {}
_HELD_SINCE: float | None = None
# Held extremes the periodic flush queued on the writer, by name. Most are
# evicted again, and on a loaded Lustre the writer runs minutes behind, so the
# eviction usually comes first: cancelling the publish then saves its copies
# and the strip that would undo them, ~27% of the writer's time (round 235).
_PUBLISHING: dict[str, tuple[Any, Path, Path]] = {}


def _cancel_held_publish(name: str) -> tuple[Path, Path] | None:
    """(held, eval_dir) if `name`'s queued publish was cancelled, else None."""
    entry = _PUBLISHING.pop(name, None)
    if entry is None or not entry[0].cancel():
        return None
    _RECORD_WRITES.remove(entry[0])  # a cancelled future's result() raises
    _WRITER_SECONDS["cancelled_held"] = _WRITER_SECONDS.get("cancelled_held", 0) + 1
    return entry[1:]


def hold_evaluation_scratch(eval_dir: Path, record: dict[str, Any]) -> None:
    global _HELD_SINCE
    scratch = evaluation_scratch_dir(eval_dir)
    if scratch is None or not scratch.is_dir():
        return
    held = scratch.parent / ".held" / scratch.name
    held.parent.mkdir(exist_ok=True)
    scratch.rename(held)
    _HELD[eval_dir.name] = (held, eval_dir)
    if _HELD_SINCE is None:
        _HELD_SINCE = time.monotonic()
    paths = record.get("paths") or {}
    for key, value in paths.items():
        if isinstance(value, str) and Path(value).is_relative_to(scratch):
            paths[key] = str(eval_dir / Path(value).relative_to(scratch))


def flush_held_evaluations(force: bool = False, keep: set[str] | None = None) -> None:
    """Move held extremes to Lustre; with `keep`, only those the run keeps."""
    import shutil

    global _HELD_SINCE
    if not _HELD or not force and (
            time.monotonic() - (_HELD_SINCE or 0.0)
            < float(os.environ.get("NS_RETAIN_FLUSH_SECONDS", "600"))):
        return
    for name, (held, eval_dir) in _HELD.items():
        if not held.is_dir():
            continue  # a self-check's scratch, already gone
        if keep is not None and name not in keep:
            shutil.rmtree(held, ignore_errors=True)
            continue
        # The periodic flush is the record writer's (round 232); a forced
        # one is the end of a run, whose readers need the files there.
        publish = lambda held=held, eval_dir=eval_dir: _publish_held(held, eval_dir)  # noqa: E731
        if force or _RECORD_WRITER is None:
            publish()
        else:
            _PUBLISHING[name] = (_submit_record_job(publish, "held"), held, eval_dir)
    for name in [name for name, entry in _PUBLISHING.items() if entry[0].done()]:
        del _PUBLISHING[name]
    _HELD.clear()
    _HELD_SINCE = None


def _publish_held(held: Path, eval_dir: Path) -> None:
    import shutil

    eval_dir.mkdir(parents=True, exist_ok=True)
    for produced in held.iterdir():
        shutil.move(str(produced), eval_dir / produced.name, copy_function=shutil.copyfile)
    held.rmdir()


def _defer_lustre(job: Any, kind: str) -> None:
    """Run a Lustre job on the record writer if there is one, else now."""
    if _RECORD_WRITER is None:
        job()
    else:
        _submit_record_job(job, kind)


def self_check_streaming_retention() -> None:
    import tempfile

    global IMAGE_KEEP_ENDS
    saved = IMAGE_KEEP_ENDS, os.environ.get("NS_KEEP_DETAIL_EVERY")
    IMAGE_KEEP_ENDS = 3
    os.environ["NS_KEEP_DETAIL_EVERY"] = "10"
    try:
        # Same outcome whether an evaluation writes beside its record or in
        # scratch that is published only if it is kept, and whether records
        # are written inline or by the run's background writer.
        os.environ["NS_RETAIN_FLUSH_SECONDS"] = "1e9"
        for scratch_root, background in ((None, False), ("scratch", False), ("scratch", True)):
            _LOWEST.clear()
            _HIGHEST.clear()
            _HELD.clear()
            with tempfile.TemporaryDirectory() as tmp:
                evaluations = Path(tmp) / "evaluations"
                if scratch_root:
                    os.environ["NS_SCRATCH_DIR"] = str(Path(tmp) / scratch_root)
                if background:
                    write_records_in_background()
                evaluations.mkdir()
                objectives = {}
                for index in range(300):
                    eval_dir = evaluations / f"eval-{index:04d}"
                    # As simulate_measurement_set(): with scratch, nothing
                    # makes the eval dir before its record or publish does.
                    if not scratch_root:
                        eval_dir.mkdir()
                    work = evaluation_scratch_dir(eval_dir) or eval_dir
                    (work / "wsclean").mkdir(parents=True)
                    (work / "wsclean" / "recon-image.fits").write_text("x")
                    (work / "wsclean.stdout.log").write_text("x")
                    objective = ((index * 7919) % 300) / 10
                    objectives[eval_dir.name] = objective
                    record = {"objective": objective,
                              "paths": {"eval_dir": str(eval_dir),
                                        "image": str(work / "wsclean" / "recon-image.fits")}}
                    write_evaluation_record(eval_dir, record)
                    # A kept evaluation's move to Lustre is the writer's job.
                    for future in list(_RECORD_WRITES):
                        future.result()
                    assert not (Path(tmp) / "scratch" / eval_dir.name).exists(), "scratch outlived scoring"
                    if eval_dir.name in _HELD:
                        # Held in tmpfs: nothing of it on Lustre but its record.
                        assert not (eval_dir / "wsclean").exists(), "a held evaluation was published"
                        assert (Path(tmp) / "scratch" / ".held" / eval_dir.name / "wsclean").is_dir()
                    if not (eval_dir / "wsclean").exists() and eval_dir.name not in _HELD:
                        assert "image" not in record["paths"], "a stripped record still names its image"
                    else:
                        assert record["paths"]["image"] == str(eval_dir / "wsclean" / "recon-image.fits")
                if scratch_root:
                    assert _HELD, "no extreme was held in scratch"
                    assert len(_HELD) <= 6, sorted(_HELD)
                held_root = Path(tmp) / "scratch" / ".held"
                if background:
                    # Another rank's extreme so far that the run does not keep:
                    # dropped in tmpfs, never published. Its record is read
                    # from disk, as an earlier segment's would be.
                    other = evaluations / "eval-other"
                    (held_root / other.name / "wsclean").mkdir(parents=True)
                    _HELD[other.name] = (held_root / other.name, other)
                    _write_record(flat_record_path(other), _json_text(
                        {"objective": 15.05, "paths": {"eval_dir": str(other)}}))
                    # Its periodic publish still queued behind a stuck writer
                    # when the run ends: cancelled, and dropped with the rest.
                    import threading

                    gate = threading.Event()
                    _submit_record_job(gate.wait)
                    os.environ["NS_RETAIN_FLUSH_SECONDS"] = "0"
                    flush_held_evaluations()
                    os.environ["NS_RETAIN_FLUSH_SECONDS"] = "1e9"
                    assert other.name in _PUBLISHING and not _HELD
                    threading.Timer(0.5, gate.set).start()
                    records = finished_run_records(evaluations)
                    assert records is not None and len(records) == len(objectives) + 1
                    assert not other.exists()
                    flat_record_path(other).unlink()
                else:
                    drain_record_writes()
                assert not _HELD and not (held_root.exists() and any(held_root.iterdir()))
                ordered = sorted(objectives, key=objectives.get)
                must = set(ordered[:3]) | set(ordered[-3:]) | {k for k in objectives if _keeps_detail(k)}
                for key in objectives:
                    if key in must:
                        kept = sorted(p.name for p in (evaluations / key).iterdir())
                        assert kept == ["wsclean", "wsclean.stdout.log"], (key, kept)
                    else:
                        # Stripped to its record, which is beside the dir: no dir.
                        assert not (evaluations / key).exists(), key
                    written = _record_reader(evaluations)[1](key)  # flat or appended
                    assert written["objective"] == objectives[key], (key, written)

                if scratch_root:
                    # Once NS_RETAIN_FLUSH_SECONDS have passed, the next record publishes.
                    os.environ["NS_RETAIN_FLUSH_SECONDS"] = "0"
                    eval_dir = evaluations / "eval-best"
                    (evaluation_scratch_dir(eval_dir) / "wsclean").mkdir(parents=True)
                    write_evaluation_record(eval_dir, {"objective": 1e3, "paths": {}})
                    os.environ["NS_RETAIN_FLUSH_SECONDS"] = "1e9"
                    assert not _HELD and (eval_dir / "wsclean").is_dir()

                if background:
                    # The periodic flush, and stripping an extreme it published,
                    # are the writer's: with it stuck (a Lustre stall), the
                    # rank's thread touches neither (round 232).
                    import threading

                    names = [n for n in (f"eval-hi-{i}" for i in range(40)) if not _keeps_detail(n)]
                    published = evaluations / names[0]  # an extreme so far, on Lustre
                    (published / "wsclean").mkdir(parents=True)
                    retain_evaluation_detail(published, {"objective": 1.5e3})
                    write_records_in_background()
                    gate = threading.Event()
                    _submit_record_job(gate.wait)
                    held = evaluations / "eval-held"
                    (evaluation_scratch_dir(held) / "wsclean").mkdir(parents=True)
                    hold_evaluation_scratch(held, {"paths": {}})
                    try:
                        os.environ["NS_RETAIN_FLUSH_SECONDS"] = "0"
                        flush_held_evaluations()
                        os.environ["NS_RETAIN_FLUSH_SECONDS"] = "1e9"
                        for rank, name in enumerate(names[1:IMAGE_KEEP_ENDS + 1]):
                            retain_evaluation_detail(evaluations / name, {"objective": 2e3 + rank})
                        assert not _HELD and not (held / "wsclean").exists(), "the rank's thread published"
                        assert (published / "wsclean").is_dir(), "the rank's thread stripped"
                    finally:
                        gate.set()
                    drain_record_writes(flush_held=False)
                    assert (held / "wsclean").is_dir() and not published.exists()

                    # An extreme the periodic flush queued but the writer has
                    # not reached, evicted again: never copied to Lustre and
                    # never stripped, just dropped in tmpfs (round 235).
                    _LOWEST.clear()
                    _HIGHEST.clear()
                    write_records_in_background()
                    gate = threading.Event()
                    _submit_record_job(gate.wait)
                    try:
                        evicted = evaluations / names[IMAGE_KEEP_ENDS + 1]
                        (evaluation_scratch_dir(evicted) / "wsclean").mkdir(parents=True)
                        retain_evaluation_detail(evicted, {"objective": 50.0})
                        hold_evaluation_scratch(evicted, {"paths": {}})
                        os.environ["NS_RETAIN_FLUSH_SECONDS"] = "0"
                        flush_held_evaluations()
                        os.environ["NS_RETAIN_FLUSH_SECONDS"] = "1e9"
                        assert evicted.name in _PUBLISHING
                        others = names[IMAGE_KEEP_ENDS + 2:3 * IMAGE_KEEP_ENDS + 2]
                        for rank, name in enumerate(others):
                            objective = 100.0 + rank if rank < IMAGE_KEEP_ENDS else rank
                            retain_evaluation_detail(evaluations / name, {"objective": objective})
                        assert evicted.name not in _PUBLISHING, "the evicted extreme's publish stayed queued"
                        assert not (held_root / evicted.name).exists(), "the evicted extreme outlived tmpfs"
                        assert "held" not in [f.kind for f in _RECORD_WRITES if not f.done()]
                    finally:
                        gate.set()
                    drain_record_writes(flush_held=False)
                    assert not evicted.exists(), "a cancelled publish reached Lustre"

                if background:
                    # A write that fails in the background still fails the run.
                    write_records_in_background()
                    blocked = evaluations / "blocked" / "eval-blocked"
                    flat_record_path(blocked).with_name(blocked.name + ".json.partial").mkdir(parents=True)
                    (blocked.parent / RECORD_PACK).mkdir()  # and the append's
                    write_evaluation_record(blocked, {"objective": 1.0})
                    try:
                        drain_record_writes()
                    except IsADirectoryError:
                        pass
                    else:
                        raise AssertionError("a failed background record write was lost")

                # A dead worker's logs outlive its scratch; its MS does not.
                eval_dir = evaluations / "eval-9999-died"
                if not scratch_root:
                    eval_dir.mkdir()
                work = evaluation_scratch_dir(eval_dir) or eval_dir
                (work / "sim.ms").mkdir(parents=True)
                (work / "simulate.stderr.log").write_text("Traceback")
                salvage_evaluation_logs(eval_dir)
                assert (eval_dir / "simulate.stderr.log").read_text() == "Traceback"
                if scratch_root:
                    assert not (eval_dir / "sim.ms").exists() and not work.exists()
            os.environ.pop("NS_SCRATCH_DIR", None)
    finally:
        drain_record_writes(timeout=0)
        os.environ.pop("NS_SCRATCH_DIR", None)
        os.environ.pop("NS_RETAIN_FLUSH_SECONDS", None)
        _HELD.clear()
        _PUBLISHING.clear()
        IMAGE_KEEP_ENDS = saved[0]
        if saved[1] is None:
            os.environ.pop("NS_KEEP_DETAIL_EVERY", None)
        else:
            os.environ["NS_KEEP_DETAIL_EVERY"] = saved[1]
        _LOWEST.clear()
        _HIGHEST.clear()
    print("streaming retention self-check passed")


_SIMULATE_WORKERS: dict[str, "FifoWorker"] = {}


def _children_of(pid: int) -> list[int]:
    """A /proc walk rather than pgrep: the ranks run inside polychord.sif,
    which has no procps, and Apptainer shares the host's pid namespace, so
    the pids are the same ones the pool wrote. A macOS host running the
    self-check has no /proc but does have pgrep."""
    if not os.path.isdir("/proc"):
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
        return [int(child) for child in out.split()]
    children = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text()
        except OSError:
            continue
        # `pid (comm) state ppid ...`; comm may hold spaces and parentheses.
        if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
            children.append(int(entry))
    return children


def _kill_process_tree(pid: int) -> None:
    """SIGKILL `pid` and its children - the worker and the meqserver it drives."""
    for victim in (*_children_of(pid), pid):
        try:
            os.kill(victim, signal.SIGKILL)
        except OSError:
            pass


class FifoWorker:
    def __init__(self, write_fd: int, reply_path: Path, base: Path) -> None:
        self.stdin = os.fdopen(write_fd, "w")
        # Opening a FIFO blocks until the other end is open, so this must be the
        # same order serve() uses - request pipe first, reply pipe second.
        self.stdout = reply_path.open("r")
        self.base = base
        self.fresh = True

    def terminate(self) -> None:
        """Let go of both ends.

        Both, not just the request end: a write end held open past the worker's
        death keeps the pipe's buffer alive for the replacement to read, and a
        request nobody was there to take would then be answered twice - once
        for the retry that resends it, and once more into the next request's
        reply. Closed, the replacement starts from an empty pipe.
        """
        for stream in (self.stdin, self.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def kill(self) -> None:
        """Kill a wedged worker; the pool starts a fresh one on the same FIFOs."""
        try:
            pid = int(Path(f"{self.base}.pid").read_text())
        except (OSError, ValueError):
            pid = 0
        if pid > 0:
            _kill_process_tree(pid)
        self.terminate()


def _forget(workers: dict[str, "FifoWorker"], key: str) -> None:
    """Drop a dead or wedged worker so the next attempt reconnects to the pool."""
    worker = workers.pop(key, None)
    if worker is not None:
        worker.terminate()


# The pools come up alongside the ranks: a SIF to start, torch or Timba to
# import, and on a cluster node every rank's worker doing it off the same
# filesystem at once. A missing pool is only ever paid for in full. A worker
# opens its FIFO only after its warm-up, and on a 112-rank CSD3 node all of them
# warm up at once, after every restart too: 60s was not enough (E5, E8 in
# docs/csd3-experiments.md), and a rank that gives up aborts the whole run.
POOL_CONNECT_SECONDS = float(os.environ.get("NS_POOL_CONNECT_SECONDS", "300"))
# Added to a newly connected worker's first reply bound, for the same reason:
# the bounds below are sized for a warm worker on a quiet node.
WORKER_FIRST_REPLY_SLACK = float(os.environ.get("NS_WORKER_FIRST_REPLY_SLACK", "120"))
# A reconnect - this rank has had a worker on these FIFOs - gives up this soon
# once the pid file names a dead worker nothing has replaced. A live pool's
# loop rewrites it the moment it spawns one, so only a pool that is gone (its
# process group killed) leaves it dead. Without this a dead pool cost each
# rank the whole POOL_CONNECT_SECONDS before the retry that restores it.
POOL_GONE_SECONDS = float(os.environ.get("NS_POOL_GONE_SECONDS", "30"))
_CONNECTED_BASES: set[str] = set()


def _pid_file_dead(base: Path) -> str | None:
    """The pid file's contents if they name a process that is gone, else None."""
    try:
        text = Path(f"{base}.pid").read_text()
        os.kill(int(text), 0)
    except ProcessLookupError:
        return text
    except (OSError, ValueError):
        pass
    return None


def _connect_shell_started_worker(fifo_dir_var: str) -> FifoWorker | None:
    """Attach to this rank's pooled worker, or return None if it never appears."""
    fifo_dir = os.environ.get(fifo_dir_var)
    if not fifo_dir:
        return None
    base = Path(fifo_dir) / str(mpi_rank())
    reconnect = str(base) in _CONNECTED_BASES
    # Nonblocking open returns ENXIO until the worker opens its end.
    deadline = time.monotonic() + POOL_CONNECT_SECONDS
    # The pid file is on Lustre, so it is read once a second, not per poll.
    next_pid_check, dead_pid, dead_since = 0.0, None, 0.0
    while True:
        try:
            write_fd = os.open(f"{base}.in", os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            now = time.monotonic()
            if now > deadline:
                return None
            if reconnect and now >= next_pid_check:
                next_pid_check = now + 1.0
                dead = _pid_file_dead(base)
                if dead is None or dead != dead_pid:
                    dead_pid, dead_since = dead, now
                elif now - dead_since > POOL_GONE_SECONDS:
                    return None
            time.sleep(0.002)
            continue
        _CONNECTED_BASES.add(str(base))
        os.set_blocking(write_fd, True)
        if _PINNED_CPU is not None:
            # The pool loop writes the pid at spawn, before the worker's warm-up
            # opens the FIFO, so it is this worker's. Forks inherit it.
            try:
                os.sched_setaffinity(int(Path(f"{base}.pid").read_text()), {_PINNED_CPU})
            except (OSError, ValueError):
                pass
        return FifoWorker(write_fd, Path(f"{base}.out"), base)


# A rank and its workers take turns - it waits while they run - so sharing one
# CPU costs nothing, and keeps the MS the simulator just wrote in that core's
# cache for WSClean. Left to the scheduler on a 36-rank CSD3 node they wander:
# steady evals/s 99.7-117.0 unpinned against 121.5-131.8 pinned (round 16 of
# docs/csd3-speed.md). The CPU is claimed with a lock held for the rank's
# life, so two searches sharing one allocation never pin onto the same one.
_PINNED_CPU: int | None = None
_UNPINNED_CPUS: list[int] = []


def pin_rank_cpu() -> int | None:
    """Claim a CPU for this rank and its workers, or None if there are too few."""
    global _PINNED_CPU, _UNPINNED_CPUS
    import fcntl

    if not hasattr(os, "sched_setaffinity"):  # macOS
        return None
    cpus = sorted(os.sched_getaffinity(0))
    ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
    if os.environ.get("NS_PIN_CPUS", "1") == "0" or not 0 < ranks <= len(cpus):
        return None
    start = mpi_rank() % len(cpus)
    for cpu in cpus[start:] + cpus[:start]:
        try:
            fd = os.open(f"/dev/shm/ri-cpu-{cpu}.lock", os.O_RDWR | os.O_CREAT, 0o666)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        # fd stays open: the lock is released when this rank exits.
        os.sched_setaffinity(0, {cpu})
        _PINNED_CPU, _UNPINNED_CPUS = cpu, cpus
        return cpu
    return None


def _process_tree_tids(pid: int | str) -> list[int]:
    """Every thread of `pid` and of its descendants."""
    tids, pending = [], [str(pid)]
    while pending:
        for task in Path(f"/proc/{pending.pop()}/task").glob("*"):
            tids.append(int(task.name))
            try:
                pending.extend((task / "children").read_text().split())
            except OSError:
                pass
    return tids


def _set_tree_affinity(pid: int | str, cpus: set[int]) -> None:
    for tid in _process_tree_tids(pid):
        try:
            os.sched_setaffinity(tid, cpus)
        except OSError:
            pass


# On a whole node each R2D2 worker sits on one socket (r2d2_serve.enter_node),
# but mpirun binds its rank to a socket by its own rule, the far one for about
# half of them, and the simulate worker floats across both: each read what the
# other wrote from the far socket's memory. Two socket-bound searches ran +9%
# over one whole-node search (round 176).
def follow_r2d2_worker_node(r2d2_image: str, meqtrees_image: str) -> set[int] | None:
    """Move this rank and its simulate worker onto its R2D2 worker's CPUs."""
    r2d2, simulate = _R2D2_WORKERS.get(r2d2_image), _SIMULATE_WORKERS.get(meqtrees_image)
    if r2d2 is None or not hasattr(os, "sched_setaffinity") or os.environ.get("NS_PIN_CPUS", "1") == "0":
        return None
    try:
        cpus = os.sched_getaffinity(int(Path(f"{r2d2.base}.pid").read_text()))
    except (OSError, ValueError):
        return None
    # Already there, or a worker free to run wherever this rank may: stay put.
    if cpus >= os.sched_getaffinity(0):
        return None
    _set_tree_affinity(os.getpid(), cpus)
    if simulate is not None:
        try:
            _set_tree_affinity(Path(f"{simulate.base}.pid").read_text().strip(), cpus)
        except OSError:
            pass
    return cpus


def simulate_worker(meqtrees_image: str, platform: str) -> FifoWorker:
    """This rank's warm simulator worker, one of the pool the run script started."""
    if meqtrees_image not in _SIMULATE_WORKERS:
        worker = _connect_shell_started_worker("NS_SIMULATE_FIFO_DIR")
        if worker is None:
            raise WorkerDied(f"no simulate worker for rank {mpi_rank()} under NS_SIMULATE_FIFO_DIR")
        atexit.register(worker.terminate)
        _SIMULATE_WORKERS[meqtrees_image] = worker
    return _SIMULATE_WORKERS[meqtrees_image]


_R2D2_WORKERS: dict[str, "FifoWorker"] = {}


def r2d2_worker(r2d2_image: str, platform: str, checkpoints_dir: str) -> FifoWorker:
    """This rank's long-lived R2D2 worker, one of the pool the run script started."""
    if r2d2_image not in _R2D2_WORKERS:
        worker = _connect_shell_started_worker("NS_R2D2_FIFO_DIR")
        if worker is None:
            raise WorkerDied(f"no R2D2 worker for rank {mpi_rank()} under NS_R2D2_FIFO_DIR")
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
            _forget(_R2D2_WORKERS, r2d2_image)
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
            worker.kill()
        # The worker died or went silent mid-request; drop it so the next
        # attempt reconnects to the one the pool starts in its place.
        _forget(_R2D2_WORKERS, r2d2_image)
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
    reply_timeout: float | None = None,
) -> int:
    if reply_timeout is None:
        reply_timeout = simulate_reply_timeout()
    for attempt in worker_attempts():
        worker = simulate_worker(meqtrees_image, platform)
        if not worker_send(worker.stdin, json.dumps(request) + "\n"):
            _forget(_SIMULATE_WORKERS, meqtrees_image)
            continue
        reply = worker_reply(worker.stdout, reply_timeout + (WORKER_FIRST_REPLY_SLACK if getattr(worker, "fresh", False) else 0))
        if reply:
            worker.fresh = False
            return int(json.loads(reply)["returncode"])
        if reply is None:
            worker.kill()
        _forget(_SIMULATE_WORKERS, meqtrees_image)
    # Appended, not written: the last attempt's worker leaves its traceback in
    # this file, and that is the only thing on disk that says why it died.
    # Clobbering it left "gave no reply" as the entire record of the failure.
    with stderr_path.open("a") as note:
        note.write(f"FATAL: simulate worker gave no reply, {len(WORKER_RETRY_DELAYS)} times\n")
    return WORKER_DIED


def simulate_measurement_set(
    params: dict[str, Any],
    eval_dir: Path,
    meqtrees_image: str,
    platform: str,
    r2d2_mat: bool = False,
) -> tuple[Path, list[str], subprocess.CalledProcessError | None]:
    scratch = evaluation_scratch_dir(eval_dir)
    if scratch is None:
        eval_dir.mkdir(parents=True, exist_ok=False)
    else:
        # The eval dir on Lustre is left to whatever first writes into it
        # (the record writer, or publishing a kept evaluation): one slow
        # metadata call there used to stall every rank (csd3-speed.md, round 14).
        # A restart that re-runs an evaluation whose scratch survived would hit
        # require_clean_output() in the simulator; the run owns this directory,
        # so clearing it is safe and normally costs an ENOENT.
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
    work = scratch or eval_dir
    ms_path = work / "sim.ms"
    n_times = evaluation_time_samples(params)
    sim_stdout = work / "simulate.stdout.log"
    sim_stderr = work / "simulate.stderr.log"
    sim_cmd = [
        "--output-ms",
        str(ms_path),
        "--metadata-json",
        str(work / "simulation.json"),
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
        # `=` because argparse reads a separate "-7.4e-06" as an option, not a value.
        f"--source-l-arcsec={params['source_l_arcsec']}",
        f"--source-m-arcsec={params['source_m_arcsec']}",
        f"--declination-deg={params['declination_deg']}",
        "--integration-seconds",
        str(params["integration_seconds"]),
        "--dynamic-range",
        str(params["dynamic_range"]),
        "--seed",
        str(params["noise_seed"]),
        # The worker can see neither the 19 ranks it competes with nor the size
        # of the job it is about to be handed, so the rank that can see both
        # sizes the bounds here - and the ladder then holds by construction
        # rather than by two constants in two images agreeing.
        "--predict-wait-seconds",
        str(predict_wait_seconds(n_times)),
        "--observation-horizon-minutes",
        str(observation_horizon_minutes()),
    ]
    if r2d2_mat:
        # Written by the simulate request itself, beside the MS: a separate
        # convert request re-read the MS for 25ms typical, 170ms heavy (round 13).
        sim_cmd += ["--mat-path", str(work / "r2d2_data.mat")]
        # R2D2 reads only the .mat, so no MS is built unless one is being kept
        # (round 21: -25ms of a 75ms stage at 36 ranks, the .mat byte-identical).
        if os.environ.get("NS_KEEP_MEASUREMENT_SETS", "0") == "0":
            sim_cmd.append("--mat-only")
    returncode = simulate_worker_request(
        meqtrees_image,
        platform,
        {"argv": sim_cmd, "stdout": str(sim_stdout), "stderr": str(sim_stderr)},
        sim_stderr,
        # A miss builds the observation at the horizon, so that is the length to wait on.
        reply_timeout=simulate_reply_timeout(
            max(n_times, evaluation_time_samples({**params, "observation_minutes": observation_horizon_minutes()}))),
    )
    if returncode != 0:
        return ms_path, sim_cmd, subprocess.CalledProcessError(returncode, sim_cmd)
    return ms_path, sim_cmd, None


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

    original = {name: globals()[name] for name in ("simulate_worker", "WORKER_RETRY_DELAYS")}
    workers: list[Worker] = []

    def spawn(reply: str | None) -> Any:
        def worker(*_args: Any, **_kwargs: Any) -> Worker:
            workers.append(Worker(reply))
            return workers[-1]

        return worker

    try:
        # No delays and a bound nothing can meet, so the retry path runs in
        # milliseconds instead of waiting out the real per-request bound.
        globals()["WORKER_RETRY_DELAYS"] = (0.0, 0.0)

        globals()["simulate_worker"] = spawn(None)
        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = Path(tmp) / "simulate.stderr.log"
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path, reply_timeout=0.05) == WORKER_DIED
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
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path, reply_timeout=0.05) == 7
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
            assert simulate_worker_request("meqtrees", "linux/amd64", {"argv": []}, stderr_path, reply_timeout=0.05) == 7
            assert not stderr_path.exists()
        assert len(workers) == 2, len(workers)
        # Nothing to kill: it was already gone, which is why the write failed.
        assert not workers[0].killed
    finally:
        globals().update(original)

    print("worker timeout self-check passed")


def self_check_worker_pool_connect() -> None:
    global POOL_GONE_SECONDS
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / str(mpi_rank())
        os.mkfifo(f"{base}.in")
        os.mkfifo(f"{base}.out")
        # O_RDWR is the one open mode a FIFO never blocks on, and it puts the
        # reader on .in and the writer on .out that a live worker would.
        ends = [os.open(f"{base}.in", os.O_RDWR), os.open(f"{base}.out", os.O_RDWR)]
        # A stand-in for the pooled worker and the meqserver it drives, so the
        # kill path can be seen to take both.
        stand_in = subprocess.Popen(["sh", "-c", "sleep 30 & wait"])
        Path(f"{base}.pid").write_text(f"{stand_in.pid}\n")
        original = dict(os.environ)
        try:
            os.environ["NS_SIMULATE_FIFO_DIR"] = tmp
            os.environ["NS_R2D2_FIFO_DIR"] = tmp
            os.environ["NS_WSCLEAN_FIFO_DIR"] = tmp
            workers = [
                simulate_worker("meqtrees-self-check", "linux/amd64"),
                r2d2_worker("r2d2-self-check", "linux/amd64", "/checkpoints"),
                zygote_worker("wsclean-self-check"),
            ]
            for worker in workers:
                assert isinstance(worker, FifoWorker), type(worker)
                atexit.unregister(worker.terminate)
            deadline = time.monotonic() + 5
            while not _children_of(stand_in.pid):
                assert time.monotonic() < deadline, "the stand-in never started its child"
                time.sleep(0.02)
            grandchildren = _children_of(stand_in.pid)
            workers[0].kill()
            assert stand_in.wait(timeout=5) == -signal.SIGKILL, stand_in.returncode
            for child in grandchildren:
                while True:
                    try:
                        os.kill(child, 0)
                    except ProcessLookupError:
                        break
                    # Killed but not yet reaped by its parent, which is gone
                    # too; init will. Zombie is as dead as it gets. Without
                    # /proc (macOS) launchd reaps it and the kill above fails.
                    try:
                        if Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                            break
                    except FileNotFoundError:
                        pass
                    assert time.monotonic() < deadline, "the worker's child survived kill()"
                    time.sleep(0.02)
            # The pool gone with it: a reconnect gives up once the dead pid has
            # gone unreplaced for POOL_GONE_SECONDS, not POOL_CONNECT_SECONDS.
            for worker in workers:
                worker.terminate()
            for fd in ends:
                os.close(fd)
            ends = []
            saved_gone, POOL_GONE_SECONDS = POOL_GONE_SECONDS, 0.5
            try:
                started = time.monotonic()
                assert _connect_shell_started_worker("NS_SIMULATE_FIFO_DIR") is None
                assert time.monotonic() - started < 10, time.monotonic() - started
            finally:
                POOL_GONE_SECONDS = saved_gone
            # No pool at all is a WorkerDied, not a rank-started fallback.
            _SIMULATE_WORKERS.pop("meqtrees-self-check", None)
            del os.environ["NS_SIMULATE_FIFO_DIR"]
            try:
                simulate_worker("meqtrees-self-check", "linux/amd64")
            except WorkerDied:
                pass
            else:
                raise AssertionError("a rank without a pool must not pretend to have a worker")
        finally:
            _SIMULATE_WORKERS.pop("meqtrees-self-check", None)
            _R2D2_WORKERS.pop("r2d2-self-check", None)
            _ZYGOTE_WORKERS.pop("wsclean-self-check", None)
            os.environ.clear()
            os.environ.update(original)
            for fd in ends:
                os.close(fd)
            if stand_in.poll() is None:
                stand_in.kill()

    print("worker pool connect self-check passed")


def self_check_cpu_pinning() -> None:
    global _PINNED_CPU
    import subprocess
    import tempfile

    if not hasattr(os, "sched_setaffinity"):
        print("cpu pinning self-check skipped (no sched_setaffinity)")
        return
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 2:
        print("cpu pinning self-check skipped (one CPU)")
        return
    original = dict(os.environ)
    stand_in = subprocess.Popen(["sleep", "30"])
    claims = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "0"
            os.mkfifo(f"{base}.in")
            os.mkfifo(f"{base}.out")
            ends = [os.open(f"{base}.in", os.O_RDWR), os.open(f"{base}.out", os.O_RDWR)]
            Path(f"{base}.pid").write_text(f"{stand_in.pid}\n")
            os.environ.update(OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_SIZE="2", NS_SIMULATE_FIFO_DIR=tmp)
            # Two claims from one process are two lock holders, as two ranks
            # are: the second must not take the first one's CPU.
            for _ in range(2):
                _PINNED_CPU = None
                os.sched_setaffinity(0, allowed)
                claims.append(pin_rank_cpu())
            assert None not in claims and claims[0] != claims[1], claims
            assert os.sched_getaffinity(0) == {claims[1]}
            write_records_in_background()
            assert _RECORD_WRITER.submit(os.sched_getaffinity, 0).result() == allowed
            drain_record_writes(flush_held=False)
            worker = _connect_shell_started_worker("NS_SIMULATE_FIFO_DIR")
            assert worker is not None
            assert os.sched_getaffinity(stand_in.pid) == {claims[1]}, os.sched_getaffinity(stand_in.pid)
            worker.stdin.close()
            worker.stdout.close()
            for fd in ends:
                os.close(fd)
            # More ranks than CPUs: no pinning at all, rather than two on one.
            _PINNED_CPU = None
            os.sched_setaffinity(0, allowed)
            os.environ["OMPI_COMM_WORLD_SIZE"] = str(len(allowed) + 1)
            assert pin_rank_cpu() is None and _PINNED_CPU is None
    finally:
        _PINNED_CPU = None
        os.sched_setaffinity(0, allowed)
        os.environ.clear()
        os.environ.update(original)
        stand_in.kill()
        stand_in.wait()
    print("cpu pinning self-check passed")


def self_check_follow_worker_node() -> None:
    import signal
    import subprocess
    import tempfile
    from types import SimpleNamespace

    allowed = os.sched_getaffinity(0) if hasattr(os, "sched_setaffinity") else set()
    if len(allowed) < 2:
        print("worker node following self-check skipped")
        return
    node = set(sorted(allowed)[: len(allowed) // 2])
    r2d2 = subprocess.Popen(["sleep", "30"])
    # A shell with a child: the simulate worker's whole tree must move.
    simulate = subprocess.Popen(["sh", "-c", "sleep 30 & wait"], start_new_session=True)
    side = threading.Thread(target=time.sleep, args=(1,))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "r.pid").write_text(f"{r2d2.pid}\n")
            Path(tmp, "s.pid").write_text(f"{simulate.pid}\n")
            _R2D2_WORKERS["r"], _SIMULATE_WORKERS["m"] = SimpleNamespace(base=Path(tmp, "r")), SimpleNamespace(base=Path(tmp, "s"))
            # A worker that may run anywhere this rank may: nothing moves.
            assert follow_r2d2_worker_node("r", "m") is None and os.sched_getaffinity(0) == allowed
            os.sched_setaffinity(r2d2.pid, node)
            deadline = time.monotonic() + 5
            while not Path(f"/proc/{simulate.pid}/task/{simulate.pid}/children").read_text().split():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            side.start()
            # mpirun bound this rank to the other socket: it moves anyway.
            _set_tree_affinity(os.getpid(), allowed - node)
            os.sched_setaffinity(r2d2.pid, node)  # a child here, so it moved too
            assert follow_r2d2_worker_node("r", "m") == node
            assert os.sched_getaffinity(0) == node and os.sched_getaffinity(side.native_id) == node
            assert all(os.sched_getaffinity(tid) == node for tid in _process_tree_tids(simulate.pid))
            assert len(_process_tree_tids(simulate.pid)) >= 2
    finally:
        _R2D2_WORKERS.pop("r", None)
        _SIMULATE_WORKERS.pop("m", None)
        _set_tree_affinity(os.getpid(), allowed)
        os.killpg(simulate.pid, signal.SIGKILL)
        for process in (r2d2, simulate):
            process.kill()
            process.wait()
    print("worker node following self-check passed")


def self_check_parameter_space() -> None:
    for spec in load_all_parameter_specs():
        assert "default" in spec, spec
    specs = load_parameter_space()
    assert len(specs) == sum(spec.get("enabled", True) for spec in load_all_parameter_specs()), specs
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
        # Against the file's own enabled set, not against every spec: more than
        # one dimension ships disabled, and naming an already-disabled one is a
        # no-op rather than a second removal.
        os.environ["NS_DISABLE_PARAMS"] = "source_offset_fraction,channel_width_hz"
        os.environ.pop("NS_ENABLE_PARAMS", None)
        enabled_in_file = [spec for spec in load_all_parameter_specs() if spec.get("enabled", True)]
        load_parameter_space.cache_clear()
        specs = load_parameter_space()
        assert {"source_offset_fraction", "channel_width_hz"}.isdisjoint({spec["name"] for spec in specs}), specs
        assert len(specs) == len(enabled_in_file) - 1, specs

        # A disabled dimension is one fewer cube dimension to draw, but still a
        # params key: simulate_measurement_set() indexes them directly, so a
        # missing one is a KeyError mid-run. Not asserted by value - defaults.toml
        # owns those and they get retuned, which is what stops this from being a
        # second file to edit every time one moves.
        raw: dict[str, Any] = {spec["name"]: float(spec.get("min", 0.0)) for spec in specs}
        fill_disabled_parameters(raw)
        assert {spec["name"] for spec in load_all_parameter_specs()} <= set(raw), raw
        # The one value that is a claim rather than a setting: the offset
        # disables back to the centred source the simulator hard-coded.
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


def self_check_image_dim() -> None:
    import tempfile

    assert image_dim() == int(load_defaults()["NS_IMAGE_DIM"]), image_dim()

    saved_dim, saved_root = os.environ.get("NS_IMAGE_DIM"), os.environ.get("REPO_ROOT")
    try:
        os.environ["NS_IMAGE_DIM"] = "64"
        image_dim.cache_clear()
        assert image_dim() == 64, "NS_IMAGE_DIM has to win, or --interleave measures nothing"

        # defaults.toml is the only place the size is written down, so one
        # without it has to stop the run rather than quietly pick a number:
        # the sky a run searches would depend on which caller it came from.
        del os.environ["NS_IMAGE_DIM"]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "defaults.toml").write_text("NS_NLIVE = 8\n")
            os.environ["REPO_ROOT"] = tmp
            load_defaults.cache_clear()
            image_dim.cache_clear()
            try:
                dim = image_dim()
            except SystemExit:
                pass
            else:
                raise AssertionError(f"defaults.toml without NS_IMAGE_DIM gave {dim}")
    finally:
        for name, value in (("NS_IMAGE_DIM", saved_dim), ("REPO_ROOT", saved_root)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        load_defaults.cache_clear()
        image_dim.cache_clear()


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

    # The 128 belongs to the recording, not to what this run is configured to
    # image: the expected pixels are where those reconstructions put the source
    # in a 128-wide image, and with no CRPIX to read, compute_image_metrics()
    # takes the centre from the array shape - so another width is another
    # centre, and these expectations no longer describe anything.
    for l_as, m_as, max_proj_baseline_lambda, expected in (
        (20.529971838368628, 35.5589543020127, 32152.12622557544, (54, 81)),
        (12.833132485007466, 22.22763748429558, 57746.84392542726, (53, 83)),
    ):
        scale = image_pixel_size_arcsec(max_proj_baseline_lambda)
        image = np.zeros((128, 128))
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
    # runs that enable it. The other representation is forced off each time,
    # since the two must not be enabled together.
    saved_on = os.environ.get("NS_ENABLE_PARAMS")
    saved_off = os.environ.get("NS_DISABLE_PARAMS")
    try:
        os.environ["NS_ENABLE_PARAMS"] = "source_offset_fraction"
        os.environ["NS_DISABLE_PARAMS"] = "source_l_pixels,source_m_pixels"
        load_parameter_space.cache_clear()
        n = len(load_parameter_space())
        params = cube_to_params(np.full(n, 0.5))
        assert params["source_l_arcsec"] != 0.0 or params["source_m_arcsec"] != 0.0, params

        # The cartesian axes land the source on the pixel they name: the box
        # is symmetric about 0, so cube 0.5 is the centre and 1.0 the maximum.
        os.environ["NS_ENABLE_PARAMS"] = "source_l_pixels,source_m_pixels"
        os.environ["NS_DISABLE_PARAMS"] = "source_offset_fraction"
        load_parameter_space.cache_clear()
        specs = load_parameter_space()
        n = len(specs)
        centred = cube_to_params(np.full(n, 0.5))
        assert centred["source_l_arcsec"] == 0.0 and centred["source_m_arcsec"] == 0.0, centred
        corner = cube_to_params(np.full(n, 1.0))
        scale = nominal_pixel_size_arcsec(corner["start_frequency_hz"])
        by_name = {str(spec["name"]): spec for spec in specs}
        for axis, arcsec in (("l", corner["source_l_arcsec"]), ("m", corner["source_m_arcsec"])):
            expected = float(by_name[f"source_{axis}_pixels"]["max"])
            assert math.isclose(arcsec / scale, expected), (axis, arcsec / scale, expected)
        # CDELT1 is negative in a real header, so source_pixel() mirrors l:
        # what is pinned is the distance from the centre.
        centre = image_dim() // 2
        header = {"CDELT1": -scale / 3600.0, "CDELT2": scale / 3600.0}
        sx, sy = source_pixel(
            header, centre, centre, corner["source_l_arcsec"], corner["source_m_arcsec"],
            image_dim(), image_dim(),
        )
        # source_pixel() clamps to the image, and the grid is not symmetric
        # about its centre pixel: `centre` pixels lie below it and one fewer
        # above, so at `fraction = 1` the box's top edge is one pixel past the
        # last row and lands on it.
        away = round(float(by_name["source_l_pixels"]["max"]))
        assert (abs(sx - centre), sy - centre) == (min(away, centre), min(away, image_dim() - 1 - centre)), (sx, sy, away)

        # Enabled alongside the polar dimension the offsets would add, and
        # the same position would come from many draws.
        os.environ["NS_ENABLE_PARAMS"] = "source_offset_fraction,source_l_pixels"
        load_parameter_space.cache_clear()
        try:
            load_parameter_space()
        except SystemExit:
            pass
        else:
            raise AssertionError("polar and cartesian offsets enabled together must abort")
        os.environ["NS_ENABLE_PARAMS"] = "source_l_pixels,source_m_pixels"
        load_parameter_space.cache_clear()

        # A fraction of the half-width, so NS_IMAGE_DIM has to move the box.
        saved_dim = os.environ.get("NS_IMAGE_DIM")
        try:
            os.environ["NS_IMAGE_DIM"] = str(image_dim() * 2)
            for clear in (image_dim, load_all_parameter_specs, load_parameter_space):
                clear.cache_clear()
            doubled = {str(spec["name"]): spec for spec in load_all_parameter_specs()}
            assert float(doubled["source_l_pixels"]["max"]) == 2.0 * away, doubled["source_l_pixels"]
            assert float(doubled["source_l_pixels"]["min"]) == -2.0 * away, doubled["source_l_pixels"]
        finally:
            if saved_dim is None:
                os.environ.pop("NS_IMAGE_DIM", None)
            else:
                os.environ["NS_IMAGE_DIM"] = saved_dim
            for clear in (image_dim, load_all_parameter_specs, load_parameter_space):
                clear.cache_clear()
    finally:
        for var, saved in (("NS_ENABLE_PARAMS", saved_on), ("NS_DISABLE_PARAMS", saved_off)):
            if saved is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = saved
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


def self_check_evidence_record() -> None:
    """The summary's evidence block reads PolyChordOutput's fields, and an
    output without them (an older pypolychord returned None) records None."""
    from types import SimpleNamespace

    output = SimpleNamespace(logZ=-12.5, logZerr=0.25, ndead=300, nlike=4567)
    assert evidence_record(output) == {"log_z": -12.5, "log_z_err": 0.25, "ndead": 300, "nlike": 4567}
    assert evidence_record(None) is None
    print("evidence record reads log(Z) from PolyChordOutput")


def self_check_polychord_finish() -> None:
    """PolyChordFinish's answers end a slice_sample (chordal_sampling.f90,
    PolyChordLite 370f6af) within three calls, wherever in it they start and
    however many of its points fall outside the cube, so a worker never
    reaches the 101-step shrink limit and its 'Non deterministic' warning."""
    import random

    logzero, contour = -1e30, -1.0
    rng = random.Random(81)
    for stop_at in range(1, 40):
        for outside in (0.0, 0.3, 0.7):
            finish, calls = PolyChordFinish.__new__(PolyChordFinish), [0]
            finish.skipped = 0

            def point() -> float:
                if rng.random() < outside:
                    return logzero  # calculate_point: outside the cube, no likelihood call
                calls[0] += 1
                return finish.discarded() if calls[0] >= stop_at else 0.0  # a real point inside the contour

            for _ in range(15):
                right, left = point(), point()
                while right >= contour and right > logzero:
                    right = point()
                while left >= contour and left > logzero:
                    left = point()
                for _ in range(101):
                    baby = point()
                    if baby >= contour and baby > logzero:
                        break
                else:
                    raise AssertionError(f"shrink hit its limit (stop at call {stop_at}, outside {outside})")
            assert finish.skipped <= 3 * 15, finish.skipped
    print("polychord finish answers end each slice_sample within three calls")


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
    assert mpi["rows"][0]["label"] == "simulate (visibilities + noise)"

    oversubscribed = profiling_breakdown(summarize_profiling(evaluations, total_wall_seconds=1.0, mpi_procs=2))
    assert oversubscribed["worker_seconds_budget"] == 19.5
    assert oversubscribed["remainder_rows"][0]["seconds"] == 0.0
    assert oversubscribed["rows"][0]["label"] == "simulate (visibilities + noise)"

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
