#!/usr/bin/env python3
"""Algorithm-agnostic nested-sampling helpers."""

from __future__ import annotations

import atexit
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class _LazyNumpy:
    """numpy, imported on first attribute access instead of at import time.

    numpy is 0.134s of this module's 0.144s import (0.03s of it once the
    report's own stdlib imports have primed the shared machinery), and the
    report only wants the formatting helpers below - a rebuild that writes
    pages from the image store never touches an array, so it should not pay
    for numpy at all. Every `np.x` here is inside a function body (annotations
    are strings under
    ``from __future__ import annotations``), so the first such access rebinds
    the module global to the real numpy and this shim drops out of the picture
    for the rest of the process, including the sampler's likelihood loop.
    """

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
    """Thread settings for a rank-started imaging worker.

    PASSIVE mirrors the pre-warmed pool's container (see the R2D2 run script):
    libgomp spins after every parallel region by default, and the regions here
    are single 128x128 NUFFTs, so with one worker per rank the spinning is a
    whole core per rank burning on nothing.
    """
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


# Pre-started by scripts/lib/start-sidecars.sh, one container per image shared by
# every rank; anything missing here is started by this rank on first use.
_SIDECAR_CONTAINERS: dict[str, str] = json.loads(os.environ.get("NS_SIDECARS", "{}"))
_IMAGE_ENTRYPOINTS: dict[str, list[str]] = {}


def sidecar_container(image: str, platform: str, extra_args: list[str] | None = None) -> str:
    """Name of the run's long-lived container for `image`.

    A per-evaluation `docker run` costs ~0.40s of create/start/teardown on this
    host regardless of image, mounts or platform; `docker exec` into an
    already-running container costs ~0.03s. Every sidecar here is short work
    against bind-mounted paths, so one reused container removes ~0.75s of the
    ~2.3s per evaluation.

    The whole repo is mounted at its host path (as the PolyChord container
    already does), so callers pass absolute paths where they previously passed
    `/work/...` against a per-evaluation `-v {eval_dir}:/work`.

    One container per image is enough for the whole run - separate `docker exec`
    processes are already isolated from each other - so the run script starts
    them before the PolyChord container and hands them over in `NS_SIDECARS`.
    Starting them per rank instead meant 16 concurrent `docker run`s on the
    default 8 ranks, which cost 1.3s against 0.36s for a single one, all of it
    in front of the first evaluation.

    `extra_args` are `docker run` arguments this image needs on top of the repo
    mount (the R2D2 image's read-only `/checkpoints`), and only apply when this
    process is the one starting the container - the run script passes the same
    arguments through `sidecar_launch`.
    """
    if image not in _SIDECAR_CONTAINERS:
        repo_root = os.environ.get("REPO_ROOT", os.getcwd())
        name = f"ri-ns-sidecar-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            [
                "docker", "run", "--detach", "--rm", "--name", name,
                # No sidecar needs networking, and docker's default bridge setup
                # costs ~0.2s per container under rootless Docker. "none" still
                # gives a loopback interface for meqserver.
                "--network", "none",
                # Everything the simulate builds - the working MS and the cached
                # makems skeletons - lives in /dev/shm, and docker's 64MB default
                # is only ~3x the largest cache this parameter space fills.
                "--shm-size", "512m",
                "--platform", platform,
                "-v", f"{repo_root}:{repo_root}",
                *(extra_args or []),
                "--entrypoint", "sleep", image, "infinity",
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        # ponytail: covers normal exit and SystemExit; a SIGKILLed rank leaks one
        # sleeping container, cleaned up with
        # `docker rm -f $(docker ps -q --filter name=ri-ns-sidecar-)`.
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
    """`docker exec` argv prefix equivalent to `docker run <image>` in `workdir`.

    `docker exec` ignores the image ENTRYPOINT, so read it back from the image
    rather than restating the Dockerfile here. `prefix` runs ahead of the
    entrypoint, the way `docker run --entrypoint` would (e.g. GNU `time`).

    Each evaluation gets its own working directory so anything a sidecar writes
    relative to the cwd stays per-evaluation, as it did when every evaluation
    had its own container.
    """
    return [
        "docker", "exec",
        *(["--interactive"] if interactive else []),
        "--workdir", str(workdir),
        sidecar_container(image, platform),
        *sidecar_command(image, prefix),
    ]


def sidecar_command(image: str, prefix: list[str] | None = None) -> list[str]:
    """The in-container argv `docker run <image>` would execute, without arguments."""
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


_SIDECAR_SHELLS: dict[str, subprocess.Popen] = {}


def sidecar_shell(image: str, platform: str) -> subprocess.Popen:
    """This rank's long-lived `sh` inside the sidecar, one `docker exec` per run.

    `docker exec` costs ~0.033s on this host - a third of the `wsclean` binary's
    own ~0.107s - and every evaluation paid it again. One `sh` reading command
    lines from its stdin pays it once per rank; a request costs a pipe write and
    a `read`.
    """
    if image not in _SIDECAR_SHELLS:
        shell = subprocess.Popen(
            # Not sidecar_exec(): this one deliberately bypasses the image
            # ENTRYPOINT, and each request cd's to its own evaluation directory.
            ["docker", "exec", "--interactive", sidecar_container(image, platform), "sh"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        # The container itself is torn down by sidecar_container()'s own atexit
        # hook, which is registered first and so runs last.
        atexit.register(shell.terminate)
        _SIDECAR_SHELLS[image] = shell
    return _SIDECAR_SHELLS[image]


def sidecar_run(
    image: str,
    platform: str,
    workdir: Path,
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> DockerRunResult:
    """Run `cmd` in this rank's sidecar and report its exit code and wall time.

    The command's own output goes to the log files, so only the exit code `echo`
    comes back down the shell's stdout and nothing a sidecar prints can be
    mistaken for a reply. A shell that dies without answering is dropped from
    the cache so the next evaluation starts a fresh one.
    """
    shell = sidecar_shell(image, platform)
    request = (
        f"cd {shlex.quote(str(workdir))} && {shlex.join(cmd)}"
        f" >{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}; echo $?\n"
    )
    started = time.perf_counter()
    shell.stdin.write(request)
    shell.stdin.flush()
    reply = shell.stdout.readline()
    wall_seconds = time.perf_counter() - started
    if not reply:
        _SIDECAR_SHELLS.pop(image, None)
        stderr_path.write_text(f"FATAL: {image} sidecar shell exited without a reply\n")
        return DockerRunResult(returncode=1, wall_seconds=wall_seconds, peak_memory_bytes=0)
    return DockerRunResult(returncode=int(reply), wall_seconds=wall_seconds, peak_memory_bytes=0)


def run_checked(cmd: list[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


def _fits_card_value(field: str) -> Any:
    """The value half of a FITS card, comment stripped.

    A quoted string may contain the `/` that otherwise starts the comment
    (`BUNIT = 'JY/BEAM '`), so quotes are closed before the comment is cut.
    """
    field = field.lstrip()
    if field.startswith("'"):
        end = field.index("'", 1)
        return field[1:end].strip()
    value = field.split("/")[0].strip()
    return value == "T" if value in ("T", "F") else float(value)


def load_fits_2d(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the primary HDU of a WSClean/R2D2 image as a 2-D float64 array.

    `from astropy.io import fits` costs ~0.45s when the 8 default ranks import
    it at once - more than every other per-rank startup put together, and more
    than 10% of the run's wall clock - to read a single-HDU uncompressed float
    image. Everything needed here is 30 lines of the FITS standard: 2880-byte
    header blocks of 80-column cards, then big-endian samples in C order.
    Anything outside that (a scaled or integer image, an extension) raises
    rather than being guessed at.
    """
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
    teardown plus the `docker stats` polling loop. Stage totals are summed
    worker-seconds; only serial runs can subtract them from run wall time to
    estimate PolyChord's own sampling/bookkeeping overhead.
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
    polychord_overhead = total_wall_seconds - accounted if mpi_procs == 1 else None

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
        "accounted_worker_seconds": accounted,
        "accounted_seconds": accounted if mpi_procs == 1 else None,
        "polychord_overhead_seconds": polychord_overhead,
        "note": (
            "stage totals are summed worker-seconds; accounted_seconds and "
            "polychord_overhead_seconds are only emitted for mpi_procs == 1."
        ),
    }


def format_duration(seconds: float | None) -> str:
    """Human-readable duration: '47ms', '1.44s', '12.3s', '7m 36s', '2h 05m 09s'.

    Seconds stop being readable at both ends of the range a profile spans - a
    per-eval metrics stage is milliseconds, a run total is hours - so each
    magnitude gets the unit that carries its digits.
    """
    if seconds is None:
        return "n/a"
    value = max(0.0, float(seconds))
    if value < 1.0:
        milliseconds = round(value * 1000)
        # 0.9996s rounds up to 1000ms, which belongs in the seconds branch.
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
    """A share of the worker-time budget as a percentage, e.g. '64.6%'."""
    if fraction is None:
        return ""
    pct = 100.0 * fraction
    if 0.0 < pct < 0.05:
        return "<0.1%"
    return f"{pct:.1f}%"


# (stage key, eval-count key, label, indented under the row above it).
# `{imager}` is filled in with the run's algorithm so the table says "wsclean
# container" or "r2d2 container" rather than the uninformative "image container".
PROFILING_VIEW_STAGES = (
    ("simulate", "simulate_seconds", "simulate (MeqTrees)", False),
    ("convert", "convert_seconds", "convert (MS -> .mat)", False),
    ("image_container", "image_container_seconds", "{imager} container (total)", False),
    ("image_binary", "image_binary_seconds", "of which: {imager} itself", True),
    ("image_container_overhead", "image_container_overhead_seconds", "of which: container overhead", True),
    ("metrics", "metrics_seconds", "metrics computation", False),
)

UNACCOUNTED_LABEL = "unaccounted (PolyChord sampling + idle)"

PROFILING_VIEW_NOTE = (
    "stage totals are summed worker-seconds across every evaluation; shares are "
    "of the run's worker-time budget (wall clock x mpi_procs), so the top-level "
    "stages plus the unaccounted remainder come to 100%. Dividing a worker-second "
    "total by mpi_procs gives what that stage cost in wall clock, since the workers "
    "spend it side by side, and those wall-clock figures add up to the run's "
    "end-to-end wall time."
)


def profiling_breakdown(profiling: dict[str, Any], algorithm: str | None = None) -> dict[str, Any]:
    """Rows and denominators for the per-stage timing view.

    Shared by the HTML report and scripts/profile-nested-sampling-run.py so the
    two cannot drift apart. Every share is a fraction of the run's total
    worker-time budget - wall clock x mpi_procs - so the top-level stages plus
    the unaccounted remainder add up to 100% of what the whole process spent.
    That holds for serial and MPI runs alike: at mpi_procs == 1 the budget is
    just the wall clock and the remainder is PolyChord's own sampling, while at
    mpi_procs > 1 the remainder also absorbs the time workers sat idle.
    """
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

    budget = None if total_wall is None else total_wall * mpi_procs
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

    unaccounted = None if denominator is None else max(0.0, denominator - accounted)
    return {
        "imager": imager,
        "mpi_procs": mpi_procs,
        "evals": max((row["evals"] for row in rows), default=0),
        "total_wall_seconds": total_wall,
        "worker_seconds_budget": denominator,
        "accounted_seconds": accounted,
        "accounted_share": share(accounted),
        "unaccounted_label": UNACCOUNTED_LABEL,
        "unaccounted_seconds": unaccounted,
        "unaccounted_share": share(unaccounted),
        "rows": rows,
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


def load_evaluations_from_dir(evaluations_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(evaluations_dir.glob("eval-*/metrics.json")):
        records.append(json.loads(metrics_path.read_text()))
    return records


def write_evaluation_record(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


_SIMULATE_WORKERS: dict[str, "subprocess.Popen | FifoWorker"] = {}


class FifoWorker:
    """A `--serve --fifo` worker the run script already started.

    Same `.stdin`/`.stdout`/`.terminate()` surface as the `subprocess.Popen`
    below, over the FIFO pair the worker is serving on. Closing stdin is what
    ends it: the worker's request loop sees EOF and exits.
    """

    def __init__(self, write_fd: int, reply_path: Path) -> None:
        self.stdin = os.fdopen(write_fd, "w")
        # Opening a FIFO blocks until the other end is open, so this must be the
        # same order serve() uses - request pipe first, reply pipe second.
        self.stdout = reply_path.open("r")

    def terminate(self) -> None:
        self.stdin.close()


def _connect_shell_started_worker(fifo_dir_var: str) -> FifoWorker | None:
    """Attach to this rank's pre-warmed worker, or None if there is not one.

    A rank-started worker is not ready to answer for a while - interpreter,
    Timba, meqserver and the first TDL compile for simulate, ~1.3s of `import
    torch` and the R2D2 modules for imaging - and PolyChord asks every rank for
    a live point at once, so all of it used to land on the wall clock in front
    of evaluation one. The run scripts make one warm worker per rank the
    sidecar's own startup command instead, and this connects to it. Falling back
    to a rank-started worker is what happens when there is no pool - an
    OUTPUT_DIR outside the bind mount, so the FIFOs are not visible in both
    containers.
    """
    fifo_dir = os.environ.get(fifo_dir_var)
    if not fifo_dir:
        return None
    base = Path(fifo_dir) / str(mpi_rank())
    # O_NONBLOCK is how a FIFO write-open says "no reader yet" (ENXIO) instead of
    # blocking forever, which is what a worker that never started would do. The
    # deadline is generous because it is only ever reached when something is
    # broken, and the fallback below is correct, just slower.
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
        return FifoWorker(write_fd, Path(f"{base}.out"))


def simulate_worker(meqtrees_image: str, platform: str) -> subprocess.Popen | FifoWorker:
    """This rank's long-lived `simulate_point_source_ms.py --serve` process.

    Even inside a reused sidecar container, a per-evaluation `docker exec` of the
    simulate script paid ~0.45s of the ~0.7s it took: the Python interpreter,
    numpy/casacore and Timba imports, starting a meqserver and reaping it again.
    One worker per rank keeps all of that warm and leaves only the per-evaluation
    compile, RIME predict and noise fill.
    """
    if meqtrees_image not in _SIMULATE_WORKERS:
        worker = _connect_shell_started_worker("NS_SIMULATE_FIFO_DIR")
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


def r2d2_serve_path(repo_root: Path) -> str:
    """The imaging worker script, read live off the repo bind mount."""
    return str(repo_root / "scripts" / "lib" / "nested_sampling" / "r2d2_serve.py")


def r2d2_checkpoint_mount(checkpoints_dir: str) -> list[str]:
    """`docker run` arguments putting the host checkpoints at `/checkpoints`.

    The mount point stays `/checkpoints` rather than becoming a host path so
    that `ckpt_path` - which every `poc-summary.json` records and
    merge-nested-sampling-runs.py compares across runs - keeps its
    machine-independent value.
    """
    return ["-v", f"{checkpoints_dir}:/checkpoints:ro"]


def r2d2_worker(r2d2_image: str, platform: str, checkpoints_dir: str) -> "subprocess.Popen | FifoWorker":
    """This rank's long-lived `r2d2_serve.py` process inside the R2D2 sidecar.

    A per-evaluation `docker run` of this image cost ~2.4s warm on this host and
    only ~0.6s of it was science: ~0.5s of container create/start plus ~1.3s of
    `import torch` and the R2D2 module imports, repeated every evaluation. One
    worker per rank pays both once.

    The thread limits go on the `docker exec`, not the container, because torch
    and finufft read them at import time and each rank gets its own share. The
    pre-warmed variant takes them from the container instead - every rank gets
    the same value, so there is nothing per-rank to lose.
    """
    if r2d2_image not in _R2D2_WORKERS:
        worker = _connect_shell_started_worker("NS_R2D2_FIFO_DIR")
        if worker is None:
            repo_root = Path(os.environ.get("REPO_ROOT", os.getcwd()))
            container = sidecar_container(r2d2_image, platform, r2d2_checkpoint_mount(checkpoints_dir))
            worker = subprocess.Popen(
                [
                    "docker", "exec", "--interactive",
                    *r2d2_docker_thread_env_flags(),
                    container,
                    "python3",
                    # Read live off the repo bind mount: the R2D2 image bakes in
                    # no copy of this repo's scripts, so nothing to rebuild.
                    r2d2_serve_path(repo_root),
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
    """Run one `imager.py` in this rank's R2D2 worker, same shape as sidecar_run()."""
    worker = r2d2_worker(r2d2_image, platform, checkpoints_dir)
    request = {"argv": argv, "stdout": str(stdout_path), "stderr": str(stderr_path)}
    started = time.perf_counter()
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()
    reply = worker.stdout.readline()
    wall_seconds = time.perf_counter() - started
    if not reply:
        # The worker died mid-request; drop it so the next evaluation gets a
        # fresh one instead of every later evaluation inheriting the corpse.
        _R2D2_WORKERS.pop(r2d2_image, None)
        stderr_path.write_text("FATAL: r2d2 worker exited without a reply\n")
        return DockerRunResult(returncode=1, wall_seconds=wall_seconds, peak_memory_bytes=0)
    answer = json.loads(reply)
    return DockerRunResult(
        returncode=answer["returncode"],
        wall_seconds=wall_seconds,
        peak_memory_bytes=answer["peak_memory_bytes"],
    )


def prewarm(*targets: Callable[[], None]) -> Callable[[], None]:
    """Start this rank's sidecar attachments concurrently; returns a joiner.

    The first evaluation on a rank cost time that later ones did not, and every
    rank paid it at the same moment, so all of it landed on the wall clock in
    front of evaluation one: the simulate worker's Python/Timba/meqserver
    startup, the R2D2 worker's `import torch`, and the `docker inspect`s for an
    image entrypoint - one after the other. Here they run in threads, so the
    rank pays the slowest instead of the sum, and the caller can overlap them
    with PolyChord's own startup by joining late.

    Nothing may touch a sidecar between this call and the returned joiner: the
    caches these threads fill are plain dicts with no lock.
    """
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
    """Send one request to this rank's simulate worker and report its exit code.

    A worker that dies without answering is dropped from the cache so the next
    evaluation gets a fresh one instead of every later evaluation inheriting the
    corpse, and reports exit 1 with the reason in the caller's stderr log.
    """
    worker = simulate_worker(meqtrees_image, platform)
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()
    reply = worker.stdout.readline()
    if not reply:
        _SIMULATE_WORKERS.pop(meqtrees_image, None)
        stderr_path.write_text("FATAL: simulate worker exited without a reply\n")
        return 1
    return json.loads(reply)["returncode"]


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
    """Convert this evaluation's MS to R2D2's `.mat` in the warm simulate worker.

    Its own `docker exec` of ms_to_r2d2_mat.py cost ~0.15s and only ~0.01s of
    that was the conversion; the rest was the exec, a fresh interpreter and the
    numpy/casacore/scipy imports. The simulate worker already has all of that
    live and has just written the MS, so it does the convert too.
    """
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
    """load_fits_2d() against astropy on a WSClean-shaped image.

    astropy is still in the polychord image; it is just not imported on the hot
    path. The trap this guards is card parsing, not the data block: a quoted
    value may contain the `/` that starts a comment (`BUNIT = 'JY/BEAM '`).
    """
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
    assert header["BUNIT"] == "JY/BEAM", header["BUNIT"]
    assert header["CRPIX1"] == 4.0 and header["CRPIX2"] == 5.0
    assert header["SIMPLE"] is True
    assert int(header["NAXIS"]) == int(expected_header["NAXIS"])


def self_check_lazy_numpy() -> None:
    """Importing this module must not import numpy, and `np` must still work.

    Needs a fresh interpreter: any caller that has already touched an array has
    numpy in sys.modules. The regression this catches is a new module-level
    `np.` use, which would re-import numpy eagerly and silently give the
    report's page-only rebuild its 0.03s back.
    """
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
    assert empty_profiling["accounted_worker_seconds"] == 0.0
    assert empty_profiling["polychord_overhead_seconds"] == 5.0

    mpi_profiling = summarize_profiling(evaluations, total_wall_seconds=5.0, mpi_procs=4)
    assert mpi_profiling["accounted_worker_seconds"] == 19.5
    assert mpi_profiling["accounted_seconds"] is None
    assert mpi_profiling["polychord_overhead_seconds"] is None

    assert format_duration(None) == "n/a"
    assert format_duration(0.0) == "0ms"
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

    # Serial run: budget is the wall clock, so stages + unaccounted make 100%.
    serial = profiling_breakdown(profiling, algorithm="wsclean")
    assert serial["worker_seconds_budget"] == 25.0
    assert serial["evals"] == 3
    labels = [row["label"] for row in serial["rows"]]
    assert "wsclean container (total)" in labels and "of which: wsclean itself" in labels
    assert "convert (MS -> .mat)" not in labels
    container = next(row for row in serial["rows"] if row["key"] == "image_container")
    assert container["seconds"] == 15.0 and container["evals"] == 3
    assert abs(container["per_eval_seconds"] - 5.0) < 1e-9
    assert abs(container["share"] - 0.6) < 1e-9
    assert abs(serial["unaccounted_seconds"] - 5.5) < 1e-9
    top_level = sum(row["share"] for row in serial["rows"] if not row["is_sub"])
    assert abs(top_level + serial["unaccounted_share"] - 1.0) < 1e-9

    # MPI run: budget is wall clock x workers, and idle time lands in unaccounted.
    mpi = profiling_breakdown(mpi_profiling, algorithm="r2d2")
    assert mpi["worker_seconds_budget"] == 20.0
    assert abs(mpi["accounted_share"] - 0.975) < 1e-9
    assert abs(mpi["unaccounted_seconds"] - 0.5) < 1e-9
    assert mpi["rows"][0]["label"] == "simulate (MeqTrees)"

    # Accounted time above the nominal budget must not push shares over 100%.
    oversubscribed = profiling_breakdown(summarize_profiling(evaluations, total_wall_seconds=1.0, mpi_procs=2))
    assert oversubscribed["worker_seconds_budget"] == 19.5
    assert oversubscribed["unaccounted_seconds"] == 0.0
    assert oversubscribed["rows"][0]["label"] == "simulate (MeqTrees)"

    # A run with no timings at all still renders rather than dividing by zero.
    degenerate = profiling_breakdown({"mpi_procs": 1, "total_wall_seconds": 0.0, "stage_totals_seconds": {}})
    assert degenerate["worker_seconds_budget"] is None
    assert degenerate["unaccounted_seconds"] is None
    assert degenerate["rows"] == []
