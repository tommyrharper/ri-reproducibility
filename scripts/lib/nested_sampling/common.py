#!/usr/bin/env python3
"""Algorithm-agnostic nested-sampling helpers."""

from __future__ import annotations

import atexit
import json
import math
import os
import select
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
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

# An evaluation the algorithm failed. PolyChord maximizes the objective and a
# real total_rms_jy is ~0.008, so this makes a failure the most interesting
# point in the search - which is the point, because failure modes are what
# this repo looks for. It is only correct for failures of the *algorithm*:
# see WORKER_DIED.
FAILURE_OBJECTIVE = 100.0

# A worker died mid-request instead of a tool running and failing. Real exit
# statuses are 0-255, so this cannot be mistaken for one.
#
# The distinction matters more than it looks. A host that runs out of memory
# has its OOM killer take a worker, and without this the rank could not tell
# that from R2D2 exiting non-zero on the parameters it was given - so it
# recorded FAILURE_OBJECTIVE, and the search treated running out of memory as
# the most interesting result it had ever found. The infrastructure dying says
# nothing about the algorithm, so it must never reach the sampler as a
# likelihood: it is retried, and if it persists the run stops.
WORKER_DIED = -1

# How long to keep putting a request to a worker before its death is called
# permanent, as the pause before each attempt. A worker is dropped from the
# cache when it dies, so every attempt gets a freshly started one.
#
# The retries are patient rather than immediate because of what kills a worker
# here: the host running out of memory, usually because another run on the
# same machine is holding it. That clears on its own - the memory this
# attempt died for is freed by its own death, and the other run eventually
# finishes - so waiting is what turns a dead run into a slow one. ~51s of
# patience, then the evaluation is treated as impossible.
WORKER_RETRY_DELAYS = (0.0, 1.0, 5.0, 15.0, 30.0)


def worker_attempts() -> Any:
    """Attempt numbers for a worker request, pausing longer before each retry."""
    for attempt, delay in enumerate(WORKER_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        yield attempt


# A worker that goes silent is not a worker that died, and until these bounds
# existed only the second one was survivable. MeqTrees deadlocks with its
# meqserver roughly once in 2,000 evaluations - the worker stays alive, its
# request never completes and no reply is ever written - which left the rank
# waiting on it blocked in readline() forever and the other ranks burning a core
# each in the MPI collective behind it. Every run left unattended came back
# stopped rather than finished.
#
# The bound is also the stall: PolyChord has every rank in the same collective,
# so one silent worker holds all of them until its timeout expires. At 30s a
# 20-rank wsclean run spent 92s of 408s - 23% of its wall clock - waiting out
# four of these. So each bound wants to be as small as it can be without ever
# firing on a stage that was only slow.
#
# Both requests this one covers are far below it, measured over 17,644
# evaluations: simulate peaks at 0.34s on wsclean and 0.60s on R2D2, and the
# R2D2-only MS-to-.mat convert - the slower of the two, and the reason this is
# not smaller still - peaks at 1.42s, with nothing over 2s. 10s leaves 7x on
# the worst of them. R2D2's actually slow stage is its imaging, which is
# IMAGING_REPLY_TIMEOUT below and nothing to do with this.
SIMULATE_REPLY_TIMEOUT = 10.0
SHELL_REPLY_TIMEOUT = 300.0
IMAGING_REPLY_TIMEOUT = 3600.0


def worker_reply(stream: Any, timeout: float) -> str | None:
    """One reply line: `""` if the worker died, None if it stopped answering.

    select() reads the file descriptor underneath the buffer, which is only
    safe because the protocol is strictly one flushed reply line per request
    and a request only goes out once the previous reply has been read - so
    there is never a second line already sitting in the buffer for select() to
    miss. A reply is short enough to be an atomic pipe write, so a readable
    stream always holds the whole line.
    """
    if not select.select([stream], [], [], timeout)[0]:
        return None
    return stream.readline()


class WorkerDied(RuntimeError):
    """A worker died and did not come back, so the host failed, not the algorithm.

    Raised out of an evaluation rather than scored, because there is no honest
    value to return. Scoring it high makes the sampler chase the OOM killer;
    scoring it low carves a hole out of exactly the expensive corner of the
    parameter space where the real failure modes live. Both are a lie about
    the algorithm, so the run stops instead.
    """


def abort_run(message: str) -> None:
    """Stop every rank now, without returning a likelihood.

    Raising out of the likelihood would only unwind this rank - it is called
    from PolyChord's Fortran - and leave the others waiting on a collective
    that never comes, so the job is torn down explicitly. Exiting with the
    results so far on disk and a reason on stderr is the honest outcome: the
    evaluations that did finish are still valid, and the chain is not
    contaminated by a value nobody measured.
    """
    print(f"FATAL: {message}", file=sys.stderr, flush=True)
    try:
        from mpi4py import MPI

        MPI.COMM_WORLD.Abort(1)
    except Exception:
        # No MPI, or it is already too broken to abort through: fall through
        # to the hard exit, which the launcher turns into a failed run anyway.
        pass
    os._exit(1)


DEFAULT_WSCLEAN_NITER = 100
DEFAULT_WSCLEAN_AUTO_THRESHOLD = 3.0

# Image geometry, shared by both imagers so that they reconstruct the same sky.
# R2D2 derives its cell size from the data it is given rather than taking one
# (src/utils/io.py in the pinned upstream commit), so WSClean has to apply the
# same formula instead of a fixed `-scale`: the search sweeps start_frequency_hz
# over three orders of magnitude, and a fixed cell is either far finer or far
# coarser than the synthesized beam at almost every frequency in that range.
#
# 1.5 is R2D2's own `CommonArgs` default (src/utils/args.py). It is stated here
# and written into the R2D2 config explicitly rather than left to that default,
# because WSClean's `-scale` is derived from it: an unpinned value on one side
# is the same mismatch in a quieter form. (The 1.52 this repo used to record as
# R2D2's super-resolution factor is a property of upstream's bundled
# `data_3c353.mat` example, not of these runs, and was never written into the
# config - so R2D2 has always run at 1.5.)
DEFAULT_IMAGE_DIM = 128
DEFAULT_SUPER_RESOLUTION = 1.5

# `source_offset_fraction` geometry (docs/parameter-space-proposal.md, section 1).
# VLA-A's longest baseline, used to pick the offset before the MS exists (the
# simulator needs an absolute source position; the real cell size is only known
# from the simulated observation, after the source has already been placed).
# Declination barely changes VLA-A's maximum baseline near 65 degrees, so this
# is accurate enough for a controlled, small offset - see
# docs/nested-sampling.md, "Toggling dimensions on and off".
VLA_A_MAX_BASELINE_M = 36_000.0
SPEED_OF_LIGHT_M_S = 299_792_458.0
# Fixed, not axis-aligned: avoids the symmetries a purely horizontal or
# vertical offset would have.
SOURCE_OFFSET_POSITION_ANGLE_DEG = 30.0


def image_pixel_size_arcsec(
    max_proj_baseline_lambda: float,
    super_resolution: float = DEFAULT_SUPER_RESOLUTION,
) -> float:
    """The cell size R2D2 would pick for this sampling pattern, in arcsec.

    Upstream `src/utils/io.py`, verbatim:

        spatial_bandwidth = 2 * max_proj_baseline
        image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)

    `max_proj_baseline` is the longest projected baseline in wavelengths, which
    the simulator records as `observation.max_proj_baseline_lambda`.
    """
    if not max_proj_baseline_lambda > 0.0:
        raise SystemExit(f"FATAL: non-positive max projected baseline: {max_proj_baseline_lambda!r}")
    return (180.0 / math.pi) * 3600.0 / (super_resolution * 2.0 * max_proj_baseline_lambda)


def source_offset_to_lm(fraction: float, start_frequency_hz: float) -> tuple[float, float]:
    """`source_offset_fraction` (0.0-0.35) to an (l, m) sky offset in arcsec.

    The offset is `fraction` of the image half-width, at a fixed
    `SOURCE_OFFSET_POSITION_ANGLE_DEG`. The half-width uses
    `image_pixel_size_arcsec()` against `VLA_A_MAX_BASELINE_M` and the sampled
    frequency - a nominal max projected baseline, not the one the simulator
    will actually record for this evaluation, because the source position has
    to be chosen before the MS (and its real baselines) exist.
    """
    max_proj_baseline_lambda = VLA_A_MAX_BASELINE_M * start_frequency_hz / SPEED_OF_LIGHT_M_S
    half_width_arcsec = image_pixel_size_arcsec(max_proj_baseline_lambda) * (DEFAULT_IMAGE_DIM / 2.0)
    radius_arcsec = fraction * half_width_arcsec
    angle_rad = math.radians(SOURCE_OFFSET_POSITION_ANGLE_DEG)
    return radius_arcsec * math.sin(angle_rad), radius_arcsec * math.cos(angle_rad)


@cache
def load_defaults() -> dict[str, Any]:
    """defaults.toml, the one file both the host and the containers read.

    REPO_ROOT is what the containers get: the repo is bind-mounted at the same
    path inside them, and this module is baked in at /opt/ri-nested-sampling,
    where the __file__-relative repo root a host run walks up to does not exist.

    Read on first use rather than at import, for the same reason numpy is: the
    report imports this module inside the R2D2 image, whose Python 3.10 has no
    tomllib, and it wants the formatting helpers - never the prior box, which
    it reads back out of each run's summary.json.
    """
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
    """The receiver bands a spectral window has to fit inside - `[[receiver_band]]`.

    Sorted by frequency so the unit-cube mapping does not depend on the order
    they happen to be written in.
    """
    bands = load_defaults()["receiver_band"]
    if not bands:
        raise SystemExit("defaults.toml: [[receiver_band]] is empty")
    return sorted(bands, key=lambda band: float(band["min"]))


@cache
def load_all_parameter_specs() -> list[dict[str, Any]]:
    """Every `[[parameter_space]]` entry in defaults.toml, enabled or not.

    `load_parameter_space()` is the ones the sampler actually searches; this
    is every one that has ever been added, disabled entries included, so
    `cube_to_params()` can still fix a disabled dimension at its pinned value
    and `./ri params` can show the full list.
    """
    return load_defaults()["parameter_space"]


def _param_name_set(env_var: str) -> set[str]:
    return {name.strip() for name in os.environ.get(env_var, "").split(",") if name.strip()}


@cache
def load_parameter_space() -> list[dict[str, Any]]:
    """The parameter space the sampler searches - `[[parameter_space]]` in defaults.toml.

    An entry is included unless `enabled = false` in defaults.toml, further
    overridden by the `NS_DISABLE_PARAMS` / `NS_ENABLE_PARAMS` comma-separated
    name lists (what `./ri search --disable-param` / `--enable-param` set), so
    a one-off search does not need a defaults.toml edit. See "Toggling
    dimensions on and off" in docs/nested-sampling.md.

    A `kind = "band_start"` dimension carries no `min`/`max` of its own: its
    box is the span of the receiver bands, filled in here so that everything
    reading a spec (paramnames, plots, summary.json) sees a plain box.
    """
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
    """The channel box against the bandwidth the bands actually have.

    A window that overflows the band its start frequency landed in is fitted
    to it (see fit_spectral_window()), so a box asking for more than a band
    holds is normal. The one box no fitting can rescue is one whose *smallest*
    window - the minimum channel count at the minimum width - fits no band at
    all: every draw would then fail, whatever start frequency came up. That is
    a configuration error, fatal here at load rather than after the images are
    warm.
    """
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


# A start frequency with too little room above it for even the smallest window
# is replaced by stepping this far around the unit interval and drawing again.
# The golden ratio conjugate spreads successive tries across every band rather
# than nudging along the one that just failed, and stepping (rather than
# drawing from an RNG) keeps the prior transform a pure function of the cube,
# which is what PolyChord requires.
START_REDRAW_STEP = 0.6180339887498949
MAX_START_REDRAWS = 64


@dataclass
class WindowFitStats:
    """What fitting sampled windows into receiver bands costs.

    `draws` counts prior transforms, not evaluations: a draw whose window is
    reduced still becomes one evaluation, it just measures narrower channels
    (or fewer of them) than the one that was drawn.
    """

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
    """Every rank's fitting tally, summed.

    PolyChord's rank 0 coordinates and barely runs the prior transform at all -
    the workers do the drawing - so this has to be collected across the run or
    the numbers in `summary.json` describe nobody's work. Collective: call it
    from every rank once PolyChord has returned, before rank 0 branches off to
    write the summary.
    """
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
    """The band a unit-cube value picks, and where in it the window starts.

    Every band gets an equal share of the dimension and the start frequency is
    uniform inside the band - equal share rather than uniform across the union
    of the bands, or the 32 MHz-wide 4-band would come up about once in every
    1500 draws and never actually be searched.
    """
    bands = load_receiver_bands()
    position = min(1.0, max(0.0, cube_value)) * len(bands)
    index = min(int(position), len(bands) - 1)
    band = bands[index]
    lower, upper = float(band["min"]), float(band["max"])
    return band, lower + (position - index) * (upper - lower)


def start_frequency_cube_value(start_frequency_hz: float) -> float:
    """The inverse of start_frequency_from_cube(), for the theta -> cube round trip."""
    bands = load_receiver_bands()
    for index, band in enumerate(bands):
        lower, upper = float(band["min"]), float(band["max"])
        if start_frequency_hz <= upper or index == len(bands) - 1:
            within = (start_frequency_hz - lower) / (upper - lower)
            return (index + min(1.0, max(0.0, within))) / len(bands)
    return 1.0


@cache
def channel_floors() -> tuple[int, float]:
    """The smallest channel count and channel width the parameter space allows."""
    by_name = {str(spec["name"]): spec for spec in load_parameter_space()}
    return int(by_name["channel_count"]["min"]), float(by_name["channel_width_hz"]["min"])


def fit_spectral_window(
    cube_value: float, channel_count: int, channel_width_hz: float, track: bool = False
) -> tuple[float, int, float]:
    """Fit a sampled window into the band its start frequency landed in.

    The start frequency is drawn first and the window is fitted to the room
    left above it, giving up as little as possible at each step:

    1. the window fits - keep the draw;
    2. it does not - narrow the channels until it does, if that stays at or
       above the minimum width;
    3. it would go below that - hold the width at the minimum and drop
       channels instead, if that stays at or above the minimum count;
    4. even the smallest window does not fit, so the start frequency is too
       close to the top of its band to hold anything - draw another one and
       start over.

    `track` is set by the prior transform, the one caller whose draws are real
    sampler work; the likelihood re-derives parameters it has already been
    given, which always fit and would otherwise pad the tally.

    Returns (start_frequency_hz, channel_count, channel_width_hz).
    """
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
    """Write PolyChord's <file_root>.paramnames beside future chain output."""
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


def fill_disabled_parameters(raw: dict[str, Any]) -> None:
    """Pin every dimension `load_parameter_space()` left out at a fixed value.

    A disabled dimension is still a key every downstream reader (the
    simulator CLI, `compute_image_metrics()`, `simulate_measurement_set()`)
    expects in `params`, so it is fixed here instead of drawn from the cube:
    at its `default` if defaults.toml gives one, otherwise at its `min` (which
    is why disabling `source_offset_fraction` alone reproduces the old
    hard-coded centred source - its `min` already is 0.0).
    """
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
        value = scale(float(cube[i]), float(spec["min"]), float(spec["max"]))
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
    # Kept under its own name, not popped like log10_dynamic_range above:
    # prior_vector() only has a reverse formula for that one special case, so
    # every other cube dimension has to survive into `raw` under its own name
    # for the theta -> cube -> params round trip self_check_spectral_window()
    # checks.
    raw["source_l_arcsec"], raw["source_m_arcsec"] = source_offset_to_lm(
        raw["source_offset_fraction"], raw["start_frequency_hz"]
    )
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
    the cache, so the retry below starts a fresh one; a death that survives
    that is reported as WORKER_DIED rather than as an exit status the command
    never returned.
    """
    request = (
        f"cd {shlex.quote(str(workdir))} && {shlex.join(cmd)}"
        f" >{shlex.quote(str(stdout_path))} 2>{shlex.quote(str(stderr_path))}; echo $?\n"
    )
    started = time.perf_counter()
    for attempt in worker_attempts():
        shell = sidecar_shell(image, platform)
        shell.stdin.write(request)
        shell.stdin.flush()
        reply = worker_reply(shell.stdout, SHELL_REPLY_TIMEOUT)
        if reply:
            wall_seconds = time.perf_counter() - started
            return DockerRunResult(returncode=int(reply), wall_seconds=wall_seconds, peak_memory_bytes=0)
        if reply is None:
            # ponytail: kills the `docker exec` client, leaving the `sh` it was
            # talking to wedged in the sidecar - the ranks' shells are
            # indistinguishable in there, so there is nothing to pgrep for. One
            # leaks per timeout; give the shell an `echo $$` handshake at
            # startup if that ever costs more than the retry does.
            shell.kill()
        _SIDECAR_SHELLS.pop(image, None)
    wall_seconds = time.perf_counter() - started
    stderr_path.write_text(
        f"FATAL: {image} sidecar shell gave no reply, {len(WORKER_RETRY_DELAYS)} times\n"
    )
    return DockerRunResult(returncode=WORKER_DIED, wall_seconds=wall_seconds, peak_memory_bytes=0)


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


def source_pixel(
    header: dict[str, Any],
    cx: int,
    cy: int,
    source_l_arcsec: float,
    source_m_arcsec: float,
    x_size: int,
    y_size: int,
) -> tuple[int, int]:
    """The pixel a source at (`source_l_arcsec`, `source_m_arcsec`) lands on.

    Plain WCS: `pixel = CRPIX - 1 + world_offset / CDELT`, applied on each axis
    with that axis's own signed `CDELT` (arcsec/pixel) - `CDELT1` is negative
    (RA increases the opposite way pixel x does), `CDELT2` positive, so the
    same formula on both axes reproduces Meow's `LMDirection` convention (the
    one `point_source_forest.py` places the source with) without a manual sign
    flip. Checked against an actual WSClean image with a non-zero offset, not
    derived from the FITS standard alone - see self_check_source_offset().

    `header` must carry both CDELT keys; `compute_image_metrics()` supplies
    them for R2D2, which writes none. It used to be true that an unoffset
    source never reached this read, which is why a header-less image only
    failed once `source_offset_fraction` was searched - do not lean on that
    again.
    """
    if source_l_arcsec == 0.0 and source_m_arcsec == 0.0:
        return cx, cy
    cdelt1_arcsec = float(header["CDELT1"]) * 3600.0
    cdelt2_arcsec = float(header["CDELT2"]) * 3600.0
    sx = cx + int(round(source_l_arcsec / cdelt1_arcsec))
    sy = cy + int(round(source_m_arcsec / cdelt2_arcsec))
    return max(0, min(x_size - 1, sx)), max(0, min(y_size - 1, sy))


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
    # WSClean writes a full WCS; R2D2 writes a bare header - SIMPLE, BITPIX,
    # NAXIS and nothing else - so for R2D2 both the reference pixel and the
    # cell size have to be supplied. Its grid is the centred one the FFT
    # implies, and its cell size is what `image_pixel_size_arcsec()` derives
    # from the same recorded baseline R2D2 sized its own pixels from (the
    # figure WSClean is passed as `-scale`). Missing CDELT with no
    # `pixel_size_arcsec` is a caller bug, not something to guess a scale for:
    # silently centring the source scores a good image as a catastrophic one.
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
    # FITS CRPIX is 1-based, so the centre of an even axis is `size / 2 + 1`:
    # 65 for the 128-pixel images here, which is what WSClean writes and what
    # R2D2's own grid uses. Defaulting to `size / 2` put a header-less image's
    # centre one pixel low on both axes.
    cx = int(round(float(header.get("CRPIX1", x_size / 2.0 + 1.0)) - 1.0))
    cy = int(round(float(header.get("CRPIX2", y_size / 2.0 + 1.0)) - 1.0))
    cx = max(0, min(x_size - 1, cx))
    cy = max(0, min(y_size - 1, cy))
    sx, sy = source_pixel(header, cx, cy, source_l_arcsec, source_m_arcsec, x_size, y_size)

    truth = np.zeros_like(image)
    truth[sy, sx] = source_flux_jy
    residual = image - truth

    yy, xx = np.ogrid[:y_size, :x_size]
    off_source = (yy - sy) ** 2 + (xx - sx) ** 2 > 25
    off_rms = rms(image[off_source])
    total_rms = rms(residual)
    peak = float(np.nanmax(np.abs(image)))
    snr = peak / off_rms if off_rms > 0 else float("inf")
    log_snr = math.log10(snr) if math.isfinite(snr) and snr > 0 else 99.0
    relative_l2_error = float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12))
    peak_flux_error = abs(float(image[sy, sx]) - source_flux_jy)

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


def adopt_completed_evaluations(
    evaluations_dir: Path,
    evaluations: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> int:
    """Take an interrupted run's finished evaluations into this run's state.

    Without this a resumed run restarts its eval ids at 1 and rebuilds
    directories the first attempt already wrote, which fails on the first
    repeated point. With it the ids carry on and a point evaluated before is
    served from the cache instead of being paid for twice - which is the whole
    reason to resume rather than start again.

    The evaluations that were still in flight when the run stopped are thrown
    away first. An evaluation directory with no metrics.json holds nothing
    worth keeping - the run died between creating it and scoring it - and
    simulate_measurement_set() creates each one with `exist_ok=False`, on
    purpose, so that two ranks cannot land on the same directory. Left in
    place, one of these would crash the resumed run the moment the sampler
    proposed that point again: the very run this is supposed to rescue.
    """
    import shutil

    for leftover in sorted(evaluations_dir.glob("eval-*")):
        if leftover.is_dir() and not (leftover / "metrics.json").exists():
            # ignore_errors because every rank runs this, and they are all
            # removing the same directories at the same moment.
            shutil.rmtree(leftover, ignore_errors=True)
    for record in load_evaluations_from_dir(evaluations_dir):
        evaluations.append(record)
        cache[params_key(record["params"])] = record
    return len(evaluations)


def self_check_resume_adoption() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        evaluations_dir = Path(tmp)
        params = {"channel_count": 2, "noise_seed": 7}
        eval_dir = evaluations_dir / "eval-0001-abc"
        eval_dir.mkdir()
        write_evaluation_record(eval_dir, {"eval_id": 1, "params": params, "objective": 0.5})

        evaluations: list[dict[str, Any]] = []
        cache: dict[str, dict[str, Any]] = {}
        assert adopt_completed_evaluations(evaluations_dir, evaluations, cache) == 1
        # Keyed the way the likelihood keys it, or the resumed run would
        # recompute the point and collide with its own directory.
        assert params_key(params) in cache
        assert cache[params_key(params)]["objective"] == 0.5
        # The next eval id continues rather than restarting at 1.
        assert len(evaluations) + 1 == 2

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

        evaluations = []
        cache = {}
        assert adopt_completed_evaluations(evaluations_dir, evaluations, cache) == 1
        assert finished.exists()
        assert not in_flight.exists()

    with tempfile.TemporaryDirectory() as tmp:
        # A fresh run adopts nothing and starts at id 1.
        evaluations = []
        cache = {}
        assert adopt_completed_evaluations(Path(tmp), evaluations, cache) == 0


def write_evaluation_record(eval_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    (eval_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


_SIMULATE_WORKERS: dict[str, "subprocess.Popen | FifoWorker"] = {}


# Set when a pooled worker had to be killed. Its FIFO pair died with it, so
# reconnecting can only find a corpse, and the ENXIO wait below would be pure
# delay in front of a retry that has to fall back to a rank-started worker
# anyway. Per rank, because each rank is its own process.
_FIFO_POOL_ABANDONED = False


def fifo_worker_pgrep_pattern(base: Path) -> str:
    """pgrep -f regex matching exactly the pooled worker serving `base`.

    The `$` does the real work, and check_fifo_kill_pattern() in
    scripts/test_watchdogs.py is the guard on it: without the anchor this
    matches every rank whose number starts with this one's, so killing rank 1
    would take ranks 10 to 19 with it. The anchor also stops the `sh -c`
    carrying this pattern from matching itself, because that command line
    continues past the pattern and ends with the `$` as a literal character.
    The bracket is belt and braces for the same self-match - it is a character
    class here and a literal `[` in any command line quoting this string - and
    a mutation of it changes no behaviour while the anchor stands.
    """
    return f"serve --fif[o] {base}$"


class FifoWorker:
    """A `--serve --fifo` worker the run script already started.

    Same `.stdin`/`.stdout`/`.kill()`/`.terminate()` surface as the
    `subprocess.Popen` below, over the FIFO pair the worker is serving on.
    Closing stdin is what ends it: the worker's request loop sees EOF and
    exits.
    """

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
        """SIGKILL this worker inside the sidecar, wedged meqserver and all.

        A worker that stopped answering is still holding its end of the FIFO
        pair open, so leaving it alive means the next attempt reconnects to the
        same wedged process rather than a fresh one - and its meqserver is
        ~0.4GB that nothing else will ever reclaim. `--fifo <base>` is unique to
        this rank's worker, so pgrep finds it with no pid file to keep in sync.
        The bracket in the pattern is what stops the `sh -c` running it from
        matching its own command line; the image has pgrep but no pkill, hence
        killing the meqserver child by pid.
        """
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
    if not fifo_dir or _FIFO_POOL_ABANDONED:
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
        return FifoWorker(write_fd, Path(f"{base}.out"), container, base)


def simulate_worker(meqtrees_image: str, platform: str) -> subprocess.Popen | FifoWorker:
    """This rank's long-lived `simulate_point_source_ms.py --serve` process.

    Even inside a reused sidecar container, a per-evaluation `docker exec` of the
    simulate script paid ~0.45s of the ~0.7s it took: the Python interpreter,
    numpy/casacore and Timba imports, starting a meqserver and reaping it again.
    One worker per rank keeps all of that warm and leaves only the per-evaluation
    compile, RIME predict and noise fill.
    """
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
        container = sidecar_container(r2d2_image, platform, r2d2_checkpoint_mount(checkpoints_dir))
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
    """Run one `imager.py` in this rank's R2D2 worker, same shape as sidecar_run().

    This is the request the host's OOM killer interrupts when memory runs
    short, so a death here is retried against a fresh worker and, if it
    happens again, reported as WORKER_DIED - never as an `imager.py` exit
    status, which is what the sampler would otherwise score as a failure mode.
    """
    request = {"argv": argv, "stdout": str(stdout_path), "stderr": str(stderr_path)}
    started = time.perf_counter()
    for attempt in worker_attempts():
        worker = r2d2_worker(r2d2_image, platform, checkpoints_dir)
        worker.stdin.write(json.dumps(request) + "\n")
        worker.stdin.flush()
        reply = worker_reply(worker.stdout, IMAGING_REPLY_TIMEOUT)
        if reply:
            answer = json.loads(reply)
            return DockerRunResult(
                returncode=answer["returncode"],
                wall_seconds=time.perf_counter() - started,
                peak_memory_bytes=answer["peak_memory_bytes"],
            )
        if reply is None:
            # ponytail: as in sidecar_run(), this kills the `docker exec`
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

    A worker that dies without answering is dropped from the cache so the retry
    gets a fresh one instead of inheriting the corpse. One that stops answering
    without dying - the MeqTrees/meqserver deadlock SIMULATE_REPLY_TIMEOUT
    exists for - is killed first, so that it leaves the same way. Either one
    surviving the retries reports WORKER_DIED, not an exit status the simulate
    never returned, with the reason in the caller's stderr log.
    """
    for attempt in worker_attempts():
        worker = simulate_worker(meqtrees_image, platform)
        worker.stdin.write(json.dumps(request) + "\n")
        worker.stdin.flush()
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
    """A worker that goes silent must be retried, then reported as WORKER_DIED.

    The run this guards against is a real one: MeqTrees deadlocked with its
    meqserver, the worker stayed alive and answered nothing, and the rank
    waiting on it sat in readline() for 82 of the run's 84 minutes with the
    other 19 ranks spinning in the collective behind it. Silence has to leave
    by the same door a death does - killed, dropped, retried against a fresh
    worker - and a silence that outlasts the retries has to reach the sampler
    as WORKER_DIED rather than as an exit status the simulate never returned.
    """
    import tempfile

    class Worker:
        """A worker whose reply never arrives, or arrives, on a real fd."""

        def __init__(self, reply: str | None) -> None:
            read_fd, self._write_fd = os.pipe()
            self.stdout = os.fdopen(read_fd, "r")
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
    finally:
        globals().update(original)

    print("worker timeout self-check passed")


def self_check_worker_pool_connect() -> None:
    """Both workers must reach their pre-warmed FIFO pool, not just simulate.

    `_connect_shell_started_worker` has two call sites, and a signature change
    that updated only the simulate one left `r2d2_worker` raising TypeError on
    the first evaluation of every pooled R2D2 run - invisible to a WSClean run,
    which never takes this path. Calling both here is what makes the two call
    sites drift loudly instead of silently.
    """
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
    """The parameter space survived the trip through defaults.toml.

    5, not 6: the default-enabled count in the committed defaults.toml, which
    is what `source_offset_fraction` being `enabled = false` leaves. Meant to
    be retuned by toggling `enabled`, same as the ranges below - bump this when
    the default-enabled set changes, so a stray toggle in a commit still trips
    a canary.
    """
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
    """`enabled = false` and `NS_ENABLE_PARAMS`/`NS_DISABLE_PARAMS` actually toggle a dimension."""
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
    """`source_offset_fraction` reaches `cube_to_params()` as an (l, m) offset that `source_pixel()` can invert."""
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
    """Every fitted window sits inside one receiver band, and inverts back."""
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


def self_check_image_pixel_size() -> None:
    """The derived cell has to be the one R2D2 picks for the same sampling pattern.

    Reproduces upstream's expression (src/utils/io.py) from raw u/v the way
    ms_to_r2d2_mat.py writes them, rather than restating this module's own
    formula, so a drift on either side shows up here.
    """
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
