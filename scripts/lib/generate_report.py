"""
Builds an HTML report from the nested-sampling runs under
results/nested-sampling/*/summary.json - one page per run, an index, and
an images/ directory of the PNGs those pages reference.

Run via scripts/generate-report.sh, which wraps this in the r2d2 image so it
can reuse the imager's own astropy + matplotlib + anesthetic rather than
requiring a host Python environment - same approach as scripts/plot-fits.sh.
"""
import argparse
import glob
import hashlib
import html
import io
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = "/workspace/repo"
NESTED_SAMPLING_DIR = os.path.join(REPO_ROOT, "results/nested-sampling")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nested_sampling"))

from common import (  # noqa: E402
    PROFILING_VIEW_NOTE,
    format_duration,
    format_share,
    profiling_breakdown,
)

LOG_Z_RE = re.compile(
    r"log\(Z\)\s*=\s*([-\d.]+E[+-]\d+)\s*\+/-\s*([-\d.]+E[+-]\d+)",
    re.IGNORECASE,
)
RUN_ID_TS_RE = re.compile(r"(\d{8}T\d{6}Z)$")

# Every generated page records the version of the code that produced it, so a
# later run can tell "already rendered" from "rendered by an older report".
# The version is the hash of this file: editing the rendering or the CSS bumps
# it by itself, which is one less thing to remember than a hand-kept number.
REPORT_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
REPORT_VERSION_RE = re.compile(r'<meta name="report-version" content="([0-9a-f]+)">')


def page_report_version(path):
    """Version stamped into an already-written page, or None if absent/unreadable."""
    try:
        with open(path) as f:
            head = f.read(2048)
    except OSError:
        return None
    m = REPORT_VERSION_RE.search(head)
    return m.group(1) if m else None


# The drawing stack is ~0.5s of import and is only touched when a PNG has to be
# drawn, so it loads on demand and a run that reuses the whole image store - the
# page-only rebuild after a REPORT_VERSION bump, and the all-current no-op -
# never pays for it. It splits in two because the two kinds of PNG need
# different halves: the corner plot needs only matplotlib (anesthetic pulls it
# in regardless), the eval rasters need astropy + PIL on top. Keeping them apart
# keeps astropy off the corner plot's critical path, which is the longest task
# in a cold build.
def load_plot_libs():
    global plt
    if "plt" in globals():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def load_chain_libs():
    """anesthetic, which every corner-plot worker needs. Imported in the parent
    so the forked pool inherits it once instead of each worker repeating the
    same 0.34s - with more runs to draw than cores, that repetition is what the
    build spends its extra CPU on. A missing anesthetic is not fatal - the
    corner plot degrades to "unavailable" - so leave that to the worker."""
    try:
        import anesthetic  # noqa: F401
    except ImportError:
        pass


def load_render_libs():
    global np, fits, AsinhStretch, ImageNormalize, ZScaleInterval, Image
    if "fits" in globals():
        return
    import numpy as np
    from astropy.io import fits
    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
    from PIL import Image

    load_plot_libs()  # render_array_png colour-maps through plt.get_cmap


def _image_norm_for_display(data):
    vmin, vmax = ZScaleInterval().get_limits(data)
    if vmin == vmax:
        # ZScaleInterval degenerates to (0, 0) on sparse images (e.g. a
        # CLEAN component model that's mostly zeros) - fall back to the
        # actual data range rather than feeding AsinhStretch a zero-width
        # interval (which divides by zero and renders every pixel NaN).
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())


# Images live in files next to the pages rather than inlined as base64 data
# URIs. Rendering them is where nearly all of the report's time goes, and their
# inputs (a run's FITS files and chains) never change once a run has finished -
# so a page rebuilt for a report-code change reuses the PNGs it rendered last
# time instead of redrawing every one. Files are content-addressed and never
# deleted; `rm -rf reports/nested-sampling-report` is the way to reclaim space.
IMAGE_SUBDIR = "images"
image_dir = None  # set by main(); the self-checks point it at a temp dir


def tight_bbox(fig):
    """The box `savefig(bbox_inches="tight")` would measure, without its extra pass.

    savefig can't trust the artist positions it is handed - a layout engine may
    still be pending - so before measuring a "tight" box it walks the whole
    figure once in a draw-disabled pass. After `fig.tight_layout()` that is
    wasted work: the layout has run and all it left behind is a do-nothing
    placeholder engine. Clear the engine and hand savefig the measured box, and
    the pass goes away for a byte-identical PNG (~20% of the corner plot's save).
    """
    fig.set_layout_engine(None)
    pad = plt.rcParams["savefig.pad_inches"]
    return fig.get_tightbbox().padded(pad, pad)


def figure_to_png_bytes(fig, **savefig_kw):
    buf = io.BytesIO()
    # compress_level=1: zlib's default (6) costs roughly double the CPU on these
    # small rasters for ~10% fewer bytes.
    savefig_kw.setdefault("pil_kwargs", {"compress_level": 1})
    fig.savefig(buf, format="png", **savefig_kw)
    plt.close(fig)
    return buf.getvalue()


# Bump when a change to the drawing code should retire the PNGs already on
# disk - keys describe the inputs, not how they were drawn.
IMAGE_RENDER_VERSION = "2"


def cached_png(key, render):
    """URL of the PNG for `key`, calling `render() -> bytes | None` only if absent."""
    name = hashlib.sha1(f"{IMAGE_RENDER_VERSION}|{key}".encode()).hexdigest()[:16] + ".png"
    path = os.path.join(image_dir, name)
    if not os.path.exists(path):
        data = render()
        if data is None:
            return None
        os.makedirs(image_dir, exist_ok=True)
        # Write-then-rename so an interrupted run can't leave a truncated PNG
        # that every later run would then happily reuse.
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return f"{IMAGE_SUBDIR}/{name}"


def file_stamp(path):
    """Identity of a file's contents for cache keys, without reading it."""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return f"{st.st_size}:{st.st_mtime_ns}"


# The eval rasters are the bulk of the report. Drawing one through a matplotlib
# Figure cost ~25ms - half of it tight_layout's trial draw - and upsampled the
# 128x128 data into a ~336px antialiased raster, 224 KiB of PNG the browser then
# scaled back down anyway. Normalising and colour-mapping straight into a PIL
# image at the data's own resolution is ~16x faster for ~5x fewer bytes, and the
# browser's own smoothing makes it indistinguishable at display size. figsize/dpi
# are kept as the caller's pixel budget: an image larger than that is scaled down
# to it, but nothing is ever scaled up.
def render_array_png(data, figsize=(4, 4), dpi=130):
    load_render_libs()
    data = np.squeeze(np.asarray(data, dtype=float))
    if data.ndim != 2:
        return None
    norm = _image_norm_for_display(data)
    # bad="white": matplotlib rendered non-finite pixels as the figure background.
    cmap = plt.get_cmap("inferno").with_extremes(bad="white")
    # [::-1]: FITS row 0 is the bottom of the image (imshow's origin="lower").
    image = Image.fromarray(cmap(norm(data), bytes=True)[::-1, :, :3])
    budget = int(round(min(figsize) * dpi))
    if max(image.size) > budget:
        image.thumbnail((budget, budget), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="png", compress_level=1)
    return buf.getvalue()


def _render_fits_png(path, figsize, dpi):
    load_render_libs()
    return render_array_png(fits.getdata(path), figsize=figsize, dpi=dpi)


def render_fits_image(path, figsize=(4, 4), dpi=130):
    return cached_png(
        f"fits|{path}|{file_stamp(path)}|{figsize}|{dpi}",
        lambda: _render_fits_png(path, figsize, dpi),
    )


def synthesize_truth_array(image_path, source_flux_jy):
    """Mirror common.compute_image_metrics truth construction."""
    load_render_libs()
    data, header = fits.getdata(image_path, header=True)
    image = np.squeeze(np.asarray(data, dtype=np.float64))
    if image.ndim != 2:
        return None
    y_size, x_size = image.shape
    cx = int(round(float(header.get("CRPIX1", x_size / 2.0)) - 1.0))
    cy = int(round(float(header.get("CRPIX2", y_size / 2.0)) - 1.0))
    cx = max(0, min(x_size - 1, cx))
    cy = max(0, min(y_size - 1, cy))
    truth = np.zeros_like(image)
    truth[cy, cx] = source_flux_jy
    return truth


def format_wall_duration(seconds):
    """Run wall-clock seconds for the card header (e.g. 4m 12s), or None when unknown."""
    return None if seconds is None else format_duration(seconds)


def format_run_id_timestamp(run_name):
    """Human-readable UTC label when run_name ends with %Y%m%dT%H%M%SZ (see run-nested-sampling-*.sh)."""
    match = RUN_ID_TS_RE.search(run_name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.strftime("%d %b %Y, %H:%M:%S UTC")


def nested_sampling_run_sort_key(summary_path):
    """Sort key for newest-first nested-sampling cards (run-id UTC, else mtime)."""
    run_name = os.path.basename(os.path.dirname(summary_path))
    match = RUN_ID_TS_RE.search(run_name)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return (0, -dt.timestamp(), run_name)
        except ValueError:
            pass
    try:
        mtime = os.path.getmtime(summary_path)
    except OSError:
        mtime = 0.0
    return (1, -mtime, run_name)


def fmt_value(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


def metrics_table_rows(d, prefix=""):
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            rows.extend(metrics_table_rows(v, prefix=f"{prefix}{k}."))
        else:
            rows.append((f"{prefix}{k}", v))
    return rows


def parse_log_evidence_from_text(text):
    match = LOG_Z_RE.search(text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def parse_log_evidence(stats_path):
    """Return (log_z, log_z_err) from a PolyChord *.stats file, or None."""
    try:
        text = Path(stats_path).read_text(errors="replace")
    except OSError:
        return None
    return parse_log_evidence_from_text(text)


def find_chain_stats(run_dir):
    """Return (stats_path, chain_root_without_extension) or (None, None)."""
    stats_files = sorted(glob.glob(os.path.join(run_dir, "chains", "*.stats")))
    if len(stats_files) != 1:
        return None, None
    stats_path = stats_files[0]
    chain_root = os.path.splitext(stats_path)[0]
    return stats_path, chain_root


def resolve_run_path(run_dir, path):
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    candidate = Path(run_dir) / path
    if candidate.is_file():
        return str(candidate)
    path_str = str(path)
    if "evaluations" in path_str:
        suffix = path_str.split("evaluations", 1)[1].lstrip("/\\")
        candidate = Path(run_dir) / "evaluations" / suffix
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_eval_path(run_dirs, path):
    """resolve_run_path against each candidate run dir (own dir, then merge sources)."""
    for run_dir in run_dirs:
        resolved = resolve_run_path(run_dir, path)
        if resolved:
            return resolved
    return None


def merged_source_run_dirs(summary):
    """Container-mounted directories of a merged run's source runs (see merged_from)."""
    dirs = []
    for entry in summary.get("merged_from") or []:
        name = entry.get("name") if isinstance(entry, dict) else str(entry)
        rel_path = entry.get("path") if isinstance(entry, dict) else None
        candidate = os.path.join(REPO_ROOT, rel_path) if rel_path else None
        if not (candidate and os.path.isdir(candidate)) and name:
            candidate = os.path.join(NESTED_SAMPLING_DIR, name)
        if candidate and os.path.isdir(candidate):
            dirs.append(candidate)
    return dirs


def render_likelihood_plot(run_dir, param_names):
    """URL of the run's corner plot. The anesthetic KDE is the single most
    expensive thing the report draws, so it goes through the image store too."""
    chains = sorted(glob.glob(os.path.join(run_dir, "chains", "*")))
    key = "likelihood|{}|{}|{}".format(
        run_dir,
        ",".join(param_names),
        ",".join(f"{os.path.basename(c)}:{file_stamp(c)}" for c in chains),
    )
    return cached_png(key, lambda: _render_likelihood_png(run_dir, param_names))


# The corner plot is roughly half of a run page's cost and the eval rasters the
# other half, so main() renders them as two independent pool tasks and stitches
# the result in. The page body carries this slot where the plot's section goes.
LIKELIHOOD_SLOT = "<!--likelihood-slot-->"


def likelihood_section(uri):
    if not uri:
        return '<section><h3>Likelihood</h3><p class="empty">Likelihood plot unavailable.</p></section>'
    return f"""
        <section>
          <h3>Likelihood</h3>
          <figure class="likelihood-plot"><img src="{uri}" alt="likelihood corner plot"></figure>
        </section>
        """


# anesthetic draws the corner plot as 15 separate pandas plot calls into one
# 20-axes grid, and after every one of them pandas re-runs its shared-axis tick
# housekeeping over *every* axis in the figure - 44% of plot_2d, for work that
# is idempotent after the first pass. Do it once per axis instead. The repeats
# are not entirely free, though: reading an axis' tick labels un-stales the
# shared view limits, and that side effect is load-bearing - without it each
# diagonal panel's CDF twin drifts off its parent's x range (~5%) and the plot
# changes. Touching viewLim keeps exactly that and drops the rest.
# Best-effort: if pandas moves the private helper, the plot is just slower.
_tick_housekeeping_deduped = False


def dedupe_pandas_tick_housekeeping():
    global _tick_housekeeping_deduped
    if _tick_housekeeping_deduped:
        return True
    _tick_housekeeping_deduped = True
    try:
        from pandas.plotting._matplotlib import tools

        remove_labels = tools._remove_labels_from_axis
    except (ImportError, AttributeError):
        return False

    def once(axis):
        if getattr(axis, "_report_labels_removed", False):
            axis.axes.viewLim  # noqa: B018 - un-stales the shared view limits
            return
        axis._report_labels_removed = True
        remove_labels(axis)

    tools._remove_labels_from_axis = once
    return True


# Drawing the corner plot makes matplotlib recompute every axis' tick positions
# and label text from scratch ~930 times: once per axis per layout pass, and the
# figure gets several (tight_layout, the tight-bbox draw, the render draw, plus
# each spine and axis-label placement in between). That is a third of the plot's
# cost for a pure function of the axis' view interval, data interval and its
# locator/formatter pair, so cache the result per axis against exactly those.
# Reading get_view_interval() to build the key keeps _update_ticks' own viewLim
# un-staling side effect, which the shared diagonal/CDF axes depend on (see
# dedupe_pandas_tick_housekeeping above).
# Best-effort: if matplotlib renames the private method, the plot is just slower.
_tick_updates_memoized = False


def memoize_matplotlib_tick_updates():
    global _tick_updates_memoized
    if _tick_updates_memoized:
        return True
    _tick_updates_memoized = True
    try:
        from matplotlib.axis import Axis

        update_ticks = Axis._update_ticks
    except (ImportError, AttributeError):
        return False

    def memoized(self):
        key = (
            tuple(self.get_view_interval()),
            tuple(self.get_data_interval()),
            self.major.locator,
            self.major.formatter,
            self.minor.locator,
            self.minor.formatter,
        )
        cached = getattr(self, "_report_tick_memo", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ticks = update_ticks(self)
        self._report_tick_memo = (key, ticks)
        return ticks

    Axis._update_ticks = memoized
    return True


# anesthetic's labelled frames resolve every `df[key]` by trying the lookup
# against each of four label-stripped views and keeping the best answer, and
# each attempt rebuilds the axis' paramname -> label mapping from scratch. Over
# one corner plot that is ~420 rebuilds of a handful of Series, so cache them.
# The mapping is a pure function of the pandas Index it is read off, and an
# Index is immutable - anything that adds or drops a column swaps in a new one,
# which misses the cache. The entry holds the Index alive so its id() cannot be
# recycled under the key. fill=False is deliberately not cached: that is the
# variant set_label() mutates in place.
# Best-effort: if anesthetic moves the private class, the plot is just slower.
_labels_map_memoized = False


def memoize_anesthetic_labels_map():
    global _labels_map_memoized
    if _labels_map_memoized:
        return True
    _labels_map_memoized = True
    try:
        from anesthetic.labelled_pandas import _LabelledObject

        get_labels_map = _LabelledObject.get_labels_map
    except (ImportError, AttributeError):
        return False

    cache = {}

    def memoized(self, axis=0, fill=True):
        try:
            index = self._get_axis(axis)
        except (ValueError, TypeError):
            # Multi-axis and axis=None lookups have no single index; anesthetic
            # relies on the raised error, so leave those alone.
            return get_labels_map(self, axis, fill)
        if not fill:
            return get_labels_map(self, axis, fill)
        key = (id(index), self._labels)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = (index, get_labels_map(self, axis, fill))
        return hit[1]

    _LabelledObject.get_labels_map = memoized
    return True


# The other half of that same `df[key]` detour: each of the four attempts first
# builds a *deep copy* of the whole frame with the labels dropped, so one corner
# plot makes ~380 copies of the same handful of frames. The result is a pure
# function of the frame's data and of which (axis, level) pairs actually get
# dropped, so cache it on exactly that.
# The key pins the frame's identity *and* both of its pandas Index objects: an
# Index is immutable, so anything that adds, drops or relabels a column swaps in
# a new one and misses the cache. That is the guard iteration 13 found was
# missing from a plain id(self) cache, which went stale the moment anesthetic
# added its weight columns. Rewriting an existing column's values in place would
# still slip past it, but nothing on the plotting path does that - the frame is
# read-only from load to savefig. The entry holds the frame and its indexes
# alive so their id()s cannot be recycled underneath the key.
# Best-effort: if anesthetic moves the private class, the plot is just slower.
_drop_labels_memoized = False


def memoize_anesthetic_drop_labels():
    global _drop_labels_memoized
    if _drop_labels_memoized:
        return True
    _drop_labels_memoized = True
    try:
        import numpy as np

        from anesthetic.labelled_pandas import _LabelledObject

        drop_labels = _LabelledObject.drop_labels
    except (ImportError, AttributeError):
        return False

    cache = {}

    def memoized(self, axis=0):
        try:
            axes = self.axes
            # Mirrors drop_labels' own loop: same order, same (axis, level)
            # pairs, so two specs share an entry only when they drop the same
            # levels off the same axes in the same order.
            dropped = tuple(
                (a, self.islabelled(a))
                for a in np.atleast_1d(axis)
                if a is not None and self.islabelled(a)
            )
            key = (id(self), tuple(id(ax) for ax in axes), dropped)
        except Exception:
            return drop_labels(self, axis)
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 64:
                cache.clear()
            hit = cache[key] = (self, axes, drop_labels(self, axis))
        return hit[2]

    _LabelledObject.drop_labels = memoized
    return True


def _render_likelihood_png(run_dir, param_names):
    load_plot_libs()
    try:
        from anesthetic_io import load_nested_samples, weight_by_likelihood
    except ImportError:
        return None

    dedupe_pandas_tick_housekeeping()
    memoize_matplotlib_tick_updates()
    memoize_anesthetic_labels_map()
    memoize_anesthetic_drop_labels()

    try:
        samples = weight_by_likelihood(load_nested_samples(run_dir))
    except Exception:
        return None

    plot_params = [name for name in param_names if name in samples.columns]
    if len(plot_params) < 2:
        return None

    # ncompress=False: anesthetic triangular compression fails on some chains.
    for kind, extra in (("kde", {"ncompress": False}), ("scatter", {})):
        try:
            grid = samples.plot_2d(plot_params, kind=kind, **extra)
            fig = grid.iloc[0, 0].figure
            fig.tight_layout()
            return figure_to_png_bytes(fig, bbox_inches=tight_bbox(fig))
        except Exception:
            pass
    return None


def objective_fill(objective, obj_min, obj_max):
    if obj_max > obj_min:
        return (float(objective) - obj_min) / (obj_max - obj_min)
    return 1.0


def run_tab_id(run_name):
    """Sanitize a run directory name into a valid unique HTML id fragment."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name).strip("-") or "run"
    digest = hashlib.sha1(run_name.encode()).hexdigest()[:8]
    return f"{sanitized}-{digest}"


def render_images_likelihood_collapsible(tab_id, eval_images_html, likelihood_html):
    """Collapsed-by-default Images / Likelihood tabs for one nested-sampling run."""
    if not eval_images_html and not likelihood_html:
        return ""
    safe_id = html.escape(tab_id)
    tabset = f"""
    <div class="run-media-tabset">
      <input type="radio" class="tab-images-radio" name="tabs-{safe_id}" id="tab-images-{safe_id}">
      <label for="tab-images-{safe_id}">Images</label>
      <input type="radio" class="tab-likelihood-radio" name="tabs-{safe_id}" id="tab-likelihood-{safe_id}" checked>
      <label for="tab-likelihood-{safe_id}">Likelihood</label>
      <div class="tab-panel tab-panel-images">{eval_images_html}</div>
      <div class="tab-panel tab-panel-likelihood">{likelihood_html}</div>
    </div>
    """
    return f"""
    <details>
      <summary>Run images and likelihood</summary>
      {tabset}
    </details>
    """


# Categorical slot per top-level stage, assigned by identity and never by size,
# so a stage keeps its colour whether or not the other stages are present.
PROFILING_STAGE_COLOURS = {
    "simulate": "var(--series-1)",
    "convert": "var(--series-2)",
    "image_container": "var(--series-3)",
    "metrics": "var(--series-4)",
    "unaccounted": "var(--series-5)",
}


# Enough lanes to read as "the workers ran side by side" without turning a
# 64-rank run into a wall of identical stripes.
PROFILING_LANE_CAP = 8


def render_profiling_lanes(segments, mpi_procs, wall_seconds):
    """Stacked worker lanes plus legend: how the stage times fit inside the wall clock.

    Each lane is one worker's averaged timeline, so the stages along a single
    lane add up to the end-to-end wall clock shown on the run header, while the
    lanes stacked on top of each other add up to the worker-time budget. That
    is the parallelisation made visible: the table's worker-seconds only reach
    the wall clock once they are spread across the lanes they ran on.

    Segments are flex-grown in proportion to their share, so the 2px gaps
    between them come out of the available width rather than overflowing it.
    """
    if not segments:
        return ""
    bars, legend = [], []
    for label, seconds, share, colour in segments:
        lane_seconds = seconds / mpi_procs
        title = (
            f"{label} - {format_duration(lane_seconds)} of each worker's "
            f"{format_duration(wall_seconds)} ({format_share(share)}), "
            f"{format_duration(seconds)} of worker-time in total"
        )
        lane_time = (
            f"{format_duration(lane_seconds)} per worker" if mpi_procs != 1
            else format_duration(lane_seconds)
        )
        bars.append(
            f'<span class="profile-seg" style="--seg-share: {share:.6f}; --seg-colour: {colour}"'
            f' title="{html.escape(title)}"></span>'
        )
        legend.append(
            f'<li><span class="profile-swatch" style="--seg-colour: {colour}"></span>'
            f'<span>{html.escape(label)}</span>'
            f'<strong>{html.escape(format_share(share))}</strong>'
            f'<span class="profile-legend-time">{html.escape(lane_time)}</span></li>'
        )
    shown = max(1, min(mpi_procs, PROFILING_LANE_CAP))
    lane = f'<div class="profile-bar">{"".join(bars)}</div>'
    summary_text = ", ".join(f"{label} {format_share(share)}" for label, _, share, _ in segments)
    if mpi_procs == 1:
        caption = "one worker, so its timeline is the run's wall clock"
    else:
        caption = (
            f"one lane per worker, {mpi_procs} of them running side by side, each spanning"
            f" the run's {format_duration(wall_seconds)} wall clock (averaged across workers)"
        )
        if shown != mpi_procs:
            caption += f" - showing {shown} of {mpi_procs} identical lanes"
    aria = (
        f"Each of the {mpi_procs} workers spends its {format_duration(wall_seconds)} "
        f"wall clock like this: {summary_text}"
    )
    return f"""
      <div class="profile-lanes" role="img" aria-label="{html.escape(aria)}">
        <div class="profile-lane-scale"><span>0</span>
          <span>{html.escape(format_duration(wall_seconds))} wall clock</span></div>
        <div class="profile-lane-stack">{lane * shown}</div>
      </div>
      <p class="profile-lane-caption">{html.escape(caption)}</p>
      <ul class="profile-legend">{''.join(legend)}</ul>
    """


def render_profiling(summary):
    """Collapsed per-stage timing breakdown, sharing its numbers with
    scripts/profile-nested-sampling-run.py via common.profiling_breakdown."""
    profiling = summary.get("profiling")
    if not profiling:
        return ""

    breakdown = profiling_breakdown(profiling, summary.get("algorithm"))
    mpi_procs = breakdown["mpi_procs"]
    budget = breakdown["worker_seconds_budget"]
    evals = breakdown["evals"]

    # Worker-seconds and wall-clock seconds only differ once the run is
    # parallel, so a serial run keeps the narrower table.
    show_wall = mpi_procs != 1

    def row(label, seconds, share, per_eval=None, evals_count="", indent=False, emphasis=False):
        # A missing stage total leaves the cell blank rather than dropping it,
        # which would shift every later cell into the wrong column.
        wall = [format_duration(seconds / mpi_procs) if seconds is not None else ""] if show_wall else []
        # format_share can return "<0.1%", so every cell goes through escaping.
        cells = [html.escape(text) for text in (
            label,
            format_duration(seconds),
            *wall,
            format_duration(per_eval) if per_eval is not None else "",
            format_share(share),
            str(evals_count),
        )]
        if emphasis:
            cells = [f"<strong>{cell}</strong>" if cell else cell for cell in cells]
        stage_class = ' class="profile-stage-sub"' if indent else ""
        numeric = "".join(f'<td class="num">{cell}</td>' for cell in cells[1:])
        return f"<tr><td{stage_class}>{cells[0]}</td>{numeric}</tr>"

    body = "".join(
        row(
            r["label"],
            r["seconds"],
            r["share"],
            r["per_eval_seconds"],
            r["evals"] or "",
            indent=r["is_sub"],
        )
        for r in breakdown["rows"]
    )
    accounted = breakdown["accounted_seconds"]
    body += row(
        "accounted (sum of stages above)",
        accounted,
        breakdown["accounted_share"],
        accounted / evals if evals else None,
        emphasis=True,
    )
    body += row(
        breakdown["unaccounted_label"],
        breakdown["unaccounted_seconds"],
        breakdown["unaccounted_share"],
        emphasis=True,
    )
    # The row the whole table exists to land on: divided across the workers, the
    # stages above come to the end-to-end wall clock on the run header.
    wall_seconds = (budget / mpi_procs) if budget else 0.0
    body += row(
        "end-to-end (accounted + unaccounted)",
        budget,
        1.0 if budget else None,
        emphasis=True,
    )

    segments = [
        (r["label"], r["seconds"], r["share"], PROFILING_STAGE_COLOURS.get(r["key"], "var(--series-1)"))
        for r in breakdown["rows"]
        if not r["is_sub"] and r["share"]
    ]
    if breakdown["unaccounted_share"]:
        segments.append((
            breakdown["unaccounted_label"],
            breakdown["unaccounted_seconds"],
            breakdown["unaccounted_share"],
            PROFILING_STAGE_COLOURS["unaccounted"],
        ))

    heading = " · ".join([
        f"{mpi_procs} worker{'s' if mpi_procs != 1 else ''}",
        f"wall clock {format_duration(breakdown['total_wall_seconds'])}",
        f"worker-time {format_duration(budget)} ({mpi_procs} × wall clock)",
        f"{evals} evaluations",
    ])
    # Spelled out left to right so the arithmetic behind the run header's wall
    # clock is readable without doing any of it yourself; the term it lands on
    # is the one the header shows, so that term carries the emphasis.
    terms = [
        f"{format_duration(accounted)} accounted",
        f"+ {format_duration(breakdown['unaccounted_seconds'])} unaccounted",
    ]
    if mpi_procs != 1:
        terms.append(f"= {format_duration(budget)} of worker-time")
        terms.append(f"÷ {mpi_procs} workers")
    equation = " ".join(html.escape(term) for term in terms)
    equation += (
        f' <strong>= {html.escape(format_duration(wall_seconds))}'
        " end-to-end wall clock</strong>"
    )
    wall_header = '<th class="num">wall clock</th>' if show_wall else ""
    return f"""
    <details>
      <summary>Profiling (where the run's time went)</summary>
      <p class="purpose">{html.escape(heading)}</p>
      {render_profiling_lanes(segments, mpi_procs, wall_seconds)}
      <p class="profile-equation">{equation}</p>
      <div class="eval-table-wrap">
        <table class="eval-table">
          <thead><tr>
            <th>stage</th><th class="num">worker-time</th>{wall_header}<th class="num">per eval</th>
            <th class="num">share</th><th class="num">evals</th>
          </tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
      <p class="purpose">{html.escape(PROFILING_VIEW_NOTE)}</p>
    </details>
    """


def format_searched_params(params, parameter_space):
    parts = []
    for spec in parameter_space:
        name = spec.get("name")
        if not name:
            continue
        value = (params or {}).get(name)
        if value is None:
            continue
        parts.append(f"{html.escape(name)}={fmt_value(value)}")
    return " · ".join(parts)


def render_eval_recon(image_path, eval_id, figsize=(2.8, 2.8), dpi=120):
    if not image_path:
        return '<span class="empty">—</span>'
    recon_uri = render_fits_image(image_path, figsize=figsize, dpi=dpi)
    if not recon_uri:
        return '<span class="empty">—</span>'
    return (
        f'<figure class="eval-recon">'
        f'<img src="{recon_uri}" alt="eval {html.escape(str(eval_id))} reconstruction">'
        f'<figcaption>recon</figcaption></figure>'
    )


def _render_truth_png(image_path, source_flux_jy, figsize, dpi):
    truth_array = synthesize_truth_array(image_path, source_flux_jy)
    return None if truth_array is None else render_array_png(truth_array, figsize=figsize, dpi=dpi)


def render_shared_truth_image(image_path, source_flux_jy, figsize=(3.2, 3.2), dpi=120):
    if not image_path:
        return ""
    truth_uri = cached_png(
        f"truth|{image_path}|{file_stamp(image_path)}|{source_flux_jy}|{figsize}|{dpi}",
        lambda: _render_truth_png(image_path, source_flux_jy, figsize, dpi),
    )
    if not truth_uri:
        return ""
    return (
        '<div class="eval-truth-shared">'
        f'<figure><img src="{truth_uri}" alt="shared ground truth">'
        f'<figcaption>Ground truth (shared across all evaluations)</figcaption></figure>'
        "</div>"
    )


def render_eval_card(ev, parameter_space, run_dirs, metric, is_best):
    eval_id = ev.get("eval_id", "?")
    params = ev.get("params") or {}
    image_path = resolve_eval_path(run_dirs, (ev.get("paths") or {}).get("image"))
    metric_label = html.escape(metric or "objective")
    best_class = " is-best" if is_best else ""
    params_caption = format_searched_params(params, parameter_space)
    recon_html = render_eval_recon(image_path, eval_id)
    return f"""
    <article class="eval-card{best_class}">
      <header class="eval-card-header">
        <span class="eval-card-id">#{html.escape(str(eval_id))}</span>
        <span class="eval-card-objective">{metric_label} <strong>{fmt_value(ev.get('objective'))}</strong></span>
      </header>
      {recon_html}
      <p class="eval-params">{params_caption or '<span class="empty">no searched parameters</span>'}</p>
    </article>
    """


def render_eval_glance_summary(evaluations, metric, failed_count):
    if not evaluations:
        if failed_count:
            return (
                '<div class="eval-glance">'
                f'<div class="headline"><span class="badge badge-warn">{failed_count} failed</span>'
                " · no successful evaluations</div></div>"
            )
        return ""

    objectives = [float(ev.get("objective", 0)) for ev in evaluations]
    obj_min, obj_max = min(objectives), max(objectives)
    best = evaluations[0]
    metric_label = html.escape(metric or "objective")

    headline_bits = [
        f'<span class="badge badge-ok">{len(evaluations)} succeeded</span>',
    ]
    if failed_count:
        headline_bits.append(f'<span class="badge badge-warn">{failed_count} failed</span>')
    headline_bits.extend(
        [
            f"optimized <strong>{metric_label}</strong>",
            (
                f"range <strong>{fmt_value(obj_min)}</strong>"
                f'<span class="delta">–</span><strong>{fmt_value(obj_max)}</strong>'
            ),
            (
                f"best eval <strong>#{html.escape(str(best.get('eval_id', '?')))}</strong>"
                f' <span class="delta">({fmt_value(best.get("objective"))})</span>'
            ),
        ]
    )
    headline_html = f'<div class="headline">{" · ".join(headline_bits)}</div>'

    strip_cells = []
    best_eval_id = best.get("eval_id")
    for ev in evaluations:
        eval_id = ev.get("eval_id", "?")
        objective = float(ev.get("objective", 0))
        fill = objective_fill(objective, obj_min, obj_max)
        best_class = " is-best" if eval_id == best_eval_id else ""
        title = html.escape(f"eval {eval_id}: {fmt_value(objective)}")
        strip_cells.append(
            f'<div class="eval-strip-cell{best_class}" style="--fill:{fill:.4f}" title="{title}"></div>'
        )
    strip_html = (
        '<div class="eval-strip-wrap">'
        f'<div class="eval-strip" aria-label="Evaluation objectives from best to worst">'
        f'{"".join(strip_cells)}</div>'
        '<p class="eval-strip-label">Each bar is one evaluation (best left); height encodes objective within this run.</p>'
        "</div>"
    )

    return f'<div class="eval-glance">{headline_html}{strip_html}</div>'


def render_eval_images(evaluations, metric, run_dirs, parameter_space):
    if not evaluations:
        return ""

    best = evaluations[0]
    truth_ref_ev = next(
        (
            ev
            for ev in evaluations
            if resolve_eval_path(run_dirs, (ev.get("paths") or {}).get("image"))
        ),
        evaluations[0],
    )
    truth_image_path = resolve_eval_path(run_dirs, (truth_ref_ev.get("paths") or {}).get("image"))
    truth_source_flux_jy = float((truth_ref_ev.get("params") or {}).get("source_flux_jy", 1.0))
    truth_html = render_shared_truth_image(truth_image_path, truth_source_flux_jy)

    best_eval_id = best.get("eval_id")
    cards = [
        render_eval_card(ev, parameter_space, run_dirs, metric, ev.get("eval_id") == best_eval_id)
        for ev in evaluations
    ]
    cards_html = f'<div class="eval-gallery">{"".join(cards)}</div>'

    return f'<div class="eval-images">{truth_html}{cards_html}</div>'


def render_nested_sampling_run(summary_path, likelihood_html=None):
    """`likelihood_html` pre-rendered by the caller, or None to draw it inline."""
    run_dir = os.path.dirname(summary_path)
    run_name = os.path.basename(run_dir)
    tab_id = run_tab_id(run_name)
    with open(summary_path) as f:
        summary = json.load(f)
    run_dirs = [run_dir] + merged_source_run_dirs(summary)

    algorithm = summary.get("algorithm", "?")
    vla_config = summary.get("vla_config", "?")
    run_type = summary.get("run_type", "")
    likelihood_framing = summary.get("likelihood_framing", "")
    metric = summary.get("metric", "")
    polychord = summary.get("polychord", {})
    parameter_space = summary.get("parameter_space", [])
    space_names = [spec["name"] for spec in parameter_space if "name" in spec]

    evaluations = [ev for ev in summary.get("evaluations", []) if "error" not in ev]
    evaluations.sort(key=lambda item: (-float(item.get("objective", 0)), item.get("eval_id", 0)))

    param_names = list(space_names)
    for ev in evaluations:
        for key in (ev.get("params") or {}):
            if key not in param_names:
                param_names.append(key)

    duration_html = ""
    duration_label = format_wall_duration(summary.get("total_wall_seconds"))
    if duration_label:
        duration_html = f'<span class="run-duration">{html.escape(duration_label)}</span>'

    run_name_bits = [f'<span class="ts">{html.escape(run_name)}</span>']
    run_ts_label = format_run_id_timestamp(run_name)
    if run_ts_label:
        run_name_bits.append(f'<span class="ts">{html.escape(run_ts_label)}</span>')
    run_name_html = " ".join(run_name_bits)

    header = f"""
    <header class="card-header">
      <div class="card-header-top">
        <h2>nested sampling: {html.escape(str(algorithm))}
          {run_name_html}</h2>
        {duration_html}
      </div>
      <div class="badges">
        {'<span class="badge badge-ok">merged</span>' if summary.get('merged_from') else ''}
        <span class="badge">{html.escape(str(vla_config))}</span>
        <span class="badge">nlive {html.escape(str(polychord.get('nlive', '?')))}</span>
        <span class="badge">repeats {html.escape(str(polychord.get('num_repeats', '?')))}</span>
        <span class="badge">max_ndead {html.escape(str(polychord.get('max_ndead', '?')))}</span>
        <span class="badge">seed {html.escape(str(polychord.get('seed', '?')))}</span>
        {f'<span class="badge">{html.escape(metric)}</span>' if metric else ''}
      </div>
    </header>
    """

    meta_bits = []
    if run_type:
        meta_bits.append(html.escape(run_type))
    if likelihood_framing:
        meta_bits.append(html.escape(likelihood_framing))
    meta_html = ""
    if meta_bits:
        meta_html = f'<p class="purpose">{" · ".join(meta_bits)}</p>'

    summary_log_z = summary.get("log_z")
    if summary_log_z is not None:
        summary_log_z_err = summary.get("log_z_err")
        err_html = (
            f'<span class="delta">± {float(summary_log_z_err):.4g}</span>'
            if summary_log_z_err is not None
            else ""
        )
        evidence_html = f"""
        <section>
          <h3>Evidence</h3>
          <div class="headline">log(Z) = <strong>{float(summary_log_z):.4g}</strong>{err_html}</div>
          <p class="purpose">From summary.json log_z</p>
        </section>
        """
    else:
        evidence_html = '<section><h3>Evidence</h3><p class="empty">Global evidence unavailable (no chains/*.stats file).</p></section>'
        stats_path, _ = find_chain_stats(run_dir)
        if stats_path:
            parsed = parse_log_evidence(stats_path)
            if parsed:
                log_z, log_z_err = parsed
                evidence_html = f"""
                <section>
                  <h3>Evidence</h3>
                  <div class="headline">log(Z) = <strong>{log_z:.4g}</strong>
                    <span class="delta">± {log_z_err:.4g}</span></div>
                  <p class="purpose">Parsed from {html.escape(os.path.relpath(stats_path, REPO_ROOT))}</p>
                </section>
                """
            else:
                evidence_html = (
                    '<section><h3>Evidence</h3>'
                    f'<p class="empty">Could not parse log(Z) from {html.escape(os.path.basename(stats_path))}.</p></section>'
                )

    metric_keys = []
    for ev in evaluations:
        for key in (ev.get("metrics") or {}):
            if key not in metric_keys:
                metric_keys.append(key)

    eval_header = (
        "<tr><th>eval</th>"
        + "".join(f"<th>{html.escape(name)}</th>" for name in param_names)
        + "".join(f"<th>{html.escape(key)}</th>" for key in metric_keys)
        + "<th>objective</th><th>image</th></tr>"
    )
    eval_rows = []
    for ev in evaluations:
        params = ev.get("params", {})
        metrics = ev.get("metrics", {})
        image_path = resolve_eval_path(run_dirs, (ev.get("paths") or {}).get("image"))
        thumb = render_eval_recon(image_path, ev.get("eval_id", "?"))
        eval_rows.append(
            "<tr>"
            f"<td>{html.escape(str(ev.get('eval_id', '?')))}</td>"
            + "".join(f"<td>{fmt_value(params.get(name))}</td>" for name in param_names)
            + "".join(f"<td>{fmt_value(metrics.get(key))}</td>" for key in metric_keys)
            + f"<td>{fmt_value(ev.get('objective'))}</td>"
            f"<td>{thumb}</td>"
            "</tr>"
        )

    failed = [ev for ev in summary.get("evaluations", []) if "error" in ev]
    failed_html = ""
    if failed:
        failed_rows = "".join(
            f"<tr><td>{html.escape(str(ev.get('eval_id', '?')))}</td>"
            f"<td>{html.escape(str(ev.get('error', '?')))}</td></tr>"
            for ev in failed
        )
        failed_html = f"""
        <details>
          <summary>{len(failed)} failed evaluation(s)</summary>
          <table class="kv">{failed_rows}</table>
        </details>
        """

    if likelihood_html is None:
        likelihood_html = likelihood_section(render_likelihood_plot(run_dir, space_names))

    evaluations_html = ""
    if eval_rows:
        glance_summary_html = render_eval_glance_summary(evaluations, metric, len(failed))
        eval_images_html = render_eval_images(evaluations, metric, run_dirs, parameter_space)
        images_collapsible = render_images_likelihood_collapsible(
            tab_id, eval_images_html, likelihood_html
        )
        evaluations_html = f"""
        <section>
          <h3>Evaluations</h3>
          {glance_summary_html}
          {images_collapsible}
          <details>
            <summary>{len(eval_rows)} evaluations (raw table)</summary>
            <div class="eval-table-wrap">
              <table class="eval-table">
                <thead>{eval_header}</thead>
                <tbody>{"".join(eval_rows)}</tbody>
              </table>
            </div>
          </details>
          {failed_html}
        </section>
        """
    elif likelihood_html:
        evaluations_html = f"""
        <section>
          <h3>Evaluations</h3>
          {render_images_likelihood_collapsible(tab_id, "", likelihood_html)}
        </section>
        """

    fixed_hp = summary.get("wsclean_fixed_hyperparameters") or summary.get("algorithm_fixed_hyperparameters")
    fixed_html = ""
    if fixed_hp:
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{fmt_value(v)}</td></tr>"
            for k, v in fixed_hp.items()
        )
        fixed_html = f"""
        <details>
          <summary>Fixed algorithm hyperparameters</summary>
          <table class="kv">{rows}</table>
        </details>
        """

    rel_summary = os.path.relpath(summary_path, REPO_ROOT)
    return f"""
    <article class="card nested-sampling-card">
      {header}
      {meta_html}
      {evidence_html}
      {evaluations_html}
      {render_profiling(summary)}
      {fixed_html}
      <p class="manifest-name">{html.escape(rel_summary)}</p>
    </article>
    """


CSS = """
:root {
  color-scheme: light dark;
  /* Categorical slots 1-5, validated for colour-vision deficiency against the
     light surface; the dark column below is the same hues re-stepped for the
     dark surface rather than an automatic flip. */
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
  }
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 980px; margin: 2rem auto; padding: 0 1rem;
  background: Canvas; color: CanvasText;
}
h1 { font-size: 1.4rem; }
.subtitle { opacity: 0.7; margin-top: -0.5rem; }
.card {
  border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
  border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;
}
.card-header h2 { margin-bottom: 0.4rem; font-size: 1.1rem; }
.card-header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 0.4rem;
}
.card-header-top h2 { margin-bottom: 0; flex: 1 1 auto; min-width: 0; }
.run-duration {
  font-size: 0.85rem;
  opacity: 0.7;
  white-space: nowrap;
  flex-shrink: 0;
  padding-top: 0.1rem;
}
.card-header .ts { opacity: 0.6; font-weight: normal; font-size: 0.85rem; }
.badges { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.badge {
  font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 999px;
  background: color-mix(in srgb, CanvasText 10%, transparent);
}
.badge-ok { background: color-mix(in srgb, #2e9e5b 25%, transparent); }
.badge-warn { background: color-mix(in srgb, #d97706 25%, transparent); }
.purpose { opacity: 0.8; font-size: 0.9rem; }
.headline { font-size: 1.1rem; margin: 0.5rem 0; }
.headline .delta { font-size: 0.85rem; opacity: 0.7; }
table.kv { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.5rem 0; }
table.kv td { padding: 0.25rem 0.5rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent); }
table.kv td:first-child { opacity: 0.65; white-space: nowrap; }
table.kv td:last-child { word-break: break-all; }
details summary { cursor: pointer; font-size: 0.9rem; margin-top: 0.5rem; }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
.gallery figure { margin: 0; min-width: 0; }
.gallery img { width: 100%; height: auto; border-radius: 6px; display: block; }
.gallery figcaption { font-size: 0.75rem; opacity: 0.7; text-align: center; margin-top: 0.25rem; word-break: break-all; }
.manifest-name { font-size: 0.75rem; opacity: 0.5; margin-top: 0.75rem; margin-bottom: 0; }
.empty { opacity: 0.6; }
.eval-table-wrap { overflow-x: auto; margin: 0.5rem 0; }
.eval-table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
.eval-table th, .eval-table td { padding: 0.35rem 0.5rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent); text-align: left; vertical-align: top; }
.eval-table th { opacity: 0.7; white-space: nowrap; }
.eval-table td.num, .eval-table th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.profile-stage-sub { padding-left: 1.5rem; opacity: 0.8; }
.profile-bar {
  display: flex; gap: 2px;
  width: 100%; height: 1.5rem;
  margin: 0.75rem 0 0.5rem;
}
.profile-seg {
  flex: var(--seg-share) 1 0;
  min-width: 3px;
  background: var(--seg-colour);
  border-radius: 3px;
}
.profile-legend {
  display: flex; flex-wrap: wrap; gap: 0.3rem 1.1rem;
  list-style: none; padding: 0; margin: 0 0 0.5rem;
  font-size: 0.78rem;
}
.profile-legend li { display: flex; align-items: center; gap: 0.35rem; }
.profile-swatch { width: 0.7rem; height: 0.7rem; border-radius: 3px; background: var(--seg-colour); flex: none; }
.profile-legend-time { opacity: 0.65; font-variant-numeric: tabular-nums; }
.profile-lanes { margin: 0.75rem 0 0.4rem; }
.profile-lane-scale {
  display: flex; justify-content: space-between;
  font-size: 0.7rem; opacity: 0.55; font-variant-numeric: tabular-nums;
  padding-bottom: 0.2rem;
  border-bottom: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
}
.profile-lane-stack { display: flex; flex-direction: column; gap: 2px; padding-top: 0.25rem; }
.profile-lane-stack .profile-bar { height: 0.6rem; margin: 0; }
.profile-lane-caption { font-size: 0.75rem; opacity: 0.7; margin: 0 0 0.5rem; }
.profile-equation {
  font-size: 0.82rem; margin: 0 0 0.75rem;
  font-variant-numeric: tabular-nums;
  padding: 0.4rem 0.6rem;
  /* Neutral, not a series colour - the line is about the total, not one stage. */
  border-left: 3px solid color-mix(in srgb, CanvasText 35%, transparent);
  background: color-mix(in srgb, CanvasText 5%, transparent);
  border-radius: 0 4px 4px 0;
}
.eval-glance { margin: 0.75rem 0 1rem; }
.eval-images { margin: 0.75rem 0; }
.eval-strip-wrap { margin: 0.75rem 0; }
.eval-strip {
  display: flex; align-items: flex-end; gap: 3px;
  height: 3rem; padding: 0 1px;
}
.eval-strip-cell {
  flex: 1 1 0; min-width: 0;
  height: calc(25% + 75% * var(--fill, 0.5));
  background: color-mix(in srgb, AccentColor calc(20% + 60% * var(--fill, 0.5)), transparent);
  border: 1px solid color-mix(in srgb, CanvasText 12%, transparent);
  border-radius: 3px 3px 0 0;
}
.eval-strip-cell.is-best {
  background: color-mix(in srgb, #2e9e5b 50%, transparent);
  border-color: color-mix(in srgb, #2e9e5b 70%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, #2e9e5b 30%, transparent);
}
.eval-strip-label { font-size: 0.75rem; opacity: 0.65; margin: 0.35rem 0 0; }
.eval-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.75rem;
  margin-top: 0.75rem;
}
.eval-card {
  border: 1px solid color-mix(in srgb, CanvasText 15%, transparent);
  border-radius: 8px;
  padding: 0.65rem 0.75rem;
  min-width: 0;
}
.eval-card.is-best {
  border-color: color-mix(in srgb, #2e9e5b 55%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, #2e9e5b 20%, transparent);
}
.eval-card-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  font-size: 0.85rem;
  margin-bottom: 0.45rem;
}
.eval-card-id { opacity: 0.75; }
.eval-card-objective { text-align: right; }
.eval-truth-shared {
  margin: 0.75rem 0;
  padding: 0.65rem 0.75rem;
  max-width: 220px;
  border: 1px dashed color-mix(in srgb, CanvasText 25%, transparent);
  border-radius: 8px;
  background: color-mix(in srgb, CanvasText 4%, transparent);
}
.eval-truth-shared figure { margin: 0; min-width: 0; }
.eval-truth-shared img {
  width: 100%;
  height: auto;
  border-radius: 4px;
  display: block;
  border: 1px solid color-mix(in srgb, CanvasText 10%, transparent);
}
.eval-truth-shared figcaption {
  font-size: 0.75rem;
  font-weight: 500;
  opacity: 0.85;
  text-align: center;
  margin-top: 0.35rem;
}
.eval-recon { margin: 0; min-width: 0; }
.eval-recon img {
  width: 100%;
  height: auto;
  border-radius: 4px;
  display: block;
  border: 1px solid color-mix(in srgb, CanvasText 10%, transparent);
}
.eval-recon figcaption {
  font-size: 0.65rem;
  opacity: 0.65;
  text-align: center;
  margin-top: 0.2rem;
}
.eval-params {
  font-size: 0.72rem;
  opacity: 0.8;
  margin: 0.45rem 0 0;
  line-height: 1.35;
  word-break: break-word;
}
.eval-table .eval-recon { min-width: 120px; }
.likelihood-plot { margin: 0.5rem 0; }
.likelihood-plot img { max-width: 100%; height: auto; border-radius: 6px; }
.run-media-tabset { margin-top: 0.5rem; }
.run-media-tabset input[type="radio"] {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
  margin: 0;
}
.run-media-tabset label {
  display: inline-block;
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0.35rem 0.75rem;
  margin: 0 0.15rem 0 0;
  border: 1px solid transparent;
  border-radius: 6px 6px 0 0;
  opacity: 0.65;
}
.run-media-tabset input.tab-images-radio:checked + label,
.run-media-tabset input.tab-likelihood-radio:checked + label {
  opacity: 1;
  background: color-mix(in srgb, CanvasText 6%, transparent);
  border-color: color-mix(in srgb, CanvasText 15%, transparent);
  border-bottom-color: Canvas;
}
.run-media-tabset .tab-panel { display: none; padding-top: 0.75rem; }
.run-media-tabset input.tab-images-radio:checked ~ .tab-panel-images { display: block; }
.run-media-tabset input.tab-likelihood-radio:checked ~ .tab-panel-likelihood { display: block; }
.nav { font-size: 0.9rem; margin: 0 0 1rem; }
.nav a { color: inherit; opacity: 0.7; text-decoration: none; }
.nav a:hover { opacity: 1; text-decoration: underline; }
.index-entry { display: block; color: inherit; text-decoration: none; }
a.index-entry:hover { border-color: color-mix(in srgb, CanvasText 45%, transparent); }
.index-entry h2 { font-size: 1.05rem; }
.index-entry-missing { opacity: 0.6; }
.index-run-name { font-size: 0.8rem; opacity: 0.6; margin: 0 0 0.5rem; word-break: break-all; }
.index-evidence { font-size: 0.95rem; margin-top: 0.6rem; }
.index-evidence .delta { font-size: 0.8rem; opacity: 0.7; margin-left: 0.35rem; }
"""


def resolve_nested_sampling_summary(run):
    """Accept a run dir, summary.json path, or directory name under nested-sampling/."""
    raw = Path(run)
    name = raw.parent.name if raw.name == "summary.json" else raw.name
    candidates = []
    if raw.name == "summary.json":
        candidates.append(raw)
    candidates.append(raw / "summary.json")
    candidates.append(Path(NESTED_SAMPLING_DIR) / name / "summary.json")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        f"No summary.json for run {run!r} "
        f"(looked under {NESTED_SAMPLING_DIR}/{name}/)"
    )


def nested_sampling_run_paths(limit=None, run=None):
    """summary.json paths, newest first, optionally filtered to one run or newest N."""
    if run:
        return [resolve_nested_sampling_summary(run)]
    paths = sorted(
        glob.glob(os.path.join(NESTED_SAMPLING_DIR, "*", "summary.json")),
        key=nested_sampling_run_sort_key,
    )
    return paths[:limit] if limit is not None else paths


def run_page_name(run_name):
    """Filename for a run's own page - same sanitising the old RUN= output used."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", run_name) + ".html"


def page_status(out_dir, run_name):
    """Page state for a run under out_dir: missing, outdated or current."""
    path = os.path.join(out_dir, run_page_name(run_name))
    if not os.path.exists(path):
        return "missing"
    return "current" if page_report_version(path) == REPORT_VERSION else "outdated"


def run_log_evidence(run_dir, summary):
    """(log_z, log_z_err) from the summary, falling back to the chains .stats file."""
    if summary.get("log_z") is not None:
        err = summary.get("log_z_err")
        return float(summary["log_z"]), (float(err) if err is not None else None)
    stats_path, _ = find_chain_stats(run_dir)
    if stats_path:
        parsed = parse_log_evidence(stats_path)
        if parsed:
            return parsed
    return None


def render_index_entry(summary_path, status):
    run_dir = os.path.dirname(summary_path)
    run_name = os.path.basename(run_dir)
    with open(summary_path) as f:
        summary = json.load(f)

    polychord = summary.get("polychord", {})
    evaluations = summary.get("evaluations", [])
    succeeded = [ev for ev in evaluations if "error" not in ev]
    failed_count = len(evaluations) - len(succeeded)

    title_bits = [html.escape(str(summary.get("algorithm", "?")))]
    ts_label = format_run_id_timestamp(run_name)
    if ts_label:
        title_bits.append(f'<span class="ts">{html.escape(ts_label)}</span>')

    duration_html = ""
    duration_label = format_wall_duration(summary.get("total_wall_seconds"))
    if duration_label:
        duration_html = f'<span class="run-duration">{html.escape(duration_label)}</span>'

    badges = []
    if summary.get("merged_from"):
        badges.append('<span class="badge badge-ok">merged</span>')
    badges.append(f'<span class="badge">{html.escape(str(summary.get("vla_config", "?")))}</span>')
    badges.append(f'<span class="badge">nlive {html.escape(str(polychord.get("nlive", "?")))}</span>')
    if summary.get("metric"):
        badges.append(f'<span class="badge">{html.escape(str(summary["metric"]))}</span>')
    badges.append(f'<span class="badge badge-ok">{len(succeeded)} evals</span>')
    if failed_count:
        badges.append(f'<span class="badge badge-warn">{failed_count} failed</span>')
    if status == "outdated":
        badges.append('<span class="badge badge-warn">outdated page</span>')

    evidence = run_log_evidence(run_dir, summary)
    if evidence:
        log_z, log_z_err = evidence
        err_html = f'<span class="delta">± {log_z_err:.4g}</span>' if log_z_err is not None else ""
        evidence_html = f'<div class="index-evidence">log(Z) = <strong>{log_z:.4g}</strong>{err_html}</div>'
    else:
        evidence_html = '<div class="index-evidence empty">log(Z) unavailable</div>'

    body = f"""
      <div class="card-header-top">
        <h2>{" ".join(title_bits)}</h2>
        {duration_html}
      </div>
      <p class="index-run-name">{html.escape(run_name)}</p>
      <div class="badges">{"".join(badges)}</div>
      {evidence_html}
    """
    if status == "missing":
        return (
            f'<div class="card index-entry index-entry-missing">{body}'
            '<p class="empty">Page not generated yet - run <code>./ri report</code>.</p></div>'
        )
    stale_html = ""
    if status == "outdated":
        stale_html = (
            '<p class="empty">Page built by an older report version - run '
            "<code>./ri report --upgrade</code>.</p>"
        )
    return (
        f'<a class="card index-entry" href="{html.escape(run_page_name(run_name))}">'
        f"{body}{stale_html}</a>"
    )


def render_nested_sampling_index(status_for):
    paths = nested_sampling_run_paths()
    if not paths:
        return (
            '<p class="empty">No nested-sampling runs found under '
            "results/nested-sampling/*/summary.json yet.</p>"
        )
    return "".join(render_index_entry(p, status_for(p)) for p in paths)


def index_nav_html():
    return '<p class="nav"><a href="index.html">&larr; All runs</a></p>'


def write_html_doc(out_path, title, subtitle, body):
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="report-version" content="{REPORT_VERSION}">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="subtitle">{subtitle}</p>
{body}
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html_doc)
    print(f"wrote {out_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_path",
        nargs="?",
        default=None,
        help="Output directory for the report pages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Newest N runs (timestamp sort). Omit for all.",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="One run directory or name under nested-sampling/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild run pages that already exist.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help=(
            "Rebuild run pages written by an older report version, leaving "
            "up-to-date ones alone."
        ),
    )
    return parser.parse_args(argv)


def run_body_task(item):
    """Page body with LIKELIHOOD_SLOT standing in for the corner plot section."""
    summary_path = item[0]
    return index_nav_html() + render_nested_sampling_run(
        summary_path, likelihood_html=LIKELIHOOD_SLOT
    )


def likelihood_task(item):
    """URL of one run's corner plot, or None. Runs alongside its page body."""
    summary_path = item[0]
    with open(summary_path) as f:
        summary = json.load(f)
    space_names = [
        spec["name"] for spec in summary.get("parameter_space", []) if "name" in spec
    ]
    return render_likelihood_plot(os.path.dirname(summary_path), space_names)


def write_run_page(item, body):
    run_name = item[2]
    assert LIKELIHOOD_SLOT not in body, "likelihood slot was never filled in"
    write_html_doc(
        item[1],
        title=f"nested-sampling run: {run_name}",
        subtitle=(
            f"Generated from <code>results/nested-sampling/{html.escape(run_name)}/"
            "summary.json</code>."
        ),
        body=body,
    )


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_path or "/workspace/out/nested-sampling-report"
    limit = args.limit
    run = args.run
    if limit is not None and run:
        raise SystemExit("refuse: --limit and --run cannot be used together")
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be >= 1")
    os.makedirs(out_dir, exist_ok=True)
    global image_dir
    image_dir = os.path.join(out_dir, IMAGE_SUBDIR)

    # An explicit --run is a deliberate "rebuild this one" request.
    force = args.force or bool(run)
    skipped = 0
    outdated = 0
    drawing = False
    todo = []
    for summary_path in nested_sampling_run_paths(limit=limit, run=run):
        run_name = os.path.basename(os.path.dirname(summary_path))
        page_path = os.path.join(out_dir, run_page_name(run_name))
        status = page_status(out_dir, run_name)
        if status == "current" and not force:
            skipped += 1
            continue
        if status == "outdated" and not (force or args.upgrade):
            outdated += 1
            skipped += 1
            continue
        todo.append((summary_path, page_path, run_name))
        # A run with no page yet has no images either, so every worker would
        # otherwise import the drawing stack separately after the fork. Load it
        # once here instead and let them inherit it. Only the matplotlib half
        # now, though: the corner plot is the longest task and never touches
        # astropy or PIL, so those wait until the plots are already running
        # (below) rather than sitting on the parent's serial prologue.
        if status == "missing":
            drawing = True
            load_plot_libs()

    # Pages are independent, and within a page the corner plot and the eval
    # rasters are independent too, so every run contributes two concurrent
    # tasks. Two pools rather than one, so that astropy can be imported in
    # between: the corner plots - the build's critical path - are already
    # running by then, and the eval-raster workers forked after it inherit the
    # import instead of each repeating it while the plots want the CPU. The
    # image store is written with write-then-rename, so two workers racing on
    # the same PNG is harmless.
    written = len(todo)
    if todo:
        workers = min(len(todo), os.cpu_count() or 1)
        if drawing:
            load_chain_libs()
        with multiprocessing.Pool(workers) as plot_pool:
            plots = [plot_pool.apply_async(likelihood_task, (item,)) for item in todo]
            if drawing:
                load_render_libs()
            with multiprocessing.Pool(workers) as body_pool:
                bodies = [body_pool.apply_async(run_body_task, (item,)) for item in todo]
                for item, plot, body in zip(todo, plots, bodies):
                    write_run_page(
                        item,
                        body.get().replace(LIKELIHOOD_SLOT, likelihood_section(plot.get())),
                    )

    # The index is cheap and must reflect every run on disk, so always rebuild it.
    write_html_doc(
        os.path.join(out_dir, "index.html"),
        title="ri-reproducibility nested-sampling runs",
        subtitle=(
            "One page per run under <code>results/nested-sampling/</code> - "
            "regenerate with <code>./ri report</code> "
            "(up-to-date pages are skipped; <code>--upgrade</code> rebuilds "
            "the ones an older report version wrote, <code>--force</code> "
            f"rebuilds every page). Report version <code>{REPORT_VERSION}</code>."
        ),
        body=render_nested_sampling_index(
            lambda p: page_status(out_dir, os.path.basename(os.path.dirname(p)))
        ),
    )
    print(f"{written} run page(s) written, {skipped} skipped")
    if outdated:
        print(
            f"{outdated} page(s) built by an older report version - rerun with "
            "UPGRADE=1 to bring them up to the current design"
        )


def _self_check_log_evidence_parser():
    sample = """
Global evidence:
----------------

log(Z)       =   0.145917983191460E+001 +/-   0.309608121862379E-001
"""
    parsed = parse_log_evidence_from_text(sample)
    assert parsed == (1.4591798319146, 0.0309608121862379), parsed


def _self_check_profiling():
    # MPI run: shares are of the worker-time budget, null stages are dropped,
    # and the imaging rows are named after the run's algorithm.
    html_out = render_profiling({"algorithm": "r2d2", "profiling": {
        "mpi_procs": 8, "total_wall_seconds": 455.58,
        "stage_totals_seconds": {
            "simulate": 33.03, "convert": None, "image_container": 2354.9, "metrics": 1.44,
        },
        "stage_eval_counts": {
            "simulate_seconds": 44, "image_container_seconds": 44, "metrics_seconds": 44,
        },
        "accounted_worker_seconds": 2389.37, "polychord_overhead_seconds": None,
    }})
    # Minutes and hours where seconds stopped being readable, and a per-eval column.
    assert "r2d2 container (total)" in html_out, html_out
    assert "image container" not in html_out, html_out
    assert "39m 15s" in html_out and "53.5s" in html_out, html_out
    # A share that rounds to nothing still renders as escaped text, not as a tag.
    assert "&lt;0.1%" in html_out and "<0.1%" not in html_out, html_out
    assert "8 workers" in html_out and "1h 00m 45s" in html_out, html_out
    assert "convert" not in html_out, html_out
    # Every top-level stage plus the unaccounted remainder is charted, adds to
    # 100%, and is repeated once per worker lane.
    assert html_out.count('class="profile-bar"') == 8, html_out
    assert html_out.count('class="profile-seg"') == 4 * 8, html_out
    assert "unaccounted (PolyChord sampling + idle)" in html_out, html_out
    shares = [float(s) for s in re.findall(r"--seg-share: ([\d.]+)", html_out)]
    assert abs(sum(shares) / 8 - 1.0) < 1e-6, shares
    # The parallelisation arithmetic is spelled out and lands on the wall clock
    # the run header shows, and the table carries the wall-clock column that
    # turns each stage's worker-seconds into its cost in wall time.
    assert (
        "39m 49s accounted + 20m 55s unaccounted = 1h 00m 45s of worker-time ÷ 8 workers"
        " <strong>= 7m 36s end-to-end wall clock</strong>"
    ) in html_out, html_out
    assert "8 of them running side by side" in html_out, html_out
    assert ">wall clock</th>" in html_out, html_out
    assert "4m 54s" in html_out, html_out  # r2d2's 39m 15s of worker-time, in wall clock
    assert "end-to-end (accounted + unaccounted)" in html_out, html_out
    # Serial run: the budget is just the wall clock, so shares are of wall time.
    single = render_profiling({"algorithm": "wsclean", "profiling": {
        "mpi_procs": 1, "total_wall_seconds": 10.0,
        "stage_totals_seconds": {"simulate": 5.0},
        "stage_eval_counts": {"simulate_seconds": 2},
        "accounted_worker_seconds": 5.0, "polychord_overhead_seconds": 5.0,
    }})
    assert '<td class="num">50.0%</td>' in single, single
    assert single.count("--seg-share: 0.500000") == 2, single  # the one stage, and the remainder
    assert single.count('class="profile-bar"') == 1, single  # one worker, one lane
    assert "1 worker ·" in single, single
    # Serial: worker-time is the wall clock, so no redundant column and no division.
    assert ">wall clock</th>" not in single, single
    assert (
        "5.00s accounted + 5.00s unaccounted <strong>= 10.0s end-to-end wall clock</strong>"
    ) in single, single
    # No profiling block: nothing rendered.
    assert render_profiling({}) == ""
    # A profiling block with nothing in it must not divide by zero.
    empty = render_profiling({"profiling": {"mpi_procs": 1, "total_wall_seconds": 0.0}})
    assert 'class="profile-seg"' not in empty, empty


def _self_check_page_status():
    tmp_dir = tempfile.mkdtemp(prefix="ns-report-selfcheck-")
    assert page_status(tmp_dir, "run-a") == "missing"
    path = os.path.join(tmp_dir, run_page_name("run-a"))
    write_html_doc(path, "t", "s", "<p>b</p>")
    assert page_status(tmp_dir, "run-a") == "current"
    with open(path) as f:
        stale = f.read().replace(REPORT_VERSION, "deadbeef0000")
    with open(path, "w") as f:
        f.write(stale)
    assert page_status(tmp_dir, "run-a") == "outdated"
    # A page written before versioning existed carries no meta tag at all.
    with open(os.path.join(tmp_dir, run_page_name("run-b")), "w") as f:
        f.write("<!doctype html><html><head><title>old</title></head></html>")
    assert page_status(tmp_dir, "run-b") == "outdated"
    shutil.rmtree(tmp_dir)


def _self_check_cached_png():
    """A second request for the same key reuses the file instead of re-rendering."""
    global image_dir
    tmp_dir = tempfile.mkdtemp(prefix="ns-report-selfcheck-")
    image_dir = os.path.join(tmp_dir, IMAGE_SUBDIR)
    calls = []

    def render():
        calls.append(1)
        return b"png-bytes"

    def name_for(key):
        digest = hashlib.sha1(f"{IMAGE_RENDER_VERSION}|{key}".encode()).hexdigest()
        return f"{digest[:16]}.png"

    url = cached_png("k", render)
    assert url == f"{IMAGE_SUBDIR}/{name_for('k')}", url
    assert cached_png("k", render) == url
    assert len(calls) == 1, calls
    assert os.path.exists(os.path.join(tmp_dir, url)), url
    # A different key renders again, and a render that declines writes nothing.
    assert cached_png("k2", render) != url
    assert len(calls) == 2, calls
    assert cached_png("k3", lambda: None) is None
    assert sorted(os.listdir(image_dir)) == sorted(
        name_for(k) for k in ("k", "k2")
    ), os.listdir(image_dir)
    shutil.rmtree(tmp_dir)
    image_dir = None


def _self_check_render_array_png():
    load_render_libs()
    assert render_array_png(np.zeros((4,))) is None
    # All-zero data is the degenerate ZScale case; it must still produce a PNG.
    flat = Image.open(io.BytesIO(render_array_png(np.zeros((8, 8, 1)))))
    assert flat.size == (8, 8), flat.size
    # Data is stored at its own resolution, capped at the caller's pixel budget.
    big = Image.open(io.BytesIO(render_array_png(np.arange(10000.0).reshape(100, 100), figsize=(1, 1), dpi=50)))
    assert big.size == (50, 50), big.size


def _self_check_tick_housekeeping():
    """The corner plot's speedup rests on one side effect - that the memoized
    path still un-stales the shared view limits. Assert it directly."""
    load_plot_libs()
    if not dedupe_pandas_tick_housekeeping():
        return
    from pandas.plotting._matplotlib import tools

    fig, (top, _bottom) = plt.subplots(2, sharex=True)
    top.plot([0, 1], [0, 1])
    tools._remove_labels_from_axis(top.xaxis)
    assert not any(t.get_visible() for t in top.xaxis.get_majorticklabels())
    top._stale_viewlims["x"] = True
    tools._remove_labels_from_axis(top.xaxis)
    assert not top._stale_viewlims["x"], "memoized path lost the viewLim un-stale"
    plt.close(fig)


def _self_check_tick_memo():
    """The tick memo must hit while the axis is unchanged, miss once its view
    limits move, and keep un-staling the shared view limits either way."""
    load_plot_libs()
    if not memoize_matplotlib_tick_updates():
        return
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    first = ax.xaxis._update_ticks()
    ax._stale_viewlims["x"] = True
    assert ax.xaxis._update_ticks() is first, "memo missed on an unchanged axis"
    assert not ax._stale_viewlims["x"], "memo hit lost the viewLim un-stale"
    ax.set_xlim(0, 10)
    assert ax.xaxis._update_ticks() is not first, "memo hit after the view moved"
    plt.close(fig)


def _self_check_tight_bbox():
    """tight_bbox must reproduce what savefig(bbox_inches="tight") measures."""
    load_plot_libs()

    def draw():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1000])
        ax.set_xlabel("x label")
        fig.tight_layout()
        return fig

    fig = draw()
    assert figure_to_png_bytes(fig, bbox_inches=tight_bbox(fig)) == figure_to_png_bytes(
        draw(), bbox_inches="tight"
    ), "tight_bbox does not match bbox_inches='tight'"


def _self_check_labels_map_memo():
    """The labels-map memo must hit while the frame's columns are unchanged and
    miss once a new column swaps the Index out, so a stale mapping can't stick."""
    if not memoize_anesthetic_labels_map():
        return
    from anesthetic.labelled_pandas import LabelledDataFrame
    from pandas import MultiIndex

    columns = MultiIndex.from_tuples(
        [("a", "$a$"), ("b", "$b$")], names=["params", "labels"]
    )
    df = LabelledDataFrame([[0.0, 1.0]], columns=columns)
    first = df.get_labels_map(axis=1)
    assert list(first) == ["$a$", "$b$"]
    assert df.get_labels_map(axis=1) is first, "memo missed on an unchanged frame"
    df[("c", "$c$")] = 2.0
    second = df.get_labels_map(axis=1)
    assert second is not first, "memo hit after a column was added"
    assert list(second) == ["$a$", "$b$", "$c$"]


def _self_check_drop_labels_memo():
    """The drop-labels memo must reuse one copy for every spec that drops the
    same levels, keep specs that drop different ones apart, and miss as soon as
    a new column swaps the columns Index out - that last one is the guard that
    stops a stale copy of the frame reaching the plot."""
    if not memoize_anesthetic_drop_labels():
        return
    from anesthetic.labelled_pandas import LabelledDataFrame
    from pandas import MultiIndex

    columns = MultiIndex.from_tuples(
        [("a", "$a$"), ("b", "$b$")], names=["params", "labels"]
    )
    df = LabelledDataFrame([[0.0, 1.0]], columns=columns)
    stripped = df.drop_labels(1)
    assert list(stripped.columns) == ["a", "b"]
    assert df.drop_labels(1) is stripped, "memo missed on an unchanged frame"
    assert df.drop_labels([0, 1]) is stripped, "same dropped levels, same copy"
    # Nothing is labelled on the index, so this one only copies.
    assert df.drop_labels(0) is not stripped, "different levels shared a copy"
    df[("c", "$c$")] = 2.0
    third = df.drop_labels(1)
    assert third is not stripped, "memo hit after a column was added"
    assert list(third.columns) == ["a", "b", "c"]


def _self_check_run_page_name():
    assert run_page_name("wsclean-vlaa-20260826T010221Z") == "wsclean-vlaa-20260826T010221Z.html"
    # Anything that would escape the output directory is flattened.
    assert run_page_name("../etc/passwd") == ".._etc_passwd.html"


if __name__ == "__main__":
    if os.environ.get("GENERATE_REPORT_SELF_CHECK") == "1":
        _self_check_log_evidence_parser()
        _self_check_run_page_name()
        _self_check_cached_png()
        _self_check_render_array_png()
        _self_check_profiling()
        _self_check_page_status()
        _self_check_tick_housekeeping()
        _self_check_tick_memo()
        _self_check_tight_bbox()
        _self_check_labels_map_memo()
        _self_check_drop_labels_memo()
        print("generate_report self-check passed")
    else:
        main()
