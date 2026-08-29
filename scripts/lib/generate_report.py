"""
Builds an HTML report from the nested-sampling runs under
results/nested-sampling/*/summary.json - one page per run, an index, and
an images/ directory of the PNGs those pages reference.

Run via scripts/generate-report.sh, which wraps this in the r2d2 image so it
can reuse the imager's own astropy + matplotlib + anesthetic rather than
requiring a host Python environment - same approach as scripts/plot-fits.sh.
"""
import argparse
import gc
import glob
import hashlib
import html
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = "/workspace/repo"
NESTED_SAMPLING_DIR = os.path.join(REPO_ROOT, "results/nested-sampling")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nested_sampling"))

from common import (  # noqa: E402
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
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    # matplotlib.projections imports mpl_toolkits.mplot3d solely to register the
    # '3d' projection, and already wraps that import in a try/except that
    # degrades to "3d unavailable" when it fails. Nothing here draws in 3d - a
    # corner plot is a grid of 2d axes - so short-circuit it: a None in
    # sys.modules makes the import raise ImportError, which is exactly the case
    # matplotlib handles, and skips 17ms (6.7%) of the 258ms `import
    # matplotlib.pyplot` that sits on the build's serial prologue. The poison is
    # removed again immediately, so a later `import mpl_toolkits.mplot3d` still
    # works; only the registry entry stays gone, which turns an accidental
    # projection='3d' into a loud ValueError rather than a silent wrong plot.
    poisoned = "mpl_toolkits.mplot3d" not in sys.modules
    if poisoned:
        sys.modules["mpl_toolkits.mplot3d"] = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to import Axes3D")
            import matplotlib.pyplot as plt
    finally:
        if poisoned:
            sys.modules.pop("mpl_toolkits.mplot3d", None)


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
# Axes the in-progress handle_shared_axes scan has selected, innermost last.
_shared_axes_recording = []


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
        if _shared_axes_recording:
            _shared_axes_recording[-1].append(axis)
        if getattr(axis, "_report_labels_removed", False):
            axis.axes.viewLim  # noqa: B018 - un-stales the shared view limits
            return
        axis._report_labels_removed = True
        remove_labels(axis)

    tools._remove_labels_from_axis = once

    # With the labels themselves deduplicated, what is left of a repeat call is
    # the scan that decides which axes to hand to `once`: for every axis, and
    # for both of its axes, walk the whole shared-axis group comparing
    # positions. That answer is fixed for as long as the figure's axes, their
    # visibility and the grid shape are, so record the axes the first scan
    # selects and replay just their viewLim touch afterwards.
    shared = getattr(tools, "handle_shared_axes", None)
    if shared is None:
        return True

    def deduped(axarr, nplots, naxes, nrows, ncols, sharex, sharey):
        axarr = list(axarr)
        if not axarr:
            return shared(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
        fig = axarr[0].get_figure()
        key = (
            tuple(id(ax) for ax in axarr),
            tuple(ax.get_visible() for ax in axarr),
            nplots,
            naxes,
            nrows,
            ncols,
            sharex,
            sharey,
        )
        cached = getattr(fig, "_report_shared_axes_memo", None)
        if cached is not None and cached[0] == key:
            for axis in cached[1]:
                axis.axes.viewLim  # noqa: B018 - the same un-stale as above
            return None
        selected = []
        _shared_axes_recording.append(selected)
        try:
            shared(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
        finally:
            _shared_axes_recording.pop()
        # The entry keeps axarr alive, so no id() in the key can be recycled
        # onto a different axes while the memo is live.
        fig._report_shared_axes_memo = (key, selected, axarr)
        return None

    # pandas' plotting core does `from .tools import handle_shared_axes`, so the
    # name has to be rebound wherever it was bound, not just on tools.
    for module in list(sys.modules.values()):
        if getattr(module, "handle_shared_axes", None) is shared:
            module.handle_shared_axes = deduped
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


# Laying a Text out - splitting it into lines, measuring each against the font,
# rotating the box - is a pure function of the string and its font/alignment
# properties, and matplotlib re-does it from scratch on every call. One corner
# plot asks for it 441 times for 70 distinct texts, because the three passes
# over the figure (tight_layout, the tight-bbox measurement, the render) each
# re-measure every tick label and axis title. The result is position-free -
# get_window_extent translates it afterwards - so one global cache serves every
# Text with the same properties. Texts with wrapping on are left alone: their
# layout also depends on the figure width, which is not in the key.
# Best-effort: if matplotlib renames the private method, the plot is just slower.
_text_layout_memoized = False


def memoize_matplotlib_text_layout():
    global _text_layout_memoized
    if _text_layout_memoized:
        return True
    _text_layout_memoized = True
    try:
        from matplotlib.text import Text

        get_layout = Text._get_layout
    except (ImportError, AttributeError):
        return False

    cache = {}

    def memoized(self, renderer):
        if self._wrap:
            return get_layout(self, renderer)
        key = (
            self._text,
            # Any two renderers of the same class at the same dpi measure
            # text identically, so the class is all a backend swap needs.
            type(renderer),
            self.get_figure(root=True).dpi,
            # FontProperties compares by hash, so this is its own equality.
            hash(self._fontproperties),
            self._usetex,
            self._parse_math,
            self._linespacing,
            # Reads the transform when transform_rotates_text is on.
            self.get_rotation(),
            self.get_rotation_mode(),
            self._horizontalalignment,
            self._verticalalignment,
            self._multialignment,
        )
        layout = cache.get(key)
        if layout is None:
            if len(cache) > 512:
                cache.clear()
            layout = cache[key] = get_layout(self, renderer)
        return layout

    Text._get_layout = memoized
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


# Even with both halves cached, `df[key]` on a labelled frame still *runs* all
# four label-stripped lookups and throws three of them away: anesthetic's `ac`
# evaluates every candidate, sorts them by dimensionality (fewest first, then
# most index levels, ties going to the earliest candidate) and returns the
# first. For the case the corner plot hits ~65 times per plot - a plain string
# column name on a frame whose *columns* carry the labels level - the winner is
# decided in advance, so run only that one.
#
# The winner is candidate `1` (drop the labels off the columns) because, with
# labelled columns, `df["x"]` on the two candidates that keep them (`0` and
# `None`) indexes a >=2-level MultiIndex and so yields a 2-D frame, which loses
# to any 1-D result outright; and of the two that strip them (`1` and `[0, 1]`)
# candidate `1` leaves the *index* untouched, so it never has fewer index
# levels than candidate `[0, 1]` and it comes first on a tie. A DataFrame lookup
# cannot return a 0-D result, so nothing can undercut a 1-D one. Any candidate
# that raises is dropped by `ac` and cannot win either.
#
# Each guard is a premise of that argument rather than a safety net: a string
# key is what forces candidates `0` and `None` to be 2-D, labelled columns are
# what make the labels a level to strip in the first place, and a Series is a
# two-candidate search over scalar results - a different comparison entirely.
# Anything outside the guards, any candidate that raises, and any result that
# turns out not to be 1-D all fall back to anesthetic's own search, so a shape
# the argument does not cover is slow rather than wrong.
# Best-effort: if anesthetic moves the private class, the plot is just slower.
_labelled_column_shortcut = False


def shortcut_anesthetic_labelled_column():
    global _labelled_column_shortcut
    if _labelled_column_shortcut:
        return True
    _labelled_column_shortcut = True
    try:
        from pandas import Series
        from pandas.errors import IndexingError

        from anesthetic.labelled_pandas import LabelledDataFrame, _LabelledObject

        getitem = _LabelledObject.__getitem__
    except (ImportError, AttributeError):
        return False

    def shortcut(self, key):
        if (
            type(key) is str
            and isinstance(self, LabelledDataFrame)
            and self.islabelled(1)
        ):
            try:
                column = super(_LabelledObject, self.drop_labels(1)).__getitem__(key)
            except (KeyError, ValueError, TypeError, IndexingError):
                return getitem(self, key)
            if getattr(column, "ndim", None) == 1:
                # The rename `ac` applies to whichever candidate it returns.
                labels = self.get_labels_map(1)
                if isinstance(labels, Series) and column.name in labels.index:
                    column.name = labels.loc[column.name]
                return column
        return getitem(self, key)

    _LabelledObject.__getitem__ = shortcut
    return True


# matplotlib looks up an Axes' axis objects through the _axis_map property,
# which rebuilds a dict - two f-strings and two getattrs - on every read. One
# corner plot reads it ~4500 times and the answer only ever changes where
# _init_axis() installs fresh XAxis/YAxis objects, so keep the dict on the
# instance and validate it by identity against the two attributes it maps: a
# rebuilt axis misses and the map is built again. Only the ordinary 2D case is
# cached; a projection with a different _axis_names falls through to the
# original property.
# Best-effort: if matplotlib drops the property, the plot is just slower.
_axis_map_cached = False


def cache_matplotlib_axis_map():
    global _axis_map_cached
    if _axis_map_cached:
        return True
    _axis_map_cached = True
    try:
        from matplotlib.axes._base import _AxesBase

        axis_map = _AxesBase._axis_map.fget
    except (ImportError, AttributeError):
        return False

    def cached(self):
        if self._axis_names == ("x", "y"):
            x_axis = self.xaxis
            y_axis = self.yaxis
            hit = self.__dict__.get("_report_axis_map")
            if hit is not None and hit[0] is x_axis and hit[1] is y_axis:
                return hit[2]
            built = {"x": x_axis, "y": y_axis}
            self.__dict__["_report_axis_map"] = (x_axis, y_axis, built)
            return built
        return axis_map(self)

    _AxesBase._axis_map = property(cached)
    return True


# Every matplotlib artist constructor and every .set() call routes its kwargs
# through cbook.normalize_kwargs, which flattens the artist class' alias map
# ({'linewidth': ['lw'], ...}) into an alias -> canonical dict from scratch each
# time. The alias map is a class attribute, so that flattening is the same
# answer for the life of the process: memoise it per class. One corner plot
# makes ~3400 of these calls. Callers that pass a plain dict (or None) keep the
# original path.
# Best-effort: if matplotlib changes the helper's shape, the plot is slower.
_alias_maps_memoized = False


def memoize_matplotlib_alias_maps():
    global _alias_maps_memoized
    if _alias_maps_memoized:
        return True
    _alias_maps_memoized = True
    try:
        from matplotlib import cbook
        from matplotlib.artist import Artist

        normalize = cbook.normalize_kwargs
    except (ImportError, AttributeError):
        return False

    cache = {}

    def memoized(kw, alias_mapping=None):
        if kw is None:
            return {}
        if isinstance(alias_mapping, type) and issubclass(alias_mapping, Artist):
            cls = alias_mapping
        elif isinstance(alias_mapping, Artist):
            cls = type(alias_mapping)
        else:
            return normalize(kw, alias_mapping)
        to_canonical = cache.get(cls)
        if to_canonical is None:
            to_canonical = cache[cls] = {
                alias: canonical
                for canonical, aliases in getattr(cls, "_alias_map", {}).items()
                for alias in aliases
            }
        if not to_canonical:
            return dict(kw)
        seen = {}
        ret = {}
        for key, value in kw.items():
            canonical = to_canonical.get(key, key)
            if canonical in seen:
                raise TypeError(
                    f"Got both {seen[canonical]!r} and {key!r}, which are "
                    "aliases of one another"
                )
            seen[canonical] = key
            ret[canonical] = value
        return ret

    # matplotlib and pandas both do `from ... import normalize_kwargs`, so the
    # defining module is not where the hot call sites look it up.
    for module in list(sys.modules.values()):
        if getattr(module, "normalize_kwargs", None) is normalize:
            module.normalize_kwargs = memoized
    return True


# Every read of an axis' viewLim asks matplotlib whether any axis sharing a
# limit with it still needs autoscaling, and that question is answered by
# walking the whole share group - twice, once per axis name - through a WeakSet.
# One corner plot asks it ~3200 times over 20 shared axes, and the answer is
# almost always "nothing is stale". Staleness is only ever *created* in one
# place, _request_autoscale_view, so count calls to it: an axis whose group was
# found settled at epoch N is still settled while the epoch reads N, and the
# scan can be skipped outright. The epoch is re-read after the wrapped call so
# an autoscale that re-stales the group on its way out is not recorded as
# settled. Nothing else in matplotlib or mpl_toolkits writes _stale_viewlims to
# True, so the counter sees every transition.
# Best-effort: if matplotlib renames either private method, the plot is slower.
_viewlim_scan_skipped = False


def skip_settled_matplotlib_viewlims():
    global _viewlim_scan_skipped
    if _viewlim_scan_skipped:
        return True
    _viewlim_scan_skipped = True
    try:
        from matplotlib.axes._base import _AxesBase

        request_autoscale = _AxesBase._request_autoscale_view
        unstale = _AxesBase._unstale_viewLim
    except (ImportError, AttributeError):
        return False

    epoch = [0]

    def counted(self, *args, **kwargs):
        epoch[0] += 1
        return request_autoscale(self, *args, **kwargs)

    def skipped(self):
        started = epoch[0]
        if getattr(self, "_report_viewlim_epoch", None) == started:
            return
        unstale(self)
        if epoch[0] == started:
            self._report_viewlim_epoch = started

    _AxesBase._request_autoscale_view = counted
    _AxesBase._unstale_viewLim = skipped
    return True


# anesthetic gives each panel its own limit-linking behaviour by defining a
# fresh Axes subclass *inside* the per-axis helper and rebinding __class__ to
# it, so a 5x5 corner plot builds 15 one-instance classes. Each one costs
# matplotlib's Artist.__init_subclass__, which regenerates set()'s signature and
# docstring by parsing the docstring of all ~265 setters - and leaves every
# panel with a class of its own, so no type-level cache in matplotlib or CPython
# is ever shared between them. The class bodies close over nothing but their
# base, so one class per (helper, base type) is enough: let the first panel
# build it as usual and rebind the rest onto the same class.
# Best-effort: if anesthetic moves the helpers, the plot is just slower.
_axes_subclasses_shared = False


def share_anesthetic_axes_subclasses():
    global _axes_subclasses_shared
    if _axes_subclasses_shared:
        return True
    _axes_subclasses_shared = True
    try:
        from anesthetic.plot import AxesDataFrame

        helpers = {
            name: AxesDataFrame.__dict__[name].__func__
            for name in ("_make_diagonal", "_make_offdiagonal")
        }
    except (ImportError, AttributeError, KeyError):
        return False

    cache = {}

    def share(name, make):
        def shared(ax):
            key = (name, type(ax))
            subclass = cache.get(key)
            if subclass is None:
                make(ax)
                cache[key] = type(ax)
            else:
                ax.__class__ = subclass

        return shared

    for name, make in helpers.items():
        setattr(AxesDataFrame, name, staticmethod(share(name, make)))
    return True


def _render_likelihood_png(run_dir, param_names):
    load_plot_libs()
    try:
        from anesthetic_io import load_nested_samples, weight_by_likelihood
    except ImportError:
        return None

    dedupe_pandas_tick_housekeeping()
    memoize_matplotlib_tick_updates()
    memoize_matplotlib_text_layout()
    memoize_anesthetic_labels_map()
    memoize_anesthetic_drop_labels()
    shortcut_anesthetic_labelled_column()
    skip_settled_matplotlib_viewlims()
    cache_matplotlib_axis_map()
    memoize_matplotlib_alias_maps()
    share_anesthetic_axes_subclasses()

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


# Categorical slot per top-level stage, assigned by identity and never by size,
# so a stage keeps its colour whether or not the other stages are present.
PROFILING_STAGE_COLOURS = {
    "simulate": "var(--series-1)",
    "convert": "var(--series-2)",
    "image_container": "var(--series-3)",
    "metrics": "var(--series-4)",
    "polychord": "var(--series-5)",
    "unaccounted": "var(--series-5)",
    # Grey, and deliberately not a sixth categorical hue: these two are the run
    # not imaging. The eye should read the coloured slices as the science and
    # the grey as everything spent getting between them.
    "harness": "color-mix(in srgb, CanvasText 45%, transparent)",
    "idle": "color-mix(in srgb, CanvasText 18%, transparent)",
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
    # Not the rank count: PolyChord's rank 0 administrates and never evaluates a
    # likelihood, so it has no worker-seconds to spend. See worker_procs().
    workers = breakdown["worker_procs"]
    budget = breakdown["worker_seconds_budget"]
    evals = breakdown["evals"]

    # Worker-seconds and wall-clock seconds only differ once the run is
    # parallel, so a serial run keeps the narrower table.
    show_wall = mpi_procs != 1

    def row(label, seconds, share, per_eval=None, evals_count="", indent=False, emphasis=False):
        # A missing stage total leaves the cell blank rather than dropping it,
        # which would shift every later cell into the wrong column.
        wall = [format_duration(seconds / workers) if seconds is not None else ""] if show_wall else []
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
    body += row(
        breakdown["subtotal_label"],
        breakdown["subtotal_seconds"],
        breakdown["subtotal_share"],
        breakdown["subtotal_per_eval_seconds"],
        emphasis=True,
    )
    # Below the sum: the time no evaluation was running in. PolyChord is
    # measured, idle is what is left over once it has been taken out.
    for remainder in breakdown["remainder_rows"]:
        body += row(remainder["label"], remainder["seconds"], remainder["share"])
    # The row the whole table exists to land on: divided across the workers, the
    # rows above come to the end-to-end wall clock on the run header.
    wall_seconds = (budget / workers) if budget else 0.0
    body += row(
        breakdown["total_label"],
        budget,
        1.0 if budget else None,
        emphasis=True,
    )

    segments = [
        (r["label"], r["seconds"], r["share"], PROFILING_STAGE_COLOURS.get(r["key"], "var(--series-1)"))
        for r in [*breakdown["rows"], *breakdown["remainder_rows"]]
        if not r.get("is_sub") and r["share"]
    ]

    heading = " · ".join([
        f"{workers} worker{'s' if workers != 1 else ''}"
        + (f" + administrator" if workers != mpi_procs else ""),
        f"wall clock {format_duration(breakdown['total_wall_seconds'])}",
        f"worker-time {format_duration(budget)} ({workers} × wall clock)",
        f"{evals} evaluations",
    ])
    # Spelled out left to right so the arithmetic behind the run header's wall
    # clock is readable without doing any of it yourself; the term it lands on
    # is the one the header shows, so that term carries the emphasis.
    terms = list(breakdown["equation_terms"])
    if mpi_procs != 1:
        terms.append(f"= {format_duration(budget)} of worker-time")
        terms.append(f"÷ {workers} workers")
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
      {render_profiling_lanes(segments, workers, wall_seconds)}
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
      <p class="purpose">{html.escape(breakdown["note"])}</p>
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


def format_param_range(spec):
    """"min-max" for a `parameter_space` entry, or "" if it has no range (fixed)."""
    lo, hi = spec.get("min"), spec.get("max")
    if lo is None or hi is None:
        return ""
    return f"{fmt_value(lo)}-{fmt_value(hi)}"


def render_param_space_badges(parameter_space):
    """One small pill per searched parameter's range, for the index card - the
    same "row of tabs" treatment as the algorithm/nlive/evals badges above it,
    so the parameter space is visible without opening the run page."""
    badges = []
    for spec in parameter_space:
        name = spec.get("name")
        if not name:
            continue
        range_label = format_param_range(spec)
        if not range_label:
            continue
        badges.append(
            f'<span class="badge badge-param">{html.escape(str(name))} {html.escape(range_label)}</span>'
        )
    if not badges:
        return ""
    return f'<div class="badges param-badges">{"".join(badges)}</div>'


def render_parameter_space_section(parameter_space):
    """The searched parameters and the range each was drawn from, for the report
    page - the per-run record of what was actually varied, distinct from any
    fixed hyperparameters shown elsewhere."""
    if not parameter_space:
        return ""
    rows = []
    for spec in parameter_space:
        name = spec.get("name")
        if not name:
            continue
        range_label = format_param_range(spec) or "-"
        kind = spec.get("kind", "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(range_label)}</td>"
            f"<td>{html.escape(str(kind))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return f"""
    <details>
      <summary>Parameter space</summary>
      <table class="kv param-space-table">
        <thead><tr><th>parameter</th><th>range</th><th>kind</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </details>
    """


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
        + "<th>objective</th></tr>"
    )
    eval_rows = []
    for ev in evaluations:
        params = ev.get("params", {})
        metrics = ev.get("metrics", {})
        eval_rows.append(
            "<tr>"
            f"<td>{html.escape(str(ev.get('eval_id', '?')))}</td>"
            + "".join(f"<td>{fmt_value(params.get(name))}</td>" for name in param_names)
            + "".join(f"<td>{fmt_value(metrics.get(key))}</td>" for key in metric_keys)
            + f"<td>{fmt_value(ev.get('objective'))}</td>"
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
        images_link_html = (
            f'<p class="nav nav-images"><a href="{html.escape(run_images_page_name(run_name))}">'
            f"View {len(eval_rows)} evaluation images &rarr;</a></p>"
        )
        evaluations_html = f"""
        <section>
          <h3>Evaluations</h3>
          {glance_summary_html}
          {images_link_html}
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
      {likelihood_html}
      {evaluations_html}
      {render_profiling(summary)}
      {render_parameter_space_section(parameter_space)}
      {fixed_html}
      <p class="manifest-name">{html.escape(rel_summary)}</p>
    </article>
    """


def render_run_images_page(summary_path):
    """Body of one run's images page: every evaluation raster, and nothing else.

    Kept off the run page so the details load without decoding one PNG per
    evaluation; the run page links here.
    """
    run_dir = os.path.dirname(summary_path)
    run_name = os.path.basename(run_dir)
    with open(summary_path) as f:
        summary = json.load(f)
    run_dirs = [run_dir] + merged_source_run_dirs(summary)

    evaluations = [ev for ev in summary.get("evaluations", []) if "error" not in ev]
    evaluations.sort(key=lambda item: (-float(item.get("objective", 0)), item.get("eval_id", 0)))
    images_html = render_eval_images(
        evaluations,
        summary.get("metric", ""),
        run_dirs,
        summary.get("parameter_space", []),
    )
    nav = (
        f'<p class="nav"><a href="{html.escape(run_page_name(run_name))}">&larr; Run details</a>'
        ' &middot; <a href="index.html">All runs</a></p>'
    )
    body = images_html or '<p class="empty">No evaluation images for this run.</p>'
    return f'{nav}<article class="card">{body}</article>'


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
.param-badges { margin-top: 0.3rem; }
.badge-param {
  background: transparent;
  border: 1px dashed color-mix(in srgb, CanvasText 30%, transparent);
  font-family: ui-monospace, monospace;
  opacity: 0.85;
}
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
.unfinished { border: 1px solid color-mix(in srgb, CanvasText 25%, transparent); border-radius: 8px; padding: 0.75rem 1rem; margin: 0 0 1rem; }
.unfinished h2 { font-size: 1rem; margin: 0 0 0.5rem; }
.unfinished ul { margin: 0.5rem 0 0; padding-left: 1.25rem; }
.unfinished li { margin: 0.25rem 0; }
.unfinished code { font-size: 0.85rem; }
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
.likelihood-plot { margin: 0.5rem 0; }
.likelihood-plot img { max-width: 100%; height: auto; border-radius: 6px; }
.nav { font-size: 0.9rem; margin: 0 0 1rem; }
.nav a { color: inherit; opacity: 0.7; text-decoration: none; }
.nav a:hover { opacity: 1; text-decoration: underline; }
.nav-images { margin: 0.75rem 0 0; }
.index-entry { display: block; color: inherit; text-decoration: none; }
/* [hidden] alone loses to the rule above: both are one-class/one-attribute
   specificity, and this author sheet already beat the UA sheet's own
   `[hidden] { display: none }` for the same reason. Pin it back down. */
.index-entry[hidden] { display: none; }
a.index-entry:hover { border-color: color-mix(in srgb, CanvasText 45%, transparent); }
.index-entry h2 { font-size: 1.05rem; }
.index-entry-missing { opacity: 0.6; }
.index-run-name { font-size: 0.8rem; opacity: 0.6; margin: 0 0 0.5rem; word-break: break-all; }
.index-evidence { font-size: 0.95rem; margin-top: 0.6rem; }
.index-evidence .delta { font-size: 0.8rem; opacity: 0.7; margin-left: 0.35rem; }
.ns-index-toolbar {
  display: flex; flex-wrap: wrap; align-items: center;
  gap: 0.5rem 1.25rem; margin: 0 0 1.25rem;
  font-size: 0.85rem;
}
.ns-index-toolbar label { display: flex; align-items: center; gap: 0.4rem; }
.ns-index-toolbar select { font: inherit; }
.ns-index-count { opacity: 0.65; margin-left: auto; }
.compare-select {
  display: flex; align-items: center; gap: 0.3rem;
  font-size: 0.8rem; opacity: 0.75; white-space: nowrap;
  flex-shrink: 0; cursor: pointer;
}
.compare-select:hover { opacity: 1; }
.ns-compare-panel {
  border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
  border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;
}
.ns-compare-header { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.ns-compare-header h3 { margin: 0; font-size: 1rem; }
.ns-compare-table-wrap { overflow-x: auto; margin-top: 0.75rem; }
table.ns-compare-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
table.ns-compare-table th, table.ns-compare-table td {
  padding: 0.3rem 0.6rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 10%, transparent);
  text-align: left; white-space: nowrap;
}
table.ns-compare-table thead th { opacity: 0.7; font-weight: 600; }
table.ns-compare-table tbody th { opacity: 0.65; font-weight: normal; }
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


def summary_is_complete(summary_path):
    """Whether `summary_path` is a whole summary.json rather than half of one.

    A run killed while writing its summary - the OOM killer, ENOSPC - leaves a
    truncated file, and one of those used to take the entire report down:
    every page and the index are built in one pass, so json.load raising on one
    run left every other run with no report at all. Runs written since
    write_json_atomic() cannot produce one; this is for the ones already on
    disk, and for a filesystem that fills up between the rename and the read.

    Tested by the last byte rather than by parsing, because this runs over
    every run in the results directory and a finished R2D2 search's summary
    carries all of its evaluations - tens of MB to parse for a question the
    tail answers in one seek. json.dumps ends every complete write with `}`.
    """
    try:
        with open(summary_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64))
            return f.read().decode("utf-8", "replace").rstrip().endswith("}")
    except OSError:
        return False


def nested_sampling_run_paths(limit=None, run=None):
    """summary.json paths, newest first, optionally filtered to one run or newest N."""
    if run:
        return [resolve_nested_sampling_summary(run)]
    paths = sorted(
        (p for p in glob.glob(os.path.join(NESTED_SAMPLING_DIR, "*", "summary.json"))
         if summary_is_complete(p)),
        key=nested_sampling_run_sort_key,
    )
    return paths[:limit] if limit is not None else paths


def run_images_page_name(run_name):
    return run_page_name(run_name)[: -len(".html")] + "-images.html"


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


# Display labels for known algorithm tokens; anything else falls back to the
# raw summary value so a future algorithm still gets a working filter option.
ALGORITHM_LABELS = {"r2d2": "R2D2", "wsclean": "WSClean"}


def render_index_entry(summary_path, status):
    """Returns (card_html, algorithm_token, parameter_space) - the token feeds the
    index toolbar's algorithm filter options ("" when the run has none worth
    filtering on), and parameter_space feeds its parameter/range filter and the
    run-comparison panel."""
    run_dir = os.path.dirname(summary_path)
    run_name = os.path.basename(run_dir)
    with open(summary_path) as f:
        summary = json.load(f)

    polychord = summary.get("polychord", {})
    evaluations = summary.get("evaluations", [])
    succeeded = [ev for ev in evaluations if "error" not in ev]
    failed_count = len(evaluations) - len(succeeded)
    parameter_space = summary.get("parameter_space", [])

    algorithm_token = str(summary.get("algorithm") or "").strip().lower()
    if algorithm_token in ("", "?"):
        algorithm_token = ""
    is_merged = bool(summary.get("merged_from"))
    param_names = [spec["name"] for spec in parameter_space if spec.get("name")]
    param_ranges = {
        spec["name"]: format_param_range(spec)
        for spec in parameter_space
        if spec.get("name") and format_param_range(spec)
    }

    title_bits = [html.escape(str(summary.get("algorithm", "?")))]
    ts_label = format_run_id_timestamp(run_name)
    if ts_label:
        title_bits.append(f'<span class="ts">{html.escape(ts_label)}</span>')

    duration_html = ""
    duration_label = format_wall_duration(summary.get("total_wall_seconds"))
    if duration_label:
        duration_html = f'<span class="run-duration">{html.escape(duration_label)}</span>'

    badges = []
    if is_merged:
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
        logz_attr_value = f"{log_z:.4g}"
    else:
        evidence_html = '<div class="index-evidence empty">log(Z) unavailable</div>'
        logz_attr_value = ""

    card_attrs = (
        f' data-algorithm="{html.escape(algorithm_token)}"'
        f' data-merged="{"1" if is_merged else "0"}"'
        f' data-evals="{len(succeeded)}"'
        f' data-run-name="{html.escape(run_name)}"'
        f' data-vla-config="{html.escape(str(summary.get("vla_config", "")))}"'
        f' data-nlive="{html.escape(str(polychord.get("nlive", "")))}"'
        f' data-metric="{html.escape(str(summary.get("metric", "")))}"'
        f' data-logz="{logz_attr_value}"'
        f' data-param-names="{html.escape(",".join(param_names))}"'
        f' data-param-ranges="{html.escape(json.dumps(param_ranges))}"'
    )

    compare_checkbox_html = (
        '<label class="compare-select" title="Select to compare runs">'
        '<input type="checkbox" class="compare-checkbox">'
        "Compare</label>"
    )

    body = f"""
      <div class="card-header-top">
        <h2>{" ".join(title_bits)}</h2>
        {compare_checkbox_html}
        {duration_html}
      </div>
      <p class="index-run-name">{html.escape(run_name)}</p>
      <div class="badges">{"".join(badges)}</div>
      {render_param_space_badges(parameter_space)}
      {evidence_html}
    """
    if status == "missing":
        return (
            f'<div class="card index-entry index-entry-missing"{card_attrs}>{body}'
            '<p class="empty">Page not generated yet - run <code>./ri report</code>.</p></div>'
        ), algorithm_token, parameter_space
    stale_html = ""
    if status == "outdated":
        stale_html = (
            '<p class="empty">Page built by an older report version - run '
            "<code>./ri report --upgrade</code>.</p>"
        )
    return (
        f'<a class="card index-entry" href="{html.escape(run_page_name(run_name))}"{card_attrs}>'
        f"{body}{stale_html}</a>"
    ), algorithm_token, parameter_space


# Wired up client-side in INDEX_SCRIPT: the toolbar only ever filters/reorders
# cards already in the page, so it needs no server round-trip and works the
# same off a plain `file://` open as it does served.
def render_index_toolbar(algorithm_tokens, param_ranges_by_name):
    algo_options = "".join(
        f'<option value="{html.escape(token)}">{html.escape(ALGORITHM_LABELS.get(token, token.title()))}</option>'
        for token in algorithm_tokens
    )
    param_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in param_ranges_by_name
    )
    # Range options depend on the chosen parameter, so only the map goes to the
    # page - INDEX_SCRIPT fills #ns-filter-param-range's options on change.
    # `<script>` content is HTML's "raw text" - entities are never decoded in
    # it, so html.escape would hand JSON.parse a literal "&quot;". Neutralise
    # "<" instead (the only character that can prematurely end the element via
    # a "</script" sequence); < is a legal JSON escape, so this still
    # parses to the exact same object.
    param_ranges_json = json.dumps(param_ranges_by_name).replace("<", "\\u003c")
    return f"""
    <div class="ns-index-toolbar">
      <label>Algorithm
        <select id="ns-filter-algorithm"><option value="">All algorithms</option>{algo_options}</select>
      </label>
      <label>Merged
        <select id="ns-filter-merged">
          <option value="">All runs</option>
          <option value="1">Merged only</option>
          <option value="0">Unmerged only</option>
        </select>
      </label>
      <label>Parameter
        <select id="ns-filter-param"><option value="">Any parameter</option>{param_options}</select>
      </label>
      <label>Range
        <select id="ns-filter-param-range" disabled><option value="">Any range</option></select>
      </label>
      <label>Sort by
        <select id="ns-sort">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="evals-desc">Most evals</option>
          <option value="evals-asc">Fewest evals</option>
        </select>
      </label>
      <span class="ns-index-count" id="ns-index-count"></span>
    </div>
    <script type="application/json" id="ns-param-ranges">{param_ranges_json}</script>
    <div class="ns-compare-panel" id="ns-compare-panel" hidden></div>
    """


# Filters by hiding cards (native `hidden` attribute, no CSS of its own needed)
# and sorts by re-appending them in the wanted order - appendChild on a node
# already in the list moves it rather than duplicating it. Array#sort is
# specified stable, so the two eval sorts keep the newest-first document order
# as their tie-break for free.
INDEX_SCRIPT = """
<script>
(function () {
  var list = document.getElementById("ns-index-list");
  var algoSel = document.getElementById("ns-filter-algorithm");
  var mergedSel = document.getElementById("ns-filter-merged");
  var paramSel = document.getElementById("ns-filter-param");
  var rangeSel = document.getElementById("ns-filter-param-range");
  var sortSel = document.getElementById("ns-sort");
  var countEl = document.getElementById("ns-index-count");
  var rangesDataEl = document.getElementById("ns-param-ranges");
  var comparePanel = document.getElementById("ns-compare-panel");
  if (!list || !algoSel || !mergedSel || !paramSel || !rangeSel || !sortSel) return;
  var items = Array.prototype.slice.call(list.children);
  var paramRanges = {};
  try {
    paramRanges = JSON.parse((rangesDataEl && rangesDataEl.textContent) || "{}");
  } catch (e) { paramRanges = {}; }

  function refreshRangeOptions() {
    var name = paramSel.value;
    var ranges = paramRanges[name] || [];
    rangeSel.disabled = !name;
    rangeSel.innerHTML = '<option value="">Any range</option>' + ranges.map(function (r) {
      return '<option value="' + r + '">' + r + "</option>";
    }).join("");
    rangeSel.value = "";
  }

  function apply() {
    var sorted = items.slice();
    if (sortSel.value === "oldest") {
      sorted.reverse();
    } else if (sortSel.value === "evals-desc" || sortSel.value === "evals-asc") {
      var dir = sortSel.value === "evals-desc" ? -1 : 1;
      sorted.sort(function (a, b) {
        return dir * (Number(a.dataset.evals || 0) - Number(b.dataset.evals || 0));
      });
    }
    sorted.forEach(function (el) { list.appendChild(el); });

    var algo = algoSel.value;
    var merged = mergedSel.value;
    var param = paramSel.value;
    var range = rangeSel.value;
    var visible = 0;
    items.forEach(function (el) {
      var names = (el.dataset.paramNames || "").split(",");
      var show = (!algo || el.dataset.algorithm === algo)
        && (!merged || el.dataset.merged === merged)
        && (!param || names.indexOf(param) !== -1)
        && (!range || (paramRangesFor(el)[param] === range));
      el.hidden = !show;
      if (show) visible += 1;
    });
    if (countEl) {
      countEl.textContent = visible + " of " + items.length + " run" + (items.length === 1 ? "" : "s");
    }
  }

  function paramRangesFor(el) {
    try {
      return JSON.parse(el.dataset.paramRanges || "{}");
    } catch (e) { return {}; }
  }

  // Compare: checkboxes live inside each `.index-entry` <a>, so a click has to
  // stop there or it navigates to the run page instead of toggling the box.
  function selectedCards() {
    return items.filter(function (el) {
      var box = el.querySelector(".compare-checkbox");
      return box && box.checked;
    });
  }

  function renderCompare() {
    var selected = selectedCards();
    if (!comparePanel) return;
    if (selected.length === 0) {
      comparePanel.hidden = true;
      comparePanel.innerHTML = "";
      return;
    }
    var paramNames = [];
    selected.forEach(function (el) {
      (el.dataset.paramNames || "").split(",").forEach(function (n) {
        if (n && paramNames.indexOf(n) === -1) paramNames.push(n);
      });
    });
    paramNames.sort();

    var headerCells = selected.map(function (el) {
      return "<th>" + (el.dataset.runName || "?") + "</th>";
    }).join("");

    function row(label, fn) {
      return "<tr><th>" + label + "</th>" + selected.map(function (el) {
        return "<td>" + fn(el) + "</td>";
      }).join("") + "</tr>";
    }

    var rows = [
      row("algorithm", function (el) { return el.dataset.algorithm || "-"; }),
      row("vla_config", function (el) { return el.dataset.vlaConfig || "-"; }),
      row("nlive", function (el) { return el.dataset.nlive || "-"; }),
      row("metric", function (el) { return el.dataset.metric || "-"; }),
      row("log(Z)", function (el) { return el.dataset.logz || "-"; }),
      row("evals", function (el) { return el.dataset.evals || "-"; })
    ];
    paramNames.forEach(function (name) {
      rows.push(row(name, function (el) {
        return paramRangesFor(el)[name] || "-";
      }));
    });

    comparePanel.hidden = false;
    comparePanel.innerHTML = '<div class="ns-compare-header"><h3>Comparing ' + selected.length
      + " run" + (selected.length === 1 ? "" : "s") + '</h3>'
      + '<button type="button" id="ns-compare-clear">Clear</button></div>'
      + '<div class="ns-compare-table-wrap"><table class="ns-compare-table"><thead><tr><th></th>'
      + headerCells + "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>";

    var clearBtn = document.getElementById("ns-compare-clear");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        selected.forEach(function (el) {
          var box = el.querySelector(".compare-checkbox");
          if (box) box.checked = false;
        });
        renderCompare();
      });
    }
  }

  // stopPropagation only - not preventDefault. The checkbox's own toggle is
  // itself a cancelable default action of this same click event: calling
  // preventDefault here would stop the card link from navigating but would
  // just as much revert the checkbox back to unchecked. Stopping propagation
  // before the event would reach the wrapping <a> keeps the checkbox's native
  // behaviour intact while the anchor never sees the click at all.
  items.forEach(function (el) {
    var label = el.querySelector(".compare-select");
    if (label) label.addEventListener("click", function (e) { e.stopPropagation(); });
  });
  list.addEventListener("change", function (e) {
    if (e.target.classList.contains("compare-checkbox")) renderCompare();
  });

  algoSel.addEventListener("change", apply);
  mergedSel.addEventListener("change", apply);
  paramSel.addEventListener("change", function () { refreshRangeOptions(); apply(); });
  rangeSel.addEventListener("change", apply);
  sortSel.addEventListener("change", apply);
  refreshRangeOptions();
  apply();
})();
</script>
"""


def unfinished_run_names():
    """Runs that stopped before finishing, newest first.

    summary.json is written only once PolyChord returns, so a run directory
    without a readable one stopped early. That matters here more than
    anywhere: the index below is built by globbing for summary.json, so an
    interrupted run is not listed as failed, it is simply missing - and a run
    that silently vanishes from the report is the one nobody goes back to. A
    torn summary (summary_is_complete) belongs in the same list: the run is
    unreportable until it is rewritten, and `./ri resume` is what rewrites it,
    from the point cache and without imaging anything again.
    """
    names = []
    for path in sorted(glob.glob(os.path.join(NESTED_SAMPLING_DIR, "*")), reverse=True):
        if os.path.isdir(path) and not summary_is_complete(os.path.join(path, "summary.json")):
            names.append(os.path.basename(path))
    return names


def render_unfinished_runs():
    names = unfinished_run_names()
    if not names:
        return ""
    count = len(names)
    items = "".join(
        f"<li><code>{html.escape(name)}</code> &mdash; "
        f"<code>./ri resume {html.escape(name)}</code></li>"
        for name in names
    )
    return (
        '<section class="unfinished">'
        f"<h2>{count} run{'' if count == 1 else 's'} stopped before finishing</h2>"
        "<p>These have no readable <code>summary.json</code>, so they have no page "
        "below. "
        "Each can be continued where it left off, keeping every evaluation it "
        "already finished:</p>"
        f"<ul>{items}</ul>"
        "</section>"
    )


def render_nested_sampling_index(status_for):
    paths = nested_sampling_run_paths()
    if not paths:
        return render_unfinished_runs() + (
            '<p class="empty">No nested-sampling runs found under '
            "results/nested-sampling/*/summary.json yet.</p>"
        )
    entries = []
    algorithm_tokens = []
    param_ranges_by_name = {}
    for p in paths:
        entry_html, algorithm_token, parameter_space = render_index_entry(p, status_for(p))
        entries.append(entry_html)
        if algorithm_token and algorithm_token not in algorithm_tokens:
            algorithm_tokens.append(algorithm_token)
        for spec in parameter_space:
            name = spec.get("name")
            range_label = format_param_range(spec)
            if not name or not range_label:
                continue
            param_ranges_by_name.setdefault(name, set()).add(range_label)
    algorithm_tokens.sort()
    param_ranges_by_name = {
        name: sorted(ranges) for name, ranges in sorted(param_ranges_by_name.items())
    }
    return (
        render_unfinished_runs()
        + render_index_toolbar(algorithm_tokens, param_ranges_by_name)
        + f'<div id="ns-index-list">{"".join(entries)}</div>'
        + INDEX_SCRIPT
    )


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
    """Both page bodies, with LIKELIHOOD_SLOT standing in for the corner plot."""
    summary_path = item[0]
    body = index_nav_html() + render_nested_sampling_run(
        summary_path, likelihood_html=LIKELIHOOD_SLOT
    )
    return body, render_run_images_page(summary_path)


def likelihood_task(item):
    """URL of one run's corner plot, or None. Runs alongside its page body."""
    summary_path = item[0]
    with open(summary_path) as f:
        summary = json.load(f)
    space_names = [
        spec["name"] for spec in summary.get("parameter_space", []) if "name" in spec
    ]
    return render_likelihood_plot(os.path.dirname(summary_path), space_names)


def write_run_page(item, body, images_body):
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
    write_html_doc(
        os.path.join(os.path.dirname(item[1]), run_images_page_name(run_name)),
        title=f"nested-sampling images: {run_name}",
        subtitle="One reconstruction per evaluation, best objective first.",
        body=images_body,
    )


def main(argv=None):
    # The report is a batch process: it builds its figures, writes the pages
    # and exits, so nothing it allocates has to be reclaimed while it runs.
    # matplotlib figures are dense with reference cycles, so the cyclic
    # collector wakes constantly and finds almost nothing that refcounting
    # would not have freed anyway - it is 12% of a corner plot's render and 9%
    # of the whole build. Disable it here rather than at import, so importing
    # this module (the self-checks, the host-side helpers) leaves the caller's
    # gc alone; the pool workers inherit the setting across fork. The cycles it
    # would have collected are simply held to exit instead, which measured as
    # +3MB of a worker's 128MB peak.
    gc.disable()
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
        # Only a build with something to draw forks anything, and the import is
        # 3.4ms of a rebuild that has nothing to do.
        import multiprocessing

        workers = min(len(todo), os.cpu_count() or 1)
        if drawing:
            load_chain_libs()
        with multiprocessing.Pool(workers) as plot_pool:
            plots = [plot_pool.apply_async(likelihood_task, (item,)) for item in todo]
            if drawing:
                load_render_libs()
            with multiprocessing.Pool(workers) as body_pool:
                bodies = [body_pool.apply_async(run_body_task, (item,)) for item in todo]
                # write_run_page already prints "wrote <path>" per page, on
                # its own newline-terminated line - a \r-redrawn line here
                # collided with it instead of overwriting it (no newline of
                # its own to end on, so the next page's "wrote ..." print
                # kept appending to the end of this one instead of starting
                # fresh), which is what made this look like nothing was
                # printed at all rather than a run-on mess. Printing our own
                # plain line, always newline-terminated, avoids that outright
                # instead of trying to redraw in place.
                start = time.monotonic()
                for i, (item, plot, body) in enumerate(zip(todo, plots, bodies), start=1):
                    page_body, images_body = body.get()
                    write_run_page(
                        item,
                        page_body.replace(LIKELIHOOD_SLOT, likelihood_section(plot.get())),
                        images_body,
                    )
                    elapsed = time.monotonic() - start
                    eta = format_duration(elapsed / i * (len(todo) - i)) if i < len(todo) else "0s"
                    print(f"[{i}/{len(todo)}] elapsed {format_duration(elapsed)}  eta {eta}")

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
    # and the imaging rows are named after the run's algorithm. 8 ranks are 7
    # workers - rank 0 administrates - so the budget is 7 x the wall clock.
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
    assert "7 workers + administrator" in html_out and "53m 09s" in html_out, html_out
    assert "convert" not in html_out, html_out
    # Every top-level stage plus the unaccounted remainder is charted, adds to
    # 100%, and is repeated once per worker lane.
    assert html_out.count('class="profile-bar"') == 7, html_out
    assert html_out.count('class="profile-seg"') == 4 * 7, html_out
    assert "unaccounted (PolyChord sampling + idle)" in html_out, html_out
    shares = [float(s) for s in re.findall(r"--seg-share: ([\d.]+)", html_out)]
    assert abs(sum(shares) / 7 - 1.0) < 1e-6, shares
    # The parallelisation arithmetic is spelled out and lands on the wall clock
    # the run header shows, and the table carries the wall-clock column that
    # turns each stage's worker-seconds into its cost in wall time.
    assert (
        "39m 49s accounted + 13m 20s unaccounted = 53m 09s of worker-time ÷ 7 workers"
        " <strong>= 7m 36s end-to-end wall clock</strong>"
    ) in html_out, html_out
    assert "7 of them running side by side" in html_out, html_out
    assert ">wall clock</th>" in html_out, html_out
    assert "5m 36s" in html_out, html_out  # r2d2's 39m 15s of worker-time, in wall clock
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
    # Records carrying their wall-clock intervals: PolyChord and idle are rows
    # and chart segments of their own rather than one "unaccounted" bucket, and
    # the harness joins the stages above the sum. See profiling_breakdown() for
    # the arithmetic behind these numbers.
    split = render_profiling({"algorithm": "wsclean", "profiling": {
        "mpi_procs": 3, "total_wall_seconds": 10.0,
        "stage_totals_seconds": {"simulate": 5.5},
        "stage_eval_counts": {"simulate_seconds": 2},
        "accounted_worker_seconds": 5.5,
        "busy_worker_seconds": 6.0, "busy_wall_seconds": 4.0,
    }})
    assert "unaccounted" not in split, split
    assert 'class="profile-stage-sub"' not in split, split
    assert "PolyChord (no evaluation in flight)</td><td class=\"num\">12.0s</td>" in split, split
    assert "idle (waiting on other workers)</td><td class=\"num\">2.00s</td>" in split, split
    assert "harness (Python around the stages)</td><td class=\"num\">500ms</td>" in split, split
    assert "evaluating (sum of the above)" in split, split
    assert (
        "6.00s evaluating + 12.0s PolyChord + 2.00s idle = 20.0s of worker-time ÷ 2 workers"
        " <strong>= 10.0s end-to-end wall clock</strong>"
    ) in split, split
    # One segment per top-level row, in both lanes, adding to the whole budget.
    assert split.count('class="profile-seg"') == 4 * 2, split
    shares = [float(s) for s in re.findall(r"--seg-share: ([\d.]+)", split)]
    assert abs(sum(shares) / 2 - 1.0) < 1e-6, shares
    # No profiling block: nothing rendered.
    assert render_profiling({}) == ""
    # A profiling block with nothing in it must not divide by zero.
    empty = render_profiling({"profiling": {"mpi_procs": 1, "total_wall_seconds": 0.0}})
    assert 'class="profile-seg"' not in empty, empty


def _self_check_page_status():
    import shutil
    import tempfile

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


def _self_check_torn_summary():
    """One half-written summary.json must not take the whole report down.

    Every page and the index are built in one pass, so json.load raising on one
    run left every other run with no report at all - reproduced by truncating a
    real 285KB summary. The run is not lost either: it joins the runs that
    stopped before finishing, with the `./ri resume` that rewrites it.
    """
    import shutil
    import tempfile

    global NESTED_SAMPLING_DIR
    saved = NESTED_SAMPLING_DIR
    tmp_dir = tempfile.mkdtemp(prefix="ns-report-selfcheck-")
    try:
        NESTED_SAMPLING_DIR = tmp_dir
        whole, torn = os.path.join(tmp_dir, "wsclean-ok"), os.path.join(tmp_dir, "wsclean-torn")
        os.makedirs(whole)
        os.makedirs(torn)
        with open(os.path.join(whole, "summary.json"), "w") as f:
            json.dump({"algorithm": "wsclean", "evaluations": []}, f, indent=2)
        # Where a kill tears it: mid-record, inside the evaluations list.
        with open(os.path.join(torn, "summary.json"), "w") as f:
            f.write('{\n  "evaluations": [\n    {\n      "eval')
        assert summary_is_complete(os.path.join(whole, "summary.json"))
        assert not summary_is_complete(os.path.join(torn, "summary.json"))
        # An empty one is what a full disk leaves, and there is no file at all
        # for a run that is still going.
        open(os.path.join(tmp_dir, "empty.json"), "w").close()
        assert not summary_is_complete(os.path.join(tmp_dir, "empty.json"))
        assert not summary_is_complete(os.path.join(tmp_dir, "no-such.json"))

        assert nested_sampling_run_paths() == [os.path.join(whole, "summary.json")], \
            nested_sampling_run_paths()
        assert unfinished_run_names() == ["wsclean-torn"], unfinished_run_names()
    finally:
        NESTED_SAMPLING_DIR = saved
        shutil.rmtree(tmp_dir)


def _self_check_index_toolbar():
    """Index cards carry the algorithm/merged/evals/parameter-space data the
    toolbar filters, sorts, and compares on, and the toolbar options match what
    was actually seen across runs."""
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="ns-report-selfcheck-")
    try:
        r2d2_dir = os.path.join(tmp_dir, "r2d2-run")
        os.makedirs(r2d2_dir)
        r2d2_summary = os.path.join(r2d2_dir, "summary.json")
        with open(r2d2_summary, "w") as f:
            json.dump({
                "algorithm": "r2d2",
                "evaluations": [{"eval_id": 1, "objective": 1.0}, {"eval_id": 2, "error": "boom"}],
                "parameter_space": [
                    {"name": "log10_dynamic_range", "min": 2.0, "max": 3.0},
                    {"name": "channel_count", "min": 2, "max": 6, "kind": "integer"},
                ],
            }, f)
        card, token, parameter_space = render_index_entry(r2d2_summary, "current")
        assert token == "r2d2", token
        assert 'data-algorithm="r2d2"' in card, card
        assert 'data-merged="0"' in card, card
        assert 'data-evals="1"' in card, card  # one succeeded, one failed
        assert 'data-param-names="log10_dynamic_range,channel_count"' in card, card
        assert '&quot;log10_dynamic_range&quot;: &quot;2-3&quot;' in card, card
        assert 'class="compare-checkbox"' in card, card
        assert '<div class="badges param-badges">' in card, card
        assert '<span class="badge badge-param">log10_dynamic_range 2-3</span>' in card, card
        assert '<span class="badge badge-param">channel_count 2-6</span>' in card, card
        assert len(parameter_space) == 2, parameter_space

        wsclean_dir = os.path.join(tmp_dir, "wsclean-run")
        os.makedirs(wsclean_dir)
        wsclean_summary = os.path.join(wsclean_dir, "summary.json")
        with open(wsclean_summary, "w") as f:
            json.dump({
                "algorithm": "wsclean",
                "merged_from": [{"name": "some-source-run"}],
                "evaluations": [],
            }, f)
        card, token, parameter_space = render_index_entry(wsclean_summary, "missing")
        assert token == "wsclean", token
        assert '<div class="card index-entry index-entry-missing" data-algorithm="wsclean"' in card, card
        assert 'data-merged="1"' in card, card
        assert 'data-evals="0"' in card, card
        assert 'data-param-names=""' in card, card
        assert parameter_space == [], parameter_space

        unknown_dir = os.path.join(tmp_dir, "mystery-run")
        os.makedirs(unknown_dir)
        unknown_summary = os.path.join(unknown_dir, "summary.json")
        with open(unknown_summary, "w") as f:
            json.dump({"evaluations": []}, f)
        card, token, _ = render_index_entry(unknown_summary, "current")
        assert token == "", token
        assert 'data-algorithm=""' in card, card

        toolbar = render_index_toolbar(
            ["r2d2", "wsclean", "sasir"],
            {"log10_dynamic_range": ["2-3"], "channel_count": ["2-6"]},
        )
        assert '<option value="r2d2">R2D2</option>' in toolbar, toolbar
        assert '<option value="wsclean">WSClean</option>' in toolbar, toolbar
        assert '<option value="sasir">Sasir</option>' in toolbar, toolbar  # unknown token: title-cased fallback
        assert '<select id="ns-filter-merged">' in toolbar, toolbar
        assert '<select id="ns-filter-param">' in toolbar, toolbar
        assert '<option value="log10_dynamic_range">log10_dynamic_range</option>' in toolbar, toolbar
        assert '<select id="ns-filter-param-range" disabled>' in toolbar, toolbar
        assert '<select id="ns-sort">' in toolbar, toolbar
        assert 'id="ns-compare-panel" hidden' in toolbar, toolbar
        # The ranges map sits inside a <script> element, whose content HTML
        # never entity-decodes - it must stay raw JSON, not html.escape'd.
        assert '"channel_count": ["2-6"]' in toolbar, toolbar
        assert "&quot;" not in toolbar, toolbar
        assert json.loads(toolbar.split('id="ns-param-ranges">')[1].split("</script>")[0]) == {
            "log10_dynamic_range": ["2-3"],
            "channel_count": ["2-6"],
        }, toolbar
    finally:
        shutil.rmtree(tmp_dir)


def _self_check_parameter_space_section():
    """The report page's parameter-space table shows each searched parameter's
    range and kind, and is empty when a run has none."""
    html_out = render_parameter_space_section([
        {"name": "log10_dynamic_range", "min": 2.0, "max": 3.0},
        {"name": "channel_count", "min": 2, "max": 6, "kind": "integer"},
    ])
    assert "<summary>Parameter space</summary>" in html_out, html_out
    assert "<td>log10_dynamic_range</td><td>2-3</td><td></td>" in html_out, html_out
    assert "<td>channel_count</td><td>2-6</td><td>integer</td>" in html_out, html_out
    assert render_parameter_space_section([]) == ""


def _self_check_cached_png():
    """A second request for the same key reuses the file instead of re-rendering."""
    import shutil
    import tempfile

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


def _self_check_shared_axes_dedupe():
    """A repeated handle_shared_axes call must skip the scan but still un-stale
    the view limits of every axis the first scan selected."""
    load_plot_libs()
    if not dedupe_pandas_tick_housekeeping():
        return
    from pandas.plotting._matplotlib import core as pandas_plot_core
    from pandas.plotting._matplotlib import tools

    fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
    flat = list(axarr.flat)
    scanned = []
    real_scan = tools._has_externally_shared_axis
    tools._has_externally_shared_axis = lambda ax, which: (
        scanned.append(ax) or real_scan(ax, which)
    )
    try:
        # sharex/sharey False with axes that do share is the anesthetic case:
        # it is what sends pandas into the per-axis _has_externally_shared_axis
        # scan this dedupe removes.
        args = dict(
            axarr=flat, nplots=4, naxes=4, nrows=2, ncols=2, sharex=False, sharey=False
        )
        pandas_plot_core.handle_shared_axes(**args)
        selected = fig._report_shared_axes_memo[1]
        assert selected, "first scan selected no axes to strip"
        for ax in flat:
            ax._stale_viewlims["x"] = ax._stale_viewlims["y"] = True
        before = len(scanned)
        pandas_plot_core.handle_shared_axes(**args)
        assert len(scanned) == before, "repeat call re-ran the shared-axis scan"
        assert not any(
            axis.axes._stale_viewlims["x"] or axis.axes._stale_viewlims["y"]
            for axis in selected
        ), "repeat call lost the viewLim un-stale"
    finally:
        tools._has_externally_shared_axis = real_scan
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


def _self_check_text_layout_memo():
    """The text-layout memo must share one layout between two texts with the
    same properties, and miss once any of those properties differs."""
    load_plot_libs()
    if not memoize_matplotlib_text_layout():
        return
    fig, ax = plt.subplots()
    renderer = fig._get_renderer()
    first, same, longer, rotated = (
        ax.text(0.1 * i, 0.5, text)
        for i, text in enumerate(("abc", "abc", "abcdef", "abc"))
    )
    rotated.set_rotation(90)
    layout = first._get_layout(renderer)
    assert same._get_layout(renderer) is layout, "memo missed on an identical text"
    assert longer._get_layout(renderer) is not layout, "memo hit on a different string"
    assert rotated._get_layout(renderer) is not layout, "memo hit on a rotated text"
    same.set_fontsize(plt.rcParams["font.size"] * 2)
    assert same._get_layout(renderer) is not layout, "memo hit after a font change"
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


def _self_check_labelled_column_shortcut():
    """The labelled-column shortcut must return exactly what anesthetic's own
    four-way search returns - same type, same index, same values, same
    relabelled name - on every shape the corner plot can hand it, and raise
    where it raises. Each case is built twice: pandas caches a frame's column
    Series, so resolving both ways on one frame would compare an object with
    itself and prove nothing."""
    if not shortcut_anesthetic_labelled_column():
        return
    from anesthetic.labelled_pandas import (
        LabelledDataFrame,
        LabelledSeries,
        _LabelledObject,
        ac,
    )
    from pandas import MultiIndex

    def searched(frame, key):
        """anesthetic's own resolution, spelled out so the patch can't shadow it."""
        return ac(
            [
                (
                    super(_LabelledObject, frame.drop_labels(i)).__getitem__,
                    frame.get_labels_map(i),
                )
                for i in frame._all_axes()
            ],
            key,
        )

    def same(build, key):
        got, want = build()[key], searched(build(), key)
        assert got is not want, "the two frames shared pandas' column cache"
        assert type(got) is type(want), (key, type(got), type(want))
        assert getattr(got, "name", None) == getattr(want, "name", None), (
            key,
            getattr(got, "name", None),
            getattr(want, "name", None),
        )
        assert [list(a) for a in got.axes] == [list(a) for a in want.axes], (
            key,
            got.axes,
            want.axes,
        )
        assert list(got.to_numpy().ravel()) == list(want.to_numpy().ravel()), key

    labelled = MultiIndex.from_tuples(
        [("a", "$a$"), ("b", "$b$")], names=["params", "labels"]
    )
    # The corner plot's own shape: labelled columns, plain index, string key.
    same(lambda: LabelledDataFrame([[0.0, 1.0], [2.0, 3.0]], columns=labelled), "a")
    # A key that names the whole (param, label) pair instead of the param.
    same(lambda: LabelledDataFrame([[0.0, 1.0]], columns=labelled), ("b", "$b$"))
    # Labels on the index as well - the shortcut still has to pick the
    # candidate that leaves the index alone.
    same(
        lambda: LabelledDataFrame(
            [[0.0, 1.0], [2.0, 3.0]], index=labelled, columns=labelled
        ),
        "b",
    )
    # A key that leaves a column level behind: every candidate is 2-D, so the
    # search falls through to its level count and keeps a *different* one.
    nested = MultiIndex.from_tuples(
        [("g", "a", "$a$"), ("g", "b", "$b$")], names=["group", "params", "labels"]
    )
    same(lambda: LabelledDataFrame([[0.0, 1.0]], columns=nested), "g")
    # Nothing labelled at all, and a Series rather than a frame: both outside
    # the shortcut's guards.
    same(lambda: LabelledDataFrame([[0.0, 1.0]], columns=["a", "b"]), "a")
    assert LabelledSeries([0.0, 1.0], index=labelled)["a"] == searched(
        LabelledSeries([0.0, 1.0], index=labelled), "a"
    )
    # A column that is not there still raises rather than resolving.
    try:
        LabelledDataFrame([[0.0]], columns=labelled[:1])["missing"]
    except KeyError:
        pass
    else:
        raise AssertionError("a missing column silently resolved")


def _self_check_viewlim_skip():
    """The settled-viewLim skip must not run the scan twice for one settled
    epoch, must still autoscale once something asks for it, and must not record
    an epoch when the autoscale it ran re-staled the group."""
    load_plot_libs()
    if not skip_settled_matplotlib_viewlims():
        return
    from matplotlib.axes._base import _AxesBase

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 5])
    scans = []
    settled = _AxesBase._unstale_viewLim

    def counting(self):
        scans.append(self)
        return settled(self)

    _AxesBase._unstale_viewLim = counting
    try:
        ax.viewLim
        first = len(scans)
        ax.viewLim
        assert len(scans) == first + 1, "the skip stopped being called at all"
        assert not ax._stale_viewlims["y"], "a settled axis was left stale"
        ax.plot([0, 1], [0, 50])
        ax.viewLim
        assert ax.get_ylim()[1] > 5, "a re-staled axis was never autoscaled"
    finally:
        _AxesBase._unstale_viewLim = settled
        plt.close(fig)


def _self_check_axis_map_cache():
    """The cached _axis_map must answer with the axes' own axis objects and
    must notice when fresh ones are swapped in."""
    load_plot_libs()
    if not cache_matplotlib_axis_map():
        return
    fig, ax = plt.subplots()
    try:
        first = ax._axis_map
        assert first == {"x": ax.xaxis, "y": ax.yaxis}, first
        assert ax._axis_map is first, "the map was rebuilt for an unchanged axes"
        ax._init_axis()  # the one place matplotlib swaps the axis objects
        assert ax._axis_map == {"x": ax.xaxis, "y": ax.yaxis}, (
            "the cache survived a swap of the axes' axis objects"
        )
    finally:
        plt.close(fig)


def _self_check_alias_map_memo():
    """The memoised normalize_kwargs must canonicalise aliases exactly as
    matplotlib does, keep the two artist classes' maps apart, and still reject
    a canonical name given twice under different aliases."""
    load_plot_libs()
    if not memoize_matplotlib_alias_maps():
        return
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    import matplotlib.cbook as cbook

    normalize = cbook.normalize_kwargs
    assert normalize({"lw": 2, "c": "r"}, Line2D) == {"linewidth": 2, "color": "r"}
    assert normalize({"lw": 2}, Line2D([], [])) == {"linewidth": 2}
    # 'lw' is not an alias for Text, so it must survive untranslated.
    assert normalize({"lw": 2}, Text) == {"lw": 2}
    assert normalize({"lw": 2}, {"width": ["lw"]}) == {"width": 2}
    try:
        normalize({"lw": 1, "linewidth": 2}, Line2D)
    except TypeError:
        pass
    else:
        raise AssertionError("two aliases of one property were accepted")


def _self_check_axes_subclass_sharing():
    """Two panels made by the same anesthetic helper must land on one shared
    subclass, and the two helpers must not share a subclass with each other."""
    load_plot_libs()
    if not share_anesthetic_axes_subclasses():
        return
    try:
        from anesthetic.plot import AxesDataFrame
    except ImportError:
        return
    fig, axs = plt.subplots(1, 3)
    try:
        base = type(axs[0])
        AxesDataFrame._make_diagonal(axs[0])
        AxesDataFrame._make_diagonal(axs[1])
        AxesDataFrame._make_offdiagonal(axs[2])
        assert type(axs[0]) is not base, "the panel was never given a subclass"
        assert type(axs[0]) is type(axs[1]), "two diagonals built two classes"
        assert type(axs[2]) is not type(axs[0]), "both helpers shared a class"
        assert issubclass(type(axs[2]), base), "the subclass lost its base"
        # The shared class must still do the linking the fresh one did.
        axs[1].set_xlim(2, 7)
        assert axs[1].get_ylim() == (2, 7), "the shared class stopped linking"
    finally:
        plt.close(fig)


def _self_check_mplot3d_skip():
    """The skipped 3d projection costs the registry nothing else, and unpoisons."""
    load_plot_libs()
    import mpl_toolkits.mplot3d  # noqa: F401  - the None must be gone again

    names = plt.matplotlib.projections.get_projection_names()
    assert "3d" not in names, names
    assert names == ["aitoff", "hammer", "lambert", "mollweide", "polar", "rectilinear"], names
    # The 2d axes the report actually draws on are unaffected.
    fig = plt.figure()
    assert type(fig.add_subplot()) is plt.matplotlib.axes.Axes
    plt.close(fig)


def _self_check_run_page_name():
    assert run_page_name("wsclean-vlaa-20260826T010221Z") == "wsclean-vlaa-20260826T010221Z.html"
    # Anything that would escape the output directory is flattened.
    assert run_page_name("../etc/passwd") == ".._etc_passwd.html"
    assert run_images_page_name("wsclean-vlaa-20260826T010221Z") == (
        "wsclean-vlaa-20260826T010221Z-images.html"
    )


def _self_check_run_page_split():
    """The run page carries no evaluation raster - only a link to the page that
    does - so opening the details does not decode one PNG per evaluation."""
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="ns-report-selfcheck-")
    try:
        run_dir = os.path.join(tmp_dir, "r2d2-run")
        os.makedirs(run_dir)
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "algorithm": "r2d2",
                "metric": "snr",
                "parameter_space": [{"name": "log10_dynamic_range", "min": 2.0, "max": 3.0}],
                "evaluations": [
                    {
                        "eval_id": 1,
                        "objective": 1.0,
                        "params": {"log10_dynamic_range": 2.5},
                        "paths": {"image": "evals/1/image.fits"},
                    },
                    {"eval_id": 2, "objective": 0.5, "params": {"log10_dynamic_range": 2.1}},
                ],
            }, f)

        page = render_nested_sampling_run(summary_path, likelihood_html=LIKELIHOOD_SLOT)
        assert "<img" not in page, page
        assert "eval-gallery" not in page, page
        assert page.index(LIKELIHOOD_SLOT) < page.index("<h3>Evaluations</h3>"), page
        assert 'href="r2d2-run-images.html"' in page, page
        assert "View 2 evaluation images" in page, page
        # The raw table lost its thumbnail column with them.
        assert "<th>image</th>" not in page, page

        images_page = render_run_images_page(summary_path)
        assert 'href="r2d2-run.html"' in images_page, images_page
        assert "eval-gallery" in images_page, images_page
        assert "#1" in images_page and "#2" in images_page, images_page
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    if os.environ.get("GENERATE_REPORT_SELF_CHECK") == "1":
        _self_check_log_evidence_parser()
        _self_check_run_page_name()
        _self_check_run_page_split()
        _self_check_mplot3d_skip()
        _self_check_cached_png()
        _self_check_render_array_png()
        _self_check_profiling()
        _self_check_page_status()
        _self_check_index_toolbar()
        _self_check_torn_summary()
        _self_check_parameter_space_section()
        _self_check_tick_housekeeping()
        _self_check_shared_axes_dedupe()
        _self_check_tick_memo()
        _self_check_text_layout_memo()
        _self_check_tight_bbox()
        _self_check_labels_map_memo()
        _self_check_drop_labels_memo()
        _self_check_labelled_column_shortcut()
        _self_check_viewlim_skip()
        _self_check_axis_map_cache()
        _self_check_alias_map_memo()
        _self_check_axes_subclass_sharing()
        print("generate_report self-check passed")
    else:
        main()
