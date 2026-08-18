"""
Builds self-contained HTML reports from either:
  - benchmarks/manifests/*.json (pipeline benchmark manifests), or
  - results/nested-sampling-poc/*/poc-summary.json (nested-sampling PoC runs).

Run via scripts/generate-benchmark-report.sh with --kind benchmarks|nested-sampling,
which wraps this in the r2d2 image so it can reuse the pipeline's own
astropy + matplotlib + anesthetic rather than requiring a host Python
environment - same approach as scripts/plot-fits.sh.
"""
import argparse
import base64
import glob
import hashlib
import html
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = "/workspace/repo"
MANIFEST_DIR = os.path.join(REPO_ROOT, "benchmarks/manifests")
NESTED_SAMPLING_DIR = os.path.join(REPO_ROOT, "results/nested-sampling-poc")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nested_sampling"))

LOG_Z_RE = re.compile(
    r"log\(Z\)\s*=\s*([-\d.]+E[+-]\d+)\s*\+/-\s*([-\d.]+E[+-]\d+)",
    re.IGNORECASE,
)
RUN_ID_TS_RE = re.compile(r"(\d{8}T\d{6}Z)$")


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


def figure_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_array_to_data_uri(data, figsize=(4, 4), dpi=130):
    data = np.squeeze(np.asarray(data, dtype=float))
    if data.ndim != 2:
        return None
    norm = _image_norm_for_display(data)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    return figure_to_data_uri(fig)


def render_fits_to_data_uri(path, figsize=(4, 4), dpi=130):
    return render_array_to_data_uri(fits.getdata(path), figsize=figsize, dpi=dpi)


def synthesize_truth_array(image_path, source_flux_jy):
    """Mirror poc_common.compute_image_metrics truth construction."""
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


def short(s, n=12):
    return s[:n] if isinstance(s, str) else s


def format_wall_duration(seconds):
    """Format run wall-clock seconds for display (e.g. 4m 12s)."""
    if seconds is None:
        return None
    total = max(0, int(round(float(seconds))))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


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


def nested_sampling_run_sort_key(poc_summary_path):
    """Sort key for newest-first nested-sampling cards (run-id UTC, else mtime)."""
    run_name = os.path.basename(os.path.dirname(poc_summary_path))
    match = RUN_ID_TS_RE.search(run_name)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return (0, -dt.timestamp(), run_name)
        except ValueError:
            pass
    try:
        mtime = os.path.getmtime(poc_summary_path)
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


def render_posterior_plot(run_dir, param_names):
    try:
        from anesthetic_io import load_nested_samples
    except ImportError:
        return None

    try:
        samples = load_nested_samples(run_dir)
    except Exception:
        return None

    plot_params = [name for name in param_names if name in samples.columns]
    if len(plot_params) < 2:
        return None

    # ncompress=False: anesthetic triangular compression fails on some PoC chains.
    for kind, extra in (("kde", {"ncompress": False}), ("scatter", {})):
        try:
            grid = samples.plot_2d(plot_params, kind=kind, **extra)
            fig = grid.iloc[0, 0].figure
            return figure_to_data_uri(fig)
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


def render_images_posterior_collapsible(tab_id, eval_images_html, posterior_html):
    """Collapsed-by-default Images / Posterior tabs for one nested-sampling run."""
    if not eval_images_html and not posterior_html:
        return ""
    safe_id = html.escape(tab_id)
    tabset = f"""
    <div class="run-media-tabset">
      <input type="radio" class="tab-images-radio" name="tabs-{safe_id}" id="tab-images-{safe_id}">
      <label for="tab-images-{safe_id}">Images</label>
      <input type="radio" class="tab-posterior-radio" name="tabs-{safe_id}" id="tab-posterior-{safe_id}" checked>
      <label for="tab-posterior-{safe_id}">Posterior</label>
      <div class="tab-panel tab-panel-images">{eval_images_html}</div>
      <div class="tab-panel tab-panel-posterior">{posterior_html}</div>
    </div>
    """
    return f"""
    <details>
      <summary>Run images and posterior</summary>
      {tabset}
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
    recon_uri = render_fits_to_data_uri(image_path, figsize=figsize, dpi=dpi)
    if not recon_uri:
        return '<span class="empty">—</span>'
    return (
        f'<figure class="eval-recon">'
        f'<img src="{recon_uri}" alt="eval {html.escape(str(eval_id))} reconstruction">'
        f'<figcaption>recon</figcaption></figure>'
    )


def render_shared_truth_image(image_path, source_flux_jy, figsize=(3.2, 3.2), dpi=120):
    if not image_path:
        return ""
    truth_array = synthesize_truth_array(image_path, source_flux_jy)
    truth_uri = render_array_to_data_uri(truth_array, figsize=figsize, dpi=dpi) if truth_array is not None else None
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


def render_nested_sampling_run(poc_summary_path):
    run_dir = os.path.dirname(poc_summary_path)
    run_name = os.path.basename(run_dir)
    tab_id = run_tab_id(run_name)
    with open(poc_summary_path) as f:
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
          <p class="purpose">From poc-summary.json log_z</p>
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
        thumb = render_eval_recon(image_path, ev.get("eval_id", "?"), figsize=(2.2, 2.2), dpi=100)
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

    posterior_html = '<section><h3>Posterior</h3><p class="empty">Posterior plot unavailable.</p></section>'
    uri = render_posterior_plot(run_dir, space_names)
    if uri:
        posterior_html = f"""
        <section>
          <h3>Posterior</h3>
          <figure class="posterior-plot"><img src="{uri}" alt="posterior corner plot"></figure>
        </section>
        """

    evaluations_html = ""
    if eval_rows:
        glance_summary_html = render_eval_glance_summary(evaluations, metric, len(failed))
        eval_images_html = render_eval_images(evaluations, metric, run_dirs, parameter_space)
        images_collapsible = render_images_posterior_collapsible(
            tab_id, eval_images_html, posterior_html
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
    elif posterior_html:
        evaluations_html = f"""
        <section>
          <h3>Evaluations</h3>
          {render_images_posterior_collapsible(tab_id, "", posterior_html)}
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

    rel_summary = os.path.relpath(poc_summary_path, REPO_ROOT)
    return f"""
    <article class="card nested-sampling-card">
      {header}
      {meta_html}
      {evidence_html}
      {evaluations_html}
      {fixed_html}
      <p class="manifest-name">{html.escape(rel_summary)}</p>
    </article>
    """


def render_manifest(manifest, path):
    tool = manifest.get("tool", "?")
    ts = manifest.get("timestamp_utc", "?")
    repo = manifest.get("repository", {})
    image = manifest.get("image", {})
    host = manifest.get("host", {})
    experiment = manifest.get("experiment", {})
    results = experiment.get("results", {})

    dirty_badge = (
        '<span class="badge badge-warn">dirty tree</span>'
        if repo.get("dirty_working_tree")
        else '<span class="badge badge-ok">clean tree</span>'
    )

    header = f"""
    <header class="card-header">
      <h2>{html.escape(tool)} <span class="ts">{html.escape(ts)}</span></h2>
      <div class="badges">
        {dirty_badge}
        <span class="badge">rev {html.escape(short(repo.get('git_revision', '?')))}</span>
        <span class="badge">{html.escape(host.get('cpu_model', '?'))} ({html.escape(str(host.get('cpu_count', '?')))} cores)</span>
        <span class="badge">{html.escape(image.get('container_architecture', '?'))}</span>
      </div>
    </header>
    """

    purpose_html = ""
    if experiment.get("purpose"):
        purpose_html = f'<p class="purpose">{html.escape(experiment["purpose"])}</p>'

    results_html = ""
    if results:
        snr = results.get("snr_db")
        paper_snr = results.get("paper_table4_mean_snr_db_R2D2_A1_T2")
        headline = ""
        if snr is not None:
            headline = f'<div class="headline">SNR <strong>{snr:.2f} dB</strong>'
            if paper_snr is not None:
                delta = snr - paper_snr
                sign = "+" if delta >= 0 else ""
                headline += f' <span class="delta">({sign}{delta:.2f} dB vs. paper mean {paper_snr:.1f} dB)</span>'
            headline += "</div>"
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{fmt_value(v)}</td></tr>"
            for k, v in metrics_table_rows(results)
        )
        results_html = f"""
        <section>
          <h3>Results</h3>
          {headline}
          <table class="kv">{rows}</table>
        </section>
        """

    provenance_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{fmt_value(v)}</td></tr>"
        for k, v in metrics_table_rows(
            {
                k: v
                for k, v in experiment.items()
                if k not in ("results", "purpose")
            }
        )
    )
    provenance_html = ""
    if provenance_rows:
        provenance_html = f"""
        <details>
          <summary>Input / checkpoint provenance</summary>
          <table class="kv">{provenance_rows}</table>
        </details>
        """

    env_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{fmt_value(v)}</td></tr>"
        for k, v in metrics_table_rows(
            {"image": image, "host": host, "config_file": manifest.get("config_file"),
             "config_file_sha256": manifest.get("config_file_sha256")}
        )
    )
    env_html = f"""
    <details>
      <summary>Environment</summary>
      <table class="kv">{env_rows}</table>
    </details>
    """

    gallery_html = ""
    images_dir = results.get("output_images_dir")
    if images_dir:
        abs_dir = os.path.join(REPO_ROOT, images_dir)
        fits_files = sorted(glob.glob(os.path.join(abs_dir, "*.fits")))
        cards = []
        for f in fits_files:
            uri = render_fits_to_data_uri(f)
            if uri is None:
                continue
            cards.append(
                f'<figure><img src="{uri}" alt="{html.escape(os.path.basename(f))}">'
                f'<figcaption>{html.escape(os.path.basename(f))}</figcaption></figure>'
            )
        if cards:
            gallery_html = f'<section><h3>Output images</h3><div class="gallery">{"".join(cards)}</div></section>'

    manifest_name = os.path.basename(path)
    return f"""
    <article class="card">
      {header}
      {purpose_html}
      {results_html}
      {gallery_html}
      {provenance_html}
      {env_html}
      <p class="manifest-name">benchmarks/manifests/{html.escape(manifest_name)}</p>
    </article>
    """


CSS = """
:root { color-scheme: light dark; }
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
.posterior-plot { margin: 0.5rem 0; }
.posterior-plot img { max-width: 100%; height: auto; border-radius: 6px; }
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
.run-media-tabset input.tab-posterior-radio:checked + label {
  opacity: 1;
  background: color-mix(in srgb, CanvasText 6%, transparent);
  border-color: color-mix(in srgb, CanvasText 15%, transparent);
  border-bottom-color: Canvas;
}
.run-media-tabset .tab-panel { display: none; padding-top: 0.75rem; }
.run-media-tabset input.tab-images-radio:checked ~ .tab-panel-images { display: block; }
.run-media-tabset input.tab-posterior-radio:checked ~ .tab-panel-posterior { display: block; }
.section-heading { font-size: 1rem; margin: 2rem 0 0.75rem; opacity: 0.85; }
"""


def render_benchmark_body():
    manifest_paths = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")), reverse=True)
    cards = []
    for p in manifest_paths:
        with open(p) as f:
            manifest = json.load(f)
        cards.append(render_manifest(manifest, p))
    if cards:
        return "".join(cards)
    return (
        '<p class="empty">No manifests found in benchmarks/manifests/ yet - '
        "run scripts/record-environment.sh as part of a benchmark run.</p>"
    )


def render_nested_sampling_body():
    nested_paths = sorted(
        glob.glob(os.path.join(NESTED_SAMPLING_DIR, "*", "poc-summary.json")),
        key=nested_sampling_run_sort_key,
    )
    nested_cards = [render_nested_sampling_run(p) for p in nested_paths]
    if nested_cards:
        return "".join(nested_cards)
    return (
        '<p class="empty">No nested-sampling PoC runs found under '
        "results/nested-sampling-poc/*/poc-summary.json yet.</p>"
    )


def write_html_doc(out_path, title, subtitle, body):
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
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
        "--kind",
        choices=("benchmarks", "nested-sampling"),
        required=True,
        help="Which report to generate.",
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default=None,
        help="Output HTML path (defaults depend on --kind).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.kind == "benchmarks":
        out_path = args.out_path or "/workspace/out/report.html"
        write_html_doc(
            out_path,
            title="ri-reproducibility benchmark report",
            subtitle=(
                "Generated from <code>benchmarks/manifests/</code> - "
                "regenerate with <code>make benchmark-report</code>."
            ),
            body=render_benchmark_body(),
        )
    else:
        out_path = args.out_path or "/workspace/out/nested-sampling-report.html"
        write_html_doc(
            out_path,
            title="ri-reproducibility nested-sampling report",
            subtitle=(
                "Generated from <code>results/nested-sampling-poc/*/poc-summary.json</code> - "
                "regenerate with <code>make nested-sampling-report</code>."
            ),
            body=render_nested_sampling_body(),
        )


def _self_check_log_evidence_parser():
    sample = """
Global evidence:
----------------

log(Z)       =   0.145917983191460E+001 +/-   0.309608121862379E-001
"""
    parsed = parse_log_evidence_from_text(sample)
    assert parsed == (1.4591798319146, 0.0309608121862379), parsed


if __name__ == "__main__":
    if os.environ.get("GENERATE_BENCHMARK_REPORT_SELF_CHECK") == "1":
        _self_check_log_evidence_parser()
        print("generate_benchmark_report self-check passed")
    else:
        main()
