"""
Builds a single self-contained HTML report from benchmarks/manifests/*.json
(written by record-environment.sh, optionally hand-extended with an
"experiment" section - see benchmarks/manifests/*.json for examples) and from
results/nested-sampling-poc/*/poc-summary.json (nested-sampling PoC runs).

Run via scripts/generate-benchmark-report.sh, which wraps this in the r2d2
image so it can reuse the pipeline's own astropy + matplotlib rather than
requiring a host Python environment - same approach as scripts/plot-fits.sh.

Not a general-purpose report format: it renders whatever each manifest's
"experiment" block happens to contain (results table, input/checkpoint
provenance, output images), and degrades to just the manifest's baseline
run/environment metadata when that block is absent, so it stays useful as
new tools (wsclean) and fields get added without needing a fixed schema.
"""
import base64
import glob
import html
import io
import json
import os
import re
import sys
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

LOG_Z_RE = re.compile(
    r"log\(Z\)\s*=\s*([-\d.]+E[+-]\d+)\s*\+/-\s*([-\d.]+E[+-]\d+)",
    re.IGNORECASE,
)


def render_fits_to_data_uri(path):
    data = fits.getdata(path)
    data = np.squeeze(np.asarray(data, dtype=float))
    if data.ndim != 2:
        return None
    vmin, vmax = ZScaleInterval().get_limits(data)
    if vmin == vmax:
        # ZScaleInterval degenerates to (0, 0) on sparse images (e.g. a
        # CLEAN component model that's mostly zeros) - fall back to the
        # actual data range rather than feeding AsinhStretch a zero-width
        # interval (which divides by zero and renders every pixel NaN).
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    fig, ax = plt.subplots(figsize=(4, 4), dpi=130)
    ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def short(s, n=12):
    return s[:n] if isinstance(s, str) else s


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


def render_array_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_posterior_plot(chain_root, param_names):
    try:
        import anesthetic
    except ImportError:
        return None

    try:
        samples = anesthetic.read_chains(chain_root)
    except Exception:
        return None

    plot_params = [name for name in param_names if name in samples.columns]
    if len(plot_params) < 2:
        return None

    try:
        grid = anesthetic.plot.plot_2d(samples, plot_params[: min(4, len(plot_params))])
        return render_array_to_data_uri(grid.fig)
    except Exception:
        return None


def render_nested_sampling_run(poc_summary_path):
    run_dir = os.path.dirname(poc_summary_path)
    run_name = os.path.basename(run_dir)
    with open(poc_summary_path) as f:
        summary = json.load(f)

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

    header = f"""
    <header class="card-header">
      <h2>nested sampling: {html.escape(str(algorithm))}
        <span class="ts">{html.escape(run_name)}</span></h2>
      <div class="badges">
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

    stats_path, chain_root = find_chain_stats(run_dir)
    evidence_html = '<section><h3>Evidence</h3><p class="empty">Global evidence unavailable (no chains/*.stats file).</p></section>'
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
        image_path = resolve_run_path(run_dir, (ev.get("paths") or {}).get("image"))
        thumb = ""
        if image_path:
            uri = render_fits_to_data_uri(image_path)
            if uri:
                thumb = f'<img class="eval-thumb" src="{uri}" alt="eval {ev.get("eval_id", "?")}">'
        if not thumb:
            thumb = '<span class="empty">—</span>'
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

    evaluations_html = ""
    if eval_rows:
        evaluations_html = f"""
        <section>
          <h3>Evaluations</h3>
          <div class="eval-table-wrap">
            <table class="eval-table">
              <thead>{eval_header}</thead>
              <tbody>{"".join(eval_rows)}</tbody>
            </table>
          </div>
          {failed_html}
        </section>
        """

    posterior_html = '<section><h3>Posterior</h3><p class="empty">Posterior plot unavailable.</p></section>'
    if chain_root:
        uri = render_posterior_plot(chain_root, space_names)
        if uri:
            posterior_html = f"""
            <section>
              <h3>Posterior</h3>
              <figure class="posterior-plot"><img src="{uri}" alt="posterior corner plot"></figure>
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
      {posterior_html}
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
.eval-thumb { width: 96px; height: auto; border-radius: 4px; display: block; }
.posterior-plot { margin: 0.5rem 0; }
.posterior-plot img { max-width: 100%; height: auto; border-radius: 6px; }
.section-heading { font-size: 1rem; margin: 2rem 0 0.75rem; opacity: 0.85; }
"""


def main():
    manifest_paths = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")), reverse=True)
    cards = []
    for p in manifest_paths:
        with open(p) as f:
            manifest = json.load(f)
        cards.append(render_manifest(manifest, p))

    nested_paths = sorted(glob.glob(os.path.join(NESTED_SAMPLING_DIR, "*", "poc-summary.json")), reverse=True)
    nested_cards = [render_nested_sampling_run(p) for p in nested_paths]

    sections = []
    if cards:
        sections.append("".join(cards))
    elif not nested_cards:
        sections.append(
            '<p class="empty">No manifests found in benchmarks/manifests/ yet - run scripts/record-environment.sh as part of a benchmark run.</p>'
        )
    if nested_cards:
        sections.append('<h2 class="section-heading">Nested-sampling PoC runs</h2>')
        sections.append("".join(nested_cards))

    body = "".join(sections)

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ri-reproducibility benchmark report</title>
<style>{CSS}</style>
</head>
<body>
<h1>ri-reproducibility benchmark report</h1>
<p class="subtitle">Generated from benchmarks/manifests/ and results/nested-sampling-poc/ - regenerate with <code>make benchmark-report</code>.</p>
{body}
</body>
</html>
"""

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/report.html"
    with open(out_path, "w") as f:
        f.write(html_doc)
    print(f"wrote {out_path}")


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
