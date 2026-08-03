"""
Builds a single self-contained HTML report from benchmarks/manifests/*.json
(written by record-environment.sh, optionally hand-extended with an
"experiment" section - see benchmarks/manifests/*.json for examples).

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
import sys

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = "/workspace/repo"
MANIFEST_DIR = os.path.join(REPO_ROOT, "benchmarks/manifests")


def render_fits_to_data_uri(path):
    data = fits.getdata(path)
    data = np.squeeze(np.asarray(data, dtype=float))
    if data.ndim != 2:
        return None
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
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
"""


def main():
    manifest_paths = sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.json")), reverse=True)
    cards = []
    for p in manifest_paths:
        with open(p) as f:
            manifest = json.load(f)
        cards.append(render_manifest(manifest, p))

    body = "".join(cards) if cards else '<p class="empty">No manifests found in benchmarks/manifests/ yet - run scripts/record-environment.sh as part of a benchmark run.</p>'

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ri-reproducibility benchmark report</title>
<style>{CSS}</style>
</head>
<body>
<h1>ri-reproducibility benchmark report</h1>
<p class="subtitle">Generated from benchmarks/manifests/ - regenerate with <code>make benchmark-report</code>.</p>
{body}
</body>
</html>
"""

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/report.html"
    with open(out_path, "w") as f:
        f.write(html_doc)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
