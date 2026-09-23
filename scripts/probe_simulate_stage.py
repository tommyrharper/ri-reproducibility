"""Per-step timing of the simulate stage on real parameter sets (docs/csd3-speed.md).

Inside meqtrees.sif, with the working tree bound over /opt/ri-nested-sampling
and NS_SCRATCH_DIR set:
    python3 scripts/probe_simulate_stage.py PARAMS.jsonl [time|warm|compare]
PARAMS.jsonl holds one evaluation's metrics.json "params" per line. `warm`
first caches each set's observation at 20 minutes, the steady state of a long
run. `compare` drops the noise and reports the analytic predict's distance
from DATA.
"""
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from casacore.tables import table

sys.path.insert(0, "/opt/ri-nested-sampling")
import simulate_point_source_ms as s  # noqa: E402

T: dict[str, float] = {}


def wrap(name):
    f = getattr(s, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[name] = T.get(name, 0.0) + time.perf_counter() - t

    setattr(s, name, g)


for n in ["write_makems_config", "run_makems", "publish_skeleton", "patch_spectral_window", "make_ms_skeleton",
          "save_observation", "load_observation", "one_timestep_template", "extend_template",
          "fill_point_source_visibilities"]:
    wrap(n)


def argv(p, out):
    return ["--output-ms", str(out), "--observation-minutes", str(p["observation_minutes"]),
            "--channel-count", str(p["channel_count"]), "--start-frequency-hz", str(p["start_frequency_hz"]),
            "--channel-width-hz", str(p["channel_width_hz"]), f"--source-l-arcsec={p['source_l_arcsec']}",
            f"--source-m-arcsec={p['source_m_arcsec']}", f"--declination-deg={p['declination_deg']}",
            "--integration-seconds", str(p["integration_seconds"]), "--dynamic-range", str(p["dynamic_range"]),
            "--seed", str(p["noise_seed"]), "--predict-wait-seconds", "60"]


def analytic(ms, l, m):
    with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as spw:
        f = spw.getcol("CHAN_FREQ")[0]
    with table(str(ms), ack=False) as t:
        uvw = t.getcol("UVW")
        n_corr = t.getcol("DATA", startrow=0, nrow=1).shape[2]
    n = math.sqrt(1 - l * l - m * m)
    ph = 2 * math.pi * (uvw[:, 0:1] * l + uvw[:, 1:2] * m + uvw[:, 2:3] * (n - 1)) * f[None, :] / s.SPEED_OF_LIGHT
    return ph, n_corr


def main():
    params = [json.loads(x) for x in open(sys.argv[1])]
    mode = sys.argv[2] if len(sys.argv) > 2 else "time"
    root = Path(os.environ["NS_SCRATCH_DIR"])
    t0 = time.perf_counter()
    s.warm_up()
    print(f"warm_forest {time.perf_counter() - t0:.2f}s", flush=True)
    if mode == "warm":
        for p in params:
            with tempfile.TemporaryDirectory(dir=root) as d:
                a = s.parse_args(argv({**p, "observation_minutes": 20.0}, Path(d) / "sim.ms"))
                s.make_ms(s.write_makems_config(a, Path(a.output_ms)), Path(a.output_ms), a)
        print(f"warmed in {time.perf_counter() - t0:.1f}s", flush=True)
    T.clear()
    rows = []
    for i, p in enumerate(params):
        with tempfile.TemporaryDirectory(dir=root) as d:
            out = Path(d) / "sim.ms"
            a = s.parse_args(argv(p, out))
            if mode == "compare":
                a.dynamic_range = 1e30
            before = dict(T)
            t = time.perf_counter()
            with s.redirect_fds(Path(d) / "o.log"):
                s.simulate(a)
            wall = time.perf_counter() - t
            steps = {k: round(T[k] - before.get(k, 0.0), 4) for k in T}
            with table(str(out), ack=False) as tb:
                nrow = tb.nrows()
            row = {"i": i, "wall": round(wall, 4), "nrow": nrow, "nchan": p["channel_count"], **steps}
            if mode == "compare":
                ph, n_corr = analytic(out, math.radians(a.source_l_arcsec / 3600), math.radians(a.source_m_arcsec / 3600))
                with table(str(out), ack=False) as tb:
                    data = tb.getcol("DATA")
                for sign in (1, -1):
                    v = np.exp(sign * 1j * ph)
                    row[f"maxdiff{sign:+d}"] = float(max(np.abs(data[:, :, 0] - v).max(), np.abs(data[:, :, -1] - v).max()))
                row["offdiag"] = float(np.abs(data[:, :, 1:-1]).max()) if n_corr > 2 else 0.0
                row["maxphase"] = float(np.abs(ph).max())
            print(json.dumps(row), flush=True)
            rows.append(row)
    keys = sorted({k for r in rows for k in r} - {"i", "nchan"})
    for k in keys:
        vals = [r.get(k, 0.0) for r in rows]
        print(f"{k:32s} mean {np.mean(vals):10.4f} median {np.median(vals):10.4f}")


main()
sys.stdout.flush()
os._exit(0)
