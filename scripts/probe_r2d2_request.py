"""cProfile of warm R2D2 imaging requests on real .mat files (docs/csd3-speed.md, round 4).

Inside r2d2.sif, with the working tree bound over /opt/ri-nested-sampling and
the checkpoints at /checkpoints:
    python3 scripts/probe_r2d2_request.py MATS_DIR N
MATS_DIR holds one subdirectory per evaluation with its r2d2_data.mat. The config is
the run's (`write_r2d2_config`); EXTRA appends lines to it. OUT_ROOT puts
the outputs elsewhere (e.g. tmpfs, as a run's scratch does since round 5).
NCPUS sets the config's ncpus (default OMP_NUM_THREADS), as a rank does since round 9.
"""
import cProfile, io, os, pstats, runpy, sys, time
from pathlib import Path
sys.path.insert(0, "/opt/ri-nested-sampling")
import r2d2_serve as r
import polychord_r2d2 as pr
# image_dim() reads defaults.toml with tomllib, which r2d2.sif's Python 3.10 lacks.
pr.image_dim = lambda: int(dim)
root = Path(sys.argv[1]); n = int(sys.argv[2]); dim = os.environ.get("DIM", "32")
r.warm_imports()
prof = cProfile.Profile()
walls = []
for d in sorted(root.iterdir())[:n]:
    out = Path(os.environ.get("OUT_ROOT", d)) / d.name / "out"; out.mkdir(parents=True, exist_ok=True)
    cfg = d / "cfg.yaml"
    # The run's own config writer, so the probe cannot drift from production
    # (a hand-copied config without target_dynamic_range spent 83% of each
    # request on an operator norm runs never compute: csd3-speed.md, round 97).
    os.environ["R2D2_OMP_THREADS"] = os.environ.get("NCPUS", os.environ.get("OMP_NUM_THREADS", "1"))
    pr.write_r2d2_config(cfg, str(d / "r2d2_data.mat"), str(out))
    if os.environ.get("EXTRA"):
        cfg.write_text(cfg.read_text() + os.environ["EXTRA"] + "\n")
    sys.argv = [str(r.IMAGER), "--config", str(cfg), "--ckpt_path", "/checkpoints/R2D2_A1"]
    t = time.perf_counter()
    with r.redirect_fds(d / "o.log", d / "e.log"):
        prof.enable()
        try:
            runpy.run_path(str(r.IMAGER), run_name="__main__")
        except SystemExit:
            pass
        prof.disable()
    walls.append(time.perf_counter() - t)
    print(d.name, round(walls[-1], 3), flush=True)
print("mean wall", sum(walls) / len(walls), "first", walls[0], "mean excl first", sum(walls[1:]) / max(1, len(walls) - 1))
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(60); print(s.getvalue())
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(30); print(s.getvalue())
sys.stdout.flush(); os._exit(0)
