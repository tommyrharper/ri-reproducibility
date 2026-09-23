"""cProfile of warm R2D2 imaging requests on real .mat files (docs/csd3-speed.md, round 4).

Inside r2d2.sif, with the working tree bound over /opt/ri-nested-sampling and
the checkpoints at /checkpoints:
    python3 scripts/probe_r2d2_request.py MATS_DIR N
MATS_DIR holds one subdirectory per evaluation with its r2d2_data.mat. EXTRA
appends lines to the config (e.g. `target_dynamic_range: 0.0`). OUT_ROOT puts
the outputs elsewhere (e.g. tmpfs, as a run's scratch does since round 5).
"""
import cProfile, io, os, pstats, runpy, sys, time
from pathlib import Path
sys.path.insert(0, "/opt/ri-nested-sampling")
import r2d2_serve as r
root = Path(sys.argv[1]); n = int(sys.argv[2]); dim = os.environ.get("DIM", "32")
r.warm_imports()
prof = cProfile.Profile()
walls = []
for d in sorted(root.iterdir())[:n]:
    out = Path(os.environ.get("OUT_ROOT", d)) / d.name / "out"; out.mkdir(parents=True, exist_ok=True)
    cfg = d / "cfg.yaml"
    cfg.write_text(f"""data_file: {d/'r2d2_data.mat'}
output_path: {out}
save_all_outputs: False
nufft_pkg: finufft
super_resolution: 1.5
meas_op_on_gpu: False
meas_dtype: double
im_dim_x: {dim}
im_dim_y: {dim}
data_weighting: True
natural_weight: True
weight_type: briggs
num_iter: 25
num_chans: 64
series: R2D2
layers: 1
architecture: unet
prune: True
sigma_res_tol: 1e-4
{os.environ.get("EXTRA","")}
ckpt_path: /checkpoints/R2D2_A1
ckpt_realisations: 1
ncpus: {os.environ.get('OMP_NUM_THREADS','1')}
""")
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
