"""Time R2D2's U-Net forward through ONNX Runtime's CUDA EP, 32x32 (docs/csd3-speed.md, round 167).

No torch: the image's torch is CUDA 13 and onnxruntime-gpu is CUDA 12, and both
load a libcudnn.so.9. ONNX is the exported net (r2d2_serve's models/<key>/N1.onnx),
REF an .npz with x and y from the CPU session (production's path) to compare with.
The CUDA 12 libraries come from pip wheels on PYTHONPATH (ort.preload_dlls()).
ALGO is cudnn_conv_algo_search (EXHAUSTIVE, HEURISTIC, DEFAULT), GRAPH=1 enables
the EP's CUDA graph, TF32=0 (default) computes float32 as r2d2_serve does.
N forwards, START (epoch seconds) holds the timed loop so P copies overlap:
    for i in $(seq 19); do apptainer exec --nv ... python3 probe_ort_cuda.py & done; wait
REF=out.npz DEV=cpu writes the reference instead.
"""
import os, time
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
if os.environ.get("DEV", "cuda") == "cpu":
    so.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(os.environ["ONNX"], so, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, 2, 32, 32), dtype=np.float32)
    np.savez(os.environ["REF"], x=x, y=sess.run(None, {"x": x})[0])
    raise SystemExit
ort.preload_dlls()
opts = {"device_id": "0", "use_tf32": os.environ.get("TF32", "0"),
        "cudnn_conv_algo_search": os.environ.get("ALGO", "EXHAUSTIVE"),
        "enable_cuda_graph": os.environ.get("GRAPH", "0")}
sess = ort.InferenceSession(os.environ["ONNX"], so, providers=[("CUDAExecutionProvider", opts)])
ref = np.load(os.environ["REF"])
xo = ort.OrtValue.ortvalue_from_numpy(ref["x"], "cuda", 0)
yo = ort.OrtValue.ortvalue_from_shape_and_type(list(ref["y"].shape), np.float32, "cuda", 0)
bind = sess.io_binding()
bind.bind_ortvalue_input("x", xo)
bind.bind_ortvalue_output("y", yo)
for _ in range(10):
    sess.run_with_iobinding(bind)
y = yo.numpy()
n = int(os.environ.get("N", "200"))
time.sleep(max(0.0, float(os.environ.get("START", "0")) - time.time()))
c, t = time.process_time(), time.perf_counter()
for _ in range(n):
    sess.run_with_iobinding(bind)
dt, cpu = (time.perf_counter() - t) / n, (time.process_time() - c) / n
# Same output again (a CUDA graph replays only its bound buffers).
sess.run_with_iobinding(bind)
y2 = yo.numpy()
print(f"{dt * 1e3:.2f} ms/image cpu_ms={cpu * 1e3:.2f} providers={sess.get_providers()[0]} "
      f"repeat_identical={np.array_equal(y, y2)} maxrel_cpu={float(np.abs(y - ref['y']).max() / np.abs(ref['y']).max()):.2g}")
