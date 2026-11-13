"""Time R2D2's U-Net forward through ONNX Runtime's CUDA EP, 32x32 (docs/csd3-speed.md, round 167).

No torch: the image's torch is CUDA 13 and onnxruntime-gpu is CUDA 12, and both
load a libcudnn.so.9. ONNX is the exported net (r2d2_serve's models/<key>/N1.onnx),
REF an .npz with x and y from the CPU session (production's path) to compare with.
The CUDA 12 libraries come from pip wheels on PYTHONPATH (ort.preload_dlls()).
ALGO is cudnn_conv_algo_search (EXHAUSTIVE, HEURISTIC, DEFAULT), GRAPH=1 enables
the EP's CUDA graph, TF32=0 (default) computes float32 as r2d2_serve does.
EP=trt (round 168) runs TensorRT's EP in front of the CUDA EP, fp32, its engine
cached in TRT_CACHE (build it solo first); libnvinfer.so.10 and libnvonnxparser.so.10
come from tensorrt-cu12-libs in TRT_LIBS (preloaded), and NVIDIA_TF32_OVERRIDE=0 keeps
TensorRT's builder off TF32.
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
if os.environ.get("EP") == "trt":
    import ctypes
    for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        ctypes.CDLL(os.path.join(os.environ["TRT_LIBS"], lib), mode=ctypes.RTLD_GLOBAL)
    ort.set_default_logger_severity(2)
opts = {"device_id": "0", "use_tf32": os.environ.get("TF32", "0"),
        "cudnn_conv_algo_search": os.environ.get("ALGO", "EXHAUSTIVE"),
        "enable_cuda_graph": os.environ.get("GRAPH", "0")}
providers = [("CUDAExecutionProvider", opts)]
if os.environ.get("EP") == "trt":
    cache = os.environ["TRT_CACHE"]
    providers.insert(0, ("TensorrtExecutionProvider", {
        "device_id": "0", "trt_fp16_enable": "False", "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": cache, "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": cache, "trt_builder_optimization_level": os.environ.get("TRT_OPT", "3"),
        "trt_cuda_graph_enable": str(os.environ.get("GRAPH") == "1")}))
sess = ort.InferenceSession(os.environ["ONNX"], so, providers=providers)
assert sess.get_providers()[0] == providers[0][0], sess.get_providers()
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
