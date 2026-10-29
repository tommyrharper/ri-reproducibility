"""Time R2D2's U-Net through TensorRT's own Python API beside CUDA 13 torch (docs/csd3-speed.md, round 169).

Round 168 ran TensorRT through onnxruntime-gpu, a CUDA 12 stack that cannot
share a process with the image's torch. This is the integration route instead:
tensorrt-cu13 (on PYTHONPATH) in the same process as torch, the engine run on
torch's device buffers on torch's stream, as r2d2_serve's pool would.
ONNX is the exported net (r2d2_serve's models/<key>/N1.onnx), ENGINE the plan
file to build it into (built when missing, TF32 off, 1x2x32x32), REF the .npz
of round 168 (x and y from CPU ORT). EAGER=1 times torch eager in the same
process instead (checkpoint at /checkpoints, as probe_unet_forward.py) and also
reports TensorRT against it. N forwards; START (epoch seconds) holds the timed
loop so P copies overlap:
    for i in $(seq 19); do apptainer exec --nv ... python3 probe_trt.py & done; wait
"""
import os, sys, time
import numpy as np
import torch
import tensorrt as trt

torch.backends.cuda.matmul.allow_tf32 = False  # as patch_cuda_network
torch.backends.cudnn.allow_tf32 = False
log = trt.Logger(trt.Logger.WARNING)
engine_path = os.environ["ENGINE"]
if not os.path.exists(engine_path):
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    with open(os.environ["ONNX"], "rb") as f:
        assert parser.parse(f.read()), [parser.get_error(i) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 2, 32, 32), (1, 2, 32, 32), (1, 2, 32, 32))
    config.add_optimization_profile(profile)
    t = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    with open(engine_path + f".{os.getpid()}", "wb") as f:
        f.write(plan)
    os.replace(engine_path + f".{os.getpid()}", engine_path)
    print(f"built {engine_path} in {time.perf_counter() - t:.1f}s, {os.path.getsize(engine_path) / 1e6:.0f}MB")
engine = trt.Runtime(log).deserialize_cuda_engine(open(engine_path, "rb").read())
ctx = engine.create_execution_context()
ctx.set_input_shape("x", (1, 2, 32, 32))
dev = torch.device("cuda")
ref = np.load(os.environ["REF"])
x = torch.from_numpy(ref["x"]).to(dev)
y = torch.empty(tuple(ref["y"].shape), dtype=torch.float32, device=dev)
ctx.set_tensor_address("x", x.data_ptr())
ctx.set_tensor_address("y", y.data_ptr())
stream = torch.cuda.current_stream()


def run_trt():
    assert ctx.execute_async_v3(stream.cuda_stream)
    return y


run = run_trt
eager = None
if os.environ.get("EAGER") == "1":
    sys.path.insert(0, "/opt/r2d2/R2D2-RI/src")
    from utils.util_model import create_net_imaging
    net = create_net_imaging()
    ck = torch.load("/checkpoints/R2D2_A1/R2D2_UNet_N1.ckpt", map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    net.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in sd.items()}, assign=True)
    net = net.eval().to(dev)
    torch.set_grad_enabled(False)
    eager = net(x).cpu().numpy()
    run = lambda: net(x)
for _ in range(10):
    run_trt()
torch.cuda.synchronize()
yt = y.cpu().numpy()
for _ in range(10):
    run()
torch.cuda.synchronize()
n = int(os.environ.get("N", "200"))
time.sleep(max(0.0, float(os.environ.get("START", "0")) - time.time()))
c, t = time.process_time(), time.perf_counter()
for _ in range(n):
    run()
    done = torch.cuda.Event(blocking=True)  # each forward waits, as patch_cuda_network's
    done.record()
    done.synchronize()
dt, cpu = (time.perf_counter() - t) / n, (time.process_time() - c) / n
peak = np.abs(ref["y"]).max()
vs_eager = "-" if eager is None else f"{float(np.abs(yt - eager).max() / peak):.2g}"
print(f"{dt * 1e3:.2f} ms/image cpu_ms={cpu * 1e3:.2f} arm={'eager' if eager is not None else 'trt'} trt={trt.__version__} "
      f"maxrel_cpu={float(np.abs(yt - ref['y']).max() / peak):.2g} trt_vs_eager={vs_eager}")
