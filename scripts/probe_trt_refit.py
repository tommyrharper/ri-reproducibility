"""One refittable TensorRT engine per worker, refitted from on-card weights before each forward (docs/csd3-speed.md, round 181).

Round 170 found 25 fp32 engines hold 5.1GB a worker (206MB each against 124MB
of weights), so an A100 fits 12 TensorRT workers against eager's 19, and
rounds 171-173 found that rank cap is what holds TensorRT's search gain to
single digits. This asks whether one REFIT engine, refitted from the 25 nets'
weights held on the card (TensorLocation.DEVICE) before each forward, keeps
TensorRT's forward speed at eager's ~3.6GB a worker:
    MODE=build  build ENGDIR/refit.plan from N1.onnx (fp32, TF32 off, 1x2x32x32)
    MODE=acc    per net, max abs difference of the refitted engine from the
                per-net plan in PLANDIR, relative to that plan's peak
    MODE=time   N forwards walking N1..N25, each after a refit (REFIT=0: no
                refit, the engine stays N1); START (epoch seconds) holds the
                timed loop so P copies under MPS overlap. Prints device memory.
"""
import os, sys, time
import numpy as np
import onnx
from onnx import numpy_helper
import torch
import tensorrt as trt

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
log = trt.Logger(trt.Logger.ERROR)
ITERS = range(1, int(os.environ.get("ITERS", "25")) + 1)
eng_dir = os.environ["ENGDIR"]
onnx_dir = os.environ["ONNX_DIR"]
mode = os.environ["MODE"]
shape = (1, 2, 32, 32)

if mode == "build":
    t = time.perf_counter()
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    assert parser.parse(open(f"{onnx_dir}/N1.onnx", "rb").read())
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.REFIT)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", shape, shape, shape)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    open(f"{eng_dir}/refit.plan", "wb").write(bytes(plan))
    print(f"built refit.plan in {time.perf_counter() - t:.0f}s, {len(bytes(plan)) / 1e6:.0f}MB")
    sys.exit()

dev = torch.device("cuda")
torch.cuda.synchronize()
free0 = torch.cuda.mem_get_info()[0]
runtime = trt.Runtime(log)
engine = runtime.deserialize_cuda_engine(open(f"{eng_dir}/refit.plan", "rb").read())
ctx = engine.create_execution_context()
ctx.set_input_shape("x", shape)
refitter = trt.Refitter(engine, log)
refitter.weights_validation = os.environ.get("VALIDATE") == "1"
names = list(refitter.get_all_weights())
mem_engine = free0 - torch.cuda.mem_get_info()[0]


def onnx_weights(i):
    m = onnx.load(f"{onnx_dir}/N{i}.onnx")
    w = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    w[n.output[0]] = numpy_helper.to_array(a.t)
    missing = [n for n in names if n not in w]
    assert not missing, f"N{i}: {len(missing)} refit weights not in the ONNX file, e.g. {missing[:5]}"
    return {n: torch.from_numpy(np.ascontiguousarray(w[n])).to(dev) for n in names}


weights = {i: onnx_weights(i) for i in ITERS}
torch.cuda.synchronize()
mem_all = free0 - torch.cuda.mem_get_info()[0]
stream = torch.cuda.Stream()
DT = {torch.float32: trt.DataType.FLOAT, torch.int64: trt.DataType.INT64, torch.int32: trt.DataType.INT32}


def refit(i):
    for n, t in weights[i].items():
        assert refitter.set_named_weights(n, trt.Weights(DT[t.dtype], t.data_ptr(), t.numel()), trt.TensorLocation.DEVICE), n
    assert refitter.refit_cuda_engine_async(stream.cuda_stream), f"refit N{i}"


x = torch.randn(shape, device=dev)
y = torch.empty(1, 1, 32, 32, device=dev)
ctx.set_tensor_address("x", x.data_ptr())
ctx.set_tensor_address("y", y.data_ptr())
print(f"refit weights={len(names)} engine_MB={mem_engine / 2**20:.0f} with_25_weight_sets_MB={mem_all / 2**20:.0f} "
      f"activations_MB={engine.device_memory_size_v2 / 2**20:.1f}", flush=True)

if mode == "acc":
    gen = torch.Generator(device=dev).manual_seed(181)
    worst = 0.0
    for i in ITERS:
        ref_eng = runtime.deserialize_cuda_engine(open(f"{os.environ['PLANDIR']}/N{i}.plan", "rb").read())
        rctx = ref_eng.create_execution_context()
        rctx.set_input_shape("x", shape)
        refit(i)
        row = []
        for scale in (1e-3, 1.0, 1e3):
            xi = torch.randn(shape, device=dev, generator=gen) * scale
            yr = torch.empty(1, 1, 32, 32, device=dev)
            for c, out in ((ctx, y), (rctx, yr)):
                c.set_tensor_address("x", xi.data_ptr())
                c.set_tensor_address("y", out.data_ptr())
                assert c.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            rel = ((y - yr).abs().max() / yr.abs().max()).item()
            worst = max(worst, rel)
            row.append(f"{scale:g}:{rel:.1e}")
        del rctx, ref_eng
        print(f"acc N{i} " + " ".join(row), flush=True)
    print(f"acc worst maxrel={worst:.2e}")
    sys.exit()

n = int(os.environ.get("N", "500"))
do_refit = os.environ.get("REFIT", "1") == "1"
done = torch.cuda.Event(blocking=True)
for k in range(30):
    if do_refit:
        refit(1 + k % 25)
    assert ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
start = float(os.environ.get("START", "0"))
while time.time() < start:
    time.sleep(0.01)
t, c = time.perf_counter(), time.process_time()
refit_s = 0.0
for k in range(n):
    if do_refit:
        r = time.perf_counter()
        refit(1 + k % 25)
        refit_s += time.perf_counter() - r
    assert ctx.execute_async_v3(stream.cuda_stream)
    done.record(stream)
    done.synchronize()
wall = time.perf_counter() - t
print(f"{1000 * wall / n:.3f} ms/image cpu_ms={1000 * (time.process_time() - c) / n:.2f} "
      f"refit_call_ms={1000 * refit_s / n:.3f} refit={int(do_refit)} peak_MB={mem_all / 2**20:.0f}", flush=True)
