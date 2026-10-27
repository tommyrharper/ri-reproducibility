"""Device memory and request-shaped speed of 25 TensorRT engines in one worker (docs/csd3-speed.md, round 170).

Round 169 left one question before TensorRT can replace eager in the GPU pool:
each worker holds all 25 iterations' nets, and a plan is 213MB against 124MB
of fp32 weights. This builds N1..N25.onnx (ONNX_DIR, r2d2_serve's export)
into ENGDIR (fp32, TF32 off, 1x2x32x32) when missing, then in one process
loads all 25 the way a worker would and prints the device memory each step
took (cudaMemGetInfo, so run it alone for MEM numbers):
    KIND=T  25 engines; SHARED=1 gives every context one user-managed
            activation buffer instead of its own
    KIND=E  25 checkpoints' weights on the card, as _CUDA_WEIGHTS holds them
Then it times N forwards walking N1..N25 in turn, as a request does, each
waiting on a blocking event as patch_cuda_network's does; START (epoch
seconds) holds the timed loop so P copies under MPS overlap.
"""
import glob, os, sys, time
import numpy as np
import torch
import tensorrt as trt

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)
log = trt.Logger(trt.Logger.ERROR)
ITERS = range(1, int(os.environ.get("ITERS", "25")) + 1)
eng_dir = os.environ["ENGDIR"]


def build(i):
    path = f"{eng_dir}/N{i}.plan"
    if os.path.exists(path):
        return
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    with open(f"{os.environ['ONNX_DIR']}/N{i}.onnx", "rb") as f:
        assert parser.parse(f.read()), [parser.get_error(k) for k in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 2, 32, 32), (1, 2, 32, 32), (1, 2, 32, 32))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    with open(path + ".tmp", "wb") as f:
        f.write(plan)
    os.replace(path + ".tmp", path)


if os.environ.get("BUILD") == "1":
    t = time.perf_counter()
    for i in ITERS:
        build(i)
    sizes = [os.path.getsize(p) for p in glob.glob(f"{eng_dir}/N*.plan")]
    print(f"built {len(sizes)} plans in {time.perf_counter() - t:.0f}s, {sum(sizes) / 1e6:.0f}MB total")
    sys.exit()

dev = torch.device("cuda")
x = torch.randn(1, 2, 32, 32, device=dev)
y = torch.empty(1, 1, 32, 32, device=dev)
torch.cuda.synchronize()
free0 = torch.cuda.mem_get_info()[0]
kind = os.environ["KIND"]
stream = torch.cuda.Stream()
steps = []
if kind == "T":
    runtime = trt.Runtime(log)
    shared = os.environ.get("SHARED") == "1"
    engines, ctxs = [], []
    for i in ITERS:
        with open(f"{eng_dir}/N{i}.plan", "rb") as f:
            engines.append(runtime.deserialize_cuda_engine(f.read()))
        a = torch.cuda.mem_get_info()[0]
        if shared:
            ctx = engines[-1].create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        else:
            ctx = engines[-1].create_execution_context()
        ctx.set_input_shape("x", (1, 2, 32, 32))
        ctx.set_tensor_address("x", x.data_ptr())
        ctx.set_tensor_address("y", y.data_ptr())
        ctxs.append(ctx)
        steps.append((free0 - a, free0 - torch.cuda.mem_get_info()[0]))
    act = max(e.device_memory_size_v2 for e in engines)
    if shared:
        scratch = torch.empty(act, dtype=torch.uint8, device=dev)
        for ctx in ctxs:
            ctx.set_device_memory(scratch.data_ptr(), act)
    eng_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20

    def run(k):
        assert ctxs[k].execute_async_v3(stream.cuda_stream)
else:
    sys.path.insert(0, "/opt/r2d2/R2D2-RI/src")
    from utils.util_model import create_net_imaging
    net = create_net_imaging().eval().to(dev)
    weights = []
    for i in ITERS:
        ck = torch.load(f"/checkpoints/R2D2_A1/R2D2_UNet_N{i}.ckpt", map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        weights.append({'.'.join(k.split('.')[1:]): v.to(dev) for k, v in sd.items()})
        steps.append((0, free0 - torch.cuda.mem_get_info()[0]))
    act = 0
    eng_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20

    def run(k):
        net.load_state_dict(weights[k], assign=True)
        with torch.cuda.stream(stream):
            y.copy_(net(x))

n_it = len(ITERS)
for k in range(2 * n_it):
    run(k % n_it)
stream.synchronize()
peak_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20
n = int(os.environ.get("N", "1000"))
time.sleep(max(0.0, float(os.environ.get("START", "0")) - time.time()))
c, t = time.process_time(), time.perf_counter()
for k in range(n):
    run(k % n_it)
    done = torch.cuda.Event(blocking=True)
    done.record(stream)
    done.synchronize()
dt, cpu = (time.perf_counter() - t) / n, (time.process_time() - c) / n
first = f"first={steps[0][1] / 2**20:.0f}MB" if steps else ""
print(f"{dt * 1e3:.2f} ms/image cpu_ms={cpu * 1e3:.2f} kind={kind} shared={os.environ.get('SHARED', '0')} "
      f"loaded_MB={eng_mb:.0f} after_warm_MB={peak_mb:.0f} act_MB={act / 2**20:.1f} {first} "
      f"per_net_MB={eng_mb / n_it:.0f}")
