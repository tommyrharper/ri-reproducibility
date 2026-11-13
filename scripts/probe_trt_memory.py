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

PREC=fp16 (round 175) builds fp16 engines instead: TensorRT 11 has no FP16
builder flag, only strongly typed networks, so each checkpoint is exported
afresh as a half-precision U-Net wrapped in float32 casts (input and output
stay float32) into ENGDIR/N<i>.onnx. ACC=1 then prints, per net, the max
abs difference of the engines in ENGDIR from torch eager float32 on the card
(TF32 off), relative to the eager output's peak, for inputs at three scales.
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


fp16 = os.environ.get("PREC") == "fp16"


def eager_net(i, dev):
    sys.path.insert(0, "/opt/r2d2/R2D2-RI/src")
    from utils.util_model import create_net_imaging
    net = create_net_imaging().eval()
    ck = torch.load(f"/checkpoints/R2D2_A1/R2D2_UNet_N{i}.ckpt", map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    net.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in sd.items()})
    return net.to(dev)


class Half(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, x):
        return self.net(x.half()).float()


def build(i):
    path = f"{eng_dir}/N{i}.plan"
    if os.path.exists(path):
        return
    onnx_path = f"{os.environ['ONNX_DIR']}/N{i}.onnx"
    if fp16:
        onnx_path = f"{eng_dir}/N{i}.onnx"
        torch.onnx.export(Half(eager_net(i, "cuda")), (torch.zeros(1, 2, 32, 32, device="cuda"),), onnx_path,
                          dynamo=False, opset_version=17, input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if fp16 else 0)
    parser = trt.OnnxParser(network, log)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), [parser.get_error(k) for k in range(parser.num_errors)]
    config = builder.create_builder_config()
    if not fp16:
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
if os.environ.get("ACC") == "1":
    runtime = trt.Runtime(log)
    gen = torch.Generator(device=dev).manual_seed(175)
    worst = 0.0
    for i in ITERS:
        net = eager_net(i, dev)
        with open(f"{eng_dir}/N{i}.plan", "rb") as f:
            eng = runtime.deserialize_cuda_engine(f.read())
        ctx = eng.create_execution_context()
        ctx.set_input_shape("x", (1, 2, 32, 32))
        row = []
        for scale in (1e-3, 1.0, 1e3):
            x = torch.randn(1, 2, 32, 32, device=dev, generator=gen) * scale
            y = torch.empty(1, 1, 32, 32, device=dev)
            ctx.set_tensor_address("x", x.data_ptr())
            ctx.set_tensor_address("y", y.data_ptr())
            assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            ref = net(x)
            torch.cuda.synchronize()
            rel = ((y - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, rel)
            row.append(f"{scale:g}:{rel:.1e}")
        print(f"acc N{i} " + " ".join(row))
    print(f"acc worst maxrel={worst:.2e} engdir={eng_dir}")
    sys.exit()
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
