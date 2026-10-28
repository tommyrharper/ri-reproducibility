"""One process, 25 shared TensorRT engines, K execution contexts each (docs/csd3-speed.md, round 207).

Round 181 closed per-worker refits; round 170 showed 25 engines cost 5.1 GB a
worker, which caps an A100 at 12 TensorRT workers. TensorRT shares an engine's
weights between the execution contexts of one process, and a context's
activations are ~1.5 MB, so a forward server holding the 25 engines once could
serve every rank. This times that server.

MODE=threads  K threads, each with its own 25 contexts, stream and buffers,
              walking N1..N25 with a blocking sync per forward (upper bound)
MODE=pipes    the same, but each thread serves a forked CPU-only client over a
              pipe pair: client writes x (2x32x32 fp32), server copies it in,
              runs the net, copies y back and replies (a rank's round trip)
Both print aggregate forwards/s, per-forward latency, device memory, and check
that every thread's output equals one reference context's bit for bit.
"""
import os, sys, threading, time
import numpy as np

K = int(os.environ.get("K", "12"))
N = int(os.environ.get("N", "2500"))
MODE = os.environ.get("MODE", "threads")
NETS = 25
XB, YB = 2 * 32 * 32 * 4, 32 * 32 * 4


def client(rfd, wfd, seed):
    rng = np.random.default_rng(seed)
    xs = [rng.standard_normal((2, 32, 32), dtype=np.float32).tobytes() for _ in range(NETS)]
    for k in range(N + 2 * NETS):
        os.write(wfd, xs[k % NETS])
        got = b""
        while len(got) < YB:
            got += os.read(rfd, YB - len(got))
    os._exit(0)


pipes = []
if MODE == "pipes":  # fork clients before CUDA exists in this process
    for t in range(K):
        a, b = os.pipe()  # server -> client
        c, d = os.pipe()  # client -> server
        if os.fork() == 0:
            client(a, d, t)
        pipes.append((c, b))

import torch
import tensorrt as trt

log = trt.Logger(trt.Logger.ERROR)
dev = torch.device("cuda")
torch.cuda.init()
free0 = torch.cuda.mem_get_info()[0]
runtime = trt.Runtime(log)
engines = []
for i in range(1, NETS + 1):
    with open(f"{os.environ['ENGDIR']}/N{i}.plan", "rb") as f:
        engines.append(runtime.deserialize_cuda_engine(f.read()))
eng_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20
act = max(e.device_memory_size_v2 for e in engines)


class Lane:
    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.x = torch.empty(1, 2, 32, 32, device=dev)
        self.y = torch.empty(1, 1, 32, 32, device=dev)
        self.hx = torch.empty(1, 2, 32, 32, pin_memory=True)
        self.hy = torch.empty(1, 1, 32, 32, pin_memory=True)
        self.scratch = torch.empty(act, dtype=torch.uint8, device=dev)
        self.ctxs = []
        for e in engines:
            c = e.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            c.set_input_shape("x", (1, 2, 32, 32))
            c.set_tensor_address("x", self.x.data_ptr())
            c.set_tensor_address("y", self.y.data_ptr())
            c.set_device_memory(self.scratch.data_ptr(), act)
            self.ctxs.append(c)

    def forward(self, k):
        assert self.ctxs[k].execute_async_v3(self.stream.cuda_stream)
        done = torch.cuda.Event(blocking=True)
        done.record(self.stream)
        done.synchronize()


lanes = [Lane() for _ in range(K)]
torch.cuda.synchronize()
lane_mb = (free0 - torch.cuda.mem_get_info()[0]) / 2**20 - eng_mb

# bit-identity: every lane's output for net k equals lane 0's
gen = torch.Generator(device=dev).manual_seed(207)
probe = torch.randn(NETS, 1, 2, 32, 32, device=dev, generator=gen)
ref = []
for k in range(NETS):
    lanes[0].x.copy_(probe[k]); torch.cuda.synchronize(); lanes[0].forward(k); ref.append(lanes[0].y.clone())
mism = 0
for ln in lanes[1:]:
    for k in range(NETS):
        ln.x.copy_(probe[k]); torch.cuda.synchronize(); ln.forward(k)
        mism += int(not torch.equal(ln.y, ref[k]))

lat = [0.0] * K
start = threading.Barrier(K + 1)


def serve(t):
    ln = lanes[t]
    if MODE == "pipes":
        rfd, wfd = pipes[t]
        hxb = ln.hx.numpy().view(np.uint8).reshape(-1)
        hyb = ln.hy.numpy().view(np.uint8).reshape(-1)
    for k in range(2 * NETS):  # warm, untimed
        ln.forward(k % NETS)
    start.wait()
    t0 = time.perf_counter()
    for k in range(N + (2 * NETS if MODE == "pipes" else 0)):
        if MODE == "pipes":
            got = 0
            while got < XB:
                got += os.readv(rfd, [memoryview(hxb)[got:]])
            with torch.cuda.stream(ln.stream):
                ln.x.copy_(ln.hx, non_blocking=True)
                ln.forward(k % NETS)  # syncs the stream
                ln.hy.copy_(ln.y, non_blocking=True)
            ln.stream.synchronize()
            os.write(wfd, hyb.tobytes())
        else:
            ln.forward(k % NETS)
    lat[t] = (time.perf_counter() - t0) / N


ths = [threading.Thread(target=serve, args=(t,)) for t in range(K)]
for th in ths:
    th.start()
start.wait()
c0, w0 = time.process_time(), time.perf_counter()
for th in ths:
    th.join()
wall, cpu = time.perf_counter() - w0, time.process_time() - c0
n_tot = K * N if MODE == "threads" else K * (N + 2 * NETS)
print(f"== shared mode={MODE} K={K} agg_forwards_per_s={n_tot / wall:.0f} mean_ms_per_forward={1e3 * sum(lat) / K:.2f} "
      f"worst_ms={1e3 * max(lat):.2f} server_cpu_s_per_s={cpu / wall:.2f} engines_MB={eng_mb:.0f} lanes_MB={lane_mb:.0f} "
      f"act_MB={act / 2**20:.1f} bit_mismatches={mism}")
if MODE == "pipes":
    for _ in range(K):
        os.wait()
