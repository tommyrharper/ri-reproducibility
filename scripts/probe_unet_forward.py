"""Time R2D2's U-Net forward alone, 32x32, one checkpoint (docs/csd3-speed.md, round 11).

Inside r2d2.sif with the checkpoints at /checkpoints:
    python3 scripts/probe_unet_forward.py
B batches B images per forward (prints ms per image). DIM, N (forwards),
T (torch threads) and DEV (cpu or cuda; csd3-speed.md, round 117) as named;
START (epoch seconds) holds the timed loop until then, so P copies overlap.
GRAPH=1 (cuda only) replays the forward from a captured CUDA graph (round 122). Run P copies at once to see what a full node does:
    for i in $(seq 24); do apptainer exec ... python3 probe_unet_forward.py & done; wait
"""
import os, sys, time
import torch
sys.path.insert(0, "/opt/r2d2/R2D2-RI/src")
from utils.util_model import create_net_imaging

torch.backends.mkldnn.deterministic = True  # as r2d2_serve.warm_imports
torch.set_num_threads(int(os.environ.get("T", "1")))
net = create_net_imaging()
ck = torch.load("/checkpoints/R2D2_A1/R2D2_UNet_N1.ckpt", map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
net.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in sd.items()}, assign=True)
dev = torch.device(os.environ.get("DEV", "cpu"))
net = net.eval().to(dev)
b, dim, n = int(os.environ.get("B", "1")), int(os.environ.get("DIM", "32")), int(os.environ.get("N", "200"))
torch.manual_seed(0)
x = torch.randn(b, 2, dim, dim).to(dev)
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
with torch.no_grad():
    ref = torch.cat([net(x[i:i + 1]) for i in range(b)])
    y = net(x)
    for _ in range(5):
        net(x)
    sync()
    run = net
    if os.environ.get("GRAPH", "0") == "1":
        static_x = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                net(static_x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = net(static_x)
        def run(inp):
            static_x.copy_(inp)
            graph.replay()
            return static_y
        y_graph = run(x).clone()
        sync()
        graph_identical = torch.equal(y_graph, y)
    time.sleep(max(0.0, float(os.environ.get("START", "0")) - time.time()))
    c = time.process_time()
    time.sleep(max(0.0, float(os.environ.get("START", "0")) - time.time()))
    t = time.perf_counter()
    for _ in range(n):
        run(x)
    sync()
    dt = (time.perf_counter() - t) / n
    cpu = (time.process_time() - c) / n
print(f"{dt / b * 1e3:.2f} ms/image B={b} params={sum(p.numel() for p in net.parameters()) * 4 / 1e6:.0f}MB "
      f"batch_identical={torch.equal(y, ref)} cpu_ms={cpu * 1e3:.2f} "
      f"graph_identical={graph_identical if os.environ.get('GRAPH', '0') == '1' else '-'} maxrel={float((y - ref).abs().max() / ref.abs().max()):.2g}")
