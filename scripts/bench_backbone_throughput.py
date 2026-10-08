"""Frozen-backbone throughput for training: tokens/s of the state pass and of a no-grad full forward, for a backbone
config, with CUDA-synchronized wall time (CUPTI is not available on this board, so the profiler cannot see kernels).
    .venv/bin/python scripts/bench_backbone_throughput.py --backbone configs/backbone_4b_nf4.yaml"""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opendecider.backbone import Backbone, BackboneConfig
from opendecider.guards import gpu_lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="BackboneConfig overrides k=v")
    a = ap.parse_args()
    with gpu_lock("bench_backbone"):
        cfg = BackboneConfig.from_yaml(a.backbone)
        for kv in a.set:
            k, v = kv.split("="); setattr(cfg, k, v)
        bb = Backbone.load(cfg)
        torch.manual_seed(0)
        for B, T in [(16, 300), (64, 40), (1, 2048)]:
            ids = torch.randint(1000, 50000, (B, T), device="cuda"); mask = torch.ones_like(ids, dtype=torch.bool)
            with torch.no_grad():
                bb.forward(ids, mask)                                        # warm-up
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(3):
                    bb.forward(ids, mask)
                torch.cuda.synchronize(); dt = (time.time() - t0) / 3
            print(f"{cfg.weight_quant:10s} batch {B:3d} x {T:5d} tokens: {dt:6.3f} s/pass, {B * T / dt:8.0f} tokens/s", flush=True)


if __name__ == "__main__":
    main()
