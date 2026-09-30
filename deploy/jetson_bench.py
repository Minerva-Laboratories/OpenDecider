"""On-device latency (no network): state encode + batched questions, p50/p95.

    python deploy/jetson_bench.py --variant v2 [--ckpt runs/x/model.pt] --state-tokens 1024 --questions 16 --k 32

Untrained heads are fine for latency. Takes the shared GPU lock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "src")
from opendecider.backbone import Backbone, BackboneConfig          # noqa: E402
from opendecider.checkpoint import load_model                       # noqa: E402
from opendecider.decider import Decider                             # noqa: E402
from opendecider.guards import gpu_lock, limit_gpu_memory          # noqa: E402
from opendecider.models import ModelConfig, build_model             # noqa: E402

WORDS = ("order refund invoice shipping delay account password login charge card plan upgrade "
         "cancel device error crash battery screen warranty return label customer agent ticket").split()


def make_request(bb, state_tokens, n_q, k, seed=0):
    import random
    rng = random.Random(seed)
    text = ""
    while len(bb.tokenize([text])[0]) < state_tokens:
        text += " ".join(rng.choice(WORDS) for _ in range(50)) + ".\n"
    ids = bb.tokenize([text])[0][:state_tokens]
    text = bb.tok.decode(ids)
    qs = {}
    for i in range(n_q):
        if i % 3 == 1:
            qs[f"q{i}"] = {"type": "noul", "prompt": f"Is aspect {i} present?"}
        elif i % 3 == 2:
            qs[f"q{i}"] = {"type": "score", "prompt": f"How severe is issue {i}?", "levels": ["low", "medium", "high", "critical"]}
        else:
            qs[f"q{i}"] = {"type": "choice", "prompt": f"Which category fits item {i}?",
                           "options": [f"{rng.choice(WORDS)} {rng.choice(WORDS)} {j}" for j in range(k)]}
    return {"state": text, "questions": qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--backbone", default="configs/backbone.yaml")
    ap.add_argument("--state-tokens", type=int, default=1024)
    ap.add_argument("--questions", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--grid", action="store_true", help="K x questions latency grid (probe battery)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    limit_gpu_memory(24)
    with gpu_lock("opendecider-bench"):
        if a.ckpt:
            model, extra = load_model(a.ckpt)
        else:
            bb = Backbone.load(BackboneConfig.from_yaml(a.backbone))
            for s in a.set:
                k, v = s.split("=", 1)
                setattr(bb.cfg, k, type(getattr(bb.cfg, k))(v) if not isinstance(getattr(bb.cfg, k), list) else [v])
            mc = ModelConfig.from_yaml(a.config) if a.config else ModelConfig(variant=a.variant)
            model = build_model(bb, mc).to(bb.device).eval()
        d = Decider(model)
        grid = [(a.k, a.questions)] if not a.grid else [(k, q) for k in (2, 8, 32, 128, 255) for q in (1, 4, 16, 64)]
        results = []
        for k, nq in grid:
            req = make_request(model.backbone, a.state_tokens, nq, k)
            for _ in range(3):
                d.decide(req)
            ts, parts = [], []
            for _ in range(a.iters):
                torch.cuda.synchronize(); t = time.perf_counter()
                r = d.decide(req)
                torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t))
                parts.append(r["timing_ms"])
            ts.sort()
            row = {"variant": model.cfg.variant, "state_tokens": a.state_tokens, "questions": nq, "k": k,
                   "p50_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * (len(ts) - 1))],
                   "state_ms_p50": statistics.median(p["state"] for p in parts),
                   "questions_ms_p50": statistics.median(p["questions"] for p in parts),
                   "input_tokens": r["input_tokens"], "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "weight_quant": model.backbone.cfg.weight_quant, "kv_quant": model.backbone.cfg.kv_quant}
            results.append(row)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
