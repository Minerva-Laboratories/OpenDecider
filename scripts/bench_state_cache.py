"""Tier 1 prefix cache on the real model: agreement with the uncached path and latency for cold / repeat / append.

    .venv/bin/python scripts/bench_state_cache.py --ckpt runs/x2b/model.pt
"""
import argparse
import json
import os
import random
import time

import torch

from opendecider.checkpoint import load_decider
from opendecider.guards import gpu_lock, limit_gpu_memory

QS = {"route": {"type": "choice", "prompt": "Which team should handle the latest customer message?",
                "options": ["billing", "technical support", "refunds", "shipping", "other"]},
      "urgent": {"type": "noul", "prompt": "Does the latest message need a reply within the hour?"},
      "severity": {"type": "score", "prompt": "How severe is the customer's problem?",
                   "levels": ["low", "medium", "high", "critical"]}}
KINDS = ["asked about delivery date", "reported the app crashes on login", "was charged twice for order #{n}",
         "wants to return a damaged blender", "changed their shipping address", "complained about late refund"]


def state(n, seed=0):
    r = random.Random(seed)
    return {"customer": {"id": "C-1042", "tier": "gold", "since": "2021-03-04"},
            "events": [{"t": f"2026-09-{1 + i % 28:02d}T{i % 24:02d}:00", "from": r.choice(["customer", "agent"]),
                        "msg": "The customer " + r.choice(KINDS).format(n=1000 + i)} for i in range(n)]}


def probs(o):
    return torch.tensor([p for a in o["answers"].values() for p in a["probs_list"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--events", type=int, nargs="+", default=[20, 100, 400])
    a = ap.parse_args()
    limit_gpu_memory(float(os.environ.get("OPENDECIDER_EVAL_GPU_GB", "24")))
    out = {}
    with gpu_lock("bench_state_cache"):
        os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
        ref = load_decider(a.ckpt)
        os.environ["OPENDECIDER_STATE_CACHE_MB"] = "2048"
        from opendecider.decider import Decider
        dec = Decider(ref.model, ref.temperature, ref.temperature_by_type)
        ref.decide({"state": state(5), "questions": QS}); dec.decide({"state": state(5, 9), "questions": QS})  # warm-up
        for n in a.events:
            s0, s1 = state(n, n), state(n, n)
            s1["events"].append({"t": "2026-09-30T10:00", "from": "customer", "msg": "I want my money back now."})
            t = time.perf_counter(); r0 = ref.decide({"state": s1, "questions": QS}); t_unc = time.perf_counter() - t
            c_cold = dec.decide({"state": s0, "questions": QS})
            c_rep = dec.decide({"state": s0, "questions": QS})
            c_app = dec.decide({"state": s1, "questions": QS})
            row = {"state_tokens": r0["input_tokens"],
                   "uncached_ms": 1e3 * t_unc, "cold_ms": c_cold["timing_ms"]["total"],
                   "repeat_ms": c_rep["timing_ms"]["total"], "append_ms": c_app["timing_ms"]["total"],
                   "repeat_state_ms": c_rep["timing_ms"]["state"], "append_state_ms": c_app["timing_ms"]["state"],
                   "cold_state_ms": c_cold["timing_ms"]["state"],
                   "append_cached_tokens": c_app["state_cached_tokens"],
                   "max_abs_diff_append_vs_uncached": float((probs(c_app) - probs(r0)).abs().max()),
                   "same_argmax": all(c_app["answers"][k]["value"] == r0["answers"][k]["value"] for k in QS)}
            out[n] = row
            print(n, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    os.makedirs("runs/state-cache", exist_ok=True)
    json.dump(out, open("runs/state-cache/bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
