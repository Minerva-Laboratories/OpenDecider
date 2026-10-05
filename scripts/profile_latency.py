"""Where decide() spends its time (2B, Orin). Wall-clock stages with CUDA syncs, plus a torch.profiler summary.

    .venv/bin/python scripts/profile_latency.py --ckpt runs/x2b/model.pt --events 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from bench_state_cache import QS, state  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402


POOL = ["billing", "technical support", "refunds", "shipping", "account access", "fraud", "cancellation", "upgrade",
        "downgrade", "invoice copy", "address change", "delivery delay", "damaged item", "wrong item", "warranty",
        "password reset", "data export", "privacy request", "complaint", "feedback", "partnership", "press",
        "legal", "security report", "outage", "bug report", "feature request", "pricing", "discount", "loyalty points",
        "gift card", "other"]


def workload(n: int, k: int) -> dict:
    """n questions cycling choice (k options), yes/no and score, with varied prompts."""
    qs = {}
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        if kind == "choice":
            qs[f"q{i}"] = {"type": "choice", "prompt": f"Question {i}: which category fits the latest message best?",
                           "options": POOL[:k]}
        elif kind == "noul":
            qs[f"q{i}"] = {"type": "noul", "prompt": f"Question {i}: does the latest message need action today?"}
        else:
            qs[f"q{i}"] = {"type": "score", "prompt": f"Question {i}: how severe is the customer's problem?",
                           "levels": ["low", "medium", "high", "critical"]}
    return qs


class Stage:
    """Accumulate CUDA-synced wall time per named stage by wrapping methods."""
    def __init__(self):
        self.t = {}

    def wrap(self, obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                torch.cuda.synchronize(); self.t[label] = self.t.get(label, 0.0) + time.perf_counter() - t0
        setattr(obj, name, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--events", type=int, default=20)
    ap.add_argument("--trace", default="")
    ap.add_argument("--questions", type=int, default=0, help="0: the 3-question default workload")
    ap.add_argument("--options", type=int, default=32, help="options per choice question in the --questions workload")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--qcache", action="store_true", help="question cache: each question's prefix read once (v3)")
    a = ap.parse_args()
    os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
    with gpu_lock("profile_latency"):
        dec = load_decider(a.ckpt)
        m, bb = dec.model, dec.model.backbone
        if a.qcache:
            m.cfg.v3_question_cache = True
        qs = workload(a.questions, a.options) if a.questions else QS
        req = {"state": state(a.events, a.events), "questions": qs}
        for _ in range(2):
            dec.decide(req)                                              # warm-up
        st = Stage()
        st.wrap(m, "_state_prefix", "state pass (backbone prefill)")
        st.wrap(m, "_rows_forward", "option rows (backbone, one row per option)")
        st.wrap(m, "_token_logprob", "LM log-prob feature (tied head over vocab)")
        st.wrap(m, "_mix_wide", "layer combine (DepthAttn)")
        st.wrap(bb, "tokenize", "tokenizer")
        if a.qcache:                                   # finer stages of the question-cache path
            from opendecider.backbone import PackedCache
            st.wrap(PackedCache, "pack", "qcache: pack question caches")
            st.wrap(PackedCache, "unpack_rows", "cache unpack per row (copies)")
            st.wrap(bb, "forward", "backbone forward calls (question + row passes)")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = dec.decide(req)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        print(f"state tokens {o['input_tokens']}, questions {len(qs)}, "
              f"rows {sum(len(q.get('options', q.get('levels', [1, 2]))) + 1 for q in qs.values())}, total {1e3 * total:.0f} ms")
        acc = 0.0
        for k, v in sorted(st.t.items(), key=lambda kv: -kv[1]):
            print(f"  {1e3 * v:7.0f} ms  {100 * v / total:4.1f}%  {k}")
            acc += v
        print(f"  {1e3 * (total - acc):7.0f} ms  {100 * (total - acc) / total:4.1f}%  rest (trunk, heads, python)")
        times = []
        for _ in range(a.repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            dec.decide(req)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        p50, p95 = float(np.percentile(times, 50)), float(np.percentile(times, 95))
        print(f"p50 {1e3 * p50:.0f} ms, p95 {1e3 * p95:.0f} ms over {a.repeats} requests")
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump({"ckpt": a.ckpt, "state_tokens": o["input_tokens"], "questions": len(qs), "options": a.options,
                       "p50_ms": 1e3 * p50, "p95_ms": 1e3 * p95, "stages_ms": {k: 1e3 * v for k, v in st.t.items()},
                       "total_ms": 1e3 * total}, open(a.out, "w"), indent=1)
        # CPU-side launch overhead vs GPU time
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            dec.decide(req)
        ev = prof.key_averages()
        cuda_us = sum(e.self_device_time_total for e in ev)
        n_kernels = sum(e.count for e in ev if e.self_device_time_total > 0)
        print(f"GPU busy (sum of kernel time) {cuda_us / 1e3:.0f} ms over {n_kernels} kernel launches")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=12))
        if a.trace:
            prof.export_chrome_trace(a.trace)


if __name__ == "__main__":
    main()
