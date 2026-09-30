"""Where decide() spends its time (2B, Orin). Wall-clock stages with CUDA syncs, plus a torch.profiler summary.

    .venv/bin/python scripts/profile_latency.py --ckpt runs/x2b/model.pt --events 20
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from bench_state_cache import QS, state  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402


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
    a = ap.parse_args()
    os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
    with gpu_lock("profile_latency"):
        dec = load_decider(a.ckpt)
        m, bb = dec.model, dec.model.backbone
        req = {"state": state(a.events, a.events), "questions": QS}
        for _ in range(2):
            dec.decide(req)                                              # warm-up
        st = Stage()
        st.wrap(m, "_state_prefix", "state pass (backbone prefill)")
        st.wrap(m, "_rows_forward", "option rows (backbone, one row per option)")
        st.wrap(m, "_token_logprob", "LM log-prob feature (tied head over vocab)")
        st.wrap(m, "_mix_wide", "layer combine (DepthAttn)")
        st.wrap(bb, "tokenize", "tokenizer")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = dec.decide(req)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        print(f"state tokens {o['input_tokens']}, questions {len(QS)}, "
              f"rows {sum(len(q.get('options', q.get('levels', [1, 2]))) + 1 for q in QS.values())}, total {1e3 * total:.0f} ms")
        acc = 0.0
        for k, v in sorted(st.t.items(), key=lambda kv: -kv[1]):
            print(f"  {1e3 * v:7.0f} ms  {100 * v / total:4.1f}%  {k}")
            acc += v
        print(f"  {1e3 * (total - acc):7.0f} ms  {100 * (total - acc) / total:4.1f}%  rest (trunk, heads, python)")
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
