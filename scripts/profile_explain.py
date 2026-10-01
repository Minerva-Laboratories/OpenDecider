"""Where /v1/explain spends its time: evidence, prompt prefill, per-token decode (backbone vs LM head), check.

    .venv/bin/python scripts/profile_explain.py [--ckpt runs/x2b/model.pt]
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from eval.typed_decisions import load_cases, questions_of  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.explain import Explainer  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402


def timed(t, key, fn):
    def g(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize(); t[key] = t.get(key, 0.0) + time.perf_counter() - t0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    a = ap.parse_args()
    with gpu_lock("profile_explain"):
        dec = load_decider(a.ckpt)
        ex = Explainer(dec)
        r = load_cases("test", 1)[0]
        name, instr, names, texts, gp, lab = next(x for x in questions_of(r) if len(x[2]) >= 3)
        q, state = Question(instr, texts, "choice"), json.loads(r["state"])
        ex.explain(state, q, names)                                    # warm-up
        t = {}
        ex.evidence = timed(t, "evidence (occlusion)", ex.evidence)
        ex.probs = timed(t, "decision probs + faithfulness check", ex.probs)
        bb = ex.bb
        bb.head_logits = timed(t, "LM head (full vocabulary, per token)", bb.head_logits)
        fwd = bb.forward
        calls = {"n": 0}

        def fwd_t(ids, *args, **kw):
            key = "prompt prefill" if ids.shape[1] > 1 else "decode steps (backbone forward)"
            calls["n"] += ids.shape[1] == 1
            return timed(t, key, fwd)(ids, *args, **kw)
        bb.forward = fwd_t
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = ex.explain(state, q, names)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        print(f"total {total:.2f} s, {calls['n']} decode steps, explanation: {out['explanation']!r}")
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            per = f"  ({1e3 * v / calls['n']:.0f} ms/token)" if "decode" in k or "LM head" in k else ""
            print(f"  {v:6.2f} s  {100 * v / total:4.1f}%  {k}{per}")


if __name__ == "__main__":
    main()
