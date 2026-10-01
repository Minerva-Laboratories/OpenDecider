"""Where an explanation's time goes: evidence (occlusion pass), decoding, faithfulness check.

    .venv/bin/python scripts/time_explain.py --ckpt runs/x2b/model.pt --weight-quant w8 --n 6
"""
import argparse
import json
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from eval.typed_decisions import load_cases, questions_of  # noqa: E402
from opendecider.backbone import Backbone, BackboneConfig  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.decider import Decider  # noqa: E402
from opendecider.explain import Explainer, _records  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--weight-quant", default="w8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load_cases("test"); rng = random.Random(1); by = {}
    for r in rows:
        by.setdefault(r["workflow"], []).append(r)
    pick = [r for wf in sorted(by) for r in rng.sample(by[wf], max(1, a.n // len(by)))]
    with gpu_lock("time_explain"):
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        bb = Backbone.load(BackboneConfig(**{**ck["backbone_cfg"], "weight_quant": a.weight_quant}))
        m, extra = load_model(a.ckpt, backbone=bb)
        ex = Explainer(Decider(m, extra.get("temperature", 1.0), extra.get("temperature_by_type")))
        sync = torch.cuda.synchronize
        t = {}

        def timed(name, f):                      # wrap an Explainer method; accumulate its synced wall time
            def g(*args, **kw):
                sync(); s = time.perf_counter(); out = f(*args, **kw); sync()
                t[name] = t.get(name, 0.0) + time.perf_counter() - s
                if name == "decode":
                    t["prompt_tokens"] = len(bb.tokenize([args[0]])[0])
                return out
            return g
        ex.evidence, ex.generate = timed("evidence", ex.evidence), timed("decode", ex.generate)
        probs = ex.probs
        ex.probs = timed("check", probs)          # also counts the occlusion pass inside evidence: subtracted below
        cold, warm = [], []
        for r in pick:
            name, instr, names, texts, *_ = next(x for x in questions_of(r) if len(x[2]) >= 3)
            q, state = Question(instr, texts, "choice"), json.loads(r["state"])
            for rep, dst in ((0, cold), (1, warm)):          # second pass: kernels autotuned for these shapes
                t.clear()
                sync(); s = time.perf_counter()
                out = ex.explain(state, q, names, n_samples=1); sync()
                t["total"] = time.perf_counter() - s
                t["check"] -= t["evidence"]                   # evidence's own probs() call
                t["new_tokens"] = len(bb.tokenize([out["explanation"] or ""])[0])
                t["state_tokens"] = len(bb.tokenize([json.dumps(state)])[0])
                t["records"] = len(_records(state))
                dst.append(dict(t))
                print(("cold " if rep == 0 else "warm ") + json.dumps({k: round(v, 2) for k, v in t.items()}),
                      flush=True)
    res = {}
    for label, rec in (("cold", cold), ("warm", warm)):
        res[label] = {k: float(np.median([x[k] for x in rec])) for k in rec[0]}
        print("MEDIAN", label, json.dumps({k: round(v, 2) for k, v in res[label].items()}))
    if a.out:
        json.dump({"median": res, "cold": cold, "warm": warm}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
