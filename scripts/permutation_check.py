"""Where the 2B's small option-order sensitivity comes from: the permutation probe with shared-prefix attention on and
off (same items, same seed). If it vanishes with the per-row path, it is float summation order in the fused attention
call, not the model reading option order.

    .venv/bin/python scripts/permutation_check.py --ckpt runs/x2b-clean3/model.pt --out runs/permutation_check_2b.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from eval.evaluate import gpu_context, load_scorer  # noqa: E402
from eval.probes import model_decide_fn, permutation_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", default="runs/probe_items.jsonl")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    res = {}
    with gpu_context("cuda", "permutation_check"):
        model, temp = load_scorer(a.ckpt, None)
        fn = model_decide_fn(model, temp, 1)
        for shared in (True, False):
            model.cfg.v3_shared_prefix = shared
            tv = [permutation_probe(fn, it, n_perm=a.n_perm, seed=0) for it in items]
            res[f"shared_prefix={shared}"] = {"mean_tv": float(np.mean([t["mean_tv"] for t in tv])),
                                              "max_tv": float(np.max([t["max_tv"] for t in tv]))}
            print(res, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
