"""O7 (run-to-run variance) probe: repeat identical calls with the stochastic sampler off/on.

    .venv/bin/python scripts/o7_probe.py runs/mix2-v3branch-vera/model.pt
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from eval.probes import model_decide_fn, repeat_probe  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock, limit_gpu_memory  # noqa: E402

limit_gpu_memory(24)
items = [json.loads(l) for l in open("runs/probe_items.jsonl")][::5][:12]
out = {}
with gpu_lock("o7-probe"):
    model, extra = load_model(sys.argv[1])
    for mode, k in [("off", 1), ("mc_dropout", 4), ("mc_dropout", 8), ("gaussian_noise", 4)]:
        fn = model_decide_fn(model, extra.get("temperature", 1.0), sampler_k=k, sampler_mode=mode)
        res = [repeat_probe(fn, it, n=20) for it in items]
        mad = [r["max_abs_diff"] for r in res]
        std = [r.get("mean_std", float("nan")) for r in res]
        out[f"{mode}_k{k}"] = {"max_abs_diff_max": float(np.max(mad)), "max_abs_diff_median": float(np.median(mad)),
                               "classes": sorted({r["class"] for r in res}), "mean_std": float(np.nanmean(std))}
        print(f"{mode:15s} k={k}: max|Δp| median {np.median(mad):.4f}, max {np.max(mad):.4f}, "
              f"per-option std {np.nanmean(std):.4f}, classes {sorted({r['class'] for r in res})}", flush=True)
json.dump(out, open("runs/o7/o7_probe.json", "w"), indent=1)
