"""SmoothQuant scales for the a8w8 path (weight_quant=a8w8sq): per-channel max |activation| of every backbone linear,
recorded with bf16 weights on TRAINING-distribution states (data/synthetic/calib.jsonl: synthetic train families,
never a benchmark), then s_j = max|X_j|^alpha / max|W_:,j|^(1-alpha) (Xiao et al. 2023, arXiv:2211.10438).

    .venv/bin/python scripts/smoothquant_calibrate.py --ckpt runs/x9b-clean3/model.pt --alphas 0.5 0.65 0.8 \
        --out runs/deploy/smooth_9b
Writes <out>_a<alpha>.pt per alpha ({"scales": {name: tensor}, "alpha", "n_records", "ckpt"}). Pick alpha on the
synthetic validation split (data/synthetic/val.jsonl, --select) before touching any benchmark.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.decider import Decider  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.quant import smooth_scales  # noqa: E402


def records(path, n, seed=0):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = np.random.default_rng(seed)
    return [rows[i] for i in rng.permutation(len(rows))[:n]]


def request(r):
    qs = {}
    for i, q in enumerate(r["questions"]):
        if len(q["options"]) > 255:                       # two-stage path: not what the scales are about
            continue
        spec = {"type": q["type"], "prompt": q["prompt"], "none": False}
        if q["type"] == "choice":
            spec["options"] = q["options"]
        elif q["type"] == "score":
            spec["levels"] = q["options"]
        qs[f"q{i}"] = spec
    return {"state": r["state"], "questions": qs}


def accuracy(dec, rows):
    hit = []
    for r in rows:
        out = dec.decide(request(r))["answers"]
        for i, q in enumerate(r["questions"]):
            if f"q{i}" not in out:
                continue
            hit.append(float(int(np.argmax(out[f"q{i}"]["probs_list"])) == q["label"]))
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", default="data/synthetic/calib.jsonl")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.65, 0.8])
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default="data/synthetic/val.jsonl", help="pick alpha by accuracy here ('' = skip)")
    ap.add_argument("--n-select", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gpu_lock("smoothquant"):
        model, _ = load_model(a.ckpt, overrides={"weight_quant": "none"})
        layers = model.backbone.lm.layers
        amax, hooks = {}, []
        for name, mod in layers.named_modules():
            if isinstance(mod, nn.Linear):
                def hook(m, args, name=name):
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).abs().amax(0).float()
                    amax[name] = x if name not in amax else torch.maximum(amax[name], x)
                hooks.append(mod.register_forward_pre_hook(hook))
        dec = Decider(model)
        rows = records(a.calib, a.n)
        with torch.no_grad():
            for r in rows:
                dec.decide(request(r))
        for h in hooks:
            h.remove()
        W = {name: mod.weight.data for name, mod in layers.named_modules() if isinstance(mod, nn.Linear)}
        paths = []
        for al in a.alphas:
            sc = {n: smooth_scales(amax[n], W[n], al).cpu() for n in amax}
            p = f"{a.out}_a{al}.pt"
            torch.save({"scales": sc, "alpha": al, "n_records": len(rows), "ckpt": a.ckpt, "calib": a.calib}, p)
            paths.append((al, p))
            print(f"alpha {al}: {len(sc)} layers -> {p}", flush=True)
        del model, dec
        torch.cuda.empty_cache()
        if not a.select:
            return
        sel = records(a.select, a.n_select, seed=1)
        res = {}
        for tag, over in [("a8w8", {"weight_quant": "a8w8"})] + [(f"a8w8sq_a{al}", {"weight_quant": "a8w8sq",
                                                                                     "smooth_path": p})
                                                                    for al, p in paths] + [("int8", {})]:
            m, _ = load_model(a.ckpt, overrides=over)
            res[tag] = accuracy(Decider(m), sel)
            print(f"select {tag}: synthetic val accuracy {res[tag]:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        json.dump(res, open(f"{a.out}_select.json", "w"), indent=1)


if __name__ == "__main__":
    main()
