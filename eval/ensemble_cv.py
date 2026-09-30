"""Ensemble OpenDecider with a zero-shot LLM, out of sample (2-fold cross-fit, same protocol as eval.profile_cv).

    .venv/bin/python -m eval.ensemble_cv --a runs/preds-x2b --b runs/zs-9b [--name ens-x2b-9b]

Per benchmark, items are joined by id. Fitted on one half, applied to the other, then swapped:
    temperature-only on each source      p = softmax(z / T)
    ensemble                             p = softmax(w_a * log p_a + w_b * log p_b + c)
(weights and bias fitted by NLL; c is a per-option-name bias only where every item shares the same option names).
Metrics with 95% bootstrap CIs on the pooled out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from .metrics import summarize


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        z = torch.tensor(r["logits"], dtype=torch.float64)
        out[r["id"]] = (torch.log_softmax(z, 0), int(r["label"]), r.get("options"))
    return out


def fit(la, lb, y, use_b=True, bias=False, iters=300):
    K = la.shape[1]
    wa = torch.ones((), dtype=torch.float64, requires_grad=True)
    wb = torch.tensor(1.0 if use_b else 0.0, dtype=torch.float64, requires_grad=use_b)
    c = torch.zeros(K, dtype=torch.float64, requires_grad=bias)
    params = [wa] + ([wb] if use_b else []) + ([c] if bias else [])
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = wa * la + (wb * lb if use_b else 0) + c
        loss = torch.nn.functional.cross_entropy(z, y) + 1e-2 * (c ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return wa.detach(), wb.detach(), c.detach()


def cross_fit(la, lb, y, use_b, bias, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = [torch.tensor(idx[: len(y) // 2]), torch.tensor(idx[len(y) // 2:])]
    P = torch.empty_like(la)
    for tr, te in ((folds[0], folds[1]), (folds[1], folds[0])):
        wa, wb, c = fit(la[tr], lb[tr], y[tr], use_b, bias)
        P[te] = torch.softmax(wa * la[te] + (wb * lb[te] if use_b else 0) + c, -1)
    return P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="dir with preds_<split>.jsonl (e.g. OpenDecider)")
    ap.add_argument("--b", required=True, help="dir with preds_<split>.jsonl (e.g. zero-shot LLM)")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    res = {}
    for fb in sorted(glob.glob(os.path.join(a.b, "preds_*.jsonl"))):
        split = os.path.basename(fb)[6:-6]
        fa = os.path.join(a.a, f"preds_{split}.jsonl")
        if not os.path.exists(fa):
            continue
        A, B = load(fa), load(fb)
        ids = sorted(set(A) & set(B))
        ids = [i for i in ids if A[i][0].shape == B[i][0].shape and A[i][1] == B[i][1]]
        la = torch.stack([A[i][0] for i in ids]); lb = torch.stack([B[i][0] for i in ids])
        y = torch.tensor([A[i][1] for i in ids])
        opts = [B[i][2] for i in ids]
        bias = all(o is not None for o in opts) and len({tuple(o) for o in opts}) == 1
        yl = y.tolist()
        S = lambda P: summarize([p.numpy() for p in P], yl, n_boot=1000)
        out = {"A raw": S(torch.softmax(la, -1)), "B raw": S(torch.softmax(lb, -1)),
               "A temp": S(cross_fit(la, lb, y, False, False)),
               "B temp": S(cross_fit(lb, la, y, False, False)),
               "ensemble": S(cross_fit(la, lb, y, True, bias))}
        res[split] = {"n": len(ids), **out}
        print(f"== {split} (n={len(ids)}{', +bias' if bias else ''})")
        for k, m in out.items():
            print(f"   {k:9s} acc {m['accuracy']['value']:.3f} [{m['accuracy']['lo']:.3f},{m['accuracy']['hi']:.3f}]  "
                  f"nll {m['nll']['value']:.3f}  ece {m['ece']['value']:.3f}")
    if a.name:
        os.makedirs(os.path.join("runs", a.name), exist_ok=True)
        json.dump(res, open(os.path.join("runs", a.name, "ensemble_cv.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
