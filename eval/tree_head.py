"""Tree-based decision head over per-option signals (stacking), vs linear log-pooling. Out of sample, 2-fold cross-fit
grouped by question (every option of a question lands in the same fold).

    .venv/bin/python -m eval.tree_head --sources runs/preds-x2b runs/public-x9b-stitch runs/zs-9b   [--typed]

Per option features, for each source s: log p_s, rank_s, gap to the question's best (log p_s - max log p_s),
the question's entropy under s, and K (number of options). Nothing depends on option order or wording, so the head
stays class-agnostic (any option set, any question).
Heads:
  linear  p ∝ exp(sum_s w_s log p_s)                    (weights fitted by log loss)
  gbdt    LightGBM gradient-boosted trees, per-option score -> softmax over the question's options
  forest  LightGBM random forest (bagged trees), per-option probability -> normalised over the question
Targets: the gold label (public benchmarks) or the gold distribution (typed-decisions, --typed: soft cross-entropy).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def load_dir(d, typed):
    files = glob.glob(os.path.join(d, "preds_typed_decisions.jsonl" if typed else "preds_*.jsonl"))
    out = {}
    for f in files:
        split = "typed_decisions" if typed else os.path.basename(f)[6:-6]
        for l in open(f):
            r = json.loads(l)
            z = np.array(r["logits"], dtype=np.float64)
            r["lp"] = z - np.logaddexp.reduce(z)
            out[(split, r["id"])] = r
    return out


def features(lps):
    """lps: list over sources of (K,) log-probs -> (K, F) features."""
    K = len(lps[0])
    cols = []
    for lp in lps:
        p = np.exp(lp)
        ent = -(p * lp).sum()
        rank = np.argsort(np.argsort(-lp))
        cols += [lp, rank.astype(float), lp - lp.max(), np.full(K, ent)]
    cols.append(np.full(K, float(K)))
    return np.stack(cols, 1)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def fit_linear(train):
    S = len(train[0][0])
    w = torch.zeros(S, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    data = [(torch.tensor(np.stack(lps)), torch.tensor(t)) for lps, t in train]

    def closure():
        opt.zero_grad()
        loss = sum(-(t * torch.log_softmax((torch.exp(w)[:, None] * L).sum(0), -1)).sum() for L, t in data) / len(data)
        loss.backward()
        return loss
    opt.step(closure)
    ww = torch.exp(w).detach().numpy()
    return lambda lps: softmax((ww[:, None] * np.stack(lps)).sum(0))


def fit_tree(train, mode):
    import lightgbm as lgb
    X = np.concatenate([features(lps) for lps, _ in train])
    y = np.concatenate([t for _, t in train])
    params = {"objective": "cross_entropy", "learning_rate": 0.05, "num_leaves": 8, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "verbose": -1, "seed": 0}
    if mode == "forest":
        params.update(boosting="rf", bagging_fraction=0.7, bagging_freq=1, num_leaves=16)
    m = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300 if mode == "gbdt" else 200)
    if mode == "gbdt":
        score = lambda lps: m.predict(features(lps), raw_score=True)
    else:
        score = lambda lps: np.log(np.clip(m.predict(features(lps)), 1e-6, 1))
    # boosted/bagged trees are poorly calibrated as is: one temperature over the per-question softmax, fitted on the
    # training fold (the same fix the calibration profiles apply to the neural model)
    Z = [(torch.tensor(score(lps)), torch.tensor(t)) for lps, t in train]
    logT = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sum(-(t * torch.log_softmax(z / torch.exp(logT), -1)).sum() for z, t in Z) / len(Z)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(logT).detach())
    return lambda lps: softmax(score(lps) / T)


def metrics(P, T, n_boot=1000, seed=0):
    """P: predicted distributions; T: target distributions (one-hot or soft). acc vs argmax(T)."""
    eps = 1e-9
    acc = np.array([float(np.argmax(p) == np.argmax(t)) for p, t in zip(P, T)])
    ll = np.array([float(-(t * np.log(p + eps)).sum()) for p, t in zip(P, T)])
    brier = np.array([float(((p - t) ** 2).sum()) for p, t in zip(P, T)])
    conf = np.array([p.max() for p in P])

    def ece(ix):
        o = ix[np.argsort(conf[ix])]
        return float(sum(len(b) / len(o) * abs(acc[b].mean() - conf[b].mean()) for b in np.array_split(o, 15) if len(b)))
    n = len(P); rng = np.random.default_rng(seed)
    pt = {"acc": acc.mean(), "log_loss": ll.mean(), "brier": brier.mean(), "ece": ece(np.arange(n))}
    bs = {k: [] for k in pt}
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        bs["acc"].append(acc[ix].mean()); bs["log_loss"].append(ll[ix].mean()); bs["brier"].append(brier[ix].mean()); bs["ece"].append(ece(ix))
    return {k: (float(v), float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))) for k, v in pt.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--typed", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    src = [load_dir(d, a.typed) for d in a.sources]
    keys = sorted(set.intersection(*(set(s) for s in src)))
    res = {}
    for split in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == split and all(len(s[k]["lp"]) == len(src[0][k]["lp"]) for s in src)]
        items = []
        for k in ks:
            r0 = src[0][k]
            t = np.array(r0["gold"]) if a.typed else np.eye(len(r0["lp"]))[r0["label"]]
            items.append(([s[k]["lp"] for s in src], t, k[1].split(":")[0]))
        groups = sorted({g for _, _, g in items})
        rng = np.random.default_rng(0); perm = rng.permutation(len(groups))
        fold_of = {groups[i]: int(j % 2) for j, i in enumerate(perm)}
        heads = {f"src{i}": None for i in range(len(src))} | {"linear": fit_linear, "gbdt": "gbdt", "forest": "forest"}
        preds = {h: [None] * len(items) for h in heads}
        for f in (0, 1):
            tr = [(lps, t) for lps, t, g in items if fold_of[g] != f]
            te = [i for i, (_, _, g) in enumerate(items) if fold_of[g] == f]
            fitted = {"linear": fit_linear(tr), "gbdt": fit_tree(tr, "gbdt"), "forest": fit_tree(tr, "forest")}
            for i in te:
                lps = items[i][0]
                for h in heads:
                    preds[h][i] = np.exp(lps[int(h[3:])]) if h.startswith("src") else fitted[h](lps)
        T = [t for _, t, _ in items]
        res[split] = {h: metrics(P, T) for h, P in preds.items()}
        print(f"== {split} (n={len(items)} questions)")
        for h, m in res[split].items():
            name = os.path.basename(a.sources[int(h[3:])]) if h.startswith("src") else h
            print(f"   {name:22s} acc {m['acc'][0]:.3f} [{m['acc'][1]:.3f},{m['acc'][2]:.3f}]  log_loss {m['log_loss'][0]:.3f}  "
                  f"brier {m['brier'][0]:.3f}  ece {m['ece'][0]:.3f}", flush=True)
    if a.name:
        os.makedirs(os.path.join("runs", a.name), exist_ok=True)
        json.dump(res, open(os.path.join("runs", a.name, "tree_head.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
