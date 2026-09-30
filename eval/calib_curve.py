"""Tree calibrators on ONE model's output probabilities, vs temperature scaling, as a function of how much labelled
data the calibrator gets (small-data learning curve).

    .venv/bin/python -m eval.calib_curve --sources runs/preds-x2b runs/public-x9b-stitch runs/zs-9b --name calib-public
    .venv/bin/python -m eval.calib_curve --typed --sources runs/td-x2b runs/td-x9b-stitch runs/td-9b --name calib-typed

Protocol: per split and source, questions are divided into two halves by group (a typed-decisions case, or a public
item). The calibrator is fitted on n questions subsampled from one half and evaluated on ALL of the other half, then the
halves swap; small n is repeated over several seeds. Heads (all per model, no ensembling):
  raw          the model's own probabilities
  temp         one temperature  p = softmax(log p / T)
  forest       LightGBM random forest on per-option features (log p, rank, gap to best, entropy, K), normalised per
               question, then a temperature
  gbdt         same with gradient boosting
  gbdt_res     residual gradient boosting: trees start from the temperature model (init_score = logit p_T per option)
               and only learn a correction; with little data the correction stays small, so it falls back to `temp`
  forest_res   residual forest: a regression forest on y - p_T, added back to p_T (LightGBM's rf mode takes no
               init_score)
  ets          ensemble temperature scaling (Mix-n-Match): simplex mix of p_T, p and uniform
  isotonic     per-option isotonic regression (PAV) on p_T, class-agnostic, renormalised per question
  iso_shrunk   isotonic blended with temp by n/(n+100): trusts the nonparametric fit only as labels accumulate
  iso_top      top-label isotonic (Gupta & Ramdas): recalibrates only the argmax probability, so accuracy is unchanged
  hist         histogram binning (10 equal-mass bins, Laplace-smoothed)
  beta         beta calibration (Kull et al. 2017): 3-parameter logistic map on [log p, -log(1-p)]
Every per-option probability is renormalised over the question's options. The calibrator math lives in
src/opendecider/calibrators.py (shared with /v1/calibrate).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from opendecider.calibrators import (NEG, fit_beta, fit_ets, fit_hist, fit_isotonic, fit_temp, logit,
                                     per_option as _per_option, scatter, softmax_masked, top_label)

from .tree_head import load_dir


def pad(items):
    """items: list of (lp (K,), target (K,)) -> L (N,Km) log-probs (pad NEG), T (N,Km) targets, M (N,Km) mask."""
    Km = max(len(lp) for lp, _ in items)
    N = len(items)
    L = np.full((N, Km), NEG); T = np.zeros((N, Km)); M = np.zeros((N, Km), bool)
    for i, (lp, t) in enumerate(items):
        L[i, :len(lp)] = lp; T[i, :len(t)] = t; M[i, :len(lp)] = True
    return L, T, M


def feats(L, M):
    """Per-option features for the valid (masked) rows, vectorised over questions -> (rows, 5)."""
    P = np.where(M, np.exp(L), 0.0)
    ent = -(P * np.where(M, L, 0.0)).sum(1, keepdims=True)
    rank = np.argsort(np.argsort(-L, 1), 1).astype(float)
    gap = L - L.max(1, keepdims=True)
    K = M.sum(1, keepdims=True).astype(float)
    F = np.stack([L, rank, gap, np.broadcast_to(ent, L.shape), np.broadcast_to(K, L.shape)], -1)
    return F[M]


PARAMS = {
    "gbdt": {"objective": "cross_entropy", "learning_rate": 0.05, "num_leaves": 4, "min_data_in_leaf": 10,
             "lambda_l2": 1.0, "feature_fraction": 0.8, "min_data_in_bin": 1, "verbose": -1, "seed": 0},
    "forest": {"objective": "cross_entropy", "boosting": "rf", "num_leaves": 8, "min_data_in_leaf": 5,
               "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.8, "min_data_in_bin": 1,
               "verbose": -1, "seed": 0},
}
ROUNDS = {"gbdt": 100, "forest": 200}
TREES = True


def fit_heads(L, T, M):
    """Fit every calibrator on (L, T, M); return {head: fn(L, M) -> (N,Km) probabilities}."""
    import lightgbm as lgb
    heads = {"raw": lambda L2, M2: softmax_masked(L2, M2)}
    Tt = fit_temp(L, T, M)
    heads["temp"] = lambda L2, M2, Tt=Tt: softmax_masked(L2 / Tt, M2)
    X, y = feats(L, M), T[M]
    init = logit(softmax_masked(L / Tt, M)[M])
    pT = softmax_masked(L / Tt, M)
    x = pT[M]
    n = len(L)
    heads["ets"] = fit_ets(L, T, M, Tt)
    for name, fitter in (("isotonic", fit_isotonic), ("hist", fit_hist), ("beta", fit_beta)):
        f = fitter(x, y)
        heads[name] = lambda L2, M2, f=f: _per_option(f, softmax_masked(L2 / Tt, M2), M2)
    # isotonic shrunk toward the temperature model: weight n/(n + n0), n0 = 100 questions
    lam = n / (n + 100.0)
    heads["iso_shrunk"] = lambda L2, M2, lam=lam: lam * heads["isotonic"](L2, M2) + (1 - lam) * softmax_masked(L2 / Tt, M2)
    # top-label isotonic: argmax-preserving
    yt = T[np.arange(n), pT.argmax(1)]
    fconf = fit_isotonic(pT.max(1), yt)
    heads["iso_top"] = lambda L2, M2, f=fconf: top_label(f, L2, M2, Tt)
    if not TREES:
        return heads
    for kind in ("gbdt", "forest"):
        m = lgb.train(PARAMS[kind], lgb.Dataset(X, y), num_boost_round=ROUNDS[kind])
        Zp = np.log(np.clip(scatter(m.predict(X), M), 1e-6, 1))
        Tk = fit_temp(Zp, T, M)

        def plain(L2, M2, m=m, Tk=Tk):
            return softmax_masked(np.log(np.clip(scatter(m.predict(feats(L2, M2)), M2), 1e-6, 1)) / Tk, M2)
        heads[kind] = plain
        if kind == "gbdt":
            mr = lgb.train(PARAMS[kind], lgb.Dataset(X, y, init_score=init), num_boost_round=ROUNDS[kind])

            def resid(L2, M2, mr=mr):
                base = logit(softmax_masked(L2 / Tt, M2)[M2])
                q = 1 / (1 + np.exp(-(base + mr.predict(feats(L2, M2), raw_score=True))))
                q = scatter(q, M2)
                return q / q.sum(1, keepdims=True)
        else:
            # LightGBM's rf mode takes no init_score: regress the residual y - p_T directly and add it back
            pT = softmax_masked(L / Tt, M)[M]
            mr = lgb.train(PARAMS[kind] | {"objective": "regression"}, lgb.Dataset(X, y - pT), num_boost_round=ROUNDS[kind])

            def resid(L2, M2, mr=mr):
                base = softmax_masked(L2 / Tt, M2)[M2]
                q = scatter(np.clip(base + mr.predict(feats(L2, M2)), 1e-6, 1), M2)
                return q / q.sum(1, keepdims=True)
        heads[kind + "_res"] = resid
    return heads


def metrics(P, T):
    eps = 1e-9
    acc = (P.argmax(1) == T.argmax(1)).astype(float)
    ll = -(T * np.log(P + eps)).sum(1)
    brier = ((P - T) ** 2).sum(1)
    conf = P.max(1)
    o = np.argsort(conf)
    ece = sum(len(b) / len(o) * abs(acc[b].mean() - conf[b].mean()) for b in np.array_split(o, 15) if len(b))
    return {"acc": acc.mean(), "log_loss": ll.mean(), "brier": brier.mean(), "ece": float(ece)}


def curve(items, groups, sizes, seeds):
    L, T, M = pad(items)
    ug = sorted(set(groups))
    perm = np.random.default_rng(0).permutation(len(ug))
    half = {ug[i]: int(j % 2) for j, i in enumerate(perm)}
    fold = np.array([half[g] for g in groups])
    res = {}
    for n in sizes:
        runs = []
        for s in range(1 if n == "all" else seeds):
            P = {}
            for f in (0, 1):
                tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
                if n != "all":
                    if n > len(tr):
                        break
                    tr = np.random.default_rng(1000 * s + f).choice(tr, n, replace=False)
                heads = fit_heads(L[tr], T[tr], M[tr])
                for h, fn in heads.items():
                    P.setdefault(h, np.zeros(T.shape))[te] = fn(L[te], M[te])
            else:
                runs.append({h: metrics(p, T) for h, p in P.items()})
        if runs:
            res[str(n)] = {h: {k: float(np.mean([r[h][k] for r in runs])) for k in runs[0][h]} for h in runs[0]}
            res[str(n)]["_seeds"] = len(runs)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--typed", action="store_true")
    ap.add_argument("--sizes", default="25,50,100,250,500,all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--splits", default="banking77_8way,prompt_injections,openbookqa,commonsenseqa,pubmedqa")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-trees", action="store_true")
    a = ap.parse_args(argv)
    global TREES
    TREES = not a.no_trees
    sizes = [s if s == "all" else int(s) for s in a.sizes.split(",")]
    keep = {"typed_decisions"} if a.typed else set(a.splits.split(","))
    out = {}
    for d in a.sources:
        src = load_dir(d, a.typed)
        for split in sorted({k[0] for k in src} & keep):
            ks = sorted(k for k in src if k[0] == split)
            items = [(src[k]["lp"], np.array(src[k]["gold"]) if a.typed else np.eye(len(src[k]["lp"]))[src[k]["label"]])
                     for k in ks]
            groups = [k[1].split(":")[0] for k in ks]
            r = curve(items, groups, sizes, a.seeds)
            out[f"{os.path.basename(d)}/{split}"] = r
            print(f"== {os.path.basename(d)} / {split} (n={len(items)})  log loss | acc", flush=True)
            for n, hs in r.items():
                print(f"   n={n:>4}  " + "  ".join(f"{h} {m['log_loss']:.3f}|{m['acc']:.3f}" for h, m in hs.items()
                                                  if h != "_seeds"), flush=True)
    if a.name:
        os.makedirs(os.path.join("runs", a.name), exist_ok=True)
        json.dump(out, open(os.path.join("runs", a.name, "curve.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
