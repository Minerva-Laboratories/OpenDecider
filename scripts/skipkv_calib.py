"""Can calibration on the option logits undo the skipped prompt pass on multiple-choice tasks?
Reads the per-arm option logits saved by DepthSkip's experiments/tasks.py (*_logits.npz). For each arm other than exact
(floor, cut C), calibrators are fitted on half the items and applied to the other half (2-fold, both halves used), toward
two targets: the exact arm's option distribution (undo the skip) and the gold label (task accuracy).
Calibrators: identity; the class-agnostic ones of opendecider.calibrators (temperature, beta, ets, iso_shrunk) and its
CV selection; vector scaling (one bias per option letter); ODIR (Kull et al. 2019, arXiv:1910.12656: matrix scaling
with off-diagonal and intercept L2, strength by inner CV). Vector and ODIR can change which option wins.
Reported on the held-out halves: flips against the exact arm's answer, accuracy, KL(exact || calibrated).

    python scripts/skipkv_calib.py /mnt/ext/opendecider/skipkv/tasks_4b_*_logits.npz
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from opendecider.calibrators import NEG, apply_calibrator, fit_calibrator, select_calibrator, softmax_masked  # noqa: E402

KINDS = ("identity", "temperature", "beta", "ets", "iso_shrunk", "vector")


def odir_fit(L, T, M, lam):
    Z, Tt, Mt = (torch.tensor(x) for x in (L, T, M))
    K = Z.shape[1]
    W = torch.eye(K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    off = ~torch.eye(K, dtype=torch.bool)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")
    Zm = torch.where(Mt, Z, torch.zeros_like(Z)).double()

    def closure():
        opt.zero_grad()
        y = (Zm @ W.T + b).masked_fill(~Mt, NEG)
        loss = -(Tt * y.log_softmax(-1)).sum(-1).mean() + lam * (W[off] ** 2).sum() + lam * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def odir_apply(p, L, M):
    W, b = p
    y = np.where(M, L, 0.0) @ W.T + b
    return softmax_masked(np.where(M, y, NEG), M)


def odir_select(L, T, M, lams=(1e-3, 1e-2, 1e-1, 1.0), k=3):
    fold = np.random.default_rng(0).permutation(len(L)) % k
    best = None
    for lam in lams:
        loss = 0.0
        for f in range(k):
            tr, te = fold != f, fold == f
            P = odir_apply(odir_fit(L[tr], T[tr], M[tr], lam), L[te], M[te])
            loss += -(T[te] * np.log(np.clip(P, 1e-12, 1))).sum()
        if best is None or loss < best[0]:
            best = (loss, lam)
    return odir_fit(L, T, M, best[1]), best[1]


def evaluate(path):
    d = np.load(path)
    gold = d["gold"]
    Le = d["exact"].astype(np.float64)
    M = Le > NEG / 2
    Pe = softmax_masked(Le, M)
    ye = Pe.argmax(1)
    n = len(gold)
    half = np.random.default_rng(1).permutation(n) % 2
    out = {"n": int(n), "exact_acc": float((ye == gold).mean())}
    for arm in [k for k in d.files if k not in ("gold", "exact")]:
        L = d[arm].astype(np.float64)
        res = {}
        for target in ("exact", "gold"):
            T = Pe if target == "exact" else np.eye(M.shape[1])[gold] * M
            P = {}
            for f in (0, 1):
                tr, te = half != f, half == f
                fits = {k: (lambda c: (lambda X, Mm: apply_calibrator(c, X, Mm)))(fit_calibrator(k, L[tr], T[tr], M[tr]))
                        for k in KINDS}
                sel, _ = select_calibrator(L[tr], T[tr], M[tr])
                fits["auto (" + sel["kind"] + ")"] = lambda X, Mm, c=sel: apply_calibrator(c, X, Mm)
                od, lam = odir_select(L[tr], T[tr], M[tr])
                fits["odir"] = lambda X, Mm, p=od: odir_apply(p, X, Mm)
                for k, fn in fits.items():
                    key = "auto" if k.startswith("auto") else k
                    P.setdefault(key, np.zeros_like(L))[te] = fn(L[te], M[te])
                    if k.startswith("auto"):
                        res.setdefault("auto_picks", []).append(k)
            res[target] = {}
            for k, Q in P.items():
                kl = (Pe * (np.log(np.clip(Pe, 1e-12, 1)) - np.log(np.clip(Q, 1e-12, 1)))).sum(1)
                res[target][k] = {"flips_vs_exact": float((Q.argmax(1) != ye).mean()),
                                  "acc": float((Q.argmax(1) == gold).mean()), "kl": float(kl.mean())}
        out[arm] = res
    return out


def main():
    res = {p: evaluate(p) for p in sys.argv[1:]}
    for p, r in res.items():
        print(p, f"n={r['n']} exact acc {r['exact_acc']:.3f}")
        for arm, v in r.items():
            if not isinstance(v, dict):
                continue
            for target in ("exact", "gold"):
                print(f"  {arm:6s} -> {target:5s}: " + ", ".join(
                    f"{k} flips {x['flips_vs_exact']:.3f} acc {x['acc']:.3f}" for k, x in v[target].items()))
    json.dump(res, open(sys.argv[1].rsplit("_", 2)[0] + "_calib.json", "w"), indent=1)


if __name__ == "__main__":
    main()
