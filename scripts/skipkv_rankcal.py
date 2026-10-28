"""Rank calibration of a skipped-prompt LM's next-token distribution (ODIR over rank positions).

A full ODIR matrix over a 248k-token vocabulary is not possible; over the arm's own top-k candidates, sorted, it is
class-agnostic in tokens: a k x k matrix plus intercepts on the sorted log-probs, which can reorder close candidates.
Data: DepthSkip's experiments/lm.py --dump-topk k (per arm and teacher-forced position: the arm's top-k ids and log-probs, the
exact model's log-probs of those ids, the exact argmax, the gold next token). Fitted on the first 2/3 of documents
toward the exact model's distribution over those candidates; reported on the rest: top-1 agreement with the exact
argmax, gold-token accuracy among the candidates, KL to the exact candidate distribution. Calibrators: identity,
temperature, vector (one bias per rank), ODIR (Kull et al. 2019; off-diagonal and intercept L2 by inner CV).

    python scripts/skipkv_rankcal.py /mnt/ext/opendecider/skipkv/lm_4b_rank_top16.pt
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from skipkv_calib import odir_apply, odir_select  # noqa: E402


def fit_scale(Z, T, kind):
    K = Z.shape[1]
    logT = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=kind == "vector")
    ps = [logT] + ([b] if kind == "vector" else [])
    opt = torch.optim.LBFGS(ps, max_iter=200, line_search_fn="strong_wolfe")
    Zt, Tt = torch.tensor(Z), torch.tensor(T)

    def closure():
        opt.zero_grad()
        loss = -(Tt * (Zt / logT.exp() + b).log_softmax(-1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp()), b.detach().numpy()


def main():
    d = torch.load(sys.argv[1])
    out = {}
    for arm, v in d.items():
        Z = v["skip_lp"].double().numpy()                         # sorted descending (topk)
        E = v["exact_lp"].double().numpy()
        ids, et, gold, doc = (v[k].numpy() for k in ("ids", "exact_top", "gold", "doc"))
        T = np.exp(E - E.max(1, keepdims=True)); T /= T.sum(1, keepdims=True)    # exact, renormalized on candidates
        docs = np.unique(doc)
        tr = np.isin(doc, docs[: 2 * len(docs) // 3])
        te = ~tr
        M = np.ones_like(Z, dtype=bool)
        cals = {"identity": lambda X: X}
        t, _ = fit_scale(Z[tr], T[tr], "temperature")
        cals["temperature"] = lambda X, t=t: X / t
        tv, bv = fit_scale(Z[tr], T[tr], "vector")
        cals["vector"] = lambda X, tv=tv, bv=bv: X / tv + bv
        od, lam = odir_select(Z[tr], T[tr], M[tr])
        cals["odir"] = lambda X, od=od: np.log(np.clip(odir_apply(od, X, np.ones_like(X, dtype=bool)), 1e-30, 1))
        r = {"positions_test": int(te.sum()), "exact_top_in_candidates": float((ids[te] == et[te, None]).any(1).mean()),
             "odir_lambda": lam, "fitted_T": t}
        for k, f in cals.items():
            Q = f(Z[te])
            pick = ids[te][np.arange(te.sum()), Q.argmax(1)]
            Qn = np.exp(Q - Q.max(1, keepdims=True)); Qn /= Qn.sum(1, keepdims=True)
            kl = (T[te] * (np.log(np.clip(T[te], 1e-30, 1)) - np.log(np.clip(Qn, 1e-30, 1)))).sum(1)
            r[k] = {"top1_agree_exact": float((pick == et[te]).mean()), "gold_acc": float((pick == gold[te]).mean()),
                    "kl_candidates": float(kl.mean())}
        ex = (et[te] == gold[te]).mean()
        r["exact_gold_acc"] = float(ex)
        out[arm] = r
        print(arm, json.dumps(r), flush=True)
    json.dump(out, open(sys.argv[1].replace(".pt", "_rankcal.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
