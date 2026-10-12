"""Does decomposition help calibration and certified automation, not only accuracy? (experiment plan step 2, part 2)

Reads the per-item output of eval/decomposition_eval.py (<out>_items.jsonl: direct, decomposed and every sub-answer's
probabilities, with ground truth for the final decision and every sub-question) and reports, per family:

  calibration  ECE (15 equal-mass bins, top-class confidence), Brier, NLL of the final decision: direct, decomposed
               (zero-shot combination), and both after temperature scaling (2-fold cross-fitted); routing also the
               fitted logistic. Bootstrap CIs (1,000 resamples).
  ranking      AUROC of the confidence for correctness; selective accuracy at 50% / 80% coverage.
  certified    DecisionGate's Learn-then-Test (fixed-sequence, Clopper-Pearson) on random thirds (plan / check / test),
               R splits: certified automatic share on the test third and its realized error, alpha 5 / 10 / 15%,
               delta 10%. Scores: direct, decomposed, decomposed + temperature, and the weakest-part confidence.
  composed     part-level certification: each of the m sub-questions certified at alpha/m on its own check labels; a
               case is automatic only if all its parts are, so P(final wrong) <= sum of part errors <= alpha (union
               bound) with probability >= 1 - delta (Bonferroni over parts, delta/m each). Share and realized error.
  sets         LAC sets per part at level alpha/m, propagated through the combining function (every combination of
               part answers in the sets): final coverage >= 1 - alpha by the union bound. Coverage and singleton rate vs
               direct LAC sets at alpha.

    .venv/bin/python eval/decomposition_analysis.py runs/decomp/x9b_items.jsonl [...] --out runs/decomp/analysis.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get("DECISIONGATE_DIR", "/home/srdelam/Desktop/DecisionGate"))
from gateway.certify import fst_threshold, lac_qhat, plan_sequence  # noqa: E402

ALPHAS = (0.02, 0.05, 0.10, 0.15)
DELTA = 0.10
EPS = 1e-12


# ============================================================ basic metrics
def ece(conf, correct, bins=15):
    o = np.argsort(conf)
    parts = np.array_split(o, bins)
    return float(sum(len(p) * abs(conf[p].mean() - correct[p].mean()) for p in parts if len(p)) / len(conf))


def auroc(score, pos):
    pos = pos.astype(bool)
    if pos.all() or (~pos).all():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def sel_acc(conf, correct, cov):
    k = max(1, int(round(cov * len(conf))))
    return float(correct[np.argsort(-conf)[:k]].mean())


def boot(f, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    return [float(f(np.arange(n))), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def temp_scale(P, y, idx_fit):
    """Fit one temperature on idx_fit (grid on log T, NLL), apply to all rows."""
    L = np.log(np.clip(P, EPS, 1))
    best = None
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        Q = np.exp(L[idx_fit] / t); Q /= Q.sum(1, keepdims=True)
        nll = -np.log(np.clip(Q[np.arange(len(idx_fit)), y[idx_fit]], EPS, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, t)
    Q = np.exp(L / best[1]); return Q / Q.sum(1, keepdims=True)


def cross_temp(P, y, seed=0):
    """2-fold cross-fitted temperature scaling (no row is scaled by a temperature fitted on itself)."""
    n = len(y); perm = np.random.default_rng(seed).permutation(n); a, b = perm[: n // 2], perm[n // 2:]
    out = np.empty_like(P)
    out[b] = temp_scale(P, y, a)[b]; out[a] = temp_scale(P, y, b)[a]
    return out


# ============================================================ family structure (parts, labels, combining function)
def parts_of(fam, r):
    """[(probs, true index)] per sub-question, and a function mapping chosen part indices -> final index."""
    pr, m = r["parts"], r["meta"]
    if fam == "two_hop":                     # chain: final = h2 given the CHOSEN h1; h2's own label is the final label
        return [(pr["h1"], m["h1_label"]), (pr["h2"], r["y"])], None
    if fam == "routing":
        a, g, f = m["parts"]
        return [(pr["amt"], 0 if a else 1), (pr["gold"], 0 if g else 1), (pr["flag"], 0 if f else 1)], \
            lambda i: 0 if ((i[0] == 0 and i[1] == 1) or i[2] == 0) else 1
    if fam == "any_over":
        c = m["cands"]
        return [(pr["max"], m["max_label"])], lambda i: 0 if c[i[0]] > m["thr"] else 1
    if fam == "date_cmp":
        d = m["dates"]
        return [(pr["ren"], m["ren_label"]), (pr["end"], m["end_label"])], lambda i: 0 if d[i[0]] < d[i[1]] else 1
    raise KeyError(fam)


# ============================================================ certification on random thirds
def certify_share(s, err, rng=None, R=200, seed=123):
    """Mean certified automatic share / realized error on the test third, per alpha (FST, plan on the plan third).
    The same R splits for every score (common random numbers), so differences between scores are not split noise."""
    n = len(s); out = {}
    for al in ALPHAS:
        shares, errs = [], []
        rng = np.random.default_rng(seed)
        for _ in range(R):
            p = rng.permutation(n); A, C, T = p[: n // 3], p[n // 3: 2 * n // 3], p[2 * n // 3:]
            seq, _, _ = plan_sequence(np.concatenate([s[A], s[T]]), s[A], err[A], len(C), al, DELTA)
            thr = fst_threshold(s[C], err[C], seq, al, DELTA)
            sel = s[T] >= thr
            shares.append(sel.mean()); errs.append(err[T][sel].mean() if sel.any() else 0.0)
        out[str(al)] = {"share": float(np.mean(shares)), "share_sd": float(np.std(shares)),
                        "realized_err": float(np.mean(errs)), "p_err_gt_alpha": float(np.mean(np.array(errs) > al))}
    return out


def composed_share(parts, final_wrong, rng=None, R=200, seed=123):
    """Each part certified at alpha/m (delta/m) on the check third; automatic iff every part clears its threshold."""
    n = len(final_wrong); m = len(parts); out = {}
    parts = [c for c in parts if len(c)]
    m = len(parts)
    S = [np.array([max(p) for p, _ in col]) for col in parts]
    E = [np.array([float(int(np.argmax(p)) != t) for p, t in col]) for col in parts]
    for al in ALPHAS:
        shares, errs = [], []
        rng = np.random.default_rng(seed)
        for _ in range(R):
            p = rng.permutation(n); A, C, T = p[: n // 3], p[n // 3: 2 * n // 3], p[2 * n // 3:]
            sel = np.ones(len(T), bool)
            for s, e in zip(S, E):
                seq, _, _ = plan_sequence(np.concatenate([s[A], s[T]]), s[A], e[A], len(C), al / m, DELTA / m)
                sel &= s[T] >= fst_threshold(s[C], e[C], seq, al / m, DELTA / m)
            shares.append(sel.mean()); errs.append(final_wrong[T][sel].mean() if sel.any() else 0.0)
        out[str(al)] = {"share": float(np.mean(shares)), "realized_err": float(np.mean(errs)),
                        "p_err_gt_alpha": float(np.mean(np.array(errs) > al))}
    return out


def propagated_sets(fam, rows, rng, R=100):
    """Part LAC sets at alpha/m (fit on half), image under the combining function; vs direct LAC sets at alpha."""
    out = {}
    if fam == "two_hop":
        return None                                              # chained: h2's options depend on the chosen h1
    n = len(rows)
    for al in ALPHAS:
        cov_d, sing_d, cov_c, sing_c = [], [], [], []
        for _ in range(R):
            p = rng.permutation(n); A, T = p[: n // 2], p[n // 2:]
            Pd = np.array([rows[i]["direct"] for i in range(n)]); y = np.array([rows[i]["y"] for i in range(n)])
            qd = lac_qhat(Pd[A], y[A], al)
            Sd = Pd[T] >= 1 - qd
            cov_d.append(Sd[np.arange(len(T)), y[T]].mean()); sing_d.append((Sd.sum(1) == 1).mean())
            pt = [parts_of(fam, rows[i]) for i in range(n)]
            m = len(pt[0][0])
            qs = []
            for j in range(m):
                Pj = [pt[i][0][j][0] for i in A]; tj = np.array([pt[i][0][j][1] for i in A])
                Kmax = max(len(x) for x in Pj)
                Pm = np.array([list(x) + [0.0] * (Kmax - len(x)) for x in Pj])
                qs.append(lac_qhat(Pm, tj, al / m))
            cov, sing = [], []
            for i in T:
                parts, fn = pt[i]
                sets = [[k for k, pk in enumerate(pp) if pk >= 1 - qs[j]] or [int(np.argmax(pp))]
                        for j, (pp, _) in enumerate(parts)]
                img = {fn(c) for c in itertools.product(*sets)}
                cov.append(rows[i]["y"] in img); sing.append(len(img) == 1)
            cov_c.append(np.mean(cov)); sing_c.append(np.mean(sing))
        out[str(al)] = {"direct": {"coverage": float(np.mean(cov_d)), "singleton": float(np.mean(sing_d))},
                        "propagated": {"coverage": float(np.mean(cov_c)), "singleton": float(np.mean(sing_c))}}
    return out


# ============================================================ one family
def analyse(fam, rows, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([r["y"] for r in rows])
    variants = {"direct": [r["direct"] for r in rows], "decomposed": [r["decomp"] for r in rows]}
    if rows[0].get("marginal"):
        variants["marginalized chain"] = [r["marginal"] for r in rows]
    if rows[0].get("fitted"):
        variants["fitted_logistic (FITTED, train split)"] = [r["fitted"] for r in rows]
    if fam != "two_hop":                                         # binary families only
        variants["stacked (FITTED, cross-fitted)"] = stack(fam, rows, seed)
    res = {"n": len(rows), "calibration": {}, "ranking": {}, "certified": {}}
    pad = lambda Ps: np.array([list(p) + [0.0] * (max(map(len, Ps)) - len(p)) for p in Ps])
    for name, Ps in list(variants.items()):
        P = pad(Ps)
        for tag, Q in ((name, P), (name + " + temperature (cross-fitted)", cross_temp(P, y, seed))):
            if "FITTED" in name and "temperature" in tag:
                continue
            conf = Q.max(1); correct = (Q.argmax(1) == y).astype(float)
            pt = np.clip(Q[np.arange(len(y)), y], EPS, 1)
            res["calibration"][tag] = {
                "acc": boot(lambda i: correct[i].mean(), len(y)),
                "ece": boot(lambda i: ece(conf[i], correct[i]), len(y)),
                "brier": boot(lambda i: ((Q[i] - np.eye(Q.shape[1])[y[i]]) ** 2).sum(1).mean(), len(y)),
                "nll": boot(lambda i: -np.log(pt[i]).mean(), len(y))}
            res["ranking"][tag] = {"auroc": boot(lambda i: auroc(conf[i], correct[i]), len(y)),
                                   "sel_acc@50": sel_acc(conf, correct, 0.5), "sel_acc@80": sel_acc(conf, correct, 0.8)}
            res["certified"][tag] = certify_share(conf, 1 - correct, rng)
    # weakest-part confidence as the certification score (decomposed answer, min over parts' top probability)
    Pdec = pad(variants["decomposed"]); dec_wrong = (Pdec.argmax(1) != y).astype(float)
    weakest = np.array([min(max(p) for p, _ in parts_of(fam, r)[0]) for r in rows])
    res["certified"]["decomposed, weakest-part confidence"] = certify_share(weakest, dec_wrong, rng)
    res["ranking"]["decomposed, weakest-part confidence"] = {"auroc": boot(lambda i: auroc(weakest[i], 1 - dec_wrong[i]),
                                                                         len(y))}
    # part-level (composed) certification: the final decision as computed from the parts' argmaxes
    cols = list(zip(*[parts_of(fam, r)[0] for r in rows]))
    if fam == "two_hop":
        fin_wrong = dec_wrong                                    # the chain's final answer is h2's argmax
    else:
        fns = [parts_of(fam, r)[1] for r in rows]
        fin = np.array([fn(tuple(int(np.argmax(p)) for p, _ in parts_of(fam, r)[0])) for fn, r in zip(fns, rows)])
        fin_wrong = (fin != y).astype(float)
    res["composed"] = {"parts": len(cols), "part_acc": [float(np.mean([int(np.argmax(p)) == t for p, t in c]))
                                                         for c in cols],
                       "certified": composed_share(cols, fin_wrong, rng)}
    res["sets"] = propagated_sets(fam, rows, rng)
    res["independence"] = independence(fam, rows)
    res["label_efficiency"] = label_efficiency(Pdec.max(1), dec_wrong, rng)
    return res


def _logreg(X, y, l2=1.0, iters=50):
    """L2 logistic regression by Newton steps (IRLS); intercept unpenalized. y in {0,1}."""
    X1 = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(X1.shape[1])
    R = l2 * np.eye(X1.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + R @ w
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Z: 1 / (1 + np.exp(-np.hstack([np.ones((len(Z), 1)), Z]) @ w))


def stack(fam, rows, seed=0, folds=2):
    """FITTED combiner (stacking): L2 logistic regression with final-label targets on the logits of the direct and the
    decomposed answer and the log top-probability of every part, cross-fitted in `folds` folds (no row is predicted
    by a model fitted on it). Binary families; two_hop returns the chained answer unchanged."""
    if fam == "two_hop":
        return [r["decomp"] for r in rows]
    lg = lambda q: np.log(max(q, EPS)) - np.log(max(1 - q, EPS))
    X = np.array([[lg(r["direct"][0]), lg(r["decomp"][0])] +
                  [np.log(max(max(v), EPS)) for k, v in sorted(r["parts"].items()) if k != "direct"] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    yes = np.array([1.0 - r["y"] for r in rows])                     # label 0 = yes
    out = np.zeros((len(yes), 2))
    perm = np.random.default_rng(seed + 11).permutation(len(yes))
    for k in range(folds):
        te = perm[k::folds]; tr = np.setdiff1d(perm, te)
        p = _logreg(X[tr], yes[tr])(X[te])
        out[te, 0], out[te, 1] = p, 1 - p
    return out.tolist()


def independence(fam, rows):
    """Are sub-answer errors independent? P(all parts right) against the product of the parts' accuracies, and the
    pairwise correlation of part errors (positive: errors come together, so independence products are overconfident)."""
    cols = list(zip(*[parts_of(fam, r)[0] for r in rows]))
    E = np.array([[float(int(np.argmax(p)) != t) for p, t in c] for c in cols])        # (parts, n)
    right = 1 - E
    out = {"p_all_right": float(right.prod(0).mean()), "product_of_marginals": float(np.prod(right.mean(1))),
           "part_error_rate": E.mean(1).tolist()}
    if len(E) > 1:
        C = np.corrcoef(E)
        out["error_corr"] = [[None if np.isnan(x) else float(x) for x in row] for row in C]
    return out


def label_efficiency(s, err, rng=None, ns=(50, 100, 200), al=0.10, R=200, seed=123):
    """Certified share at alpha 10% vs the number of check labels (plan on 1/3, test on the last 1/3)."""
    n = len(s); out = {}
    for k in ns:
        sh = []
        rng = np.random.default_rng(seed)
        for _ in range(R):
            p = rng.permutation(n); A, T = p[: n // 3], p[2 * n // 3:]
            C = p[n // 3: n // 3 + min(k, n // 3)]
            seq, _, _ = plan_sequence(np.concatenate([s[A], s[T]]), s[A], err[A], len(C), al, DELTA)
            sh.append(float((s[T] >= fst_threshold(s[C], err[C], seq, al, DELTA)).mean()))
        out[str(k)] = float(np.mean(sh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for path in a.items:
        rows = defaultdict(list)
        for l in open(path):
            r = json.loads(l)
            rows[r["family"]].append(r)
        model = os.path.basename(path).replace("_items.jsonl", "")
        out[model] = {fam: analyse(fam, rs) for fam, rs in rows.items() if fam != "routing_train"}
        for fam, r in out[model].items():
            c = r["calibration"]; k = r["certified"]
            print(f"{model} {fam}: ECE direct {c['direct']['ece'][0]:.3f} decomp {c['decomposed']['ece'][0]:.3f} | "
                  f"AUROC {r['ranking']['direct']['auroc'][0]:.3f} / {r['ranking']['decomposed']['auroc'][0]:.3f} | "
                  f"certified@10% direct {k['direct']['0.1']['share']:.2f} decomp {k['decomposed']['0.1']['share']:.2f} "
                  f"composed {r['composed']['certified']['0.1']['share']:.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
