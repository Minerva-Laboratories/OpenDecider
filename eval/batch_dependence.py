"""Analyse the batch-dependence probe (data/builders/batch_dependence.py) from saved eval predictions.

    python -m eval.batch_dependence runs/<eval name> [...]      # dirs written by eval.evaluate --save-preds

Per condition (alone / p10 / p50 / p90 positives among the 7 batch-mates): accuracy and Brier of the target, and the
per-item shift of P(yes) relative to scoring it alone (mean signed shift, mean |shift|). "slope" = shift(p90) -
shift(p10): > 0 means the item looks more positive among positives (assimilation), < 0 means contrast. A target-
isolated model gives ~0 everywhere. 95% bootstrap CIs over items (1,000 resamples).
"""
import json
import os
import sys

import numpy as np

from .metrics import bootstrap_mean_ci

LEVELS = ("alone", "p10", "p50", "p90")


def p_yes(logits):
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max())
    return float(e[0] / e.sum())


def analyse(run_dir):
    P, Y = {}, {}
    for lev in LEVELS:
        path = os.path.join(run_dir, f"preds_batchdep_{lev}.jsonl")
        for line in open(path):
            r = json.loads(line)
            key = r["id"].split(":")[0]
            P.setdefault(lev, {})[key] = p_yes(r["logits"])
            Y[key] = 1.0 if r["label"] == 0 else 0.0
    keys = sorted(set.intersection(*(set(P[l]) for l in LEVELS)))
    ci = lambda v: dict(zip(("mean", "lo", "hi"), bootstrap_mean_ci(v, n=1000)))
    out = {"n_items": len(keys)}
    for lev in LEVELS:
        p = np.array([P[lev][k] for k in keys]); y = np.array([Y[k] for k in keys])
        base = np.array([P["alone"][k] for k in keys])
        out[lev] = {"acc": ci(((p > 0.5) == (y > 0.5)).astype(float)), "brier": ci((p - y) ** 2),
                    "shift": ci(p - base), "abs_shift": ci(np.abs(p - base)), "mean_p_yes": float(p.mean())}
    out["slope_p90_minus_p10"] = ci(np.array([P["p90"][k] - P["p10"][k] for k in keys]))
    return out


def main(argv=None):
    for d in (argv or sys.argv[1:]):
        res = analyse(d)
        with open(os.path.join(d, "batch_dependence.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"== {d}  ({res['n_items']} items)")
        for lev in LEVELS:
            r = res[lev]
            print(f"  {lev:5s} acc {r['acc']['mean']:.3f}  brier {r['brier']['mean']:.3f}  "
                  f"shift {r['shift']['mean']:+.3f} [{r['shift']['lo']:+.3f},{r['shift']['hi']:+.3f}]  "
                  f"|shift| {r['abs_shift']['mean']:.3f}")
        s = res["slope_p90_minus_p10"]
        print(f"  slope p90-p10 {s['mean']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}]")


if __name__ == "__main__":
    main()
