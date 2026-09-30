"""Do per-deployment calibration profiles help OUT of sample? 2-fold cross-fitting on saved predictions.

    python -m eval.profile_cv runs/<eval name> [...]        # dirs written by eval.evaluate --save-preds

Per benchmark: split items in two (seeded), fit a profile (src/opendecider/calibration.fit_logit_profile, the same fit
the /v1/calibrate endpoint uses) on one half, apply to the other, swap; metrics on the pooled out-of-fold predictions
(bootstrap 95% CIs). 'vector' (per-option bias, keyed by option NAME like the API) only where every item has the same
option-name set (predictions must carry `options`); logits are then re-ordered to one canonical name order first.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from opendecider.calibration import fit_logit_profile

from .metrics import summarize


def load(path):
    rows = [json.loads(l) for l in open(path)]
    K = max(len(r["logits"]) for r in rows)
    Z = torch.full((len(rows), K), -1e9)
    for i, r in enumerate(rows):
        Z[i, : len(r["logits"])] = torch.tensor(r["logits"], dtype=torch.float32)
    mask = Z > -1e8
    y = torch.tensor([r["label"] for r in rows])
    sets = {tuple(sorted(r["options"])) for r in rows} if all("options" in r for r in rows) else set()
    same = len(sets) == 1 and len(next(iter(sets))) == len(set(next(iter(sets))))
    if same:                                           # canonical name order: bias j always means the same option
        names = list(next(iter(sets)))
        perm = torch.tensor([[r["options"].index(n) for n in names] for r in rows])
        Z = Z.gather(1, perm)
        y = torch.tensor([names.index(r["options"][r["label"]]) for r in rows])
    return Z, mask, y, same


def cross_fit(Z, mask, y, method, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = [idx[: len(y) // 2], idx[len(y) // 2:]]
    P = torch.empty_like(Z)
    for a, b in ((0, 1), (1, 0)):
        tr, te = torch.tensor(folds[a]), torch.tensor(folds[b])
        T, bias = fit_logit_profile(Z[tr], y[tr], method, mask=mask[tr])
        P[te] = torch.softmax(Z[te] / T + torch.tensor(bias), -1)
    return P


def main(argv=None):
    for d in (argv or sys.argv[1:]):
        res = {}
        for f in sorted(glob.glob(os.path.join(d, "preds_*.jsonl"))):
            split = os.path.basename(f)[6:-6]
            Z, mask, y, same = load(f)
            probs = lambda P: [p[m].numpy() for p, m in zip(P, mask)]
            yl = y.tolist()
            out = {"raw": summarize(probs(torch.softmax(Z, -1)), yl, n_boot=1000),
                   "temperature": summarize(probs(cross_fit(Z, mask, y, "temperature")), yl, n_boot=1000)}
            if same:
                out["vector"] = summarize(probs(cross_fit(Z, mask, y, "vector")), yl, n_boot=1000)
            res[split] = out
            line = f"{split:28s} n={len(yl):4d}"
            for k, v in out.items():
                line += f" | {k} acc {v['accuracy']['value']:.3f} nll {v['nll']['value']:.3f} ece {v['ece']['value']:.3f}"
            print(line, flush=True)
        json.dump(res, open(os.path.join(d, "profile_cv.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
