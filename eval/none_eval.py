"""Abstention under shift: the explicit `none` option and conformal sets on held-out public benchmarks (eval only).

    .venv/bin/python -m eval.none_eval --model checkpoints/opendecider-2b --n 150 --name none-2b
    .venv/bin/python -m eval.none_eval --ckpt runs/<run>/model.pt --n 150 --name none-<run>

Everything goes through Decider.decide (the serving path, `none` text appended as in production). Per item:
  answerable  : the benchmark question as is
  gold_removed: the correct option removed
  hard        : the correct option removed AND only the two options the model itself ranked highest kept
  unrelated   : the same question asked about the state of an item from another benchmark
Items need >= 3 options for the removal conditions (yes/no sets only contribute answerable + unrelated).
Reported, per benchmark and pooled, with 1,000-resample bootstrap CIs:
  - accuracy on answerable items; mean P(none) per condition; AUROC of P(none) vs answerable
  - conformal coverage / mean set size / abstention rate per condition at alpha 0.1, with
      (a) the checkpoint's own calibration scores (fitted on our training-family calib split: tests shift), and
      (b) scores refit on the first --refit labelled answerable items of that benchmark (a caller's profile),
          evaluated on the remaining items
  - selective accuracy at 50 / 80 / 95% coverage (by top probability) on answerable items
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "src")
from opendecider.conformal import coverage, prediction_set, scores, threshold  # noqa: E402
from opendecider.guards import gpu_lock, limit_gpu_memory  # noqa: E402

from .evaluate import read_jsonl, to_question  # noqa: E402
from .metrics import selective_accuracy  # noqa: E402

FILES = ["banking77_8way", "openbookqa", "commonsenseqa", "pubmedqa", "prompt_injections"]
ALPHA = 0.1


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):                                   # average ranks for ties
        m = allv == v
        r[m] = r[m].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auroc(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot(fn, n_items, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n_items, n_items)) for _ in range(n)]
    vals = [v for v in vals if v == v]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def ask(dec, state, qs):
    """qs: {name: (prompt, options)} -> {name: (probs over options, P(none), value)}. Yes/no items are asked as a
    two-option choice so every condition shares one format."""
    req = {"state": state, "questions": {k: {"type": "choice", "prompt": p, "options": o} for k, (p, o) in qs.items()}}
    out = dec.decide(req)["answers"]
    return {k: (a["probs_list"], a.get("none", float("nan")), a["value"]) for k, a in out.items()}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="hub checkpoint dir (opendecider.hub)")
    g.add_argument("--ckpt", help="runs/<x>/model.pt")
    ap.add_argument("--backbone", default=None, help="hub backbone variant")
    ap.add_argument("--n", type=int, default=150, help="items per benchmark")
    ap.add_argument("--refit", type=int, default=50, help="labelled items per benchmark for the refit profile")
    ap.add_argument("--name", default="none-eval")
    a = ap.parse_args()
    limit_gpu_memory(float(os.environ.get("OPENDECIDER_EVAL_GPU_GB", "24")))
    rng = random.Random(0)
    data = {}
    for f in FILES:
        recs = read_jsonl(f"data/public/{f}.jsonl")
        rng.shuffle(recs)
        data[f] = recs[: a.n]
    rows = []
    with gpu_lock("none_eval"):
        if a.model:
            from opendecider.hub import load_decider
            dec = load_decider(a.model, backbone=a.backbone)
        else:
            from opendecider.checkpoint import load_decider
            dec = load_decider(a.ckpt)
        conf = dec.conformal.get("lac", {}).get("choice") if dec.conformal else None
        others = [(f, r) for f in FILES for r in data[f]]
        for f in FILES:
            for i, r in enumerate(data[f]):
                q = to_question(r["questions"][0])
                opts, lab = list(q.options), q.label
                qs = {"answerable": (q.prompt, opts)}
                if len(opts) >= 3:
                    qs["gold_removed"] = (q.prompt, [o for j, o in enumerate(opts) if j != lab])
                got = ask(dec, r["state"], qs)
                p, pn, _ = got["answerable"]
                row = {"bench": f, "i": i, "label": lab, "K": len(opts), "p": p, "none": {"answerable": pn}}
                if "gold_removed" in got:
                    row["none"]["gold_removed"] = got["gold_removed"][1]
                    row["p_removed"] = got["gold_removed"][0]
                    order = [j for j in np.argsort(-np.asarray(p)) if j != lab][:2]
                    row["hard_opts"] = [opts[j] for j in sorted(order)]
                    h = ask(dec, r["state"], {"hard": (q.prompt, row["hard_opts"])})["hard"]
                    row["none"]["hard"], row["p_hard"] = h[1], h[0]
                while True:                                              # state from another benchmark
                    of, orec = others[rng.randrange(len(others))]
                    if of != f:
                        break
                u = ask(dec, orec["state"], {"unrelated": (q.prompt, opts)})["unrelated"]
                row["none"]["unrelated"], row["p_unrelated"] = u[1], u[0]
                rows.append(row)
            print(f"[none_eval] {f}: {len(data[f])} items", flush=True)

    def summarize(rs):
        y = np.array([r["label"] for r in rs])
        acc = [float(np.argmax(r["p"]) == r["label"]) for r in rs]
        out = {"n": len(rs), "accuracy": float(np.mean(acc)), "accuracy_ci": boot(lambda ix: float(np.mean(np.asarray(acc)[ix])), len(rs))}
        pa = [r["none"]["answerable"] for r in rs]
        out["p_none"] = {c: float(np.mean([r["none"][c] for r in rs if c in r["none"]]))
                         for c in ("answerable", "gold_removed", "hard", "unrelated") if any(c in r["none"] for r in rs)}
        out["auroc_none"] = {}
        for c in ("gold_removed", "hard", "unrelated"):
            idx = [k for k, r in enumerate(rs) if c in r["none"]]
            if idx:
                neg = np.asarray(pa)
                pos = np.asarray([rs[k]["none"][c] for k in idx])
                out["auroc_none"][c] = {"value": auroc(pos, neg), "ci": boot_auroc(pos, neg)}
        P = [np.asarray(r["p"]) for r in rs]
        out["selective_accuracy"] = {str(c): selective_accuracy(P, y, c / 100) for c in (50, 80, 95)} \
            if len({len(x) for x in P}) == 1 else None

        def conf_block(cal):
            res = {"answerable": coverage(P, y, cal, ALPHA, "lac")}
            thr = threshold(cal, ALPHA)
            for c, key in (("gold_removed", "p_removed"), ("hard", "p_hard"), ("unrelated", "p_unrelated")):
                sel = [r for r in rs if key in r]
                if sel:                       # abstain = not exactly one option (gold is absent in the first two)
                    sizes = [len(prediction_set(r[key], thr, "lac")) for r in sel]
                    res[c] = {"abstain": float(np.mean([s != 1 for s in sizes])), "mean_size": float(np.mean(sizes))}
            res["answerable"]["abstain"] = 1 - res["answerable"]["singleton"]
            return res
        if conf:
            out["conformal_checkpoint"] = conf_block(conf)
        return out

    report = {"alpha": ALPHA, "benchmarks": {}}
    for f in FILES:
        rs = [r for r in rows if r["bench"] == f]
        report["benchmarks"][f] = summarize(rs)
        cal, test = rs[: a.refit], rs[a.refit:]
        if len(test) >= 20:                                              # caller profile: refit on own labels
            Pc = [np.asarray(r["p"]) for r in cal]
            K = max(len(p) for p in Pc)
            Pm = np.stack([np.pad(p, (0, K - len(p))) for p in Pc])
            sc = scores(Pm, np.array([r["label"] for r in cal]), "lac")
            Pt, yt = [np.asarray(r["p"]) for r in test], np.array([r["label"] for r in test])
            report["benchmarks"][f]["conformal_refit"] = {"n_cal": len(cal), **coverage(Pt, yt, sc, ALPHA, "lac")}
    report["pooled"] = summarize(rows)
    print(json.dumps({k: v for k, v in report.items()}, indent=1, default=float))
    os.makedirs(f"runs/{a.name}", exist_ok=True)
    json.dump({"report": report, "rows": rows}, open(f"runs/{a.name}/results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
