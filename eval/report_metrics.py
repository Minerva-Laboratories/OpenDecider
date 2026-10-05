"""Task-appropriate metrics from saved per-item predictions (no model runs), with 95% bootstrap intervals.

    .venv/bin/python -m eval.report_metrics --out runs/metrics_report.json
    python3 -m eval.report_metrics --out runs/metrics_report.json --figures docs/img --figures-only   # matplotlib

Reads runs/<run>/preds_<benchmark>.jsonl (raw logits, temperature 1, as in the headline tables). Per task type:
  binary yes/no (prompt-injection detection, PubMedQA; positive class = "yes"):
      confusion matrix at P = 0.5, precision, recall, specificity, F1, MCC, balanced accuracy,
      ROC-AUC and PR-AUC (average precision) of P(yes), recall at 1% and 5% false-positive rate
  intent classification (Banking77, 8 fixed classes): confusion matrix by class, per-class precision / recall / F1,
      macro-F1, balanced accuracy
  multiple choice (OpenBookQA, CommonsenseQA): answer positions are not classes, so no F1 or ROC; accuracy by gold
      position and the gold-position x predicted-position matrix (position bias)
  all: Brier, risk-coverage area (AURC: mean error rate over all coverage levels when the least confident answers are
      dropped first; lower is better), ECE (15 equal-mass bins, as everywhere in this repo), reliability bins
      (equal-width, for the figure)
The decision threshold is fixed at 0.5: choosing it on these items would fit the benchmark.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SYSTEMS = [("public-x9b-clean3", "OpenDecider 9B"), ("public-x2b-clean3-stitch9b-awq", "OpenDecider 9B AWQ"),
           ("public-x2b-clean3", "OpenDecider 2B"), ("zs-9b", "Qwen3.5-9B, letter scores")]
BINARY = ["prompt_injections", "pubmedqa"]
CLASSES = ["banking77_8way"]
MCQ = ["openbookqa", "commonsenseqa"]
N_BOOT, SEED, BINS = 1000, 0, 15


def load(run: str, bench: str):
    rows = [json.loads(l) for l in open(f"runs/{run}/preds_{bench}.jsonl")]
    rows.sort(key=lambda r: r["id"])
    P = []
    for r in rows:
        z = np.asarray(r["logits"], dtype=np.float64)
        e = np.exp(z - z.max())
        P.append(e / e.sum())
    return rows, P, np.array([r["label"] for r in rows])


# ---------------------------------------------------------------- binary
def _auc(score, pos):
    """ROC-AUC by ranks (ties averaged)."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ap(score, pos):
    """Average precision: sum over distinct thresholds of (recall gained) x (precision there); tied scores form one
    threshold, so the result does not depend on the order of tied items."""
    if pos.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    s, p = score[o], pos[o].astype(float)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]          # last index of each tied group
    tp = np.cumsum(p)[last]
    prec = tp / (last + 1)
    rec = tp / p.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _recall_at_fpr(score, pos, fpr):
    """Recall at the strictest threshold whose false-positive rate is <= fpr (on these items)."""
    neg = np.sort(score[~pos])[::-1]
    k = int(np.floor(fpr * len(neg)))                 # negatives allowed above the threshold
    thr = neg[k] if k < len(neg) else -np.inf
    return float((score[pos] > thr).mean())


def roc_points(score, pos):
    """(false-positive rate, recall) at every distinct threshold, from (0, 0) to (1, 1)."""
    o = np.argsort(-score, kind="stable")
    s, p = score[o], pos[o].astype(float)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = np.cumsum(p)[last], np.cumsum(1 - p)[last]
    return np.r_[0.0, fp / max(fp[-1], 1)], np.r_[0.0, tp / max(tp[-1], 1)]


def pr_points(score, pos):
    """(recall, precision) at every distinct threshold."""
    o = np.argsort(-score, kind="stable")
    s, p = score[o], pos[o].astype(float)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(p)[last]
    return tp / max(p.sum(), 1), tp / (last + 1)


def binary_metrics(p_yes, y_yes):
    pred = p_yes >= 0.5
    tp, fn = int((pred & y_yes).sum()), int((~pred & y_yes).sum())
    fp, tn = int((pred & ~y_yes).sum()), int((~pred & ~y_yes).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"precision": prec, "recall": rec, "specificity": spec, "f1": f1, "mcc": float(mcc),
            "balanced_accuracy": (rec + spec) / 2, "roc_auc": _auc(p_yes, y_yes), "pr_auc": _ap(p_yes, y_yes),
            "recall_at_fpr_1pct": _recall_at_fpr(p_yes, y_yes, 0.01),
            "recall_at_fpr_5pct": _recall_at_fpr(p_yes, y_yes, 0.05)}, {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


# ---------------------------------------------------------------- multiclass
def class_metrics(pred, gold, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (gold, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"macro_f1": float(f1[present].mean()), "balanced_accuracy": float(rec[present].mean())}, cm, prec, rec, f1


# ---------------------------------------------------------------- all tasks
def common_metrics(P, y):
    conf = np.array([p.max() for p in P])
    correct = np.array([int(np.argmax(p) == t) for p, t in zip(P, y)], dtype=float)
    brier = float(np.mean([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(P, y)]))
    o = np.argsort(-conf, kind="stable")
    risk = np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)
    from eval.metrics import _ece, per_example          # the project's ECE: 15 equal-mass bins, ties kept together
    ece = _ece(per_example(P, y))
    return {"accuracy": float(correct.mean()), "brier": brier, "aurc": float(risk.mean()), "ece": ece}, conf, correct


def reliability_fixed(conf, correct, edges=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Equal-width bins for display (confidence ranges people read directly), with counts."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                    "accuracy": float(correct[m].mean()) if m.any() else None,
                    "confidence": float(conf[m].mean()) if m.any() else None})
    return out


def reliability(conf, correct, bins=BINS):
    o = np.argsort(conf, kind="stable")
    out = []
    for chunk in np.array_split(o, bins):
        if len(chunk):
            out.append({"confidence": float(conf[chunk].mean()), "accuracy": float(correct[chunk].mean()),
                        "n": int(len(chunk))})
    return out


def bootstrap(fn, n_items, n=N_BOOT, seed=SEED):
    """fn(idx) -> dict of scalars; returns {name: [lo, hi]} (95% percentile intervals)."""
    rng = np.random.default_rng(seed)
    draws = [fn(rng.integers(0, n_items, n_items)) for _ in range(n)]
    return {k: [float(np.nanpercentile([d[k] for d in draws], 2.5)), float(np.nanpercentile([d[k] for d in draws], 97.5))]
            for k in draws[0]}


def evaluate(run: str, bench: str) -> dict:
    rows, P, y = load(run, bench)
    out = {"n": len(rows)}

    def scalars(idx):
        Pi, yi = [P[i] for i in idx], y[idx]
        m, _, _ = common_metrics(Pi, yi)
        if bench in BINARY:
            p_yes = np.array([p[0] for p in Pi])
            b, _ = binary_metrics(p_yes, yi == 0)
            m.update(b)
        elif bench in CLASSES:
            names = sorted({rows[i]["options"][rows[i]["label"]] for i in range(len(rows))})
            gi = np.array([names.index(rows[i]["options"][rows[i]["label"]]) for i in idx])
            pi = np.array([names.index(rows[i]["options"][int(np.argmax(P[i]))])
                           if rows[i]["options"][int(np.argmax(P[i]))] in names else len(names) for i in idx])
            c, *_ = class_metrics(pi, gi, len(names) + 1)
            m.update(c)
        return m

    full = scalars(np.arange(len(rows)))
    ci = bootstrap(scalars, len(rows))
    out["metrics"] = {k: {"value": v, "ci": ci[k]} for k, v in full.items()}
    _, conf, correct = common_metrics(P, y)
    out["reliability"] = reliability(conf, correct)
    out["reliability_fixed"] = reliability_fixed(conf, correct)
    if bench in BINARY:
        p_yes = np.array([p[0] for p in P])
        _, out["confusion"] = binary_metrics(p_yes, y == 0)
        out["scores"] = {"p_yes": p_yes.tolist(), "is_yes": (y == 0).astype(int).tolist()}
    elif bench in CLASSES:
        names = sorted({r["options"][r["label"]] for r in rows})
        gi = np.array([names.index(r["options"][r["label"]]) for r in rows])
        pi = np.array([names.index(r["options"][int(np.argmax(p))]) for r, p in zip(rows, P)])
        _, cm, prec, rec, f1 = class_metrics(pi, gi, len(names))
        out["classes"] = names
        out["confusion"] = cm.tolist()
        out["per_class"] = {nm: {"precision": float(a), "recall": float(b), "f1": float(c)}
                            for nm, a, b, c in zip(names, prec, rec, f1)}
    else:                                                # positions, not classes
        K = max(len(r["logits"]) for r in rows)
        pred = np.array([int(np.argmax(p)) for p in P])
        cm = np.zeros((K, K), dtype=int)
        np.add.at(cm, (y, pred), 1)
        out["position_confusion"] = cm.tolist()
        out["accuracy_by_gold_position"] = [float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None for k in range(K)]
        out["predicted_position_share"] = (cm.sum(0) / cm.sum()).tolist()
        out["gold_position_share"] = (cm.sum(1) / cm.sum()).tolist()
    return out


def figures(rep: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2, GRID = "#52514e", "#e4e3df"
    COL = {"OpenDecider 9B": "#1baf7a", "OpenDecider 9B AWQ": "#7c5cd6", "OpenDecider 2B": "#2a78d6",
           "Qwen3.5-9B, letter scores": "#eda100"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})
    # ROC and PR curves, injection detection
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for name, res in rep.items():
        s = res["prompt_injections"]["scores"]
        sc, pos = np.array(s["p_yes"]), np.array(s["is_yes"])
        fpr, tpr = roc_points(sc, pos.astype(bool))
        m = res["prompt_injections"]["metrics"]
        a1.plot(fpr, tpr, color=COL[name], lw=1.2, label=f"{name} ({m['roc_auc']['value']:.2f})")
        rc, pr = pr_points(sc, pos.astype(bool))
        a2.plot(rc, pr, color=COL[name], lw=1.2, label=f"{name} ({m['pr_auc']['value']:.2f})")
    a1.plot([0, 1], [0, 1], color=GRID, lw=0.8)
    base = float(np.mean(pos))
    a2.axhline(base, color=GRID, lw=0.8)
    a1.set(xlabel="false-positive rate", ylabel="recall (true-positive rate)", title="ROC, injection detection")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall, injection detection", ylim=(0.4, 1.02))
    a1.legend(fontsize=6, loc="lower right", frameon=False); a2.legend(fontsize=6, loc="lower left", frameon=False)
    fig.savefig(os.path.join(outdir, "injection_roc_pr.png")); plt.close(fig)
    benches = BINARY + CLASSES + MCQ
    titles = {"prompt_injections": "Injection detection", "pubmedqa": "PubMedQA", "banking77_8way": "Banking77",
              "openbookqa": "OpenBookQA", "commonsenseqa": "CommonsenseQA"}
    # calibration error summary: ECE per benchmark and system, 95% intervals
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    w = 0.2
    for si, (name, res) in enumerate(rep.items()):
        xs = np.arange(len(benches)) + (si - 1.5) * w
        v = [res[b]["metrics"]["ece"]["value"] for b in benches]
        lo = [res[b]["metrics"]["ece"]["ci"][0] for b in benches]
        hi = [res[b]["metrics"]["ece"]["ci"][1] for b in benches]
        ax.bar(xs, v, width=w * 0.92, color=COL[name], label=name)
        ax.errorbar(xs, v, yerr=[np.maximum(np.subtract(v, lo), 0), np.maximum(np.subtract(hi, v), 0)], fmt="none", ecolor=INK2, elinewidth=0.7,
                    capsize=1.5)
    ax.set_xticks(range(len(benches))); ax.set_xticklabels([titles[b] for b in benches])
    ax.set_ylabel("expected calibration error\n(lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(os.path.join(outdir, "calibration_ece.png")); plt.close(fig)
    # reliability bars: one row per system, one column per benchmark
    names = list(rep)
    fig, axs = plt.subplots(len(names), len(benches), figsize=(9.6, 6.4), sharex=True, sharey=True)
    for r, name in enumerate(names):
        for c, b in enumerate(benches):
            ax = axs[r, c]
            ax.grid(axis="x", visible=False)
            ax.plot([0.2, 1], [0.2, 1], color=INK2, lw=0.7, ls=(0, (2, 2)))
            for bin_ in rep[name][b]["reliability_fixed"]:
                if bin_["n"] < 5:                       # too few items to show an accuracy
                    continue
                lo, hi, acc, cf = max(bin_["lo"], 0.2), bin_["hi"], bin_["accuracy"], bin_["confidence"]
                ax.bar(lo, acc, width=hi - lo, align="edge", color=COL[name], alpha=0.9, edgecolor="white", lw=0.8)
                g0, g1 = sorted((acc, cf))              # gap to the bin's mean confidence
                ax.bar(lo, g1 - g0, bottom=g0, width=hi - lo, align="edge", color="#c0392b", alpha=0.28,
                       edgecolor="white", lw=0.8)
            ece = rep[name][b]["metrics"]["ece"]["value"]
            ax.text(0.24, 0.93, f"ECE {ece:.3f}", fontsize=6.5, color=INK2, va="top")
            ax.set_xlim(0.2, 1); ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(titles[b], fontsize=8)
            if c == 0:
                ax.set_ylabel(name.replace(", letter scores", "\nletter scores"), fontsize=7)
            if r == len(names) - 1:
                ax.set_xlabel("confidence", fontsize=7)
            ax.set_xticks([0.2, 0.6, 1]); ax.set_yticks([0, 0.5, 1])
    fig.text(0.005, 0.5, "accuracy in bin", rotation=90, va="center", fontsize=7, color=INK2)
    fig.suptitle("Bars: accuracy of the answers in each confidence bin (bins with fewer than 5 answers omitted). "
                 "Dashed: perfect calibration.\nLight red above a bar: overconfident (confidence above accuracy). "
                 "Darker top inside a bar: underconfident.", fontsize=7, color=INK2, y=0.995)
    fig.savefig(os.path.join(outdir, "reliability.png")); plt.close(fig)
    # Banking77 confusion matrix, 9B
    res = rep["OpenDecider 9B"]["banking77_8way"]
    cm = np.array(res["confusion"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.6 else "black")
    names = res["classes"]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    ax.set(xlabel="predicted", ylabel="true", title="Banking77 (8 intents), OpenDecider 9B")
    fig.savefig(os.path.join(outdir, "banking77_confusion_9b.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/metrics_report.json")
    ap.add_argument("--figures", default="", help="directory for the PNGs (needs matplotlib)")
    ap.add_argument("--figures-only", action="store_true", help="draw from an existing --out file")
    a = ap.parse_args()
    if a.figures_only:
        figures(json.load(open(a.out)), a.figures)
        return
    rep = {}
    for run, name in SYSTEMS:
        rep[name] = {b: evaluate(run, b) for b in BINARY + CLASSES + MCQ}
        print(name, {b: {k: round(v["value"], 3) for k, v in r["metrics"].items()} for b, r in rep[name].items()},
              flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    if a.figures:
        figures(rep, a.figures)


if __name__ == "__main__":
    main()
