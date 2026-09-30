"""Calibration / accuracy metrics over variable-K probability vectors (pure numpy).

Inputs everywhere: `probs` = list of 1-D arrays (one per question, K may differ), `labels` = list
of int (index of the correct option). Every metric is also available as a function of per-example
arrays (confidence, correctness, nll, brier) so the 1,000-resample bootstrap (docs/SPEC.md section 11)
is cheap.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

EPS = 1e-12
ECE_BINS = 15
COVERAGES = (0.5, 0.8, 0.95)


# ---------------------------------------------------------------------------- per-example arrays
def per_example(probs: Sequence, labels: Sequence[int]) -> dict[str, np.ndarray]:
    """conf = top-1 probability, correct = argmax == label, nll = -log p[label],
    brier = sum_k (p_k - onehot_k)^2 (multiclass Brier, range [0, 2])."""
    n = len(probs)
    if n != len(labels):
        raise ValueError(f"{n} prob vectors vs {len(labels)} labels")
    conf = np.empty(n); correct = np.empty(n); nll = np.empty(n); brier = np.empty(n)
    for i, (p, y) in enumerate(zip(probs, labels)):
        p = np.asarray(p, dtype=np.float64)
        y = int(y)
        if not 0 <= y < p.shape[0]:
            raise ValueError(f"label {y} out of range for K={p.shape[0]}")
        top = int(np.argmax(p))
        conf[i] = p[top]
        correct[i] = float(top == y)
        nll[i] = -math.log(max(p[y], EPS))
        sq = float(np.sum(p * p))
        brier[i] = sq - 2.0 * p[y] + 1.0
    return {"conf": conf, "correct": correct, "nll": nll, "brier": brier}


def _acc(pe):
    return float(pe["correct"].mean()) if len(pe["correct"]) else float("nan")


def _nll(pe):
    return float(pe["nll"].mean()) if len(pe["nll"]) else float("nan")


def _brier(pe):
    return float(pe["brier"].mean()) if len(pe["brier"]) else float("nan")


def _ece(pe, n_bins: int = ECE_BINS):
    """Equal-mass binning: sort by confidence, split into n_bins groups of (nearly) equal size
    (tied confidences are kept together, so bins with ties can be larger)."""
    conf, correct = pe["conf"], pe["correct"]
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf, kind="stable")
    cs = conf[order]
    pos_bin = np.concatenate([np.full(len(c), b) for b, c in enumerate(np.array_split(np.arange(n), min(n_bins, n)))])
    # tied confidences stay in one bin (the bin of their first sorted position), otherwise an
    # arbitrary tie order could split right/wrong examples into different bins
    _, first, inv = np.unique(cs, return_index=True, return_inverse=True)
    b = pos_bin[first][inv]
    cor = correct[order]
    acc_sum = np.bincount(b, weights=cor, minlength=n_bins)
    conf_sum = np.bincount(b, weights=cs, minlength=n_bins)
    return float(np.abs(acc_sum - conf_sum).sum() / n)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties sharing the mean rank."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts          # 0-based first position of each tie group
    return (start + (counts + 1) / 2.0)[inv]


def _auroc(pe):
    """AUROC of confidence as a score for correctness (Mann-Whitney U / (n_pos * n_neg))."""
    conf, correct = pe["conf"], pe["correct"].astype(bool)
    n_pos, n_neg = int(correct.sum()), int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(conf)
    u = r[correct].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _selective(pe, coverage: float):
    """Accuracy on the ceil(coverage * n) most confident examples."""
    conf, correct = pe["conf"], pe["correct"]
    n = len(conf)
    if n == 0:
        return float("nan")
    m = max(1, int(math.ceil(coverage * n - 1e-9)))
    order = np.argsort(-conf, kind="stable")
    return float(correct[order[:m]].mean())


def _wrap(pe_fn: Callable, name: str) -> Callable:
    def f(probs, labels):
        return pe_fn(per_example(probs, labels))
    f._pe = pe_fn
    f.__name__ = name
    return f


accuracy = _wrap(_acc, "accuracy")
nll = _wrap(_nll, "nll")
brier = _wrap(_brier, "brier")
ece = _wrap(_ece, "ece")
auroc = _wrap(_auroc, "auroc")


def selective_accuracy(probs, labels, coverage: float) -> float:
    return _selective(per_example(probs, labels), coverage)


def _selective_fn(c: float) -> Callable:
    return _wrap(lambda pe: _selective(pe, c), f"selective_acc@{int(round(c * 100))}")


METRICS: dict[str, Callable] = {"accuracy": accuracy, "nll": nll, "brier": brier, "ece": ece, "auroc": auroc}
for _c in COVERAGES:
    METRICS[f"selective_acc@{int(round(_c * 100))}"] = _selective_fn(_c)


# ---------------------------------------------------------------------------- bootstrap
def _percentile_ci(vals: np.ndarray, alpha: float) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def bootstrap_ci(metric_fn: Callable, probs, labels, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over questions. Returns (point, lo, hi)."""
    N = len(probs)
    point = metric_fn(probs, labels)
    if N == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pe_fn = getattr(metric_fn, "_pe", None)
    pe = per_example(probs, labels) if pe_fn is not None else None
    vals = np.empty(n)
    for b in range(n):
        ii = rng.integers(0, N, size=N)
        if pe_fn is not None:
            vals[b] = pe_fn({k: v[ii] for k, v in pe.items()})
        else:
            vals[b] = metric_fn([probs[j] for j in ii], [labels[j] for j in ii])
    lo, hi = _percentile_ci(vals, alpha)
    return point, lo, hi


def bootstrap_mean_ci(values: Sequence[float], n: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) of a plain list of numbers (used for probe aggregates and latency)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    lo, hi = _percentile_ci(means, alpha)
    return float(v.mean()), lo, hi


def summarize(probs, labels, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Every metric with a (1 - alpha) percentile-bootstrap CI: {name: {value, lo, hi}, "n": N}."""
    out: dict = {"n": len(probs)}
    if len(probs) == 0:
        return out
    pe = per_example(probs, labels)
    rng = np.random.default_rng(seed)
    vals = {name: np.empty(n_boot) for name in METRICS}
    for b in range(n_boot):
        ii = rng.integers(0, len(probs), size=len(probs))
        s = {k: v[ii] for k, v in pe.items()}
        for name, fn in METRICS.items():
            vals[name][b] = fn._pe(s)
    for name, fn in METRICS.items():
        lo, hi = _percentile_ci(vals[name], alpha)
        out[name] = {"value": fn._pe(pe), "lo": lo, "hi": hi}
    return out


# ---------------------------------------------------------------------------- temperature scaling
def _pad(logits_list) -> tuple[np.ndarray, np.ndarray]:
    K = max(len(l) for l in logits_list)
    Z = np.full((len(logits_list), K), -np.inf)
    for i, l in enumerate(logits_list):
        Z[i, : len(l)] = np.asarray(l, dtype=np.float64)
    return Z, np.isfinite(Z)


def _nll_at(Z: np.ndarray, y: np.ndarray, beta: float) -> float:
    s = Z * beta
    m = np.max(s, 1, keepdims=True)
    lse = (m[:, 0] + np.log(np.sum(np.exp(s - m), 1)))
    return float(np.mean(lse - s[np.arange(len(y)), y]))


def fit_temperature(logits_list, labels, t_min: float = 0.05, t_max: float = 20.0, iters: int = 100) -> float:
    """Single temperature T minimising held-out NLL of softmax(logits / T).

    NLL is convex in beta = 1/T (log-sum-exp of a linear function), so golden-section search on
    beta in [1/t_max, 1/t_min] finds the global optimum."""
    Z, _ = _pad(logits_list)
    Z = np.where(np.isfinite(Z), Z, -1e30)
    y = np.asarray(labels, dtype=np.int64)
    a, b = 1.0 / t_max, 1.0 / t_min
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll_at(Z, y, c), _nll_at(Z, y, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = _nll_at(Z, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = _nll_at(Z, y, d)
        if b - a < 1e-7:
            break
    return float(1.0 / ((a + b) / 2))


def softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def apply_temperature(logits_list, T: float) -> list[np.ndarray]:
    return [softmax(np.asarray(l, dtype=np.float64) / T) for l in logits_list]
