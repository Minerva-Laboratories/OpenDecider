"""Learn-then-Test selective automation and LAC conformal sets, self-contained (numpy / scipy).

Fixed-sequence testing over confidence thresholds with exact binomial p-values (Bauer 1991; Angelopoulos et al.,
Learn then Test, arXiv:2110.01052): with probability >= 1 - delta over the calibration labels, the error rate among
cases whose score is >= the certified threshold is <= alpha. The threshold sequence is planned without the
calibration labels. Used by eval/decomposition_analysis.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


# ============================================================ tests
def binom_pvalue(k, n, alpha):
    """Exact p-value for H0: error rate >= alpha, given k errors among n (n == 0 -> 1)."""
    k, n = np.asarray(k), np.asarray(n)
    return np.where(n > 0, binom.cdf(k, np.maximum(n, 1), alpha), 1.0)


def counts_at(s, err, thresholds):
    """(n_t, k_t): cases with score >= t and errors among them, per threshold."""
    o = np.argsort(-s)
    ss, e = s[o], np.concatenate([[0], np.cumsum(err[o])])
    n = np.searchsorted(-ss, -np.asarray(thresholds), side="right")
    return n, e[n]


def fst_threshold(s_cal, err_cal, seq, alpha, delta):
    """Last threshold of `seq` (strict -> loose) whose p-value <= delta before the first failure; +inf if none."""
    seq = np.asarray(seq)
    if len(seq) == 0 or len(s_cal) == 0:
        return math.inf
    n, k = counts_at(np.asarray(s_cal), np.asarray(err_cal, float), seq)
    ok = binom_pvalue(k, n, alpha) <= delta
    if not ok[0]:
        return math.inf
    j = len(ok) if ok.all() else int(np.argmin(ok))
    return float(seq[j - 1])


def pass_prob(m, e, alpha, delta):
    """P(the exact test at level delta passes) with m cases of true error e."""
    m = np.asarray(m, int)
    e = np.clip(np.asarray(e, float), 1e-6, 1 - 1e-6)
    kmax = np.array([int(binom.ppf(delta, mm, alpha)) if mm > 0 else -1 for mm in m])
    kmax = np.where((m > 0) & (binom.cdf(kmax, np.maximum(m, 1), alpha) > delta), kmax - 1, kmax)
    return np.where(kmax >= 0, binom.cdf(kmax, np.maximum(m, 1), e), 0.0)


def plan_sequence(s_unl, s_fit, err_fit, n_cal, alpha, delta, levels=np.arange(1, 101) / 100):
    """Threshold sequence fixed without the check labels: unlabeled-score quantiles at coverage 1..100%, error per
    level estimated on the teach labels' out-of-fold predictions (Beta(1,1)-smoothed), started where the expected
    certified coverage is largest. Returns (sequence, start coverage, expected coverage)."""
    t = np.quantile(s_unl, 1 - levels)
    if s_fit is None or len(s_fit) == 0:
        o = np.sort(s_unl)[::-1]
        cm = np.cumsum(1 - o) / np.arange(1, len(o) + 1)
        e = cm[np.clip(np.round(levels * len(o)).astype(int) - 1, 0, len(o) - 1)]
    else:
        n_f, k_f = counts_at(np.asarray(s_fit), np.asarray(err_fit, float), t)
        e = (k_f + 1) / (n_f + 2)
    p = pass_prob(np.round(levels * n_cal).astype(int), e, alpha, delta)
    dc = np.diff(np.concatenate([[0], levels]))
    best, j0 = 0.0, 0
    for j in range(len(levels)):
        v = (dc[j:] * np.minimum.accumulate(p[j:])).sum()
        if p[j] > 0 and v > best:
            best, j0 = v, j
    return t[j0:], float(levels[j0]), float(best)


def lac_qhat(P_cal, y_cal, alpha):
    """Split-conformal LAC: sets {y : p_y >= 1 - qhat} cover the true answer with probability >= 1 - alpha
    (marginal over cases). Returns qhat (inf: every option)."""
    if len(y_cal) == 0:
        return math.inf
    y_cal = np.asarray(y_cal, int)
    sc = 1 - np.where(y_cal >= 0, P_cal[np.arange(len(y_cal)), np.maximum(y_cal, 0)], 0.0)   # -1: not an option
    n = len(sc)
    lv = math.ceil((n + 1) * (1 - alpha)) / n
    return math.inf if lv > 1 else float(np.quantile(sc, lv, method="higher"))

