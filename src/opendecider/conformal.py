"""Split-conformal prediction sets over caller-defined options (class-agnostic, like the rest of the head).

Calibration stores nonconformity scores of the true option on labelled data; at request time a level alpha turns them
into a threshold, and every option whose score is under it enters the set. If the calibration data and the requests
are exchangeable, the set contains the true option with probability >= 1 - alpha. Under distribution shift there is
no such guarantee: refit on the caller's own labels (POST /v1/calibrate stores scores in the profile).

Scores (p = probabilities renormalised over the caller's options):
  lac : 1 - p[y]                                      smallest average sets; may be empty (-> abstain)
  aps : total mass of options ranked at or above y     adaptive sets (larger when the model is unsure); never empty
"""
from __future__ import annotations

import math

import numpy as np

METHODS = ("lac", "aps")


def scores(P: np.ndarray, y: np.ndarray, method: str = "lac") -> np.ndarray:
    """Nonconformity score of the true option. P (n, K) rows sum to 1 over valid options (pad with 0); y (n,)."""
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y)
    py = P[np.arange(len(y)), y]
    if method == "lac":
        return 1.0 - py
    if method == "aps":
        return (P * (P >= py[:, None])).sum(1)
    raise ValueError(f"unknown conformal method {method!r}")


def threshold(cal_scores, alpha: float) -> float:
    """Finite-sample corrected quantile: the ceil((n+1)(1-alpha))-th smallest score; inf if n is too small."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def prediction_set(p, q: float, method: str = "lac") -> list[int]:
    """Indices of the options in the set, most probable first."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    if method == "lac":
        return [int(j) for j in order if 1.0 - p[j] <= q + 1e-12]
    if method == "aps":
        cum = np.cumsum(p[order])
        k = int(np.searchsorted(cum, q - 1e-12)) + 1          # smallest prefix whose mass reaches q
        return [int(j) for j in order[: min(k, len(p))]]
    raise ValueError(f"unknown conformal method {method!r}")


def pack(P, y, types, methods=METHODS, decimals: int = 5) -> dict:
    """Scores grouped by question type (Mondrian by type), sorted and rounded, for storage in a checkpoint."""
    P_list, y, types = list(P), np.asarray(y), list(types)
    out = {}
    for m in methods:
        out[m] = {}
        for t in sorted(set(types)):
            idx = [i for i, x in enumerate(types) if x == t]
            K = max(len(P_list[i]) for i in idx)
            Pm = np.zeros((len(idx), K))
            for r, i in enumerate(idx):
                Pm[r, : len(P_list[i])] = P_list[i]
            out[m][t] = np.round(np.sort(scores(Pm, y[idx], m)), decimals).tolist()
    return out


def coverage(P, y, cal_scores, alpha: float, method: str = "lac") -> dict:
    """Empirical coverage, mean set size and empty-set rate of the sets built from cal_scores on (P, y)."""
    q = threshold(cal_scores, alpha)
    sets = [prediction_set(p, q, method) for p in P]
    hit = [int(t) in s for s, t in zip(sets, y)]
    return {"coverage": float(np.mean(hit)), "mean_size": float(np.mean([len(s) for s in sets])),
            "empty": float(np.mean([len(s) == 0 for s in sets])), "singleton": float(np.mean([len(s) == 1 for s in sets])),
            "threshold": q, "n_cal": len(cal_scores)}
