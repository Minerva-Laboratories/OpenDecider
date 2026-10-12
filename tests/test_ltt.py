"""eval/ltt.py: Learn-then-Test thresholds keep the selected error <= alpha with probability >= 1 - delta; LAC
coverage."""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.ltt import fst_threshold, lac_qhat, plan_sequence  # noqa: E402


def test_fst_validity_by_simulation():
    rng = np.random.default_rng(0)
    alpha, delta, viol, runs = 0.10, 0.10, 0, 200
    for _ in range(runs):
        def draw(n):
            s = rng.uniform(0.2, 1, n)
            return s, (rng.uniform(0, 1, n) > s ** 0.5).astype(float)
        s_u, _ = draw(2000); s_f, e_f = draw(300); s_c, e_c = draw(400)
        seq, _, _ = plan_sequence(s_u, s_f, e_f, len(s_c), alpha, delta)
        thr = fst_threshold(s_c, e_c, seq, alpha, delta)
        if math.isinf(thr):
            continue
        s_p, e_p = draw(100000)
        viol += int(e_p[s_p >= thr].mean() > alpha)
    assert viol / runs <= delta + 0.05


def test_lac_coverage():
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(100):
        Z = rng.normal(0, 1, (600, 6)); y = rng.integers(0, 6, 600); Z[np.arange(600), y] += 1.5
        P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        q = lac_qhat(P[:300], y[:300], 0.1)
        hits.append((P[300:] >= 1 - q)[np.arange(300), y[300:]].mean())
    assert np.mean(hits) >= 0.89
