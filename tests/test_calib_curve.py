import numpy as np
import pytest

from eval.calib_curve import feats, fit_heads, pad


def _items(n=80, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        K = int(rng.integers(2, 6)); y = int(rng.integers(K))
        z = rng.normal(size=K); z[y] += 2.0
        out.append((z - np.logaddexp.reduce(z), np.eye(K)[y]))
    return out


def test_feats_match_unpadded_order():
    L, _, M = pad(_items(5))
    assert feats(L, M).shape == (M.sum(), 5)


def test_calibrators_return_distributions_and_are_order_equivariant():
    pytest.importorskip("lightgbm")
    heads = fit_heads(*pad(_items()))
    L, _, M = pad(_items(10, seed=2))
    perm = np.array([1, 0, 2, 3, 4])[: L.shape[1]]
    for h, fn in heads.items():
        P = fn(L, M)
        assert np.allclose(P.sum(1), 1, atol=1e-6) and (P[~M] == 0).all(), h
        full = M.all(1)       # questions with every slot filled can be permuted freely
        np.testing.assert_allclose(fn(L[full][:, perm], M[full])[:, perm], P[full], atol=1e-6, err_msg=h)


def test_top_label_preserves_argmax():
    pytest.importorskip("lightgbm")
    from eval.calib_curve import top_label
    L, _, M = pad(_items(200, seed=5))
    Q = top_label(lambda c: np.full_like(c, 0.05), L, M, 1.0)       # an absurdly low confidence map
    P = np.where(M, np.exp(L), 0)
    assert (Q.argmax(1) == P.argmax(1)).all() and np.allclose(Q.sum(1), 1)
