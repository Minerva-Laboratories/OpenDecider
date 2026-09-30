import numpy as np
import pytest

from eval.tree_head import features, fit_linear, fit_tree


def _items(n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        K = int(rng.integers(2, 6))
        y = int(rng.integers(K))
        lps = []
        for s in range(2):
            z = rng.normal(size=K); z[y] += 1.5 * (s + 1)
            lps.append(z - np.logaddexp.reduce(z))
        out.append((lps, np.eye(K)[y]))
    return out


def test_features_permutation_equivariant():
    lps, _ = _items(1)[0]
    perm = np.random.default_rng(1).permutation(len(lps[0]))
    np.testing.assert_allclose(features([lp[perm] for lp in lps]), features(lps)[perm])


@pytest.mark.parametrize("mode", ["linear", "gbdt", "forest"])
def test_heads_return_distributions(mode):
    pytest.importorskip("lightgbm")
    tr = _items()
    head = fit_linear(tr) if mode == "linear" else fit_tree(tr, mode)
    for lps, _ in _items(10, seed=3):
        p = head(lps)
        assert p.shape == lps[0].shape and abs(p.sum() - 1) < 1e-6 and (p >= 0).all()
