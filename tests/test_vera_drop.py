"""Serving leaves out VeRA adapters whose updates are below bf16 resolution (decider.prepare_inference_)."""
import torch

from opendecider import decider as D
from tests.conftest import make
from tests.tiny import tiny_backbone


def _model(b_std):
    bb = tiny_backbone(seed=5, layers=(4, "final"))
    m = make(bb, "v3", v3_features="branched", v3_vera_layers=1, v3_vera_scope="rows_top", v3_vera_rank=8,
             v3_row_format="answer", v3_list_cap=255)
    with torch.no_grad():
        for p in m.vera.b.values():
            p.normal_(0, b_std)
    return m


def test_relative_size_matches_weight_ratio():
    m = _model(0.0)
    assert m.vera.relative_size() == 0.0
    m = _model(0.5)
    mod, tag = m.vera.targets[0]
    k = tag.replace(".", "_")
    i, o = mod.in_features, mod.out_features
    dW = (getattr(m.vera, f"B_{i}x{o}") * m.vera.b[k][:, None]) @ (getattr(m.vera, f"A_{i}x{o}") * m.vera.d[k][:, None])
    exact = float(dW.norm() / mod.weight.float().norm())
    assert m.vera.relative_size(n=4096) >= 0.8 * exact          # the max over linears, each estimated within ~5%


def test_prepare_inference_drops_inert_vera(monkeypatch):
    monkeypatch.delenv("OPENDECIDER_VERA", raising=False)
    m = _model(1e-7)
    D.prepare_inference_(m)
    assert getattr(m, "vera", None) is None and m.vera_dropped
    m = _model(0.5)
    D.prepare_inference_(m)
    assert m.vera is not None and m.vera_relative_size > D.VERA_INERT
    monkeypatch.setenv("OPENDECIDER_VERA", "0")
    m = _model(0.5)
    D.prepare_inference_(m)
    assert getattr(m, "vera", None) is None
