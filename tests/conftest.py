import pytest
import torch

from tests.tiny import tiny_backbone
from opendecider.models import ModelConfig, build_model


@pytest.fixture(scope="session")
def bb():
    return tiny_backbone()


def make(bb, variant, **kw):
    torch.manual_seed(0)
    kw.setdefault("dropout", 0.0)
    cfg = ModelConfig(variant=variant, d_model=32, n_heads=4, decision_layers=2, interaction_layers=1, **kw)
    m = build_model(bb, cfg).eval()
    # V2 gates start at 0 (identity); open them so fusion paths are exercised by the tests.
    for n, p in m.named_parameters():
        if n.endswith("alpha_xattn") or n.endswith("alpha_dense"):
            p.data.fill_(0.5)
    return m
