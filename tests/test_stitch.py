"""Model stitching: a V3 trunk reads another backbone through a ridge-fitted linear map (scripts/stitch_backbone.py)."""
import importlib.util
import os

import torch

from opendecider.batching import Question
from tests.conftest import make
from tests.test_models import OPTS, STATE
from tests.test_masks import _mem
from tests.tiny import tiny_backbone

spec = importlib.util.spec_from_file_location("stitch", os.path.join(os.path.dirname(__file__), "..", "scripts", "stitch_backbone.py"))
stitch = importlib.util.module_from_spec(spec); spec.loader.exec_module(stitch)


def test_stitch_onto_same_backbone_recovers_the_original_model():
    """A == B (same backbone, one feature layer): the ridge map must be ~identity on unit features, and the stitched
    model's logits must match the original's (in_norm is scale-invariant)."""
    bb = tiny_backbone(seed=4, layers=("final",), kv_quant="none")
    orig = make(bb, "v3", v3_features="branched").eval()
    stitched = make(bb, "v3", v3_features="branched", v3_layer_combine="stitch", v3_stitch_dim=bb.hidden_size).eval()
    sd = stitched.state_dict()
    for k, v in orig.state_dict().items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    qs_all = [[Question("Which team?", OPTS)], [Question("Is it urgent?", ["yes", "no"])], [Question("Pick one", ["red", "green", "blue"])]]
    D = bb.hidden_size + 1
    XtX = torch.zeros(D, D, dtype=torch.float64); XtY = torch.zeros(D, bb.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        for qs in qs_all:
            s, _, q, _, o, _ = orig._conditioned_feats(qs, _mem(orig, bb, [STATE]))
            Y = stitch.unit(torch.cat([s, q, o]).double())
            w, _, wq, _, wo, _ = stitched._conditioned_feats_raw(qs, _mem(stitched, bb, [STATE]))
            X = stitch.stitch_inputs(torch.cat([w, wq, wo]), 1, bb.hidden_size).double()
            XtX += X.T @ X; XtY += X.T @ Y
        W = stitch.solve_ridge(XtX, XtY, 1e-6)
    sd["stitch.weight"] = W[:-1].T.float(); sd["stitch.bias"] = W[-1].float()
    stitched.load_state_dict(sd)
    with torch.no_grad():
        for qs in qs_all:
            a = orig.run(qs, _mem(orig, bb, [STATE]))[1].logits
            b = stitched.run(qs, _mem(stitched, bb, [STATE]))[1].logits
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
