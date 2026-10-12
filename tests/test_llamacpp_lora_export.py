"""VeRA -> LoRA GGUF export (scripts/llamacpp_probe/export_vera_lora.py): head reordering matches the GGUF weights."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "llamacpp_probe"))
from export_vera_lora import to_gguf_order  # noqa: E402

from opendecider.gguf_load import _reorder_v_heads, _ungroup  # noqa: E402

NK, NV, HK, HV, R = 2, 4, 3, 5, 7


def _to_gguf_weight(name, W):
    """The converter's forward transform of an HF weight (inverse of gguf_load._convert's reorder)."""
    nvk = NV // NK
    if name.endswith("in_proj_qkv"):
        q = NK * HK
        return torch.cat([W[:2 * q], _reorder_v_heads(W[2 * q:], 0, NK, nvk, HV)], 0)
    if name.endswith("in_proj_z"):
        return _reorder_v_heads(W, 0, NK, nvk, HV)
    if name.endswith("out_proj"):
        return _reorder_v_heads(W, 1, NK, nvk, HV)
    return W


def test_reorder_roundtrip_matches_loader():
    W = torch.randn(NV * HV, 11)
    assert torch.equal(_ungroup(_to_gguf_weight("linear_attn.in_proj_z", W), 0, NK, NV // NK, HV), W)


def test_lora_delta_in_gguf_order():
    torch.manual_seed(0)
    shapes = {"linear_attn.in_proj_qkv": (2 * NK * HK + NV * HV, 11), "linear_attn.in_proj_z": (NV * HV, 11),
              "linear_attn.out_proj": (11, NV * HV), "mlp.up_proj": (13, 11)}
    for name, (o, i) in shapes.items():
        la, lb = torch.randn(R, i), torch.randn(o, R)
        ga, gb = to_gguf_order(name, la, lb, NK, NV, HK, HV)
        assert torch.allclose(gb @ ga, _to_gguf_weight(name, lb @ la), atol=1e-5), name
