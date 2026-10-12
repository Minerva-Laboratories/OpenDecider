"""Export a checkpoint's VeRA adapters as an exactly equivalent LoRA GGUF for llama.cpp (P8.2 feasibility probe).

VeRA (src/opendecider/vera.py): delta W = diag(b) B diag(d) A, A (r x in) and B (out x r) regenerated from seed 0.
As LoRA: lora_a = diag(d) A (r x in), lora_b = diag(b) B (out x r), scale 1 (alpha 0 in llama.cpp means scale =
adapter scale). Tensor names follow the GGUF the backbone was loaded from; linear-attention V heads are put in the
GGUF's tiled order (the converter's reorder) so the adapter matches the stored weights.

    python scripts/llamacpp_probe/export_vera_lora.py runs/x9b-clean3/model.pt /mnt/ext/opendecider/p8/llamacpp/vera_9b.gguf
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from opendecider.gguf_load import _reorder_v_heads  # noqa: E402


def to_gguf_order(name, la, lb, nk, nv, hk, hv):
    """LoRA factors of one linear-attention projection, HF (grouped V heads) -> GGUF (tiled) order, so that
    lb @ la equals the GGUF-stored weight's delta. Other modules are returned unchanged."""
    nvk = nv // nk
    if nk == nv or "linear_attn" not in name:
        return la, lb
    if name.endswith("in_proj_qkv"):
        q = nk * hk
        lb = torch.cat([lb[:2 * q], _reorder_v_heads(lb[2 * q:], 0, nk, nvk, hv)], 0)
    elif name.endswith("in_proj_z"):
        lb = _reorder_v_heads(lb, 0, nk, nvk, hv)
    elif name.endswith("out_proj"):
        la = _reorder_v_heads(la, 1, nk, nvk, hv)
    return la, lb


def main(ckpt, out, dtype="f16"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    bb, mc = ck["backbone_cfg"], ck["model_cfg"]
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(os.path.dirname(bb["path"]))
    c = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        lm = Qwen3_5TextModel(c)

    # regenerate A/B exactly as VeRA.__init__ does (same module order, same generator), on CPU; the meta model
    # only provides shapes
    import math
    rank, n = mc["v3_vera_rank"], mc["v3_vera_layers"]
    targets = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
               "down_proj")
    layers = lm.layers; start = len(layers) - n
    mods, shapes = [], {}
    for li in range(start, len(layers)):
        for name, mod in layers[li].named_modules():
            if name.split(".")[-1] in targets and hasattr(mod, "in_features"):
                mods.append((li, name, mod.in_features, mod.out_features))
                shapes[(mod.in_features, mod.out_features)] = True
    g = torch.Generator().manual_seed(0)
    AB = {}
    for (i, o) in shapes:
        a = torch.empty(rank, i); b = torch.empty(o, rank)
        a.uniform_(-math.sqrt(6 / i), math.sqrt(6 / i), generator=g); b.uniform_(-math.sqrt(6 / rank), math.sqrt(6 / rank), generator=g)
        AB[(i, o)] = (a, b)
    sd = ck["state_dict"]
    nk, nv = c.linear_num_key_heads, c.linear_num_value_heads
    hk, hv = c.linear_key_head_dim, c.linear_value_head_dim
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, c.num_hidden_layers + 1)
    w = gguf.GGUFWriter(out, "qwen35")
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 0.0)
    np_dt = np.float16 if dtype == "f16" else np.float32
    for li, name, i, o in mods:
        k = f"L{li}_{name}".replace(".", "_")
        d, bvec = sd[f"vera.d.{k}"].float(), sd[f"vera.b.{k}"].float()
        A, B = AB[(i, o)]
        la = A * d[:, None]                         # r x in
        lb = B * bvec[:, None]                      # out x r
        la, lb = to_gguf_order(name, la, lb, nk, nv, hk, hv)
        base = tmap.get_name(f"model.layers.{li}.{name}")
        assert base is not None, name
        w.add_tensor(base + ".weight.lora_a", la.numpy().astype(np_dt))
        w.add_tensor(base + ".weight.lora_b", lb.numpy().astype(np_dt))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"{len(mods)} adapted linears (layers {start}..{len(layers) - 1}, rank {rank}) -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
