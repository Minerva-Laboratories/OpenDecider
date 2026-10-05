"""Load pre-quantized (AWQ) Qwen3.5 checkpoints in llm-compressor's compressed-tensors `pack-quantized` format.

Each quantized Linear stores weight_packed (int32, 8 codes per word along the input dim, little-endian nibbles),
weight_scale (bf16, one per group) and weight_shape. Symmetric int4 codes are stored with a +8 offset, so
W = (code - 8) * scale. The AWQ activation scales are already folded into the preceding norms by the quantizer,
so the layers load as plain int4 weights. They run on GemLite's A16W4 Triton kernel with the checkpoint's own
codes and scales (zero point 8): no re-quantization. Unquantized tensors (norms, conv, A_log, dt_bias and the
small linear-attention gate projections) load as stored; the embedding table becomes int8 as in the other modes.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .quant import quantize_table


def unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """(N, K/8) int32 -> (N, K) uint8 codes in [0, 15]."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    codes = (packed.unsqueeze(-1) >> shifts) & 0xF                      # (N, K/8, 8), nibble i = column 8j + i
    return codes.reshape(packed.shape[0], -1)[:, :K].to(torch.uint8)


def gemlite_from_packed(packed, scale, shape, dtype, device) -> nn.Module:
    from gemlite import DType, GemLiteLinear
    N, K = int(shape[0]), int(shape[1])
    group = K // scale.shape[1]
    codes = unpack_int4(packed.to(device), K)
    dt = DType.BF16 if dtype == torch.bfloat16 else DType.FP16
    lin = GemLiteLinear(W_nbits=4, group_size=group, in_features=K, out_features=N, input_dtype=dt, output_dtype=dt)
    lin.pack(codes, scale.to(device=device, dtype=dtype), 8, bias=None)
    lin.in_features, lin.out_features = K, N
    return lin


def dequant(packed, scale, shape) -> torch.Tensor:
    N, K = int(shape[0]), int(shape[1])
    codes = unpack_int4(packed, K).float() - 8
    return (codes.view(N, scale.shape[1], -1) * scale.float()[..., None]).view(N, K)


@torch.no_grad()
def load_qwen35_ct(path: str, device: str = "cuda", dtype=torch.bfloat16, table_quant: str = "int8"):
    """(text model with GemLite int4 linears, None) from a compressed-tensors checkpoint directory."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(path)
    c = getattr(cfg, "text_config", cfg)
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    handles = [safe_open(f, "pt", device="cpu") for f in files]
    where = {k: h for h in handles for k in h.keys()}
    pre = "model.language_model."
    with torch.device("meta"):
        lm = Qwen3_5TextModel(c)
    lm.to(dtype)

    def get(name):
        return where[pre + name].get_tensor(pre + name)

    def fill(module: nn.Module, prefix: str):
        # quantized linears -> GemLite; everything else as stored
        for cname, child in list(module.named_children()):
            full = f"{prefix}{cname}"
            if isinstance(child, nn.Linear) and pre + full + ".weight_packed" in where:
                setattr(module, cname, gemlite_from_packed(get(full + ".weight_packed"), get(full + ".weight_scale"),
                                                           get(full + ".weight_shape"), dtype, device))
            elif len(list(child.children())):
                fill(child, full + ".")
            else:
                child.to_empty(device=device)
                for pn, p in child.named_parameters(recurse=False):
                    p.copy_(get(f"{full}.{pn}").to(p.dtype))
        for pn, p in module.named_parameters(recurse=False):
            if p.device.type == "meta":
                setattr(module, pn, nn.Parameter(get(prefix + pn).to(device=device, dtype=dtype), requires_grad=False))

    lm.embed_tokens.to_empty(device=device)
    lm.embed_tokens.weight.copy_(get("embed_tokens.weight"))
    lm.embed_tokens = quantize_table(lm.embed_tokens, table_quant)
    for i, layer in enumerate(lm.layers):
        fill(layer, f"layers.{i}.")
    lm.norm.to_empty(device=device)
    lm.norm.weight.copy_(get("norm.weight"))
    lm.rotary_emb = type(lm.rotary_emb)(config=c, device=device)
    head = None
    if not getattr(cfg, "tie_word_embeddings", True):
        # untied LM head (e.g. Qwen3.5-9B): stored unquantized as `lm_head.weight`; int8 per row, like the GGUF path
        name = "lm_head.weight" if "lm_head.weight" in where else pre + "lm_head.weight"
        w = where[name].get_tensor(name)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=device, dtype=dtype)
        lin.weight.copy_(w.to(device=device, dtype=dtype))
        head = quantize_table(lin, table_quant)
        del lin, w
    return lm.eval(), head


def is_compressed_tensors(path: str) -> bool:
    try:
        q = json.load(open(os.path.join(path, "config.json"))).get("quantization_config", {})
    except (OSError, ValueError):
        return False
    return q.get("quant_method") == "compressed-tensors" and q.get("format") == "pack-quantized"
