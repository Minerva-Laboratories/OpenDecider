"""Load a Qwen3.5 text backbone from a llama.cpp GGUF file (quantized weights; no bf16 download needed).

The GGUF was written by llama.cpp's converter (conversion/qwen.py: Qwen3NextModel, _LinearAttentionVReorderBase,
Qwen3_5TextModel). This module inverts its tensor transforms:
  - names: gguf TensorNameMap for MODEL_ARCH.QWEN35, plus A_log <-> ssm_a and dt_bias <-> ssm_dt.bias
  - RMSNorm weights were stored as w + 1 (HF computes (1 + w) x); the gated linear-attention norm was not
  - A_log was stored as -exp(A_log)
  - conv1d had its singleton dimension squeezed
  - linear-attention V heads were reordered from grouped-by-K-head to tiled order (when K heads != V heads)
Weights are dequantized one decoder layer at a time and immediately re-quantized to int8 (or kept bf16), so peak
memory is about one layer in bf16 plus the model at its final precision. Requires `gguf` from llama.cpp's gguf-py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .quant import Int8Embedding, Int8Linear, gemlite_quantize_, quantize_int8_


def _reorder_v_heads(t: torch.Tensor, dim: int, n_a: int, n_b: int, head: int) -> torch.Tensor:
    """llama.cpp's _reorder_v_heads: view dim as (n_a, n_b, head) and swap the first two."""
    shape = list(t.shape)
    dim = dim % len(shape)
    t = t.reshape(*shape[:dim], n_a, n_b, head, *shape[dim + 1:])
    perm = list(range(t.dim()))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def _ungroup(t, dim, nk, nvk, head):
    """Inverse of the converter's reorder (grouped -> tiled): tiled (nvk, nk, head) -> grouped (nk, nvk, head)."""
    return _reorder_v_heads(t, dim, nvk, nk, head)


class _GGUF:
    def __init__(self, path):
        import gguf
        self.gguf = gguf
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name) -> torch.Tensor:
        t = self.t[name]
        if t.tensor_type.name in ("F32", "F16", "BF16"):
            a = np.asarray(t.data, dtype=np.float32)
        else:
            a = self.gguf.quants.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _hf_to_gguf_name(tmap, name: str) -> str:
    if name.endswith(".A_log"):
        return f"blk.{name.split('.')[2]}.ssm_a"
    if name.endswith(".dt_bias"):
        return f"blk.{name.split('.')[2]}.ssm_dt.bias"
    base, suffix = (name[:-7], ".weight") if name.endswith(".weight") else (name[:-5], ".bias")
    g = tmap.get_name(base)
    if g is None:
        raise KeyError(f"no GGUF name for {name}")
    return g + suffix


def _convert(name: str, x: torch.Tensor, shape, c) -> torch.Tensor:
    """GGUF tensor (float32, numpy row-major = reversed ggml dims) -> HF parameter."""
    nk, nv = c.linear_num_key_heads, c.linear_num_value_heads
    hk, hv = c.linear_key_head_dim, c.linear_value_head_dim
    nvk = nv // nk
    reorder = nk != nv and ".linear_attn." in name
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        x = x - 1.0
    if name.endswith(".A_log"):
        x = torch.log(-x)
    if name.endswith("conv1d.weight"):
        x = x.reshape(shape[0], shape[-1])                                     # (channels, kernel)
    if reorder:
        if name.endswith("in_proj_qkv.weight"):
            q = nk * hk
            x = x.reshape(shape)
            x = torch.cat([x[:2 * q], _ungroup(x[2 * q:], 0, nk, nvk, hv)], 0)
        elif name.endswith("in_proj_z.weight"):
            x = _ungroup(x.reshape(shape), 0, nk, nvk, hv)
        elif name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
            x = _ungroup(x.reshape(shape), 0, nk, nvk, 1)
        elif name.endswith((".A_log", ".dt_bias")):
            x = _ungroup(x.reshape(-1, 1), 0, nk, nvk, 1).reshape(-1)
        elif name.endswith("conv1d.weight"):
            qk = 2 * nk * hk
            x = torch.cat([x[:qk], _ungroup(x[qk:], 0, nk, nvk, hv)], 0)
        elif name.endswith("out_proj.weight"):
            x = _ungroup(x.reshape(shape), 1, nk, nvk, hv)
    return x.reshape(shape)


@torch.no_grad()
def load_qwen35_gguf(gguf_path: str, config_dir: str, device: str = "cuda", dtype=torch.bfloat16,
                     weight_quant: str = "int8") -> nn.Module:
    """(text model: embed_tokens, layers, norm; untied LM head or None) from a GGUF file."""
    import gguf
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(config_dir)
    c = getattr(cfg, "text_config", cfg)
    src = _GGUF(gguf_path)
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, c.num_hidden_layers + 1)
    with torch.device("meta"):
        lm = Qwen3_5TextModel(c)
    lm.to(dtype)

    def fill(module: nn.Module, prefix: str):
        module.to_empty(device=device)
        for n, p in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True)):
            full = f"model.{prefix}{n}"
            if isinstance(p, nn.Parameter) or n in dict(module.named_parameters()):
                x = _convert(full, src.get(_hf_to_gguf_name(tmap, full)), tuple(p.shape), c)
                p.copy_(x.to(p.dtype))

    fill(lm.embed_tokens, "embed_tokens.")
    if weight_quant != "none":
        lm.embed_tokens = Int8Embedding(lm.embed_tokens)
    for i, layer in enumerate(lm.layers):                     # one layer in bf16 at a time
        fill(layer, f"layers.{i}.")
        if weight_quant == "int8":
            quantize_int8_(layer)
        elif weight_quant in ("w4", "w8"):
            gemlite_quantize_(layer, 4 if weight_quant == "w4" else 8)
        torch.cuda.empty_cache()
    fill(lm.norm, "norm.")
    # non-persistent buffers (rotary inv_freq) are recomputed from the config
    lm.rotary_emb = type(lm.rotary_emb)(config=c, device=device)
    head = None
    if not getattr(cfg, "tie_word_embeddings", True) and "output.weight" in src.t:   # untied LM head (9B)
        W = src.get("output.weight").view(c.vocab_size, c.hidden_size)
        head = nn.Linear(c.hidden_size, c.vocab_size, bias=False, device="meta")
        head.weight = nn.Parameter(W.to(device=device, dtype=dtype), requires_grad=False)
        del W
        if weight_quant != "none":
            head = Int8Linear(head)
        torch.cuda.empty_cache()
    return lm.eval(), head
