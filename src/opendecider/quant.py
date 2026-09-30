"""8-bit / 4-bit weight quantization and int8 cache storage.

Weights
  int8: weight-only, symmetric per-output-channel absmax. Dequantised inside the matmul.
        Measured on Orin (0.8B, 1024-token state): hidden-state cosine vs bf16 mean 0.9992,
        min 0.996; 0.98 GB vs 1.41 GB; ~25% slower (prefill is compute-bound, so weight
        quantization saves memory, not time).
        bitsandbytes LLM.int8 is NOT used: its cuBLASLt int8 matmul fails on sm_87
        ("BLAS Lt API failed with status 15").
  nf4:  bitsandbytes 4-bit (applied at load time in backbone.py). On 0.8B it distorts hidden
        states noticeably (cosine min 0.78), so prefer it only for the 2B/9B backbones.

Caches ("all KV caches in 8 bit")
  Int8Tensor stores any cache tensor as int8 + one bf16/fp32 scale per row over the last dim
  (per token, per head for K/V). Used for: the state memory S, precomputed cross-attention
  K/V in fusion blocks, and V0's prefix K/V for the full-attention layers.
  The Gated-DeltaNet recurrent/conv states are not KV caches (fixed size, fp32 by config);
  they stay in native dtype unless `quantize_linear_state=True`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- int8 tensors (caches)
@dataclass
class Int8Tensor:
    q: torch.Tensor          # int8, same shape as the original
    scale: torch.Tensor      # shape[:-1] + (1,), original dtype
    dtype: torch.dtype

    @classmethod
    def quantize(cls, x: torch.Tensor) -> "Int8Tensor":
        xf = x.float()
        scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(xf / scale).clamp_(-127, 127).to(torch.int8)
        return cls(q, scale.to(x.dtype if x.dtype != torch.float32 else torch.float32), x.dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dt = dtype or self.dtype
        return self.q.to(dt) * self.scale.to(dt)

    @property
    def shape(self):
        return self.q.shape

    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * self.scale.element_size()

    def to(self, device) -> "Int8Tensor":
        return Int8Tensor(self.q.to(device), self.scale.to(device), self.dtype)

    def index_select(self, dim: int, idx: torch.Tensor) -> "Int8Tensor":
        return Int8Tensor(self.q.index_select(dim, idx), self.scale.index_select(dim, idx), self.dtype)


def maybe_quantize(x: torch.Tensor, enabled: bool):
    return Int8Tensor.quantize(x) if enabled else x


def materialize(x, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, Int8Tensor):
        return x.dequantize(dtype)
    return x if dtype is None else x.to(dtype)


# ---------------------------------------------------------------- int8 weights
class Int8Linear(nn.Module):
    """Frozen weight-only int8 linear (per-output-channel absmax)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).clamp_(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(lin.weight.dtype))
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.data.clone())
        else:
            self.bias = None
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self) -> torch.Tensor:  # some HF code paths read .weight.dtype/.device
        return self.qweight.to(self.scale.dtype) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 weight-only"


class Int8Embedding(nn.Module):
    """Frozen int8 embedding table (per-row absmax). Saves ~0.25 GB on the 0.8B backbone."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        w = emb.weight.data.float()
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).clamp_(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(emb.weight.dtype))
        self.padding_idx = emb.padding_idx
        self.num_embeddings, self.embedding_dim = emb.num_embeddings, emb.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight.to(self.scale.dtype) * self.scale

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.qweight[ids].to(self.scale.dtype) * self.scale[ids]


def quantize_int8_(module: nn.Module, skip: tuple[str, ...] = ()) -> nn.Module:
    """Replace every nn.Linear / nn.Embedding under `module` in place (skipping names in `skip`)."""
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
        elif isinstance(child, nn.Embedding):
            setattr(module, name, Int8Embedding(child))
        else:
            quantize_int8_(child, skip)
    return module
