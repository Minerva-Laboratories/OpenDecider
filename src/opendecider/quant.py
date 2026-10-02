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


class Int4Tensor(Int8Tensor):
    """Symmetric int4 with one scale per group of G values along the last dim; two values per byte.
    Same interface as Int8Tensor (a subclass, so existing isinstance checks and row selection keep working)."""
    G = 32

    def __init__(self, q, scale, dtype, last):
        super().__init__(q, scale, dtype)
        self.last = last                      # original last-dim size

    @classmethod
    def quantize(cls, x: torch.Tensor) -> "Int4Tensor":
        last = x.shape[-1]
        pad = (-last) % cls.G
        xf = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
        g = xf.view(*xf.shape[:-1], -1, cls.G)                         # (..., n_groups, G)
        scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
        v = (torch.round(g / scale).clamp_(-7, 7) + 8).to(torch.uint8).view(*xf.shape)   # 1..15
        packed = v[..., 0::2] | (v[..., 1::2] << 4)
        return cls(packed, scale.squeeze(-1).to(x.dtype if x.dtype != torch.float32 else torch.float32), x.dtype, last)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dt = dtype or self.dtype
        lo, hi = (self.q & 0xF), (self.q >> 4)
        v = torch.stack([lo, hi], -1).flatten(-2).to(dt) - 8
        g = v.view(*v.shape[:-1], -1, self.G) * self.scale.to(dt)[..., None]
        return g.flatten(-2)[..., : self.last]

    @property
    def shape(self):
        return torch.Size([*self.q.shape[:-1], self.last])

    def to(self, device) -> "Int4Tensor":
        return Int4Tensor(self.q.to(device), self.scale.to(device), self.dtype, self.last)

    def index_select(self, dim: int, idx: torch.Tensor) -> "Int4Tensor":
        assert dim != -1 and dim != self.q.dim() - 1, "int4 rows only"
        return Int4Tensor(self.q.index_select(dim, idx), self.scale.index_select(dim, idx), self.dtype, self.last)


def kv_mode(cfg) -> str | bool:
    """Cache quantization from a BackboneConfig: False | "int8" | "int4"."""
    return cfg.kv_quant if cfg.kv_quant in ("int8", "int4") else False


def maybe_quantize(x: torch.Tensor, enabled):
    """enabled: False/None (keep), True or "int8" (Int8Tensor), "int4" (Int4Tensor)."""
    if not enabled:
        return x
    return Int4Tensor.quantize(x) if enabled == "int4" else Int8Tensor.quantize(x)


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
        if self.__dict__.get("_dq") is not None:
            return self._dq
        return self.qweight.to(self.scale.dtype) * self.scale

    def materialize(self, dtype: torch.dtype) -> None:
        """Training: expand the int8 weight ONCE to `dtype` (the exact product the forward computes per call) and
        drop the int8 copy. Same numbers as the deployed int8 layer, without a dequantize kernel on every forward,
        recompute and backward call. Inference keeps int8 (fused GemLite kernels or this per-call path)."""
        self.__dict__["_dq"] = (self.qweight.to(dtype) * self.scale.to(dtype)).detach()
        self.qweight = torch.empty(0, dtype=torch.int8, device=self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq = self.__dict__.get("_dq")
        if dq is not None and dq.dtype == x.dtype:
            return F.linear(x, dq, None if self.bias is None else self.bias.to(x.dtype))
        return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8 weight-only"


def materialize_int8_(module: nn.Module, dtype: torch.dtype = torch.bfloat16, budget_gb: float | None = None) -> int:
    """Expand Int8Linear layers under `module` once (see Int8Linear.materialize), in module order, until the extra
    memory (bf16 minus int8 bytes) would exceed `budget_gb` (None: all). Returns the number expanded."""
    n, extra = 0, 0.0
    per = torch.finfo(dtype).bits // 8 - 1                      # extra bytes per weight
    for m in module.modules():
        if isinstance(m, Int8Linear) and m.__dict__.get("_dq") is None:
            add = m.qweight.numel() * per / 2**30
            if budget_gb is not None and extra + add > budget_gb:
                continue
            m.materialize(dtype)
            extra += add
            n += 1
    return n


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


# ---------------------------------------------------------------- low-bit weights with Triton GEMMs (GemLite)
def gemlite_linear(lin: nn.Linear, bits: int, group: int = 64) -> nn.Module:
    """nn.Linear -> GemLiteLinear (A16W{bits}): asymmetric per-group weights, Triton GEMM kernels (native on sm_87).
    bits=8 uses one group per row (per-channel)."""
    from gemlite import DType, GemLiteLinear
    W = lin.weight.data.float()
    N, K = W.shape
    g = group if bits < 8 else K
    Wg = W.view(N, K // g, g)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    s = (mx - mn).clamp_min(1e-8) / (2 ** bits - 1)
    z = -mn / s
    Wq = torch.round(Wg / s + z).clamp_(0, 2 ** bits - 1).to(torch.uint8).view(N, K)
    dt = DType.BF16 if lin.weight.dtype == torch.bfloat16 else DType.FP16
    out = GemLiteLinear(W_nbits=bits, group_size=g, in_features=K, out_features=N, input_dtype=dt, output_dtype=dt)
    out.pack(Wq, s.view(N, -1).to(lin.weight.dtype), z.view(N, -1).to(lin.weight.dtype),
             bias=None if lin.bias is None else lin.bias.data)
    out.in_features, out.out_features = K, N
    return out


def gemlite_quantize_(module: nn.Module, bits: int, group: int = 64) -> nn.Module:
    """Replace every nn.Linear under `module` in place with a GemLite A16W{bits} layer."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, gemlite_linear(child, bits, group))
            del child
        else:
            gemlite_quantize_(child, bits, group)
    return module


def gemlite_a8w8_(module: nn.Module) -> nn.Module:
    """Replace every nn.Linear under `module` with GemLite A8W8: int8 per-channel weights, int8 per-token dynamic
    activations, int8 tensor-core GEMM (Triton). Fastest prefill GEMM measured on the Orin (scripts/bench_gemm.py)."""
    from gemlite.helper import A8W8_int8_dynamic
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            K, N = child.in_features, child.out_features
            q = A8W8_int8_dynamic(device=str(child.weight.device), dtype=child.weight.dtype).from_linear(child)
            q.in_features, q.out_features = K, N
            setattr(module, name, q)
        else:
            gemlite_a8w8_(child)
    return module
