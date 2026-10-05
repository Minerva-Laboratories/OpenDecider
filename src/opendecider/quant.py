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
        if self.__dict__.get("_fused") is not None:
            return self._fused.dense_weight().t()
        return self.qweight.to(self.scale.dtype) * self.scale

    def materialize(self, dtype: torch.dtype) -> None:
        """Training: expand the int8 weight ONCE to `dtype` (the exact product the forward computes per call) and
        drop the int8 copy. Same numbers as the deployed int8 layer, without a dequantize kernel on every forward,
        recompute and backward call. Inference keeps int8 (fused GemLite kernels or this per-call path)."""
        self.__dict__["_dq"] = (self.qweight.to(dtype) * self.scale.to(dtype)).detach()
        self.qweight = torch.empty(0, dtype=torch.int8, device=self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.__dict__.get("_fused")
        if fused is not None:
            return fused(x)
        dq = self.__dict__.get("_dq")
        if dq is not None and dq.dtype == x.dtype:
            return F.linear(x, dq, None if self.bias is None else self.bias.to(x.dtype))
        return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))

    def dequant_rows(self, idx, dtype=torch.bfloat16) -> torch.Tensor:
        """Weight rows `idx` (output units) in `dtype` (LM-head use: log-probs and logits in chunks)."""
        return self.qweight[idx].to(dtype) * self.scale[idx].to(dtype)

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

    def dequant_rows(self, idx, dtype=torch.bfloat16) -> torch.Tensor:
        return self.qweight[idx].to(dtype) * self.scale[idx].to(dtype)


class Q4Table(nn.Module):
    """Frozen 4-bit table (embedding rows or LM-head rows) in llama.cpp's Q4_0 layout: blocks of 32 values along a row,
    one fp16 scale d per block, w = (q - 8) * d with q in 0..15 (low nibbles hold block positions 0-15, high nibbles
    16-31). 4.5 bits per value. Quantizing values that came from a Q4_0 GGUF tensor reproduces them exactly (the block
    extreme is -8 d, so d is recovered); other tables get round-to-nearest Q4_0."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        N, K = w.shape
        assert K % 32 == 0, "Q4Table needs rows whose length is a multiple of 32"
        self.num_embeddings, self.embedding_dim = N, K
        self.in_features, self.out_features = K, N
        self.padding_idx = None
        qs, ds = [], []
        for a in range(0, N, 16384):                        # chunks: bounded float32 temporaries
            b = w[a:a + 16384].float().view(-1, K // 32, 32)
            idx = b.abs().argmax(-1, keepdim=True)
            mx = b.gather(-1, idx)                          # signed value of largest magnitude (ggml's choice)
            d = mx / -8
            inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
            q = torch.clamp(torch.floor(b * inv + 8.5), 0, 15).to(torch.uint8)
            qs.append((q[..., :16] | (q[..., 16:] << 4)).view(b.shape[0], K // 2))
            ds.append(d.squeeze(-1).to(torch.float16))
        self.register_buffer("qs", torch.cat(qs))
        self.register_buffer("d", torch.cat(ds))

    def dequant_rows(self, idx, dtype=torch.bfloat16) -> torch.Tensor:
        """Rows `idx` (index tensor or slice) as (n, K) in `dtype`."""
        q = self.qs[idx]
        n, K = q.shape[0], self.embedding_dim
        q = q.view(n, K // 32, 16)
        v = torch.cat([q & 15, q >> 4], -1).to(dtype) - 8               # (n, blocks, 32)
        return (v * self.d[idx].to(dtype).unsqueeze(-1)).view(n, K)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequant_rows(slice(None), torch.float16 if self.d.device.type == "cpu" else torch.bfloat16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:              # embedding lookup
        return self.dequant_rows(ids.reshape(-1), torch.bfloat16 if ids.is_cuda else torch.float32).view(
            *ids.shape, self.embedding_dim)


def quantize_table(module: nn.Module, mode: str) -> nn.Module:
    """An embedding or LM-head module as the frozen table `mode` (int8 | q4); `module` holds float weights."""
    if mode == "q4":
        return Q4Table(module.weight.data)
    return Int8Embedding(module) if isinstance(module, nn.Embedding) else Int8Linear(module)


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


# ---------------------------------------------------------------- fused int8 inference GEMMs (same weights)
def int8_to_gemlite(m: Int8Linear) -> nn.Module:
    """Int8Linear -> GemLite A16W8 holding the SAME int8 weights (stored as uint8 q+128 with zero point 128, one group
    per row), so W = (q+128-128)*scale exactly. The GEMM reads int8 directly instead of expanding the whole weight to
    bf16 on every call; outputs differ from the dequantize path only by bf16 rounding inside the kernel."""
    from gemlite import DType, GemLiteLinear
    q, s = m.qweight, m.scale
    N, K = q.shape
    dt = DType.BF16 if s.dtype == torch.bfloat16 else DType.FP16
    g = GemLiteLinear(W_nbits=8, group_size=K, in_features=K, out_features=N, input_dtype=dt, output_dtype=dt)
    g.pack((q.to(torch.int16) + 128).to(torch.uint8), s.view(N, 1), torch.full((N, 1), 128.0, dtype=s.dtype,
                                                                               device=q.device), bias=m.bias)
    g.in_features, g.out_features = K, N
    return g


class Int8GemLite(nn.Module):
    """A GemLite A16W8 layer plus a dense path for large batches, from the SAME packed weights (no second copy).
    GemLite is fastest for small and mid batch sizes; above `dense_from` tokens per call, expanding the weight once
    and using cuBLAS is faster (crossover measured per shape by scripts/tune_kernels.py)."""

    def __init__(self, g: nn.Module, dense_from: int | None = None):
        super().__init__()
        self.g, self.dense_from = g, dense_from
        self.in_features, self.out_features = g.in_features, g.out_features

    def dense_weight(self) -> torch.Tensor:
        """(in, out) weight from GemLite's buffer: 4 uint8 values per int32 along `in` (k = 4*row + byte)."""
        W, K, N = self.g.W_q, self.in_features, self.out_features
        q = W.view(torch.uint8).view(K // 4, N, 4).permute(0, 2, 1).reshape(K, N)
        return (q.to(self.g.scales.dtype) - 128) * self.g.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dense_from is not None and x.numel() // self.in_features >= self.dense_from:
            y = x @ self.dense_weight()
            return y if self.g.bias is None else y + self.g.bias
        return self.g(x)


def int8_to_gemlite_(module: nn.Module, dense_from: dict | None = None) -> int:
    """Give every (non-materialized) Int8Linear under `module` an Int8GemLite on the same weights and free its own int8
    copy. The Int8Linear OBJECTS stay in place, so forward hooks on them (VeRA adapters) keep working; dense_from maps
    "out x in" to the batch size from which the dense path is used. Inference only: no backward. Returns the count."""
    n = 0
    for m in module.modules():
        if isinstance(m, Int8Linear) and m.qweight.numel() and m.__dict__.get("_dq") is None:
            key = f"{m.out_features}x{m.in_features}"
            m.__dict__["_fused"] = Int8GemLite(int8_to_gemlite(m), (dense_from or {}).get(key))
            m.qweight = torch.empty(0, dtype=torch.int8, device=m.qweight.device)
            n += 1
    return n


def gemlite_config_path() -> str:
    """Per-GPU file of tuned GemLite kernel configs (written by scripts/tune_kernels.py)."""
    import os
    import re
    name = re.sub(r"[^a-z0-9]+", "_", torch.cuda.get_device_name().lower()).strip("_")
    root = os.environ.get("OPENDECIDER_KERNEL_DIR", os.path.expanduser("~/.cache/opendecider/kernels"))
    return os.path.join(root, f"gemlite_{name}_sm{''.join(map(str, torch.cuda.get_device_capability()))}.json")


def load_gemlite_config() -> dict | None:
    """Load this GPU's tuned configs into GemLite (autotuning off) and return the per-shape dense crossovers. None if
    the GPU has not been tuned: untuned GemLite is slower than dequantize + cuBLAS at large M on the Orin
    (scripts/bench_gemm.py), so callers keep the dequantize path."""
    import json
    import os
    path = gemlite_config_path()
    if not os.path.exists(path):
        return None
    import gemlite
    gemlite.set_autotune(False)
    gemlite.load_config(path)
    cross = path.replace(".json", "_dense_from.json")
    return json.load(open(cross)) if os.path.exists(cross) else {}


def fast_int8_kernels_(backbone) -> int:
    """Inference loaders: run the int8 decoder layers on tuned GemLite kernels when this GPU has a tuned config
    (scripts/tune_kernels.py). OPENDECIDER_INT8_GEMM=dequant keeps the dequantize path. Returns layers swapped."""
    import os
    if os.environ.get("OPENDECIDER_INT8_GEMM", "auto") == "dequant" or backbone.device.type != "cuda":
        return 0
    if not any(isinstance(m, Int8Linear) for m in backbone.lm.layers.modules()):
        return 0
    try:
        dense_from = load_gemlite_config()
    except ImportError:
        return 0
    if dense_from is None:
        return 0
    n = int8_to_gemlite_(backbone.lm.layers, dense_from)
    torch.cuda.empty_cache()
    return n
