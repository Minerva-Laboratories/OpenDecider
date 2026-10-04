"""Flamingo-style gated cross-attention and a Perceiver resampler.

GatedXAttnDense (Alayrac et al. 2022): y = x + tanh(a) * XAttn(LN(x), M);  y = y + tanh(b) * FFW(LN(y)),
with a = b = 0 at init, so the frozen LM is unchanged at initialisation.

Cross-attention is per query token, so rows that share a state can be flattened into one
long query sequence against that state's memory. That avoids copying a (T x d) memory per
option row (255 options x 16 questions would otherwise materialise GBs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import materialize


class CrossAttention(nn.Module):
    def __init__(self, d_q: int, d_kv: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_q % n_heads == 0
        self.h, self.dh = n_heads, d_q // n_heads
        self.q = nn.Linear(d_q, d_q, bias=False)
        self.kv = nn.Linear(d_kv, 2 * d_q, bias=False)
        self.o = nn.Linear(d_q, d_q, bias=False)
        self.dropout = dropout

    def project_kv(self, mem: torch.Tensor) -> torch.Tensor:
        """(S,T,d_kv) -> (S, 2, h, T, dh). This is the cacheable cross-attention KV."""
        S, T, _ = mem.shape
        return self.kv(mem).view(S, T, 2, self.h, self.dh).permute(0, 2, 3, 1, 4)

    def forward(self, x: torch.Tensor, kv, mem_mask: torch.Tensor, row_state: torch.Tensor) -> torch.Tensor:
        """x: (R,L,d_q); kv: (S,2,h,T,dh) tensor or Int8Tensor; mem_mask: (S,T); row_state: (R,)."""
        R, L, d = x.shape
        kv = materialize(kv, x.dtype)
        q = self.q(x).view(R, L, self.h, self.dh)
        out = torch.empty(R, L, self.h, self.dh, dtype=x.dtype, device=x.device)
        p = self.dropout if self.training else 0.0
        for s in torch.unique(row_state).tolist():
            rows = (row_state == s).nonzero(as_tuple=True)[0]
            qs = q[rows].reshape(1, -1, self.h, self.dh).transpose(1, 2)        # (1,h,rows*L,dh)
            k, v = kv[s, 0][None], kv[s, 1][None]                               # (1,h,T,dh)
            m = mem_mask[s][None, None, None, :]                                # (1,1,1,T)
            o = F.scaled_dot_product_attention(qs, k, v, attn_mask=m, dropout_p=p)
            out[rows] = o.transpose(1, 2).reshape(len(rows), L, self.h, self.dh)
        return self.o(out.reshape(R, L, d))


class GatedXAttnDense(nn.Module):
    def __init__(self, d: int, d_mem: int, n_heads: int = 8, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln_x = nn.LayerNorm(d)
        self.xattn = CrossAttention(d, d_mem, n_heads, dropout)
        self.ln_f = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.alpha_xattn = nn.Parameter(torch.zeros(()))
        self.alpha_dense = nn.Parameter(torch.zeros(()))

    def project_kv(self, mem: torch.Tensor):
        return self.xattn.project_kv(mem)

    def forward(self, x, kv, mem_mask, row_state):
        dt = x.dtype
        h = x.float()
        h = h + torch.tanh(self.alpha_xattn) * self.xattn(self.ln_x(h), kv, mem_mask, row_state).float()
        h = h + torch.tanh(self.alpha_dense) * self.ff(self.ln_f(h)).float()
        return h.to(dt)


class PerceiverResampler(nn.Module):
    """Compress a variable-length memory (S,T,d) to (S,N,d) learned latents (Jaegle 2021 / Flamingo)."""

    def __init__(self, d: int, n_latents: int = 256, depth: int = 2, n_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d), "ln_m": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, n_heads, batch_first=True),
                "ln_f": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))

    def forward(self, mem: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        S = mem.shape[0]
        lat = self.latents[None].expand(S, -1, -1).to(mem.dtype)
        for l in self.layers:
            q = l["ln_q"](lat)
            kv = torch.cat([l["ln_m"](mem), q], 1)          # latents also attend to themselves
            kpm = torch.cat([~mask, torch.zeros(S, q.shape[1], dtype=torch.bool, device=mask.device)], 1)
            lat = lat + l["attn"](q, kv, kv, key_padding_mask=kpm, need_weights=False)[0]
            lat = lat + l["ff"](l["ln_f"](lat))
        return lat, torch.ones(S, lat.shape[1], dtype=torch.bool, device=mem.device)
