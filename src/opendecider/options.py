"""Option tokens: span pooling and slot embeddings.

Slot table has 256 rows; row 0 is reserved (padding), so at most 255 options per Choice call.
That cap, together with order sensitivity (O6), is what the slot hypothesis is meant to explain.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MAX_SLOTS = 256
MAX_OPTIONS = MAX_SLOTS - 1


class SlotEmbedding(nn.Module):
    """mode: learned | none | random_per_call."""

    def __init__(self, d: int, mode: str = "learned"):
        super().__init__()
        assert mode in ("learned", "none", "random_per_call"), mode
        self.mode = mode
        self.table = nn.Embedding(MAX_SLOTS, d, padding_idx=0)   # default init; padding row 0 is zero

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """slots: (N,K) long, 1..K for real options, 0 for padding."""
        if self.mode == "none":
            return torch.zeros(*slots.shape, self.table.embedding_dim, device=slots.device,
                               dtype=self.table.weight.dtype)
        if self.mode == "random_per_call":
            perm = torch.randperm(MAX_OPTIONS, device=slots.device) + 1
            slots = torch.where(slots > 0, perm[(slots - 1).clamp_min(0)], slots)
        return self.table(slots)


def span_mean(h: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """Mean-pool token spans.

    h: (R,L,d); spans: (M,3) long rows of (row, start, end) with end exclusive. Returns (M,d).
    Implemented with a cumulative sum so it is one gather regardless of M.
    """
    R, L, d = h.shape
    cs = torch.cat([torch.zeros(R, 1, d, dtype=torch.float32, device=h.device), h.float().cumsum(1)], 1)
    r, s, e = spans[:, 0], spans[:, 1], spans[:, 2]
    tot = cs[r, e] - cs[r, s]
    return (tot / (e - s).clamp_min(1).unsqueeze(-1).float()).to(h.dtype)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1)


def span_pool(h: torch.Tensor, spans: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool token spans of a CAUSAL model.

    mean     : plain mean (early tokens have seen little of the span)
    last     : last token of the span (the only one that has seen the whole span + its prefix)
    weighted : position-weighted mean, weights 1..n within the span (SGPT-style for causal LMs)
    """
    if mode == "mean":
        return span_mean(h, spans)
    r, s, e = spans[:, 0], spans[:, 1], spans[:, 2]
    if mode == "last":
        return h[r, (e - 1).clamp_min(s)]
    if mode == "weighted":
        # sum_{t=s}^{e-1} (t-s+1) h_t = sum t*h_t - (s-1) sum h_t, via two prefix sums (no (M,L,d) gather)
        R, L, d = h.shape
        hf = h.float()
        pos = torch.arange(L, device=h.device, dtype=torch.float32)[None, :, None]
        z = torch.zeros(R, 1, d, device=h.device)
        c0 = torch.cat([z, hf.cumsum(1)], 1)
        c1 = torch.cat([z, (hf * pos).cumsum(1)], 1)
        s0 = c0[r, e] - c0[r, s]
        s1 = c1[r, e] - c1[r, s]
        n = (e - s).float()
        num = s1 - (s.float() - 1).unsqueeze(-1) * s0
        return (num / (n * (n + 1) / 2).clamp_min(1).unsqueeze(-1)).to(h.dtype)
    raise ValueError(mode)
