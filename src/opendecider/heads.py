"""Pointer heads and question-type readouts.

Pointer (Vinyals et al. 2015): logit_k = w . tanh(W1 h_ok + W2 h_q), one logit per option token,
so the output "vocabulary" is whatever options the caller supplies.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .options import MAX_OPTIONS

NEG = -1e9


class PointerHead(nn.Module):
    def __init__(self, d: int, kind: str = "ptr", d_hidden: int | None = None):
        super().__init__()
        self.kind = kind
        if kind == "ptr":
            dh = d_hidden or d
            self.w1 = nn.Linear(d, dh, bias=False)
            self.w2 = nn.Linear(d, dh)
            self.v = nn.Linear(dh, 1, bias=False)
        elif kind == "bilinear":
            self.W = nn.Linear(d, d, bias=False)
        else:
            raise ValueError(kind)
        self.relevance = nn.Linear(d, 1)     # stage-1 independent sigmoid score (>255 options)
        self.noul = nn.Linear(d, 1)          # optional sigmoid-on-question Noul readout

    def forward(self, h_q: torch.Tensor, h_o: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        """h_q: (N,d); h_o: (N,K,d); opt_mask: (N,K) bool -> logits (N,K) with NEG at padding."""
        if self.kind == "ptr":
            z = self.v(torch.tanh(self.w1(h_o) + self.w2(h_q).unsqueeze(1))).squeeze(-1)
        else:
            z = torch.einsum("nd,nkd->nk", self.W(h_q), h_o) / h_o.shape[-1] ** 0.5
        return z.float().masked_fill(~opt_mask, NEG)

    def relevance_logits(self, h_o, opt_mask):
        return self.relevance(h_o).squeeze(-1).float().masked_fill(~opt_mask, NEG)


def masked_probs(logits: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float().masked_fill(~opt_mask, NEG), -1) * opt_mask


def expected_level(probs: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * idx).sum(-1)


def two_stage_select(relevance: torch.Tensor, keep: int = MAX_OPTIONS) -> torch.Tensor:
    """Indices of the top-`keep` options by stage-1 relevance (kept in original order)."""
    k = min(keep, relevance.numel())
    top = torch.topk(relevance, k).indices
    return torch.sort(top).values
