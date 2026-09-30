"""Proper scoring losses over the option softmax (log score + Brier) and an ordinal auxiliary."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .heads import NEG


def choice_loss(logits: torch.Tensor, opt_mask: torch.Tensor, target: torch.Tensor,
                brier_weight: float = 1.0) -> dict:
    """target: (N,) option index, or (N,K) probability vector (soft labels). Returns dict of means."""
    logits = logits.float().masked_fill(~opt_mask, NEG)
    logp = F.log_softmax(logits, -1)
    if target.dim() == 1:
        onehot = F.one_hot(target, logits.shape[-1]).float()
    else:
        onehot = target.float()
    onehot = onehot * opt_mask
    ce = -(onehot * logp.masked_fill(~opt_mask, 0)).sum(-1)
    p = logp.exp() * opt_mask
    brier = ((p - onehot) ** 2).sum(-1)
    loss = ce + brier_weight * brier
    return {"loss": loss.mean(), "ce": ce.mean().detach(), "brier": brier.mean().detach()}


def ordinal_emd_loss(logits: torch.Tensor, opt_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared earth mover's distance between predicted and one-hot CDFs (ordered levels)."""
    p = torch.softmax(logits.float().masked_fill(~opt_mask, NEG), -1) * opt_mask
    t = F.one_hot(target, logits.shape[-1]).float()
    return (((p.cumsum(-1) - t.cumsum(-1)) ** 2) * opt_mask).sum(-1).mean()


def relevance_loss(rel_logits: torch.Tensor, opt_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-question mean BCE over its options, then mean over questions (so micro-batching is exact)."""
    y = F.one_hot(target, rel_logits.shape[-1]).float()
    l = F.binary_cross_entropy_with_logits(rel_logits.clamp(-30, 30), y, reduction="none")
    return ((l * opt_mask).sum(-1) / opt_mask.sum(-1)).mean()
