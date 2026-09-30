"""'Parallel sampler' (O7 hypothesis b): average K stochastic passes of the small decision module
over one cached state memory. Returns mean probabilities and their std across passes.

modes: off | mc_dropout | gaussian_noise. The backbone is never re-run; only the trainable
decision module is stochastic, so the cost is K x (decision module), not K x (backbone).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplerConfig:
    mode: str = "off"
    k: int = 1
    noise_std: float = 0.05


def enable_dropout(module: torch.nn.Module, on: bool):
    """Train-mode every decision-module submodule (the frozen backbone stays in eval).

    Whole modules, not just nn.Dropout: TransformerEncoderLayer and our CrossAttention read
    `self.training` to decide whether dropout is applied at all. The decision modules have no
    BatchNorm, so train mode changes nothing but dropout."""
    for name, m in module.named_modules():
        if name == "backbone" or name.startswith("backbone."):
            continue
        m.training = on


def run_sampler(fn, cfg: SamplerConfig, decision_module: torch.nn.Module | None = None):
    """fn(noise_std) -> probs (N,K). Returns (mean, std)."""
    if cfg.mode == "off" or cfg.k <= 1:
        p = fn(0.0)
        return p, torch.zeros_like(p)
    outs = []
    if cfg.mode == "mc_dropout" and decision_module is not None:
        enable_dropout(decision_module, True)
    try:
        for _ in range(cfg.k):
            outs.append(fn(cfg.noise_std if cfg.mode == "gaussian_noise" else 0.0))
    finally:
        if decision_module is not None:
            enable_dropout(decision_module, False)
    P = torch.stack(outs, 0)
    return P.mean(0), P.std(0, unbiased=False)
