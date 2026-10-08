"""LoRA on the decision head (never the backbone): parameter-efficient fine-tuning of a warm-started head.

Every nn.Linear of the model outside `backbone.` gets a low-rank update  y += (x A^T) B^T * alpha / r  (A Kaiming,
B zero, so training starts exactly at the warm start), added by a forward hook so the head's own parameter names do
not change. All other head parameters are frozen, except Linear layers too small for a low-rank update
(min(in, out) <= r, e.g. the 1-unit score), which train directly. Checkpoints store the MERGED weights
(W + B A * alpha / r), so a LoRA-tuned head loads like any other.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class HeadLoRA(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank, self.scale = rank, alpha / rank
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        self.targets: dict[str, nn.Linear] = {}
        self.full: list[str] = []
        self._handles = []
        for name, mod in model.named_modules():
            if name.startswith("backbone") or name.startswith("head_lora") or type(mod) is not nn.Linear:
                continue
            if min(mod.in_features, mod.out_features) <= rank:
                self.full.append(name)
                continue
            key = name.replace(".", "__")
            dev, dt = mod.weight.device, torch.float32
            a = torch.empty(rank, mod.in_features, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A[key] = nn.Parameter(a)
            self.B[key] = nn.Parameter(torch.zeros(mod.out_features, rank, device=dev, dtype=dt))
            self.targets[key] = mod
            self._handles.append(mod.register_forward_hook(self._hook(key)))

    def _hook(self, key):
        def fn(mod, args, out):
            x = args[0]
            d = (x.to(self.A[key].dtype) @ self.A[key].t()) @ self.B[key].t()
            return out + (d * self.scale).to(out.dtype)
        return fn

    def delta(self, key) -> torch.Tensor:
        return (self.B[key] @ self.A[key]) * self.scale

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def add_head_lora_(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> HeadLoRA:
    """Freeze the head, attach LoRA (registered as model.head_lora), unfreeze the tiny Linears. Returns the module."""
    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            p.requires_grad_(False)
    lora = HeadLoRA(model, rank, alpha)
    model.head_lora = lora
    for name in lora.full:
        model.get_submodule(name).requires_grad_(True)
    return lora


def merged_state_dict(model: nn.Module) -> dict:
    """The head's state dict with every LoRA update folded into its Linear weight and no LoRA tensors."""
    lora = getattr(model, "head_lora", None)
    sd = {k: v for k, v in model.trainable_state_dict().items() if not k.startswith("head_lora.")}
    if lora is None:
        return sd
    with torch.no_grad():
        for key, mod in lora.targets.items():
            w = key.replace("__", ".") + ".weight"
            sd[w] = (sd[w].float() + lora.delta(key).to(sd[w].device)).to(sd[w].dtype)
    return sd
