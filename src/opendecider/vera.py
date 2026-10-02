"""VeRA adapters (Kopiczko, Blankevoort, Asano, ICLR 2024, arXiv:2310.11454) on the frozen backbone.

    h = W0 x + Λ_b B Λ_d A x
A (r × in) and B (out × r): frozen random matrices shared by every adapted linear of the same shape
(Kaiming-uniform, fixed seed). Trainable per linear: d ∈ R^r (init 0.1) and b ∈ R^out (init 0), as in the paper,
so the adapted model equals the frozen one at init.

Scope for OpenDecider: only the TOP `n_layers` decoder layers, and only while `active` is True, i.e. during the ROW
pass (question/options continuing from the state cache). The state pass runs with adapters off, so cached state
features / KV / GDN states stay those of the frozen model. Nothing below the adapted layers requires grad, so autograd
records no graph there: backprop is shallow.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class VeRA(nn.Module):
    def __init__(self, backbone, n_layers: int = 6, rank: int = 256, d_init: float = 0.1, seed: int = 0,
                 targets: tuple[str, ...] = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj",
                                             "o_proj", "gate_proj", "up_proj", "down_proj")):
        super().__init__()
        self.active = False
        self.rank = rank
        layers = backbone.lm.layers
        start = len(layers) - n_layers
        self.targets: list[tuple[nn.Module, str]] = []
        shapes = {}
        for li in range(start, len(layers)):
            for name, mod in layers[li].named_modules():
                leaf = name.split(".")[-1]
                if leaf in targets and hasattr(mod, "in_features"):
                    self.targets.append((mod, f"L{li}.{name}"))
                    shapes[(mod.in_features, mod.out_features)] = True
        g = torch.Generator().manual_seed(seed)
        self._A, self._B = {}, {}
        for (i, o) in shapes:
            a = torch.empty(rank, i); b = torch.empty(o, rank)
            bound_a, bound_b = math.sqrt(6 / i), math.sqrt(6 / rank)          # Kaiming-uniform bounds
            a.uniform_(-bound_a, bound_a, generator=g); b.uniform_(-bound_b, bound_b, generator=g)
            key = f"{i}x{o}"
            self.register_buffer(f"A_{key}", a, persistent=False)            # frozen, regenerated from the seed
            self.register_buffer(f"B_{key}", b, persistent=False)
        self.d = nn.ParameterDict()
        self.b = nn.ParameterDict()
        self._handles = []
        for mod, tag in self.targets:
            k = tag.replace(".", "_")
            self.d[k] = nn.Parameter(torch.full((rank,), d_init))
            self.b[k] = nn.Parameter(torch.zeros(mod.out_features))
            self._handles.append(mod.register_forward_hook(self._hook(k, mod.in_features, mod.out_features)))

    def _hook(self, k, i, o):
        def fn(mod, args, out):
            if not self.active:
                return out
            x = args[0]
            A, B = self._frozen(i, o, x.dtype)
            delta = ((x @ A) * self.d[k].to(x.dtype)) @ B * self.b[k].to(x.dtype)
            return out + delta
        return fn

    def _frozen(self, i, o, dtype):
        """A^T, B^T of one shape in the compute dtype, cast once (the random projections never change)."""
        key = (i, o, dtype)
        c = self.__dict__.setdefault("_cast", {})
        hit = c.get(key)
        A = getattr(self, f"A_{i}x{o}")
        if hit is None or hit[0].device != A.device:
            hit = c[key] = (A.t().to(dtype).contiguous(), getattr(self, f"B_{i}x{o}").t().to(dtype).contiguous())
        return hit

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
