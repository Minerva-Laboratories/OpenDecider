"""AWQ: activation-aware weight quantization (Lin et al., MLSys 2024, arXiv:2306.00978) for the frozen backbone.

Per linear layer, with calibration inputs X recorded from the full-precision model on OpenDecider requests:
  scale search: s = mean|X|^alpha (per input channel, normalized), alpha on a grid in [0, 1]; the weight is
                quantized as Q(W diag(s)) and the input is divided by s, which protects the channels with large
                activations. alpha minimizes ||X W^T - (X / s) Q(W s)^T||.
  clip search : per layer, the ratio r in {1.0, ..., 0.8} that clips each group's range before quantization,
                chosen by the same output error.
Quantization is asymmetric per group (4 bits, groups of 64). At inference the layer is a GemLite A16W4 GEMM with
the 1/s multiply in front (AWQLinear). Scales and clip ratios are saved so loading does not re-calibrate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ALPHAS = [i / 20 for i in range(21)]
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8)


def fake_quant(W: torch.Tensor, bits: int = 4, group: int = 64, clip: float = 1.0) -> torch.Tensor:
    """Round-to-nearest asymmetric group quantization (dequantized back), as the GemLite kernel computes it."""
    N, K = W.shape
    Wg = W.view(N, K // group, group)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    if clip < 1.0:
        c = (mx + mn) / 2
        mn, mx = c + (mn - c) * clip, c + (mx - c) * clip
    s = (mx - mn).clamp_min(1e-8) / (2 ** bits - 1)
    z = torch.round(-mn / s)
    return ((torch.round(Wg / s + z).clamp_(0, 2 ** bits - 1) - z) * s).view(N, K)


@torch.no_grad()
def search_layer(W: torch.Tensor, X: torch.Tensor, bits: int = 4, group: int = 64):
    """-> (scales (K,), clip, error with AWQ, error with plain rounding). W (N, K) float, X (n, K) float."""
    ref = X @ W.t()
    err = lambda Wq, Xs: float(((Xs @ Wq.t()) - ref).pow(2).mean())
    rtn = err(fake_quant(W, bits, group), X)
    xmean = X.abs().mean(0).clamp_min(1e-5)
    best = (rtn, torch.ones_like(xmean))
    for a in ALPHAS[1:]:
        s = xmean.pow(a)
        s = s / (s.max() * s.min()).sqrt()
        e = err(fake_quant(W * s, bits, group), X / s)
        if e < best[0]:
            best = (e, s)
    e_s, s = best
    best_c = (e_s, 1.0)
    for c in CLIPS[1:]:
        e = err(fake_quant(W * s, bits, group, c), X / s)
        if e < best_c[0]:
            best_c = (e, c)
    return s, best_c[1], best_c[0], rtn


class AWQLinear(nn.Module):
    """y = GemLite_W4(W diag(s)) (x / s)."""

    def __init__(self, lin: nn.Linear, s: torch.Tensor, clip: float, bits: int = 4, group: int = 64):
        super().__init__()
        from .quant import gemlite_linear
        scaled = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                           device=lin.weight.device, dtype=lin.weight.dtype)
        W = lin.weight.data.float() * s.to(lin.weight.device)[None]
        if clip < 1.0:                                     # clip each group's range, as searched
            N, K = W.shape
            Wg = W.view(N, K // group, group)
            mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
            c = (mx + mn) / 2
            W = Wg.clamp(c + (mn - c) * clip, c + (mx - c) * clip).view(N, K)
        scaled.weight.data.copy_(W.to(lin.weight.dtype))
        if lin.bias is not None:
            scaled.bias.data.copy_(lin.bias.data)
        self.q = gemlite_linear(scaled, bits, group)
        self.register_buffer("inv_s", (1.0 / s).to(lin.weight.dtype).to(lin.weight.device))
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        return self.q(x * self.inv_s)


class ActivationRecorder:
    """Forward pre-hooks on every nn.Linear under `root`: keep a random subsample of input rows per layer."""

    def __init__(self, root: nn.Module, max_rows: int = 4096, seed: int = 0):
        self.x, self.max_rows, self.handles = {}, max_rows, []
        self.g = torch.Generator().manual_seed(seed)
        for name, mod in root.named_modules():
            if isinstance(mod, nn.Linear):
                self.handles.append(mod.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            keep = x[torch.randperm(x.shape[0], generator=self.g)[: self.max_rows // 8].to(x.device)]
            cur = self.x.get(name)
            cur = keep.float().cpu() if cur is None else torch.cat([cur, keep.float().cpu()])
            if cur.shape[0] > self.max_rows:
                cur = cur[torch.randperm(cur.shape[0], generator=self.g)[: self.max_rows]]
            self.x[name] = cur
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def calibrate(root: nn.Module, X: dict, bits: int = 4, group: int = 64, max_eval_rows: int = 1024) -> dict:
    """Search scales/clip for every recorded linear -> {name: {"s", "clip", "err", "err_rtn"}}."""
    out = {}
    for name, mod in root.named_modules():
        if not isinstance(mod, nn.Linear) or name not in X or mod.in_features % group:
            continue
        W = mod.weight.data.float()
        x = X[name][:max_eval_rows].to(W.device)
        s, clip, e, rtn = search_layer(W, x, bits, group)
        out[name] = {"s": s.cpu(), "clip": clip, "err": e, "err_rtn": rtn}
    return out


def apply_awq_(root: nn.Module, params: dict, bits: int = 4, group: int = 64) -> nn.Module:
    """Replace calibrated nn.Linear layers under `root` with AWQLinear; others fall back to plain GemLite W4."""
    from .quant import gemlite_linear
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear):
                p = params.get(full)
                new = AWQLinear(child, p["s"], p["clip"], bits, group) if p is not None else \
                    gemlite_linear(child, bits, group)
                setattr(mod, cname, new)
    return root
