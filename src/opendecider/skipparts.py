"""Skip parts of the layers above a cut for state tokens, instead of whole layers (skipkv.py): the MLPs, the
full-attention mixers, or both, each replaced by a ridge map from its input to its output (fitted on exact passes).
The DeltaNet mixers always run, so the residual stream keeps propagating through context mixing.

- mlp:  layer j's MLP -> post-attention-norm output @ W + b (as skipkv.MLPSurrogates);
- attn: a full-attention layer's mixer still writes its cache from the TRUE normed input (K/V for stock attention,
        latent / RoPE-key codes for Mercurius MLA), but its queries, scores and output projection are not computed;
        the output is the normed input @ W + b. Saves the O(T^2) attention of the prompt in those layers.
Surrogates are per layer and independent of the cut (exact inputs), so one fit from the lowest cut serves every cut
above it. Used through `PartSkip(...).ctx(lm)` (skipdepth.segment_cache accepts it as a segment's maps)."""
from __future__ import annotations

import contextlib
import sys

import numpy as np
import torch
from torch import nn

from .skipkv import LAMS, CacheOnlyLayer, _LinearMLP, _merc, _stock


class _AttnSurrogate(nn.Module):
    def __init__(self, att, W, b):
        super().__init__()
        self.att, self.W, self.b = att, W, b
        self.stock = _stock(att)
        self.merc = _merc(att, "mercurius_cache", "_compressed_forward")
        assert self.stock or self.merc, "attention surrogate: stock Qwen3.5 attention or Mercurius compressed MLA"

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                position_ids=None, **kw):
        x = hidden_states
        if self.stock:
            CacheOnlyLayer._attn(self.att, x, position_embeddings, past_key_values, sys.modules[type(self.att).__module__])
        else:
            CacheOnlyLayer._merc_attn(self.att, x, position_ids, past_key_values)
        y = torch.addmm(self.b.to(x.dtype), x.reshape(-1, x.shape[-1]), self.W.to(x.dtype)).view(x.shape)
        return y, None


class PartSkip:
    """cut; mlp = (W (U, D, D), b (U, D)) for layers base..L-1 or None; attn = {j: (W, b)} or None."""

    def __init__(self, cut, base, mlp=None, attn=None):
        self.cut, self.base, self.mlp, self.attn = cut, base, mlp, attn

    @contextlib.contextmanager
    def ctx(self, lm):
        saved = []
        try:
            for j in range(self.cut, len(lm.layers)):
                layer = lm.layers[j]
                if self.mlp is not None:
                    saved.append((layer, "mlp", layer.mlp))
                    layer.mlp = _LinearMLP(self.mlp[0][j - self.base], self.mlp[1][j - self.base])
                if self.attn is not None and j in self.attn:
                    saved.append((layer, "self_attn", layer.self_attn))
                    layer.self_attn = _AttnSurrogate(layer.self_attn, *self.attn[j])
            yield
        finally:
            for layer, k, m in reversed(saved):
                setattr(layer, k, m)


@torch.no_grad()
def fit_parts(bb, texts, held_texts, base: int, lams=LAMS, log=print):
    """Ridge maps for every MLP and every full-attention mixer of layers >= base, input -> output on exact passes
    (bb.prefix_cache_batch), one ridge strength per map by held-out squared error. Returns a dict for save()."""
    lm, D, dev = bb.lm, bb.hidden_size, bb.device
    L = len(lm.layers)
    keys = [("mlp", j) for j in range(base, L)] + [("attn", j) for j in range(base, L) if hasattr(lm.layers[j], "self_attn")]
    XtX = {k: torch.zeros(D + 1, D + 1, device=dev, dtype=torch.float64) for k in keys}
    XtY = {k: torch.zeros(D + 1, D, device=dev, dtype=torch.float64) for k in keys}

    def pairs(text):
        st, hs = {}, []
        one = lambda x: torch.cat([x, torch.ones(len(x), 1, device=dev)], 1)
        for kind, j in keys:
            if kind == "mlp":
                hs.append(lm.layers[j].mlp.register_forward_hook(
                    lambda m, a, o, j=j: st.__setitem__(("mlp", j), (one(a[0][0].float()), o[0].float()))))
            else:
                hs.append(lm.layers[j].self_attn.register_forward_hook(
                    lambda m, a, k, o, j=j: st.__setitem__(
                        ("attn", j), (one((a[0] if a else k["hidden_states"])[0].float()), o[0][0].float())),
                    with_kwargs=True))
        try:
            bb.prefix_cache_batch(bb.tokenize([text]))
        finally:
            for h in hs:
                h.remove()
        return st
    n = 0
    for i, t in enumerate(texts):
        for k, (X, Y) in pairs(t).items():
            XtX[k] += (X.T @ X).double()
            XtY[k] += (X.T @ Y).double()
        n += len(X)
        if (i + 1) % 50 == 0:
            log(f"  fit parts from {base}: {i + 1}/{len(texts)} states, {n} tokens")
    held = [pairs(t) for t in held_texts]
    out = {"base": base, "n_tokens": n, "mlp_W": torch.empty(L - base, D, D, dtype=torch.bfloat16),
           "mlp_b": torch.empty(L - base, D, dtype=torch.bfloat16), "attn": {}, "rel": {}, "lam": {}}
    for k in keys:
        A, B = XtX.pop(k), XtY.pop(k)
        sc = float(A.diagonal()[:-1].mean())
        best = None
        for lam in lams:
            W = torch.linalg.solve(A + lam * sc * torch.eye(D + 1, dtype=A.dtype, device=dev), B).float()
            e = sum(float((h[k][0] @ W - h[k][1]).pow(2).sum()) for h in held)
            if best is None or e < best[0]:
                best = (e, lam, W)
        e, lam, W = best
        kind, j = k
        if kind == "mlp":
            out["mlp_W"][j - base], out["mlp_b"][j - base] = W[:-1].bfloat16().cpu(), W[-1].bfloat16().cpu()
        else:
            out["attn"][j] = (W[:-1].bfloat16().cpu(), W[-1].bfloat16().cpu())
        out["rel"][f"{kind}{j}"] = (e / max(sum(float(h[k][1].pow(2).sum()) for h in held), 1e-30)) ** 0.5
        out["lam"][f"{kind}{j}"] = lam
        del A, B
    log(f"  held-out relative error: {out['rel']}")
    return out


def part_skip(fit: dict, cut: int, mlp: bool, attn: bool, device="cuda") -> PartSkip:
    base = fit["base"]
    m = (fit["mlp_W"].to(device), fit["mlp_b"].to(device)) if mlp else None
    a = {j: (W.to(device), b.to(device)) for j, (W, b) in fit["attn"].items() if j >= cut} if attn else None
    return PartSkip(cut, base, m, a)


def part_share(cfg_text, cut: int, mlp: bool, attn: bool, whole_attn_known: bool = True) -> float:
    """Skipped share of the state pass's weight-multiply work (attention scores not counted; stock projection sizes):
    mlp saves 3 D I - D^2 per layer, attn saves the q (with gate) and output projections - D^2 per attention layer."""
    c, D = cfg_text, cfg_text.hidden_size
    mlp_w = 3 * D * c.intermediate_size
    hd = getattr(c, "head_dim", D // c.num_attention_heads)
    attn_full = D * (2 * c.num_attention_heads * hd) + 2 * D * c.num_key_value_heads * hd + c.num_attention_heads * hd * D
    attn_keep = 2 * D * c.num_key_value_heads * hd
    kd, vd = c.linear_num_key_heads * c.linear_key_head_dim, c.linear_num_value_heads * c.linear_value_head_dim
    gdn_full = D * (2 * kd + vd) + 2 * D * vd + 2 * D * c.linear_num_value_heads
    full = saved = 0
    for j, t in enumerate(c.layer_types):
        full += (attn_full if t == "full_attention" else gdn_full) + mlp_w
        if j >= cut:
            if mlp:
                saved += mlp_w - D * D
            if attn and t == "full_attention":
                saved += attn_full - attn_keep - D * D
    return saved / full
