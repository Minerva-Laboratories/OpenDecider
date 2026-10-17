"""Per-token depth for the skipped state pass ("foveated depth"), an extension of skipkv.py.

skipkv.py gives every state token the same cut c: layers 0..c-1 run, every layer >= c writes only its cache from a
predicted input. Here each token t has its own cut c_t in the set of fitted cuts (or L = all layers): a token is "live"
in layer j while j < c_t (it runs the whole layer, attending to / recurring over everybody's cache), and from its cut up
its cache comes from the map of its own cut applied to its input at that cut. Depths are chosen by recency windows
(graded: the last 32 tokens all layers, the last 512 cut 24, the rest cut 16) and optionally by an error-model score
(E23 of the FoveaKV notes: sum over the last W queries and heads of p_qi ||v_i - o_q||, from the exact attention layer
right below the base cut, stock Qwen3.5 attention only), so the deep tokens are the ones later tokens read most.

Two implementations:
- depth_cache(): quality reference. Every layer >= base runs whole for ALL tokens with the input norm replaced by
  [true normed input for live tokens, predicted input for the rest]; only live rows of the stream are updated. The
  cache written equals the per-token-depth design's exactly; the compute is not representative (counted analytically
  with skipkv.skipped_share).
- segment_cache(): the real, fast form for recency-only plans (contiguous segments): consecutive prompt segments run
  through bb._prefill with their own SkipMaps (or none), each on the cache of the previous ones. Same semantics as
  depth_cache for those plans.
"""
from __future__ import annotations

import contextlib
import sys

import torch
from torch import nn

from .skipkv import SkipMaps, _rms, skipping


class _Fixed(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, _):
        return self.x


class DepthPlan:
    """maps {cut: SkipMaps (normed)}; windows [(n_last_tokens, cut)] (cut = L or None for all layers);
    select (fraction, cut, W) or None: the top `fraction` of the tokens not already deeper, by the error-model score
    at the attention layer base-1, get depth `cut`."""

    def __init__(self, maps: dict, n_layers: int, windows=(), select=None):
        assert all(m.mode == "normed" and not m.alphas for m in maps.values())
        self.maps, self.L = maps, n_layers
        self.base = min(maps)
        self.windows = [(n, n_layers if c is None else c) for n, c in windows]
        self.select = select
        self.ct = None

    def depths(self, T, device, score=None):
        ct = torch.full((T,), self.base, dtype=torch.long, device=device)
        pos = torch.arange(T, device=device)
        for n, c in self.windows:
            ct = torch.where(pos >= T - n, ct.clamp_min(c), ct)
        if self.select is not None and score is not None:
            frac, c, _ = self.select
            k = int(round(frac * T))
            s = score.masked_fill(ct >= c, -float("inf"))
            k = min(k, int((ct < c).sum()))
            if k > 0:
                ct[s.topk(k).indices] = c
        return ct


@torch.no_grad()
def error_score(att, x, pe, W):
    """E23 score per key position from stock Qwen3.5 attention `att` on its normed input x (1, T, D): attention of the
    last W queries (causal), sum over heads and queries of p_qi ||v_i - o_q||."""
    mq = sys.modules[type(att).__module__]
    T, hd = x.shape[1], att.head_dim
    q, _ = torch.chunk(att.q_proj(x).view(1, T, -1, hd * 2), 2, dim=-1)
    q = att.q_norm(q).transpose(1, 2)
    k = att.k_norm(att.k_proj(x).view(1, T, -1, hd)).transpose(1, 2)
    v = att.v_proj(x).view(1, T, -1, hd).transpose(1, 2)
    cos, sin = pe
    q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1).float(), v.repeat_interleave(rep, 1).float()
    W = min(W, T)
    q = q[:, :, T - W:].float()
    s = (q @ k.transpose(-1, -2)) * att.scaling                              # (1, H, W, T)
    qpos = torch.arange(T - W, T, device=x.device)[:, None]
    s = s.masked_fill(torch.arange(T, device=x.device)[None] > qpos, -float("inf"))
    p = s.softmax(-1)
    o = p @ v                                                                 # (1, H, W, hd)
    dev = torch.cdist(o, v)                                                   # (1, H, W, T)
    return (p * dev).sum((0, 1, 2))


class DepthLayer(nn.Module):
    def __init__(self, layer, j, plan):
        super().__init__()
        self.layer, self.j, self.plan = layer, j, plan

    def forward(self, hidden_states, *args, **kw):
        P, j, h = self.plan, self.j, hidden_states
        assert h.shape[0] == 1, "depth_cache: one sequence"
        if j == P.base:
            P.hin = {}
            P.ct = P.depths(h.shape[1], h.device, getattr(P, "score", None))
        if j in P.maps:
            P.hin[j] = h
        live = P.ct > j
        layer = self.layer
        x = layer.input_layernorm(h)
        for c, m in P.maps.items():
            rows = (P.ct == c) & (c <= j)
            if rows.any():
                i = j - c
                pred = torch.addmm(m.b[i].to(h.dtype), _rms(P.hin[c][0]).to(h.dtype), m.W[i].to(h.dtype))
                x = torch.where(rows[None, :, None], pred[None], x)
        saved = layer.input_layernorm
        layer.input_layernorm = _Fixed(x)
        try:
            out = layer(h, *args, **kw)
        finally:
            layer.input_layernorm = saved
        out = out[0] if isinstance(out, tuple) else out
        return torch.where(live[None, :, None], out, h)


@contextlib.contextmanager
def depth_skipping(lm, plan: DepthPlan):
    saved, hook = {}, None
    plan.score = None
    if plan.select is not None:
        att = getattr(lm.layers[plan.base - 1], "self_attn", None)
        assert att is not None, "selection needs an attention layer right below the base cut"

        def pre(mod, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            plan.score = error_score(mod, x, kwargs["position_embeddings"], plan.select[2])
        hook = att.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        for j in range(plan.base, len(lm.layers)):
            saved[j] = lm.layers[j]
            lm.layers[j] = DepthLayer(saved[j], j, plan)
        yield
    finally:
        for j, layer in saved.items():
            lm.layers[j] = layer
        if hook is not None:
            hook.remove()


@torch.no_grad()
def depth_cache(bb, ids, plan: DepthPlan):
    x = torch.tensor([ids], device=bb.device)
    m = torch.ones_like(x, dtype=torch.bool)
    with depth_skipping(bb.lm, plan):
        _, cache = bb._prefill(x, m, ("final",), None, False, x.shape[1])
    return cache


def seg_lengths(T, segs):
    """Token counts of segs [(n or -1, maps)] for a T-token prompt. Segments after the -1 one are filled from the END
    (the last segments are the deepest windows; a short prompt drops the earlier ones first), then segments before it
    (the first tokens of the prompt, e.g. kept at full depth like attention sinks) in order; -1 takes the rest."""
    out, left = [0] * len(segs), T
    r = next((i for i, (n, _) in enumerate(segs) if n < 0), None)
    for i in range(len(segs) - 1, -1 if r is None else r, -1):
        out[i] = min(segs[i][0], left)
        left -= out[i]
    if r is not None:
        for i in range(r):
            out[i] = min(segs[i][0], left)
            left -= out[i]
        out[r] = left
    return out


@torch.no_grad()
def segment_cache(bb, ids, segs):
    """segs [(n_tokens, SkipMaps or None)] in prompt order (n = -1: the rest). Each segment runs on the previous ones'
    cache; a segment with maps runs layers below its cut and writes only caches above."""
    x = torch.tensor([ids], device=bb.device)
    m = torch.ones_like(x, dtype=torch.bool)
    T, a, cache = x.shape[1], 0, None
    C = bb.cfg.prefill_chunk or T
    for n, maps in zip(seg_lengths(T, segs), (mp for _, mp in segs)):
        if n <= 0:
            continue
        sl = slice(a, a + n)
        if maps is None:
            _, cache = bb._prefill(x[:, sl], m[:, sl], ("final",), cache, False, C)
        elif hasattr(maps, "ctx"):                   # skipparts.PartSkip: MLP / attention surrogates
            with maps.ctx(bb.lm):
                _, cache = bb._prefill(x[:, sl], m[:, sl], ("final",), cache, False, C)
        else:
            with skipping(bb.lm, maps):
                _, cache = bb._prefill(x[:, sl], m[:, sl], ("final",), cache, False, C, maps)
        a += n
    return cache


def plan_skipped(cfg_text, ct, L, whole_mixer):
    """Mean skipped share of the state pass over tokens with depths ct (skipkv.skipped_share per cut; depth L = 0)."""
    from .skipkv import skipped_share
    tot = 0.0
    for c in ct.unique().tolist():
        s = 0.0 if c >= L else skipped_share(cfg_text, c, whole_mixer=whole_mixer)["skipped"]
        tot += s * int((ct == c).sum())
    return tot / len(ct)
