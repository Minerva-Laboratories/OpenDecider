"""Shared-prefix attention for branched rows (docs/branched_shared_prefix_math.md §2; Hydragen, arXiv:2402.05099).

Every option row continues from its state's cache. The stock path gives each row its own bf16 copy of the state's
K/V (and concatenates the row's keys onto it), so memory grows with rows × state tokens. Here each full-attention
layer attends to ONE dequantized copy of the state K/V per state plus the row's own keys, in one fused SDPA call per
state (memory-efficient kernel; scores are never materialized). The g query heads of each KV head are folded into the
query length, so GQA needs no K/V expansion, and a boolean mask limits each row's suffix keys to its own tokens. The row cache holds a stride-0 placeholder for the attention layers, so position bookkeeping
(get_seq_length, RoPE positions) still sees the prefix length.
GDN layers are untouched (their per-row state is fixed-size).
"""
from __future__ import annotations

import contextlib
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .quant import materialize

MASK_ELEMS = 1 << 25            # max boolean-mask elements per SDPA call (32M)


def _kernels(dev):
    """Fused kernels only on CUDA: memory-efficient attention (the one with boolean-mask support on sm_87). If it is
    unavailable SDPA raises instead of falling back to the math kernel (which materializes the scores)."""
    if dev.type != "cuda":
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def _attn_modules(bb):
    return [(i, l.self_attn) for i, l in enumerate(bb.lm.layers)
            if bb.layer_types[i] == "full_attention" and hasattr(l, "self_attn")]


def _forward(mod, ctx, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
    """Same math as Qwen3_5Attention.forward (gated q, q/k RMSNorm, RoPE, output gate), shared-prefix attention."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    R, t, _ = hidden_states.shape
    D = mod.head_dim
    q, gate = torch.chunk(mod.q_proj(hidden_states).view(R, t, -1, D * 2), 2, dim=-1)
    gate = gate.reshape(R, t, -1)
    q = mod.q_norm(q).transpose(1, 2)                                          # (R, Hq, t, D)
    k = mod.k_norm(mod.k_proj(hidden_states).view(R, t, -1, D)).transpose(1, 2)  # (R, Hkv, t, D)
    v = mod.v_proj(hidden_states).view(R, t, -1, D).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    Kp, Vp = ctx["kv"][mod.layer_idx]                                          # (S, Hkv, T, D), one per state
    Hkv, g = k.shape[1], q.shape[1] // k.shape[1]
    T = Kp.shape[2]
    dev = q.device
    out = torch.empty(R, Hkv, g, t, D, device=dev, dtype=v.dtype)
    qg = q.view(R, Hkv, g, t, D)
    rmax = ctx["rows_of_max"]
    krow = torch.arange(rmax, device=dev).repeat_interleave(t)                 # suffix key -> row, position
    kpos = torch.arange(t, device=dev).repeat(rmax)
    for s in ctx["states"]:                                                    # few states (1 at inference)
        r = ctx["rows_of"][s]
        off = ctx["pad"][s]                                                    # left padding of this state's prefix
        Kps, Vps = Kp[s][:, off:], Vp[s][:, off:]                              # (Hkv, T_s, D): real prefix only
        Ts = T - off
        # rows in groups so the boolean mask (queries x keys) stays under MASK_ELEMS
        per = max(1, MASK_ELEMS // (g * t * (Ts + t)))
        for a in range(0, r.numel(), per):
            rr = r[a:a + per]
            n = rr.numel()
            # GQA without copies: the g query heads of each kv head go into the query length dim
            Q = qg.index_select(0, rr).permute(1, 2, 0, 3, 4).reshape(Hkv, 1, g * n * t, D)
            ks = k.index_select(0, rr).permute(1, 0, 2, 3).reshape(Hkv, n * t, D)
            vs = v.index_select(0, rr).permute(1, 0, 2, 3).reshape(Hkv, n * t, D)
            K = torch.cat([Kps, ks], 1)[:, None]                                  # (Hkv, 1, Ts + n t, D)
            V = torch.cat([Vps, vs], 1)[:, None]
            # mask: every query sees the whole prefix; in the suffix only its own row, causally
            qrow = torch.arange(n, device=dev).view(1, n, 1).expand(g, n, t).reshape(-1)
            qpos = torch.arange(t, device=dev).view(1, 1, t).expand(g, n, t).reshape(-1)
            own = (qrow[:, None] == krow[:n * t][None]) & (kpos[:n * t][None] <= qpos[:, None])
            mask = torch.cat([torch.ones(own.shape[0], Ts, dtype=torch.bool, device=dev), own], 1)
            with _kernels(dev):
                o = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask[None, None], scale=mod.scaling)
            out[rr] = o.view(Hkv, g, n, t, D).permute(2, 0, 1, 3, 4)
    o = out.view(R, Hkv * g, t, D).transpose(1, 2).reshape(R, t, -1)
    o = o * torch.sigmoid(gate)
    return mod.o_proj(o), None


@contextlib.contextmanager
def shared_prefix_attention(bb, pc, rows: torch.Tensor, pmask: torch.Tensor):
    """While active, full-attention layers attend to pc's per-state K/V (dequantized once) for rows `rows`
    (row i reads state rows[i]); pmask (S, T) marks real prefix tokens."""
    kv = {}
    for li, _ in _attn_modules(bb):
        K = materialize(pc.tensors[(li, "keys", None)]).to(bb.dtype)
        V = materialize(pc.tensors[(li, "values", None)]).to(bb.dtype)
        kv[li] = (K, V)
    states = torch.unique(rows).tolist()
    rows_of = {s: (rows == s).nonzero().squeeze(1) for s in states}
    pad = (~pmask.bool()).sum(1).tolist()                                     # prefixes are LEFT-padded
    ctx = {"kv": kv, "pad": pad, "states": states, "rows_of": rows_of,
           "rows_of_max": max(int(v.numel()) for v in rows_of.values())}
    mods = _attn_modules(bb)
    for _, m in mods:
        m.forward = types.MethodType(lambda self, *a, **k: _forward(self, ctx, *a, **k), m)
    try:
        yield
    finally:
        for _, m in mods:
            del m.forward
