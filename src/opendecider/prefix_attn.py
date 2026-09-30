"""Shared-prefix attention for branched rows (docs/branched_shared_prefix_math.md §2; Hydragen, arXiv:2402.05099).

Every option row continues from its state's cache. The stock path gives each row its own bf16 copy of the state's
K/V (and concatenates the row's keys onto it), so memory grows with rows × state tokens. Here each full-attention
layer attends to ONE dequantized copy of the state K/V per state plus the row's own keys, in a single softmax over
[prefix ; suffix] scores: exact, with no per-row prefix copy. The row cache holds a stride-0 placeholder for the
attention layers so position bookkeeping (get_seq_length, RoPE positions) still sees the prefix length.
GDN layers are untouched (their per-row state is fixed-size).
"""
from __future__ import annotations

import contextlib
import types

import torch

from .quant import materialize

NEG = -1e30


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
    out = torch.empty(R, Hkv, g, t, D, device=q.device, dtype=v.dtype)
    qg = q.view(R, Hkv, g, t, D)
    rows = ctx["rows"]
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    for s in ctx["states"]:                                                    # few states (1 at inference)
        r = ctx["rows_of"][s]
        n = r.numel()
        qs = qg.index_select(0, r)                                             # (n, Hkv, g, t, D)
        # prefix scores: all rows of this state against its K, one bmm per kv head (no copy of K)
        qh = qs.permute(1, 0, 2, 3, 4).reshape(Hkv, n * g * t, D)
        sp = torch.bmm(qh, Kp[s].transpose(1, 2)).view(Hkv, n, g, t, T).permute(1, 0, 2, 3, 4).float()
        sp = sp.masked_fill(~ctx["pmask"][s].view(1, 1, 1, 1, T), NEG)
        # suffix scores: causal attention inside each row
        ks, vs = k.index_select(0, r), v.index_select(0, r)                    # (n, Hkv, t, D)
        ss = torch.einsum("nhgtd,nhsd->nhgts", qs, ks).float().masked_fill(~causal, NEG)
        w = torch.softmax(torch.cat([sp, ss], -1) * mod.scaling, -1)
        wp, ws = w[..., :T].to(v.dtype), w[..., T:].to(v.dtype)
        op = torch.bmm(wp.permute(1, 0, 2, 3, 4).reshape(Hkv, n * g * t, T), Vp[s]).view(Hkv, n, g, t, D)
        os_ = torch.einsum("nhgts,nhsd->nhgtd", ws, vs)
        out[r] = op.permute(1, 0, 2, 3, 4) + os_
    o = out.view(R, Hkv * g, t, D).transpose(1, 2).reshape(R, t, -1)
    o = o * torch.sigmoid(gate)
    del rows
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
    ctx = {"kv": kv, "rows": rows, "pmask": pmask.bool(), "states": states,
           "rows_of": {s: (rows == s).nonzero().squeeze(1) for s in states}}
    mods = _attn_modules(bb)
    for _, m in mods:
        m.forward = types.MethodType(lambda self, *a, **k: _forward(self, ctx, *a, **k), m)
    try:
        yield
    finally:
        for _, m in mods:
            del m.forward
