"""Shared-prefix attention for branched rows (docs/branched_shared_prefix_math.md §2; Hydragen, arXiv:2402.05099).

Every option row continues from its state's cache. The stock path gives each row its own bf16 copy of the state's
K/V (and concatenates the row's keys onto it), so memory grows with rows × state tokens. Here each full-attention
layer reads ONE dequantized copy of the state K/V per state. With one state per call (inference), attention is split
as in Hydragen: all rows' queries attend the shared prefix in one call, each row attends its own tokens causally, and
the two parts are merged exactly by their log-sum-exp (`_split_attention`; the wide flash kernel on CUDA). Every row
then sees the same key layout, so outputs do not depend on row order (the earlier single fused call placed each row's
own keys at an index-dependent offset: permutation TV 0.0028, worst 0.024 on the 2B). With several states and small
prefixes, one batched masked SDPA call reads each row's state K/V gathered (row_state). The row cache holds a
stride-0 placeholder for the attention layers, so position bookkeeping (get_seq_length, RoPE positions) still sees
the prefix length.
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
BATCHED_BYTES = 1 << 30         # max gathered prefix K/V (+ copies) for the one-call multi-state path


def _kernels(dev):
    """Fused kernels only on CUDA: memory-efficient attention (the one with boolean-mask support on sm_87). If it is
    unavailable SDPA raises instead of falling back to the math kernel (which materializes the scores)."""
    if dev.type != "cuda":
        return contextlib.nullcontext()
    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


def _attn_modules(bb):
    return [(i, l.self_attn) for i, l in enumerate(bb.lm.layers)
            if bb.layer_types[i] == "full_attention" and hasattr(l, "self_attn")]


def _split_attention(q, k, v, Kp, Vp, pm, scale):
    """Hydragen split for rows continuing ONE shared prefix: q (n, Hq, t, D) attends the prefix Kp/Vp (Hkv, T, D;
    pm (T,) real keys or None) and its own row's k/v (n, Hkv, t, D) causally; the two softmax parts are merged
    exactly by their log-sum-exp. Every row sees the same key layout whatever its index, so outputs do not depend
    on row order (the old fused call put each row's own keys at an index-dependent offset). Returns (n, Hq, t, D)."""
    n, Hq, t, D = q.shape
    Hkv, T = Kp.shape[0], Kp.shape[1]
    if q.is_cuda and D in (64, 128, 256) and 64 % (Hq // Hkv) == 0:          # wide kernel: BM=64 rows
        from .wide_attention import wide_attention
        qs = q.transpose(0, 1).reshape(1, Hq, n * t, D)                          # all rows' queries, prefix keys only
        o_p, l_p = wide_attention(qs, Kp[None], Vp[None], scale, None if pm is None else pm[None], causal=False,
                                  return_lse=True)
        o_p, l_p = o_p.view(Hq, n, t, D).transpose(0, 1).float(), l_p.view(Hq, n, t).transpose(0, 1)
        o_o, l_o = wide_attention(q, k, v, scale, None, causal=True, return_lse=True)
        o_o = o_o.float()
    else:                                                                          # CPU / other widths: same math
        g = Hq // Hkv
        Ke, Ve = Kp.repeat_interleave(g, 0).float(), Vp.repeat_interleave(g, 0).float()
        o_p = torch.empty(n, Hq, t, D, device=q.device, dtype=torch.float32)
        l_p = torch.empty(n, Hq, t, device=q.device, dtype=torch.float32)
        per = max(1, MASK_ELEMS // (Hq * t * max(T, 1)))                           # bound the score tensor
        for a in range(0, n, per):
            sp = torch.einsum("nhtd,hsd->nhts", q[a:a + per].float(), Ke) * scale
            if pm is not None:
                sp = sp.masked_fill(~pm.view(1, 1, 1, T), float("-inf"))
            l_p[a:a + per] = torch.logsumexp(sp, -1)
            o_p[a:a + per] = torch.softmax(sp, -1) @ Ve
        so = torch.einsum("nhtd,nhud->nhtu", q.float(), k.repeat_interleave(g, 1).float()) * scale
        so = so.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=q.device).tril(), float("-inf"))
        l_o = torch.logsumexp(so, -1)
        o_o = torch.softmax(so, -1) @ v.repeat_interleave(g, 1).float()
    m = torch.maximum(l_p, l_o)
    wp, wo = torch.exp(l_p - m), torch.exp(l_o - m)
    return ((o_p * wp[..., None] + o_o * wo[..., None]) / (wp + wo)[..., None]).to(v.dtype)


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
    if ctx.get("capture") is not None:                    # this pass's own keys/values, for continuations after it
        ctx["capture"][mod.layer_idx] = (k, v)
    Kp, Vp = ctx["kv"][mod.layer_idx]                                          # (S, Hkv, T, D), one per state
    if ctx.get("q_kv") is not None:
        # two-level prefix: state K/V (shared) + each question's own K/V, joined for this layer only
        Kq, Vq = ctx["q_kv"][mod.layer_idx]
        qs = ctx["q_state"]
        Kp, Vp = torch.cat([Kp.index_select(0, qs), Kq.to(Kp.dtype)], 2), torch.cat([Vp.index_select(0, qs),
                                                                                     Vq.to(Vp.dtype)], 2)
    Hkv, g = k.shape[1], q.shape[1] // k.shape[1]
    T = Kp.shape[2]
    dev = q.device
    if ctx.get("row_state") is not None:
        # all rows in ONE call: each row reads its state's prefix K/V (gathered), pads masked; no per-state loop
        st = ctx["row_state"]
        K = torch.cat([Kp.index_select(0, st), k], 2)                         # (R, Hkv, T + t, D)
        V = torch.cat([Vp.index_select(0, st), v], 2)
        mask = ctx["bmask"].get(t)
        if mask is None:                                                       # same for every layer of the pass
            qpos = torch.arange(t, device=dev).repeat(g)                       # query (head-in-group, pos) -> pos
            own = torch.arange(t, device=dev)[None, :] <= qpos[:, None]        # (g t, t) causal within the row
            mask = ctx["bmask"][t] = torch.cat([ctx["pmask"].index_select(0, st)[:, None, None, :].expand(
                R, 1, g * t, T), own[None, None].expand(R, 1, g * t, t)], 3)
        with _kernels(dev):
            o = F.scaled_dot_product_attention(q.reshape(R, Hkv, g * t, D), K, V, attn_mask=mask, scale=mod.scaling)
        o = o.reshape(R, Hkv * g, t, D).transpose(1, 2).reshape(R, t, -1)
        return mod.o_proj(o * torch.sigmoid(gate)), None
    out = torch.empty(R, Hkv * g, t, D, device=dev, dtype=v.dtype)
    for s in ctx["states"]:                                                    # few states (1 at inference)
        r = ctx["rows_of"][s]
        if ctx.get("static"):                  # CUDA graphs: fixed shapes, prefix pads masked instead of sliced
            Kps, Vps, pm = Kp[s], Vp[s], ctx["pmask"][s]
        else:
            off = ctx["pad"][s]                                                # left padding of this state's prefix
            Kps, Vps = Kp[s][:, off:], Vp[s][:, off:]                          # (Hkv, T_s, D): from the 1st real key
            pm = ctx["pmask"][s, off:] if ctx["holes"][s] else None            # pads after it (padded question pass)
        out[r] = _split_attention(q.index_select(0, r), k.index_select(0, r), v.index_select(0, r), Kps, Vps, pm,
                                  mod.scaling)
    o = out.transpose(1, 2).reshape(R, t, -1)
    o = o * torch.sigmoid(gate)
    return mod.o_proj(o), None


@contextlib.contextmanager
def shared_prefix_attention(bb, pc, rows: torch.Tensor, pmask: torch.Tensor, static: bool = False,
                            capture: dict | None = None, q_kv: dict | None = None, q_state: torch.Tensor | None = None):
    """While active, full-attention layers attend to pc's per-state K/V (dequantized once) for rows `rows`
    (row i reads state rows[i]); pmask (S, T) marks real prefix tokens. static=True (CUDA graphs): shapes stay fixed,
    prefix pads are masked instead of sliced, and nothing is read back to the host. Otherwise leading pads are sliced
    off and any later pads (a right-padded question pass, see V3._branched_feats_qcache) are masked.
    capture: dict filled with each attention layer's (k, v) of this pass (the question pass of the question cache).
    q_kv / q_state: a second prefix level. Then `rows` index QUESTIONS, question j's prefix is state q_state[j]'s K/V
    followed by q_kv[layer][j], and pmask (questions, T + Lq) covers both; the join is made one layer at a time, so
    the state's K/V is never copied per question for all layers at once."""
    kv = {}
    for li, _ in _attn_modules(bb):
        K = materialize(pc.tensors[(li, "keys", None)]).to(bb.dtype)
        V = materialize(pc.tensors[(li, "values", None)]).to(bb.dtype)
        kv[li] = (K, V)
    if static:                                                                # one state, every row reads it
        states, rows_of, pad, holes = [0], {0: torch.arange(rows.numel(), device=rows.device)}, None, None
    else:
        states = torch.unique(rows).tolist()
        rows_of = {s: (rows == s).nonzero().squeeze(1) for s in states}
        pm = pmask.bool()
        pad = pm.int().argmax(1).tolist()                                     # leading pads (left-padded states)
        holes = ((~pm).sum(1).cpu() > torch.tensor(pad)).tolist()              # pads after the first real key
    ctx = {"kv": kv, "pad": pad, "holes": holes, "pmask": pmask.bool(), "states": states,
           "rows_of": rows_of, "static": static, "capture": capture, "q_kv": q_kv,
           "q_state": None if q_state is None else q_state.to(pmask.device)}
    if not static and kv and len(states) > 1:
        # several states (e.g. one per question with the question cache): one batched call per layer when the
        # gathered per-row prefix K/V stays small, instead of one call per state
        K0 = next(iter(kv.values()))[0]
        per_row = K0[0].numel() // K0.shape[2] * pmask.shape[1]                # one row's prefix, both levels
        if rows.numel() * per_row * K0.element_size() * 4 <= BATCHED_BYTES:
            ctx["row_state"], ctx["bmask"] = rows.to(K0.device), {}
    mods = _attn_modules(bb)
    for _, m in mods:
        m.forward = types.MethodType(lambda self, *a, **k: _forward(self, ctx, *a, **k), m)
    try:
        yield
    finally:
        for _, m in mods:
            del m.forward
