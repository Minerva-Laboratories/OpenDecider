"""Mercurius-1-4B's compressed KV cache: the cache holds the 4-bit TurboQuant codes, not decompressed K/V.

Mercurius (decisions #64, #66, #68) caches, per token and attention layer: the MLA latent and the decoupled RoPE key,
each TurboQuant-MSE 4-bit in a fixed rotated basis (per-vector fp16 norm + Beta Lloyd-Max codebook), and the per-head
key normaliser 1/rms. Its research forward computes exactly those values but then writes the full K/V into a stock
cache. Here the cache stores the codes (indices packed two per byte), and each attention layer decodes its own past
when it runs, one layer at a time, so the decompressed K/V of a layer exists only during that layer's call.

Per token and layer: latent r/2 bytes + RoPE key (n_rope*rd)/2 bytes + 2 fp16 norms + G fp32 normalisers + an int32
position (the RoPE key is quantized BEFORE rotary, as trained, so the rotary angle is re-applied at read).
The attention math is Mercurius's decoupled forward (surgery/mla_rope.py::_decoupled_forward) line for line; only
where K/V come from changes. Equality with the uncompressed path: tests/test_mercurius_cache.py and
scripts/mercurius_cache_check.py.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicLayer

CODE_ATTRS = ("lat_codes", "lat_norm", "rope_codes", "rope_norm", "inv_r", "pos")


# ---------------------------------------------------------------------------------------------- TurboQuant codes
def _codebook(r, bits, device):
    from mercurius.models.qat import tq_codebook
    cen, bnd = tq_codebook(r, bits)
    return cen.to(device), bnd.to(device)


def tq_encode(x: torch.Tensor, bits: int = 4):
    """x (..., r) in the rotated basis -> (packed uint8 codes (..., r/2), fp16 norms (...)). Same arithmetic as
    mercurius.models.qat.tq_roundtrip: per-vector norm rounded to fp16, unit vector bucketized against the codebook."""
    assert bits == 4, "two codes per byte"
    r = x.shape[-1]
    _, bnd = _codebook(r, bits, x.device)
    cf = x.float()
    n = cf.norm(dim=-1, keepdim=True).half()
    idx = torch.bucketize(cf / n.float().clamp_min(1e-12), bnd).to(torch.uint8)
    if r % 2:                                        # latent widths vary by layer and can be odd: one pad nibble
        idx = F.pad(idx, (0, 1))
    return idx[..., 0::2] | (idx[..., 1::2] << 4), n.squeeze(-1)


def tq_decode(codes: torch.Tensor, norm: torch.Tensor, r: int, dtype, bits: int = 4) -> torch.Tensor:
    """Inverse of tq_encode for width r: (q * n).to(dtype), as tq_roundtrip returns it."""
    lo, hi = codes & 15, codes >> 4
    idx = torch.stack([lo, hi], -1).flatten(-2)[..., :r].long()
    cen, _ = _codebook(r, bits, codes.device)
    return (cen[idx] * norm.float().clamp_min(1e-12).unsqueeze(-1)).to(dtype)


# ---------------------------------------------------------------------------------------------- cache layer
class LatentCodesLayer(DynamicLayer):
    """A cache layer holding Mercurius's compressed attention state (batch-first tensors, CODE_ATTRS)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for a in CODE_ATTRS:
            setattr(self, a, None)

    def append(self, **parts):
        for a in CODE_ATTRS:
            old = getattr(self, a)
            setattr(self, a, parts[a] if old is None else torch.cat([old, parts[a]], 1))
        self.is_initialized = True

    def get_seq_length(self) -> int:
        return 0 if self.pos is None else self.pos.shape[1]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for a in CODE_ATTRS if (t := getattr(self, a)) is not None)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for a in CODE_ATTRS:
            if getattr(self, a) is not None:
                setattr(self, a, getattr(self, a).repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for a in CODE_ATTRS:
            if getattr(self, a) is not None:
                setattr(self, a, getattr(self, a)[indices])

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("compressed Mercurius cache: crop is not supported")


# ---------------------------------------------------------------------------------------------- attention
import os as _os
from opendecider.wide_attention import wide_attention

# Prefill (the state pass) runs in the expanded form: per-group K/V of width D=256 cost ~2.7x fewer attention FLOPs
# than the absorbed keys (rd + r up to 720 wide). Absorbed is for decode-like calls (the option rows). Standard MLA split.
_UNABSORBED = _os.environ.get("MERC_ABSORBED_PREFILL", "0") != "1"
ATTN_MAX_ELEMS = 1 << 26                                         # score elements per query chunk (memory bound)
# Q/K are 448 wide (rope + shared residual + per-group nope): above the 256 of the stock flash kernels, where PyTorch
# falls back to a ~2x slower kernel. wide_attention is a Triton flash kernel for that width.
_WIDE_ATTN = _os.environ.get("MERC_WIDE_ATTN", "1") == "1"


def _key_valid(attention_mask, S):
    """(B, S) bool key-valid mask from HF's 4D causal + padding mask (its last query row sees every valid key);
    None when there is no padding."""
    if attention_mask is None:
        return None
    m = attention_mask[:, 0, -1, :S]
    ok = m if m.dtype == torch.bool else m > torch.finfo(m.dtype).min / 2
    return None if bool(ok.all()) else ok


def _absorbed_attention(sa, q_raw, position_embeddings, allp, attention_mask):
    """Attention of the whole call against the layer's decoded cache in absorbed form: keys [rope key | latent]
    (rd + r, shared by every head, never expanded per head), queries [q_rope | q_abs] from the folded per-head maps,
    per-group 1/rms on the keys, attention on the latent, per-group V up-projection. Query tokens in chunks so the
    score tensor stays under ATTN_MAX_ELEMS at any length. Returns (B, T, H*D) before the output gate."""
    B, T = q_raw.shape[:2]
    H, G, D, rd = sa._H, sa._G, sa.head_dim, sa._rd
    assert sa._n_rope == 1, "absorbed form assumes one decoupled RoPE key (RoRoPE n_rope=1)"
    rep = H // G
    A = absorbed_maps(sa)
    r, bits, dt = A["r"], sa._qat["kv_bits"], torch.bfloat16
    S = allp["pos"].shape[1]
    lat = tq_decode(allp["lat_codes"], allp["lat_norm"], r, dt, bits)                     # (B, S, r) rotated basis
    kt = tq_decode(allp["rope_codes"], allp["rope_norm"], rd, dt, bits) @ sa._qat_R0.to(dt)
    cos_s, sin_s = sa._rotary(lat, allp["pos"].long()[None].expand(3, B, S))
    k_full = torch.cat([_rope_neox(kt, cos_s.to(dt), sin_s.to(dt)), lat], -1)            # (B, S, rd + r)
    inv = allp["inv_r"].float()                                                           # (B, S, G)
    y = type(sa.q_norm).forward(sa.q_norm, q_raw)                                         # RMSNorm only
    qq = torch.einsum("bthd,hed->bthe", y.float(), A["P"])
    cos, sin = position_embeddings
    q_full = torch.cat([_rope_neox(qq[..., :rd], cos.float()[:, :, None], sin.float()[:, :, None]), qq[..., rd:]], -1)
    q_full = q_full.view(B, T, G, rep, rd + r).to(dt)
    out = torch.empty(B, T, G, rep, D, device=q_raw.device, dtype=torch.float32)
    chunk = max(1, ATTN_MAX_ELEMS // (H * S * B))
    for a in range(0, T, chunk):
        b_ = min(T, a + chunk)
        s = torch.einsum("btgpe,bse->bgpts", q_full[:, a:b_], k_full).float()          # (B, G, rep, t, S)
        s = s * inv.permute(0, 2, 1)[:, :, None, None, :] * sa.scaling
        if attention_mask is not None:
            m = attention_mask[:, :, a:b_, :S]                                            # (B, 1, t, S)
            s = s.masked_fill(~m[:, :, None], float("-inf")) if m.dtype == torch.bool else s + m[:, :, None].float()
        else:
            qpos = torch.arange(a, b_, device=s.device) + (S - T)
            s = s.masked_fill(torch.arange(S, device=s.device)[None, :] > qpos[:, None], float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)                                          # fully masked pad rows
        o_lat = torch.einsum("bgpts,bsc->bgptc", p.to(dt), lat).float()
        out[:, a:b_] = torch.einsum("bgptc,gdc->btgpd", o_lat, A["Wv"])
    return out.reshape(B, T, H * D)


def _encode_codes(sa, hidden_states) -> dict:
    """This call's tokens in cache form: TurboQuant codes of the latent and of the RoPE key, and the per-group 1/rms."""
    from mercurius.models.qat import nf4_roundtrip
    B, T = hidden_states.shape[:2]
    lat, bits = sa.k_proj.latent, sa._qat["kv_bits"]
    kraw = sa.k_rms(hidden_states).view(B, T, sa._G, sa.head_dim).float()
    inv_r = torch.rsqrt(kraw.pow(2).mean(-1) + getattr(sa.k_norm, "eps", 1e-6))                     # (B,T,G)
    Wr = sa.__dict__.get("_wr_nf4")
    if Wr is None:                                   # the trained forward's NF4(k_rope), fixed weights: once.
        w = lat.k_rope.weight                        # Same arithmetic as its ste(w, nf4(w)) (w + (q - w) is not
        Wr = sa.__dict__["_wr_nf4"] = w + (nf4_roundtrip(w) - w)     # bit-identical to q, and 4-bit codes flip on it)
    kt0 = F.linear(hidden_states, Wr.to(hidden_states.dtype))                                       # (B,T,nr*rd)
    rope_codes, rope_norm = tq_encode(kt0 @ sa._qat_R0.to(kt0.dtype).T, bits)
    c = lat.down(hidden_states)
    lat_codes, lat_norm = tq_encode(c @ lat._tq_rot.to(c.device, c.dtype).T, bits)
    return {"lat_codes": lat_codes, "lat_norm": lat_norm, "rope_codes": rope_codes, "rope_norm": rope_norm,
            "inv_r": inv_r}


def _expand_kv(sa, codes, cos, sin, dt):
    """Decoded cache -> expanded per-group keys (B, G, S, rd + (G-1)rd + D-rd) with 1/rms folded in, values (B, G, S, D):
    exactly the values the research forward attends to. cos/sin: rotary at the cached tokens' positions."""
    from mercurius.surgery.mla_rope import _rope_blocks
    G, D, rd, nr, bits = sa._G, sa.head_dim, sa._rd, sa._n_rope, sa._qat["kv_bits"]
    lat = sa.k_proj.latent
    Rl = lat._tq_rot.to(cos.device, dt)
    kt = tq_decode(codes["rope_codes"], codes["rope_norm"], nr * rd, dt, bits) @ sa._qat_R0.to(dt)        # (B,S,nr*rd)
    cq = tq_decode(codes["lat_codes"], codes["lat_norm"], Rl.shape[0], dt, bits) @ Rl
    B, S = cq.shape[:2]
    kres, v = lat.up_k(cq), lat.up_v(cq)
    nres = (G - nr) * rd
    k_res, k_nope = kres[..., :nres], kres[..., nres:].view(B, S, G, D - rd)
    k_rope = _rope_blocks(kt.view(B, S, 1, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
    K = torch.cat([k_rope.expand(B, S, G, nr * rd), k_res.view(B, S, 1, nres).expand(B, S, G, nres), k_nope], -1)
    K = (K.float() * codes["inv_r"].float().unsqueeze(-1)).to(dt)
    return K.transpose(1, 2), v.view(B, S, G, D).transpose(1, 2)


def _expand_q(sa, q_raw, position_embeddings):
    """Queries in the expanded form, as in the research forward: (B, H, T, rd + (G-1)rd + D-rd)."""
    from mercurius.surgery.mla_rope import _rope_blocks
    B, T, H = q_raw.shape[:3]
    G, D, rd, nr = sa._G, sa.head_dim, sa._rd, sa._n_rope
    half, nres = rd // 2, (G - nr) * rd
    q = sa.q_norm(q_raw)
    Ug = sa._U_heads.to(q.dtype)
    qc = torch.cat([q[..., :half].unsqueeze(-2) * Ug, q[..., half:rd].unsqueeze(-2) * Ug], -1)
    cos, sin = position_embeddings
    q_rope = _rope_blocks(qc[..., :nr, :].reshape(B, T, H, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
    return torch.cat([q_rope, qc[..., nr:, :].reshape(B, T, H, nres), q[..., rd:]], -1).transpose(1, 2)


def _compressed_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                        position_ids=None, **kwargs):
    """mla_rope._decoupled_forward with the deployed (TurboQuant) cache. Writes this call's codes, decodes the
    layer's whole past (including them) and runs the same attention."""
    B, T = hidden_states.shape[:2]
    H, D = self._H, self.head_dim
    q_raw, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, H, 2 * D), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    if position_ids is None:
        raise ValueError("the compressed Mercurius cache needs position_ids (re-applies rotary at read)")
    parts = _encode_codes(self, hidden_states)
    parts["pos"] = position_ids.to(torch.int32).expand(B, T)
    if past_key_values is not None:
        layer = past_key_values.layers[self.layer_idx]
        if not isinstance(layer, LatentCodesLayer):
            if layer.get_seq_length():
                raise ValueError("cache layer already holds uncompressed K/V")
            layer = past_key_values.layers[self.layer_idx] = LatentCodesLayer()
        layer.append(**parts)
        allp = {a: getattr(layer, a) for a in CODE_ATTRS}
    else:
        allp = parts
    S = allp["pos"].shape[1]
    dt = hidden_states.dtype
    if not _UNABSORBED:
        out = _absorbed_attention(self, q_raw, position_embeddings, allp, attention_mask)
        return self.o_proj(out.to(dt) * torch.sigmoid(gate)), None
    cos_s, sin_s = self._rotary(hidden_states, allp["pos"].long()[None].expand(3, B, S))
    K, V = _expand_kv(self, allp, cos_s, sin_s, dt)                                  # K/V keep G heads (GQA)
    Q = _expand_q(self, q_raw, position_embeddings)
    if Q.is_cuda and _WIDE_ATTN:
        out = wide_attention(Q, K, V, self.scaling, _key_valid(attention_mask, S))
    else:
        out = Q.new_empty(B, H, T, D)
        chunk = max(1, ATTN_MAX_ELEMS // (H * S * B)) if T * S * H * B > ATTN_MAX_ELEMS else T
        for a in range(0, T, chunk):
            b_ = min(T, a + chunk)
            if attention_mask is not None:
                m = attention_mask[:, :, a:b_, :S]
            elif T > 1:
                m = torch.arange(S, device=Q.device)[None] <= torch.arange(a, b_, device=Q.device)[:, None] + (S - T)
            else:
                m = None
            out[:, :, a:b_] = F.scaled_dot_product_attention(Q[:, :, a:b_], K, V, attn_mask=m, scale=self.scaling,
                                                             enable_gqa=True)
    out = out.transpose(1, 2).reshape(B, T, -1).contiguous() * torch.sigmoid(gate)
    return self.o_proj(out), None


def install_compressed_cache(trunk) -> int:
    """Use the compressed cache in every decoupled-RoPE MLA layer of `trunk` (the deployed Mercurius text model)."""
    n = 0
    for layer in trunk.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is None or not getattr(sa, "_rope_decoupled", False):
            continue
        if getattr(sa, "_qat", None) is None or getattr(sa.k_proj.latent, "_tq_rot", None) is None:
            raise ValueError("expects the deployed Mercurius (mercurius_port.deploy_mercurius)")
        sa.__dict__["_rotary"] = trunk.rotary_emb          # not a submodule: no duplicate registration
        sa.forward = types.MethodType(_compressed_forward, sa)
        n += 1
    return n


# ---------------------------------------------------------------------------------------------- shared-prefix rows
# OpenDecider's option rows all continue the same state. Instead of giving every row its own copy of the state's
# cache (which the plain forward would then decode once per row), the state's codes are decoded ONCE per layer and
# every row attends to them in the absorbed form the llama.cpp graph uses:
#     [q_rope | q_abs] = P_h RMSNorm(q_h)          P_h folds the post-norm map, the RoRoPE mixing and the key
#                                                   up-projections (latent in its rotated basis)
#     s = ([q_rope | q_abs] . [rope(R0^T kr) | c]) * inv_r_g * scale ; o_h = W_v,g' sum_s p_s c_s
# The rows' own tokens are TurboQuant'd like every cached token.

@torch.no_grad()
def absorbed_maps(sa) -> dict:
    """Per attention layer, cached on the module: P (H, rd + r, D), W_v' (G, D, r) acting on the rotated latent."""
    a = sa.__dict__.get("_merc_absorbed")
    if a is not None:
        return a
    from opendecider.mercurius_port import _bnb_dense
    H, G, D = sa._H, sa._G, sa.head_dim
    rd, nr = sa._rd, sa._n_rope
    half = rd // 2
    lat = sa.k_proj.latent
    R = lat._tq_rot.float()
    r = R.shape[0]
    Wk = _bnb_dense(lat.up_k)
    nres = (G - nr) * rd
    W_res, W_nope = Wk[:nres], Wk[nres:].view(G, D - rd, r)
    Rq, U = sa.q_norm.R.float(), sa._U_heads.float()
    g_of_h = torch.arange(H, device=R.device) // (H // G)
    I = torch.eye(D, device=R.device)
    P = torch.empty(H, rd + r, D, device=R.device)
    for h in range(H):
        q = I @ Rq[h]
        qc = torch.cat([q[:, :half].unsqueeze(1) * U[h], q[:, half:rd].unsqueeze(1) * U[h]], -1)   # (D, G, rd)
        q_abs = qc[:, 1:, :].reshape(D, nres) @ W_res + q[:, rd:] @ W_nope[g_of_h[h]]
        P[h, :rd] = qc[:, 0, :].T
        P[h, rd:] = (q_abs @ R.T).T
    a = {"P": P, "Wv": _bnb_dense(lat.up_v).view(G, D, r) @ R.T, "r": r}
    sa.__dict__["_merc_absorbed"] = a
    return a


def _rope_neox(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


@torch.no_grad()
def _decode_state(sa, pc, li):
    """The layer's cached state, decoded once: keys [rope key | latent] (S, T, rd + r), latent (S, T, r), inv_r (S, T, G)."""
    t = {a: pc.tensors[(li, a, None)] for a in CODE_ATTRS}
    rd, r, bits = sa._rd, sa.k_proj.latent._tq_rot.shape[0], sa._qat["kv_bits"]
    dt = torch.bfloat16
    lat = tq_decode(t["lat_codes"], t["lat_norm"], r, dt, bits)                         # rotated basis
    kt = tq_decode(t["rope_codes"], t["rope_norm"], rd, dt, bits) @ sa._qat_R0.to(dt)
    S, T = t["pos"].shape
    cos, sin = sa._rotary(lat, t["pos"].long()[None].expand(3, S, T))
    k = torch.cat([_rope_neox(kt, cos.to(dt), sin.to(dt)), lat], -1)
    return k, lat, t["inv_r"].float()


def _row_forward(sa, ctx, li, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
    from mercurius.models.qat import nf4_roundtrip
    R_, t, _ = hidden_states.shape
    H, G, D, rd = sa._H, sa._G, sa.head_dim, sa._rd
    rep = H // G
    A = absorbed_maps(sa)
    r, bits, dt = A["r"], sa._qat["kv_bits"], torch.bfloat16
    lat = sa.k_proj.latent
    q, gate = torch.chunk(sa.q_proj(hidden_states).view(R_, t, H, 2 * D), 2, dim=-1)
    gate = gate.reshape(R_, t, -1)
    y = type(sa.q_norm).forward(sa.q_norm, q)                                           # RMSNorm only (map is in P)
    qq = torch.einsum("bthd,hed->bthe", y.float(), A["P"])                              # (R, t, H, rd + r)
    cos, sin = position_embeddings
    q_full = torch.cat([_rope_neox(qq[..., :rd], cos.float()[:, :, None], sin.float()[:, :, None]), qq[..., rd:]], -1)
    # the rows' own tokens, cached form (TurboQuant round trip), as the compressed forward writes them
    Wr = sa.__dict__.get("_wr_nf4")
    if Wr is None:
        w = lat.k_rope.weight
        Wr = sa.__dict__["_wr_nf4"] = w + (nf4_roundtrip(w) - w)
    R0 = sa._qat_R0.to(dt)
    kt = tq_decode(*tq_encode(F.linear(hidden_states, Wr.to(dt)) @ R0.T, bits), rd, dt, bits) @ R0
    c = lat.down(hidden_states)
    Rl = lat._tq_rot.to(c.device, c.dtype)
    c_rot = tq_decode(*tq_encode(c @ Rl.T, bits), r, dt, bits)                           # (R, t, r)
    k_own = torch.cat([_rope_neox(kt, cos.to(dt), sin.to(dt)), c_rot], -1)              # (R, t, rd + r)
    kraw = sa.k_rms(hidden_states).view(R_, t, G, D).float()
    inv_own = torch.rsqrt(kraw.pow(2).mean(-1) + getattr(sa.k_norm, "eps", 1e-6))       # (R, t, G)

    if li not in ctx["dec"]:
        ctx["dec"][li] = _decode_state(sa, ctx["pc"], li)
    Ks, Ls, Is = ctx["dec"][li]
    rows = ctx["rows"]
    out = torch.empty(R_, t, H, D, device=hidden_states.device, dtype=torch.float32)
    causal = torch.ones(t, t, dtype=torch.bool, device=hidden_states.device).tril()
    for s in torch.unique(rows).tolist():
        rr_all = (rows == s).nonzero().squeeze(1)
        T = Ks.shape[1]
        per = max(1, ctx["max_elems"] // (rep * t * (T + t)))
        for a0 in range(0, rr_all.numel(), per):
            rr = rr_all[a0:a0 + per]
            n = rr.numel()
            Q = q_full.index_select(0, rr).view(n, t, G, rep, rd + r).permute(0, 2, 3, 1, 4)   # (n, G, rep, t, e)
            sp = torch.einsum("ngpte,se->ngpts", Q.to(dt), Ks[s]).float()                    # prefix (n,G,rep,t,T)
            sp = sp * Is[s].T[None, :, None, None, :]
            sp = sp.masked_fill(~ctx["pmask"][s][None, None, None, None, :], float("-inf"))
            ko = k_own.index_select(0, rr)                                                    # (n, t', e)
            so = torch.einsum("ngpte,nue->ngptu", Q.to(dt), ko).float()
            so = so * inv_own.index_select(0, rr).permute(0, 2, 1)[:, :, None, None, :]
            so = so.masked_fill(~causal[None, None, None], float("-inf"))
            p = torch.softmax(torch.cat([sp, so], -1) * sa.scaling, -1)
            o_lat = torch.einsum("ngpts,sc->ngptc", p[..., :T].to(dt), Ls[s]).float() + \
                torch.einsum("ngptu,nuc->ngptc", p[..., T:].to(dt), c_rot.index_select(0, rr)).float()
            o = torch.einsum("ngptc,gdc->ngptd", o_lat, A["Wv"])                              # (n, G, rep, t, D)
            out[rr] = o.permute(0, 3, 1, 2, 4).reshape(n, t, H, D)
    o = out.reshape(R_, t, H * D).to(hidden_states.dtype) * torch.sigmoid(gate)
    return sa.o_proj(o), None


def _row_forward_flash(sa, ctx, li, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                       **kw):
    """Rows continuing the shared state(s), expanded form, nothing of size rows x state materialized: every row's
    queries of one state go through the wide flash kernel against that state's keys (decoded and expanded once per
    layer, transient), the rows' own tokens (TurboQuant round trip, as the cache would hold them) are attended
    causally in PyTorch, and the two parts are merged by their log-sum-exp."""
    R_, t, _ = hidden_states.shape
    H, G, D = sa._H, sa._G, sa.head_dim
    rep, dt = H // G, hidden_states.dtype
    q_raw, gate = torch.chunk(sa.q_proj(hidden_states).view(R_, t, H, 2 * D), 2, dim=-1)
    gate = gate.reshape(R_, t, -1)
    Q = _expand_q(sa, q_raw, position_embeddings)                                         # (R, H, t, E)
    E = Q.shape[-1]
    cos, sin = position_embeddings
    Ko, Vo = _expand_kv(sa, _encode_codes(sa, hidden_states), cos, sin, dt)                # (R, G, t, E/D)
    so = torch.einsum("rgpte,rgue->rgptu", Q.view(R_, G, rep, t, E), Ko).float() * sa.scaling
    causal = torch.ones(t, t, dtype=torch.bool, device=Q.device).tril()
    so = so.masked_fill(~causal, float("-inf"))
    lse_o = torch.logsumexp(so, -1)                                                       # (R, G, rep, t)
    o_own = torch.einsum("rgptu,rgud->rgptd", torch.softmax(so, -1).to(dt), Vo).float()
    o_pre = torch.empty(R_, H, t, D, device=Q.device, dtype=torch.float32)
    lse_p = torch.empty(R_, H, t, device=Q.device, dtype=torch.float32)
    pc, rows = ctx["pc"], ctx["rows"]
    for st in torch.unique(rows).tolist():
        rr = (rows == st).nonzero().squeeze(1)
        n = rr.numel()
        codes = {a: pc.tensors[(li, a, None)][st:st + 1] for a in CODE_ATTRS}
        S = codes["pos"].shape[1]
        c_s, s_s = sa._rotary(hidden_states, codes["pos"].long()[None].expand(3, 1, S))
        Ks, Vs = _expand_kv(sa, codes, c_s, s_s, dt)                                       # (1, G, S, E/D)
        Qs = Q.index_select(0, rr).transpose(0, 1).reshape(1, H, n * t, E)
        o, lse = wide_attention(Qs, Ks, Vs, sa.scaling, ctx["pmask"][st:st + 1], causal=False, return_lse=True)
        o_pre[rr] = o.view(H, n, t, D).transpose(0, 1).float()
        lse_p[rr] = lse.view(H, n, t).transpose(0, 1)
        del Ks, Vs
    lse_o, o_own = lse_o.view(R_, H, t), o_own.view(R_, H, t, D)
    m = torch.maximum(lse_p, lse_o)
    wp, wo = torch.exp(lse_p - m), torch.exp(lse_o - m)
    out = (o_pre * wp[..., None] + o_own * wo[..., None]) / (wp + wo)[..., None]
    o = out.transpose(1, 2).reshape(R_, t, H * D).to(dt) * torch.sigmoid(gate)
    return sa.o_proj(o), None


def pc_device(pc):
    return next(iter(pc.tensors.values())).device


import contextlib


@contextlib.contextmanager
def merc_shared_prefix_attention(bb, pc, rows, pmask, max_elems: int = 1 << 26):
    """While active, every Mercurius attention layer of bb attends the state(s) in `pc` once-decoded (rows[i] reads
    state rows[i]; pmask (S, T) marks real state tokens). Rows' caches are placeholders (PackedCache.unpack_rows with
    attn_placeholder=True), so the state's codes are never copied per row."""
    ctx = {"pc": pc, "rows": rows, "pmask": pmask.bool(), "dec": {}, "max_elems": max_elems}
    mods = [(i, l.self_attn) for i, l in enumerate(bb.lm.layers)
            if hasattr(l, "self_attn") and getattr(l.self_attn, "_rope_decoupled", False)]
    saved = [(sa, sa.__dict__.get("forward")) for _, sa in mods]
    fwd = _row_forward_flash if (_WIDE_ATTN and pc_device(pc).type == "cuda") else _row_forward
    for li, sa in mods:
        sa.forward = types.MethodType(lambda self, *a, _li=li, _f=fwd, **k: _f(self, ctx, _li, *a, **k), sa)
    try:
        yield
    finally:
        for sa, f in saved:
            if f is None:
                sa.__dict__.pop("forward", None)
            else:
                sa.forward = f
