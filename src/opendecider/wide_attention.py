"""Flash attention (Triton) for query/key widths above the 256 the stock kernels are built for: Mercurius's expanded
MLA heads score on 448 dims (64 RoPE + 192 shared residual + 192 per-group nope) against 256-wide values. The Q.K
reduction is tiled over the head dimension, the softmax is online (FlashAttention-2), the query heads of one GQA group
share a tile (rows = token x head) so each K/V tile is read once per group, causal blocks past the diagonal are
skipped, and left padding comes in as a per-row key-valid mask.
    out = wide_attention(q (B,H,T,Dk), k (B,G,S,Dk), v (B,G,S,Dv), scale, kvalid=None (B,S) bool)
Queries are the last T of the S positions (causal with offset S - T)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _wide_fwd(Q, K, V, O, LSE, KV_OK, scale_log2, T, S, G, DK,
              sqb, sqh, sqt, skb, skg, sks, svb, svg, svs, sob, soh, sot, smb,
              HAS_MASK: tl.constexpr, REP: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
              DKP: tl.constexpr, DV: tl.constexpr, CAUSAL: tl.constexpr, WRITE_LSE: tl.constexpr):
    BM: tl.constexpr = BT * REP
    pid_m = tl.program_id(0)
    bg = tl.program_id(1)
    b = bg // G
    g = bg % G
    rows = tl.arange(0, BM)
    tok = pid_m * BT + rows // REP                                # row = (token, head of the group)
    head = g * REP + rows % REP
    qpos = tok + (S - T)
    offs_d = tl.arange(0, BD)
    offs_v = tl.arange(0, DV)
    q_rows = Q + b * sqb + head * sqh + tok * sqt
    k_base = K + b * skb + g * skg
    v_base = V + b * svb + g * svg
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, DV], tl.float32)
    if CAUSAL:
        end = tl.minimum(S, (pid_m + 1) * BT + S - T)
    else:
        end = S
    for n0 in range(0, end, BN):
        offs_n = n0 + tl.arange(0, BN)
        qk = tl.zeros([BM, BN], tl.float32)
        for d0 in tl.static_range(0, DKP, BD):
            dd = d0 + offs_d
            q = tl.load(q_rows[:, None] + dd[None, :], mask=(tok[:, None] < T) & (dd[None, :] < DK), other=0.0)
            k = tl.load(k_base + offs_n[:, None] * sks + dd[None, :], mask=(offs_n[:, None] < S) & (dd[None, :] < DK), other=0.0)
            qk += tl.dot(q, tl.trans(k))
        ok = offs_n[None, :] < S
        if CAUSAL:
            ok = ok & (offs_n[None, :] <= qpos[:, None])
        if HAS_MASK:
            kv = tl.load(KV_OK + b * smb + offs_n, mask=offs_n < S, other=0)
            ok = ok & (kv[None, :] != 0)
        qk = tl.where(ok, qk * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * svs + offs_v[None, :], mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]          # rows with no visible key (padding) -> 0
    tl.store(O + b * sob + head[:, None] * soh + tok[:, None] * sot + offs_v[None, :], out.to(O.dtype.element_ty),
             mask=tok[:, None] < T)
    if WRITE_LSE:                                                 # natural-log LSE of the scaled scores; -inf if none
        lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
        tl.store(LSE + (b * (T * G * REP) + head * T + tok), lse, mask=tok < T)


def wide_attention(q, k, v, scale: float, kvalid=None, causal: bool = True, return_lse: bool = False, BM: int = 64,
                   BN: int = 64, BD: int = 32, num_warps: int = 4, num_stages: int = 2):
    """BM: rows per tile (tokens x the group's query heads). causal=False: every query sees every (valid) key.
    return_lse: also the (B, H, T) fp32 log-sum-exp of the scaled scores, to merge with attention over other keys."""
    B, H, T, DK = q.shape
    G, S, DV = k.shape[1], k.shape[2], v.shape[-1]
    rep = H // G
    assert H % G == 0 and v.shape[2] == S and DV in (64, 128, 256) and BM % rep == 0
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    o = q.new_empty(B, H, T, DV)
    if kvalid is not None:
        kvalid = kvalid.to(torch.int8).contiguous()
    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32) if return_lse else o
    bt = BM // rep
    grid = (triton.cdiv(T, bt), B * G)
    _wide_fwd[grid](q, k, v, o, lse, kvalid if kvalid is not None else q, scale * 1.4426950408889634, T, S, G, DK,
                    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
                    kvalid.stride(0) if kvalid is not None else 0,
                    HAS_MASK=kvalid is not None, REP=rep, BT=bt, BN=BN, BD=BD, DKP=triton.cdiv(DK, BD) * BD, DV=DV,
                    CAUSAL=causal, WRITE_LSE=return_lse, num_warps=num_warps, num_stages=num_stages)
    return (o, lse) if return_lse else o
