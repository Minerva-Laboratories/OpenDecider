"""Mercurius-1-4B -> GGUF for the llama.cpp `mercurius` architecture (TurboQuant fork + Mercurius additions).

Attention layers (absorbed MLA with a decoupled RoPE key, decisions #64-#68 of the Mercurius repo) are folded into the
form the llama.cpp graph computes:
    y_h      = RMSNorm_(1+w)(q_h)                         normalized query of head h (256)
    [qr | qa] = P_h y_h                                    P_h: (64 + r) x 256 -- the post-norm map R_h, the RoRoPE
                                                           frequency mixing U and the absorbed key up-projections
                                                           (residual + NoPE rows of group g(h), latent rotation folded)
    kr       = TQdec(TQenc(W_r' x)) ; k = rope(R0^T kr)    RoPE key, TurboQuant'd before rotary (R0 folded into W_r')
    c        = TQdec(TQenc(W_d' x))                         latent in its rotated basis (R folded into W_d')
    s_h(t,s) = [rope(qr_t) | qa_t] . [k_s | c_s] * inv_r_g(s) * scale,  inv_r_g = rsqrt(mean(W_k x)_g^2 + eps)
    o_h      = W_v,g' (sum_s softmax(s_h)_s c_s) * sigmoid(gate_h) ; out = W_o [o_1..o_H]
TurboQuant (4-bit, per-vector fp16 norm, Beta Lloyd-Max codebook of the vector's width) is unchanged; only the
rotations moved into the weights (R into the latent down-projection, R^T into the up-projections, R0 into the RoPE
key projection; R0^T stays explicit because rotary sits between it and the query).

    MERCURIUS_ROOT=... PYTHONPATH=<mercurius>/src:src .venv/bin/python deploy/mercurius_gguf.py check
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def dense(lin) -> torch.Tensor:
    """fp32 weight of a Linear / bitsandbytes Linear4bit (the deployed values)."""
    try:
        import bitsandbytes as bnb
        if isinstance(lin, bnb.nn.Linear4bit):
            return bnb.functional.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
    except ImportError:
        pass
    return lin.weight.data.float()


@torch.no_grad()
def fold_attention(sa, rope_nf4: bool = True) -> dict:
    """All fp32 tensors of one Mercurius attention layer in the folded form above. rope_nf4=False skips the RoPE-key
    weight's NF4 rounding (only to compare with the module run without its QAT)."""
    from mercurius.models.qat import nf4_roundtrip
    H, G, D = sa._H, sa._G, sa.head_dim
    rd, nr = sa._rd, sa._n_rope
    half = rd // 2
    assert nr == 1, "the graph assumes one exact RoPE component per frequency (n_rope=1)"
    lat = sa.k_proj.latent
    R = lat._tq_rot.float()                          # (r, r): cache holds codes of R c
    R0 = sa._qat_R0.float()                          # (64, 64)
    r = R.shape[0]
    dev = R.device
    # query side: y (normalized, before the post-norm map) -> [pre-RoPE (64) | absorbed (r)], per head
    Wk = dense(lat.up_k)                              # (nres + G*(D-rd), r)
    nres = (G - nr) * rd
    W_res, W_nope = Wk[:nres], Wk[nres:].view(G, D - rd, r)
    Rq = sa.q_norm.R.float()                          # (H, D, D): q' = R_h^T y
    U = sa._U_heads.float()                           # (H, G, half)
    g_of_h = torch.arange(H, device=dev) // (H // G)
    I = torch.eye(D, device=dev)
    P = torch.empty(H, rd + r, D, device=dev)
    Qx = torch.empty(H, rd + nres + (D - rd), D, device=dev)   # expanded query per head: [rope | shared res | nope]
    for h in range(H):
        q = torch.einsum("bd,de->be", I, Rq[h])       # basis -> post-map query, rows = basis vectors
        qre, qim = q[:, :half], q[:, half:rd]
        qc = torch.cat([qre.unsqueeze(1) * U[h].unsqueeze(0), qim.unsqueeze(1) * U[h].unsqueeze(0)], -1)  # (D, G, rd)
        q_rope = qc[:, 0, :]                          # component 0 keeps RoPE
        q_res = qc[:, 1:, :].reshape(D, nres)
        q_nope = q[:, rd:]
        q_abs = q_res @ W_res + q_nope @ W_nope[g_of_h[h]]          # (D, r) in the latent's original basis
        P[h, :rd] = q_rope.T
        P[h, rd:] = (q_abs @ R.T).T                   # rotated basis: c = R^T c_rot  =>  q.c = (R q).c_rot
        Qx[h] = torch.cat([q_rope, q_res, q_nope], 1).T
    Wv = dense(lat.up_v).view(G, D, r) @ R.T          # (G, D, r) acting on c_rot
    Wr = lat.k_rope.weight.float()
    if rope_nf4:
        Wr = (Wr + (nf4_roundtrip(Wr) - Wr))          # the trained forward's NF4 rounding of the RoPE key weight
    return {
        "wq": dense(sa.q_proj),                       # (H*2D, n_embd): per head [q(D) | gate(D)]
        "q_norm": sa.q_norm.weight.float(),           # Qwen3.5 RMSNorm gain is (1 + w)
        "q_norm_eps": float(sa.q_norm.eps if hasattr(sa.q_norm, "eps") else sa.q_norm.variance_epsilon),
        "q_map": P,                                   # (H, 64 + r, D)
        # expanded (prefill) form: per-group key [rope (64) | W_res c (192) | W_nope_g c (192)], value W_v,g c
        "q_exp": Qx,                                  # (H, 448, D)
        "k_up": Wk @ R.T,                             # (192 + G*192, r) acting on c_rot
        "w_rope": R0 @ Wr,                            # (64, n_embd): cache codes of R0 kt
        "rope_unrot": R0.T.contiguous(),              # kt = R0^T dec
        "w_latent": R @ dense(lat.down),              # (r, n_embd): cache codes of R c
        "w_v": Wv,                                    # (G, D, r)
        "w_krms": dense(sa.k_rms),                    # (G*D, n_embd)
        "k_norm_eps": float(getattr(sa.k_norm, "eps", getattr(sa.k_norm, "variance_epsilon", 1e-6))),
        "wo": dense(sa.o_proj),
        "scale": float(sa.scaling), "H": H, "G": G, "D": D, "rd": rd, "r": r,
    }


def _rope(x, cos, sin):
    """NEOX (rotate_half) rotary over the last dim of x (..., 64); cos/sin (..., 64)."""
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


@torch.no_grad()
def absorbed_attention(f, h, cos, sin, tq: bool = True):
    """Reference of the llama.cpp graph for one layer, fp32. h (T, n_embd); cos/sin (T, 64). Causal, one sequence."""
    from opendecider.mercurius_cache import tq_decode, tq_encode
    T = h.shape[0]
    H, G, D, rd, r = f["H"], f["G"], f["D"], f["rd"], f["r"]
    x = h.float()
    qg = (x @ f["wq"].T).view(T, H, 2 * D)
    q, gate = qg[..., :D], qg[..., D:]
    y = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + f["q_norm_eps"]) * (1 + f["q_norm"])
    qq = torch.einsum("thd,hed->the", y, f["q_map"])                 # (T, H, 64 + r)
    q_rope = _rope(qq[..., :rd], cos[:, None], sin[:, None])
    q_abs = qq[..., rd:]
    kr, c = x @ f["w_rope"].T, x @ f["w_latent"].T
    if tq:
        kr = tq_decode(*tq_encode(kr.to(torch.bfloat16), 4), rd, torch.bfloat16, 4).float()
        c = tq_decode(*tq_encode(c.to(torch.bfloat16), 4), r, torch.bfloat16, 4).float()
    k = _rope(kr @ f["rope_unrot"].T, cos, sin)                      # (T, 64)
    kk = x @ f["w_krms"].T
    inv_r = torch.rsqrt(kk.view(T, G, D).pow(2).mean(-1) + f["k_norm_eps"])     # (T, G)
    s = torch.einsum("the,se->hts", q_rope, k) + torch.einsum("thc,sc->hts", q_abs, c)
    s = s * inv_r.T.repeat_interleave(H // G, 0)[:, None, :] * f["scale"]
    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
    o_lat = torch.einsum("hts,sc->thc", s.softmax(-1), c)             # (T, H, r)
    o = torch.einsum("thc,hdc->thd", o_lat, f["w_v"].repeat_interleave(H // G, 0))
    o = (o * torch.sigmoid(gate)).reshape(T, H * D)
    return o @ f["wo"].T


@torch.no_grad()
def check(root: str):
    """Folded / absorbed reference vs the deployed layer, on real hidden states of a text."""
    from opendecider.guards import gpu_lock
    from opendecider.mercurius_port import load_deployed
    with gpu_lock("mercurius_gguf_check"):
        trunk, tok = load_deployed(root, compressed_cache=False)
        ids = tok("State:\nThe customer was charged twice for the March invoice and asks for a refund. They have "
                  "been a customer for six years.\nQuestion: Which team should handle it?", return_tensors="pt")["input_ids"].cuda()
        T = ids.shape[1]
        layers = [(i, l) for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
        cap = {}
        hooks = [l.self_attn.register_forward_pre_hook(
            lambda m, a, kw, i=i: cap.__setitem__(i, (kw.get("hidden_states", a[0] if a else None),
                                                     kw["position_embeddings"])), with_kwargs=True)
                 for i, l in layers]
        outs = {}
        hooks += [l.self_attn.register_forward_hook(lambda m, a, kw, o, i=i: outs.__setitem__(i, o[0]), with_kwargs=True)
                  for i, l in layers]
        trunk(input_ids=ids)
        for hk in hooks:
            hk.remove()
        rel = lambda a, b: ((a.float() - b.float()).norm() / a.float().norm()).item()
        for i, l in layers:
            sa = l.self_attn
            f = fold_attention(sa)
            h, (cos, sin) = cap[i]
            ref = absorbed_attention(f, h[0], cos[0].float(), sin[0].float(), tq=True)
            # exact-math check: the module itself with its KV rounding switched off, vs the reference without TQ
            lat = sa.k_proj.latent
            fw, qat = lat.forward, sa._qat
            lat.forward = lambda x, _l=lat: (_l.up_k(_l.down(x)), _l.up_v(_l.down(x)))
            sa._qat = None
            raw = sa(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)[0][0]
            lat.forward, sa._qat = fw, qat
            ref_raw = absorbed_attention(fold_attention(sa, rope_nf4=False), h[0], cos[0].float(), sin[0].float(), tq=False)
            print(f"layer {i:2d} r={f['r']}: folded vs module, no KV rounding {rel(raw, ref_raw):.2e} | "
                  f"with TurboQuant {rel(outs[i][0], ref):.2e}", flush=True)


# ---------------------------------------------------------------------------------------------- exact NF4 export
class NF4:
    """A bitsandbytes NF4 Linear4bit to be written as GGML_TYPE_NF4 blocks: its codes and dequantized block scales
    exactly as stored (no re-quantization), rows / columns optionally permuted in whole 64-value blocks."""

    def __init__(self, lin, row_perm=None, col_perm=None):
        self.lin, self.row_perm, self.col_perm = lin, row_perm, col_perm

    def blocks(self):
        import numpy as np
        import bitsandbytes.functional as bnbf
        w = self.lin.weight
        st = w.quant_state
        out, inn = st.shape
        assert st.blocksize == 64 and inn % 64 == 0
        q = w.data.reshape(-1).cpu().numpy()                                # bitsandbytes: element 2i in the high nibble
        codes = np.stack([q >> 4, q & 15], -1).reshape(out, inn)
        scale = (bnbf.dequantize_blockwise(st.absmax, st.state2) + st.offset).float().cpu().numpy().reshape(out, inn // 64)
        if self.row_perm is not None:
            rp = self.row_perm.cpu().numpy()
            codes, scale = codes[rp], scale[rp]
        if self.col_perm is not None:
            cp = self.col_perm.cpu().numpy()
            bp = cp.reshape(-1, 64)
            assert (bp[:, 0] % 64 == 0).all() and (bp == bp[:, :1] + np.arange(64)).all(), "column permutation splits blocks"
            codes, scale = codes[:, cp], scale[:, bp[:, 0] // 64]
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8).reshape(out, inn // 64, 32)
        rec = np.empty((out, inn // 64), dtype=[("d", "<f4"), ("qs", "u1", (32,))])
        rec["d"], rec["qs"] = scale, packed
        return rec.view(np.uint8).reshape(out, (inn // 64) * 36), (out, inn)


def _is_nf4(lin):
    try:
        import bitsandbytes as bnb
        return isinstance(lin, bnb.nn.Linear4bit)
    except ImportError:
        return False


# ---------------------------------------------------------------------------------------------- GGUF writer
QWEN_GGUF = os.path.expanduser("~/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_0.gguf")    # same tokenizer as Qwen3.5-4B


def _v_perm(n_k: int, n_v: int, head_dim: int) -> torch.Tensor:
    """Row permutation from grouped value heads (by key head) to tiled order, as the Qwen3.5 converter does
    (_LinearAttentionVReorderBase): ggml broadcasts keys over value heads in tiled order."""
    idx = torch.arange(n_v * head_dim).view(n_k, n_v // n_k, head_dim)
    return idx.permute(1, 0, 2).reshape(-1)


def _head_perm(n_k: int, n_v: int) -> torch.Tensor:
    return torch.arange(n_v).view(n_k, n_v // n_k).T.reshape(-1)


@torch.no_grad()
def linear_layer_tensors(la, cfg, nf4: bool = False) -> dict:
    """GDN-2 layer: projections, conv, decay (-exp(A_log), dt_bias), factored gates (tiled base + VeRA vectors,
    decay LoRA), gated norm, out-projection. Value-head-indexed rows/cols reordered to tiled order."""
    from opendecider.mercurius_port import TiledLinear
    nk, nv, dk, dv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], \
        cfg["linear_value_head_dim"]
    pv, pk = _v_perm(nk, nv, dv), _v_perm(nk, nv, dk)          # value-axis / key-channel-axis (per value head)
    ph = _head_perm(nk, nv)
    kd = nk * dk
    qkv_perm = torch.cat([torch.arange(2 * kd), 2 * kd + pv])
    conv = la.conv1d.weight.data.float().squeeze(1)
    conv = torch.cat([conv[:2 * kd], conv[2 * kd:][pv]])
    pick = lambda lin, rp=None, cp=None: NF4(lin, rp, cp) if nf4 and _is_nf4(lin) else \
        (dense(lin)[rp] if rp is not None else dense(lin))[:, cp] if cp is not None else \
        (dense(lin)[rp] if rp is not None else dense(lin))
    out = {"attn_qkv": pick(la.in_proj_qkv, qkv_perm), "attn_gate": pick(la.in_proj_z, pv), "ssm_conv1d": conv,
           "ssm_a": (-la.A_log.float().exp())[ph].reshape(-1), "ssm_dt": la.dt_bias.float()[ph].reshape(-1),
           "ssm_norm": la.norm.weight.float(), "ssm_out": pick(la.out_proj, None, pv),
           "ssm_a_lora_a": la.a_lora_A.float(), "ssm_a_lora_b": la.a_lora_B.float()[pk]}
    # the three gates (decay a, erase be, write bw) stacked, so the graph runs each factor once per layer:
    #   gate_base (3 n_v, n_embd)  rows g*n_v + h      gate_vd (3, vera rank)      gate_vb (3, n_v * d_k)
    base, vd, vb = [], [], []
    for g, perm in (("a", pk), ("be", pk), ("bw", pv)):
        m = getattr(la, f"in_proj_{g}")
        assert isinstance(m.base, TiledLinear), f"in_proj_{g}: expected a tiled base"
        U = m.base.unique.float()                              # (n_v heads, n_embd), each repeated rep times
        assert U.shape[0] == nv and m.base.rep * nv == m.base.out_features
        base.append(U[ph]); vd.append(m.vera_d.float()); vb.append(m.vera_b.float()[perm])
    out["gate_base"], out["gate_vd"], out["gate_vb"] = torch.cat(base), torch.stack(vd), torch.stack(vb)
    return out


def _shared_vera(la, cfg):
    """The VeRA factors every gate shares: A[:, :n_embd] and B[:n_out] (rows reordered like the gate outputs)."""
    nk, nv, dk = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"]
    m = la.in_proj_be
    O = m.base.out_features
    assert O == nv * dk == la.in_proj_bw.base.out_features == la.in_proj_a.base.out_features
    return m.vera_A[:, :m.base.in_features].float(), m.vera_B[:O].float()[_v_perm(nk, nv, dk)]


def _norm_vec(norm, n):
    """ScaleNorm (one scalar g: y = g x / rms) or zero-centred Qwen3.5 RMSNorm (1 + w) -> llama.cpp RMS weight."""
    w = norm.weight.data.float()
    return w.expand(n).clone() if w.numel() == 1 else (1.0 + w)


@torch.no_grad()
def convert(root: str, out_path: str, nf4: bool = False):
    """Deployed Mercurius -> GGUF. nf4=False: F16 big matrices, F32 small ones (quantize afterwards with llama-quantize).
    nf4=True: every 4-bit projection and the embedding as exact GGML_TYPE_NF4 (the deployed weights bit for bit),
    the folded attention matrices, gates and VeRA factors F16."""
    sys.path.insert(0, "/mnt/ext/opendecider/llama-tq/tom/gguf-py")
    import json
    import gguf
    import numpy as np
    from mercurius.models.qat import tq_codebook
    from opendecider.mercurius_port import load_deployed
    trunk, tok = load_deployed(root, compressed_cache=False)
    hf = json.load(open(os.path.join(root, "models", "qwen3.5-4b", "config.json")))
    cfg = hf.get("text_config", hf)
    n_embd, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    w = gguf.GGUFWriter(out_path, "mercurius")

    def put(name, t, half_big=True):
        if isinstance(t, NF4):
            data, shape = t.blocks()
            w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.NF4)    # byte shape; gguf-py derives (out, in)
            return
        big = t.dim() >= 2 and t.numel() >= 1 << 20
        keep32 = name.endswith(("ssm_conv1d.weight",                  # ggml_ssm_conv reads an F32 kernel
                                "gate_vd.weight", "gate_vb.weight"))  # VeRA scaling vectors: fp32 as trained
        w.add_tensor(name, (t.half() if not keep32 and ((big and half_big) or (nf4 and t.dim() >= 2)) else t).cpu().numpy())
    A = "mercurius"
    w.add_name("Mercurius-1-4B")
    w.add_uint32(f"{A}.block_count", L)
    w.add_uint32(f"{A}.context_length", 262144)
    w.add_uint32(f"{A}.embedding_length", n_embd)
    w.add_uint32(f"{A}.feed_forward_length", cfg["intermediate_size"])
    w.add_uint32(f"{A}.attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(f"{A}.attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_uint32(f"{A}.attention.key_length", cfg["head_dim"])
    w.add_uint32(f"{A}.attention.value_length", cfg["head_dim"])
    w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_uint32(f"{A}.full_attention_interval", cfg["full_attention_interval"])
    w.add_float32(f"{A}.rope.freq_base", float(cfg["rope_parameters"]["rope_theta"]))
    w.add_uint32(f"{A}.rope.dimension_count", int(cfg["head_dim"] * cfg["rope_parameters"]["partial_rotary_factor"]))
    w.add_uint32(f"{A}.ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
    w.add_uint32(f"{A}.ssm.state_size", cfg["linear_key_head_dim"])
    w.add_uint32(f"{A}.ssm.group_count", cfg["linear_num_key_heads"])
    w.add_uint32(f"{A}.ssm.time_step_rank", cfg["linear_num_value_heads"])
    w.add_uint32(f"{A}.ssm.inner_size", cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"])
    rd_ = tq = None
    latent_ranks, words = [], []
    for il, layer in enumerate(trunk.layers):
        pre = f"blk.{il}."
        w.add_tensor(pre + "attn_norm.weight", _norm_vec(layer.input_layernorm, n_embd).cpu().numpy())
        w.add_tensor(pre + "post_attention_norm.weight", _norm_vec(layer.post_attention_layernorm, n_embd).cpu().numpy())
        for k, m in (("ffn_gate", layer.mlp.gate_proj), ("ffn_up", layer.mlp.up_proj), ("ffn_down", layer.mlp.down_proj)):
            put(pre + k + ".weight", NF4(m) if nf4 and _is_nf4(m) else dense(m))
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            if tq is None:
                tq = _shared_vera(la, cfg)
                w.add_tensor("vera_a.weight", tq[0].half().cpu().numpy())
                w.add_tensor("vera_b.weight", tq[1].half().cpu().numpy())
            for k, t in linear_layer_tensors(la, cfg, nf4).items():
                name = {"ssm_a": pre + "ssm_a", "ssm_dt": pre + "ssm_dt.bias"}.get(k, pre + k + ".weight")
                put(name, t)
            latent_ranks.append(0); words.append(0)
        else:
            f = fold_attention(layer.self_attn)
            r, rd, G = f["r"], f["rd"], f["G"]
            rd_ = rd
            sa = layer.self_attn
            for k, lin in (("wq", sa.q_proj), ("wo", sa.o_proj), ("w_krms", sa.k_rms)):
                name = pre + {"wq": "attn_q", "wo": "attn_output", "w_krms": "attn_k_rms"}[k] + ".weight"
                put(name, NF4(lin) if nf4 and _is_nf4(lin) else f[k])
            w.add_tensor(pre + "attn_q_norm.weight", (1.0 + f["q_norm"]).cpu().numpy())
            put(pre + "attn_q_map.weight", f["q_map"], half_big=False)        # F32 (H, 64 + r, D); F16 with --nf4
            for k, n in (("w_rope", "attn_k_rope"), ("rope_unrot", "attn_rope_unrot"), ("w_latent", "attn_latent"),
                         ("w_v", "attn_v_up"), ("q_exp", "attn_q_exp"), ("k_up", "attn_k_up")):
                put(pre + n + ".weight", f[k], half_big=False)                  # folded: F32, or F16 with --nf4
            for n, dim in (("tq_latent_cb", r), ("tq_rope_cb", rd)):
                cen, bnd = tq_codebook(dim, 4)
                w.add_tensor(pre + n + ".weight", torch.cat([cen, bnd]).float().cpu().numpy())   # 16 centroids + 15 bounds
            latent_ranks.append(r)
            nbytes = 4 + 4 * G + 2 + 2 + rd // 2 + (r + 1) // 2      # pos, inv_r, 2 norms, rope codes, latent codes
            words.append((nbytes + 3) // 4)
    w.add_array(f"{A}.attention.latent_ranks", latent_ranks)
    w.add_array(f"{A}.attention.cache_words", words)
    w.add_float32(f"{A}.attention.k_norm_epsilon", f["k_norm_eps"])
    w.add_float32(f"{A}.attention.scale", f["scale"])
    w.add_uint32(f"{A}.vera.rank", int(tq[0].shape[0]))
    E = trunk.embed_tokens
    if nf4:                                   # NF4Table: bitsandbytes codes (high nibble first) + fp32 block scales
        import numpy as np
        V, D = E.num_embeddings, E.embedding_dim
        q = E.codes.cpu().numpy()
        codes = np.stack([q >> 4, q & 15], -1).reshape(V, D)
        rec = np.empty((V, D // 64), dtype=[("d", "<f4"), ("qs", "u1", (32,))])
        rec["d"] = E.absmax.cpu().numpy()
        rec["qs"] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).reshape(V, D // 64, 32)
        w.add_tensor("token_embd.weight", rec.view(np.uint8).reshape(V, (D // 64) * 36),
                     raw_dtype=gguf.GGMLQuantizationType.NF4)
    else:
        w.add_tensor("token_embd.weight", E.dequant_rows(slice(None), torch.float16).cpu().numpy())
    w.add_tensor("output_norm.weight", _norm_vec(trunk.norm, n_embd).cpu().numpy())
    r = gguf.GGUFReader(QWEN_GGUF)                                                         # tokenizer (identical)
    for k, fld in r.fields.items():
        if k.startswith("tokenizer."):
            v = fld.contents()
            w.add_key_value(k, v, fld.types[0], sub_type=fld.types[-1] if fld.types[0] == gguf.GGUFValueType.ARRAY else None)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        check(os.environ["MERCURIUS_ROOT"])
    elif sys.argv[1:2] == ["convert"]:
        convert(os.environ["MERCURIUS_ROOT"], sys.argv[2], nf4="--nf4" in sys.argv)
