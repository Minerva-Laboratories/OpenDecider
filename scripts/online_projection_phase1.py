"""Phase 1 of docs/online_projection_spec.md: offline attention fidelity on real activations.

Qwen3.5 attention layers rotate only the first 64 of 256 head dims (partial RoPE); the other 192 are position-free.
We capture normalized pre-RoPE queries/keys and values for every attention layer on a long text, then compress the
192 position-free key dims (and, separately, the values) to rank r with:
  static   PCA basis fitted on a different text (the global, MLA-like projection)
  oracle   PCA of this sequence's own keys, non-causal (upper bound for any per-sequence basis)
  online   chunked subspace iteration (spec §3): codes written with the basis of their chunk, read with that basis
  online-1 the same codes read with the reader's current basis only (no per-chunk correction: drift error)
  qw       online, query-weighted (spec §2): keys whitened by the running query covariance
and measure exact causal attention against the uncompressed computation: relative output error and top-k overlap.
    PYTHONPATH=src .venv/bin/python scripts/online_projection_phase1.py --model /mnt/ext/.../Qwen__Qwen3.5-2B@... """
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opendecider.guards import gpu_lock

ROT = 64           # rotary dims (Qwen3.5: partial_rotary_factor 0.25 of head_dim 256)
THETA = 1e7


def capture(model, tok, text, T):
    """Per attention layer: q (T, Hq, 256), k (T, Hk, 256), v (T, Hk, 256), normalized, pre-RoPE, fp32 on GPU."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :T].cuda()
    trunk = model.model.language_model if hasattr(model.model, "language_model") else model.model
    caps, hooks = {}, []
    for i, layer in enumerate(trunk.layers):
        sa = getattr(layer, "self_attn", None)
        if sa is None:
            continue
        caps[i] = {}
        hooks.append(sa.q_norm.register_forward_hook(lambda m, a, o, i=i: caps[i].__setitem__("q", o[0].float())))
        hooks.append(sa.k_norm.register_forward_hook(lambda m, a, o, i=i: caps[i].__setitem__("k", o[0].float())))
        hooks.append(sa.v_proj.register_forward_hook(
            lambda m, a, o, i=i, sa=sa: caps[i].__setitem__("v", o[0].float().view(o.shape[1], -1, sa.head_dim))))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    return caps, ids.shape[1]


def rope(x):
    """NeoX RoPE on the first ROT dims of (T, H, 256), positions 0..T-1."""
    T = x.shape[0]
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, device=x.device, dtype=torch.float32) / ROT))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1)[:, None], torch.cat([ang.sin(), ang.sin()], -1)[:, None]
    r = x[..., :ROT]
    rh = torch.cat([-r[..., ROT // 2:], r[..., :ROT // 2]], -1)
    return torch.cat([r * cos + rh * sin, x[..., ROT:]], -1)


def attention(q, k, v, scale, qb=1024):
    """Causal softmax attention for one KV group: q (T, n, D) with keys/values (T, D). Returns out (T, n, Dv), and the
    top-32 key indices per (query, head) for the overlap metric."""
    T, n, _ = q.shape
    out = torch.empty(T, n, v.shape[-1], device=q.device)
    top = torch.empty(T, n, 32, dtype=torch.long, device=q.device)
    for a in range(0, T, qb):
        b = min(T, a + qb)
        s = torch.einsum("tnd,jd->tnj", q[a:b], k[:b]) * scale
        mask = torch.arange(b, device=q.device)[None, :] > torch.arange(a, b, device=q.device)[:, None]
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        p = torch.softmax(s, -1)
        out[a:b] = torch.einsum("tnj,jd->tnd", p, v[:b])
        top[a:b] = s.topk(min(32, b), -1).indices if b >= 32 else s.topk(b, -1).indices
    return out, top


def pca_basis(X, r):
    """Top-r principal directions (uncentered second moment) of X (N, d) -> (r, d)."""
    C = X.T @ X / X.shape[0]
    w, U = torch.linalg.eigh(C)
    return U[:, -r:].flip(-1).T.contiguous()


def online_bases(X, r, C=64, gamma=0.9, eta=None, whiten=None):
    """Spec §3: per chunk b, the basis E_b used to write chunk b's codes (chunk 0 = warm-up: exact, basis None).
    gamma: EMA of the second moment per chunk; eta=None means a full subspace-iteration step E <- orth(E M).
    whiten: optional (T, d, d) running whitening matrices (QW variant), applied to keys before the moment."""
    T, d = X.shape
    M = torch.zeros(d, d, device=X.device)
    bases, E = [None], None
    for b0 in range(0, T, C):
        Xb = X[b0:b0 + C]
        M = gamma * M + (1 - gamma) * (Xb.T @ Xb) / Xb.shape[0]
        if E is None:
            w, U = torch.linalg.eigh(M)
            E = U[:, -r:].flip(-1).T.contiguous()
        else:
            step = M if eta is None else torch.eye(d, device=X.device) + eta * M / M.diagonal().sum()
            Q, _ = torch.linalg.qr((E @ step).T)
            E = Q.T.contiguous()
        bases.append(E.clone())                  # basis for the NEXT chunk (causal: built from chunks <= b)
    return bases[:-1]


def project_chunks(X, bases, C=64):
    """Codes written per chunk and read with the same basis: X_hat = E_b^T E_b x for chunk b (chunk 0 exact)."""
    out = X.clone()
    for b, E in enumerate(bases):
        if E is not None:
            Xb = X[b * C:(b + 1) * C]
            out[b * C:(b + 1) * C] = (Xb @ E.T) @ E
    return out


def attention_drift(q, k_nope_in, k_rope, v, bases, scale, C=64, qb=1024):
    """online-1: keys coded with their chunk's basis, read with the QUERY's current basis only:
    score_nope = (E_t q)^T c_j with c_j = E_b(j) k_j  ->  k_eff = E_t^T E_b(j) k_j."""
    T, n, _ = q.shape
    out = torch.empty(T, n, v.shape[-1], device=q.device)
    codes = torch.zeros(T, bases[1].shape[0] if len(bases) > 1 else 1, device=q.device)
    exact0 = k_nope_in[:C]
    for b, E in enumerate(bases):
        if E is not None:
            codes[b * C:(b + 1) * C] = k_nope_in[b * C:(b + 1) * C] @ E.T
    for a in range(0, T, C):
        b = min(T, a + C)
        bq = a // C
        E_t = bases[bq] if bases[bq] is not None else None
        if E_t is None:
            k_eff = k_nope_in[:b]
        else:
            k_eff = torch.cat([exact0, codes[C:b] @ E_t], 0) if b > C else exact0[:b]
        kk = torch.cat([k_rope[:b], k_eff], -1)
        s = torch.einsum("tnd,jd->tnj", q[a:b], kk) * scale
        mask = torch.arange(b, device=q.device)[None, :] > torch.arange(a, b, device=q.device)[:, None]
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        out[a:b] = torch.einsum("tnj,jd->tnd", torch.softmax(s, -1), v[:b])
    return out


def attention_remap(q, k_hat, k_rope, v, bases, scale, C=64):
    """Option (b): keep only the current basis E_t plus an r x r map per old chunk. Reading chunk b's codes through
    Q_b = E_t E_b^T gives E_t^T E_t (E_b^T c_j): the write-time reconstruction projected onto the current subspace."""
    T, n, _ = q.shape
    out = torch.empty(T, n, v.shape[-1], device=q.device)
    for a in range(0, T, C):
        b = min(T, a + C)
        E_t = bases[a // C]
        if E_t is None or b <= C:
            k_eff = k_hat[:b]
        else:
            k_eff = torch.cat([k_hat[:C], (k_hat[C:b] @ E_t.T) @ E_t], 0)   # chunk 0 (warm-up) stays exact
        kk = torch.cat([k_rope[:b], k_eff], -1)
        s = torch.einsum("tnd,jd->tnj", q[a:b], kk) * scale
        mask = torch.arange(b, device=q.device)[None, :] > torch.arange(a, b, device=q.device)[:, None]
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        out[a:b] = torch.einsum("tnj,jd->tnd", torch.softmax(s, -1), v[:b])
    return out


def metrics(o_hat, o_ref, top_hat=None, top_ref=None, skip=64):
    rel = ((o_hat - o_ref).norm(dim=-1) / o_ref.norm(dim=-1).clamp_min(1e-6))[skip:].mean().item()
    res = {"rel_err": round(rel, 4)}
    if top_hat is not None:
        th, tr = top_hat[skip:], top_ref[skip:]
        ov = (th.unsqueeze(-1) == tr.unsqueeze(-2)).any(-1).float().mean().item()
        res["top32_overlap"] = round(ov, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ext/opendecider/models/Qwen__Qwen3.5-2B@15852e8c1636")
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--ranks_k", type=int, nargs="+", default=[16, 32, 64, 96])
    ap.add_argument("--ranks_v", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--keys_only", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--out", default="runs/online_projection_phase1.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = os.path.join(os.path.dirname(__file__), "..")
    rd = lambda *p: "".join(open(os.path.join(root, x)).read() for x in p)
    eval_text = rd("docs/RESULTS.md", "docs/LANDSCAPE.md", "README.md")       # prose + tables
    calib_text = rd("src/opendecider/v3.py", "src/opendecider/backbone.py", "src/opendecider/api.py",
                    "src/opendecider/train.py")                               # code: a different distribution
    with gpu_lock("online_projection_phase1"):
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
        ev, T = capture(model, tok, eval_text, a.T)
        cal, Tc = capture(model, tok, calib_text, a.T)
        del model
        torch.cuda.empty_cache()
        print(f"captured {len(ev)} attention layers, T={T} (eval), {Tc} (calibration)", flush=True)
        results = {"T": T, "T_calib": Tc, "gamma": a.gamma, "layers": {}}
        t0 = time.time()
        for li in sorted(ev):
            q, k, v = ev[li]["q"], ev[li]["k"], ev[li]["v"]
            Hq, Hk, D = q.shape[1], k.shape[1], k.shape[2]
            rep, scale = Hq // Hk, 1.0 / math.sqrt(D)
            qr, kr = rope(q), rope(k)
            lay = {}
            for g in range(Hk):
                qg = qr[:, g * rep:(g + 1) * rep]                               # (T, rep, D)
                k_rope, k_nope = kr[:, g, :ROT], kr[:, g, ROT:]
                vg = v[:, g]
                o_ref, top_ref = attention(qg, kr[:, g], vg, scale)
                kc = cal[li]["k"][:, g, ROT:]
                vc = cal[li]["v"][:, g]
                # query covariance for QW (position-free part, all heads of the group), running per chunk
                qn = q[:, g * rep:(g + 1) * rep, ROT:].reshape(T, rep, -1)
                for r in a.ranks_k:
                    res = {}
                    variants = {
                        "static": k_nope @ pca_basis(kc, r).T @ pca_basis(kc, r),
                        "oracle": k_nope @ pca_basis(k_nope, r).T @ pca_basis(k_nope, r),
                    }
                    bases = online_bases(k_nope, r, gamma=a.gamma)
                    variants["online"] = project_chunks(k_nope, bases)
                    # QW: whiten keys by the running query second moment A = Sq^{1/2}; oblique projection
                    # k_hat = A^{-1} U U^T A k with U from online subspace iteration on A k (per chunk A)
                    Ck = 64
                    k_qw = k_nope.clone()
                    Mq = torch.zeros(D - ROT, D - ROT, device=q.device)
                    Mk = torch.zeros_like(Mq)
                    E = None
                    for b0 in range(0, T, Ck):
                        qb_ = qn[b0:b0 + Ck].reshape(-1, D - ROT)
                        Mq = a.gamma * Mq + (1 - a.gamma) * qb_.T @ qb_ / qb_.shape[0]
                        w, U = torch.linalg.eigh(Mq + 1e-4 * Mq.diagonal().mean() * torch.eye(D - ROT, device=q.device))
                        A = (U * w.clamp_min(1e-8).sqrt()) @ U.T
                        Ai = (U * w.clamp_min(1e-8).rsqrt()) @ U.T
                        if E is not None:                                      # write this chunk with the basis so far
                            k_qw[b0:b0 + Ck] = ((k_nope[b0:b0 + Ck] @ A_prev) @ E.T @ E) @ Ai_prev
                        kb = k_nope[b0:b0 + Ck] @ A
                        Mk = a.gamma * Mk + (1 - a.gamma) * kb.T @ kb / kb.shape[0]
                        if E is None:
                            w2, U2 = torch.linalg.eigh(Mk)
                            E = U2[:, -r:].flip(-1).T.contiguous()
                        else:
                            Qm, _ = torch.linalg.qr((E @ Mk).T)
                            E = Qm.T.contiguous()
                        A_prev, Ai_prev = A, Ai
                    variants["qw"] = k_qw
                    for name, kn in variants.items():
                        o, top = attention(qg, torch.cat([k_rope, kn], -1), vg, scale)
                        res[name] = metrics(o, o_ref, top, top_ref)
                    o = attention_drift(qg, k_nope, k_rope, vg, bases, scale)
                    res["online-1"] = metrics(o, o_ref)
                    o = attention_remap(qg, variants["online"], k_rope, vg, bases, scale)
                    res["online-remap"] = metrics(o, o_ref)
                    lay[f"g{g}_k_r{r}"] = res
                for r in ([] if a.keys_only else a.ranks_v):
                    res = {}
                    vb = online_bases(vg, r, gamma=a.gamma)
                    for name, vv in {"static": vg @ pca_basis(vc, r).T @ pca_basis(vc, r),
                                     "oracle": vg @ pca_basis(vg, r).T @ pca_basis(vg, r),
                                     "online": project_chunks(vg, vb)}.items():
                        o, _ = attention(qg, kr[:, g], vv, scale)
                        res[name] = metrics(o, o_ref)
                    lay[f"g{g}_v_r{r}"] = res
                print(f"layer {li} group {g} done ({time.time() - t0:.0f}s)", flush=True)
            results["layers"][li] = lay
            json.dump(results, open(a.out, "w"), indent=1)
        # summary: mean over layers and groups
        summ = {}
        for li, lay in results["layers"].items():
            for key, res in lay.items():
                kind = key.split("_", 1)[1]
                for name, m in res.items():
                    summ.setdefault(kind, {}).setdefault(name, []).append(m["rel_err"])
        results["summary_rel_err"] = {kind: {n: round(sum(v) / len(v), 4) for n, v in d.items()} for kind, d in summ.items()}
        json.dump(results, open(a.out, "w"), indent=1)
        for kind, d in results["summary_rel_err"].items():
            print(kind, d, flush=True)


if __name__ == "__main__":
    main()
