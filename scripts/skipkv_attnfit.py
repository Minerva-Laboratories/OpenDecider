"""Fine-tune the attention layers' skip maps on what attention reads (a local, SwiftKV-style fit; frozen model).

The closed-form maps (skipkv.fit_skip_maps) minimize the error of each upper layer's input uniformly over directions
and tokens. The generation tests (RESULTS 21.2) put most of the damage in the full-attention layers' caches. Here each
full-attention layer j above the cut gets its map W_j trained so that queries of LATER tokens (exact, as at deployment:
tokens after the prompt run every layer) get the same attention output from the predicted prompt keys / values as from
the exact ones:
    loss_j = || attn(Q_q, [K(x^_P); K(x_q)], [V(x^_P); V(x_q)]) - attn(Q_q, K(x), V(x)) ||^2 / || attn(Q_q, K(x), V(x)) ||^2
x^_P = map_j(cut input) at prompt positions, x_q the exact inputs of the query positions. Only W_j, b_j train
(initialized from the closed-form map, L2 pull toward it); the targets come from the model's own exact pass.
Mercurius MLA: keys and values through its float path (latent down / up, decoupled RoPE key, per-group 1/rms; the
deployed TurboQuant 4-bit codes encode the same latent and RoPE key). Data: FineWeb-Edu documents (the maps' own
calibration documents), --T tokens, the last --q of them as queries.

    python scripts/skipkv_attnfit.py --ckpt runs/x4b-mercurius-lora/model.pt --cut 16
"""
import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps, _merc, _rms  # noqa: E402
from skipkv_lm import DATA  # noqa: E402


def merc_kv(sa, x, cos, sin):
    """Mercurius MLA keys / values in float (mercurius_cache._encode_codes + _expand_kv without the 4-bit codes)."""
    from mercurius.surgery.mla_rope import _rope_blocks
    B, T = x.shape[:2]
    G, D, rd, nr = sa._G, sa.head_dim, sa._rd, sa._n_rope
    lat = sa.k_proj.latent
    kraw = sa.k_rms(x).view(B, T, G, D).float()
    inv_r = torch.rsqrt(kraw.pow(2).mean(-1) + getattr(sa.k_norm, "eps", 1e-6))
    kt = F.linear(x, sa.__dict__["_wr_nf4"].to(x.dtype))
    cq = lat.down(x)
    kres, v = lat.up_k(cq), lat.up_v(cq)
    nres = (G - nr) * rd
    k_res, k_nope = kres[..., :nres], kres[..., nres:].view(B, T, G, D - rd)
    k_rope = _rope_blocks(kt.view(B, T, 1, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
    K = torch.cat([k_rope.expand(B, T, G, nr * rd), k_res.view(B, T, 1, nres).expand(B, T, G, nres), k_nope], -1)
    return (K.float() * inv_r.unsqueeze(-1)).transpose(1, 2), v.view(B, T, G, D).transpose(1, 2).float()


def merc_q(sa, x, pe):
    import opendecider.mercurius_cache as mc
    B, T = x.shape[:2]
    q_raw = torch.chunk(sa.q_proj(x).view(B, T, sa._H, 2 * sa.head_dim), 2, dim=-1)[0]
    return mc._expand_q(sa, q_raw, pe).float()


def attend(sa, Q, K, V, q0):
    """Queries at positions q0.. over keys 0..; causal."""
    Tq, S = Q.shape[-2], K.shape[-2]
    m = torch.arange(S, device=Q.device)[None] <= torch.arange(q0, q0 + Tq, device=Q.device)[:, None]
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=m, scale=sa.scaling, enable_gqa=True)


@torch.no_grad()
def capture(bb, ids, cut, layers):
    """Exact pass: the cut's input, and for each attention layer its (normed) input, position embeddings, positions."""
    st, hs = {}, []
    hs.append(bb.lm.layers[cut].register_forward_pre_hook(
        lambda m, a, k: st.__setitem__("h", (a[0] if a else k["hidden_states"]).detach()), with_kwargs=True))
    for j in layers:
        def pre(m, a, k, j=j):
            st[j] = ((a[0] if a else k["hidden_states"]).detach(), tuple(t.detach() for t in k["position_embeddings"]),
                     k.get("position_ids"))
        hs.append(bb.lm.layers[j].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        x = torch.tensor([ids], device=bb.device)
        bb.forward(x, torch.ones_like(x, dtype=torch.bool), ("final",))
    finally:
        for h in hs:
            h.remove()
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cut", type=int, default=16)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--T", type=int, default=768)
    ap.add_argument("--q", type=int, default=192)
    ap.add_argument("--n-docs", type=int, default=120)
    ap.add_argument("--n-held", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pull", type=float, default=1e-3, help="L2 pull of W toward the closed-form map (relative)")
    a = ap.parse_args()
    with gpu_lock("skipkv_attnfit"):
        model, _ = load_model(a.ckpt)
        bb = model.backbone
        name = os.path.basename(os.path.dirname(a.ckpt))
        maps = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut{a.cut}_normed.pt"), bb.device)
        assert maps.mode == "normed" and not maps.alphas
        L = len(bb.lm.layers)
        layers = [j for j in range(a.cut, L) if hasattr(bb.lm.layers[j], "self_attn")]
        sa0 = bb.lm.layers[layers[0]].self_attn
        assert _merc(sa0, "mercurius_cache", "_compressed_forward"), "Mercurius MLA only for now"
        enc = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
        docs = []
        for t in pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[:3000]:
            ids = enc(t)
            if len(ids) >= a.T:
                docs.append(ids[: a.T])
            if len(docs) == a.n_docs + a.n_held:
                break
        P = a.T - a.q
        t0 = time.time()
        caps = []
        for ids in docs:                                              # exact pass once per document, kept on the CPU
            st = capture(bb, ids, a.cut, layers)
            caps.append({"h": st["h"].cpu(), **{j: (st[j][0].cpu(), tuple(t.cpu() for t in st[j][1]), st[j][2])
                                                for j in layers}})
        print(f"captured {len(caps)} documents in {time.time() - t0:.0f} s", flush=True)
        tr, he = caps[: a.n_docs], caps[a.n_docs:]
        out = {"layers": layers, "per_layer": {}}
        newW, newb = maps.W.clone(), maps.b.clone()
        dev = bb.device
        for j in layers:
            sa = bb.lm.layers[j].self_attn
            i = j - a.cut
            W0, b0 = maps.W[i].float(), maps.b[i].float()
            W = W0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            opt = torch.optim.Adam([W, b], lr=a.lr)
            scale = float(W0.pow(2).mean())

            def loss_of(c, train):
                h = c["h"].to(dev)
                x, pe, pos = c[j]
                x, pe = x.to(dev), tuple(t.to(dev) for t in pe)
                B, T = x.shape[:2]
                p = (pos if pos is not None else torch.arange(T, device=dev)[None]).long().to(dev)
                cos_s, sin_s = sa._rotary(x, p.expand(B, T)[None].expand(3, B, T))
                with torch.no_grad():
                    K, V = merc_kv(sa, x, cos_s, sin_s)
                    Q = merc_q(sa, x[:, P:], tuple(t[:, P:] for t in pe))
                    target = attend(sa, Q, K, V, P)
                with torch.set_grad_enabled(train):
                    xh = (_rms(h[:, :P]) @ W + b).to(x.dtype)
                    Kp, Vp = merc_kv(sa, xh, cos_s[:, :P], sin_s[:, :P])
                    o = attend(sa, Q, torch.cat([Kp, K[:, :, P:]], 2), torch.cat([Vp, V[:, :, P:]], 2), P)
                    return (o - target).pow(2).sum() / target.pow(2).sum()

            def held_loss():
                with torch.no_grad():
                    return sum(float(loss_of(c, False)) for c in he) / len(he)
            l0 = held_loss()
            best = (l0, W0.clone(), b0.clone(), 0)
            g = torch.Generator().manual_seed(j)
            for step in range(1, a.steps + 1):
                c = tr[int(torch.randint(len(tr), (1,), generator=g))]
                loss = loss_of(c, True) + a.pull * ((W - W0).pow(2).mean() / scale)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    hl = held_loss()
                    if hl < best[0]:
                        best = (hl, W.detach().clone(), b.detach().clone(), step)
                    print(f"  layer {j} step {step}: held-out attention error {hl:.4f} (start {l0:.4f})", flush=True)
            newW[i], newb[i] = best[1].bfloat16(), best[2].bfloat16()
            out["per_layer"][str(j)] = {"held_start": l0, "held_best": best[0], "best_step": best[3]}
            print(f"layer {j}: held-out relative attention error {l0:.4f} -> {best[0]:.4f} (step {best[3]})", flush=True)
            del opt, W, b
            torch.cuda.empty_cache()
        fit = SkipMaps(maps.cut, maps.mode, newW, newb, maps.lam)
        fit.meta = {**maps.meta, "attnfit": out | {"T": a.T, "q": a.q, "n_docs": a.n_docs, "steps": a.steps,
                                                    "lr": a.lr, "pull": a.pull}}
        fit.save(os.path.join(a.maps_dir, f"{name}_lm_cut{a.cut}_normed_attnfit.pt"))
        json.dump(out, open(os.path.join(a.maps_dir, f"{name}_attnfit_cut{a.cut}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
