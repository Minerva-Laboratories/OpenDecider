"""Skipped prompt pass (src/opendecider/skipkv.py) for ordinary text generation: does it hold for a plain LM workload?

As in SwiftKV, prompt tokens run layers 0..cut-1 and every layer above the cut writes only its cache from a predicted
input; the LAST prompt token and every generated token run all layers. Measured on FineWeb-Edu documents (ODC-BY,
local sample; calibration and evaluation documents disjoint): per document, a --ctx-token prompt and the next --cont
tokens scored by teacher forcing on top of the prompt's cache. Reported per cut: continuation NLL (exact vs skipped),
mean KL(exact || skipped) of the next-token distribution, top-1 agreement, greedy continuation (--gen tokens) match,
and the prompt pass time at --ctx and 4x --ctx tokens.

    python scripts/skipkv_lm.py --ckpt runs/x4b-mercurius-lora/model.pt --cuts 24 16 --out /mnt/ext/opendecider/skipkv/lm_4b.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import MLPSurrogates, SkipMaps, _stock, fit_mlp_surrogates, fit_skip_maps, skipped_share  # noqa: E402

DATA = "/mnt/ext/mercurius/data/fineweb-edu/sample/10BT/000_00000.parquet"


@torch.no_grad()
def prompt_cache(bb, ids):
    x = torch.tensor([ids], device=bb.device)
    _, cache = bb.prefill(x, torch.ones_like(x, dtype=torch.bool), ("final",), keep_hidden=False)
    return cache


@torch.no_grad()
def continue_logits(bb, cache, ids, return_hidden=False):
    """Logits of tokens continuing `cache`. With bb.hcorr = (W, b) set (output calibration of a skipped prompt), the
    final hidden states are corrected first: h + h W + b (W (D, D) or a vector for a per-dimension scale)."""
    x = torch.tensor([ids], device=bb.device)
    f, cache = bb.forward(x, torch.ones_like(x, dtype=torch.bool), ("final",), past_key_values=cache, use_cache=True)
    h = f[0]
    hc = getattr(bb, "hcorr", None)
    if hc is not None:
        W, b = hc
        hf = h.float()
        h = (hf + (hf @ W if W.dim() == 2 else hf * W) + b).to(h.dtype)
    lg = bb.head_logits(h).float()
    return (lg, cache, f[0]) if return_hidden else (lg, cache)


@torch.no_grad()
def fit_hcorr(bb, docs, ctx, cont, maps, n_held=10, lams=(1e-1, 1, 10, 100)):
    """Output calibration for a skipped prompt that can reorder tokens: ridge from the skipped model's final hidden
    states to the exact ones (residual form, full D x D, and a per-dimension scale + shift), on teacher-forced
    continuations of calibration documents; ridge strength by held-out KL to the exact next-token distribution."""
    Hs, He, Ls = [], [], []
    for i, ids in enumerate(docs):
        x_in = ids[ctx - 1: ctx + cont - 1]
        le, _, he = continue_logits(bb, prompt_cache(bb, ids[: ctx - 1]), x_in, True)
        bb.skip_state = maps
        try:
            _, _, hs = continue_logits(bb, prompt_cache(bb, ids[: ctx - 1]), x_in, True)
        finally:
            bb.skip_state = None
        Hs.append(hs.float()); He.append(he.float())
        if i >= len(docs) - n_held:                  # full distributions only for the held-out documents
            Ls.append(le.log_softmax(-1).half().cpu())
    fit, held = slice(0, len(docs) - n_held), slice(len(docs) - n_held, len(docs))
    X, Y = torch.cat(Hs[fit]), torch.cat(He[fit])
    X1 = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    R = Y - X
    A, B = (X1.T @ X1).double(), (X1.T @ R).double()
    sc = float(A.diagonal()[:-1].mean())
    xm, rm = X.mean(0), R.mean(0)                                            # per-dimension least squares
    a = ((X - xm) * (R - rm)).sum(0) / ((X - xm) ** 2).sum(0).clamp(min=1e-6)
    diag = (a, rm - a * xm)

    def held_kl(hc):
        bb.hcorr, tot = hc, 0.0
        try:
            for hs, le in zip(Hs[held], Ls):
                W, b = hc
                h = hs + (hs @ W if W.dim() == 2 else hs * W) + b
                q = bb.head_logits(h.to(torch.bfloat16)).float().log_softmax(-1)
                p = le.to(q.device).float()
                tot += float((p.exp() * (p - q)).sum(-1).mean())
        finally:
            bb.hcorr = None
        return tot / len(Ls)

    best = None
    for lam in lams:
        Wb = torch.linalg.solve(A + lam * sc * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), B).float()
        hc = (Wb[:-1], Wb[-1])
        k = held_kl(hc)
        if best is None or k < best[0]:
            best = (k, lam, hc)
    zero = torch.zeros(X.shape[1], device=X.device)
    return {"full": best[2], "diag": diag, "lambda": best[1], "held_kl": {"none": held_kl((zero, zero)),
                                                                            "full": best[0], "diag": held_kl(diag)}}


@torch.no_grad()
def mixed_cache(bb, ids, exact_types):
    """The skipped prompt's cache, with the layers of the given types ("full_attention" / "linear_attention") taken from
    an exact pass instead: which layer type's predicted cache causes the drift. Mercurius MLA parts ("mla:inv_r",
    "mla:lat", "mla:rope"): only those fields of the compressed attention cache taken from the exact pass."""
    if "chunk256" in exact_types:             # noise floor: the exact model, prompt pass in 256-token slices
        maps, chunk, bb.skip_state, bb.cfg.prefill_chunk = bb.skip_state, bb.cfg.prefill_chunk, None, 256
        try:
            return prompt_cache(bb, ids)
        finally:
            bb.skip_state, bb.cfg.prefill_chunk = maps, chunk
    cache = prompt_cache(bb, ids)
    if exact_types:
        maps, bb.skip_state = bb.skip_state, None
        try:
            ref = prompt_cache(bb, ids)
        finally:
            bb.skip_state = maps
        parts = {"mla:inv_r": ("inv_r",), "mla:lat": ("lat_codes", "lat_norm"), "mla:rope": ("rope_codes", "rope_norm")}
        for j, t in enumerate(bb.layer_types):
            if t in exact_types:
                cache.layers[j] = ref.layers[j]
            elif t == "full_attention":
                for e in exact_types:
                    for f in parts.get(e, ()):
                        setattr(cache.layers[j], f, getattr(ref.layers[j], f))
    return cache


@torch.no_grad()
def run_doc(bb, ids, ctx, cont, gen, exact_types=()):
    """Teacher-forced continuation logits after a skipped-or-exact prompt (last prompt token through all layers), and
    a greedy continuation."""
    lg, _ = continue_logits(bb, mixed_cache(bb, ids[: ctx - 1], exact_types), ids[ctx - 1: ctx + cont - 1])
    cache = mixed_cache(bb, ids[: ctx - 1], exact_types)
    out, last = [], ids[ctx - 1]
    for _ in range(gen):
        l, cache = continue_logits(bb, cache, [last])
        last = int(l[-1].argmax()); out.append(last)
    return lg.log_softmax(-1), out


def timed(f, reps=3):
    f(); torch.cuda.synchronize()
    lat = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); lat.append(time.perf_counter() - t)
    return float(np.median(lat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cuts", nargs="+", type=int, default=[24, 16])
    ap.add_argument("--mode", default="normed")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--calib-tokens", type=int, default=256)
    ap.add_argument("--n-held", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--mix", nargs="+", default=["none"], choices=["none", "exact_attn", "exact_linear", "exact_inv_r", "exact_lat", "exact_rope", "floor",
                             "corr_full", "corr_diag"],
                    help="per cut, also evaluate with the attention (or linear-attention) layers' cache exact")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--n-corr", type=int, default=60, help="calibration documents for the corr_* arms")
    ap.add_argument("--variants", nargs="*", default=[],
                    help="extra arms per cut: tail<R> (last R prompt tokens through every layer), ema (maps with causal "
                         "EMA context features, alphas --alphas), ema_tail<R>")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.9, 0.99])
    ap.add_argument("--dump-topk", type=int, default=0,
                    help="save, per arm and teacher-forced position, the arm's top-k candidates (ids, log-probs) and the "
                         "exact log-probs of those candidates (rank-calibration study, scripts/skipkv_rankcal.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock("skipkv_lm"):
        model, _ = load_model(a.ckpt)
        bb = model.backbone
        texts = pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[:5000]
        enc = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
        calib = [bb.tok.decode(enc(t)[: a.calib_tokens]) for t in texts[: a.n_calib]]
        held = [bb.tok.decode(enc(t)[: a.calib_tokens]) for t in texts[a.n_calib: a.n_calib + a.n_held]]
        ev = []
        for t in texts[1000:]:                       # documents long enough for prompt + continuation
            ids = enc(t)
            if len(ids) >= a.ctx + a.cont:
                ev.append(ids[: a.ctx + a.cont])
            if len(ev) == a.n_eval:
                break
        long_ids = sum((enc(t) for t in texts[3000:3200]), [])[: 4 * a.ctx]
        res = {"ckpt": a.ckpt, "layers": len(bb.lm.layers), "mode": a.mode, "n_eval": len(ev), "ctx": a.ctx,
               "cont": a.cont, "gen": a.gen}
        name = os.path.basename(os.path.dirname(a.ckpt))
        maps = {}
        for c in a.cuts:                              # fit (or load) every cut first; ~|cut..L| x D^2 bf16 each
            path = os.path.join(a.maps_dir, f"{name}_lm_cut{c}_{a.mode}.pt")
            if os.path.exists(path):
                maps[c] = SkipMaps.load(path, bb.device)
            else:
                t = time.time()
                maps[c] = fit_skip_maps(bb, calib, held, c, a.mode)
                maps[c].meta |= {"fit_seconds": time.time() - t, "ckpt": a.ckpt, "data": DATA}
                maps[c].save(path)
            torch.cuda.empty_cache()
        emaps = {}
        if any(v.startswith("ema") for v in a.variants):
            for c in a.cuts:
                path = os.path.join(a.maps_dir, f"{name}_lm_cut{c}_{a.mode}_ema{'-'.join(map(str, a.alphas))}.pt")
                if os.path.exists(path):
                    emaps[c] = SkipMaps.load(path, bb.device)
                else:
                    t = time.time()
                    emaps[c] = fit_skip_maps(bb, calib, held, c, a.mode, alphas=tuple(a.alphas))
                    emaps[c].meta |= {"fit_seconds": time.time() - t, "ckpt": a.ckpt, "data": DATA}
                    emaps[c].save(path)
                torch.cuda.empty_cache()

        smaps = {}
        if any(v.startswith("mlp") for v in a.variants):
            for c in a.cuts:
                path = os.path.join(a.maps_dir, f"{name}_mlp{c}.pt")
                if os.path.exists(path):
                    smaps[c] = MLPSurrogates.load(path, bb.device)
                else:
                    t = time.time()
                    smaps[c] = fit_mlp_surrogates(bb, calib, held, c)
                    smaps[c].meta |= {"fit_seconds": time.time() - t, "ckpt": a.ckpt, "data": DATA}
                    smaps[c].save(path)
                torch.cuda.empty_cache()
        amaps = {}
        for v in a.variants:
            if v.startswith("attnfit"):
                for c in a.cuts:
                    amaps[c] = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut{c}_normed_attnfit.pt"), bb.device)

        def variant(c, v):
            m = re.search(r"tail(\d+)", v)
            if v.startswith("mlp"):
                base = smaps[c]
                return MLPSurrogates(base.cut, base.W, base.b, base.lam, int(m.group(1)) if m else 0)
            base = emaps[c] if v.startswith("ema") else amaps[c] if v.startswith("attnfit") else maps[c]
            return SkipMaps(base.cut, base.mode, base.W, base.b, base.lam, base.alphas, int(m.group(1)) if m else 0)
        hcorr = {}
        if any(m.startswith("corr_") for m in a.mix):
            cdocs = []
            for t in texts[4000:]:
                ids = enc(t)
                if len(ids) >= a.ctx + a.cont:
                    cdocs.append(ids[: a.ctx + a.cont])
                if len(cdocs) == a.n_corr:
                    break
            for c in a.cuts:
                hcorr[c] = fit_hcorr(bb, cdocs, a.ctx, a.cont, maps[c])
                res[f"hcorr_cut{c}"] = {"lambda": hcorr[c]["lambda"], "held_kl": hcorr[c]["held_kl"]}
                print(f"hcorr cut {c}", res[f"hcorr_cut{c}"], flush=True)
        MIX = {"none": (), "exact_attn": ("full_attention",), "exact_linear": ("linear_attention",),
               "exact_inv_r": ("mla:inv_r",), "exact_lat": ("mla:lat",), "exact_rope": ("mla:rope",),
               "floor": ("chunk256",), "corr_full": (), "corr_diag": ()}
        keys = [(c, m) for c in a.cuts for m in a.mix] + [(c, v) for c in a.cuts for v in a.variants]
        TS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
        acc = {k: {"nll": [], "kl": [], "top1": [], "first_diff": [], "kl_T": [], "nll_T": []} for k in ["exact"] + keys}
        dump = {}
        for i, ids in enumerate(ev):                  # compared per document: no full-vocabulary logits kept
            y = torch.tensor(ids[a.ctx: a.ctx + a.cont], device=bb.device)
            p, g0 = run_doc(bb, ids, a.ctx, a.cont, a.gen)
            acc["exact"]["nll"].append(-p.gather(1, y[:, None]).mean().item())
            for c, mx in keys:
                bb.skip_state = variant(c, mx) if mx in a.variants else maps[c]
                bb.hcorr = hcorr[c][mx[5:]] if mx.startswith("corr_") else None
                try:
                    q, g1 = run_doc(bb, ids, a.ctx, a.cont, a.gen, MIX.get(mx, ()))
                finally:
                    bb.skip_state, bb.hcorr = None, None
                d = acc[(c, mx)]
                d["nll"].append(-q.gather(1, y[:, None]).mean().item())
                d["kl"].append((p.exp() * (p - q)).sum(-1).mean().item())
                d["top1"].append((p.argmax(-1) == q.argmax(-1)).float().mean().item())
                d["first_diff"].append(next((j for j, (u, v) in enumerate(zip(g0, g1)) if u != v), a.gen))
                if a.dump_topk:
                    tq, ti = q.topk(a.dump_topk, -1)
                    dd = dump.setdefault(f"{c}_{mx}", {"skip_lp": [], "ids": [], "exact_lp": [], "exact_top": [],
                                                       "doc": [], "gold": []})
                    dd["gold"].append(y.int().cpu())
                    dd["skip_lp"].append(tq.half().cpu()); dd["ids"].append(ti.int().cpu())
                    dd["exact_lp"].append(p.gather(1, ti).half().cpu()); dd["exact_top"].append(p.argmax(-1).int().cpu())
                    dd["doc"].append(torch.full((len(ti),), i, dtype=torch.int32))
                qT = [(q / T).log_softmax(-1) for T in TS]            # one temperature on the skipped logits
                d["kl_T"].append([(p.exp() * (p - x)).sum(-1).mean().item() for x in qT])
                d["nll_T"].append([-x.gather(1, y[:, None]).mean().item() for x in qT])
                del q
            del p
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(ev)} documents", flush=True)
        if dump:
            torch.save({k: {f: torch.cat(v) for f, v in dd.items()} for k, dd in dump.items()},
                       a.out.replace(".json", f"_top{a.dump_topk}.pt"))
        tm = (lambda f: None) if a.no_timing else timed
        res["exact"] = {"nll": float(np.mean(acc["exact"]["nll"])),
                        "prompt_s": tm(lambda: prompt_cache(bb, ev[0][: a.ctx])),
                        "prompt4x_s": tm(lambda: prompt_cache(bb, long_ids))}
        print("exact", res["exact"], flush=True)
        for c, mx in keys:
            d = acc[(c, mx)]
            bb.skip_state = variant(c, mx) if mx in a.variants else maps[c]
            try:
                r = {"prompt_s": tm(lambda: prompt_cache(bb, ev[0][: a.ctx])) if mx == "none" else None,
                     "prompt4x_s": tm(lambda: prompt_cache(bb, long_ids)) if mx == "none" else None}
            finally:
                bb.skip_state = None
            dn = np.array(d["nll"]) - np.array(acc["exact"]["nll"])
            boot = np.random.default_rng(0).choice(len(dn), (1000, len(dn)))
            L_ = bb.lm.layers[c]
            stock = _stock(L_.linear_attn if hasattr(L_, "linear_attn") else L_.self_attn)
            r |= {"skipped_compute": skipped_share(bb.lm.config, c, whole_mixer=not stock)["skipped"], "lambda": maps[c].lam,
                  "nll": float(np.mean(d["nll"])), "nll_delta": float(dn.mean()),
                  "nll_delta_ci95": [float(x) for x in np.percentile(dn[boot].mean(1), [2.5, 97.5])],
                  "kl": float(np.mean(d["kl"])), "kl_doc_max": float(np.max(d["kl"])),
                  "top1_agree": float(np.mean(d["top1"])),
                  "greedy_identical": float(np.mean([f == a.gen for f in d["first_diff"]])),
                  "greedy_first_diff_mean": float(np.mean(d["first_diff"]))}
            # temperature fitted on the first half of the documents (KL to exact), reported on the second half
            h = len(d["kl_T"]) // 2
            kT, nT = np.array(d["kl_T"]), np.array(d["nll_T"])
            bi = int(kT[:h].mean(0).argmin())
            en = np.array(acc["exact"]["nll"])
            r |= {"temperature": TS[bi], "kl_heldout_raw": float(kT[h:, TS.index(1.0)].mean()),
                  "kl_heldout_calibrated": float(kT[h:, bi].mean()),
                  "nll_delta_heldout_raw": float((nT[h:, TS.index(1.0)] - en[h:]).mean()),
                  "nll_delta_heldout_calibrated": float((nT[h:, bi] - en[h:]).mean())}
            name = f"cut{c}" + ("" if mx == "none" else f"_{mx}")
            res[name] = r
            print(name, r, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
