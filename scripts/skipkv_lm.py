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
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps, _stock, fit_skip_maps, skipped_share  # noqa: E402

DATA = "/mnt/ext/mercurius/data/fineweb-edu/sample/10BT/000_00000.parquet"


@torch.no_grad()
def prompt_cache(bb, ids):
    x = torch.tensor([ids], device=bb.device)
    _, cache = bb.prefill(x, torch.ones_like(x, dtype=torch.bool), ("final",), keep_hidden=False)
    return cache


@torch.no_grad()
def continue_logits(bb, cache, ids):
    x = torch.tensor([ids], device=bb.device)
    f, cache = bb.forward(x, torch.ones_like(x, dtype=torch.bool), ("final",), past_key_values=cache, use_cache=True)
    return bb.head_logits(f[0]).float(), cache


@torch.no_grad()
def run_doc(bb, ids, ctx, cont, gen):
    """Teacher-forced continuation logits after a skipped-or-exact prompt (last prompt token through all layers), and
    a greedy continuation."""
    lg, _ = continue_logits(bb, prompt_cache(bb, ids[: ctx - 1]), ids[ctx - 1: ctx + cont - 1])
    cache = prompt_cache(bb, ids[: ctx - 1])
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
        acc = {k: {"nll": [], "kl": [], "top1": [], "first_diff": []} for k in ["exact"] + a.cuts}
        for i, ids in enumerate(ev):                  # compared per document: no full-vocabulary logits kept
            y = torch.tensor(ids[a.ctx: a.ctx + a.cont], device=bb.device)
            p, g0 = run_doc(bb, ids, a.ctx, a.cont, a.gen)
            acc["exact"]["nll"].append(-p.gather(1, y[:, None]).mean().item())
            for c in a.cuts:
                bb.skip_state = maps[c]
                try:
                    q, g1 = run_doc(bb, ids, a.ctx, a.cont, a.gen)
                finally:
                    bb.skip_state = None
                d = acc[c]
                d["nll"].append(-q.gather(1, y[:, None]).mean().item())
                d["kl"].append((p.exp() * (p - q)).sum(-1).mean().item())
                d["top1"].append((p.argmax(-1) == q.argmax(-1)).float().mean().item())
                d["first_diff"].append(next((j for j, (u, v) in enumerate(zip(g0, g1)) if u != v), a.gen))
                del q
            del p
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(ev)} documents", flush=True)
        res["exact"] = {"nll": float(np.mean(acc["exact"]["nll"])),
                        "prompt_s": timed(lambda: prompt_cache(bb, ev[0][: a.ctx])),
                        "prompt4x_s": timed(lambda: prompt_cache(bb, long_ids))}
        print("exact", res["exact"], flush=True)
        for c in a.cuts:
            d = acc[c]
            bb.skip_state = maps[c]
            try:
                r = {"prompt_s": timed(lambda: prompt_cache(bb, ev[0][: a.ctx])),
                     "prompt4x_s": timed(lambda: prompt_cache(bb, long_ids))}
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
            res[f"cut{c}"] = r
            print(f"cut {c}", r, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
