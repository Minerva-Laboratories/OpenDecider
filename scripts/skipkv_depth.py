"""Per-token depth for the skipped prompt pass ("foveated depth", src/opendecider/skipdepth.py) against the plain cuts
and the two-call exact tail (RESULTS 21.2-21.3).

Arms (L = all layers):
  cut24, cut16                 every prompt token cut there (skipkv)
  cut16_tail32                 the 21.3 exact tail: last 32 prompt tokens through every layer
  g256 / g512                  graded: last 32 all layers, the previous 224 / 480 cut 24, the rest cut 16 (segments)
  f4/f64/f256_g512, f64_tail32 the FIRST 4 / 64 / 256 prompt tokens also through every layer (segments)
  d_tail32                     cut16_tail32 through the per-token quality path (equivalence check)
  d_tail83                     recency control for the selection arms: last 83 tokens (32 + 5% of 1k) all layers
  d_sel5                       last 32 all layers + 5% of tokens by the error-model score all layers   (stock attn only)
  d_sel10_24                   last 32 all layers + 10% by score cut 24                                (stock attn only)
  d_g512_sel5                  g512 + 5% by score all layers                                          (stock attn only)
Tests: generation (FineWeb-Edu, --ctx prompt, --cont teacher-forced tokens: KL to exact, NLL delta, top-1, greedy
--gen identical), lettered multiple choice (OpenBookQA, CommonsenseQA: accuracy, answer flips vs exact), prompt-pass
time of the real (segment) forms at --lengths, and a torch.profiler table of cut16 vs cut16_tail32 (the tail's cost).

    python scripts/skipkv_depth.py --ckpt runs/x4b-mercurius-lora/model.pt --out /mnt/ext/opendecider/skipkv/depth_4b.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps, _stock, skipped_share  # noqa: E402
from opendecider.skipdepth import DepthPlan, depth_cache, plan_skipped, seg_lengths, segment_cache  # noqa: E402
from skipkv_lm import DATA, continue_logits, prompt_cache, timed  # noqa: E402
from skipkv_tasks import LETTERS, chat, mc_items  # noqa: E402


def boot_ci(x, seed=0):
    x = np.asarray(x)
    b = np.random.default_rng(seed).choice(len(x), (1000, len(x)))
    return [float(v) for v in np.percentile(x[b].mean(1), [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--n-mc", type=int, default=300)
    ap.add_argument("--tasks", nargs="*", default=["openbookqa", "commonsenseqa"])
    ap.add_argument("--lengths", nargs="*", type=int, default=[1024, 4096, 15360])
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--parts", action="store_true",
                    help="add the part-skip arms (skipparts: MLP / attention surrogates above cut 16 / 24, fitted here)")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock("skipkv_depth"):
        model, _ = load_model(a.ckpt)
        bb = model.backbone
        bb.skip_state = None
        L = len(bb.lm.layers)
        name = os.path.basename(os.path.dirname(a.ckpt))
        m16 = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut16_normed.pt"), bb.device)
        m24 = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut24_normed.pt"), bb.device)
        att15 = getattr(bb.lm.layers[15], "self_attn", None)
        stock = att15 is not None and _stock(att15)
        whole = not _stock(bb.lm.layers[16].linear_attn if hasattr(bb.lm.layers[16], "linear_attn")
                           else bb.lm.layers[16].self_attn)
        cfg = bb.lm.config
        sh = {c: skipped_share(cfg, c, whole_mixer=whole)["skipped"] for c in (16, 24)}

        def seg_share(T, segs):                      # segs as segment_cache, maps -> cut
            return sum(n * (0.0 if mp is None else sh[getattr(mp, "share_key", mp.cut)])
                       for n, (_, mp) in zip(seg_lengths(T, segs), segs)) / T

        SEG = {"cut24": [(-1, m24)], "cut16": [(-1, m16)], "cut16_tail32": [(-1, m16), (32, None)],
               "g256": [(-1, m16), (224, m24), (32, None)], "g512": [(-1, m16), (480, m24), (32, None)],
               # first N prompt tokens through every layer too (sink-like), on g512 / cut16_tail32
               "f4_g512": [(4, None), (-1, m16), (480, m24), (32, None)],
               "f64_g512": [(64, None), (-1, m16), (480, m24), (32, None)],
               "f256_g512": [(256, None), (-1, m16), (480, m24), (32, None)],
               "f64_tail32": [(64, None), (-1, m16), (32, None)]}
        PLAN = {"d_tail32": lambda: DepthPlan({16: m16}, L, [(32, None)]),
                "d_tail83": lambda: DepthPlan({16: m16}, L, [(83, None)])}
        if stock:
            PLAN |= {"d_sel5": lambda: DepthPlan({16: m16}, L, [(32, None)], (0.05, L, 64)),
                     "d_sel10_24": lambda: DepthPlan({16: m16, 24: m24}, L, [(32, None)], (0.10, 24, 64)),
                     "d_g512_sel5": lambda: DepthPlan({16: m16, 24: m24}, L, [(32, None), (512, 24)], (0.05, L, 64))}
        if a.parts:
            from opendecider.skipparts import fit_parts, part_share, part_skip
            path = os.path.join(a.maps_dir, f"{name}_parts16.pt")
            if os.path.exists(path):
                fit = torch.load(path, weights_only=False)
            else:
                texts0 = pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[:210]
                enc0 = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
                cal = [bb.tok.decode(enc0(t)[:256]) for t in texts0]
                t0 = time.time()
                fit = fit_parts(bb, cal[:200], cal[200:210], 16)
                fit["fit_seconds"] = time.time() - t0
                torch.save(fit, path)
                torch.cuda.empty_cache()
            res_parts = {"held_rel_err": fit["rel"], "fit_seconds": fit.get("fit_seconds")}
            for c in (16, 24):
                for kind, mm, aa in (("mlp", True, False), ("attn", False, True), ("mlpattn", True, True)):
                    ps = part_skip(fit, c, mm, aa, bb.device)
                    sh[f"{kind}{c}"] = part_share(cfg, c, mm, aa)
                    ps.share_key = f"{kind}{c}"
                    SEG[f"{kind}{c}"] = [(-1, ps)]
                    if c == 16:
                        SEG[f"{kind}{c}_tail32"] = [(-1, ps), (32, None)]
        names = ["exact"] + list(SEG) + list(PLAN)
        if a.arms:
            names = ["exact"] + [n for n in names if n in a.arms]
        share = {n: [] for n in names}

        def build(n, ids):
            if n == "exact":
                share[n].append(0.0)
                return prompt_cache(bb, ids)
            if n in SEG:
                share[n].append(seg_share(len(ids), SEG[n]))
                return segment_cache(bb, ids, SEG[n])
            p = PLAN[n]()
            c = depth_cache(bb, ids, p)
            share[n].append(plan_skipped(cfg, p.ct, L, whole))
            return c

        res = {"parts": res_parts if a.parts else None, "ckpt": a.ckpt, "layers": L, "stock_attention_below_cut": stock, "skipped_share": sh, "arms": names}
        # ---------------------------------------------------------------- generation
        texts = pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[:5000]
        enc = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
        ev = []
        for t in texts[1000:]:
            ids = enc(t)
            if len(ids) >= a.ctx + a.cont:
                ev.append(ids[: a.ctx + a.cont])
            if len(ev) == a.n_eval:
                break
        acc = {n: {"nll": [], "kl": [], "top1": [], "greedy": []} for n in names}
        t0 = time.time()
        for i, ids in enumerate(ev):
            y = torch.tensor(ids[a.ctx: a.ctx + a.cont], device=bb.device)
            ref = None
            for n in names:
                lg, _ = continue_logits(bb, build(n, ids[: a.ctx - 1]), ids[a.ctx - 1: a.ctx + a.cont - 1])
                q = lg.log_softmax(-1)
                cache, last, out = build(n, ids[: a.ctx - 1]), ids[a.ctx - 1], []
                share[n].pop()
                for _ in range(a.gen):
                    l, cache = continue_logits(bb, cache, [last])
                    last = int(l[-1].argmax()); out.append(last)
                if ref is None:
                    ref = (q, out)
                p, g0 = ref
                d = acc[n]
                d["nll"].append(-q.gather(1, y[:, None]).mean().item())
                d["kl"].append((p.exp() * (p - q)).sum(-1).mean().item())
                d["top1"].append((p.argmax(-1) == q.argmax(-1)).float().mean().item())
                d["greedy"].append(float(out == g0))
            print(f"gen {i + 1}/{len(ev)} {time.time() - t0:.0f}s", flush=True)
        gen = {}
        for n in names:
            d = acc[n]
            dn = np.array(d["nll"]) - np.array(acc["exact"]["nll"])
            gen[n] = {"skipped": float(np.mean(share[n])), "kl": float(np.mean(d["kl"])), "kl_ci95": boot_ci(d["kl"]),
                      "nll_delta": float(dn.mean()), "nll_delta_ci95": boot_ci(dn),
                      "top1": float(np.mean(d["top1"])), "greedy_identical": float(np.mean(d["greedy"]))}
            print("gen", n, gen[n], flush=True)
        res["generation"] = gen
        json.dump(res, open(a.out, "w"), indent=1)
        # ---------------------------------------------------------------- multiple choice
        for task in a.tasks:
            for n in names:
                share[n] = []
            items = mc_items(f"data/public/{task}.jsonl", a.n_mc)
            pred = {n: [] for n in names}
            t0 = time.time()
            for text, labels, gold in items:
                ids = enc(chat(bb, text))
                tok = [enc(l)[0] for l in labels]
                for n in names:
                    lg, _ = continue_logits(bb, build(n, ids[:-1]), ids[-1:])
                    pred[n].append(int(lg[-1][tok].float().argmax()))
            golds = [g for _, _, g in items]
            r = {n: {"acc": float(np.mean([p == g for p, g in zip(pred[n], golds)])),
                     "acc_ci95": boot_ci([p == g for p, g in zip(pred[n], golds)]),
                     "flips": float(np.mean([p != q for p, q in zip(pred[n], pred["exact"])])),
                     "skipped": float(np.mean(share[n]))} for n in names}
            r["n"], r["seconds"] = len(items), time.time() - t0
            res[task] = r
            print(task, json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        # ---------------------------------------------------------------- timing (real segment forms) + profile
        long_ids = sum((enc(t) for t in texts[3000:3600]), [])
        tim = {}
        for T in a.lengths:
            ids = long_ids[:T]
            tim[T] = {"exact": timed(lambda: prompt_cache(bb, ids), reps=1 if T > 20000 else 3)}
            for n, segs in SEG.items():
                if n not in names:
                    continue
                tim[T][n] = timed(lambda: segment_cache(bb, ids, segs), reps=1 if T > 20000 else 3)
            print("time", T, tim[T], flush=True)
            torch.cuda.empty_cache()
        res["prompt_seconds"] = tim
        json.dump(res, open(a.out, "w"), indent=1)
        if a.no_profile:
            print("DEPTH_DONE", flush=True)
            return
        from torch.profiler import ProfilerActivity, profile
        ids = long_ids[:4096]
        prof_txt = []
        for n in ("cut16", "cut16_tail32"):
            segment_cache(bb, ids, SEG[n]); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                segment_cache(bb, ids, SEG[n]); torch.cuda.synchronize()
            ka = pr.key_averages()
            prof_txt.append(f"===== {n} (4096 tokens) by CUDA time\n" +
                            ka.table(sort_by="cuda_time_total", row_limit=25) +
                            f"\n===== {n} by CPU time\n" + ka.table(sort_by="cpu_time_total", row_limit=25))
        open(a.out.replace(".json", "_profile.txt"), "w").write("\n".join(prof_txt))
        print("DEPTH_DONE", flush=True)


if __name__ == "__main__":
    main()
