"""Time and memory of the skipped prompt pass (src/opendecider/skipkv.py) against the exact one.

Per variant and prompt length: prompt-pass wall time (median of 3 after a warm-up), time to first token (prompt pass
+ the last prompt token through every layer), peak GPU memory during the pass, the memory the maps hold, and the
cache size. Prompts are FineWeb-Edu text (concatenated documents, cut to length). Variants: exact; cut<C>; cut<C>
_tail<R> (last R prompt tokens through every layer); mlp<C> (layers from C up keep their mixers, MLPs replaced by
linear surrogates; skipkv.MLPSurrogates). The arithmetic share of skipped weight work (skipkv.skipped_share) is printed
alongside.

    python scripts/skipkv_bench.py --ckpt runs/x4b-mercurius-lora/model.pt --variants cut24 cut16 cut16_tail128 \
        --out /mnt/ext/opendecider/skipkv/bench_4b.json
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

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps, _stock, skipped_share  # noqa: E402
from skipkv_lm import DATA, continue_logits, prompt_cache  # noqa: E402

GB = 2 ** 30


def cache_bytes(cache):
    n = 0
    for layer in cache.layers:
        for v in vars(layer).values():
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
            elif isinstance(v, (list, dict)):
                for t in (v.values() if isinstance(v, dict) else v):
                    if torch.is_tensor(t):
                        n += t.numel() * t.element_size()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variants", nargs="+", default=["cut24", "cut16", "cut16_tail32", "cut16_tail128"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 15000])
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--serving", action="store_true", help="install the serving kernels first (decider.prepare_inference_: "
                    "tuned int8 GEMMs; without it every forward call expands the int8 weights again)")
    ap.add_argument("--chunk", type=int, default=0, help="prefill slices of this many tokens (all variants); 0 = one pass")
    ap.add_argument("--long-from", type=int, default=32768, help="lengths from here are timed once (no repeats)")
    ap.add_argument("--no-exact", action="store_true", help="skip the exact baseline (already measured)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock("skipkv_bench"):
        model, _ = load_model(a.ckpt)
        if a.serving:
            from opendecider.decider import prepare_inference_
            prepare_inference_(model)
        bb = model.backbone
        bb.cfg.prefill_chunk = a.chunk
        name = os.path.basename(os.path.dirname(a.ckpt))
        enc = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
        ids_all = []
        for t in pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[5000:]:
            ids_all += enc(t)
            if len(ids_all) >= max(a.lengths):
                break
        res = {"ckpt": a.ckpt, "weights_gb": torch.cuda.memory_allocated() / GB, "variants": {}}
        for v in ([] if a.no_exact else ["exact"]) + a.variants:
            maps = None
            if v != "exact":
                m = re.match(r"(cut|mlp)(\d+)(?:_tail(\d+))?", v)
                kind, c, tail = m.group(1), int(m.group(2)), int(m.group(3) or 0)
                f = f"{name}_lm_cut{c}_normed.pt" if kind == "cut" else f"{name}_mlp{c}.pt"
                if kind == "cut":
                    maps = SkipMaps.load(os.path.join(a.maps_dir, f), bb.device, tail=tail)
                else:
                    from opendecider.skipkv import MLPSurrogates
                    maps = MLPSurrogates.load(os.path.join(a.maps_dir, f), bb.device, tail=tail)
            torch.cuda.empty_cache()
            held = torch.cuda.memory_allocated()
            r = {"maps_gb": (maps.nbytes() / GB) if maps is not None else 0.0}
            if v != "exact":
                L_ = bb.lm.layers[c]
                stock = _stock(L_.linear_attn if hasattr(L_, "linear_attn") else L_.self_attn)
                r["skipped_weight_work"] = (skipped_share(bb.lm.config, c, whole_mixer=not stock)["skipped"]
                                            if kind == "cut" else None)
            bb.skip_state = maps
            try:
                for n in a.lengths:
                    ids = ids_all[:n]
                    prompt_cache(bb, ids[:256]); torch.cuda.synchronize()          # warm-up
                    lat, ttft = [], []
                    for _ in range(1 if n >= a.long_from else 3):
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        cache = prompt_cache(bb, ids[:-1])
                        torch.cuda.synchronize(); t1 = time.perf_counter()
                        continue_logits(bb, cache, ids[-1:])
                        torch.cuda.synchronize(); t2 = time.perf_counter()
                        lat.append(t1 - t0); ttft.append(t2 - t0)
                    r[f"prompt_s_{n}"] = float(np.median(lat))
                    r[f"ttft_s_{n}"] = float(np.median(ttft))
                    r[f"peak_over_weights_gb_{n}"] = (torch.cuda.max_memory_allocated() - held) / GB
                    r[f"cache_gb_{n}"] = cache_bytes(cache) / GB
                    del cache
                    torch.cuda.empty_cache()
                    print(f"  {v} {n} tokens: prompt {r[f'prompt_s_{n}']:.2f} s, peak {r[f'peak_over_weights_gb_{n}']:.2f} GB",
                          flush=True)
                    json.dump(res | {"partial": {v: r}}, open(a.out, "w"), indent=1)
            finally:
                bb.skip_state = None
            res["variants"][v] = r
            print(v, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
            del maps
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
