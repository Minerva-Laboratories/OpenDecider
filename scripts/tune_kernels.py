"""Tune the fused int8 GEMM kernels (GemLite, Triton) for THIS GPU, once per device.

    .venv/bin/python scripts/tune_kernels.py --ckpt checkpoints/opendecider-9b          # writes the per-GPU file
    .venv/bin/python scripts/tune_kernels.py --ckpt checkpoints/opendecider-9b --check  # + end-to-end latency check

The int8 backbones (`int8` and the GGUF variants, which run int8 on the GPU) otherwise expand every weight matrix to
bf16 on every forward call. With a tuned file present, the inference loaders (`load_decider`) swap those layers for
GemLite A16W8 kernels that read the same int8 weights directly (src/opendecider/quant.py, `fast_int8_kernels_`).
Untuned GemLite is slower than the dequantize path at large batch sizes on the Orin, so nothing is swapped without
this file. Output: ~/.cache/opendecider/kernels/gemlite_<gpu>_sm<cc>.json (or $OPENDECIDER_KERNEL_DIR), plus
gemlite_<gpu>_sm<cc>_dense_from.json: per shape, the batch size from which expanding the int8 weight and using cuBLAS
is faster than GemLite (measured here; the layer switches per call).

Every distinct (out, in) shape of the checkpoint's decoder layers is tuned at each of GemLite's batch-size buckets
(tokens per call: 1, 2, 4, 8, 16, 24, 32, 48, ... 3072, 4096); a call uses the config of the next bucket up, and an
untuned bucket falls back to a default config that can be 2x slower. "fast" autotuning takes ~10-60 minutes on
the Orin for the 9B; "max" can take hours per shape there.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_INT8_GEMM"] = "dequant"          # load the plain int8 layers; we swap them ourselves below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/opendecider-9b", help="checkpoint folder/repo or runs/*/model.pt")
    ap.add_argument("--backbone", default=None, help="backbone variant (checkpoint folders only)")
    ap.add_argument("--max-m", type=int, default=4096, help="largest tokens-per-call bucket (GemLite caps at 4096)")
    ap.add_argument("--mode", default="fast", choices=["fast", "max"])
    ap.add_argument("--check", action="store_true", help="time a request with the dequantize path and the tuned one")
    a = ap.parse_args()
    import gemlite
    from opendecider.guards import gpu_lock
    from opendecider.quant import (Int8GemLite, Int8Linear, gemlite_config_path, int8_to_gemlite,
                                   int8_to_gemlite_)
    gemlite.set_autotune(a.mode)
    out = gemlite_config_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        gemlite.load_config(out)                          # resume: tuned buckets are skipped by the cache lookup
    with gpu_lock("tune_kernels"):
        if a.ckpt.endswith(".pt"):
            from opendecider.checkpoint import load_decider
            dec = load_decider(a.ckpt)
        else:
            from opendecider.hub import load_decider
            dec = load_decider(a.ckpt, backbone=a.backbone)
        bb = dec.model.backbone
        shapes = {}
        for m in bb.lm.layers.modules():
            if isinstance(m, Int8Linear):
                shapes.setdefault(tuple(m.qweight.shape), m)
        if not shapes:
            raise SystemExit("no int8 layers in this backbone variant (4-bit variants already use GemLite)")
        from gemlite.triton_kernels import utils as gu      # the batch-size buckets GemLite looks configs up by
        ms = sorted({gu.M_MAPPING[m] for m in range(1, gu.M_MAXVAL + 1)} | {gu.M_MAXVAL})
        ms = [m for m in ms if m <= a.max_m]
        print(f"{len(shapes)} shapes x {len(ms)} batch sizes -> {out}", flush=True)
        t0 = time.time()
        cross_path = out.replace(".json", "_dense_from.json")
        dense_from = json.load(open(cross_path)) if os.path.exists(cross_path) else {}
        for (n, k), m in shapes.items():
            lay = Int8GemLite(int8_to_gemlite(m))
            for M in ms:
                x = torch.randn(M, k, device=bb.device, dtype=m.scale.dtype)
                ts = time.time()
                lay(x); torch.cuda.synchronize()
                print(f"  {n}x{k} M={M}: {time.time() - ts:.1f}s", flush=True)
            gemlite.cache_config(out)                     # save after every shape (resumable)
            # crossover: the smallest bucket from which expanding the weight + cuBLAS beats GemLite at every larger one
            faster = []
            for M in [b for b in ms if b >= 64] + [2 * ms[-1], 4 * ms[-1]]:
                x = torch.randn(M, k, device=bb.device, dtype=m.scale.dtype)
                t = {}
                for path, lim in (("gemlite", None), ("dense", 1)):
                    lay.dense_from = lim
                    lay(x); torch.cuda.synchronize(); ts = time.perf_counter()
                    for _ in range(5):
                        lay(x)
                    torch.cuda.synchronize(); t[path] = (time.perf_counter() - ts) / 5
                faster.append((M, t["dense"] < t["gemlite"]))
                print(f"  {n}x{k} M={M}: gemlite {1e3 * t['gemlite']:.2f} ms, dense {1e3 * t['dense']:.2f} ms", flush=True)
            cut = None
            for M, dense_wins in reversed(faster):
                if not dense_wins:
                    break
                cut = M
            dense_from[f"{n}x{k}"] = cut
            json.dump(dense_from, open(cross_path, "w"), indent=1)
            del lay
        print(f"tuned in {(time.time() - t0) / 60:.1f} min; dense from {dense_from}", flush=True)
        if a.check:
            from profile_latency import workload
            from bench_state_cache import state
            req = {"state": state(25, 25), "questions": workload(16, 32)}
            res = {}
            for label in ("dequant", "gemlite"):
                if label == "gemlite":
                    gemlite.set_autotune(False)
                    gemlite.load_config(out)
                    int8_to_gemlite_(bb.lm.layers, dense_from)
                    torch.cuda.empty_cache()
                for _ in range(2):
                    dec.decide(req)
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(3):
                    r = dec.decide(req)
                torch.cuda.synchronize()
                res[label] = {"s": (time.perf_counter() - t) / 3,
                              "probs": {q: v["probs"] for q, v in r["answers"].items()}}
            diff = max(abs(res["dequant"]["probs"][q][o] - res["gemlite"]["probs"][q][o])
                       for q in res["dequant"]["probs"] for o in res["dequant"]["probs"][q])
            print(json.dumps({"dequant_s": round(res["dequant"]["s"], 2), "gemlite_s": round(res["gemlite"]["s"], 2),
                              "max_prob_diff": round(diff, 5)}), flush=True)


if __name__ == "__main__":
    main()
