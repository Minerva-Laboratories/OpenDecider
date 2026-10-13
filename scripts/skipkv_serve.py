"""The skipped state pass for real (src/opendecider/skipkv.py): fit the maps, then accuracy against the exact path and
wall-clock latency, on the serving path load_decider installs.

Per mode (input: the probe's maps + each layer's own input norm; normed: straight to the normed input) the maps are
fitted on --n-calib synthetic calibration states (data/synthetic/calib.jsonl; ridge strength by held-out error on
val.jsonl), saved to --maps-dir, and the model is evaluated with them: typed decisions (first --cases), Banking77
8-way and prompt injections (scripts/skipkv_probe.py's paths and agreement metrics), then the state pass alone and
decide() (bench_state_cache.QS: 3 questions) for states of about 0.9k / 3.7k / 15k tokens, median of 3 after a warm-up.

    python scripts/skipkv_serve.py --ckpt runs/x9b-clean3/model.pt --cut 16 --out /mnt/ext/opendecider/skipkv/serve_9b.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from bench_state_cache import QS, state  # noqa: E402
from eval.typed_decisions import load_cases  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.formatting import record_state_text  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps, fit_skip_maps, skipped_share  # noqa: E402
from skipkv_probe import evaluate, score, state_texts  # noqa: E402

SIZES = (20, 100, 400)


def timed(f, reps=3):
    f()
    lat = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
    return float(np.median(lat))


def latency(dec):
    model, bb = dec.model, dec.model.backbone
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    out = {}
    for n in SIZES:
        ids = bb.tokenize([record_state_text({"state": state(n, n), "questions": []}, fmt, cap)])
        out[f"tokens_{n}"] = len(ids[0])
        torch.cuda.reset_peak_memory_stats()
        out[f"state_pass_s_{n}"] = timed(lambda: bb.prefix_cache_batch(ids))
        out[f"decide_s_{n}"] = timed(lambda: dec.decide({"state": state(n, n), "questions": QS}))
        out[f"peak_gb_{n}"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cut", type=int, default=16)
    ap.add_argument("--modes", nargs="+", default=["normed", "input"])
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-held", type=int, default=10)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--n-public", type=int, default=None)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.maps_dir, exist_ok=True)
    with gpu_lock("skipkv_serve"):
        dec = load_decider(a.ckpt)
        model, bb = dec.model, dec.model.backbone
        cases = load_cases("test", a.cases)
        b77 = [json.loads(l) for l in open("data/public/banking77_8way.jsonl")][: a.n_public]
        inj = [json.loads(l) for l in open("data/public/prompt_injections.jsonl")][: a.n_public]
        res = {"ckpt": a.ckpt, "cut": a.cut, "skipped_compute": skipped_share(bb.lm.config, a.cut)}
        ref = evaluate(model, cases, b77, inj)
        res["exact"] = score(ref) | latency(dec)
        print("exact", res["exact"], flush=True)
        for mode in a.modes:
            path = os.path.join(a.maps_dir, f"{os.path.basename(os.path.dirname(a.ckpt))}_cut{a.cut}_{mode}.pt")
            if os.path.exists(path):
                maps = SkipMaps.load(path, bb.device)
            else:
                t = time.time()
                maps = fit_skip_maps(bb, state_texts(model, "data/synthetic/calib.jsonl", a.n_calib, 0),
                                     state_texts(model, "data/synthetic/val.jsonl", a.n_held, 1), a.cut, mode)
                maps.meta |= {"fit_seconds": time.time() - t, "ckpt": a.ckpt}
                maps.save(path)
                torch.cuda.empty_cache()
            bb.skip_state = maps
            try:
                ev = evaluate(model, cases, b77, inj)
                res[mode] = score(ev, ref) | latency(dec) | {"maps": path, "lambda": maps.lam,
                                                             "held_rel_err_last": maps.meta.get("held_rel_err", {}).get(
                                                                 str(len(bb.lm.layers) - 1))}
            finally:
                bb.skip_state = None
            print(mode, res[mode], flush=True)
            del maps; torch.cuda.empty_cache()
            json.dump(res, open(a.out, "w"), indent=1)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
