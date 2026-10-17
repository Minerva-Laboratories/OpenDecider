"""Where does the exact tail's extra time go? (RESULTS 21.3: cut16 + tail32 costs +0.35 to +0.95 s over cut16, growing with
the prompt, far more than 32 tokens of compute.)

Per length: the skipped head pass alone (T - 32 tokens), the 32-token tail pass on its cache, the same 32-token
continuation on an EXACT cache (no skip anywhere), and one decode step; for the tail pass, CUDA-event time per layer
(attention / DeltaNet) and the remainder outside the layers (embedding, mask construction, norm).

    python scripts/skipkv_overhead.py --ckpt runs/x4b-mercurius-lora/model.pt --out /mnt/ext/opendecider/skipkv/overhead_4b.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps  # noqa: E402
from opendecider.skipdepth import segment_cache  # noqa: E402
from skipkv_lm import DATA, prompt_cache  # noqa: E402


def cuda_time(f, reps=3):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--lengths", nargs="*", type=int, default=[1024, 4096, 15360])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock("skipkv_overhead"):
        model, _ = load_model(a.ckpt)
        bb = model.backbone
        bb.skip_state = None
        name = os.path.basename(os.path.dirname(a.ckpt))
        m16 = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut16_normed.pt"), bb.device)
        texts = pq.read_table(DATA, columns=["text"]).column("text").to_pylist()[3000:3600]
        ids_all = sum((bb.tok(t, add_special_tokens=False)["input_ids"] for t in texts), [])
        res = {"ckpt": a.ckpt, "attn_impl": getattr(bb.lm.config, "_attn_implementation", None),
               "prefill_chunk": bb.cfg.prefill_chunk}
        for T in a.lengths:
            ids = ids_all[:T]
            head, tail = ids[:-32], ids[-32:]
            r = {}
            r["exact_full"] = cuda_time(lambda: prompt_cache(bb, ids))
            r["skip_head_only"] = cuda_time(lambda: segment_cache(bb, head, [(-1, m16)]))
            r["skip_head_plus_tail"] = cuda_time(lambda: segment_cache(bb, ids, [(-1, m16), (32, None)]))
            x = torch.tensor([tail], device=bb.device)
            m = torch.ones_like(x, dtype=torch.bool)

            def cont(cache_fn):
                c = cache_fn()
                torch.cuda.synchronize()
                ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                ev[0].record()
                bb._prefill(x, m, ("final",), c, False, 32)
                ev[1].record(); torch.cuda.synchronize()
                return ev[0].elapsed_time(ev[1]) / 1e3
            r["tail32_on_skipped_cache"] = float(np.median([cont(lambda: segment_cache(bb, head, [(-1, m16)])) for _ in range(3)]))
            r["tail32_on_exact_cache"] = float(np.median([cont(lambda: prompt_cache(bb, head)) for _ in range(3)]))
            # one decode token on the exact cache
            c = prompt_cache(bb, head)
            x1 = x[:, :1]; m1 = m[:, :1]
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); bb._prefill(x1, m1, ("final",), c, False, 1); e1.record(); torch.cuda.synchronize()
            r["decode1_on_exact_cache"] = e0.elapsed_time(e1) / 1e3
            # per-layer CUDA time of the 32-token continuation on the skipped cache
            c = segment_cache(bb, head, [(-1, m16)])
            evs, hs = [], []
            for j, layer in enumerate(bb.lm.layers):
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                evs.append((j, st, en))
                hs.append(layer.register_forward_pre_hook(lambda mod, a_, st=st: st.record()))
                hs.append(layer.register_forward_hook(lambda mod, a_, o, en=en: en.record()))
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record(); bb._prefill(x, m, ("final",), c, False, 32); t1.record(); torch.cuda.synchronize()
            for h_ in hs:
                h_.remove()
            per = {j: s.elapsed_time(e) / 1e3 for j, s, e in evs}
            types = bb.layer_types
            r["tail32_total"] = t0.elapsed_time(t1) / 1e3
            r["tail32_attention_layers"] = sum(v for j, v in per.items() if types[j] != "linear_attention")
            r["tail32_deltanet_layers"] = sum(v for j, v in per.items() if types[j] == "linear_attention")
            r["tail32_outside_layers"] = r["tail32_total"] - sum(per.values())
            r["tail32_slowest_layers"] = sorted(((round(v, 4), j, types[j]) for j, v in per.items()), reverse=True)[:5]
            res[T] = r
            print(T, json.dumps(r), flush=True)
            torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)
        print("OVERHEAD_DONE", flush=True)


if __name__ == "__main__":
    main()
