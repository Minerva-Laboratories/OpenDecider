"""The llama.cpp serving backend (opendecider.llamacpp) against the PyTorch path.

  parity   Banking77 8-way through the backend (same benchmark path as probe.py: one state, one question, raw logits)
           against the PyTorch logits that probe.py dump saved (ref_veraoff.json: PyTorch without VeRA, the function
           the backend serves; ref.json: with VeRA): accuracy, agreement, total variation.
  latency  Decider.decide end to end on the states of scripts/bench_quant.py (20 / 100 / 400 records), against its
           PyTorch serving numbers (RESULTS 20.1)
  accuracy bench_quant's typed decisions (100 cases) and prompt-injection set

    python scripts/llamacpp_probe/backend_check.py --ckpt runs/x9b-clean3/model.pt \\
        --gguf models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_0.gguf --ref-dir /mnt/ext/opendecider/p8/llamacpp --out res.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from eval.evaluate import _logits_for  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.formatting import record_state_text  # noqa: E402


def parity(dec, a):
    model = dec.model
    rows = [json.loads(l) for l in open(a.data)]
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    logits, t = [], time.time()
    for r in rows:
        q = r["questions"][0]
        with torch.no_grad():
            mem = model.encode_states([record_state_text({"state": r["state"], "questions": []}, fmt, cap)])
            lg = _logits_for(model, [Question(q["prompt"], q["options"], "choice")], mem, 64)[0]
        logits.append(np.array(lg[: len(q["options"])], float))
    sec = (time.time() - t) / len(rows)
    T = (dec.temperature_by_type or {}).get("choice", dec.temperature)
    sm = lambda z: np.exp(z / T - np.logaddexp.reduce(z / T))
    y = np.array([r["questions"][0]["label"] for r in rows])
    out = {"n": len(rows), "acc_llamacpp": float(np.mean([np.argmax(l) == t for l, t in zip(logits, y)])),
           "s_per_item": sec}
    for name in ("ref_veraoff", "ref"):
        R = json.load(open(os.path.join(a.ref_dir, name + ".json")))
        pt = [np.array(l) for l in R["logits"]]
        out[name] = {"acc_pytorch": float(np.mean([np.argmax(l) == t for l, t in zip(pt, y)])),
                     "agreement": float(np.mean([np.argmax(p) == np.argmax(l) for p, l in zip(pt, logits)])),
                     "tv_mean": float(np.mean([0.5 * np.abs(sm(p) - sm(l)).sum() for p, l in zip(pt, logits)])),
                     "tv_max": float(np.max([0.5 * np.abs(sm(p) - sm(l)).sum() for p, l in zip(pt, logits)]))}
    return out


def latency(dec, a):
    """The states and questions of scripts/bench_quant.py (bench_state_cache.state / QS), a different state for each
    timed call (llama.cpp keeps the last state, which would make repeats free)."""
    from bench_state_cache import QS, state
    out = {}
    dec.decide({"state": state(5, 1), "questions": QS})
    for n in (20, 100, 400):
        dec.decide({"state": state(n, n), "questions": QS})
        ts = []
        for i in range(3):
            req = {"state": state(n, 1000 * n + i), "questions": QS}
            t = time.time(); o = dec.decide(req); ts.append(time.time() - t)
        out[f"latency_s_{n}"] = float(np.median(ts))
        out[f"tokens_{n}"] = int(o["input_tokens"])
        t = time.time(); dec.decide(req); out[f"latency_s_{n}_same_state"] = time.time() - t
    return out


def accuracy(dec, a):
    """bench_quant's typed decisions (100 test cases) and the prompt-injection set, as in RESULTS 20.1."""
    from bench_quant import public, typed
    from eval.typed_decisions import load_cases
    return {"typed": typed(dec.model, load_cases("test", 100)),
            "prompt_injections": public(dec.model, "data/public/prompt_injections.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--data", default="data/public/banking77_8way.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    a = ap.parse_args()
    t = time.time()
    dec = load_decider(a.ckpt, llamacpp_gguf=a.gguf)
    res = {"load_s": time.time() - t}
    res["parity"] = parity(dec, a)
    print(json.dumps(res, indent=1), flush=True)
    if not a.skip_latency:
        res["latency"] = latency(dec, a)
        print(json.dumps(res["latency"]), flush=True)
    if not a.skip_accuracy:
        res["accuracy"] = accuracy(dec, a)
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
