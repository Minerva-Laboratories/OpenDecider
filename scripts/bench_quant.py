"""Memory, latency and accuracy of the 2B model under weight / cache quantization.

    .venv/bin/python scripts/bench_quant.py --ckpt runs/x2b/model.pt [--configs int8:int8 nf4:int4] [--cases 100]

Each config is "weights:cache[:graphs]": weights int8 | nf4 (bitsandbytes) | w8 | w4 (GemLite Triton GEMMs, round to
nearest) | awq (pre-quantized AWQ checkpoint on GemLite int4 kernels, configs/*_awq.yaml), cache
int8 | int4, ":graphs" = CUDA-graph engine (src/opendecider/deploy.py), ":qcache" = question-cache row path
(exact; question tokens computed once per question instead of once per option). Embeddings stay int8
(every cached K/V and the state memory; GDN recurrent state stays bf16/fp32). Per config:
  memory  : GPU memory after load, and peak during one decide() for states of about 0.9k / 3.9k / 15k tokens
  latency : decide() wall time for the same states (prefix cache off; median of 3 after a warm-up call per size)
  accuracy: typed-decisions test (first --cases cases, 5 questions each; argmax vs gold argmax, KL, Brier vs gold)
            and the public Banking77 8-way and prompt-injection sets (accuracy)
The model weights were trained with int8 weights and cache; nothing is retrained.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from bench_state_cache import QS, state  # noqa: E402
from eval.evaluate import _logits_for  # noqa: E402
from eval.typed_decisions import load_cases, questions_of  # noqa: E402
from opendecider.backbone import Backbone, BackboneConfig  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.decider import Decider  # noqa: E402
from opendecider.formatting import record_state_text  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402

GB = 2 ** 30
AWQ = {"Qwen/Qwen3.5-2B": "configs/backbone_2b_awq.yaml"}


def logits(model, state_text, qs):
    with torch.no_grad():
        mem = model.encode_states([state_text])
        return _logits_for(model, qs, mem, 64)


def typed(model, rows):
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    acc, kl, brier = [], [], []
    for r in rows:
        items = questions_of(r)
        qs = [Question(instr, texts, "choice", lab) for _, instr, _, texts, _, lab in items]
        for (_, _, names, _, gp, _), lg in zip(items, logits(model, record_state_text({"state": r["state"], "questions": []}, fmt, cap), qs)):
            z = np.array(lg[: len(names)]); p = np.exp(z - np.logaddexp.reduce(z))
            acc.append(float(p.argmax() == gp.argmax()))
            kl.append(float((gp * (np.log(np.clip(gp, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()))
            brier.append(float(((p - gp) ** 2).sum()))
    return {"acc": float(np.mean(acc)), "kl": float(np.mean(kl)), "brier": float(np.mean(brier)), "n": len(acc)}


def public(model, path):
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    hit = []
    for l in open(path):
        r = json.loads(l)
        q = r["questions"][0]
        lg = logits(model, record_state_text({"state": r["state"], "questions": []}, fmt, cap),
                    [Question(q["prompt"], q["options"], "choice")])[0]
        hit.append(float(int(np.argmax(lg[: len(q["options"])])) == q["label"]))
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--configs", nargs="+", default=["int8:int8", "int8:int4", "nf4:int8", "nf4:int4"])
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--no-acc", action="store_true", help="memory and latency only (run one config per process)")
    ap.add_argument("--out", default="runs/quant/bench.json")
    a = ap.parse_args()
    os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    rows = [] if a.no_acc else load_cases("test", a.cases)
    out = {}
    with gpu_lock("bench_quant"):
        for c in a.configs:
            w, kv, *mode = c.split(":")
            graphs, qcache = "graphs" in mode, "qcache" in mode
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            over = {"weight_quant": w, "kv_quant": kv}
            if w == "awq":                  # pre-quantized AWQ checkpoint for this backbone (configs/*_awq.yaml)
                import yaml
                over.update({k: v for k, v in yaml.safe_load(open(AWQ[ck["backbone_cfg"]["repo_id"]])).items()
                             if k in ("path", "repo_id", "revision")})
            bb = Backbone.load(BackboneConfig(**{**ck["backbone_cfg"], **over}))
            model, extra = load_model(a.ckpt, backbone=bb)
            if qcache:                      # question read once per question; option rows carry only their tokens
                model.cfg.v3_question_cache = True
            dec = Decider(model, extra.get("temperature", 1.0), extra.get("temperature_by_type"))
            if graphs:
                from opendecider.deploy import GraphEngine
                dec.engine = GraphEngine(model).install()
            torch.cuda.synchronize()
            res = {"weights_gb": torch.cuda.memory_allocated() / GB}
            dec.decide({"state": state(5, 1), "questions": QS})                      # warm-up
            for n in (20, 100, 400):
                req = {"state": state(n, n), "questions": QS}
                dec.decide(req)                                                  # warm-up (captures graphs)
                torch.cuda.reset_peak_memory_stats()
                lat = []
                for _ in range(3):
                    torch.cuda.synchronize(); t = time.perf_counter(); o = dec.decide(req)
                    torch.cuda.synchronize(); lat.append(time.perf_counter() - t)
                res[f"tokens_{n}"] = o["input_tokens"]
                res[f"peak_gb_{n}"] = torch.cuda.max_memory_allocated() / GB
                res[f"latency_s_{n}"] = float(np.median(lat))
            if graphs:
                res["graph_stats"] = dict(dec.engine.stats)
            with torch.no_grad():                                                   # stored state cache, 15k tokens
                pc, _ = bb.prefix_cache_batch(bb.tokenize([record_state_text({"state": state(400, 400), "questions": []})]))
                res["state_cache_mb_400"] = pc.nbytes() / 2 ** 20
                del pc
            if not a.no_acc:
                res["typed"] = typed(model, rows)
                res["banking77_8way"] = public(model, "data/public/banking77_8way.jsonl")
                res["prompt_injections"] = public(model, "data/public/prompt_injections.jsonl")
            out[c] = res
            print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
            del dec, model, bb
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
