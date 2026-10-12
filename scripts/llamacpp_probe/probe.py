"""P8.2 feasibility probe: the 9B's trained head on features computed by llama.cpp instead of PyTorch.

  dump    load the model (PyTorch path, as benchmarked), run Banking77 8-way, record per item the exact token ids the
          head's backbone pass used (state; question row prefix; every option) and the reference features / LM
          log-probs / logits; write the int32 request file for od_features.
  (then)  od_features <Q4_0 gguf> <vera lora gguf> <dir>/req.bin <dir>/feat.bin
  replay  load the model again, feed od_features' output to the head in place of the backbone pass, report
          accuracy, agreement with the PyTorch path, per-layer cosine similarity and LM log-prob differences.

    python scripts/llamacpp_probe/probe.py dump   --ckpt runs/x9b-clean3/model.pt --dir /mnt/ext/opendecider/p8/llamacpp
    python scripts/llamacpp_probe/probe.py replay --ckpt runs/x9b-clean3/model.pt --dir /mnt/ext/opendecider/p8/llamacpp
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from eval.evaluate import _logits_for  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.formatting import answer_option_text, answer_row_prefix, record_state_text  # noqa: E402
from opendecider.v3 import V3  # noqa: E402

D = 4096


def items(path, n=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:n] if n else rows


def tok_layout(model, questions):
    bb, cap = model.backbone, model.cfg.v3_list_cap
    q_tok = bb.tokenize([answer_row_prefix(q.prompt, q.options, cap) for q in questions])
    o_tok = [bb.tokenize([answer_option_text(o) for o in q.options]) for q in questions]
    return q_tok, o_tok


def run(model, rows, hook):
    """Benchmark path (scripts/bench_quant.public) with V3._branched_feats wrapped by hook(orig, self, args)."""
    orig = V3._branched_feats
    V3._branched_feats = lambda self, *a: hook(orig, self, a)
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    out = []
    try:
        for i, r in enumerate(rows):
            q = r["questions"][0]
            run.cur = i
            with torch.no_grad():
                mem = model.encode_states([record_state_text({"state": r["state"], "questions": []}, fmt, cap)])
                run.state_ids = mem.extra_prefix[0]
                lg = _logits_for(model, [Question(q["prompt"], q["options"], "choice")], mem, 64)[0]
            out.append(np.array(lg[: len(q["options"])], float))
    finally:
        V3._branched_feats = orig
    return out


def dump(a, model, rows):
    os.makedirs(a.dir, exist_ok=True)
    if a.vera_off:                                   # control: the PyTorch path without the adapters
        model.vera.remove()
    reqs, ref = [], {}

    def hook(orig, self, args):
        questions = args[0]
        assert not (self.cfg.v3_question_cache and self.cfg.v3_row_format == "answer"), "expects the plain row path"
        res = orig(self, *args)
        q_tok, o_tok = tok_layout(self, questions)
        reqs.append((list(run.state_ids), q_tok, o_tok))
        _, _, q_flat, lq, o_flat, lo = res
        ref[run.cur] = (q_flat.half().cpu().numpy(), o_flat.half().cpu().numpy(), self._lm_lp.float().cpu().numpy(),
                        lq.numpy(), lo.numpy())
        return res

    logits = run(model, rows, hook)
    sfx = "_veraoff" if a.vera_off else ""
    with open(os.path.join(a.dir, f"req{sfx}.bin"), "wb") as f:
        f.write(struct.pack("i", len(reqs)))
        w = lambda ids: f.write(struct.pack("i", len(ids)) + np.asarray(ids, np.int32).tobytes())
        for st, q_tok, o_tok in reqs:
            w(st); f.write(struct.pack("i", len(q_tok)))
            for qt, ots in zip(q_tok, o_tok):
                w(qt); f.write(struct.pack("i", len(ots)))
                for ot in ots:
                    w(ot)
    np.savez(os.path.join(a.dir, f"ref{sfx}.npz"), **{f"{k}_{i}": v for i, t in ref.items()
                                                for k, v in zip(("q", "o", "lp", "lq", "lo"), t)})
    json.dump({"logits": [l.tolist() for l in logits], "labels": [r["questions"][0]["label"] for r in rows],
               "n_state_tokens": [len(s) for s, _, _ in reqs]}, open(os.path.join(a.dir, f"ref{sfx}.json"), "w"))
    acc = np.mean([int(np.argmax(l)) == r["questions"][0]["label"] for l, r in zip(logits, rows)])
    print(f"PyTorch path: Banking77 8-way accuracy {acc:.3f} on {len(rows)} items; request file written")


def read_feats(a, n_items):
    """od_features output, per item: q (nq x 16384 fp16), o (sum no x 16384 fp16), lp (sum no) -- lengths from ref."""
    ref = np.load(os.path.join(a.dir, "ref.npz"))
    raw = open(os.path.join(a.dir, a.feat), "rb").read()
    off, out = 0, []
    for i in range(n_items):
        lq, lo = ref[f"lq_{i}"], ref[f"lo_{i}"]
        assert len(lq) == 1
        nq = int(lq[0]); q = np.frombuffer(raw, np.float16, nq * 4 * D, off).reshape(nq, 4 * D); off += q.nbytes
        os_, lps = [], []
        for no in lo.tolist():
            o = np.frombuffer(raw, np.float16, no * 4 * D, off).reshape(no, 4 * D); off += o.nbytes
            lp = np.frombuffer(raw, np.float32, no, off); off += lp.nbytes
            os_.append(o); lps.append(lp)
        out.append((q, np.concatenate(os_), np.concatenate(lps)))
    assert off == len(raw), f"read {off} of {len(raw)} bytes"
    return out, ref


def replay(a, model, rows):
    feats, ref = read_feats(a, len(rows))
    R = json.load(open(os.path.join(a.dir, "ref.json")))

    def hook(orig, self, args):
        questions, mem, pc, smask, sh = args
        q, o, lp = feats[run.cur]
        dev = self.backbone.device
        self._lm_lp = torch.from_numpy(lp.copy()).to(dev).float()
        lq = torch.from_numpy(ref[f"lq_{run.cur}"]); lo = torch.from_numpy(ref[f"lo_{run.cur}"])
        return (None, torch.tensor([len(run.state_ids)]), torch.from_numpy(q.copy()).to(dev).bfloat16(), lq,
                torch.from_numpy(o.copy()).to(dev).bfloat16(), lo)

    logits = run(model, rows, hook)
    labels = np.array(R["labels"])
    pt = [np.array(l) for l in R["logits"]]
    T = model_temperature(a)
    sm = lambda z: np.exp(z / T - np.logaddexp.reduce(z / T))
    acc_pt = np.mean([np.argmax(l) == y for l, y in zip(pt, labels)])
    acc_ll = np.mean([np.argmax(l) == y for l, y in zip(logits, labels)])
    agree = np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(pt, logits)])
    tv = [0.5 * np.abs(sm(x) - sm(y)).sum() for x, y in zip(pt, logits)]
    cos = {k: [] for k in ("L8", "L16", "L24", "final")}
    dlp = []
    for i, (q, o, lp) in enumerate(feats):
        X = np.concatenate([ref[f"q_{i}"], ref[f"o_{i}"]]).astype(np.float32)
        Y = np.concatenate([q, o]).astype(np.float32)
        for j, k in enumerate(cos):
            x, y = X[:, j * D:(j + 1) * D], Y[:, j * D:(j + 1) * D]
            cos[k].extend((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9))
        dlp.extend(np.abs(ref[f"lp_{i}"] - lp))
    res = {"n": len(rows), "acc_pytorch": float(acc_pt), "acc_llamacpp": float(acc_ll), "argmax_agreement": float(agree),
           "prob_tv_mean": float(np.mean(tv)), "prob_tv_max": float(np.max(tv)),
           "cosine_mean": {k: float(np.mean(v)) for k, v in cos.items()},
           "cosine_min": {k: float(np.min(v)) for k, v in cos.items()},
           "lm_logprob_absdiff_mean": float(np.mean(dlp)), "lm_logprob_absdiff_max": float(np.max(dlp))}
    print(json.dumps(res, indent=1))
    res["feat"] = a.feat
    json.dump(res, open(os.path.join(a.dir, f"probe_result_{os.path.splitext(a.feat)[0]}.json"), "w"), indent=1)


def model_temperature(a):
    ex = torch.load(a.ckpt, map_location="cpu", weights_only=False)["extra"]
    return (ex.get("temperature_by_type") or {}).get("choice", ex.get("temperature", 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "replay"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default="data/public/banking77_8way.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--feat", default="feat.bin", help="replay: od_features output file in --dir")
    ap.add_argument("--vera-off", action="store_true", help="dump: control run with the VeRA adapters removed")
    a = ap.parse_args()
    model, _ = load_model(a.ckpt)               # plain row path (v3_question_cache off in this checkpoint)
    model.eval()
    rows = items(a.data, a.n)
    (dump if a.mode == "dump" else replay)(a, model, rows)


if __name__ == "__main__":
    main()
