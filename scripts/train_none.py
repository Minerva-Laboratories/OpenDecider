"""Minimal training of the `none` sink on a frozen trained model: ONLY none_x and none_lp learn (2 tensors).

    .venv/bin/python scripts/train_none.py --ckpt runs/x2b/model.pt --out runs/x2b-none/model.pt

Programmatic targets, no teacher (docs/roadmap_designs.md §1):
  answerable  : all options, target = gold
  gold removed: the gold option is dropped (other options kept), target = none
  unknowable  : question paired with an unrelated state (data/synthetic/uncertainty_train.jsonl), target = none
Questions with more than 16 options keep the gold plus 15 random distractors (one LLM row per option).
Evaluation on held-out val records (+ held-out unknowable items): accuracy on answerable items (renormalised over
real options), mean P(none) per kind, AUROC of P(none) for gold-removed / unknowable vs answerable.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from opendecider.backbone import Backbone, BackboneConfig
from opendecider.batching import Question
from opendecider.formatting import state_text
from opendecider.guards import gpu_lock, limit_gpu_memory, require_free_gb
from opendecider.models import ModelConfig, build_model

FAMS = ("clinc", "massive", "dbpedia", "arc", "arc_challenge", "snli")


def load(ckpt, sink=True):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    bb = Backbone.load(BackboneConfig(**ck["backbone_cfg"]))
    cfg = ModelConfig(**{**ck["model_cfg"], "v3_none": sink})
    m = build_model(bb, cfg)
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and {k for k in missing if not k.startswith("backbone.")} <= {"none_x", "none_lp"}, missing
    return m.to(bb.device).eval(), ck


def items(path, n, rng, fams=FAMS, max_opts=16):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r["family"] in fams]
    rng.shuffle(recs)
    out = []
    for r in recs[:n]:
        q = r["questions"][0]
        opts, lab = q["options"], q["label"]
        if len(opts) < 3:
            continue
        if len(opts) > max_opts:                     # gold + random distractors (one LLM row per option)
            keep = sorted(rng.sample([i for i in range(len(opts)) if i != lab], max_opts - 1) + [lab])
            opts, lab = [opts[i] for i in keep], keep.index(lab)
        out.append((state_text(r["state"]), q["prompt"], opts, lab, "answerable"))
        keep = [o for i, o in enumerate(opts) if i != lab]
        out.append((state_text(r["state"]), q["prompt"], keep, -1, "gold_removed"))
    return out


def unknowable(path, n, rng):
    recs = [json.loads(l) for l in open(path) if '"unknowable-' in l[:40]]
    rng.shuffle(recs)
    return [(state_text(r["state"]), r["questions"][0]["prompt"], r["questions"][0]["options"], -1, "unknowable")
            for r in recs[:n]]


EXPLICIT = ""                       # --explicit: append this text as a real option instead of the learned sink


def forward(m, batch):
    mem = m.encode_states([b[0] for b in batch])
    qs = [Question(b[1], b[2] + ([EXPLICIT] if EXPLICIT else []), "choice", state_idx=i) for i, b in enumerate(batch)]
    _, out = m.run(qs, mem)
    if EXPLICIT:                       # the explicit option sits right after each question's own options
        K = torch.tensor([len(b[2]) for b in batch], device=out.logits.device)
        p = torch.softmax(out.logits.float(), -1)
        pn = p.gather(1, K[:, None]).squeeze(1)
        rest = p.clone(); rest.scatter_(1, K[:, None], 0.0)
        logits = torch.log(torch.cat([rest[:, :-1] if rest.shape[1] > 1 else rest, pn[:, None]], 1).clamp_min(1e-12))
        tgt = torch.tensor([b[3] if b[3] >= 0 else logits.shape[1] - 1 for b in batch], device=logits.device)
        return logits, tgt
    none_col = out.logits.shape[1] - 1
    tgt = torch.tensor([b[3] if b[3] >= 0 else none_col for b in batch], device=out.logits.device)
    return out.logits, tgt


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    r = s.argsort().argsort() + 1
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(m, data, bs):
    res = {"answerable": [], "gold_removed": [], "unknowable": []}
    acc = []
    for a in range(0, len(data), bs):
        batch = data[a:a + bs]
        logits, tgt = forward(m, batch)
        p = torch.softmax(logits.float(), -1)
        for i, b in enumerate(batch):
            res[b[4]].append(float(p[i, -1]))
            if b[4] == "answerable":
                acc.append(int(p[i, :len(b[2])].argmax()) == b[3])
    out = {k: float(np.mean(v)) for k, v in res.items() if v}
    out["acc_answerable"] = float(np.mean(acc))
    out["auroc_gold_removed"] = auroc(np.array(res["gold_removed"]), np.array(res["answerable"]))
    if res["unknowable"]:
        out["auroc_unknowable"] = auroc(np.array(res["unknowable"]), np.array(res["answerable"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--out", default="runs/x2b-none/model.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--explicit", default="", help='eval only: append this option text (e.g. "none of the above")')
    ap.add_argument("--eval-out", default="runs/none-explicit/none_eval.json")
    a = ap.parse_args()
    global EXPLICIT
    EXPLICIT = a.explicit
    require_free_gb(0.5)
    limit_gpu_memory(float(os.environ.get("OPENDECIDER_EVAL_GPU_GB", "24")))
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    train = items("data/public_train/train.jsonl", 3000, rng) + unknowable("data/synthetic/uncertainty_train.jsonl", 600, rng)
    val = items("data/public_train/val.jsonl", 300, random.Random(1)) + \
        unknowable("data/synthetic/uncertainty_train.jsonl", 3000, random.Random(2))[-150:]
    rng.shuffle(train)
    if a.explicit:                                   # zero-training baseline: a real text option read by the backbone
        with gpu_lock("eval_none_explicit"):
            m, _ = load(a.ckpt, sink=False)
            res = evaluate(m, val, a.bs)
        print("explicit", json.dumps(res), flush=True)
        os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
        json.dump({"explicit": a.explicit, "ckpt": a.ckpt, "eval": res}, open(a.eval_out, "w"), indent=1)
        return
    with gpu_lock("train_none"):
        m, ck = load(a.ckpt)
        for n, p in m.named_parameters():
            p.requires_grad_(n in ("none_x", "none_lp"))
        params = [m.none_x, m.none_lp]
        before = evaluate(m, val, a.bs)
        print("before", json.dumps(before), flush=True)
        opt = torch.optim.Adam(params, lr=a.lr)
        t0 = time.time()
        for step in range(a.steps):
            batch = [train[(step * a.bs + i) % len(train)] for i in range(a.bs)]
            logits, tgt = forward(m, batch)
            loss = torch.nn.functional.cross_entropy(logits.float(), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                print(f"step {step} loss {loss.item():.4f} none_lp {m.none_lp.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        after = evaluate(m, val, a.bs)
        print("after", json.dumps(after), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ck["model_cfg"] = {**ck["model_cfg"], "v3_none": True}
    ck["state_dict"] = {**ck["state_dict"], "none_x": m.none_x.detach().cpu(), "none_lp": m.none_lp.detach().cpu()}
    ck.setdefault("extra", {})["none_training"] = {"args": vars(a), "before": before, "after": after}
    torch.save(ck, a.out)
    json.dump({"before": before, "after": after, "args": vars(a)}, open(os.path.join(os.path.dirname(a.out), "none_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
