"""Model stitching: move a trained OpenDecider V3 (trunk + heads) onto a bigger backbone WITHOUT retraining the trunk.

The trunk only sees per-token features AFTER the layer combine (width d_A, e.g. 2048 for the 2B) through a scale-only
norm. For a new backbone B with the same tokenizer (Qwen3.5 2B -> 9B), run both on the SAME token rows, then fit
    unit(target_A) ~= W . [unit(h_B^l1); unit(h_B^l2); ...] + b        (ridge regression, closed form)
and plug W into a V3 with v3_layer_combine="stitch". Everything downstream of the combine is reused as is.

    python scripts/stitch_backbone.py targets --ckpt runs/x2b/model.pt --out /dev/shm/stitch/targets.pt --n-states 300
    python scripts/stitch_backbone.py fit --ckpt runs/x2b/model.pt --targets /dev/shm/stitch/targets.pt \
        --backbone-path models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_0.gguf --layers 8,16,24,final --out runs/x9b-stitch/model.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRAIN_FILES = ["data/public_train/train.jsonl", "data/synthetic/train.jsonl", "data/public_train/injection_train.jsonl",
               "data/public_train/knowledge_train.jsonl"]


def sample_records(n: int, seed: int = 0, max_opts: int = 32, files=None):
    """A fixed mixed sample of training states (no test data): n/len(files) per file; <= 4 questions per state and
    only questions with <= max_opts options (bounded rows per call)."""
    rng = random.Random(seed)
    files = files or TRAIN_FILES
    recs, per = [], max(1, n // len(files))
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        rng.shuffle(rows)
        got = 0
        for r in rows:
            qs = [q for q in r["questions"] if len(q["options"]) <= max_opts and q.get("label") is not None][:4]
            if qs:
                recs.append(dict(r, questions=qs)); got += 1
            if got >= per:
                break
    return recs


def calls(model, recs, per_call: int = 2):
    """Yield (questions, mem) exactly as evaluation builds them (same rows for any backbone)."""
    from opendecider.batching import Question
    from opendecider.formatting import record_state_text
    from opendecider.train import to_questions
    fmt, cap = model.row_format, model.cfg.v3_list_cap
    for i in range(0, len(recs), per_call):
        chunk = recs[i:i + per_call]
        qs = [q for j, r in enumerate(chunk) for q in to_questions(r, j, None, 10 ** 6)]
        mem = model.encode_states([record_state_text(r, fmt, cap) for r in chunk])
        yield qs, mem


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def stitch_inputs(wide, n_l, d_b):
    """raw wide features (m, n_l*d_b) -> [unit per layer ; 1] (m, n_l*d_b + 1)"""
    X = unit(wide.float().view(wide.shape[0], n_l, d_b)).flatten(1)
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)


def solve_ridge(XtX, XtY, lam):
    reg = lam * torch.eye(XtX.shape[0], dtype=XtX.dtype); reg[-1, -1] = 0      # bias unpenalised
    return torch.linalg.solve(XtX + reg, XtY)                                     # (D, d_a); last row = bias


@torch.no_grad()
def cmd_targets(a):
    from eval.evaluate import load_scorer
    model, _ = load_scorer(a.ckpt, None)
    model.eval()
    recs = sample_records(a.n_states, a.seed, files=a.files)
    Y, t0 = [], time.time()
    for k, (qs, mem) in enumerate(calls(model, recs)):
        s, _, q, _, o, _ = model._conditioned_feats(qs, mem)          # after the layer combine, before in_norm
        Y.append(torch.cat([x for x in (s, q, o) if x is not None]).half().cpu())   # s is None on state-free models
        if (k + 1) % 25 == 0:
            print(f"  {k + 1} calls, {sum(y.shape[0] for y in Y)} tokens, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"Y": Y, "recs": recs, "ckpt": a.ckpt}, a.out)
    print(f"targets: {sum(y.shape[0] for y in Y)} tokens x {Y[0].shape[1]} -> {a.out}")


@torch.no_grad()
def cmd_fit(a):
    from opendecider.backbone import Backbone, BackboneConfig
    from opendecider.models import ModelConfig, build_model
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    T = torch.load(a.targets, weights_only=False)
    layers = [int(x) if x.isdigit() else x for x in a.layers.split(",")]
    bcfg = BackboneConfig(**{**ck["backbone_cfg"], "path": a.backbone_path, "feature_layers": layers,
                             "weight_quant": a.weight_quant, "repo_id": a.repo_id, "revision": a.revision})
    bb = Backbone.load(bcfg)
    mcfg = dict(ck["model_cfg"])
    d_a = ck["state_dict"]["proj_option.weight"].shape[1]
    mcfg.update(v3_layer_combine="stitch", v3_stitch_dim=d_a, v3_vera_layers=0)
    model = build_model(bb, ModelConfig(**mcfg)).to(bb.device).eval()
    n_l, d_b = len(layers), bb.hidden_size
    D = n_l * d_b + 1
    XtX = torch.zeros(D, D, dtype=torch.float64, device="cpu")
    XtY = torch.zeros(D, d_a, dtype=torch.float64, device="cpu")
    hold_X, hold_Y = [], []
    n_calls = len(T["Y"])
    t0 = time.time()
    for k, (qs, mem) in enumerate(calls(model, T["recs"])):
        s, _, q, _, o, _ = model._conditioned_feats_raw(qs, mem)       # raw wide features (L * d_b)
        X = stitch_inputs(torch.cat([x for x in (s, q, o) if x is not None]), n_l, d_b)
        Y = T["Y"][k].float().to(X.device)
        assert Y.shape[0] == X.shape[0], f"token mismatch in call {k}: {Y.shape[0]} vs {X.shape[0]}"
        Y = unit(Y)
        if k % 10 == 9:                                                # every 10th call held out
            hold_X.append(X.half().cpu()); hold_Y.append(Y.half().cpu())
        else:
            XtX += (X.T @ X).double().cpu()
            XtY += (X.T @ Y).double().cpu()
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{n_calls} calls, {time.time() - t0:.0f}s", flush=True)
    best = None
    Hx = torch.cat(hold_X).double(); Hy = torch.cat(hold_Y).double()
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        W = solve_ridge(XtX, XtY, lam)
        P = Hx @ W
        cos = torch.nn.functional.cosine_similarity(P, Hy, dim=-1).mean().item()
        print(f"  ridge lambda={lam:g}: held-out cosine(pred, target) = {cos:.4f}", flush=True)
        if best is None or cos > best[0]:
            best = (cos, lam, W)
    cos, lam, W = best
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():                              # reuse everything with matching shape
        if k in sd and sd[k].shape == v.shape and not k.startswith("backbone."):
            sd[k] = v
    sd["stitch.weight"] = W[:-1].T.float().contiguous()
    sd["stitch.bias"] = W[-1].float().contiguous()
    reused = [k for k in ck["state_dict"] if k in sd and sd[k].shape == ck["state_dict"][k].shape]
    skipped = [k for k in ck["state_dict"] if k not in reused]
    from opendecider.checkpoint import save
    model.load_state_dict(sd, strict=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save(model, a.out, {**ck.get("extra", {}), "stitched_from": a.ckpt, "stitch_lambda": lam, "stitch_holdout_cosine": cos})
    print(f"stitched checkpoint -> {a.out}  (lambda {lam}, held-out cosine {cos:.4f}; reused {len(reused)} tensors, "
          f"re-fitted/new: stitch + {len(skipped)} skipped: {sorted({s.split('.')[0] for s in skipped})})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets"); t.add_argument("--ckpt", required=True); t.add_argument("--out", required=True)
    t.add_argument("--n-states", type=int, default=300); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--files", nargs="*", default=None, help="training files to sample states from (default: the "
                   "original corpus; pass the clean_train files for a clean-provenance stitch)")
    f = sub.add_parser("fit"); f.add_argument("--ckpt", required=True); f.add_argument("--targets", required=True)
    f.add_argument("--backbone-path", required=True); f.add_argument("--layers", default="8,16,24,final")
    f.add_argument("--out", required=True)
    f.add_argument("--weight-quant", default="int8", help="backbone weight mode the stitch is fitted on (e.g. awq)")
    f.add_argument("--repo-id", default="Qwen/Qwen3.5-9B")            # provenance of the target backbone
    f.add_argument("--revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    a = ap.parse_args()
    {"targets": cmd_targets, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
