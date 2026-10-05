"""Evaluate a checkpoint or baseline on JSONL data; metrics with bootstrap CIs (docs/SPEC.md section 7).

    python -m eval.evaluate --ckpt runs/v2/ckpt.pt --data data/synthetic/test_id.jsonl data/synthetic/test_ood.jsonl \
        --temperature-from data/synthetic/calib.jsonl
    python -m eval.evaluate --baseline label_seq --data data/synthetic/test_ood.jsonl

JSONL record: {"id", "family", "state", "questions": [{"type", "prompt", "options", "label"}], ["split"]}.
`label` is an option index (or the option string; for noul also a bool). Split = record["split"]
if present, else the data file's stem. Metrics are reported per split, per family (within split)
and per question type, raw and (if a calibration file is given) temperature-scaled (section 5.4).

GPU runs take `opendecider.guards.gpu_lock("eval")`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from typing import Sequence

import torch

from opendecider.batching import Question
from opendecider.formatting import YES_NO
from opendecider.heads import NEG, two_stage_select
from opendecider.options import MAX_OPTIONS

from .metrics import apply_temperature, fit_temperature, summarize


# ---------------------------------------------------------------------------- loading
def gpu_context(device: str, tag: str = "eval"):
    if str(device).startswith("cuda"):
        from opendecider.guards import gpu_lock, limit_gpu_memory
        limit_gpu_memory(float(os.environ.get("OPENDECIDER_EVAL_GPU_GB", "24")))   # unified memory: OOM, not a hang
        return gpu_lock(tag)
    return contextlib.nullcontext()


def load_scorer(ckpt: str | None, baseline: str | None, backbone_config: str | None = None,
                device: str = "cuda", backbone=None):
    """Returns (model_with_encode_states_and_run, checkpoint_temperature)."""
    if ckpt:
        from opendecider.checkpoint import load_model
        model, extra = load_model(ckpt, backbone=backbone, device=device)
        return model, float(extra.get("temperature", 1.0))
    from opendecider.backbone import Backbone, BackboneConfig
    from .baselines import build_baseline
    if backbone is None:
        cfg = BackboneConfig.from_yaml(backbone_config) if backbone_config and os.path.exists(backbone_config) \
            else BackboneConfig()
        cfg.device = device
        backbone = Backbone.load(cfg)
    return build_baseline(baseline, backbone), 1.0


def read_jsonl(path: str, split: str | None = None) -> list[dict]:
    split = split or os.path.splitext(os.path.basename(path))[0]
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r.setdefault("split", split)
                out.append(r)
    return out


def to_question(qd: dict, family: str = "") -> Question:
    t = qd.get("type", "choice")
    opts = list(qd.get("options") or qd.get("levels") or (YES_NO if t == "noul" else []))
    lab = qd.get("label")
    if isinstance(lab, bool):
        lab = 0 if lab else 1
    elif isinstance(lab, str):
        lab = opts.index(lab)
    return Question(qd["prompt"], opts, t, label=lab, family=family)


# ---------------------------------------------------------------------------- inference
@torch.no_grad()
def _logits_for(model, qs: list[Question], mem, max_q: int) -> list[list[float]]:
    small = [i for i, q in enumerate(qs) if len(q.options) <= MAX_OPTIONS or not hasattr(model, "slots")]
    small_set = set(small)
    large = [i for i in range(len(qs)) if i not in small_set]
    res: list = [None] * len(qs)
    sigmoid_noul = getattr(getattr(model, "cfg", None), "noul_mode", "choice") == "sigmoid"
    for s in range(0, len(small), max_q):
        idx = small[s:s + max_q]
        _, out = model.run([qs[i] for i in idx], mem)
        for j, i in enumerate(idx):
            K = len(qs[i].options)
            if sigmoid_noul and qs[i].type == "noul" and out.noul_logit is not None:
                res[i] = [float(out.noul_logit[j]), 0.0]      # softmax([z, 0]) = sigmoid(z)
            else:
                res[i] = out.logits[j, :K].double().tolist()
    for i in large:                                           # two-stage (O2), as in Decider
        q = qs[i]
        rel = []
        for s in range(0, len(q.options), MAX_OPTIONS):
            chunk = Question(q.prompt, q.options[s:s + MAX_OPTIONS], q.type)
            _, out = model.run([chunk], mem)
            rel.append(out.relevance[0, : len(chunk.options)])
        keep = two_stage_select(torch.cat(rel)).tolist()
        _, out = model.run([Question(q.prompt, [q.options[k] for k in keep], q.type)], mem)
        full = [float(NEG)] * len(q.options)
        for j, k in enumerate(keep):
            full[k] = float(out.logits[0, j])
        res[i] = full
    return res


def collect(model, records: Sequence[dict], max_q: int = 64, log_every: int = 0) -> list[dict]:
    """One row per question: {id, split, family, type, K, label, logits}."""
    rows = []
    t0 = time.perf_counter()
    for r_i, rec in enumerate(records):
        fam = rec.get("family", "")
        qs = [to_question(qd, fam) for qd in rec["questions"]]
        from opendecider.formatting import record_state_text
        mem = model.encode_states([record_state_text(rec, getattr(model, "row_format", "list"),
                                                     getattr(getattr(model, "cfg", None), "v3_list_cap", 16))])
        logits = _logits_for(model, qs, mem, max_q)
        for j, (q, lg) in enumerate(zip(qs, logits)):
            if q.label is None:
                continue
            rows.append({"id": f"{rec.get('id', r_i)}:{j}", "split": rec["split"], "family": fam,
                         "type": q.type, "K": len(q.options), "label": int(q.label), "logits": lg,
                         "options": list(q.options)})
        if log_every and (r_i + 1) % log_every == 0:
            print(f"  {r_i + 1}/{len(records)} states, {time.perf_counter() - t0:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------- metrics
def metrics_for(rows: list[dict], T: float | None, n_boot: int, seed: int, T_type: dict | None = None) -> dict:
    def summ(rs, temp):
        return summarize(apply_temperature([r["logits"] for r in rs], temp), [r["label"] for r in rs],
                         n_boot=n_boot, seed=seed)

    def groups(key):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        return dict(sorted(g.items()))

    out = {"n_questions": len(rows), "raw": summ(rows, 1.0),
           "by_family": {k: summ(v, 1.0) for k, v in groups("family").items()},
           "by_type": {k: summ(v, 1.0) for k, v in groups("type").items()}}
    if T is not None:
        out["temperature"] = T
        out["scaled"] = summ(rows, T)
        out["by_family_scaled"] = {k: summ(v, T) for k, v in groups("family").items()}
    if T_type:
        def summ_typed(rs):
            lg = [x for t, grp in groupby_type(rs).items() for x in apply_temperature([r["logits"] for r in grp], T_type.get(t, T or 1.0))]
            lb = [r["label"] for t, grp in groupby_type(rs).items() for r in grp]
            return summarize(lg, lb, n_boot=n_boot, seed=seed)
        out["temperature_by_type"] = T_type
        out["scaled_per_type"] = summ_typed(rows)
        out["by_type_scaled_per_type"] = {k: summ_typed(v) for k, v in groups("type").items()}
    return out


def groupby_type(rs):
    g = defaultdict(list)
    for r in rs:
        g[r["type"]].append(r)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--baseline", choices=["label_first", "label_seq"])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--temperature-from", help="held-out calibration JSONL; fit a single T on its NLL")
    ap.add_argument("--temperature", type=float, help="use this T instead of fitting")
    ap.add_argument("--backbone-config", default="configs/backbone.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name")
    ap.add_argument("--max-questions-per-call", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max records per file (debug)")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--serving", action="store_true",
                    help="evaluate the serving path (decider.prepare_inference_: question cache, tuned int8 kernels)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    name = a.name or (os.path.splitext(os.path.basename(a.ckpt))[0] if a.ckpt else a.baseline)
    splits: dict[str, list[dict]] = defaultdict(list)
    for path in a.data:
        recs = read_jsonl(path)
        if a.limit:
            recs = recs[: a.limit]
        for r in recs:
            splits[r["split"]].append(r)

    t0 = time.perf_counter()
    with gpu_context(a.device, "eval"):
        model, ckpt_T = load_scorer(a.ckpt, a.baseline, a.backbone_config, a.device)
        if a.serving and a.ckpt:
            from opendecider.decider import prepare_inference_
            prepare_inference_(model)
        model.eval()
        T, calib_info, T_type = None, None, None
        if a.temperature is not None:
            T = a.temperature
        elif a.temperature_from:
            crecs = read_jsonl(a.temperature_from, "calib")
            crecs = crecs[: a.limit] if a.limit else crecs          # --limit applies to calibration too
            crow = collect(model, crecs, a.max_questions_per_call)
            T = fit_temperature([r["logits"] for r in crow], [r["label"] for r in crow])
            T_type = {t: fit_temperature([r["logits"] for r in g], [r["label"] for r in g])
                      for t, g in groupby_type(crow).items() if len(g) >= 30}      # per question type (choice/noul/score)
            calib_info = {"file": a.temperature_from, "n_questions": len(crow), "fitted_T": T, "fitted_T_by_type": T_type}
        rows = {s: collect(model, recs, a.max_questions_per_call, log_every=200) for s, recs in splits.items()}
    elapsed = time.perf_counter() - t0

    from opendecider.guards import require_free_gb
    require_free_gb(0.05)
    out_dir = os.path.join(a.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"name": name, "ckpt": a.ckpt, "baseline": a.baseline, "data": a.data, "seed": a.seed,
            "n_boot": a.n_boot, "device": a.device, "checkpoint_temperature": ckpt_T, "calibration": calib_info,
            "model_cfg": getattr(getattr(model, "cfg", None), "__dict__", None), "elapsed_s": elapsed}
    for split, rs in rows.items():
        res = {"meta": meta, "split": split, **metrics_for(rs, T, a.n_boot, a.seed, T_type)}
        path = os.path.join(out_dir, f"eval_{split}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)
        m = res["raw"]
        line = f"[{split}] n={m['n']} acc={m['accuracy']['value']:.4f} [{m['accuracy']['lo']:.4f},{m['accuracy']['hi']:.4f}] " \
               f"nll={m['nll']['value']:.4f} ece={m['ece']['value']:.4f}"
        if "scaled" in res:
            line += f" | T={T:.3f} ece_scaled={res['scaled']['ece']['value']:.4f}"
        print(line, "->", path)
        if a.save_preds:
            with open(os.path.join(out_dir, f"preds_{split}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
