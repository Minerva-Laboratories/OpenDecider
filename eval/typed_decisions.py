"""typed-decisions benchmark (LocalLLaMA/typed-decisions, Apache-2.0, rev f7a2487edd7a): 400 test cases x 5 typed
questions, gold = mean of 3 teacher samples (full distributions). Zero-shot ("generalist" mode): no model here has
seen these workflows or schemas.

    .venv/bin/python -m eval.typed_decisions llm   --split test  --name td-9b        # LLM letter log-probs (server)
    .venv/bin/python -m eval.typed_decisions od    --split test  --name td-x2b  --ckpt runs/x2b/model.pt
    .venv/bin/python -m eval.typed_decisions score --a runs/td-x2b --b runs/td-9b [--fit-on runs/td-x2b-train,runs/td-9b-train]

Every question becomes a choice over its options, rendered "name: description" (score levels "k: description").
Metrics (same terms as the dataset leaderboard): accuracy (argmax vs gold label), KL(gold || pred), Brier vs gold
distribution, plus log loss vs gold and ECE vs the gold label; 95% bootstrap CIs over cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REV = "f7a2487edd7a"


def load_cases(split: str, limit: int = 0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet", repo_type="dataset",
                        revision=REV, local_dir="data/typed_decisions")
    rows = pq.read_table(p).to_pylist()
    return rows[:limit] if limit else rows


def questions_of(row):
    """-> list of (qname, instructions, option_names, option_texts, gold_probs aligned to option_names, gold_label)."""
    qs, gold = json.loads(row["questions"]), json.loads(row["gold"])
    out = []
    for name, q in qs.items():
        crit = q.get("criteria")
        if q["type"] == "score":
            names = [str(i) for i in range(len(crit))]
            texts = [f"{i}: {c}" for i, c in enumerate(crit)]
        elif q["type"] == "noul":
            crit = crit or {"true": "Yes.", "false": "No."}
            names = ["true", "false"]
            texts = [f"{n}: {crit.get(n, n)}" for n in names]
        else:
            names = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            texts = [f"{n}: {crit[n]}" if isinstance(crit, dict) else n for n in names]
        g = gold[name]["probabilities"]
        gp = np.array([float(g.get(n, 0.0)) for n in names]); gp = gp / gp.sum()
        instr = q.get("instructions", "")
        if q["type"] == "noul":                    # noul instructions are statements: ask whether it holds
            instr = f"Is the following statement true? {instr}"
        out.append((name, instr, names, texts, gp, names.index(str(gold[name]["label"]))))
    return out


def run_llm(a):
    from .zeroshot_llm import LETTERS, SYSTEM
    import urllib.request
    key = open(os.path.expanduser("~/.config/qwen-server/api_key")).read().strip()
    rows, preds = load_cases(a.split, a.limit), []
    for r in rows:
        for qname, instr, names, texts, gp, lab in questions_of(r):
            opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
            user = (f"{r['state']}\n\nQuestion: {instr}\nOptions:\n{opts}\n\n"
                    f"Answer with a single letter ({LETTERS[0]}-{LETTERS[len(texts) - 1]}).")
            body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "cache_prompt": True}
            req = urllib.request.Request(a.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            tops = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            lp = {}
            for t in tops:
                tok = t["token"].strip()
                if len(tok) == 1 and tok in LETTERS[:len(texts)]:
                    lp[tok] = max(lp.get(tok, -1e9), t["logprob"])
            floor = min(t["logprob"] for t in tops) - 2.0
            raw = np.array([lp.get(LETTERS[i], floor) for i in range(len(texts))])
            logp = raw - np.logaddexp.reduce(raw)
            preds.append({"id": f"{r['id']}:{qname}", "workflow": r["workflow"], "label": lab, "gold": gp.tolist(),
                          "logits": logp.tolist(), "options": names})
    save(a.name, preds)


def run_od(a):
    sys.path.insert(0, "src")
    from .evaluate import load_scorer, _logits_for
    from opendecider.batching import Question
    from opendecider.formatting import record_state_text
    model, _ = load_scorer(a.ckpt, None)
    model.eval()
    rows, preds = load_cases(a.split, a.limit), []
    fmt, cap = getattr(model, "row_format", "list"), getattr(getattr(model, "cfg", None), "v3_list_cap", 16)
    import torch
    with torch.no_grad():
        for i, r in enumerate(rows):
            items = questions_of(r)
            qs = [Question(instr, texts, "choice", lab) for _, instr, _, texts, _, lab in items]
            mem = model.encode_states([record_state_text({"state": r["state"], "questions": []}, fmt, cap)])
            logits = _logits_for(model, qs, mem, 64)
            for (qname, _, names, _, gp, lab), lg in zip(items, logits):
                z = np.array(lg[: len(names)], dtype=np.float64)
                preds.append({"id": f"{r['id']}:{qname}", "workflow": r["workflow"], "label": lab, "gold": gp.tolist(),
                              "logits": (z - np.logaddexp.reduce(z)).tolist(), "options": names})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)} cases", flush=True)
    save(a.name, preds)


def save(name, preds):
    os.makedirs(os.path.join("runs", name), exist_ok=True)
    with open(os.path.join("runs", name, "preds_typed_decisions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print_metrics(name, preds)


def metrics(P, G, y, cases, n_boot=1000, seed=0):
    """P, G: lists of prob vectors (pred, gold); y: gold label idx; cases: case id per decision (bootstrap unit)."""
    eps = 1e-9
    acc = np.array([float(np.argmax(p) == t) for p, t in zip(P, y)])
    kl = np.array([float(np.sum(g * (np.log(g + eps) - np.log(p + eps)))) for p, g in zip(P, G)])
    brier = np.array([float(np.sum((p - g) ** 2)) for p, g in zip(P, G)])
    ll = np.array([float(-np.sum(g * np.log(p + eps))) for p, g in zip(P, G)])
    conf = np.array([float(np.max(p)) for p in P])
    uc = sorted(set(cases)); ci = {c: i for i, c in enumerate(uc)}
    ix = np.array([ci[c] for c in cases])

    def ece(sel):
        o = np.argsort(conf[sel]); bins = np.array_split(o, 15)
        return float(sum(len(b) / len(o) * abs(acc[sel][b].mean() - conf[sel][b].mean()) for b in bins if len(b)))
    point = {"accuracy": acc.mean(), "kl": kl.mean(), "brier": brier.mean(), "log_loss": ll.mean(), "ece": ece(np.arange(len(acc)))}
    rng = np.random.default_rng(seed); boots = {k: [] for k in point}
    members = [np.where(ix == i)[0] for i in range(len(uc))]
    for _ in range(n_boot):
        sel = np.concatenate([members[i] for i in rng.integers(0, len(uc), len(uc))])
        for k, v in (("accuracy", acc), ("kl", kl), ("brier", brier), ("log_loss", ll)):
            boots[k].append(v[sel].mean())
        boots["ece"].append(ece(sel))
    return {k: (float(point[k]), float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))) for k in point}


def print_metrics(name, preds, extra=""):
    P = [np.exp(np.array(p["logits"])) for p in preds]
    m = metrics(P, [np.array(p["gold"]) for p in preds], [p["label"] for p in preds], [p["id"].split(":")[0] for p in preds])
    print(f"{name}{extra}: " + "  ".join(f"{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for k, v in m.items()), flush=True)
    return m


def load_preds(d):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(d, "preds_typed_decisions.jsonl"))}


def score(a):
    """Per-source metrics, per-workflow accuracy, and the log-linear ensemble p ∝ exp(wa log pa + wb log pb) with the
    two scalars fitted by KL to the gold distributions on the TRAIN split (--fit-on), then applied to test."""
    import torch
    A, B = load_preds(a.a), load_preds(a.b)
    ids = sorted(set(A) & set(B))
    res = {"A": print_metrics(os.path.basename(a.a), [A[i] for i in ids]),
           "B": print_metrics(os.path.basename(a.b), [B[i] for i in ids])}
    for wf in sorted({A[i]["workflow"] for i in ids}):
        sub = [i for i in ids if A[i]["workflow"] == wf]
        acc = lambda S: np.mean([np.argmax(S[i]["logits"]) == S[i]["label"] for i in sub])
        print(f"   {wf:28s} A {acc(A):.3f}  B {acc(B):.3f}  (n={len(sub)})")
    if a.fit_on:
        fa, fb = a.fit_on.split(",")
        FA, FB = load_preds(fa), load_preds(fb)
        fids = sorted(set(FA) & set(FB))
        w = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
        data = [(torch.tensor(FA[i]["logits"]), torch.tensor(FB[i]["logits"]), torch.tensor(FA[i]["gold"])) for i in fids]

        def closure():
            opt.zero_grad()
            loss = sum(-(g * torch.log_softmax(torch.exp(w[0]) * la + torch.exp(w[1]) * lb, -1)).sum() for la, lb, g in data) / len(data)
            loss.backward()
            return loss
        opt.step(closure)
        wa, wb = [float(x) for x in torch.exp(w.detach())]
        ens = []
        for i in ids:
            z = wa * np.array(A[i]["logits"]) + wb * np.array(B[i]["logits"])
            ens.append({**A[i], "logits": (z - np.logaddexp.reduce(z)).tolist()})
        res["ensemble"] = print_metrics("ensemble", ens, f" (w_a={wa:.2f}, w_b={wb:.2f}, fitted on {len(fids)} train decisions)")
    if a.name:
        os.makedirs(os.path.join("runs", a.name), exist_ok=True)
        json.dump(res, open(os.path.join("runs", a.name, "typed_decisions_scores.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["llm", "od", "score"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--fit-on", default="")
    a = ap.parse_args(argv)
    {"llm": run_llm, "od": run_od, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
