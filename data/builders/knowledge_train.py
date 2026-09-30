"""Knowledge / commonsense multiple-choice TRAINING data (targets the OpenBookQA / CommonsenseQA gap).

    python data/builders/knowledge_train.py --out data/public_train --cap 8000

Sources (licenses from HF cards 2026-09-24): allenai/qasc (CC-BY-4.0; facts withheld: they reveal the answer),
allenai/cosmos_qa (CC-BY-4.0, via parquet branch if the script repo fails), allenai/quartz (CC-BY-4.0),
derek-thomas/ScienceQA (CC-BY-SA-4.0; text-only items), ChilleD/StrategyQA (MIT; yes/no).
Held-out benchmarks stay out: every question also present in OpenBookQA or CommonsenseQA (normalised exact match of
the question text) is dropped. Output: knowledge_{train,val,calib}.jsonl.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
MC = ["Which option correctly answers the question?", "Pick the right answer.", "Which answer is correct?"]


def build(out, cap, seed):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    require_free_gb(1.5)
    rng = random.Random(seed)
    api = HfApi()
    held = {norm(x["question_stem"]) for x in load_dataset("allenai/openbookqa", "main", split="test")}
    held |= {norm(x["question"]) for x in load_dataset("tau/commonsense_qa", split="validation")}
    recs, revs, dropped = [], {}, 0

    def load(repo, split="train"):
        sha = api.dataset_info(repo).sha
        try:
            ds = load_dataset(repo, split=split, revision=sha)
            revs[repo] = sha
        except RuntimeError as e:
            if "scripts are no longer supported" not in str(e):
                raise
            ds = load_dataset(repo, split=split, revision="refs/convert/parquet")
            revs[repo] = f"{sha} (refs/convert/parquet)"
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return [ds[i] for i in idx[:cap]]

    def mc(fam, i, state, question_text, choices, lab):
        nonlocal dropped
        if norm(question_text) in held:
            dropped += 1
            return
        order = list(range(len(choices)))
        rng.shuffle(order)
        recs.append({"id": f"{fam}-{i}", "family": fam, "state": state,
                     "questions": [{"type": "choice", "prompt": rng.choice(MC),
                                    "options": [choices[j] for j in order], "label": order.index(lab)}]})

    for i, x in enumerate(load("allenai/qasc")):
        ch = x["choices"]
        mc("qasc", i, {"question": x["question"]}, x["question"], list(ch["text"]), ch["label"].index(x["answerKey"]))
    try:
        for i, x in enumerate(load("allenai/cosmos_qa")):
            ans = [x[f"answer{k}"] for k in range(4)]
            mc("cosmos_qa", i, {"context": x["context"], "question": x["question"]}, x["question"], ans, int(x["label"]))
    except Exception as e:
        print(f"SKIP cosmos_qa: {type(e).__name__}: {str(e)[:120]}")
    for i, x in enumerate(load("allenai/quartz")):
        ch = x["choices"]
        mc("quartz", i, {"passage": x["para"], "question": x["question"]}, x["question"], list(ch["text"]),
           ch["label"].index(x["answerKey"]))
    k = 0
    for x in load("derek-thomas/ScienceQA"):
        if x.get("image") is not None:              # text-only items: the model never sees images
            continue
        choices = x["choices"] if isinstance(x["choices"], list) else ast.literal_eval(x["choices"])
        state = {"question": x["question"]} | ({"context": x["hint"]} if x.get("hint") else {})
        mc("scienceqa", k, state, x["question"], list(choices), int(x["answer"]))
        k += 1
    for i, x in enumerate(load("ChilleD/StrategyQA")):
        if norm(x["question"]) in held:
            dropped += 1
            continue
        yes = str(x["answer"]).lower() in ("true", "1", "yes")
        recs.append({"id": f"strategyqa-{i}", "family": "strategyqa", "state": {"question": x["question"]},
                     "questions": [{"type": "noul", "prompt": "Is the answer to the question yes?",
                                    "options": ["yes", "no"], "label": 0 if yes else 1}]})

    splits = {"train": [], "val": [], "calib": []}
    for r in recs:
        h = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100
        splits["val" if h < 5 else "calib" if h < 10 else "train"].append(r)
    os.makedirs(out, exist_ok=True)
    for kk, v in splits.items():
        with open(os.path.join(out, f"knowledge_{kk}.jsonl"), "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
    fams = {}
    for r in recs:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print({kk: len(v) for kk, v in splits.items()}, fams, f"dropped {dropped} overlapping held-out questions")
    json.dump(revs, open(os.path.join(out, "knowledge_revisions.json"), "w"), indent=1)
    print("revisions:", revs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_train")
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.cap, a.seed)
