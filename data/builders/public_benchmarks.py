"""Build held-out public benchmark eval files (never used for training).

    python data/builders/public_benchmarks.py --out data/public

Mirrors the setups of independent Jev tests where documented:
- banking77 77-way (all test items, capped) and 8-way (8 intents x 20 items), Choice
- PubMedQA pqa_labeled, yes/no items only, Noul
- OpenBookQA main/test, Choice (4)
- CommonsenseQA validation, Choice (5)
- deepset/prompt-injections test, Noul
Output JSONL rows: {"id","family","split","state","questions":[{"type","prompt","options","label"}]}.
Revisions are pinned (commit sha recorded in data/public/revisions.json). Licenses: data/MANIFEST.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

SOURCES = {  # repo, config, split
    "banking77": ("PolyAI/banking77", None, "test"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),     # the labelled set only has a "train" split
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "commonsenseqa": ("tau/commonsense_qa", None, "validation"),
    "prompt_injections": ("deepset/prompt-injections", None, "test"),
}


B77_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"      # PolyAI-LDN/task-specific-datasets, CC-BY-4.0
B77_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{B77_COMMIT}/banking_data/test.csv"


def load_banking77():
    """HF repo is a legacy loading script over this CSV; read the CSV at a pinned commit instead."""
    import csv
    import io
    import urllib.request
    text = urllib.request.urlopen(B77_URL, timeout=60).read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    names = sorted({r["category"] for r in rows})
    items = [{"text": r["text"], "label": names.index(r["category"])} for r in rows]
    return items, names, f"github PolyAI-LDN/task-specific-datasets@{B77_COMMIT}"


def load(name):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    repo, cfg, split = SOURCES[name]
    api = HfApi()
    sha = api.dataset_info(repo).sha
    try:
        ds = load_dataset(repo, cfg, split=split, revision=sha)
        return ds, sha
    except RuntimeError as e:                       # legacy script dataset -> HF's auto-converted parquet branch
        if "scripts are no longer supported" not in str(e):
            raise
        refs = api.list_repo_refs(repo, repo_type="dataset")
        conv = next(c.target_commit for c in refs.converts if c.name == "parquet")
        ds = load_dataset(repo, cfg or "default", split=split, revision=conv)
        return ds, f"{sha} (parquet {conv})"


def human(label: str) -> str:
    return label.replace("_", " ")


def build(out: str, cap: int, seed: int):
    require_free_gb(0.5)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    revs = {}

    items, raw_names, revs["banking77"] = load_banking77()
    names = [human(n) for n in raw_names]
    rng.shuffle(items)
    rows = [{"id": f"b77-{i}", "family": "banking77_77way", "split": "banking77_77way",
             "state": {"customer_message": x["text"]},
             "questions": [{"type": "choice", "prompt": "Which banking intent does the customer's message express?",
                            "options": names, "label": x["label"]}]} for i, x in enumerate(items[:cap])]
    write(out, "banking77_77way", rows)
    eight = sorted(rng.sample(range(len(names)), 8))
    rows = []
    for k, lab in enumerate(eight):
        sel = [x for x in items if x["label"] == lab][:20]
        for j, x in enumerate(sel):
            rows.append({"id": f"b8-{k}-{j}", "family": "banking77_8way", "split": "banking77_8way",
                         "state": {"customer_message": x["text"]},
                         "questions": [{"type": "choice", "prompt": "Which banking intent does the customer's message express?",
                                        "options": [names[e] for e in eight], "label": eight.index(lab)}]})
    write(out, "banking77_8way", rows)

    ds, revs["pubmedqa"] = load("pubmedqa")
    rows = []
    for x in ds:
        if x["final_decision"] not in ("yes", "no"):
            continue
        ctx = " ".join(x["context"]["contexts"])
        rows.append({"id": f"pmq-{x['pubid']}", "family": "pubmedqa", "split": "pubmedqa",
                     "state": {"abstract": ctx},
                     "questions": [{"type": "noul", "prompt": x["question"], "options": ["yes", "no"],
                                    "label": 0 if x["final_decision"] == "yes" else 1}]})
    write(out, "pubmedqa", rows[:cap])

    ds, revs["openbookqa"] = load("openbookqa")
    rows = []
    for x in ds:
        lab = x["choices"]["label"].index(x["answerKey"])
        rows.append({"id": f"obqa-{x['id']}", "family": "openbookqa", "split": "openbookqa",
                     "state": {"question": x["question_stem"]},
                     "questions": [{"type": "choice", "prompt": "Which option correctly answers the question?",
                                    "options": list(x["choices"]["text"]), "label": lab}]})
    write(out, "openbookqa", rows[:cap])

    ds, revs["commonsenseqa"] = load("commonsenseqa")
    rows = []
    for x in ds:
        lab = x["choices"]["label"].index(x["answerKey"])
        rows.append({"id": f"csqa-{x['id']}", "family": "commonsenseqa", "split": "commonsenseqa",
                     "state": {"question": x["question"]},
                     "questions": [{"type": "choice", "prompt": "Which option best answers the question?",
                                    "options": list(x["choices"]["text"]), "label": lab}]})
    write(out, "commonsenseqa", rows[:cap])

    ds, revs["prompt_injections"] = load("prompt_injections")
    rows = [{"id": f"pi-{i}", "family": "prompt_injections", "split": "prompt_injections",
             "state": {"user_input": x["text"]},
             "questions": [{"type": "noul", "prompt": "Is this input a prompt-injection attempt?",
                            "options": ["yes", "no"], "label": 0 if x["label"] == 1 else 1}]} for i, x in enumerate(ds)]
    write(out, "prompt_injections", rows[:cap])
    json.dump(revs, open(os.path.join(out, "revisions.json"), "w"), indent=1)
    print("revisions:", revs)


def write(out, name, rows):
    with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.cap, a.seed)
