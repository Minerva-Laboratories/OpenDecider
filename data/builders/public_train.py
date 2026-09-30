"""Build the permissive public TRAINING corpus (mixed with synthetic data). Benchmarks stay out.

    python data/builders/public_train.py --out data/public_train --cap 8000

Sources (licenses verified 2026-09-23, see data/MANIFEST.md): clinc_oos (CC-BY-3.0), AmazonScience/massive en-US
(CC-BY-4.0), boolq (CC-BY-SA-3.0), snli (CC-BY-SA-4.0), ai2_arc (CC-BY-SA-4.0), winogrande (Apache-2.0),
dbpedia_14 (CC-BY-SA-3.0). EXCLUDED as held-out benchmarks: banking77, PubMedQA, OpenBookQA, CommonsenseQA,
prompt-injections. Note: ARC shares a domain with OpenBookQA and CLINC/MASSIVE with Banking77 (dataset-level,
not domain-level, holdout).
Augmentations (docs/SPEC.md §5.1): intent/topic questions use random option subsets (2..40, always containing the
answer) most of the time and the full label set otherwise; option order shuffled; several prompt templates.
Output: {train,val,calib}.jsonl in the synthetic schema, split deterministically by hash of the record id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

INTENT_PROMPTS = ["Which intent does the user's message express?", "What is the user trying to do?",
                  "Classify the request:", "Which of these best describes the user's intent?"]
TOPIC_PROMPTS = ["Which category does this text belong to?", "What is the topic of the text?",
                 "Classify the article's subject:"]
NLI_PROMPTS = ["How does the hypothesis relate to the premise?", "Does the premise support the hypothesis?",
               "Classify the relationship between premise and hypothesis:"]
MC_PROMPTS = ["Which option correctly answers the question?", "Pick the right answer.", "Which answer is correct?"]


def load(repo, cfg, split):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    sha = api.dataset_info(repo).sha
    try:
        return load_dataset(repo, cfg, split=split, revision=sha), sha
    except RuntimeError as e:
        if "scripts are no longer supported" not in str(e):
            raise
        # legacy script repos: use HF's parquet conversion branch if it exists
        ds = load_dataset(repo, cfg, split=split, revision="refs/convert/parquet")
        return ds, f"{sha} (refs/convert/parquet)"


def label_question(rng, names, lab, prompts, full_p=0.3, kmin=2, kmax=40):
    """Choice over a random subset of labels containing the answer (or the full set with prob full_p)."""
    if len(names) <= kmax or rng.random() < full_p:
        opts = list(range(len(names)))
    else:
        k = rng.randint(kmin, kmax)
        opts = rng.sample([i for i in range(len(names)) if i != lab], k - 1) + [lab]
    rng.shuffle(opts)
    return {"type": "choice", "prompt": rng.choice(prompts), "options": [names[i] for i in opts],
            "label": opts.index(lab)}


def mc_question(rng, choices, lab, prompts):
    order = list(range(len(choices)))
    rng.shuffle(order)
    return {"type": "choice", "prompt": rng.choice(prompts), "options": [choices[i] for i in order],
            "label": order.index(lab)}


def build(out, cap, seed):
    require_free_gb(1.0)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    recs, revs = [], {}

    def add(family, i, state, questions):
        recs.append({"id": f"{family}-{i}", "family": family, "state": state, "questions": questions})

    def take(ds):
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return [ds[i] for i in idx[:cap]]

    specs = [
        ("clinc", "clinc/clinc_oos", "plus", "train"),
        ("massive", "AmazonScience/massive", "default", "train"),    # parquet branch: all locales -> en-US
        ("boolq", "google/boolq", None, "train"),
        ("snli", "stanfordnlp/snli", None, "train"),
        ("arc", "allenai/ai2_arc", "ARC-Easy", "train"),
        ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train"),
        ("winogrande", "allenai/winogrande", "winogrande_xl", "train"),
        ("dbpedia", "fancyzhx/dbpedia_14", None, "train"),
    ]
    for fam, repo, cfg, split in specs:
        try:
            ds, revs[fam] = load(repo, cfg, split)
        except Exception as e:                       # report and continue; nothing silently substituted
            print(f"SKIP {fam}: {type(e).__name__}: {str(e)[:160]}")
            continue
        if fam == "massive":
            ds = ds.filter(lambda x: x["locale"] == "en-US", desc=None)
        rows = take(ds)
        if fam == "clinc":
            names = [n.replace("_", " ") for n in ds.features["intent"].names]
            for i, x in enumerate(rows):
                add(fam, i, {"user_message": x["text"]}, [label_question(rng, names, x["intent"], INTENT_PROMPTS)])
        elif fam == "massive":
            names = [n.replace("_", " ") for n in ds.features["intent"].names]
            for i, x in enumerate(rows):
                add(fam, i, {"user_message": x["utt"]}, [label_question(rng, names, x["intent"], INTENT_PROMPTS)])
        elif fam == "boolq":
            for i, x in enumerate(rows):
                add(fam, i, {"passage": x["passage"]},
                    [{"type": "noul", "prompt": x["question"].rstrip("?") + "?", "options": ["yes", "no"],
                      "label": 0 if x["answer"] else 1}])
        elif fam == "snli":
            names = ["entailment", "neutral", "contradiction"]
            k = 0
            for x in rows:
                if x["label"] not in (0, 1, 2):
                    continue
                add(fam, k, {"premise": x["premise"], "hypothesis": x["hypothesis"]},
                    [mc_question(rng, names, x["label"], NLI_PROMPTS)])
                k += 1
        elif fam.startswith("arc"):
            for i, x in enumerate(rows):
                if x["answerKey"] not in x["choices"]["label"]:
                    continue
                add(fam, i, {"question": x["question"]},
                    [mc_question(rng, list(x["choices"]["text"]), x["choices"]["label"].index(x["answerKey"]), MC_PROMPTS)])
        elif fam == "winogrande":
            for i, x in enumerate(rows):
                if x["answer"] not in ("1", "2"):
                    continue
                add(fam, i, {"sentence": x["sentence"]},
                    [mc_question(rng, [x["option1"], x["option2"]], int(x["answer"]) - 1,
                                 ["Which option correctly fills the blank (_)?", "Who or what does the blank refer to?"])])
        elif fam == "dbpedia":
            names = [n.replace("_", " ") for n in ds.features["label"].names]
            for i, x in enumerate(rows):
                add(fam, i, {"title": x["title"], "content": x["content"]},
                    [label_question(rng, names, x["label"], TOPIC_PROMPTS, full_p=0.5)])
        print(f"{fam}: {sum(r['family'] == fam for r in recs)} records")

    splits = {"train": [], "val": [], "calib": []}
    for r in recs:
        h = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100
        splits["val" if h < 5 else "calib" if h < 10 else "train"].append(r)
    for k, v in splits.items():
        with open(os.path.join(out, f"{k}.jsonl"), "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
        print(f"{k}: {len(v)} records")
    json.dump(revs, open(os.path.join(out, "revisions.json"), "w"), indent=1)
    print("revisions:", revs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_train")
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.cap, a.seed)
