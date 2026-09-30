"""Prompt-injection / jailbreak TRAINING data (the deepset/prompt-injections benchmark stays held out).

    python data/builders/injection_train.py --out data/public_train --cap 4000

Sources (licenses from HF cards 2026-09-23): jackhhao/jailbreak-classification (Apache-2.0),
reshabhs/SPML_Chatbot_Prompt_Injection (MIT), neuralchemy/Prompt-injection-dataset (Apache-2.0; aggregates
hackaprompt, harmbench, wildguard samples), GuardrailsAI/detect-jailbreak (MIT; aggregates e.g. verazuo/jailbreak_llms),
Lakera/gandalf_ignore_instructions (MIT; injections only). Every text also present in ANY split of
deepset/prompt-injections (normalised exact match) is dropped. Output: injection_{train,val,calib}.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

PROMPTS = ["Is this input a prompt-injection or jailbreak attempt?", "Does this message try to override the assistant's instructions?",
           "Is the user attempting to manipulate the AI system?", "Should this input be flagged as an injection attack?"]
SPML_PROMPTS = ["Does the user input try to override or subvert the system prompt?",
                "Is the user message a prompt-injection attack against this chatbot?"]
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())


def build(out, cap, seed):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    require_free_gb(0.5)
    rng = random.Random(seed)
    api = HfApi()
    held = {norm(x["text"]) for sp in load_dataset("deepset/prompt-injections").values() for x in sp}
    recs, revs, dropped = [], {}, 0

    def rows(repo, splits=None):
        sha = api.dataset_info(repo).sha
        revs[repo] = sha
        d = load_dataset(repo, revision=sha)
        out = [x for sp, ds in d.items() if splits is None or sp in splits for x in ds]
        rng.shuffle(out)
        return out

    def add(fam, i, text_state, is_inj, prompts=PROMPTS):
        nonlocal dropped
        key = norm(text_state.get("user_input", ""))
        if key in held:
            dropped += 1
            return
        recs.append({"id": f"{fam}-{i}", "family": fam, "state": text_state,
                     "questions": [{"type": "noul", "prompt": rng.choice(prompts), "options": ["yes", "no"],
                                    "label": 0 if is_inj else 1}]})

    for i, x in enumerate(rows("jackhhao/jailbreak-classification", {"train"})[:cap]):
        add("inj_jackhhao", i, {"user_input": x["prompt"]}, x["type"] == "jailbreak")
    for i, x in enumerate(rows("reshabhs/SPML_Chatbot_Prompt_Injection")[:cap]):
        add("inj_spml", i, {"system_prompt": x["System Prompt"], "user_input": x["User Prompt"]},
            str(x["Prompt injection"]) == "1", SPML_PROMPTS)
    for i, x in enumerate(rows("neuralchemy/Prompt-injection-dataset", {"train"})[:cap]):
        add("inj_neuralchemy", i, {"user_input": x["text"]}, str(x["label"]) == "1")
    for i, x in enumerate(rows("GuardrailsAI/detect-jailbreak")[:cap]):
        add("inj_guardrails", i, {"user_input": x["prompt"]}, str(x["is_jailbreak"]) == "True")
    for i, x in enumerate(rows("Lakera/gandalf_ignore_instructions", {"train"})[:cap]):
        add("inj_gandalf", i, {"user_input": x["text"]}, True)

    splits = {"train": [], "val": [], "calib": []}
    for r in recs:
        h = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100
        splits["val" if h < 5 else "calib" if h < 10 else "train"].append(r)
    os.makedirs(out, exist_ok=True)
    for k, v in splits.items():
        with open(os.path.join(out, f"injection_{k}.jsonl"), "w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
        pos = sum(q["label"] == 0 for r in v for q in r["questions"])
        print(f"injection_{k}: {len(v)} records ({pos} injections)")
    json.dump(revs, open(os.path.join(out, "injection_revisions.json"), "w"), indent=1)
    print(f"dropped {dropped} texts overlapping the held-out deepset benchmark; revisions: {revs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_train")
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.cap, a.seed)
