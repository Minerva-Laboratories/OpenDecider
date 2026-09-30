"""Batch-dependence probe data (eval only, NO training): does a comparative batch shift one item's probability?

    python data/builders/batch_dependence.py          # -> data/public_context/batchdep_{items,schema}.jsonl

Each prompt_injections test item is scored ALONE and inside a batch of 7 other test items whose share of
positives (injections) is ~10% (1/7), ~50% (3/7 or 4/7) or ~90% (6/7). The target keeps the same slot and the same
question in every condition, so only the batch composition changes. Two renderings:
  items  : {"items": {"item_1": ..., ...}}, "Regarding item_i: ..."  (the trained batch format)
  schema : the other items as schema-consistent cases with an EMPTY answer, then the target (no labels)
A calibrated, target-isolated model should give the same P(yes) in every condition. Analysis: eval/batch_dependence.py.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from context_data import item_text, q_about  # noqa: E402

LEVELS = {"alone": None, "p10": 1, "p50": None, "p90": 6}   # positives among the 7 others (p50: 3 or 4 alternating)


def main(src="data/public/prompt_injections.jsonl", out="data/public_context", seed=0):
    rng = random.Random(seed)
    items = [json.loads(l) for l in open(src)]
    items = [{"state": {"user_input": r["state"]["user_input"]}, "questions": r["questions"]} for r in items]
    pos = [i for i, r in enumerate(items) if r["questions"][0]["label"] == 0]      # option 0 = "yes" (injection)
    neg = [i for i, r in enumerate(items) if r["questions"][0]["label"] != 0]
    recs = {"items": [], "schema": []}
    for i, t in enumerate(items):
        q = t["questions"][0]
        slot = rng.randint(0, 7)
        P, N = [j for j in pos if j != i], [j for j in neg if j != i]
        for lev in LEVELS:
            if lev == "alone":
                others = []
            else:
                k = LEVELS[lev] if lev != "p50" else 3 + i % 2
                others = [items[j] for j in rng.sample(P, k) + rng.sample(N, 7 - k)]
                rng.shuffle(others)
            base = {"family": "prompt_injections", "split": f"batchdep_{lev}", "level": lev}
            if others:
                group = others[:slot] + [t] + others[slot:]
                st = {"items": {f"item_{j + 1}": item_text(x) for j, x in enumerate(group)}}
                qq = q_about(q, f"item_{slot + 1}")
            else:
                st, qq = {"items": {"item_1": item_text(t)}}, q_about(q, "item_1")
            recs["items"].append(dict(base, id=f"bd-{i}", state=st, questions=[qq]))
            recs["schema"].append(dict(base, id=f"bd-{i}", state=t["state"], questions=[q],
                                       examples=[{"state": x["state"], "answers": [None]} for x in others]))
    for fmt, rs in recs.items():
        with open(f"{out}/batchdep_{fmt}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(f"batchdep_{fmt}: {len(rs)} records ({len(items)} items x {len(LEVELS)} conditions)")


if __name__ == "__main__":
    main()
