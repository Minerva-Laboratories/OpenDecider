"""Second looks at hop 2 of the two-hop family, with the same model (no second model, no generation).

On the 9B, most remaining two-hop errors are confident hop-2 lookups for the RIGHT manager (RESULTS 19.5). For the top
hop-1 manager m and the top final answer d of each item (from an earlier eval/decomposition_eval.py run), ask in one
request against the same state:
  shuffled    "Which department does m work in?" with the department options in a different order
  paraphrase  "In which department is m employed?"
  verify      "Does m work in the d department?" (yes/no)
Agreement with the first look, and the verifier's P(yes), are extra confidence signals; whether they separate the
confident errors is measured downstream (RESULTS 19.5: they do not).

    python eval/hop2_second_look.py --ckpt runs/x9b-clean3/model.pt \\
        --items /mnt/ext/opendecider/p7/x9b-clean3_twohop_items.jsonl --out /mnt/ext/opendecider/p9/x9b_hop2_second.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", required=True, help="two-hop items of an earlier decomposition_eval run (h1, marginal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1, help="the seed that run used to build the items")
    a = ap.parse_args()
    from eval.decomposition_eval import build
    from opendecider.checkpoint import load_decider
    rows = [r for r in map(json.loads, open(a.items)) if r["family"] == "two_hop"]
    items = build("two_hop", len(rows), seed=a.seed)
    dec = load_decider(a.ckpt)
    rng = random.Random(0)
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, ((state, qs, y, meta), r) in enumerate(zip(items, rows)):
            assert y == r["y"] and meta["h1_label"] == r["meta"]["h1_label"], "items differ from the run"
            ppl, depts = qs["h1"]["options"], qs["direct"]["options"]
            m = ppl[int(np.argmax(r["parts"]["h1"]))]
            d = depts[int(np.argmax(r["marginal"]))]
            perm = list(range(len(depts)))
            while len(depts) > 1 and perm == list(range(len(depts))):
                rng.shuffle(perm)
            ask = {"shuffled": {"type": "choice", "prompt": f"Which department does {m} work in?",
                                "options": [depts[k] for k in perm], "none": False},
                   "paraphrase": {"type": "choice", "prompt": f"In which department is {m} employed?",
                                  "options": depts, "none": False},
                   "verify": {"type": "noul", "prompt": f"Does {m} work in the {d} department?"}}
            out = dec.decide({"state": state, "questions": ask})["answers"]
            sh = np.empty(len(depts)); sh[perm] = out["shuffled"]["probs_list"]
            f.write(json.dumps({"i": i, "manager": m, "dept": d, "shuffled": sh.tolist(),
                                "paraphrase": out["paraphrase"]["probs_list"], "verify_p_yes": out["verify"]["p_yes"]})
                    + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} items, {(time.time() - t0) / (i + 1):.1f} s/item", flush=True)


if __name__ == "__main__":
    main()
