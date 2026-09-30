"""Tier 2 store recall@budget (docs/roadmap_designs.md §3, Tier 2 "Evaluation").

    python -m eval.store_recall [--sizes 1000 10000 100000] [--budgets 512 2048 8192] [--trials 45]
                                [--embedder hash | module:factory] [--unions max sum] [--option-weight 1.0]

Builds a synthetic haystack of order records (programmatic, ground truth known), buries one deciding record per
trial, asks a question whose answer is only in that record, and checks whether StateStore.select() keeps the
deciding record within the token budget. Queries = question prompt + option texts (the request union), as in the
design. Methods: bm25-only, dense-only, hybrid (RRF). Recall is reported with 95% bootstrap CIs (1,000 resamples).

The repo's synthetic generator (data/synthetic/generate.py) renders small self-contained states whose deciding
record is not tagged at record level, so a record-level generator is used here instead.

Question kinds (deciding record differs from distractors only in the asked fact):
  by_id    "What is the status of order ORD-xxxxxx?"   id appears only in the deciding record
  by_name  "Which city was <rare surname>'s order shipped to?"   surname is unique to the deciding record
  by_event "Which customer's order had a <noun> issue?"   note uses an inflected rare word (quarantined vs
           quarantine): exact-token BM25 misses it, stem-sharing dense (even the hash placeholder) can find it;
           options are customer names (incl. the deciding one)

--embedder hash is a PLACEHOLDER (hashed bag of words + 5-char prefixes); real dense vectors will come from the
backbone. A custom factory `module:fn` must return embed_fn(list[str]) -> (n, d) array.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import time

import numpy as np

from opendecider.store import StateStore, hash_embedder

from .metrics import bootstrap_mean_ci

FIRST = ["Nadia", "Mateo", "Wen", "Hugo", "Ines", "Omar", "Lena", "Kofi", "Sara", "Ivan", "Maya", "Tariq", "Elif",
         "Jonas", "Priya", "Diego", "Aiko", "Farah", "Luca", "Zara"]
LAST = ["Smith", "Garcia", "Chen", "Muller", "Rossi", "Silva", "Khan", "Kim", "Novak", "Dubois", "Ali", "Brown"]
CITIES = ["Lyon", "Porto", "Osaka", "Quito", "Accra", "Hanoi", "Perth", "Lima", "Oslo", "Tunis", "Graz", "Cork"]
STATUS = ["pending", "shipped", "delivered", "returned", "cancelled", "on hold"]
NOTES = ["left at the front door", "customer asked for gift wrap", "delivery window moved", "address verified",
         "courier called ahead", "signed by recipient", "partial shipment", "paid by card", "invoice emailed",
         "customer requested a refund", "damaged packaging reported", "express upgrade applied"]
# (note form in the deciding record, question form): shared 5-char stem, different token
EVENTS = [("quarantined", "quarantine"), ("escalated", "escalation"), ("rerouted", "rerouting"),
          ("impounded", "impound"), ("embargoed", "embargo"), ("tampered", "tampering"),
          ("misrouted", "misrouting"), ("counterfeit", "counterfeiting"), ("overcharged", "overcharging"),
          ("duplicated", "duplicate"), ("suspended", "suspension"), ("abandoned", "abandonment")]
SYL = ["ka", "ro", "vi", "zu", "mel", "dra", "sto", "qui", "ne", "bal", "tor", "ix", "yan", "fe", "gru", "ph"]


def rare_surname(rng):
    return "".join(rng.choice(SYL) for _ in range(4)).capitalize()


def base_record(rng, i):
    return {"id": f"ORD-{i:06d}", "ts": 1.7e9 + rng.randrange(10 ** 7),
            "customer": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "city": rng.choice(CITIES),
            "status": rng.choice(STATUS), "items": rng.randint(1, 9), "note": rng.choice(NOTES)}


def build(n, trials, seed):
    """-> (records, questions). Record ids are a random permutation of 0..n-1 so ids carry no position info."""
    rng = random.Random(seed)
    nums = rng.sample(range(100000, 100000 + 10 * n), n)
    recs = [base_record(rng, nums[i]) for i in range(n)]
    slots = rng.sample(range(n), trials)
    qs, used = [], set()
    for t, j in enumerate(slots):
        r, kind = recs[j], ("by_id", "by_name", "by_event")[t % 3]
        if kind == "by_id":
            prompt, opts = f"What is the status of order {r['id']}?", list(STATUS)
        elif kind == "by_name":
            sn = rare_surname(rng)
            while sn in used:
                sn = rare_surname(rng)
            used.add(sn)
            r["customer"] = f"{rng.choice(FIRST)} {sn}"
            prompt, opts = f"Which city was {sn}'s order shipped to?", list(CITIES)
        else:
            ev, q = EVENTS[(t // 3) % len(EVENTS)]
            r["note"] = f"parcel {ev} at the depot"
            others = [recs[x]["customer"] for x in rng.sample(range(n), 4)]
            opts = rng.sample([r["customer"]] + others, 5)
            prompt = f"Which customer's order had a {q} issue?"
        qs.append({"kind": kind, "prompt": prompt, "options": opts, "deciding": r["id"]})
    return recs, qs


def load_embedder(spec):
    if spec == "hash":
        return hash_embedder(dim=256)
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)()


METHODS = {"bm25": {"bm25": 1.0, "dense": 0.0}, "dense": {"bm25": 0.0, "dense": 1.0}, "hybrid": {}}


def run(sizes, budgets, trials, embedder, seed, unions=("max", "sum"), option_weight=1.0):
    emb = load_embedder(embedder)
    rows = []
    for n in sizes:
        recs, qs = build(n, trials, seed)
        t0 = time.time()
        s = StateStore(":memory:", embed_fn=emb)
        s.add_records(recs, prefix="orders")
        build_s = time.time() - t0
        for union in unions:
            for m, w in METHODS.items():
                for b in budgets:
                    hit, lat, kinds = [], [], []
                    for q in qs:
                        t1 = time.perf_counter()
                        qt = [q["prompt"]] + [(o, option_weight) for o in q["options"]]
                        sel = s.select(qt, token_budget=b, weights=w, union=union)
                        lat.append(1e3 * (time.perf_counter() - t1))
                        hit.append(float(q["deciding"] in sel.ids)); kinds.append(q["kind"])
                    hit, kinds = np.asarray(hit), np.asarray(kinds)
                    mean, lo, hi = bootstrap_mean_ci(hit, n=1000, seed=seed)
                    row = {"n": n, "union": union, "method": m, "budget": b, "recall": mean, "ci": [lo, hi],
                           "by_kind": {k: float(hit[kinds == k].mean()) for k in ("by_id", "by_name", "by_event")},
                           "select_ms_p50": float(np.percentile(lat, 50)),
                           "select_ms_p95": float(np.percentile(lat, 95)), "build_s": build_s, "trials": len(qs)}
                    rows.append(row)
                    print(f"n={n:>7} {union} {m:>6} budget={b:>5}  recall={mean:.3f} [{lo:.3f},{hi:.3f}]  "
                          + " ".join(f"{k}={v:.2f}" for k, v in row["by_kind"].items())
                          + f"  p50={row['select_ms_p50']:.0f}ms", flush=True)
        s.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--budgets", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--trials", type=int, default=45)
    ap.add_argument("--embedder", default="hash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unions", nargs="+", default=["max", "sum"], choices=["max", "sum"])
    ap.add_argument("--option-weight", type=float, default=1.0)
    ap.add_argument("--out", default="runs/store-recall")
    a = ap.parse_args()
    rows = run(a.sizes, a.budgets, a.trials, a.embedder, a.seed, a.unions, a.option_weight)
    print("\n| n | union | method | " + " | ".join(f"R@{b}" for b in a.budgets) + " | select p50 ms |")
    print("|---|---|---|" + "---|" * (len(a.budgets) + 1))
    for n in a.sizes:
        for u in a.unions:
            for m in METHODS:
                cell = {r["budget"]: r for r in rows if r["n"] == n and r["method"] == m and r["union"] == u}
                print(f"| {n} | {u} | {m} | " + " | ".join(
                    f"{cell[b]['recall']:.2f} [{cell[b]['ci'][0]:.2f},{cell[b]['ci'][1]:.2f}]" for b in a.budgets)
                      + f" | {cell[a.budgets[-1]]['select_ms_p50']:.0f} |")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"recall_{a.embedder.replace(':', '_')}_seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(a), "placeholder_embedder": a.embedder == "hash", "rows": rows}, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
