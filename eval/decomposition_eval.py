"""Experiment plan step 2: caller-side decomposition, evaluated (docs/experiment_plan.md).

One question asked directly vs the same decision decomposed into narrow sub-questions, on synthetic families whose
generator knows the sub-answers (ground truth by construction, no teacher). Every request answers BOTH forms
against the same state in one call (questions are isolated from each other, O3), so comparisons are paired.

  two_hop   "Which department does the manager of X work in?" vs chained: "Who is X's manager?" then
            "Which department does {manager} work in?" (the chosen option is substituted; no generation).
  routing   a stated rule over three record facts ("manual review if amount > $500 and not Gold, or flagged") vs
            three yes/no parts combined (a) by the rule in code, zero-shot, with independence, and (b) by a logistic
            fitted on a separate train split (FITTED: labelled as such; the composite question type in the API).
  any_over  "Did <customer> place any order over $X?" vs "What is <customer>'s largest order amount?" (choice over
            that customer's amounts) and the comparison in code.
  date_cmp  "Is <account>'s renewal date before its contract end date?" vs one choice question per date and the
            comparison in code (P = sum over date pairs, assuming the two answers independent).

    .venv/bin/python eval/decomposition_eval.py --ckpt runs/x9b-clean3/model.pt --n 600 --out runs/decomp/x9b.json
Metrics: accuracy and Brier of the final yes/no (or department) decision, paired bootstrap CIs (1,000 resamples).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIRST = ["Ana", "Ben", "Chloe", "Dmitri", "Elena", "Farid", "Grace", "Hugo", "Ines", "Jonas", "Kira", "Liam", "Maya",
         "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Ximena", "Yusuf", "Zoe"]
LAST = ["Alvarez", "Brooks", "Chen", "Dubois", "Evans", "Fischer", "Garcia", "Haddad", "Ito", "Jensen", "Kowalski",
        "Lopez", "Moreau", "Novak", "Okafor", "Patel", "Rossi", "Silva", "Tanaka", "Weber"]
DEPTS = ["Finance", "Legal", "Marketing", "Engineering", "Sales", "Support", "Operations", "Research", "Design", "HR"]
TIERS = ["Gold", "Silver", "Bronze", "Standard"]
COUNTRIES = ["Spain", "Mexico", "Germany", "Japan", "Brazil", "Canada", "India", "Kenya"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def names(rng, n):
    return [f"{f} {rng.choice(LAST)}" for f in rng.sample(FIRST, n)]


def fmt_date(d, rng):
    s = rng.choice(["iso", "long", "short"])
    return d.isoformat() if s == "iso" else (f"{MONTHS[d.month - 1]} {d.day}, {d.year}" if s == "long"
                                             else f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}")


# ============================================================ families: (state, direct question, plan, label)
def two_hop(rng):
    n = rng.randint(6, 12)
    ppl = names(rng, n)
    depts = rng.sample(DEPTS, rng.randint(4, 7))
    dept = {p: rng.choice(depts) for p in ppl}
    mgr = {p: rng.choice([q for q in ppl if q != p]) for p in ppl}
    x = rng.choice(ppl)
    state = {"directory": [{"name": p, "department": dept[p], "manager": mgr[p]} for p in ppl]}
    y = depts.index(dept[mgr[x]])
    qs = {"direct": {"type": "choice", "prompt": f"Which department does the manager of {x} work in?",
                     "options": depts, "none": False},
          "h1": {"type": "choice", "prompt": f"Who is the manager of {x}?", "options": ppl, "none": False},
          "h2": {"type": "choice", "prompt": "Which department does {h1} work in?", "options": depts, "none": False}}
    for j, p in enumerate(ppl):            # marginalized chain: hop 2 for every hop-1 candidate (same state pass)
        qs[f"h2_{j}"] = {"type": "choice", "prompt": f"Which department does {p} work in?", "options": depts,
                         "none": False}
    return state, qs, y, {"K": len(depts), "h1_label": ppl.index(mgr[x])}


ROUTE_RULE = ("Rule: an order goes to manual review if its amount is over $500 and the customer is not Gold tier, "
              "or if the order is flagged by the fraud check.")


def routing(rng):
    amount = rng.choice([rng.randint(20, 499), rng.randint(501, 3000)])
    tier = rng.choice(TIERS)
    flagged = rng.random() < 0.3
    others = [{"order": f"O-{rng.randint(1000, 9999)}", "amount": f"${rng.randint(20, 3000)}",
               "tier": rng.choice(TIERS), "flagged": rng.random() < 0.3} for _ in range(rng.randint(0, 3))]
    oid = f"O-{rng.randint(1000, 9999)}"
    rec = {"order": oid, "amount": f"${amount}", "customer_tier": tier, "country": rng.choice(COUNTRIES),
           "fraud_check": "flagged" if flagged else "clear"}
    state = {"order": rec, "recent_orders_other_customers": others}
    a, g, f = amount > 500, tier == "Gold", flagged
    truth = (a and not g) or f
    qs = {"direct": {"type": "noul", "prompt": f"{ROUTE_RULE} Should order {oid} go to manual review?", "none": False},
          "amt": {"type": "noul", "prompt": f"Is the amount of order {oid} over $500?", "none": False},
          "gold": {"type": "noul", "prompt": f"Is the customer of order {oid} Gold tier?", "none": False},
          "flag": {"type": "noul", "prompt": f"Was order {oid} flagged by the fraud check?", "none": False}}
    return state, qs, 0 if truth else 1, {"K": 2, "parts": [a, g, f]}


def any_over(rng):
    custs = names(rng, rng.randint(3, 6))
    orders = []
    for c in custs:
        for _ in range(rng.randint(1, 5)):
            orders.append({"customer": c, "amount": rng.randint(5, 250), "date": dt.date(2026, rng.randint(1, 9),
                                                                                         rng.randint(1, 28)).isoformat()})
    rng.shuffle(orders)
    x = rng.choice(custs)
    mine = sorted({o["amount"] for o in orders if o["customer"] == x})
    want = rng.random() < 0.5                                   # balanced: threshold below or at/above the max
    grid = [t for t in range(25, 250, 25) if (t < max(mine)) == want]
    thr = rng.choice(grid) if grid else (25 if want else 250)
    truth = max(mine) > thr
    state = {"orders": [{**o, "amount": f"${o['amount']}"} for o in orders]}
    cands = sorted(set(mine) | set(rng.sample(range(5, 251), 4)))
    qs = {"direct": {"type": "noul", "prompt": f"Did {x} place any order over ${thr}?", "none": False},
          "max": {"type": "choice", "prompt": f"What is the largest amount of any order placed by {x}?",
                  "options": [f"${v}" for v in cands], "none": False}}
    return state, qs, 0 if truth else 1, {"K": 2, "thr": thr, "cands": cands, "max_label": cands.index(max(mine))}


def date_cmp(rng):
    accts = [f"A-{v}" for v in rng.sample(range(100, 999), rng.randint(3, 6))]
    recs = []
    for a in accts:
        r = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 700))
        e = r + dt.timedelta(days=rng.randint(-200, 200) or 1)
        recs.append({"account": a, "renewal_date": r, "contract_end": e})
    x = rng.choice(recs)
    pool = sorted({x["renewal_date"], x["contract_end"]} | {r["renewal_date"] for r in recs} |
                  {r["contract_end"] for r in recs})
    state = {"accounts": [{"account": r["account"], "renewal_date": fmt_date(r["renewal_date"], rng),
                           "contract_end": fmt_date(r["contract_end"], rng)} for r in recs]}
    opts = [d.isoformat() for d in pool]
    qs = {"direct": {"type": "noul", "prompt": f"Is the renewal date of account {x['account']} before its contract "
                                               f"end date?", "none": False},
          "ren": {"type": "choice", "prompt": f"What is the renewal date of account {x['account']}?", "options": opts,
                  "none": False},
          "end": {"type": "choice", "prompt": f"What is the contract end date of account {x['account']}?",
                  "options": opts, "none": False}}
    truth = x["renewal_date"] < x["contract_end"]
    return state, qs, 0 if truth else 1, {"K": 2, "dates": opts, "ren_label": opts.index(x["renewal_date"].isoformat()),
                                          "end_label": opts.index(x["contract_end"].isoformat())}


FAMILIES = {"two_hop": two_hop, "routing": routing, "any_over": any_over, "date_cmp": date_cmp}


def build(fam, n, seed):
    rng = random.Random(f"{fam}-{seed}")
    return [FAMILIES[fam](rng) for _ in range(n)]


# ============================================================ decomposed answers
def p_of(ans):
    return np.asarray(ans["probs_list"], float)


def decomposed(fam, ans, meta):
    """-> probability vector of the final decision from the sub-answers (zero-shot, in code)."""
    if fam == "two_hop":
        return p_of(ans["h2"])                                  # the chain conditions on the chosen manager
    if fam == "routing":
        a, g, f = (p_of(ans[k])[0] for k in ("amt", "gold", "flag"))
        p = 1 - (1 - a * (1 - g)) * (1 - f)
        return np.array([p, 1 - p])
    if fam == "any_over":
        P = p_of(ans["max"])
        p = float(sum(P[i] for i, v in enumerate(meta["cands"]) if v > meta["thr"]))
        return np.array([p, 1 - p])
    if fam == "date_cmp":
        P1, P2 = p_of(ans["ren"]), p_of(ans["end"])
        D = np.array(meta["dates"])
        less = (D[:, None] < D[None, :]).astype(float)           # ISO dates compare as strings
        p = float(P1 @ less @ P2)
        return np.array([p, 1 - p])
    raise KeyError(fam)


def fit_routing_logistic(train_answers, train_labels):
    """FITTED combination (composite question): logistic on the three parts' P(yes), fitted on a train split."""
    from opendecider.decompose import fit_logistic
    return fit_logistic(train_answers, [1 - y for y in train_labels], ["amt", "gold", "flag"])


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=300, help="routing: train split for the fitted combination")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from eval.evaluate import gpu_context
    from opendecider.checkpoint import load_decider
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = {"ckpt": a.ckpt, "n": a.n, "families": {}}
    items_path = a.out.replace(".json", "_items.jsonl")
    open(items_path, "w").close()
    with gpu_context("cuda", "decomposition_eval"):
        dec = load_decider(a.ckpt)
        for fam in a.families:
            t0 = time.time()
            items = build(fam, a.n, seed=1)
            rows = []
            for state, qs, y, meta in items:
                out = dec.decide({"state": state, "questions": qs})["answers"]
                Pd = p_of(out["direct"]); Pc = decomposed(fam, out, meta)
                row = {"y": y, "direct": Pd.tolist(), "decomp": Pc.tolist(), "answers": out, "meta": meta}
                if fam == "two_hop":                                     # sum over hop-1: P(m) P(dept | m)
                    P1 = p_of(out["h1"])
                    row["marginal"] = sum(P1[j] * p_of(out[f"h2_{j}"]) for j in range(len(P1))).tolist()
                rows.append(row)
            r = {}
            for k in ("direct", "decomp"):                               # option counts vary per item (two_hop)
                acc = np.array([float(np.argmax(row[k]) == row["y"]) for row in rows])
                brier = np.array([float(((np.asarray(row[k]) - np.eye(len(row[k]))[row["y"]]) ** 2).sum())
                                  for row in rows])
                r[k] = {"acc": boot_ci(acc), "brier": boot_ci(brier)}
                r[k + "_raw"] = {"acc": acc.tolist(), "brier": brier.tolist()}
            r["diff_acc"] = boot_ci(np.array(r["decomp_raw"]["acc"]) - np.array(r["direct_raw"]["acc"]))
            r["diff_brier"] = boot_ci(np.array(r["decomp_raw"]["brier"]) - np.array(r["direct_raw"]["brier"]))
            if fam == "two_hop":                                         # hop-level accuracy, to see where it fails
                r["hop1_acc"] = boot_ci([float(np.argmax(row["answers"]["h1"]["probs_list"]) == row["meta"]["h1_label"])
                                         for row in rows])
            if fam == "routing":                                         # FITTED combination on a separate split
                tr = build(fam, a.n_train, seed=2)
                tr_ans = [dec.decide({"state": s, "questions": q})["answers"] for s, q, _, _ in tr]
                spec = fit_routing_logistic([{k: an[k] for k in ("amt", "gold", "flag")} for an in tr_ans],
                                            [y for _, _, y, _ in tr])
                from opendecider.decider import composite_feature
                z = np.array([spec["bias"] + sum(w * composite_feature(f, row["answers"]) for f, w in
                                                 spec["weights"].items()) for row in rows])
                p = 1 / (1 + np.exp(-z))
                for row, pi in zip(rows, p):
                    row["fitted"] = [float(pi), float(1 - pi)]
                with open(items_path, "a") as f:                         # train split too (for refits)
                    for (st, q, y, meta), an in zip(tr, tr_ans):
                        f.write(json.dumps({"family": "routing_train", "y": y, "direct": an["direct"]["probs_list"],
                                            "parts": {k: v["probs_list"] for k, v in an.items()},
                                            "meta": meta}) + "\n")
                P = np.stack([p, 1 - p], 1); yy = np.array([row["y"] for row in rows])
                acc = (P.argmax(1) == yy).astype(float); brier = ((P - np.eye(2)[yy]) ** 2).sum(1)
                r["fitted_logistic"] = {"acc": boot_ci(acc), "brier": boot_ci(brier), "spec": spec,
                                        "n_train": a.n_train, "diff_acc_vs_direct":
                                        boot_ci(acc - np.array(r["direct_raw"]["acc"]))}
                r["part_acc"] = {k: boot_ci([float((np.argmax(row["answers"][k]["probs_list"]) == 0) ==
                                                   row["meta"]["parts"][i]) for row in rows])
                                 for i, k in enumerate(("amt", "gold", "flag"))}
            with open(items_path, "a") as f:                             # per item: calibration / certification analysis
                for row in rows:
                    f.write(json.dumps({"family": fam, "y": row["y"], "direct": row["direct"], "decomp": row["decomp"],
                                        "fitted": row.get("fitted"), "marginal": row.get("marginal"),
                                        "parts": {k: v["probs_list"] for k, v in row["answers"].items()},
                                        "meta": row["meta"]}) + "\n")
            r["seconds"] = round(time.time() - t0, 1)
            for k in ("direct_raw", "decomp_raw"):
                r.pop(k)
            res["families"][fam] = r
            print(fam, json.dumps({k: v for k, v in r.items() if k in ("direct", "decomp", "diff_acc", "fitted_logistic",
                                                                       "hop1_acc", "part_acc")}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
