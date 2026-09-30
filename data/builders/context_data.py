"""Context-learning data: UNLABELLED comparative batches and LABELLED in-context examples.

    python data/builders/context_data.py train     # training records from our permissive pools (data/public_train)
    python data/builders/context_data.py eval      # eval-only variants of held-out benchmarks (data/public_context)

Formats (state is JSON; questions refer to items by id):
  single : {"items": {"item_1": text}}                                   -> 1 question
  batch  : {"items": {"item_1": t1, ..., "item_n": tn}}   (no labels)    -> one question per asked item ("Regarding item_i: ...")
  icl    : {"examples": [{"text": t, "label": name}, ...], "item": text} -> 1 question about "item"
Items and examples are drawn i.i.d. from the pool (natural base rates, random order, batch size 1 included): NOT
curated or balanced, so deployment-like batches keep probabilities calibrated. Eval ICL examples come from each
benchmark's TRAIN split (standard few-shot; nothing is trained on them); eval batch context = other test items.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

TEXT_KEYS = ("user_message", "user_input", "content")


def item_text(rec):
    st = rec["state"]
    for k in TEXT_KEYS:
        if isinstance(st, dict) and k in st:
            return str(st[k])[:600]
    return None


def pools_from(paths, fams):
    pools = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["family"] in fams and len(r["questions"]) == 1 and item_text(r):
                pools.setdefault(r["family"], []).append(r)
    return pools


def q_about(q, ref):
    out = dict(q)
    out["prompt"] = f"Regarding {ref}: {q['prompt']}"
    return out


def make_batch(rng, pool, fam, idx, max_n=12, max_q=6):
    n = rng.randint(1, max_n)
    recs = [rng.choice(pool) for _ in range(n)]
    items = {f"item_{i + 1}": item_text(r) for i, r in enumerate(recs)}
    asked = rng.sample(range(n), min(n, max_q))
    return {"id": f"ctxbatch-{fam}-{idx}", "family": f"batch_{fam}", "context": "batch",
            "state": {"items": items}, "questions": [q_about(recs[i]["questions"][0], f"item_{i + 1}") for i in asked]}


def make_icl(rng, pool, fam, idx, k=None, demo_pool=None, target=None):
    k = rng.randint(1, 8) if k is None else k
    target = target or rng.choice(pool)
    demos = [rng.choice(demo_pool or pool) for _ in range(k)]
    ex = []
    for d in demos:
        q = d["questions"][0]
        ex.append({"text": item_text(d), "label": q["options"][q["label"]]})
    return {"id": f"ctxicl-{fam}-{idx}", "family": f"icl_{fam}", "context": f"icl{k}",
            "state": {"examples": ex, "item": item_text(target)}, "questions": [q_about(target["questions"][0], "item")]}


def make_schema(rng, pool, fam, idx, labelled, max_k=16, text_cap=400):
    """Examples in the MODEL'S OWN query schema (rendered by formatting.solved_cases_prefix at train/eval time):
    labelled = k solved cases (answer filled), unlabelled = k other items (answer empty). Large label sets are cut to
    a candidate subset (answer + example labels + random distractors, <= max_k) so every example lists the same
    options; texts are capped so the whole prefix fits the state-token budget."""
    target = rng.choice(pool)
    q = target["questions"][0]
    k = rng.randint(1, 4) if labelled else rng.randint(1, 7)
    demos = [rng.choice(pool) for _ in range(k)]
    names = q["options"]
    need = {names[q["label"]]} | ({d["questions"][0]["options"][d["questions"][0]["label"]] for d in demos} if labelled else set())
    # records carry different candidate subsets of one family label space: demo labels join the candidate set
    rest = [o for o in names if o not in need]
    names_all = len(set(names) | need)
    K = min(names_all, max(len(need), rng.randint(max(2, min(8, names_all)), max_k)))
    opts = sorted(need) + rng.sample(rest, max(0, K - len(need)))
    rng.shuffle(opts)
    qq = dict(q, options=opts, label=opts.index(names[q["label"]]))
    cap = lambda st: {kk: (str(v)[:text_cap] if isinstance(v, str) else v) for kk, v in st.items()} if isinstance(st, dict) else st
    exs = []
    for d in demos:
        dq = d["questions"][0]
        ans = dq["options"][dq["label"]]
        exs.append({"state": cap(d["state"]), "answers": [ans if labelled and ans in opts else None]})
    return {"id": f"ctxschema-{fam}-{idx}", "family": f"schema_{fam}", "context": "schema_icl" if labelled else "schema_batch",
            "state": cap(target["state"]), "examples": exs, "questions": [qq]}


def build_train(out="data/public_train", n_per_fam=3000, seed=0):
    rng = random.Random(seed)
    fams = {"clinc", "massive", "dbpedia", "inj_jackhhao", "inj_neuralchemy", "inj_guardrails", "inj_spml"}
    pools = pools_from([f"{out}/train.jsonl", f"{out}/injection_train.jsonl"], fams)
    recs = []
    for fam, pool in sorted(pools.items()):
        for i in range(n_per_fam):
            kind = i % 4                        # item-list batch / JSON icl / schema labelled / schema unlabelled
            recs.append(make_batch(rng, pool, fam, i) if kind == 0 else make_icl(rng, pool, fam, i) if kind == 1
                        else make_schema(rng, pool, fam, i, labelled=kind == 2))
    rng.shuffle(recs)
    with open(f"{out}/context_train.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"context_train: {len(recs)} records from {sorted(pools)}")


B77_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"


def banking_train_pool():
    url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{B77_COMMIT}/banking_data/train.csv"
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode())))
    return rows


def build_eval(out="data/public_context", seed=0):
    """Eval-only variants: single / batch8 (unlabelled) / icl4 / icl8 (labelled, from TRAIN split)."""
    from datasets import load_dataset
    require_free_gb(0.3)
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)
    # --- Banking77 8-way (same 8 intents and 160 test items as data/public/banking77_8way.jsonl)
    test = [json.loads(l) for l in open("data/public/banking77_8way.jsonl")]
    opts = test[0]["questions"][0]["options"]
    tr = [r for r in banking_train_pool() if r["category"].replace("_", " ") in opts]
    demo_pool = [{"state": {"user_message": r["text"]}, "questions": [{"type": "choice", "prompt": "",
                  "options": opts, "label": opts.index(r["category"].replace("_", " "))}]} for r in tr]
    test_items = [{"state": {"user_message": r["state"]["customer_message"]}, "questions": r["questions"]} for r in test]
    # --- prompt-injections (116 test items; demos from its TRAIN split)
    ds_tr = load_dataset("deepset/prompt-injections", split="train")
    pi_demo = [{"state": {"user_input": x["text"]}, "questions": [{"type": "noul", "prompt": "", "options": ["yes", "no"],
               "label": 0 if x["label"] == 1 else 1}]} for x in ds_tr]
    pi_test = [json.loads(l) for l in open("data/public/prompt_injections.jsonl")]
    pi_items = [{"state": {"user_input": r["state"]["user_input"]}, "questions": r["questions"]} for r in pi_test]
    for name, items, demos in (("banking77_8way", test_items, demo_pool), ("prompt_injections", pi_items, pi_demo)):
        # schema-consistent variants: examples rendered by the MODEL in its own query schema (answer filled / empty)
        rng_s = random.Random(seed + 1)
        for variant in ("single_schema", "icl4_schema", "icl8_schema", "batch8_schema"):
            recs = []
            for i, t in enumerate(items):
                q = t["questions"][0]
                if variant == "batch8_schema":                # unlabelled: 7 other TEST items, answer slot empty
                    exs = [{"state": x["state"], "answers": [None]}
                           for x in rng_s.sample([x for j, x in enumerate(items) if j != i], 7)]
                elif variant.startswith("icl"):               # labelled: k solved cases from the TRAIN split
                    k = int(variant[3:variant.index("_")])
                    exs = []
                    for d in (rng_s.choice(demos) for _ in range(k)):
                        dq = d["questions"][0]
                        exs.append({"state": d["state"], "answers": [dq["options"][dq["label"]]]})
                else:
                    exs = []
                recs.append({"id": f"{name}-{variant}-{i}", "family": name, "split": f"{name}_{variant}",
                             "state": t["state"], "examples": exs, "questions": [q]})
            with open(f"{out}/{name}_{variant}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"{name}_{variant}: {len(recs)}")
        for variant in ("single", "batch8", "icl4", "icl8", "batch8_native", "icl4_native", "icl8_native"):
            recs = []
            for i, t in enumerate(items):
                q = t["questions"][0]
                if variant == "single":
                    st, qq = {"items": {"item_1": item_text(t)}}, q_about(q, "item_1")
                elif variant == "batch8":                     # target + 7 other TEST items, target at a random slot
                    others = rng.sample([x for j, x in enumerate(items) if j != i], 7)
                    slot = rng.randint(0, 7)
                    group = others[:slot] + [t] + others[slot:]
                    st = {"items": {f"item_{j + 1}": item_text(x) for j, x in enumerate(group)}}
                    qq = q_about(q, f"item_{slot + 1}")
                elif variant == "batch8_native":              # same batch, rendered as plain text blocks
                    others = rng.sample([x for j, x in enumerate(items) if j != i], 7)
                    slot = rng.randint(0, 7)
                    group = others[:slot] + [t] + others[slot:]
                    st = "".join(f"Item {j + 1}: {item_text(x)}\n" for j, x in enumerate(group))
                    qq = q_about(q, f"Item {slot + 1}")
                elif variant.endswith("_native"):             # native few-shot: Example/Answer pairs, then the item
                    k = int(variant[3:variant.index("_")])
                    demo = [rng.choice(demos) for _ in range(k)]
                    blocks = "".join(f"Example: {item_text(d)}\nAnswer: {d['questions'][0]['options'][d['questions'][0]['label']]}\n\n"
                                     for d in demo)
                    st = blocks + f"Now classify this one.\nItem: {item_text(t)}"
                    qq = q_about(q, "the item")
                else:
                    r = make_icl(rng, None, name, i, k=int(variant[3:]), demo_pool=demos, target=t)
                    st, qq = r["state"], r["questions"][0]
                recs.append({"id": f"{name}-{variant}-{i}", "family": name, "split": f"{name}_{variant}",
                             "state": st, "questions": [qq]})
            with open(f"{out}/{name}_{variant}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"{name}_{variant}: {len(recs)}")


def tfidf_neighbours(queries, pool, k):
    """Indices of the k most similar pool texts per query (TF-IDF cosine on lowercase word unigrams+bigrams;
    numpy only). A cheap, model-free retriever: the point is WHICH examples are shown, not a learned embedding."""
    import re as _re
    import numpy as np
    tok = lambda t: (lambda w: w + [a + " " + b for a, b in zip(w, w[1:])])(_re.findall(r"[a-z0-9']+", t.lower()))
    docs = [tok(t) for t in pool]
    vocab = {}
    for d in docs:
        for w in set(d):
            vocab[w] = vocab.get(w, 0) + 1
    idx = {w: i for i, w in enumerate(vocab)}
    idf = np.log((1 + len(docs)) / (1 + np.array([vocab[w] for w in idx]))) + 1

    def vec(words):
        v = np.zeros(len(idx))
        for w in words:
            if w in idx:
                v[idx[w]] += 1
        v *= idf
        return v / (np.linalg.norm(v) + 1e-9)
    P = np.stack([vec(d) for d in docs])
    Qm = np.stack([vec(tok(t)) for t in queries])
    return np.argsort(-(Qm @ P.T), axis=1)[:, :k]


def build_knn_eval(out="data/public_context", ks=(4, 8)):
    """Retrieved labelled examples (schema-consistent, answer filled), no training: for each test item the k most
    similar TRAIN-split items (TF-IDF). Banking77 8-way, Banking77 77-way, prompt injections."""
    from datasets import load_dataset
    require_free_gb(0.3)
    tr = banking_train_pool()
    b8 = [json.loads(l) for l in open("data/public/banking77_8way.jsonl")]
    b77 = [json.loads(l) for l in open("data/public/banking77_77way.jsonl")]
    ds_tr = load_dataset("deepset/prompt-injections", split="train")
    pi = [json.loads(l) for l in open("data/public/prompt_injections.jsonl")]
    norm = lambda x: x.replace("_", " ").strip().lower()
    sets = []
    for name, test, key in (("banking77_8way", b8, "customer_message"), ("banking77_77way", b77, "customer_message")):
        opts = test[0]["questions"][0]["options"]
        by = {norm(o): o for o in opts}
        pool = [(r["text"], by[norm(r["category"])]) for r in tr if norm(r["category"]) in by]
        sets.append((name, [(t["state"][key], t) for t in test], pool, "user_message"))
    pool_pi = [(x["text"], "yes" if x["label"] == 1 else "no") for x in ds_tr]
    sets.append(("prompt_injections", [(t["state"]["user_input"], t) for t in pi], pool_pi, "user_input"))
    for name, test, pool, skey in sets:
        nb = tfidf_neighbours([q for q, _ in test], [p for p, _ in pool], max(ks))
        for k in ks:
            recs = []
            for i, (text, t) in enumerate(test):
                q = t["questions"][0]
                exs = [{"state": {skey: pool[j][0]}, "answers": [pool[j][1]]} for j in nb[i][:k][::-1]]   # nearest LAST
                recs.append({"id": f"{name}-knn{k}-{i}", "family": name, "split": f"{name}_knn{k}_schema",
                             "state": {skey: text}, "examples": exs, "questions": [q]})
            with open(f"{out}/{name}_knn{k}_schema.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"{name}_knn{k}_schema: {len(recs)}")


if __name__ == "__main__":
    {"train": build_train, "eval": build_eval, "knn": build_knn_eval}[sys.argv[1]]()
