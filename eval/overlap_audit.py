"""Text overlap between the training corpus and every benchmark we report.

    .venv/bin/python -m eval.overlap_audit --out runs/overlap_audit.json

Training side: every JSONL file a release config trains, validates or calibrates on (all splits), including the
generated injection sets. Benchmark side: the held-out evaluation files in data/public/ and the typed-decisions test
cases. Every string in a record's state and questions is compared (options and prompts included):
  exact   : identical after lower-casing and whitespace normalisation (strings of 20+ characters)
  span13  : the two texts share a 13-word span
  span8   : the two texts share an 8-word span (looser; catches near-duplicates and boilerplate)
  near    : short-text benchmarks (prompt injections, Banking77 8-way): every benchmark text is compared with every
            training text of the same kind by word overlap (Jaccard); pairs >= 0.4 are listed for reading
Strings that recur in more than 20 benchmark items (our own question templates) are skipped. Reported per training
file and per benchmark, with up to 5 examples per pair. A shared 8-word span is often harmless (common phrases,
abstract boilerplate), so the examples should be read before drawing conclusions.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import yaml

sys.path.insert(0, ".")
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from strings(v)


def grams(text, n):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def benchmark_index(n_values=(13, 8)):
    """{benchmark: [(item id, text)]} and gram -> set of (benchmark, item id) per n."""
    items = collections.defaultdict(list)
    for f in sorted(glob.glob("data/public/*.jsonl")):
        name = os.path.basename(f)[:-6]
        for line in open(f):
            r = json.loads(line)
            for t in strings([r.get("state"), r.get("questions")]):
                if len(norm(t)) >= 20:
                    items[name].append((r.get("id"), t))
    from eval.typed_decisions import load_cases
    for r in load_cases("test"):
        for t in strings([json.loads(r["state"]) if isinstance(r["state"], str) else r["state"], r.get("questions")]):
            if len(norm(t)) >= 20:
                items["typed_decisions"].append((r["id"], t))
    freq = collections.Counter(norm(t) for lst in items.values() for _, t in lst)
    template = {t for t, c in freq.items() if c > 20}               # our shared question templates
    items = {b: [(i, t) for i, t in lst if norm(t) not in template] for b, lst in items.items()}
    exact = {norm(t): (b, i) for b, lst in items.items() for i, t in lst}
    idx = {n: collections.defaultdict(set) for n in n_values}
    for b, lst in items.items():
        for i, t in lst:
            for n in n_values:
                for g in grams(t, n):
                    idx[n][g].add((b, i))
    return items, exact, idx


def near_duplicates(threshold=0.4):
    """Word-Jaccard pairs between short benchmark texts and training texts of the same kind."""
    words = lambda t: set(re.findall(r"[a-z0-9äöüß']+", norm(t)))
    pairs = []
    plan = [("prompt_injections", lambda r: r["state"]["user_input"], "data/clean_train/*inj*_*.jsonl"),
            ("banking77_8way", lambda r: next(strings(r["state"])), "data/clean_train/kept_*_*.jsonl")]
    for bench, get, pattern in plan:
        train = []
        for f in sorted(glob.glob(pattern)):
            for line in open(f):
                r = json.loads(line)
                w = words(" ".join(strings(r["state"])))
                if len(w) >= 5:
                    train.append((os.path.basename(f), r["id"], r["questions"][0].get("label"), w))
        for line in open(f"data/public/{bench}.jsonl"):
            r = json.loads(line)
            bw = words(get(r))
            if len(bw) < 5:
                continue
            j, f, i, lab = max(((len(bw & tw) / len(bw | tw)), f, i, lab) for f, i, lab, tw in train)
            if j >= threshold:
                pairs.append({"benchmark": bench, "item": r["id"], "label": r["questions"][0].get("label"),
                              "jaccard": round(j, 2), "training_file": f, "record": i, "record_label": lab})
    return pairs


def training_files():
    files = set()
    for cfg in ("configs/train_2b_clean.yaml", "configs/train_9b_clean.yaml"):
        t = yaml.safe_load(open(cfg))["train"]
        for k in ("data", "val", "calib"):
            for f in t.get(k) or []:
                files.add(f)
                for s in ("train", "val", "calib"):                    # all splits of each source
                    files.add(f.replace("_train.jsonl", f"_{s}.jsonl").replace("/train.jsonl", f"/{s}.jsonl"))
    return sorted(f for f in files if os.path.exists(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/overlap_audit.json")
    a = ap.parse_args()
    items, exact, idx = benchmark_index()
    print({b: len(v) for b, v in items.items()}, flush=True)
    report = {"benchmarks": {b: len(v) for b, v in items.items()}, "files": {}}
    for f in training_files():
        hits = {"exact": collections.Counter(), "span13": collections.Counter(), "span8": collections.Counter()}
        examples = collections.defaultdict(list)
        n_rec = 0
        for line in open(f):
            r = json.loads(line)
            n_rec += 1
            found = {}
            for t in strings([r.get("state"), r.get("questions"), r.get("examples")]):
                if len(norm(t)) < 20:
                    continue
                if norm(t) in exact:
                    found.setdefault("exact", (exact[norm(t)], t))
                for n, key in ((13, "span13"), (8, "span8")):
                    if key in found:
                        continue
                    for g in grams(t, n):
                        if g in idx[n]:
                            found[key] = (next(iter(idx[n][g])), g)
                            break
            for key, ((bench, item), what) in found.items():
                hits[key][bench] += 1                              # records with at least one hit, per benchmark
                if len(examples[(key, bench)]) < 5:
                    examples[(key, bench)].append({"record": r.get("id"), "benchmark_item": item, "text": what[:160]})
        report["files"][f] = {"records": n_rec, **{k: dict(v) for k, v in hits.items()},
                              "examples": {f"{k}|{b}": v for (k, b), v in examples.items()}}
        print(f"{f}: {n_rec} records; " + "; ".join(f"{k} {dict(v)}" for k, v in hits.items() if v), flush=True)
    report["near"] = near_duplicates()
    for pair in report["near"]:
        print("near:", pair, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
