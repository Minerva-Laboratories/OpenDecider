"""Clean-provenance TRAINING corpus: no share-alike, non-commercial or unlicensed sources, no LLM-generated labels.

    python data/builders/clean_train.py --out data/clean_train

1. Keeps the permissive part of the existing corpus, filtered by family (no re-download):
   public_train: clinc, massive, winogrande (drops boolq, snli, arc, dbpedia: CC-BY-SA)
   knowledge   : qasc, cosmos_qa, quartz, strategyqa (drops scienceqa: CC-BY-SA)
   context     : clinc and massive in-context items only
   injection   : Gandalf only (upstream audit in data/MANIFEST.md)
2. Adds new sources (licenses and revisions verified 2026-10-01; data/MANIFEST.md):
   knowledge MC : CODAH (ODC-BY), Balanced-COPA (CC-BY-4.0), AQuA-RAT (Apache-2.0), CaseHOLD via LexGLUE (CC-BY-4.0)
   yes/no + NLI : CUAD clause presence (CC-BY-4.0; also "not in the passage" negatives), CSQA2 (CC-BY-4.0, GitHub;
                  ConceptNet prompt fields dropped), RuleTaker (Apache-2.0), HANS (MIT, GitHub)
   topic        : LEDGAR 100 classes via LexGLUE (CC-BY-4.0), arXiv title+abstract categories (CC0 metadata)
   injection    : TrustAIRLab in-the-wild jailbreak/regular prompts (MIT), LLMail-Inject attack emails (MIT; raw
                  submissions only, never the LLM-judge labels) with programmatic benign mailboxes, OASST1 human
                  prompts as negatives (Apache-2.0)
   harmful req. : XSTest (CC-BY-4.0, GitHub), JBB-Behaviors (MIT; AdvBench rows dropped: LLM-generated)
   ordinal      : Civil Comments toxicity (CC0), OASST1 human quality/helpfulness ratings (Apache-2.0)
Contamination: every text is checked against the held-out benchmarks in data/public/ and all splits of
deepset/prompt-injections (normalised exact match, plus any shared 13-word span for texts of 13+ words).
Streaming: large sources are read with `datasets` streaming (nothing cached); small files are fetched to memory.
Output: <group>_{train,val,calib}.jsonl split 90/5/5 by hash of the record id, revisions.json, stats.json.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opendecider.guards import require_free_gb  # noqa: E402

HF = {  # repo: (config, pinned sha)
    "jaredfern/codah": ("codah", "4b0e0e7f33b3ae8cd8e69bcdfa09af9611d8f3bd"),
    "pkavumba/balanced-copa": (None, "813bd03cd6e07d9bd8d7333896ad5d40abb95ea9"),
    "deepmind/aqua_rat": ("raw", "33301c6a050c96af81f63cad5562cb5363e88971"),
    "coastalcph/lex_glue": (None, "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"),
    "gfissore/arxiv-abstracts-2021": (None, "e4c5fbd4dec8e46a5dc869216fe1c94cc585757a"),
    "theatticusproject/cuad": (None, "a3c393f5d103fd0c516374e4fdff676c8176dcb1"),
    "tasksource/ruletaker": (None, "a3e0880baeb6ec3d478f4c4d85afe04b21b6cf7f"),
    "TrustAIRLab/in-the-wild-jailbreak-prompts": (None, "a10aab8eff1c73165a442d4464dce192bd28b9c5"),
    "microsoft/llmail-inject-challenge": (None, "1063bdf01ec8762b812d5e06ee768a06faa5a6f7"),
    "JailbreakBench/JBB-Behaviors": ("behaviors", "886acc352a31533ffbcf4ef22c744658688086fc"),
    "OpenAssistant/oasst1": (None, "fdf72ae0827c1cda404aff25b6603abec9e3399b"),
    "google/civil_comments": (None, "f2970eb3a55777454c94069077cc8d9b5866312d"),
}
GH = {
    "csqa2": "https://raw.githubusercontent.com/allenai/csqa2/3f57a4a6a1d51fe6a9b462b59be67c08d41fdc39/dataset/CSQA2_train.json.gz",
    "hans": "https://raw.githubusercontent.com/tommccoy1/hans/7299f6f657089ce06a0f98e7e81f8d0f5b7741ce/heuristics_train_set.jsonl",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv",
}
KEEP_PUBLIC = {"clinc", "massive", "winogrande"}
KEEP_KNOWLEDGE = {"qasc", "cosmos_qa", "quartz", "strategyqa"}
# Upstream audit 2026-10-01 (data/MANIFEST.md): SPML prompts were written by GPT-4 and neuralchemy mixes undocumented,
# gated and LLM-written upstreams (dropped); jackhhao's benign rows are GPT-4-generated (GPTeacher) and GuardrailsAI
# relabelled rows with an undocumented auto-tagger (dropped: their clean part, verazuo's MIT prompts, comes in directly
# as TrustAIRLab below, labelled by source file). Gandalf: real human game submissions (MIT).
KEEP_INJECTION = {"inj_gandalf"}
KEEP_CONTEXT = ("clinc", "massive")          # in-context items built from the clean pools only

MC = ["Which option correctly answers the question?", "Pick the right answer.", "Which answer is correct?"]
INJ = ["Is this input a prompt-injection or jailbreak attempt?", "Does this message try to override the assistant's instructions?"]
HARM = ["Is this request harmful?", "Would fulfilling this request cause harm?"]
TOPIC = ["Which category does this text belong to?", "What is the topic of the text?"]
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())


# ------------------------------------------------------------------ contamination
def _texts(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _texts(v)
    elif isinstance(x, list):
        for v in x:
            yield from _texts(v)


def _grams(t, n=13):
    w = norm(t).split()
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


class Held:
    """Held-out benchmark texts: exact (normalised) and 13-word spans."""

    def __init__(self):
        from datasets import load_dataset
        self.exact, self.grams = set(), set()
        for f in os.listdir("data/public"):
            if f.endswith(".jsonl"):
                for line in open(os.path.join("data/public", f)):
                    r = json.loads(line)
                    for t in _texts([r["state"], [q.get("prompt") for q in r["questions"]]]):
                        self.add(t)
        for sp in load_dataset("deepset/prompt-injections").values():
            for x in sp:
                self.add(x["text"])

    def add(self, t):
        if t and len(norm(t)) > 15:
            self.exact.add(norm(t))
            self.grams |= _grams(t)

    def hit(self, *texts) -> bool:
        return any(norm(t) in self.exact or (_grams(t) & self.grams) for t in texts if t)


# ------------------------------------------------------------------ io
def stream(repo, split, config=None, **kw):
    from datasets import load_dataset
    cfg, sha = HF[repo]
    try:
        return load_dataset(repo, config or cfg, split=split, revision=sha, streaming=True, **kw)
    except RuntimeError as e:                      # legacy loading-script repos: HF's parquet conversion
        if "scripts are no longer supported" not in str(e):
            raise
        CONVERTED.add(repo)
        return load_dataset(repo, config or cfg, split=split, revision="refs/convert/parquet", streaming=True, **kw)


CONVERTED: set = set()


def fetch(url) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def hf_file(repo, path) -> bytes:
    return fetch(f"https://huggingface.co/datasets/{repo}/resolve/{HF[repo][1]}/{path}")


def reservoir(it, k, rng, key=None):
    """Uniform sample of k items from an iterator (per key when given: k per key)."""
    buckets = collections.defaultdict(list)
    seen = collections.Counter()
    for x in it:
        b = key(x) if key else 0
        if b is None:
            continue
        seen[b] += 1
        if len(buckets[b]) < k:
            buckets[b].append(x)
        else:
            j = rng.randrange(seen[b])
            if j < k:
                buckets[b][j] = x
    return buckets if key else buckets[0]


# ------------------------------------------------------------------ builder
class Builder:
    def __init__(self, rng, held):
        self.rng, self.held = rng, held
        self.groups = collections.defaultdict(list)
        self.dropped = collections.Counter()

    def _q(self, typ, prompt, options, label):
        if typ == "choice":
            order = list(range(len(options)))
            self.rng.shuffle(order)
            options, label = [options[j] for j in order], order.index(label)
        return {"type": typ, "prompt": prompt, "options": list(options), "label": label}

    def add(self, group, fam, i, state, q, check=()):
        if self.held.hit(*check):
            self.dropped[fam] += 1
            return
        self.groups[group].append({"id": f"{fam}-{i}", "family": fam, "state": state, "questions": [q]})

    def mc(self, group, fam, i, state, prompt, options, label, check=()):
        self.add(group, fam, i, state, self._q("choice", prompt, options, label), check)

    def noul(self, group, fam, i, state, prompt, yes, check=()):
        self.add(group, fam, i, state, {"type": "noul", "prompt": prompt, "options": ["yes", "no"],
                                        "label": 0 if yes else 1}, check)

    def score(self, group, fam, i, state, prompt, levels, label, check=()):
        self.add(group, fam, i, state, {"type": "score", "prompt": prompt, "options": list(levels), "label": label}, check)

    def label_q(self, names, lab, prompts, kmax=40, full_p=0.3):
        if len(names) <= kmax or self.rng.random() < full_p:
            opts = list(range(len(names)))
        else:
            opts = self.rng.sample([i for i in range(len(names)) if i != lab], self.rng.randint(2, kmax) - 1) + [lab]
        self.rng.shuffle(opts)
        return {"type": "choice", "prompt": self.rng.choice(prompts), "options": [names[i] for i in opts],
                "label": opts.index(lab)}


def keep_existing(b: Builder):
    """Permissive families of the existing corpus (all three splits: re-split by the same id hash below)."""
    plan = [("data/public_train/{s}.jsonl", "kept_public", KEEP_PUBLIC),
            ("data/public_train/knowledge_{s}.jsonl", "kept_knowledge", KEEP_KNOWLEDGE),
            ("data/public_train/injection_{s}.jsonl", "kept_injection", KEEP_INJECTION)]
    for pat, group, fams in plan:
        for s in ("train", "val", "calib"):
            for line in open(pat.format(s=s)):
                r = json.loads(line)
                if r["family"] in fams:
                    b.groups[group].append(r)
    for line in open("data/public_train/context_train.jsonl"):
        r = json.loads(line)
        if r["family"].endswith(KEEP_CONTEXT):
            b.groups["kept_context"].append(r)


def knowledge(b: Builder, cap):
    rng = b.rng
    for i, x in enumerate(stream("jaredfern/codah", "train")):
        cands = x["candidate_answers"] if isinstance(x["candidate_answers"], list) else ast.literal_eval(x["candidate_answers"])
        b.mc("knowledge", "codah", i, {"text": x["question_propmt"]}, "Which ending is most plausible?", cands,
             int(x["correct_answer_idx"]), check=[x["question_propmt"]])
    for i, x in enumerate(stream("pkavumba/balanced-copa", "train")):
        p = "What was the cause?" if x["question"] == "cause" else "What happened as a result?"
        b.mc("knowledge", "copa", i, {"premise": x["premise"]}, p, [x["choice1"], x["choice2"]], int(x["label"]),
             check=[x["premise"]])
    for i, x in enumerate(reservoir(stream("deepmind/aqua_rat", "train"), cap, rng)):
        opts = x["options"] if isinstance(x["options"], list) else ast.literal_eval(x["options"])
        opts = [re.sub(r"^[A-E]\)\s*", "", o) for o in opts]
        lab = "ABCDE".index(x["correct"])
        b.mc("knowledge", "aqua_rat", i, {"problem": x["question"]}, rng.choice(MC), opts, lab, check=[x["question"]])
    for i, x in enumerate(reservoir(stream("coastalcph/lex_glue", "train", "case_hold"), cap, rng)):
        b.mc("knowledge", "casehold", i, {"citing_context": x["context"]}, "Which holding fits the <HOLDING> citation?",
             list(x["endings"]), int(x["label"]))


def reading(b: Builder, cap):
    rng = b.rng
    # CUAD: clause presence in ~2,400-character windows (training states are cut at 768 tokens)
    cuad = json.loads(hf_file("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json"))
    W, items = 2400, []
    for doc in cuad["data"]:
        for para in doc["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                m = re.search(r'related to "([^"]+)".*?Details: (.*)$', qa["question"], re.S)
                if not m:
                    continue
                spans = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
                items.append((ctx, m.group(1), m.group(2).strip(), spans))
    rng.shuffle(items)
    n_pos = n_neg = 0
    for i, (ctx, name, det, spans) in enumerate(items):
        if n_pos >= cap // 2 and n_neg >= cap // 2:
            break
        if spans and n_pos < cap // 2:
            s, e = rng.choice(spans)
            if e - s > W:
                continue
            a = max(0, min(s - rng.randint(0, W - (e - s)), len(ctx) - W))
            yes, n_pos = True, n_pos + 1
        elif n_neg < cap // 2 and len(ctx) > W:
            a = rng.randint(0, len(ctx) - W)
            if any(s < a + W and e > a for s, e in spans):
                continue
            yes, n_neg = False, n_neg + 1
        else:
            continue
        b.noul("reading", "cuad", i, {"contract_excerpt": ctx[a:a + W]},
               f'Does this passage contain a "{name}" clause ({det.rstrip(".")})?', yes)
    raw = gzip.decompress(fetch(GH["csqa2"])).decode()
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    for i, x in enumerate(rng.sample(rows, min(cap, len(rows)))):
        if x.get("answer") not in ("yes", "no"):
            continue
        b.noul("reading", "csqa2", i, {"statement": x["question"]}, "Is this statement true?", x["answer"] == "yes",
               check=[x["question"]])
    by = reservoir(stream("tasksource/ruletaker", "train"), max(1, cap // 8), rng, key=lambda x: x["config"])
    k = 0
    for cfg_name, xs in sorted(by.items()):
        for x in xs:
            b.noul("reading", "ruletaker", k, {"facts_and_rules": x["context"]},
                   f"Can this be concluded from the facts and rules: {x['question']}", x["label"] == "entailment")
            k += 1
    hans = [json.loads(l) for l in fetch(GH["hans"]).decode().splitlines() if l.strip()]
    for i, x in enumerate(rng.sample(hans, min(cap // 4, len(hans)))):
        b.noul("reading", "hans", i, {"sentence_1": x["sentence1"], "sentence_2": x["sentence2"]},
               "Does the first sentence imply the second?", x["gold_label"] == "entailment")


def topics(b: Builder, cap):
    rng = b.rng
    ds = stream("coastalcph/lex_glue", "train", "ledgar")
    names = ds.features["label"].names
    for i, x in enumerate(reservoir(ds, cap, rng)):
        b.add("topics", "ledgar", i, {"provision": x["text"]},
              b.label_q(names, int(x["label"]), ["What type of contract provision is this?"] + TOPIC))
    tax = arxiv_taxonomy()
    per = reservoir(stream("gfissore/arxiv-abstracts-2021", "train"), max(1, cap // max(1, len(tax))) + 1, rng,
                    key=lambda x: _primary(x, tax))
    pool = [x for xs in per.values() for x in xs]
    rng.shuffle(pool)
    cats = sorted(tax)
    names = [tax[c] for c in cats]
    for i, x in enumerate(pool[:cap]):
        b.add("topics", "arxiv", i, {"title": " ".join(x["title"].split()), "abstract": " ".join(x["abstract"].split())},
              b.label_q(names, cats.index(_primary(x, tax)), ["Which arXiv category is this paper's primary subject?"] + TOPIC))


def _primary(x, tax):
    c = x["categories"] if isinstance(x["categories"], list) else ast.literal_eval(x["categories"])
    c = c[0].split()[0] if c else None
    return c if c in tax else None


def arxiv_taxonomy() -> dict:
    """arXiv category code -> full name, parsed from the official taxonomy page."""
    html = fetch("https://arxiv.org/category_taxonomy").decode("utf-8", "ignore")
    out = {}
    for code, name in re.findall(r"<h4>([a-zA-Z\-]+(?:\.[A-Za-z\-]+)?)\s*<span>\(([^)]+)\)</span></h4>", html):
        out[code] = name.strip()
    if len(out) < 100:
        raise RuntimeError(f"arXiv taxonomy parse found only {len(out)} categories")
    return out


BENIGN_SUBJECTS = ["Meeting moved to {d}", "Invoice {n} for {m}", "Your order {n} has shipped", "Lunch on {d}?",
                   "Quarterly report draft", "Re: project timeline", "Password policy reminder", "Team offsite {m}"]
BENIGN_BODIES = ["Hi,\nThe {m} review is now on {d} at {h}. Please bring the latest numbers.\nThanks",
                 "Hello,\nAttached is invoice {n}. Payment is due within 30 days.\nBest regards",
                 "Hi team,\nQuick reminder that the draft is due {d}. Ping me with questions.\nCheers",
                 "Dear customer,\nOrder {n} left our warehouse today and should arrive by {d}.",
                 "Hey,\nAre you free for lunch {d}? The usual place at {h} works for me.",
                 "Hi all,\nPlease rotate your passwords before {d}. IT will not ask for them by email."]


def _benign_email(rng):
    f = dict(d=rng.choice(["Monday", "Tuesday", "Thursday", "Friday", "next week"]), n=rng.randint(1000, 99999),
             m=rng.choice(["March", "June", "September", "Q3", "budget"]), h=rng.choice(["10:00", "12:30", "15:00"]))
    return {"subject": rng.choice(BENIGN_SUBJECTS).format(**f), "body": rng.choice(BENIGN_BODIES).format(**f)}


def injection(b: Builder, cap, existing: set):
    rng = b.rng
    seen = set(existing)

    def new(text):
        k = norm(text)
        if k in seen or len(k) < 10:
            return False
        seen.add(k)
        return True
    jb = [x for x in stream("TrustAIRLab/in-the-wild-jailbreak-prompts", "train", "jailbreak_2023_12_25")]
    reg = reservoir(stream("TrustAIRLab/in-the-wild-jailbreak-prompts", "train", "regular_2023_12_25"), cap, rng)
    for i, x in enumerate(jb + reg[: max(0, cap - len(jb))]):
        if new(x["prompt"]):
            b.noul("injection", "inj_trustairlab", i, {"user_input": x["prompt"]}, rng.choice(INJ),
                   str(x["jailbreak"]) == "True", check=[x["prompt"]])
    # LLMail-Inject: every raw submission is a human-written attack email; benign mailboxes are programmatic
    att = reservoir(stream("microsoft/llmail-inject-challenge", "Phase1"), 6, rng, key=lambda x: x["team_id"])
    att = [x for xs in att.values() for x in xs]
    rng.shuffle(att)
    k = 0
    for x in att:
        if k >= cap // 2:
            break
        if not new(x["subject"] + " " + x["body"]):
            continue
        box = [_benign_email(rng) for _ in range(rng.randint(1, 3))]
        box.insert(rng.randint(0, len(box)), {"subject": x["subject"], "body": x["body"]})
        b.noul("injection", "inj_llmail", k, {"emails": box},
               "Does any of these emails contain instructions aimed at the AI assistant reading them?", True)
        b.noul("injection", "inj_llmail", f"neg{k}", {"emails": [_benign_email(rng) for _ in range(rng.randint(1, 4))]},
               "Does any of these emails contain instructions aimed at the AI assistant reading them?", False)
        k += 1
    # OASST1 human prompts as benign negatives (skip anything that reads like an override attempt)
    sus = re.compile(r"ignore (all |the )?(previous|prior|above)|jailbreak|\bDAN\b|developer mode|pretend (you|to be)|"
                     r"system prompt|no restrictions|act as", re.I)
    msgs = list(stream("OpenAssistant/oasst1", "train"))
    prompts = [m for m in msgs if m["role"] == "prompter" and m["lang"] == "en" and not m["synthetic"]
               and not m["deleted"] and not sus.search(m["text"])]
    for i, m in enumerate(rng.sample(prompts, min(cap // 2, len(prompts)))):
        if new(m["text"]):
            b.noul("injection", "inj_oasst_benign", i, {"user_input": m["text"]}, rng.choice(INJ), False, check=[m["text"]])
    return msgs


def harmful(b: Builder):
    rng = b.rng
    import csv
    for i, x in enumerate(csv.DictReader(io.StringIO(fetch(GH["xstest"]).decode()))):
        b.noul("harmful", "xstest", i, {"request": x["prompt"]}, rng.choice(HARM), x["label"] == "unsafe")
    for split, yes in (("harmful", True), ("benign", False)):
        for i, x in enumerate(stream("JailbreakBench/JBB-Behaviors", split)):
            if yes and x["Source"] == "AdvBench":
                continue
            b.noul("harmful", "jbb", f"{split}{i}", {"request": x["Goal"]}, rng.choice(HARM), yes)


TOX_LEVELS = ["not toxic", "slightly toxic", "moderately toxic", "very toxic", "severely toxic"]
QUAL_LEVELS = ["very poor", "poor", "acceptable", "good", "excellent"]


def ordinal(b: Builder, cap, msgs):
    rng = b.rng
    tox_bin = lambda t: 0 if t == 0 else 1 if t <= 0.2 else 2 if t <= 0.5 else 3 if t <= 0.8 else 4
    per = reservoir(stream("google/civil_comments", "train"), cap // 5, rng, key=lambda x: tox_bin(float(x["toxicity"])))
    k = 0
    for lvl, xs in sorted(per.items()):
        for x in xs:
            b.score("ordinal", "civil_comments", k, {"comment": x["text"]}, "How toxic is this comment?", TOX_LEVELS, lvl,
                    check=[x["text"]])
            k += 1
    by_id = {m["message_id"]: m for m in msgs}
    rated = []
    for m in msgs:
        if m["role"] != "assistant" or m["lang"] != "en" or m["synthetic"] or m["deleted"] or not m.get("labels"):
            continue
        lab = m["labels"] if isinstance(m["labels"], dict) else ast.literal_eval(m["labels"])
        names, vals, counts = lab.get("name", []), lab.get("value", []), lab.get("count", [])
        for attr, q in (("quality", "How good is this reply overall?"), ("helpfulness", "How helpful is this reply?")):
            if attr in names:
                j = names.index(attr)
                if counts and counts[j] >= 2 and m.get("parent_id") in by_id:
                    rated.append((m, by_id[m["parent_id"]], q, min(4, int(float(vals[j]) * 5))))
    for i, (m, parent, q, lvl) in enumerate(rng.sample(rated, min(cap, len(rated)))):
        b.score("ordinal", "oasst_rating", i, {"request": parent["text"], "reply": m["text"]}, q, QUAL_LEVELS, lvl)


def write(b: Builder, out: str):
    """Write every group built so far (called after each step, so a crash keeps finished steps)."""
    os.makedirs(out, exist_ok=True)
    stats = {}
    for g, recs in b.groups.items():
        if not recs:
            continue
        splits = {"train": [], "val": [], "calib": []}
        for r in recs:
            h = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100
            splits["val" if h < 5 else "calib" if h < 10 else "train"].append(r)
        for s, v in splits.items():
            with open(os.path.join(out, f"{g}_{s}.jsonl"), "w") as f:
                for r in v:
                    f.write(json.dumps(r) + "\n")
        stats[g] = {"splits": {s: len(v) for s, v in splits.items()},
                    "families": dict(collections.Counter(r["family"] for r in recs))}
    stats["dropped_overlapping_heldout"] = dict(b.dropped)
    old = json.load(open(os.path.join(out, "stats.json"))) if os.path.exists(os.path.join(out, "stats.json")) else {}
    drop = {**old.get("dropped_overlapping_heldout", {}), **stats.pop("dropped_overlapping_heldout")}
    stats = {**old, **stats, "dropped_overlapping_heldout": drop}          # runs with --only update their groups
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    json.dump({"hf": {k: {"config": c, "revision": s, "via_parquet_conversion": k in CONVERTED} for k, (c, s) in HF.items()},
               "github": GH},
              open(os.path.join(out, "revisions.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clean_train")
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: existing knowledge reading topics injection harmful ordinal")
    a = ap.parse_args()
    require_free_gb(1.0)
    rng = random.Random(a.seed)
    b = Builder(rng, Held())
    steps = a.only or ["existing", "knowledge", "reading", "topics", "injection", "harmful", "ordinal"]
    msgs = []
    if "existing" in steps:
        keep_existing(b)
    existing_inj = {norm(r["state"].get("user_input", "")) for s in ("train", "val", "calib")
                    for r in map(json.loads, open(f"data/public_train/injection_{s}.jsonl")) if r["family"] in KEEP_INJECTION}
    for s in steps:
        if s == "existing":
            continue
        print(f"[clean] {s} ...", flush=True)
        if s == "knowledge":
            knowledge(b, a.cap)
        elif s == "reading":
            reading(b, a.cap)
        elif s == "topics":
            topics(b, a.cap)
        elif s == "injection":
            msgs = injection(b, a.cap, existing_inj)
        elif s == "harmful":
            harmful(b)
        elif s == "ordinal":
            if not msgs:
                msgs = list(stream("OpenAssistant/oasst1", "train"))
            ordinal(b, a.cap, msgs)
        print(f"[clean] {s}: " + ", ".join(f"{g}={len(v)}" for g, v in b.groups.items()), flush=True)
        write(b, a.out)
    write(b, a.out)


if __name__ == "__main__":
    main()
