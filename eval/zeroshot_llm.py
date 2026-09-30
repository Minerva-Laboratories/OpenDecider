"""Zero-shot baseline through an OpenAI-compatible LLM server (e.g. the llama.cpp Qwen3.5-9B): options as letters,
read the next-token log-probabilities of the letters, renormalise over the options. No training.

    .venv/bin/python -m eval.zeroshot_llm --data data/public/openbookqa.jsonl ... --name zs-9b [--limit N]

Writes runs/<name>/preds_<split>.jsonl in the same format as eval.evaluate --save-preds (id, label, logits=log p,
options), plus eval_<split>.json metrics, so eval.profile_cv and ensembling work unchanged.
Only questions with <= 26 options (letters A-Z) are scored; others are skipped and counted.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import string
import time
import urllib.request

from opendecider.batching import state_texts

from .metrics import summarize

LETTERS = string.ascii_uppercase
SYSTEM = "You are a careful, accurate classifier. Reply with the letter of the correct option only."


def build_prompt(state, q) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return (f"{state_texts([state])[0]}\n\nQuestion: {q['prompt']}\nOptions:\n{opts}\n\n"
            f"Answer with a single letter ({LETTERS[0]}-{LETTERS[len(q['options']) - 1]}).")


def letter_logprobs(base, key, prompt, K, top=20, slot=None):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": top, "cache_prompt": False}
    if slot is not None:
        body["id_slot"] = slot
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    tops = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {}
    for t in tops:
        tok = t["token"].strip()
        if len(tok) == 1 and tok in LETTERS[:K]:
            lp[tok] = max(lp.get(tok, -1e9), t["logprob"])
    floor = min(t["logprob"] for t in tops) - 2.0             # letters outside the top list: below the smallest seen
    raw = [lp.get(LETTERS[i], floor) for i in range(K)]
    m = max(raw)
    z = math.log(sum(math.exp(x - m) for x in raw)) + m
    return [x - z for x in raw], out.get("usage", {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-file", default=os.path.expanduser("~/.config/qwen-server/api_key"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    key = open(a.key_file).read().strip() if os.path.exists(a.key_file) else ""
    out_dir = os.path.join("runs", a.name)
    os.makedirs(out_dir, exist_ok=True)
    for path in a.data:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if a.limit:
            recs = recs[: a.limit]
        jobs, skipped = [], 0
        for r in recs:
            for j, q in enumerate(r["questions"]):
                if q.get("label") is None:
                    continue
                if len(q["options"]) > len(LETTERS):
                    skipped += 1
                    continue
                jobs.append((f"{r['id']}:{j}", r.get("split", os.path.basename(path)[:-6]), r["state"], q))
        if not jobs:
            print(f"[{os.path.basename(path)}] nothing to score ({skipped} questions with > 26 options skipped)")
            continue
        t0 = time.time()

        def run(job):
            jid, split, state, q = job
            lp, _ = letter_logprobs(a.base_url, key, build_prompt(state, q), len(q["options"]))
            return {"id": jid, "split": split, "label": int(q["label"]), "logits": lp, "options": list(q["options"])}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(run, jobs))
        split = rows[0]["split"]
        with open(os.path.join(out_dir, f"preds_{split}.jsonl"), "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        probs = [[math.exp(v) for v in x["logits"]] for x in rows]
        m = summarize(probs, [x["label"] for x in rows], n_boot=a.n_boot)
        json.dump({"split": split, "n_skipped_over_26": skipped, "raw": m, "elapsed_s": time.time() - t0},
                  open(os.path.join(out_dir, f"eval_{split}.json"), "w"), indent=1, default=str)
        print(f"[{split}] n={m['n']} acc={m['accuracy']['value']:.4f} [{m['accuracy']['lo']:.4f},{m['accuracy']['hi']:.4f}] "
              f"nll={m['nll']['value']:.4f} ece={m['ece']['value']:.4f} ({time.time() - t0:.0f}s, {skipped} skipped)", flush=True)


if __name__ == "__main__":
    main()
