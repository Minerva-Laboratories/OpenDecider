"""Skipped prompt pass (src/opendecider/skipkv.py) on ordinary LLM tasks: do answers change beyond the model's own
rounding noise?

Arms: exact; floor (the exact model with its prompt pass in 32-token slices: same math, different rounding: how often
answers flip with no approximation at all); cut C (prompt tokens skip the layers above C, maps from
scripts/skipkv_lm.py, fitted on FineWeb-Edu; the last prompt token and every generated token run all layers).
Tasks (chat template, thinking off):
  openbookqa, commonsenseqa  multiple choice, answer = the letter with the highest next-token probability
  pubmedqa                   abstract + question, yes / no by next-token probability (longer prompts)
  gsm8k                      greedy generation (--max-new tokens), final number after the last "####" or the last
                             number in the reply (openai/gsm8k test, MIT; evaluation only)
Reported per task and arm: accuracy, and the share of items whose answer differs from the exact arm's ("flips").

    python scripts/skipkv_tasks.py --ckpt runs/x4b-mercurius-lora/model.pt --cuts 24 16 \
        --out /mnt/ext/opendecider/skipkv/tasks_4b.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps  # noqa: E402
from skipkv_lm import continue_logits, prompt_cache  # noqa: E402

LETTERS = "ABCDE"


def chat(bb, user):
    try:
        return bb.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    except TypeError:
        return bb.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def mc_items(path, n):
    out = []
    for l in open(path):
        r = json.loads(l)
        q = r["questions"][0]
        st = r["state"]
        if "abstract" in st:                                      # pubmedqa: yes / no
            if q["options"][q["label"]] not in ("yes", "no"):
                continue
            text = (f"Abstract:\n{st['abstract']}\n\nQuestion: {st.get('question', q['prompt'])}\n"
                    "Answer with yes or no only.")
            out.append((text, ["yes", "no"], ["yes", "no"].index(q["options"][q["label"]])))
        else:
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
            text = f"{st['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."
            out.append((text, list(LETTERS[: len(q["options"])]), q["label"]))
        if len(out) == n:
            break
    return out


def gsm_items(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir="/mnt/ext/opendecider/hf_datasets")
    out = []
    for r in ds.select(range(n)):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append((r["question"] + "\n\nSolve step by step, then give the final answer after '####'.", ans))
    return out


def number(text):
    m = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", m)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


class Arm:
    def __init__(self, bb, maps=None, chunk=None):
        self.bb, self.maps, self.chunk = bb, maps, chunk

    def cache(self, ids):
        bb = self.bb
        prev = bb.skip_state, bb.cfg.prefill_chunk
        bb.skip_state = self.maps
        if self.chunk:
            bb.cfg.prefill_chunk = self.chunk
        try:
            return prompt_cache(bb, ids)
        finally:
            bb.skip_state, bb.cfg.prefill_chunk = prev

    @torch.no_grad()
    def next_logits(self, ids):
        lg, _ = continue_logits(self.bb, self.cache(ids[:-1]), ids[-1:])
        return lg[-1]

    @torch.no_grad()
    def generate(self, ids, max_new, stop_ids):
        cache, last, out = self.cache(ids[:-1]), ids[-1], []
        for _ in range(max_new):
            lg, cache = continue_logits(self.bb, cache, [last])
            last = int(lg[-1].argmax())
            if last in stop_ids:
                break
            out.append(last)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cuts", nargs="+", type=int, default=[24, 16])
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--n-mc", type=int, default=300)
    ap.add_argument("--n-gsm", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tasks", nargs="+", default=["openbookqa", "commonsenseqa", "pubmedqa", "gsm8k"])
    ap.add_argument("--variants", nargs="*", default=[],
                    help="extra arms per cut: tail<R>, ema, ema_tail<R> (ema maps from scripts/skipkv_lm.py --variants ema)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.9, 0.99])
    ap.add_argument("--no-base", action="store_true", help="drop the plain cut<C> arms (variants only)")
    ap.add_argument("--gsm-arms", nargs="+", default=None, help="arms run on gsm8k (generation is slow); default all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gpu_lock("skipkv_tasks"):
        model, _ = load_model(a.ckpt)
        bb = model.backbone
        name = os.path.basename(os.path.dirname(a.ckpt))
        arms = {"exact": Arm(bb), "floor": Arm(bb, chunk=32)}
        for c in a.cuts:
            if not a.no_base:
                arms[f"cut{c}"] = Arm(bb, SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_cut{c}_normed.pt"), bb.device))
            for v in a.variants:
                f = f"{name}_lm_cut{c}_normed" + (f"_ema{'-'.join(map(str, a.alphas))}" if v.startswith("ema") else "")
                m = re.search(r"tail(\d+)", v)
                arms[f"cut{c}_{v}"] = Arm(bb, SkipMaps.load(os.path.join(a.maps_dir, f + ".pt"), bb.device,
                                                            tail=int(m.group(1)) if m else 0))
        enc = lambda t: bb.tok(t, add_special_tokens=False)["input_ids"]
        res = {"ckpt": a.ckpt}
        for task in a.tasks:
            t0 = time.time()
            if task == "gsm8k":
                items = gsm_items(a.n_gsm)
                stop = {bb.tok.eos_token_id} | {bb.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
                                                 if bb.tok.convert_tokens_to_ids(t) is not None}
                garms = {k: v for k, v in arms.items() if a.gsm_arms is None or k in a.gsm_arms or k == "exact"}
                ans = {k: [] for k in garms}
                gens = {k: [] for k in garms}
                for q, gold in items:
                    ids = enc(chat(bb, q))
                    for k, arm in garms.items():
                        g = arm.generate(ids, a.max_new, stop)
                        gens[k].append(g)
                        ans[k].append(number(bb.tok.decode(g)))
                golds = [g for _, g in items]
                r = {}
                for k in garms:
                    r[k] = {"acc": float(np.mean([x == g for x, g in zip(ans[k], golds)])),
                            "answer_flips": float(np.mean([x != y for x, y in zip(ans[k], ans["exact"])])),
                            "generation_identical": float(np.mean([x == y for x, y in zip(gens[k], gens["exact"])])),
                            "mean_len": float(np.mean([len(x) for x in gens[k]]))}
            else:
                path = f"data/public/{task}.jsonl"
                items = mc_items(path, a.n_mc)
                pred = {k: [] for k in arms}
                logits = {k: np.full((len(items), len(LETTERS)), -1e4, dtype=np.float32) for k in arms}
                for i, (text, labels, gold) in enumerate(items):
                    ids = enc(chat(bb, text))
                    tok = [enc(l)[0] for l in labels]
                    for k, arm in arms.items():
                        z = arm.next_logits(ids)[tok].float().cpu().numpy()
                        logits[k][i, : len(z)] = z
                        pred[k].append(int(z.argmax()))
                np.savez(a.out.replace(".json", f"_{task}_logits.npz"), gold=np.array([g for _, _, g in items]),
                         **logits)                                # option logits per arm (scripts/skipkv_calib.py)
                golds = [g for _, _, g in items]
                r = {k: {"acc": float(np.mean([p == g for p, g in zip(pred[k], golds)])),
                         "answer_flips": float(np.mean([p != q for p, q in zip(pred[k], pred["exact"])]))}
                     for k in arms}
            r["n"], r["seconds"] = len(items), time.time() - t0
            res[task] = r
            print(task, json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
