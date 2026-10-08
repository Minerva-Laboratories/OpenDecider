"""Zero-shot letter scores from an in-process Hugging Face / PyTorch causal LM, same prompt and scoring as
eval/zeroshot_llm.py (options as letters, next-token log-probabilities of the letters renormalised over the options).

    .venv/bin/python -m eval.zeroshot_hf --model orig --name zs-4b --data data/public/*.jsonl
    PYTHONPATH=<mercurius>/src MERCURIUS_ROOT=<dir> .venv/bin/python -m eval.zeroshot_hf --model mercurius ...

--model:
  orig       Qwen3.5-4B from --path, bf16
  orig-nf4   the same, NF4 4-bit weights (Mercurius's same-precision baseline, mercurius.eval.retrieval_ab)
  mercurius  Mercurius-1-4B rebuilt with mercurius.eval.retrieval_ab.build and the deployed 4-bit settings
  mercurius-deployed  Mercurius's deployed form (opendecider.mercurius_port.deploy_mercurius), MERCURIUS_ROOT
Writes runs/<name>/preds_<split>.jsonl and eval_<split>.json like eval.zeroshot_llm. Questions with more than 26
options are skipped and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import types

import torch

from opendecider.guards import gpu_lock

from .metrics import summarize
from .zeroshot_llm import LETTERS, SYSTEM, build_prompt


class _TiedLM(torch.nn.Module):
    """Text model + tied head (embedding rows) -> .logits, for the deployed Mercurius (its text model only)."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, input_ids):
        h = self.trunk(input_ids=input_ids).last_hidden_state
        E = self.trunk.embed_tokens
        W = E.dequant_rows(slice(None), h.dtype) if hasattr(E, "dequant_rows") else E.weight
        return types.SimpleNamespace(logits=h @ W.to(h.dtype).t())


def load(a):
    if a.model == "mercurius-deployed":  # Mercurius's deployed form (NF4 merged weights, factored gates, TurboQuant KV)
        from opendecider.mercurius_port import load_deployed
        trunk, tok = load_deployed(os.environ["MERCURIUS_ROOT"])          # compact storage + compressed KV cache
        nbytes = sum(t.numel() * t.element_size() for t in list(trunk.parameters()) + list(trunk.buffers()))
        print(f"deployed text model: {nbytes / 2**30:.2f} GiB of parameters and buffers", flush=True)
        return _TiedLM(trunk).eval(), tok
    if a.model == "orig":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(a.path, dtype=torch.bfloat16, device_map="cuda").eval()
        return m, AutoTokenizer.from_pretrained(a.path)
    from transformers import AutoTokenizer
    from mercurius.eval.retrieval_ab import build, build_original_nf4
    from mercurius.paths import STAGE_AB
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    if a.model == "orig-nf4":
        return build_original_nf4().eval(), tok
    # the deployed configuration, as in mercurius experiments/eval_v3_bench.sh
    qat = {"kv_bits": 4, "kv_group": 32, "kv_quant": "tq", "kv_rot": "none", "gate_bits": 16, "embed_bits": 4}
    here = os.getcwd()
    os.chdir(os.environ["MERCURIUS_ROOT"])          # mercurius reads ckpt/tq_rotations_<key>.pt relative to the cwd
    try:
        m = build(a.adapters, 512, a.covs, quantize=True, groups=a.groups, dial="c0", qat=qat)
    finally:
        os.chdir(here)
    import transformers
    if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 7):
        from opendecider.mercurius_port import port_linear_attention
        print("mercurius port:", port_linear_attention(m), flush=True)       # its KDA forward targets 5.6
    return m.eval(), tok


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["orig", "orig-nf4", "mercurius", "mercurius-deployed"], required=True)
    ap.add_argument("--path", default="/mnt/ext/opendecider/models/qwen3.5-4b")
    ap.add_argument("--adapters", default="")
    ap.add_argument("--covs", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    with gpu_lock("zeroshot_hf"):
        model, tok = load(a)
        letter_ids = [tok.convert_tokens_to_ids(c) for c in LETTERS]
        out_dir = f"runs/{a.name}"
        os.makedirs(out_dir, exist_ok=True)
        for path in a.data:
            split = os.path.splitext(os.path.basename(path))[0]
            recs = [json.loads(l) for l in open(path)][: a.limit or None]
            rows, skipped, t0 = [], 0, time.time()
            for r in recs:
                q = r["questions"][0]
                K = len(q["options"])
                if K > len(LETTERS):
                    skipped += 1
                    continue
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(r["state"], q)}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                logits = model(input_ids=ids).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)[letter_ids[:K]]
                rows.append({"id": r["id"], "split": split, "label": int(q["label"]), "logits": lp.tolist(),
                             "options": q["options"]})
            with open(f"{out_dir}/preds_{split}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            probs = [torch.softmax(torch.tensor(r["logits"]), -1).tolist() for r in rows]
            res = {"split": split, "n_skipped_over_26": skipped,
                   "raw": summarize(probs, [r["label"] for r in rows], n_boot=a.n_boot),
                   "meta": {"model": a.model, "path": a.path, "adapters": a.adapters, "seconds": time.time() - t0}}
            json.dump(res, open(f"{out_dir}/eval_{split}.json", "w"), indent=1)
            print(split, len(rows), "acc", round(res["raw"]["accuracy"]["value"], 3), flush=True)


if __name__ == "__main__":
    main()
