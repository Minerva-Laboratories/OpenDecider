"""Export the trained heads to shareable checkpoints (safetensors + config + model card) under checkpoints/.

    .venv/bin/python scripts/export_checkpoints.py

The backbone is not included: each checkpoint lists pinned public backbone variants that opendecider.hub downloads on
first use. Every number in a model card is read from the evaluation outputs listed in RELEASES (runs/, gitignored),
so cards and docs/RESULTS.md come from the same files.
"""
import json
import os
import sys

sys.path.insert(0, "src"); sys.path.insert(0, ".")
from opendecider.hub import export  # noqa: E402

QWEN2B = {"repo_id": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
AWQ2B = {"repo_id": "cyankiwi/Qwen3.5-2B-AWQ-4bit", "revision": "718fd9b52c5ca16f1f856f6c4e4209129802a6a2"}
QWEN9B = {"repo_id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
AWQ9B = {"repo_id": "cyankiwi/Qwen3.5-9B-AWQ-4bit", "revision": "156edc4bbeb8d1910ee7be9196bafaf1bc052156"}
GGUF9B = {"repo_id": "unsloth/Qwen3.5-9B-MTP-GGUF", "revision": "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
          "gguf_file": "Qwen3.5-9B-Q4_0.gguf", "config_repo": QWEN9B["repo_id"], "config_revision": QWEN9B["revision"]}
PUBLIC = [("banking77_8way", "Banking77 8-way"), ("prompt_injections", "Prompt-injection detection"),
          ("openbookqa", "OpenBookQA"), ("commonsenseqa", "CommonsenseQA"), ("pubmedqa", "PubMedQA")]

RELEASES = [
    {"name": "opendecider-2b", "title": "OpenDecider 2B", "base_model": QWEN2B["repo_id"], "run": "runs/x2b-clean3", "default": "int8",
     "variants": {"int8": ("int8", QWEN2B), "w8": ("w8", QWEN2B), "nf4": ("nf4", QWEN2B), "awq": ("awq", AWQ2B)},
     "bench": ("runs/quant/release_2b.json", "runs/quant/fresh/2b_{cfg}.json",
               {"int8": "int8:int8", "w8": "w8:int8", "nf4": "nf4:int8", "awq": "awq:int8"}),
     "evals": "x2b-clean3",
     "summary": "Trained from scratch for 1,250 steps on the clean corpus, on the deployed int8 backbone features, with "
                "VeRA adapters on every backbone layer (25.4M trainable parameters).",
     "backbone_note": f"frozen Qwen3.5-2B ({QWEN2B['repo_id']}, Apache-2.0)",
     "requirements": "CUDA GPU with about 2.5 GB free for states up to 1k tokens (3.1 GB at 15k). The `int8` variant "
                     "also runs on CPU (slow). Disk: 4.3 GB for the backbone download."},
    {"name": "opendecider-9b", "title": "OpenDecider 9B", "base_model": QWEN9B["repo_id"], "run": "runs/x9b-clean3", "default": "gguf-q4_0",
     # embedding table and LM head as Q4_0 blocks: same benchmark accuracy, 0.83 GB less (docs/RESULTS.md §16)
     "variants": {"gguf-q4_0": ("int8", GGUF9B, {"table_quant": "q4"})},
     "bench": ("runs/quant/release_9b.json", "runs/quant/serving/9b_{cfg}_q4t.json", {"gguf-q4_0": "int8:int8"}),
     "bench_note": "serving path (question cache, tuned fused int8 kernels), Q4_0 embedding table and LM head, Orin in "
                   "50 W mode",
     "evals": "x9b-clean3",
     "summary": "The 2B checkpoint's trunk and heads moved to the 9B with a closed-form ridge map between the two backbones' "
                "features, then trained for 1,500 steps on the same corpus with VeRA adapters on the top 16 layers "
                "(58.1M trainable parameters).",
     "backbone_note": f"frozen Qwen3.5-9B, Q4_0 GGUF from {GGUF9B['repo_id']} (Apache-2.0)",
     "requirements": "CUDA GPU with about 9 GB free for states up to 1k tokens (10.2 GB at 15k). Disk: 5.6 GB for the "
                     "GGUF backbone."},
    {"name": "opendecider-9b-awq", "title": "OpenDecider 9B AWQ (4-bit)", "base_model": QWEN9B["repo_id"], "run": "runs/x2b-clean3-stitch9b-awq",
     "default": "awq", "variants": {"awq": ("awq", AWQ9B, {"table_quant": "q4"})},
     "bench": ("runs/quant/9b_clean3_awq.json", "runs/quant/serving/9b_awq_{cfg}_q4t.json", {"awq": "awq:int8"}),
     "bench_note": "serving path (question cache), Q4_0 embedding table and LM head, Orin in 50 W mode",
     "evals": "x2b-clean3-stitch9b-awq",
     "summary": "The 2B checkpoint's trunk and heads moved to the 4-bit AWQ 9B with a closed-form ridge map fitted on the "
                "AWQ backbone's own features (no gradient steps on the 9B). About 3 GB less GPU memory than "
                "opendecider-9b and faster on short states, 1 to 2 points lower on knowledge-heavy multiple choice.",
     "backbone_note": f"frozen Qwen3.5-9B, AWQ int4 from {AWQ9B['repo_id']} (Apache-2.0)",
     "requirements": "CUDA GPU with about 6 GB free for states up to 1k tokens (7.3 GB at 15k). Disk: about 8.5 GB for "
                     "the AWQ backbone."},
]

DATA = """## Training data and licenses

Clean provenance: no share-alike, non-commercial, unlicensed or gated sources, and no labels produced by a model.
Pinned revisions and the full review (including rejected sources) are in `data/MANIFEST.md` of the repository.

| Source | License | Used for |
|---|---|---|
| Programmatic synthetic tasks (this repository) | Apache-2.0 | decisions over generated records, uncertainty items |
| clinc/clinc_oos | CC-BY-3.0 | intents |
| AmazonScience/massive (en-US) | CC-BY-4.0 | intents |
| allenai/winogrande | Apache-2.0 | commonsense |
| allenai/qasc, allenai/cosmos_qa, allenai/quartz | CC-BY-4.0 | knowledge multiple choice |
| ChilleD/StrategyQA | MIT | yes/no |
| jaredfern/codah | ODC-BY | commonsense multiple choice |
| pkavumba/balanced-copa | CC-BY-4.0 (COPA: BSD-2-Clause) | causal multiple choice |
| deepmind/aqua_rat | Apache-2.0 | algebra multiple choice |
| coastalcph/lex_glue (case_hold, ledgar) | CC-BY-4.0 | legal multiple choice, 100-class topics |
| gfissore/arxiv-abstracts-2021 | CC0 (arXiv metadata terms) | ~150-class topics |
| theatticusproject/cuad | CC-BY-4.0 | clause presence, unanswerable questions |
| allenai/csqa2 (GitHub) | CC-BY-4.0 | commonsense yes/no |
| tasksource/ruletaker | Apache-2.0 | rule entailment |
| tommccoy1/hans (GitHub) | MIT | entailment |
| TrustAIRLab/in-the-wild-jailbreak-prompts | MIT | jailbreak detection |
| microsoft/llmail-inject-challenge | MIT | indirect injection in emails |
| Lakera/gandalf_ignore_instructions | MIT | injection detection |
| paul-rottger/xstest (GitHub), JailbreakBench/JBB-Behaviors | CC-BY-4.0, MIT | harmful-request detection |
| OpenAssistant/oasst1 | Apache-2.0 | benign prompts, human quality ratings |
| google/civil_comments | CC0 | toxicity ratings |

Generated data: 6,000 prompt-injection attacks written by Qwen3.5-9B (Apache-2.0) into human-written texts from the
sources above, half of them made hard to notice. The model only wrote the attacks; every label comes from how the
example was built (a code-chosen string must appear in the attack, and the untouched text is the paired negative).

The evaluation benchmarks (Banking77, deepset/prompt-injections, OpenBookQA, CommonsenseQA, PubMedQA,
typed-decisions) were never used for training, tuning or data generation; training texts that overlap them were
removed.

## License

Apache-2.0, like the code. The backbone keeps its own license (Apache-2.0 for every variant listed above). CC-BY
sources are credited in the table above."""

FRAMING = """OpenDecider is a public hypothesis test of a "System One" decision model in the style of TypeSafe's Jev. Results
are reported only as consistent or inconsistent with public observations; nothing here describes how Jev is built.
This checkpoint is a research release, not a product."""


def _ci(v):
    return f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def results(rel, ck):
    """Markdown tables for one release, read from its evaluation outputs."""
    from eval.typed_decisions import load_preds, print_metrics
    tag = rel["evals"]
    rows = []
    for key, label in PUBLIC:
        raw = json.load(open(f"runs/public-{tag}/eval_{key}.json"))["raw"]
        rows.append(f"| {label} | {_ci(raw['accuracy'])} | {raw['nll']['value']:.3f} |")
    td = print_metrics(tag, list(load_preds(f"runs/td-{tag}").values()))
    none_path = f"runs/none-{tag}/results.json"
    none_md = ""
    if os.path.exists(none_path):
        au = {k: v["value"] for k, v in json.load(open(none_path))["report"]["pooled"]["auroc_none"].items()}
        none_md = (f"`none` option (\"none of the above\", on by default), AUROC of P(none) against answerable questions on "
                   f"the five public benchmarks: correct option removed {au['gold_removed']:.3f}, only plausible wrong "
                   f"options left {au['hard']:.3f}, state unrelated to the question {au['unrelated']:.3f}.")
    conf = ck["extra"].get("conformal", {}).get("lac", {})
    n_conf = ", ".join(f"{k} {len(v)}" for k, v in sorted(conf.items()))
    return f"""## Results (zero-shot, default variant)

Accuracy with 95% bootstrap confidence intervals (1,000 resamples).

| Benchmark | Accuracy | NLL |
|---|---|---|
{chr(10).join(rows)}

Secondary, not a correctness benchmark: on typed-decisions (2,000 decisions whose gold labels are the answers of a ~4B
teacher model, so the score measures agreement with that teacher; the dataset card lists the teacher's self-agreement at 0.735)
this checkpoint scores {td['accuracy'][0]:.3f} [{td['accuracy'][1]:.3f}, {td['accuracy'][2]:.3f}] (KL to the teacher
distribution {td['kl'][0]:.3f}). We do not train or tune on it.

{none_md}

Conformal prediction sets (`"conformal": {{"alpha": 0.1}}` in a request) use calibration scores stored in this
checkpoint ({n_conf} questions from the training families' calibration split). Coverage holds for requests like that
split; for other distributions, refit with a few dozen labels (`POST /v1/calibrate`)."""


def variants_table(rel):
    """Memory and latency from one fresh process per config; accuracy from the shared-process run (memory left by one
    config inflates the next in a shared process, accuracy is unaffected)."""
    acc_path, mem_pat, cfg_of = rel["bench"]
    if not os.path.exists(acc_path):
        return ""
    acc = json.load(open(acc_path))
    rows = []
    for name, (wq, src, *_) in rel["variants"].items():
        cfg = cfg_of[name]
        mem_path = mem_pat.format(cfg=cfg.replace(":", "_"))
        if cfg not in acc or not os.path.exists(mem_path):
            continue
        m, a = next(iter(json.load(open(mem_path)).values())), acc[cfg]
        rows.append(f"| `{name}`{' (default)' if name == rel['default'] else ''} | {src['repo_id']} | {m['weights_gb']:.2f} GB | "
                    f"{m['peak_gb_20']:.1f} / {m['peak_gb_400']:.1f} GB | {m['latency_s_20']:.2f} / {m['latency_s_400']:.2f} s | "
                    f"{a['typed']['acc']:.3f} / {a['banking77_8way']:.3f} / {a['prompt_injections']:.3f} |")
    return f"""## Backbone variants

Jetson AGX Orin 64 GB, int8 KV cache, {rel.get("bench_note", "eager kernels")}; one fresh process per variant for
memory and latency.
Accuracy: 500 typed-decisions questions (secondary) / Banking77 8-way / prompt-injection detection. The head was
trained on the default variant's features.

| Variant | Backbone source | Weights on GPU | Peak GPU memory, 0.8k / 15k-token state | Latency, same states | Accuracy |
|---|---|---|---|---|---|
{chr(10).join(rows)}"""


def card(rel, ck):
    return f"""---
license: apache-2.0
library_name: opendecider
base_model: {rel['base_model']}
pipeline_tag: text-classification
tags: [decision-model, classification, calibration, qwen3.5]
---

# {rel['title']}

Calibrated typed decisions (choice, yes/no, ordinal score) with an explicit "none of the above" option, on a
{rel['backbone_note']}. {rel['summary']} This checkpoint contains the trained trunk, heads and adapters (fp16
safetensors). The backbone is downloaded from its original repository at a pinned revision on first use.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/{rel['name']}")
print(dec.decide({{"state": {{"ticket": "I was charged twice and want my money back."}},
                   "questions": {{"route": {{"type": "choice", "prompt": "Which team handles this?",
                                            "options": ["billing", "technical", "refund", "other"]}}}}}}))
```

Or serve it: `python -m opendecider.serve --model checkpoints/{rel['name']}`.

{results(rel, ck)}

{variants_table(rel)}

## Requirements

{rel['requirements']}

{FRAMING}

{DATA}
"""


def variants(base_cfg, entries):
    """entries: name -> (weight_quant, source[, extra BackboneConfig fields, e.g. {"table_quant": "q4"}])."""
    keep = ("dtype", "kv_quant", "quantize_linear_state", "feature_layers")
    return {name: {**{k: base_cfg[k] for k in keep if k in base_cfg}, "weight_quant": e[0], "source": e[1],
                   **(e[2] if len(e) > 2 else {})}
            for name, e in entries.items()}


def main():
    import torch
    for rel in RELEASES:
        ck = torch.load(f"{rel['run']}/model.pt", map_location="cpu", weights_only=False)
        out = f"checkpoints/{rel['name']}"
        export(f"{rel['run']}/model.pt", out, variants(ck["backbone_cfg"], rel["variants"]), rel["default"], card(rel, ck),
               info={"name": rel["name"], "trained_from": f"{rel['run']} (step {ck['extra'].get('step')})"})
        files = sorted(os.listdir(out))
        print(out, [(f, round(os.path.getsize(os.path.join(out, f)) / 2 ** 20, 1)) for f in files])


if __name__ == "__main__":
    main()
