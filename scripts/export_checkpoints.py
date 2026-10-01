"""Export the trained heads to shareable checkpoints (safetensors + config + model card) under checkpoints/.

    .venv/bin/python scripts/export_checkpoints.py

The backbone is not included: each checkpoint lists pinned public backbone variants that opendecider.hub downloads
on first use. Numbers in the cards come from docs/RESULTS.md.
"""
import os
import sys

import torch

sys.path.insert(0, "src")
from opendecider.hub import export  # noqa: E402

QWEN2B = {"repo_id": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"}
AWQ2B = {"repo_id": "cyankiwi/Qwen3.5-2B-AWQ-4bit", "revision": "718fd9b52c5ca16f1f856f6c4e4209129802a6a2"}
QWEN9B = {"repo_id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
GGUF9B = {"repo_id": "unsloth/Qwen3.5-9B-MTP-GGUF", "revision": "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
          "gguf_file": "Qwen3.5-9B-Q4_0.gguf", "config_repo": QWEN9B["repo_id"], "config_revision": QWEN9B["revision"]}

DATA = """## Training data and licenses

The head was trained on programmatic synthetic data (this repository, no LLM teacher) and on these public datasets
(pinned revisions in `data/MANIFEST.md`). Attribution as required by their licenses:

| Dataset | License |
|---|---|
| clinc/clinc_oos | CC-BY-3.0 |
| AmazonScience/massive (en-US) | CC-BY-4.0 |
| google/boolq | CC-BY-SA-3.0 |
| stanfordnlp/snli | CC-BY-SA-4.0 |
| allenai/ai2_arc | CC-BY-SA-4.0 |
| fancyzhx/dbpedia_14 | CC-BY-SA-3.0 |
| allenai/winogrande | Apache-2.0 |
| jackhhao/jailbreak-classification, neuralchemy/Prompt-injection-dataset | Apache-2.0 |
| reshabhs/SPML_Chatbot_Prompt_Injection, GuardrailsAI/detect-jailbreak, Lakera/gandalf_ignore_instructions | MIT |

Non-commercial datasets were excluded. The evaluation benchmarks were never used for training.

## License

The weights in this checkpoint are released under Apache-2.0, like the code. Four training sets are CC-BY-SA
(share-alike). Whether trained weights are "adapted material" under CC-BY-SA is legally unsettled; this head is a
classifier that does not generate text and cannot reproduce its training sentences. If that matters for your use,
consult your own counsel. The backbone keeps its own license (Apache-2.0 for every variant listed above).
"""

FRAMING = """OpenDecider is a public hypothesis test of a "System One" decision model in the style of TypeSafe's Jev. Results
are reported only as consistent or inconsistent with public observations; nothing here describes how Jev is built.
This checkpoint is a research release, not a product."""


def card(title, body):
    return f"""---
license: apache-2.0
library_name: opendecider
tags: [decision-model, classification, calibration, qwen3.5]
---

# {title}

{body}

{FRAMING}

{DATA}"""


CARD_2B = card("OpenDecider 2B", f"""Calibrated typed decisions (choice, yes/no, ordinal score) from a frozen Qwen3.5-2B backbone. This checkpoint
contains the trained trunk, heads and adapters (25.4M parameters, fp16 safetensors). The backbone is downloaded from
its original repository at a pinned revision on first use.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-2b")            # or backbone="awq" | "nf4" | "w8"
print(dec.decide({{"state": {{"ticket": "I was charged twice and want my money back."}},
                   "questions": {{"route": {{"type": "choice", "prompt": "Which team handles this?",
                                            "options": ["billing", "technical", "refund", "other"]}}}}}}))
```

Or serve it: `python -m opendecider.serve --model checkpoints/opendecider-2b`.

## Backbone variants

All on a Jetson AGX Orin 64 GB; accuracy on 500 typed-decisions questions / Banking77 8-way / prompt-injection
detection (`docs/RESULTS.md` §12). The head was trained on int8 features: 4-bit backbones lose accuracy until the
head is retrained on them.

| Variant | Backbone source | Weights on GPU | Accuracy | Latency, 0.8k / 3.7k / 15k-token state |
|---|---|---|---|---|
| `int8` (default) | {QWEN2B['repo_id']} | 1.88 GB | 0.434 / 0.825 / 0.776 | 0.80 / 1.47 / 4.59 s |
| `w8` (GemLite) | {QWEN2B['repo_id']} | 1.88 GB | 0.430 / 0.812 / 0.767 | 0.66 / 1.80 / 6.35 s |
| `awq` (int4) | {AWQ2B['repo_id']} | 1.32 GB | 0.420 / 0.750 / 0.776 | not measured eagerly |
| `nf4` | {QWEN2B['repo_id']} | 1.26 GB | 0.414 / 0.775 / 0.698 | not measured eagerly |

## Results (default variant, zero-shot)

| Benchmark | Accuracy |
|---|---|
| typed-decisions (2,000 decisions; KL 0.324, Brier 0.168) | 0.577 |
| Banking77 8-way | 0.819 |
| Prompt-injection detection | 0.767 |
| OpenBookQA | 0.686 |
| CommonsenseQA | 0.639 |
| PubMedQA | 0.849 |

`none` option ("none of the above", on by default): AUROC 0.845 when the correct option is removed, 0.795 for an
unrelated state.

## Requirements

CUDA GPU with about 2.5 GB free for states up to 1k tokens (3.1 GB at 15k). The `int8` variant also runs on CPU
(slow). Disk: 4.3 GB for the backbone download.""")

CARD_9B = card("OpenDecider 9B (stitched)", f"""The OpenDecider 2B head moved onto a frozen Qwen3.5-9B backbone with a closed-form ridge map ("stitch"), with
no gradient steps. This checkpoint contains the trunk, heads and the stitch map (57.3M parameters, fp16
safetensors). The default backbone is the Q4_0 GGUF from {GGUF9B['repo_id']} (5.6 GB), loaded directly by
OpenDecider; the stitch was fitted on those weights.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-9b-stitched")
```

## Backbone variants

| Variant | Backbone source | Status |
|---|---|---|
| `gguf-q4_0` (default) | {GGUF9B['repo_id']} (`{GGUF9B['gguf_file']}`) + config/tokenizer from {QWEN9B['repo_id']} | evaluated |
| `int8` | {QWEN9B['repo_id']} (18 GB bf16 download, int8 at load) | not evaluated: the stitch was fitted on the GGUF weights |

## Results (default variant, zero-shot)

| Benchmark | Accuracy |
|---|---|
| typed-decisions (2,000 decisions; KL 0.240, Brier 0.135) | 0.643 |
| Banking77 8-way | 0.869 |
| Prompt-injection detection | 0.750 |
| OpenBookQA | 0.850 |
| CommonsenseQA | 0.785 |
| PubMedQA | 0.898 |

`none` option: AUROC 0.945 when the correct option is removed, 0.925 for an unrelated state.

## Requirements

CUDA GPU with about 10 GB free for states up to 1k tokens (11 GB at 15k); latency on a Jetson AGX Orin 2.6 / 5.4 /
15.2 s at 0.8k / 3.7k / 15k tokens (eager, int8). Disk: 5.6 GB for the GGUF.""")


def variants(base_cfg, entries):
    keep = ("dtype", "kv_quant", "quantize_linear_state", "feature_layers")
    return {name: {**{k: base_cfg[k] for k in keep if k in base_cfg}, "weight_quant": wq, "source": src}
            for name, (wq, src) in entries.items()}


def main():
    ck2 = torch.load("runs/x2b/model.pt", map_location="cpu", weights_only=False)
    export("runs/x2b/model.pt", "checkpoints/opendecider-2b",
           variants(ck2["backbone_cfg"], {"int8": ("int8", QWEN2B), "w8": ("w8", QWEN2B), "nf4": ("nf4", QWEN2B),
                                          "awq": ("awq", AWQ2B)}),
           "int8", CARD_2B, info={"name": "opendecider-2b", "trained_from": "runs/x2b (step 1500)"})
    ck9 = torch.load("runs/x9b-gguf-stitch/model.pt", map_location="cpu", weights_only=False)
    export("runs/x9b-gguf-stitch/model.pt", "checkpoints/opendecider-9b-stitched",
           variants(ck9["backbone_cfg"], {"gguf-q4_0": ("int8", GGUF9B), "int8": ("int8", QWEN9B)}),
           "gguf-q4_0", CARD_9B, info={"name": "opendecider-9b-stitched", "stitched_from": "opendecider-2b",
                                       "stitch_holdout_cosine": ck9.get("extra", {}).get("stitch_holdout_cosine")})
    for d in ("checkpoints/opendecider-2b", "checkpoints/opendecider-9b-stitched"):
        files = sorted(os.listdir(d))
        print(d, [(f, round(os.path.getsize(os.path.join(d, f)) / 2 ** 20, 1)) for f in files])


if __name__ == "__main__":
    main()
