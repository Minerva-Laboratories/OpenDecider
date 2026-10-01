---
license: apache-2.0
library_name: opendecider
tags: [decision-model, classification, calibration, qwen3.5]
---

# OpenDecider 9B (stitched)

The OpenDecider 2B head moved onto a frozen Qwen3.5-9B backbone with a closed-form ridge map ("stitch"), with
no gradient steps. This checkpoint contains the trunk, heads and the stitch map (57.3M parameters, fp16
safetensors). The default backbone is the Q4_0 GGUF from unsloth/Qwen3.5-9B-MTP-GGUF (5.6 GB), loaded directly by
OpenDecider; the stitch was fitted on those weights.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-9b-stitched")
```

## Backbone variants

| Variant | Backbone source | Status |
|---|---|---|
| `gguf-q4_0` (default) | unsloth/Qwen3.5-9B-MTP-GGUF (`Qwen3.5-9B-Q4_0.gguf`) + config/tokenizer from Qwen/Qwen3.5-9B | evaluated |
| `int8` | Qwen/Qwen3.5-9B (18 GB bf16 download, int8 at load) | not evaluated: the stitch was fitted on the GGUF weights |

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
15.2 s at 0.8k / 3.7k / 15k tokens (eager, int8). Disk: 5.6 GB for the GGUF.

OpenDecider is a public hypothesis test of a "System One" decision model in the style of TypeSafe's Jev. Results
are reported only as consistent or inconsistent with public observations; nothing here describes how Jev is built.
This checkpoint is a research release, not a product.

## Training data and licenses

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
