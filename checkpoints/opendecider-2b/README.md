---
license: apache-2.0
library_name: opendecider
tags: [decision-model, classification, calibration, qwen3.5]
---

# OpenDecider 2B

Calibrated typed decisions (choice, yes/no, ordinal score) from a frozen Qwen3.5-2B backbone. This checkpoint
contains the trained trunk, heads and adapters (25.4M parameters, fp16 safetensors). The backbone is downloaded from
its original repository at a pinned revision on first use.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-2b")            # or backbone="awq" | "nf4" | "w8"
print(dec.decide({"state": {"ticket": "I was charged twice and want my money back."},
                   "questions": {"route": {"type": "choice", "prompt": "Which team handles this?",
                                            "options": ["billing", "technical", "refund", "other"]}}}))
```

Or serve it: `python -m opendecider.serve --model checkpoints/opendecider-2b`.

## Backbone variants

All on a Jetson AGX Orin 64 GB; accuracy on 500 typed-decisions questions / Banking77 8-way / prompt-injection
detection (`docs/RESULTS.md` §12). The head was trained on int8 features: 4-bit backbones lose accuracy until the
head is retrained on them.

| Variant | Backbone source | Weights on GPU | Accuracy | Latency, 0.8k / 3.7k / 15k-token state |
|---|---|---|---|---|
| `int8` (default) | Qwen/Qwen3.5-2B | 1.88 GB | 0.434 / 0.825 / 0.776 | 0.80 / 1.47 / 4.59 s |
| `w8` (GemLite) | Qwen/Qwen3.5-2B | 1.88 GB | 0.430 / 0.812 / 0.767 | 0.66 / 1.80 / 6.35 s |
| `awq` (int4) | cyankiwi/Qwen3.5-2B-AWQ-4bit | 1.32 GB | 0.420 / 0.750 / 0.776 | not measured eagerly |
| `nf4` | Qwen/Qwen3.5-2B | 1.26 GB | 0.414 / 0.775 / 0.698 | not measured eagerly |

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
(slow). Disk: 4.3 GB for the backbone download.

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
| derek-thomas/ScienceQA (text-only items) | CC-BY-SA-4.0 |
| allenai/qasc, allenai/cosmos_qa, allenai/quartz | CC-BY-4.0 |
| ChilleD/StrategyQA | MIT |
| allenai/winogrande | Apache-2.0 |
| jackhhao/jailbreak-classification, neuralchemy/Prompt-injection-dataset | Apache-2.0 |
| reshabhs/SPML_Chatbot_Prompt_Injection, GuardrailsAI/detect-jailbreak, Lakera/gandalf_ignore_instructions | MIT |

Also used: in-context batches built from the clinc, massive, dbpedia and injection pools above, and programmatic
uncertainty items built from our synthetic data. Non-commercial datasets were excluded. The evaluation benchmarks were never used for training.

## License

The weights in this checkpoint are released under Apache-2.0, like the code. Five training sets are CC-BY-SA
(share-alike). Whether trained weights are "adapted material" under CC-BY-SA is legally unsettled; this head is a
classifier that does not generate text and cannot reproduce its training sentences. If that matters for your use,
consult your own counsel. The backbone keeps its own license (Apache-2.0 for every variant listed above).

These checkpoints will be retrained without the CC-BY-SA datasets and re-released with clean provenance.
