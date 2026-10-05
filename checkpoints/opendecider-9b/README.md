---
license: apache-2.0
library_name: opendecider
base_model: Qwen/Qwen3.5-9B
pipeline_tag: text-classification
tags: [decision-model, classification, calibration, qwen3.5]
---

# OpenDecider 9B

Calibrated typed decisions (choice, yes/no, ordinal score) with an explicit "none of the above" option, on a
frozen Qwen3.5-9B, Q4_0 GGUF from unsloth/Qwen3.5-9B-MTP-GGUF (Apache-2.0). The 2B checkpoint's trunk and heads moved to the 9B with a closed-form ridge map between the two backbones' features, then trained for 1,500 steps on the same corpus with VeRA adapters on the top 16 layers (58.1M trainable parameters). This checkpoint contains the trained trunk, heads and adapters (fp16
safetensors). The backbone is downloaded from its original repository at a pinned revision on first use.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-9b")
print(dec.decide({"state": {"ticket": "I was charged twice and want my money back."},
                   "questions": {"route": {"type": "choice", "prompt": "Which team handles this?",
                                            "options": ["billing", "technical", "refund", "other"]}}}))
```

Or serve it: `python -m opendecider.serve --model checkpoints/opendecider-9b`.

## Results (zero-shot, default variant)

Accuracy with 95% bootstrap confidence intervals (1,000 resamples).

| Benchmark | Accuracy | NLL |
|---|---|---|
| Banking77 8-way | 0.875 [0.825, 0.925] | 0.773 |
| Prompt-injection detection | 0.810 [0.741, 0.871] | 0.537 |
| OpenBookQA | 0.884 [0.856, 0.910] | 0.360 |
| CommonsenseQA | 0.800 [0.775, 0.826] | 0.666 |
| PubMedQA | 0.891 [0.871, 0.911] | 0.286 |

Secondary, not a correctness benchmark: on typed-decisions (2,000 decisions whose gold labels are the answers of a ~4B
teacher model, so the score measures agreement with that teacher; the dataset card lists the teacher's self-agreement at 0.735)
this checkpoint scores 0.614 [0.592, 0.638] (KL to the teacher
distribution 0.407). We do not train or tune on it.

`none` option ("none of the above", on by default), AUROC of P(none) against answerable questions on the five public benchmarks: correct option removed 0.893, only plausible wrong options left 0.880, state unrelated to the question 0.808.

Conformal prediction sets (`"conformal": {"alpha": 0.1}` in a request) use calibration scores stored in this
checkpoint (choice 406, noul 393, score 84 questions from the training families' calibration split). Coverage holds for requests like that
split; for other distributions, refit with a few dozen labels (`POST /v1/calibrate`).

## Backbone variants

Jetson AGX Orin 64 GB, int8 KV cache, eager kernels, one fresh process per variant for memory and latency.
Accuracy: 500 typed-decisions questions (secondary) / Banking77 8-way / prompt-injection detection. The head was
trained on the default variant's features.

| Variant | Backbone source | Weights on GPU | Peak GPU memory, 0.8k / 15k-token state | Latency, same states | Accuracy |
|---|---|---|---|---|---|
| `gguf-q4_0` (default) | unsloth/Qwen3.5-9B-MTP-GGUF | 8.62 GB | 9.9 / 11.1 GB | 3.68 / 21.35 s | 0.560 / 0.875 / 0.810 |

## Requirements

CUDA GPU with about 10 GB free for states up to 1k tokens (11 GB at 15k). Disk: 5.6 GB for the GGUF backbone.

OpenDecider is a public hypothesis test of a "System One" decision model in the style of TypeSafe's Jev. Results
are reported only as consistent or inconsistent with public observations; nothing here describes how Jev is built.
This checkpoint is a research release, not a product.

## Training data and licenses

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
sources are credited in the table above.
