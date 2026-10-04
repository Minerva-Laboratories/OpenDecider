---
license: apache-2.0
library_name: opendecider
tags: [decision-model, classification, calibration, qwen3.5]
---

# OpenDecider 2B

Calibrated typed decisions (choice, yes/no, ordinal score) with an explicit "none of the above" option, on a
frozen Qwen3.5-2B (Qwen/Qwen3.5-2B, Apache-2.0). Trained from scratch for 1,250 steps on the clean corpus, on the deployed int8 backbone features, with VeRA adapters on every backbone layer (25.4M trainable parameters). This checkpoint contains the trained trunk, heads and adapters (fp16
safetensors). The backbone is downloaded from its original repository at a pinned revision on first use.

## Quickstart

```python
from opendecider.hub import load_decider
dec = load_decider("checkpoints/opendecider-2b")
print(dec.decide({"state": {"ticket": "I was charged twice and want my money back."},
                   "questions": {"route": {"type": "choice", "prompt": "Which team handles this?",
                                            "options": ["billing", "technical", "refund", "other"]}}}))
```

Or serve it: `python -m opendecider.serve --model checkpoints/opendecider-2b`.

## Results (zero-shot, default variant)

Accuracy with 95% bootstrap confidence intervals (1,000 resamples).

| Benchmark | Accuracy | NLL |
|---|---|---|
| Banking77 8-way | 0.831 [0.775, 0.887] | 0.633 |
| Prompt-injection detection | 0.681 [0.595, 0.767] | 0.817 |
| OpenBookQA | 0.670 [0.628, 0.710] | 0.868 |
| CommonsenseQA | 0.651 [0.621, 0.679] | 0.896 |
| PubMedQA | 0.845 [0.821, 0.867] | 0.368 |

Secondary, not a correctness benchmark: on typed-decisions (2,000 decisions whose gold labels are the answers of a ~4B
teacher model, so the score measures agreement with that teacher; the dataset card lists the teacher's self-agreement at 0.735)
this checkpoint scores 0.485 [0.460, 0.510] (KL to the teacher
distribution 0.342). We do not train or tune on it.

`none` option ("none of the above", on by default), AUROC of P(none) against answerable questions on the five public
benchmarks: correct option removed 0.817, only plausible wrong options left 0.803, state
unrelated to the question 0.816.

Conformal prediction sets (`"conformal": {"alpha": 0.1}` in a request) use calibration scores stored in this
checkpoint (choice 406, noul 393, score 84 questions from the training families' calibration split). Coverage holds for requests like that
split; for other distributions, refit with a few dozen labels (`POST /v1/calibrate`).



## Requirements

CUDA GPU with about 2.5 GB free for states up to 1k tokens (3.1 GB at 15k). The `int8` variant also runs on CPU (slow). Disk: 4.3 GB for the backbone download.

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
