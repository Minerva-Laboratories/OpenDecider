# Data manifest

Every source used for training or evaluation must be listed here with its license and pinned
revision before use (docs/SPEC.md §5.1, §11). Anything non-redistributable is excluded from released artifacts.

| Source | Kind | License | Revision / generator seed | Used for | Redistributable |
|---|---|---|---|---|---|
| `data/synthetic/generate.py` | programmatic synthetic (labels computed by code, no LLM teacher) | this repo | seed recorded in `data/synthetic/stats.json` | train / val / calib / test_id / test_ood | yes |
| `opendecider.rl.GridEnv` | multi-step environment (outcome reward) | this repo | seed per run | RL stage | yes |

Public datasets (planned; not downloaded yet). Before adding one, fetch the license from the
dataset card at a pinned revision and record it here; do not rely on memory:
- yes/no QA (BoolQ-style), NLI, intent classification (77–150 classes), topic classification,
  multiple-choice QA.

## Public datasets: license check (2026-09-23; NOT downloaded yet; pin revision at download)
License as listed on the Hugging Face dataset card (or source repo where marked). Card sha at check time in brackets.

| Dataset | License | Role | Notes |
|---|---|---|---|
| PolyAI/banking77 [90d4e2ee55] | CC-BY-4.0 | benchmark (held out) | |
| qiaojin/PubMedQA [9001f2853f] | MIT | benchmark (held out) | |
| allenai/OpenBookQA | Apache-2.0 (GitHub repo) | benchmark (held out) | HF card says "unknown" |
| tau/commonsense_qa [94630fe30d] | MIT | benchmark (held out) | |
| deepset/prompt-injections [4f61ecb038] | Apache-2.0 | benchmark (held out) | |
| clinc/clinc_oos [155b9c7104] | CC-BY-3.0 | training candidate | |
| AmazonScience/massive | CC-BY-4.0 | training candidate | |
| google/boolq [35b264d036] | CC-BY-SA-3.0 | training candidate | share-alike |
| stanfordnlp/snli [cdb5c3d5ee] | CC-BY-SA-4.0 | training candidate | share-alike |
| allenai/ai2_arc [210d026faf] | CC-BY-SA-4.0 | training candidate | share-alike |
| cais/mmlu [c30699e835] | MIT | training candidate | |
| allenai/winogrande | Apache-2.0 (GitHub repo) | training candidate | |
| fancyzhx/dbpedia_14 [9abd46cf7f] | CC-BY-SA-3.0 | training candidate | share-alike |
| allenai/natural-instructions | Apache-2.0 (GitHub repo) | training candidate | |
| bigscience/P3 [db485208b9] | Apache-2.0 | training candidate | underlying datasets keep their own licenses |
| allenai/sciq, facebook/anli | CC-BY-NC | EXCLUDED | non-commercial |
| ag_news, piqa, glue, race, imdb, emotion, tweet_eval, multi_nli, fever, hellaswag | unknown / other / mixed / unverified | NOT USED until verified | |

### Downloaded 2026-09-23: held-out benchmark eval files (`data/builders/public_benchmarks.py` → `data/public/`, not redistributed)
Pinned revisions (also in `data/public/revisions.json`):
- banking77: GitHub PolyAI-LDN/task-specific-datasets@9d081458ff52e53cf7e848f414e6e9344e4e6696 (CC-BY-4.0; HF repo is a legacy loading script over this CSV). 77-way: 1000 test items; 8-way: 8 intents x 20.
- qiaojin/PubMedQA pqa_labeled@9001f2853fb87cab8d220904e0de81ac6973b318 (MIT): 890 yes/no items (maybe dropped).
- allenai/openbookqa main@388097ea7776314e93a529163e0fea805b8a6454 (Apache-2.0): 500 test.
- tau/commonsense_qa@94630fe30dad47192a8546eb75f094926d47e155 (MIT): 1000 of the validation split.
- deepset/prompt-injections@4f61ecb038e9c3fb77e21034b22511b523772cdd (Apache-2.0): 116 test.

### Built 2026-09-23: public TRAINING corpus (`data/builders/public_train.py` → `data/public_train/`, not redistributed)
8000 records per source (ARC smaller), split 90/5/5 train/val/calib by id hash: train 46163, val 2622, calib 2566.
Pinned revisions (also in `data/public_train/revisions.json`):
- clinc/clinc_oos plus@155b9c710419136e17307b80d0a13e68cd46b4ec (CC-BY-3.0)
- AmazonScience/massive@ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740 via refs/convert/parquet, en-US only (CC-BY-4.0)
- google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5 (CC-BY-SA-3.0)
- stanfordnlp/snli@cdb5c3d5eed6ead6e5a341c8e56e669bb666725b (CC-BY-SA-4.0)
- allenai/ai2_arc (Easy + Challenge)@210d026faf9955653af8916fad021475a3f00453 (CC-BY-SA-4.0)
- allenai/winogrande xl@01e74176c63542e6b0bcb004dcdea22d94fb67b5 (Apache-2.0, GitHub repo)
- fancyzhx/dbpedia_14@9abd46cf7fc8b4c64290f26993c540b92aa145ac (CC-BY-SA-3.0)
Share-alike sources: any redistributed derivative (e.g. model weights trained on them, depending on interpretation) must be
reviewed before release. MMLU excluded (its train split bundles RACE, license "other").

### Built 2026-09-23: injection / jailbreak TRAINING data (`data/builders/injection_train.py` → `data/public_train/injection_*.jsonl`)
train 12447 (6588 injections), val 695, calib 679. Normalised exact-match dedupe against ALL splits of the held-out
deepset/prompt-injections benchmark. Revisions in `data/public_train/injection_revisions.json`.
- jackhhao/jailbreak-classification@2f2ceeb396 (Apache-2.0), train split
- reshabhs/SPML_Chatbot_Prompt_Injection@02ce8084e9 (MIT); system prompt + user prompt
- neuralchemy/Prompt-injection-dataset@7d70432dfc (Apache-2.0), train split; aggregates hackaprompt / harmbench /
  wildguard samples: check those upstream licenses before any redistribution
- GuardrailsAI/detect-jailbreak@5f68a89245 (MIT); aggregates e.g. verazuo/jailbreak_llms
- Lakera/gandalf_ignore_instructions@04737b65e9 (MIT), train split, injections only
Skipped (no stated license or gated): xTRam1/safe-guard-prompt-injection, JasperLS/prompt-injections (mirror of the
benchmark), geekyrakshit/prompt-injection-dataset, rubend18/ChatGPT-Jailbreak-Prompts, hackaprompt (gated), wildjailbreak (gated).

### Built 2026-09-24: knowledge / commonsense MC TRAINING data (`data/builders/knowledge_train.py` → `data/public_train/knowledge_*.jsonl`)
train 22082, val 1117, calib 1201; dropped 2 questions overlapping the held-out OpenBookQA/CommonsenseQA (normalised exact match).
- allenai/qasc@a34ba204eb (CC-BY-4.0), 7998; supporting facts withheld
- allenai/cosmos_qa@28d9d5e2aa via refs/convert/parquet (CC-BY-4.0), 8000
- allenai/quartz@28c1dbb56c (CC-BY-4.0), 2696
- derek-thomas/ScienceQA@f18b0a7035 (CC-BY-SA-4.0; share-alike), 4103 text-only items
- ChilleD/StrategyQA@705562638f (MIT), 1603 yes/no
Excluded (license unknown/other/NC): social_i_qa, piqa, hellaswag, swag, art, riddle_sense, sciq, wiqa.

### 2026-09-25: context-learning and uncertainty data
- `data/builders/context_data.py train` → `data/public_train/context_train.jsonl` (21000): unlabelled comparative
  batches and labelled in-context examples built ONLY from our permissive training pools (clinc, massive, dbpedia,
  injection sets). Items drawn i.i.d. (natural base rates), random order, batch size 1 included.
- `context_data.py eval` → `data/public_context/` (eval only, not redistributed): Banking77 8-way and deepset
  prompt-injections test items in 4 variants (single / unlabelled batch of 8 / 4 / 8 labelled examples). Labelled
  examples come from each benchmark's TRAIN split (Banking77 CSV @9d081458; deepset train split). This is standard
  few-shot evaluation: those splits are NEVER used for training.
- `data/builders/uncertainty_data.py` → `data/synthetic/uncertainty_train.jsonl` (6000): unknowable items (unrelated
  state, uniform soft target) and buried states (2–5k chars of unrelated records around the real state). Derived from our
  own synthetic data only.

### 2026-09-27: typed-decisions benchmark (evaluation only)
- LocalLLaMA/typed-decisions@f7a2487edd7a (Apache-2.0): 400 test cases × 5 typed questions; gold = mean of 3 teacher
  samples. Downloaded to `data/typed_decisions/` (gitignored). **Eval only**: its train split is used solely to fit
  decision-profile weights (out-of-fold), never to train the model.

### 2026-10-01: clean-provenance TRAINING corpus (`data/builders/clean_train.py` → `data/clean_train/`, not redistributed)
For the retrained checkpoints: no share-alike, non-commercial, ND, copyleft, unknown-license or gated sources, and no
LLM-generated texts or labels. Licenses checked on each card at the pinned sha AND at the upstream source (stricter one
binding). Full candidate review (with ~50 rejected sets and reasons): kept with the builder's revisions.json.
Contamination: normalised exact match + any shared 13-word span against every held-out benchmark file and all splits
of deepset/prompt-injections.

Kept from the earlier corpus: clinc_oos, massive, winogrande, qasc, cosmos_qa, quartz, StrategyQA, Gandalf, and the
clinc/massive in-context items. Dropped: boolq, snli, ai2_arc, dbpedia_14, ScienceQA (CC-BY-SA), and after an upstream
audit of the injection sets:
- reshabhs/SPML_Chatbot_Prompt_Injection: system and user prompts written by GPT-4 (card, method section).
- neuralchemy/Prompt-injection-dataset: mixes undocumented "v1" data, gated hackaprompt rows, JailbreakBench
  judge-comparison prompts (partly written by an attacker LLM) and AdvBench goals (LLM-generated).
- jackhhao/jailbreak-classification: benign rows from GPTeacher ("100% GPT4 generated").
- GuardrailsAI/detect-jailbreak: relabels verazuo rows with an undocumented auto-tagger; its clean part (verazuo's MIT
  prompts) is taken directly via TrustAIRLab, labelled by source file.
- in-context items built from those pools.

New sources (pinned revisions in the builder; all verified 2026-10-01):
| Source | License | Use | Notes |
|---|---|---|---|
| jaredfern/codah@4b0e0e7f33 | ODC-BY | commonsense MC | |
| pkavumba/balanced-copa@813bd03cd6 | CC-BY-4.0 (COPA: BSD-2) | causal MC | |
| deepmind/aqua_rat@33301c6a05 (raw) | Apache-2.0 | algebra MC | rationale never shown |
| coastalcph/lex_glue@c23fdff1a6 case_hold, ledgar | CC-BY-4.0 (CaseHOLD Apache-2.0; LEDGAR MIT) | legal MC; 100-class topic | case text from the Caselaw Access Project, unrestricted since 2024-03 |
| gfissore/arxiv-abstracts-2021@e4c5fbd4de | CC0 (arXiv metadata terms) | ~150-class topic | title + abstract only |
| theatticusproject/cuad@a3c393f5d1 | CC-BY-4.0 (data, labels) | clause presence yes/no, absent-clause negatives | contract text: EDGAR filings, no license representation by the authors; never redistribute excerpts |
| allenai/csqa2 (GitHub @3f57a4a6a1) | CC-BY-4.0 | commonsense yes/no | ConceptNet prompt fields dropped (CC-BY-SA) |
| tasksource/ruletaker@a3e0880bae | Apache-2.0 | rule entailment yes/no | programmatic labels |
| tommccoy1/hans (GitHub @7299f6f657) | MIT | NLI yes/no | HF mirror says "unknown": not used |
| TrustAIRLab/in-the-wild-jailbreak-prompts@a10aab8eff | MIT | jailbreak vs regular prompts | posts scraped from public prompt communities |
| microsoft/llmail-inject-challenge@1063bdf01e | MIT | indirect injection in emails | raw human submissions only (never the LLM-judge labels); benign mailboxes are programmatic templates |
| paul-rottger/xstest (GitHub @d7bb5bd738) | CC-BY-4.0 | harmful-request yes/no | HF copy gated: not used |
| JailbreakBench/JBB-Behaviors@886acc352a | MIT | harmful-request yes/no | AdvBench rows dropped (LLM-generated) |
| OpenAssistant/oasst1@fdf72ae082 | Apache-2.0 | benign prompts; human quality/helpfulness ratings (ordinal) | synthetic rows and the detoxify column excluded |
| google/civil_comments@f2970eb3a5 | CC0 | toxicity (ordinal) | |
Excluded after review (selection): SciNLI (CC-BY-SA upstream), MultiNLI (SA/other), WANLI (GPT-3 text), ContractNLI
(NC-SA), SciTail (no license), ConditionalQA/DROP/Belebele (SA), CondaQA/ROPES/Qasper (upstream text rights), PIQA
(AFL-3.0), MedMCQA (question copyright unclear), HelpSteer2 (LLM-written responses). bigscience/P3 and
allenai/natural-instructions are struck from the candidate list above: they contain held-out benchmark tasks.
