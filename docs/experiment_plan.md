# Experiment plan (agreed 2026-09-23)

Current best designs:

| Design | Description |
|---|---|
| V0 | The frozen backbone reads state→question→options in one causal row. A head reads the result. |
| V3 branched | The frozen backbone reads state→question→each option alone. A dense trunk follows: ScaleNorm pre-norm residual blocks, xATGLU, RoPE, no pooling until the per-option CLS. |

Details: `docs/v3_dense_design.md`.

## Step 1: transfer to held-out public benchmarks (no extra training)

- Checkpoints: `runs/all-v0/`, `runs/all-v0-1500/` (equal budget), `runs/all-v3branch/`. All are trained on
  synthetic data only.
- Benchmarks match those used by independent Jev tests, so both run on the same tasks: Banking77 (8-way and
  77-way), PubMedQA (yes/no), OpenBookQA, CommonsenseQA, deepset/prompt-injections.
- These datasets stay out of all training (held out by dataset, docs/SPEC.md §7.1).
- Metrics: accuracy, NLL, Brier, ECE (15 equal-mass bins), AUROC and selective accuracy, with bootstrap CIs.
- Wording: results are "consistent / inconsistent with public observations" of Jev. We make no claims about its
  internals.

## Step 2: caller-side decomposition (no text generation)

Motivation: an independent phishing test (BERI, jev-phishing-bench) took Jev from 62.6% with one question to 95.0%
with 5 narrow sub-questions. Weights fitted on 1,000 labeled examples combined the sub-questions. TypeSafe's
guidance says questions must be atomic.

1. Composite questions in `/v1/decide`. A question is declared as a combination of sub-questions (for example a
   fitted logistic). All sub-questions run in the same batch against the state, which is read once (O3 preserved).
2. Sequential chaining. A question's chosen option is substituted into a later question's text. Every step is a
   Choice, so there is no generation. Each hop costs one extra question pass, and the state stays cached.
   Arithmetic stays in code. This follows Jev's own guidance, and arithmetic is also among our weakest families.

Evaluation: synthetic families where the generator can emit the sub-questions (lookup chains, rule routing, set
membership, dates). Also replicate a phishing-style decomposition experiment.

## Step 3: learned in-model decomposition (would go beyond Jev)

- Sub-question supervision. The synthetic generator knows the hidden world. It emits intermediate sub-questions and
  labels (which record, which rule fired, which rows qualify) as auxiliary training targets.
- Latent sub-questions. A fixed set of learned question tokens runs as extra rows next to the real question. The
  trunk combines their answers (Q-Former-like automatic decomposition, no text generation).
- Multi-step decisions with outcome-only reward use the RL stage (§5.3, `src/opendecider/rl.py`).

## Step 4: mixed training corpus, then longer runs (DGX Spark)

The corpus mixes synthetic and permissive public data. Licenses were checked on 2026-09-23 against Hugging Face
dataset cards and source repos. Pin revisions at download time and record them in `data/MANIFEST.md`.

| Status | Datasets |
|---|---|
| Usable | banking77 CC-BY-4.0 · clinc_oos CC-BY-3.0 · AmazonScience/massive CC-BY-4.0 · PubMedQA MIT · boolq CC-BY-SA-3.0 · snli CC-BY-SA-4.0 · commonsense_qa MIT · OpenBookQA Apache-2.0 (repo) · ai2_arc CC-BY-SA-4.0 · mmlu MIT · winogrande Apache-2.0 (repo) · dbpedia_14 CC-BY-SA-3.0 · deepset/prompt-injections Apache-2.0 · allenai/natural-instructions Apache-2.0 · bigscience/P3 Apache-2.0 (templates; underlying datasets keep their licenses) |
| Avoid (non-commercial) | sciq CC-BY-NC-3.0, anli CC-BY-NC-4.0 |
| Verify first | ag_news, piqa, glue, race, imdb, emotion, tweet_eval (unknown/"other"); multi_nli (mixed, incl. "other"); fever (CC-BY-SA + GPL); hellaswag (no license on the card, repo blocked; often cited as MIT, not verified) |

- Share-alike datasets are fine for training. Any redistributed derivative carries the SA terms (note this in
  MANIFEST).
- Step 1 benchmarks are never used for training.

## Open items

- Equal-budget comparison: V0 @1500 steps vs V3 branched @1500, on ID and OOD tests with CIs. In progress.
- Probe battery (permutation, dummy, duplicate, label length, repeat calls, latency) on V0 and V3 branched.
- V3 trunk optimization issue. Cross-attention failed to learn matching on real isolated features, even where a
  linear readout could. A toy task learns fine. Investigate before using cross-attention elsewhere.
- M4 latency work: CUDA graphs and torch.compile (see `docs/latency_notes.md`).

## Results: Step 1 (synthetic-only transfer) and Step 4a (mixed corpus), 2026-09-23

Public benchmarks, held out. Accuracy with 95% bootstrap CIs is in the eval logs `runs/public-*/eval_*.json`.

| Benchmark | chance | V0 synth-only | V0 mixed (3000 st) | V3 branched mixed (2000 st) | Jev (independent reports) |
|---|---|---|---|---|---|
| Banking77 77-way | 0.013 | 0.130 | 0.380 | **0.471** | 0.79 (TDS), 0.788 (AY) |
| Banking77 8-way | 0.125 | 0.600 | 0.769 | **0.788** (ECE 0.086) | 0.838 (AY) |
| PubMedQA yes/no | ~0.5 | 0.642 | 0.645 | **0.718** (ECE 0.034) | 69.0 Jevals decision score (different metric) |
| OpenBookQA | 0.25 | 0.312 | **0.422** | 0.390 | 0.942 (scienthoon) |
| CommonsenseQA | 0.20 | 0.410 | **0.474** | 0.457 | 0.881 (scienthoon) |
| prompt-injections | ~0.5 | 0.483 | 0.483 | 0.491 | 0.870 (AY) |

Reading:
- Data coverage drives recognition tasks (intents about x3, science MC +11 points).
- Branched V3 is best on recognition and calibration.
- Knowledge-heavy MC stays far below Jev. The likely cause is the frozen 0.8B backbone.
- There is no injection-related training data, so that task stays at chance.
- Jev numbers come from different systems and conditions. Comparisons are "consistent / inconsistent with public
  observations" only.

Proposed next steps:
1. Permissive injection and jailbreak training data (not the benchmark).
2. Qwen3.5-2B backbone.
3. Decomposition experiment.
4. Probe battery on mix-v0 vs mix-v3branch.

## Improvement candidates (literature check 2026-09-23)

1. Rejected on 2026-09-23: soft-label distillation from a larger open teacher. No teacher models are used. The goal
   is calibration and classification learned from labels. The rejected proposal: a Qwen3.5-9B teacher scores each
   option (teacher calibrated first, no label smoothing), with a KL + CE + Brier loss. Evidence cited:
   arXiv:2412.09807 (LLM→DeBERTa MC, MMLU 28.9→39.3), 2005.10419, 2508.20224, 1906.02629. It targeted the
   knowledge-heavy MC gap (the frozen 0.8B caps recall).
2. Shared-S0 branched rows (SpecLA-style) plus shared-prefix attention (Hydragen 2402.05099). This removes per-row
   cache copies.
3. Score questions: a cumulative-link / CORN ordinal head (1901.07884, 2111.08851) plus a per-question-type
   temperature (or Dirichlet calibration, 1910.12656). CORAL/CORN report MAE, not ECE, so measure calibration
   ourselves.
4. VeRA (arXiv:2310.11454; h = W0x + Λ_b B Λ_d A x, shared frozen random A,B, b init 0, d init 0.1) or LoRA-FA on
   the row pass only, last ~4–6 layers (top-layer tuning evidence: 1911.03090, 2411.15558, 2206.06522). This keeps
   state caches valid and backprop shallow. Re-check calibration: upper layers do "confidence correction"
   (2511.00280).
5. Multi-layer features: a learned mix that includes mid-depth layers (2502.02013, 2601.13288).
6. Calibration under shift: a question-conditioned temperature (Thermometer 2403.08819) plus sampler ensembles
   (1906.02530).
7. Order-bias probe and fix for slot variants: PriDe (2309.03882), Batch Calibration (2309.17249).
8. Focal loss with automatic γ (2002.09437) instead of CE (keep Brier).
