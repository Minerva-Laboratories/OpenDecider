# Open decision-model landscape and direction for OpenDecider

This is a snapshot dated 2026-09-29. TypeSafe launched Jev on 2026-09-15, two weeks earlier.

Sources searched: GitHub, Hugging Face, arXiv, Hacker News, blogs and aggregators (awesome-system-one,
awesome-jev-robustness, systemonemodels.org). Reddit could not be fetched directly.

Verification tags:

| Tag | Meaning |
|---|---|
| [R] | We read the project's own README, page or paper. |
| [A] | The claim comes only from an aggregator. |
| [S] | The claim comes only from search snippets. |

Almost every number here is self-reported by its authors. Statements about Jev are third-party observations.
Following `docs/SPEC.md`, we describe our results only as consistent or inconsistent with those observations. We make
no claims about Jev's architecture.

---

## Overview

The field split into three camps within two weeks:

| Camp | Method | Examples |
|---|---|---|
| Logit readers | Prompt an LLM and read option-letter probabilities. | open-alternative-jev, Featherless, Decider, SemIf and about 20 more |
| LoRA and head fine-tunes on Qwen | Train adapters or a head on the backbone. | Open-Jev, JevK5, Kev, Nimble, Tev1, AutoJev |
| Small encoder specialists | Classifier-style encoders. | Laya, Von, Verdict, GLiNER2.5-Decide, Julia 1 |

Frontier labs are entering. Upstage launched Solar Decide on 2026-09-28. OpenAI launched a Decisions API on
2026-09-29.

Our standing:
- Zero-shot on typed-decisions, our 2B→9B stitched model (0.616, KL 0.28) is in the open pack. It is better
  calibrated than most, but not the most accurate.
- Specialists fine-tuned on the benchmark's train split beat our fitted tree profile (0.675): Verdict (0.771,
  Brier 0.064) and Laya (0.766). We fitted the tree profile out-of-fold on test halves. That is a different
  protocol, but the gap is real.

Features we found in no other public project:
1. A closed-form stitch moves a trained head across model sizes with no gradient steps.
2. Exact option-order invariance in one pass. OpenJev pays K isolated passes for it, and AnyJev pays K rotations.
   Kev and Jev are both order-sensitive.
3. A Jetson-first local stack on the hybrid Qwen3.5 backbone.

Open gaps that nobody handles well. Each fits our architecture:
- abstention without a "none" option
- coherent probabilities across questions
- per-tenant recalibration with few labels
- incremental or reused state across requests
- evidence attribution
- benchmarks of the "one state, many questions" workload

§5 turns these gaps into a ranked roadmap.

---

## 1. Map of solutions

### 1.1 Closest designs to OpenDecider

| Project | What it is | Reported (self) | Relation to us |
|---|---|---|---|
| Kev (jaredpalmer/kev) [R], Apache-2.0 | Qwen3.5 0.8B–27B with rank-16 LoRA. A pointer head scores each option's `</opt>` state against the question's `<decide>` state. Serves `/v1/systemone`. | 27B: 0.848 vs Jev 0.857 on new sources. 4B/9B are about 4 points behind. MMLU 9B 0.74 vs Jev 0.90. | Same head idea as ours. Its README says option order can change the answer. Ours is invariant by construction. Kev fine-tunes the backbone; we keep it frozen. |
| AnyJev (nokia-applied-research) [R], Apache-2.0 | L0 averages over option rotations and removes a label-free prior. L1 fits a temperature. L2 fits a closed-form head on intermediate hidden states from 100–300 labels. | Qwen3-8B on Banking77: order flips 0.230→0.073, ECE 0.240→0.095. Traffic decided automatically at ≤5% risk: 7.7%→52%. | This is our decision profile plus hidden-state probes, packaged as levels. Its "auto-decidable at risk" metric is the right product metric. We should adopt it. |
| CLM (Contrastive-LM) [R], Apache-2.0 | Frozen Qwen3-8B with 20M projection heads. State and action embeddings are cached separately (a dual encoder trained with InfoNCE). | Agent benchmarks, up to 9× faster. | Similar to our V1 late fusion. Our V2/V3 deep fusion is the untested alternative in that design space. |
| qwen-rlcd [R], no license | "Prefix-fork" on hybrid Qwen3.5-0.8B: prefill the state once, then copy the GDN and attention cache per question row. | Matches sequential results to 4.8e-5, 8–19× faster. | Same trick as our branched shared prefix (`docs/branched_shared_prefix_math.md`). Not unique to us. |
| Decider (Mapika) [R], Apache-2.0 | Letter logits with per-type temperatures. Has state-first and schema-first (cached prefix) layouts. SFT on about 95 datasets plus 27B-teacher labels, then RL on calibrated outcomes (MiniWoB++). | #2 on Decision Index v0.2.1 (57.33 vs Jev 57.91). | The only public RL-for-calibration stage. Relevant to our §5.3 plan, but it depends on a teacher, which we avoid. |

### 1.2 Other projects, by camp

Logit readers (inference only; minimal calibration evidence unless noted):
- open-alternative-jev [R]: Qwen3.6-27B, 73.7% on typed-decisions, ECE 0.020, 582 ms per case. Reversing option
  order swings 4B yes/no answers by 13.5 points.
- Featherless simple-jev [R]
- SemIf [A]: WebGPU/MLX
- openjev-sglang, litjev, mini-jev, jevmlx, jevfire (vLLM) [A]
- open-spark-jev [A]: DGX Spark
- PocketJev (iPhone) and systemANE [A]
- Diffusion-LM variants [A]: djev, razorback16/openjev

LoRA and head fine-tunes:
- Open-Jev [R]: JevBench 2B 64.9 / 9B 77.5 / 27B 85.3. 2B p50 85 ms on an H100.
- JevK5 [R]: JevBench 62.04. Teacher-distilled.
- Nimble [R]: Qwen3.5-9B with LoRA on 2,676 contrastive pairs, where one changed fact flips the label. 90.1% vs
  Jev 93.2% agreement on 324 items. Its labels are model-checked.
- Tev1 (Together) [R]: its docs say "logprobs are preferences, not calibrated confidence".
- AutoJev/JEV-27B [R]: 24-slot fp32 head. 84.07% vs Jev 83.85% on 6 benchmarks.
- Imajev-4B [S]: photo input. JevBench #1 at 67.37. Has an explicit "can't tell" answer.
- Eikos [S]: finance.
- Shisa DE-1 [S].

Encoder specialists:
- Laya [R]: ModernBERT 421M, 33 ms on a T4. Its typed-decisions score of 0.766 comes from a checkpoint fine-tuned
  on the benchmark's train split. The base checkpoints score 0.36. Banking77 0.425.
- Von [R]: 395M, about 23 ms on an A10G.
- Verdict 2.0 [R]: 151M, GLiClass-style. typed-decisions 77.1% / Brier 0.064 as a specialist. Has an explicit
  `__insufficient_evidence__` option, whose recall falls to 18% on hard negatives. External TypeSafe benchmark:
  48% vs Jev 90.8%.
- GLiNER2.5-Decide [S]: 340M. Decodes typed questions jointly.
- Julia 1, jeff, jevlike [A].

Game and control:
- NanoJev [R]: Qwen3-0.6B with set-attention heads, trained on 18.7k decisions from game episodes. ViZDoom
  128/128 vs Jev 56/128.

Multimodal: Imajev, Jev-Omni, jev-visual, PlayJev [A].

Aggregators and servers:
- Ollaya [S]: "Ollama for decision models", TypeSafe-compatible API.
- OpenRouter Decisions [A].
- OpenJev/OpenJevPro [R]: PolyForm non-commercial. Isolated per-option passes give exact order invariance. A
  "System 2→1 flywheel" escalates borderline cases to a reasoning model. The fast path takes under 35 ms; an
  escalated case takes about 1.1 s.

Commercial:
- OpenAI Decisions API [S]: announced at DevDay 2026-09-29. About 150 ms. Limited preview.
- Upstage Solar Decide [S]: same schema, 512K context.
- Hanzo Kai, Respan Span-01 [A].
- meraGPT Decider 1: typed-decisions leader.

### 1.3 Benchmarks and harnesses

| Benchmark | Notes |
|---|---|
| typed-decisions [R] | Apache-2.0. The one we use. The leaderboard mixes generalist and specialist rows, so check the mode. |
| JevBench v1.4.2.2 [R] | MIT. Harmonic mean of intelligence, calibration, speed and cost. Issue #47: its speed axis never tests the "one state, many questions" workload. |
| Jev Decision Index v0.2 [R] | 49 entrants, 40 benchmarks, chance-corrected. MIT code; some upstream datasets are restricted. |
| jabr/classifier-benchmark [R] | CC0. 49 tasks. Jev 0.966, Von 0.720, Laya 0.583. |
| LangWatch OOD comparison [R] | 11 tasks. Jev 81.9% vs Eikos-27B 80.0%. |
| NavyaAI [R] | Jev's raw ECE is 0.080, against 0.053 for an untuned self-hosted model. End-to-end speedup is only 1.7–1.9×. |
| Axiom and adversarial papers | 2609.33209 [R] (P(X)+P(¬X) is off by 0.064), JevAdvBench 2609.31142 [A], "Type-Safe Is Not Error-Free" 2609.26758 [A], jev-verify [A]. |

### 1.4 Integrations

TypeSafe ships official Python and JS SDKs, a Vercel AI SDK provider and a LangChain provider. Community work:
- MCP servers: jev-mcp [R] (MIT, 12 tools, several backends) and typesafe-mcp (Go)
- llm-typesafe, pg-jev, jevframe
- jev-tree: client-side recursion for more than 255 options
- DSPy and LlamaIndex adapters [S]

Almost every open server that wants drop-in adoption now speaks the `/v1/systemone` wire format.

---

## 2. Third-party measurements of Jev, next to our observations

Rows map to O1–O8 in `docs/SPEC.md` §1. They are observations only.

| Observation | Third-party evidence | Our status |
|---|---|---|
| O3/O5: shared state, flat in the number of questions | Hume [R]: 57 ms at 360 tokens → 218 ms at 29.8k tokens. 87 ms for 1 question → 610 ms for 1,500. Caps of about 32k tokens per branch and 65k per request. | Consistent with our design: the state is encoded once, and questions run as branches. |
| Options interact before the readout | Hume [R]: adding an irrelevant fifth option moved the log-odds between two other options by −0.28 (95% CI −0.36 to −0.19). | Consistent with our option-interaction block. Inconsistent with independent per-option scoring such as OpenJev's. |
| O6: order sensitivity | Hume [R]: a reference option placed last scored 16/16, first or middle 11–12/16. Other reports [A]: +0.37 for the first option on value-laden questions, 88% vs 57% on arithmetic by position, and the first option wins 2,000 of 2,000 times when no option is correct. | `slot_emb=learned` reproduces order sensitivity; `none` removes it. Hume's pattern (last position favored) is consistent with causal processing over the options. |
| O7: run-to-run variance | [A] SD 0.001–0.015. 15 distinct answer sets in 50 identical calls. Batching does not interfere: JS divergence 0.0058 vs 0.0035 [R]. | Consistent with a small stochastic sampler rather than float jitter. This matches our `sampler.k` hypothesis (b). |
| Confidence field | [R/A] Exactly (K·p_max − 1)/(K − 1). Outputs are quantized to 0.01. | A formula, not a separate uncertainty head. We can offer real uncertainty (§5). |
| Choice overconfident, Noul underconfident | [A] Hidden fair die: Choice puts 0.83 on its pick and is right 19% of the time. Noul stays near 1/6. The gap between Noul and Choice averages 0.125. | Not probed yet. Add it to the probe battery. |
| Incoherent probabilities across questions | 2609.33209 [R]: P(X)+P(¬X) is off by 0.064. Three mutually exclusive Nouls sum to 1.14. [A] reports ranges of 0.71–1.42. | Not probed yet. Also an opportunity (§5.3). |
| Weak abstention | [A] Removing the "unknown" option on KoBBQ took accuracy 0.95→0.00. 0 of 30 out-of-scope inputs were flagged. | We have uncertainty training data (`uncertainty_train.jsonl`). The sink option (`none`) now has code (§5.2). |
| Injection dressed as evidence | [A] JevOut: 61.4% of decisions flip. One appended opinion flips 12.1% (JevAdvBench). | Not measured. |
| Tokenizer | Hume [R]: closest to Qwen (348/415 probes agree), but no exact match among 192 public tokenizers. | We make no claim either way. |

---

## 3. Gaps nobody handles well

1. Abstention without an explicit "none" option. Only Verdict and Imajev add one, and Verdict's collapses on hard
   negatives.
2. Coherent probabilities across questions, such as complements and mutually exclusive Nouls.
3. Option-set effects. Irrelevant options shift answers, and large option lists collapse toward zero. Invariance
   currently costs K× compute (OpenJev, AnyJev).
4. Per-tenant recalibration with few labels. Everyone fits one global temperature. Thresholds tuned
   in-distribution fail out of distribution.
5. Real uncertainty. Confidence is a function of p_max. Conformal sets and abstention exist only as external
   add-ons.
6. Sequential and multi-step decisions. Jev is at 13% on sequential state. Only NanoJev and Decider touch this.
7. Incremental or reused state across requests. Nobody offers it: caching is per request or per schema only.
8. More than 255 options. Only client-side recursion handles this (jev-tree).
9. Evidence attribution: which span of the state drove the decision.
10. Benchmarks of the real workload ("one state, many questions"), with ECE reported against its sampling noise
    floor.

---

## 4. Differentiators to claim now

| Claim | Status | Evidence |
|---|---|---|
| Train small, deploy big: a closed-form stitch moves a trained head to a larger backbone without training. | Unique among public projects we found. | 0.577→0.616 on typed-decisions zero-shot. Stitching onto the same backbone reproduces the head's logits within 1e-3 (`tests/test_stitch.py`). |
| Order-invariant in one pass, with options still interacting (Hume found Jev's options interact). | Unique combination. OpenJev needs K passes and has no interaction. AnyJev needs K rotations. | Permutation TV 0.000. The dummy-option probe shows interaction. |
| Calibrated zero-shot. | Strong but not unique. open-alternative-jev reports ECE 0.020 at 27B; Verdict reports Brier 0.064 as a specialist. | KL 0.28 vs Jev 1.44 on typed-decisions. |
| Local and edge-first on the hybrid Qwen3.5 backbone. | Shared with SemIf, PocketJev and open-spark-jev. | Jetson Orin stack. |
| Behavioral probe battery compared against Jev observations. | Rare. Most projects report accuracy only. | `eval/probes.py` |

Do not claim:
- "best open accuracy". Specialists beat us, and Kev and Open-Jev at 27B lead the generalists.
- "only calibrated model".
- "fastest". Encoders run at 8–30 ms.

---

## 5. Roadmap, ranked by gain × ease

Each item lists the source ideas, how it plugs into OpenDecider, and the experiment that decides it. No item uses
an LLM teacher.

Detailed designs for four items are in [`roadmap_designs.md`](roadmap_designs.md): the sink option (`none`),
few-label recalibration, state reuse (radix prefix cache plus hybrid BM25 and dense retrieval) and backbone
explanations. Code status as of 2026-09-30 (code and unit tests only; no results are claimed here):

| Item | Code |
|---|---|
| 5.2 abstention: sink option (`none`) | `v3_none` in `src/opendecider/v3.py`; `tests/test_none_sink.py` |
| 5.4 few-label recalibration | `src/opendecider/calibrators.py` (used by `/v1/calibrate`); measured in `eval/calib_curve.py` |
| 5.8 state reuse, Tier 1 (exact prefix cache) | `src/opendecider/state_cache.py`, `src/opendecider/chunking.py`; `tests/test_state_cache.py` |
| 5.8 state reuse, Tier 2 (retrieval-built states) | `src/opendecider/store.py`, `src/opendecider/embed.py`; `tests/test_store.py`, `eval/store_recall.py` |
| 5.10 evidence and explanations | `src/opendecider/explain.py`, behind the opt-in `/v1/explain` endpoint (separate from `/v1/decide`); `tests/test_explain.py` |

Conformal sets (5.2), coherence constraints (5.3) and the edge cascade (5.7) have no code yet.

### 5.1 Fix the 9B zero-shot source: rotations, prior removal, full-label scoring (very easy, inference only)
Sources:
- Permutation Self-Consistency (Tang et al., NAACL 2024, arXiv:2310.07712)
- PriDe (Zheng et al., ICLR 2024, arXiv:2309.03882)
- AnyJev L0 (order flips 0.23→0.073)
- Normalized Contextual Calibration (NCC) for label length (Sanz-Guerrero & von der Wense, 2025, arXiv:2511.14385)
- DC-PMI (Holtzman et al., EMNLP 2021, arXiv:2104.08315)

Why: our head is invariant, but the 9B letter source is not. Its position bias leaks into the pool. NCC also
targets our known label-length weakness.

Experiment: compare rotation-averaged, PriDe and NCC 9B sources. Re-run `ensemble_cv`, `tree_head` and the
label-length probe. Cost: K cyclic shifts batched in one server call.

### 5.2 Native abstention and conformal outputs in the API (easy; strong differentiator; gaps 1 and 5)
Sources:
- RAPS (Angelopoulos et al., ICLR 2021, arXiv:2009.14193)
- clustered conformal (Ding et al., NeurIPS 2023, arXiv:2306.09335)
- conformal abstention (Abbasi-Yadkori et al., arXiv:2405.01563)
- AnyJev's "auto-decidable at risk" metric

Plug-in: an optional `risk` per question. The response adds `set` (an option set with guaranteed coverage) and
`abstain`. Pair this with our uncertainty training data (unknowable and buried states) so the model can abstain
without a caller-supplied "none" option.

Experiment: the fraction of traffic decided automatically at ≤5% risk, against the raw 9B and AnyJev's numbers.
Also run the KoBBQ-style "no correct option" probe.

### 5.3 Coherence across questions (medium; gap 2)
Sources: arXiv:2609.33209 (probability-axiom violations). The mechanism is ours.

Plug-in: the schema already has `CompositeQ` and `CombineSpec` for dependencies. Add declared `constraints`
(complement, mutually exclusive, implies). Project the per-question distributions onto the consistent set. One
option is a KL projection solved with a few vectorized iterations of iterative proportional fitting.

Experiment: run the complement and mutually-exclusive probe from 2609.33209 on OpenDecider and compare with
published Jev numbers. Measure Brier before and after projection.

### 5.4 Few-label calibration per tenant (easy; gap 4; builds on our §6 learning curve)
Sources:
- ensemble temperature scaling (Mix-n-Match, Zhang et al., ICML 2020, arXiv:2003.07329)
- Thermometer, zero-label temperature prediction (Shen et al., ICML 2024, arXiv:2403.08819)
- CORDIAL for ordinal Score questions (Wang et al., arXiv:2609.29807, 2026-09-24; not yet peer-reviewed): lowest
  log loss in 76 of 80 settings with 5–100 labels
- Batch Calibration, label-free prior (Zhou et al., ICLR 2024, arXiv:2309.17249)
- hierarchical shrinkage of per-tenant temperatures (standard empirical Bayes; our design)

Plug-in: `/v1/calibrate` chooses automatically:

| Labels | Calibrator |
|---|---|
| under ~50 | shrunk temperature, with the Thermometer prediction as prior |
| 50–250 | temperature plus bias |
| over 250 | residual trees |
| Score questions, any size | CORDIAL |

Experiment: extend `eval/calib_curve.py` with ensemble temperature scaling, the shrunk temperature, Thermometer and
CORDIAL.

### 5.5 Few-label hidden-state heads as an extra pooled source (easy)
Sources:
- Hidden Calibration, nearest centroid on hidden states (Cho et al., NAACL 2025, arXiv:2406.16535): 20–50% over
  token-based methods
- AnyJev L2: a closed-form head at about 64% depth
- Layer by Layer (Skean et al., ICML 2025, arXiv:2502.02013): middle layers beat the last layer by up to 16%

Plug-in: for tenants with a fixed option set, keep per-option centroids or a ridge head on our cached trunk option
features. Add it as one more source for the linear or tree profile.

Experiment: typed-decisions train split → test (specialist mode), so the comparison with Laya and Verdict is fair.

### 5.6 Better stitching (easy to medium; strengthens our unique claim)
Sources:
- Chen et al., *Transferring Linear Features Across LMs With Model Stitching* (arXiv:2506.06609). Per-layer affine
  maps. Transferred features make large-model training about 50% cheaper.
- Linear Representation Transferability (arXiv:2506.00653)
- Procrustes-style latent translation (Maiorca et al., NeurIPS 2023, arXiv:2311.00664)
- relative representations (Moschella et al., ICLR 2023, arXiv:2209.15430)

Plug-in: fit per-layer affine or orthogonal maps on 5–10k generic anchors, not only task states. Then use the
stitch as the initialization for a short warm-start fine-tune on the other machine (already planned).

Experiment: measure whether it recovers prompt injections (0.595, vs 0.767 for the 2B). Count the steps the warm
start saves compared with training from scratch.

### 5.7 Edge-first cascade (medium; large latency gain)
Sources:
- FrugalGPT (arXiv:2305.05176)
- cascade deferral rules (Gupta et al., ICLR 2024, arXiv:2404.10136)
- Gatekeeper (arXiv:2502.19335)
- learning to defer (Mozannar & Sontag, ICML 2020, arXiv:2006.01862)
- OpenJevPro's "System 2→1" escalation

Plug-in: the 2B head on the Jetson answers first. The request escalates to the 9B profile only when the 2B head is
uncertain (a conformal threshold on its calibrated confidence).

Experiment: the latency vs accuracy curve on the Orin, and the fraction deferred at iso-accuracy with the 9B
profile.

### 5.8 Incremental and reusable state (medium; gap 7; native to the architecture)
Why us: on the hybrid Qwen3.5 backbone, the Gated DeltaNet layers carry a fixed-size recurrent state. Appending
events to a state costs only the new tokens there. The attention layers still need their KV cache, which we
already store in int8.

Plug-in: `/v1/decide` returns a `state_id`, and `POST /v1/state/{id}/append` accepts streaming agent traces and
logs. The cache has a TTL and a memory cap. No other project offers this. The Jev API has no prefix caching across
requests (NavyaAI).

Experiment: latency for "append 200 tokens to a 20k-token state" vs re-encoding it. Results must match a full
re-encode within 1e-3 (a cache-equivalence test, as `tests/test_cache.py` already does).

### 5.9 Programmatic contrastive pairs for training (medium)
Source: Nimble. 2,676 minimal-edit pairs took a 9B from 66% to 90% agreement with Jev.

Our variant: our synthetic generators already build states from structured records. We can flip one fact and
recompute the label by program, with no model checking. Target the pairs at:
- evidence-framed injection (fake approvals, editor notes)
- irrelevant options
- near-miss distractors

Experiment: OOD accuracy, the injection flip rate (JevAdvBench style) and the dummy-option probe.

### 5.10 Evidence attribution without generation (medium; gap 9)
Plug-in: the trunk already cross-attends from the question and options into the state. Return the top-k state spans
by attention rollout or gradient×input as `evidence: [{start, end, weight}]`. These are character offsets, not
generated text, so the "no text generation" rule holds.

Experiment: a deletion test. Removing the attributed spans should flip or flatten the decision more than removing
random spans of equal length.

### 5.11 Benchmark contribution (easy; gap 10)
Publish the workload that JevBench #47 asks for: one state with {1, 4, 16, 64, 256} questions, latency vs question
count, and ECE with its bootstrap noise floor. The probe battery and latency tooling already exist. The benchmark
also shows the shape advantage of our design.

### 5.12 Compatibility and adoption (easy)
- Offer a `/v1/systemone`-compatible adapter. Kev, Ollaya, Solar Decide and OpenRouter all speak it. Then ship an
  MCP server and an Ollaya bundle.
- Wire-format compatibility is interface mirroring only. As `docs/SPEC.md` §6 says, it is not TypeSafe's code.

### Lower priority (considered)
- Proxy-tuning logit arithmetic (arXiv:2401.08565), as a pool with no fitted weights
- paraphrase ensembles (AMA, arXiv:2210.02441)
- focal loss (arXiv:2002.09437)
- Venn-Abers for Noul
- early exit (PABEE, LayerSkip; LayerSkip needs backbone training)
- SetFit-style contrastive per-tenant heads

Teacher distillation (JevK5, Decider, Open-Jev style) is out of scope by design.

---

## 6. Suggested order

1. First, inference only:
   - 5.1: fix the 9B source
   - 5.2: conformal sets and abstention in the API
   - 5.4: calibration grid
   - 5.11: workload benchmark
2. Next:
   - 5.5: hidden-state heads, specialist mode on typed-decisions for a fair comparison with Laya and Verdict
   - 5.6: better stitch plus warm start, on the training machine
   - 5.3: coherence constraints
3. Then:
   - 5.8: incremental state
   - 5.7: edge cascade
   - 5.9: contrastive programmatic data
   - 5.10: attribution

After steps 1–2, the positioning becomes "the reliable decision layer": calibrated, abstains, coherent,
order-invariant, portable across model sizes, and local.
