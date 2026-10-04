# Open decision-model landscape and direction for OpenDecider

This is a snapshot dated 2026-09-29. TypeSafe launched Jev on 2026-09-15, two weeks earlier.

> Sources for every named item are listed in §Sources at the end. Numbers about OpenDecider in this snapshot are superseded by docs/RESULTS.md §13.

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
- (Updated 2026-10-03 for the clean-provenance release; details in docs/RESULTS.md §13.) On benchmarks with real
  labels, OpenDecider 9B is within about 1 to 2 points of its own backbone's letter scores and below the third-party
  Jev numbers on knowledge-heavy multiple choice and injection detection. On typed-decisions, a teacher-agreement
  benchmark we treat as secondary, it scores 0.614 zero-shot.
- Specialists fine-tuned on typed-decisions' own workflows score higher on it (Verdict 0.771, Laya 0.766). We do not
  train on any benchmark we report, so these are not comparable with our zero-shot numbers.

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
| Kev (jaredpalmer/kev) [R], Apache-2.0 | Qwen3.5 0.8B–27B with rank-16 LoRA. A pointer head scores each option's `</opt>` state against the question's `<decide>` state. Serves `/v1/systemone`. | 27B: 0.848 (source says 0.851) vs Jev 0.857 on new sources. 4B/9B are about 4 points behind. MMLU 9B 0.74 (source says 0.73) vs Jev 0.90. | Same head idea as ours. Its README says option order can change the answer. Ours is invariant by construction. Kev fine-tunes the backbone; we keep it frozen. |
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
- JEV-27B [R] (autotrust): 24-slot fp32 head. 84.07% vs Jev 83.85% (six-group mean).
- AutoJev [R] (denis-pplx): 84.60% vs Jev 82.79%; no slot head. (An earlier version of this survey merged these two
  projects into one entry.)
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
  escalated case takes about 1.1 s (source says 1.8 s).

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
| Jev Decision Index v0.2 [R] | 49 entrants (source says 64 on the 0.2 board and 67 on 0.2.1, 2026-10-03), 40 benchmarks, chance-corrected. MIT code; some upstream datasets are restricted. |
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
| O7: run-to-run variance | [A] SD 0.001–0.015. 15 distinct answer sets in 50 identical calls. Batching does not interfere: JS divergence 0.0058 vs 0.0035 (not found at source, 2026-10-03) [R]. | Consistent with a small stochastic sampler rather than float jitter. This matches our `sampler.k` hypothesis (b). |
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

---

## Sources

Links were checked on 2026-10-03. "Verified" means the link resolved and the page matched the description in this
document. Pages change; numbers are as the page showed on that date. A note "source says X" marks a number in the text
that differs from the page.

| Name in this document | Source | Verified (date) | Note |
|---|---|---|---|
| awesome-system-one | https://github.com/andyrewlee/awesome-system-one | 2026-10-03 | Aggregator. |
| awesome-jev-robustness | https://github.com/Yifan-Lan/awesome-jev-robustness | 2026-10-03 | Aggregator. Most [A] rows in §2 trace to it. |
| systemonemodels.org | https://systemonemodels.org | 2026-10-03 | Aggregator. |
| Jev, TypeSafe launch post | https://typesafe.ai/blog/introducing-system-one-models-and-jev | 2026-10-03 | Dated 2026-09-15. |
| TypeSafe docs | https://docs.typesafe.ai | 2026-10-03 | |
| Kev (jaredpalmer/kev) | https://github.com/jaredpalmer/kev | 2026-10-03 | Apache-2.0, rank-16 LoRA, `</opt>`/`<decide>` pointer head, `/v1/systemone`, order caveat all present. Source gives 27B 0.851 (dev) vs Jev 0.857 and MMLU 9B 0.73. The README may have changed since 2026-09-29. |
| AnyJev (nokia-applied-research) | https://github.com/nokia-applied-research/AnyJev | 2026-10-03 | Apache-2.0. Numbers match (Qwen3-8B, BANKING77 20-way). |
| CLM (Contrastive-LM) | https://github.com/Contrastive-LM/CLM | 2026-10-03 | Apache-2.0. Frozen Qwen3-8B, 20M heads, InfoNCE, "up to 9× faster". |
| qwen-rlcd | https://github.com/shamazharikh/qwen-rlcd | 2026-10-03 | GitHub reports no license. Prefix-fork, 4.8e-5, 8–19× present. |
| Decider (Mapika) | https://github.com/Mapika/decider | 2026-10-03 | Apache-2.0. Decision Index v0.2.1: 57.33, #2 of 70; Jev 57.91. |
| open-alternative-jev | https://github.com/ikermoel/open-alternative-jev | 2026-10-03 | Apache-2.0. 73.7%, ECE 0.020, 582 ms, 13.5-point swing present. |
| Featherless simple-jev | https://github.com/featherless-ai/simple-jev | 2026-10-03 | Apache-2.0. |
| SemIf | https://github.com/TheoLeeCJ/SemIf | 2026-10-03 | Formerly named OpenJev. Has a WebGPU demo and an MLX backend. |
| openjev-sglang | https://github.com/ekzhang/openjev-sglang | 2026-10-03 | README calls it an early experiment. |
| litjev | https://github.com/zhengxuyu/litjev | 2026-10-03 | |
| mini-jev | https://github.com/r-ms/mini-jev | 2026-10-03 | |
| jevmlx | https://github.com/bnsd55/jevmlx | 2026-10-03 | |
| jevfire | https://github.com/kikoncuo/jevfire | 2026-10-03 | vLLM-based. |
| open-spark-jev | https://github.com/abhishek085/open-spark-jev | 2026-10-03 | Built for DGX Spark. |
| PocketJev | https://github.com/NullPo-jp/PocketJev | 2026-10-03 | iPhone app, MLX, Qwen3-VL. |
| systemANE | https://github.com/kerryrm/systemANE | 2026-10-03 | Apple Neural Engine. |
| djev | https://github.com/mmastrac/djev | 2026-10-03 | DiffusionGemma decisions server. Related repos: https://github.com/mmastrac/djev-spark, https://github.com/taeold/djev-run. |
| razorback16/openjev | https://github.com/razorback16/openjev | 2026-10-03 | DiffusionGemma decision server. |
| Open-Jev | https://github.com/Zefan-Cai/Open-Jev; https://zefan-cai.github.io/open-jev/benchmarks/ | 2026-10-03 | JevBench public 64.94% / 77.49% / 85.28% on the benchmarks page. 2B median 85.0 ms on one H100 (customer-service workload) in `docs/inference-latency.md`. Not the same project as OpenJev/OpenJevPro. |
| JevK5 | https://github.com/allebee/jevk5 | 2026-10-03 | JevBench v1.4: 62.04. Teacher data and distilled LoRA stated. |
| Nimble | https://github.com/bespokelabsai/nimble | 2026-10-03 | 2,676 pairs, 90.1% vs 93.2% on 324 items. GitHub reports no license. |
| Tev1 (Together) | https://github.com/togethercomputer/tev1 | 2026-10-03 | Exact wording: "Logprobs are model preferences, not calibrated confidence." |
| AutoJev | https://github.com/denis-pplx/autojev | 2026-10-03 | Separate project from JEV-27B. Reports 84.60% vs Jev 82.79% on its own suite; no 24-slot head. |
| JEV-27B | https://huggingface.co/autotrust/JEV-27B | 2026-10-03 | Source of the "24-slot fp32 head" and "84.07% vs 83.85%" (six-group mean) in §1.2. The text's "AutoJev/JEV-27B" joins two projects. |
| Imajev-4B | https://github.com/mohit67890/imajev | 2026-10-03 | #1 on JevBench v1.4.2.2 at 67.37; explicit "can't tell" answer. |
| Eikos | https://github.com/caiovicentino/eikos | 2026-10-03 | Finance; 4B and 27B. |
| Shisa DE-1 | https://huggingface.co/shisa-ai/shisa-de-1 | 2026-10-03 | Gemma-4-26B-A4B base; reads option letters. |
| Laya | https://github.com/NandhaKishorM/laya; https://huggingface.co/convaiinnovations/laya-typed-decisions | 2026-10-03 | 421M, 33 ms on a T4, 0.766 fine-tuned vs 0.362 base, Banking77 0.425. |
| Von | https://github.com/wfzyx/von | 2026-10-03 | 395M; p50 0.023 s on an A10G. |
| Verdict 2.0 | https://github.com/Heman10x-NGU/openJev-verdict-2.0; https://github.com/Heman10x-NGU/Verdict-open-jev | 2026-10-03 | Same author. The first repo has 77.10% and Brier 0.0636. The second has `__insufficient_evidence__`, 18.00% recall with hard negatives, and 48.07% vs Jev 90.80%. |
| GLiNER2.5-Decide | https://huggingface.co/fastino/GLiNER2.5-Decide | 2026-10-03 | 340M confirmed. The card does not state joint decoding of typed questions. |
| Julia 1 | https://huggingface.co/SupersonicLabs/Julia-1 | 2026-10-03 | 144.3M, mmBERT-small encoder. |
| jeff | https://github.com/logan-markewich/jeff | 2026-10-03 | GLiFormer 400M encoder, which fits the encoder camp. Other repos with this name: https://github.com/firelex/jeff (Qwen/Gemma fine-tunes), https://github.com/saembit/jeff-cli (CLI). |
| jevlike | https://github.com/vinnylarouge/jevlike | 2026-10-03 | Small model with a byte-level encoder. |
| NanoJev | https://github.com/TianyuCodings/NanoJev | 2026-10-03 | MIT. Source says 18,760 decision questions per data variant. ViZDoom Basic 128/128 vs Jev 56/128. |
| Jev-Omni | https://huggingface.co/akhilaaa3/Jev-Omni | 2026-10-03 | |
| jev-visual | https://github.com/hr98w/jev-visual | 2026-10-03 | Name match. Another candidate: https://github.com/andrueandersoncs/visual-jev. Not certain which one the aggregator meant. |
| PlayJev | https://github.com/OmniJev/PlayJev | 2026-10-03 | |
| Ollaya | https://github.com/ollaya-dev/ollaya | 2026-10-03 | "Run open decision models locally, the way Ollama runs LLMs"; wire-identical `/v1/systemone`. |
| OpenRouter Decisions | https://openrouter.ai/typesafe/jev-1.13 | 2026-10-03 | Model page says it runs on the OpenRouter Decisions API. See also https://openrouter.ai/labs/jev/compile. |
| OpenJev/OpenJevPro | https://github.com/zhangcy122/OpenJev; https://openjev.pro | 2026-10-03 | PolyForm Noncommercial 1.0.0, isolated per-option passes, sub-35 ms. The website gives "1.8s Deliberate"; 1.1 s not found. |
| OpenAI Decisions API | https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev | 2026-10-03 | Article: DevDay 2026-09-29, 150 ms, limited preview. OpenAI's recap (https://openai.com/index/devday-2026-recap/) returned HTTP 403 to automated fetches, so it is unverified. |
| Upstage Solar Decide | https://openrouter.ai/upstage/solar-decide | 2026-10-03 | Released Sep 28, 2026; 524,288-token (512K) context; `/v1/systemone` schema. No Upstage-hosted page found. |
| Hanzo Kai | https://github.com/hanzoai/kai | 2026-10-03 | Proprietary; repo is a stub description. |
| Respan Span-01 | https://openrouter.ai/respan/span-01 | 2026-10-03 | See also https://github.com/respanai/span-01-showcases. |
| meraGPT Decider 1 | https://meragpt.com/models/state-decider-1 | 2026-10-03 | Ranked 1 among zero-shot general models on the typed-decisions card. |
| typed-decisions | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | 2026-10-03 | Apache-2.0. The leaderboard separates general and specialist rows. |
| JevBench v1.4.2.2 | https://github.com/fstandhartinger/jevbench | 2026-10-03 | MIT; equal-weight harmonic mean of four axes. Live board: https://benchmarkheaven.com/jev-models. The site jevbench.dev is an unrelated game benchmark. |
| JevBench issue #47 | https://github.com/fstandhartinger/jevbench/issues/47 | 2026-10-03 | "Speed: measure the 'one state, many questions' workload". |
| Jev Decision Index v0.2 | https://huggingface.co/spaces/multimodalart/jev-decision-index; https://github.com/apolinario/decision-index | 2026-10-03 | 40 benchmarks, chance-corrected, MIT code, upstream datasets under their own terms. The kit counts 64 entrants on 0.2 and 67 on 0.2.1. |
| jabr/classifier-benchmark | https://github.com/jabr/classifier-benchmark | 2026-10-03 | CC0. v2 suite, 49 tasks: Jev 0.966, Von 1.1 0.720, Laya 0.583 (macro accuracy). |
| LangWatch OOD comparison | https://langwatch.ai/compare/jev-vs-all | 2026-10-03 | Jev 81.9% (11 of 11 tasks); Eikos-27B 80.0% (8 of 11 tasks). |
| NavyaAI | https://www.navyaai.com/blog/jev-typesafe-limitations-production | 2026-10-03 | Jev raw ECE 0.080. The 0.053 row is "jeff, self-hosted". 1.7–1.9× speedup; no prefix caching. |
| arXiv 2609.33209 | https://arxiv.org/abs/2609.33209 | 2026-10-03 | "Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?" 0.064 and 1.14 in the abstract. |
| JevAdvBench (arXiv 2609.31142) | https://arxiv.org/abs/2609.31142 | 2026-10-03 | One appended opinion flips 12.1%. |
| "Type-Safe Is Not Error-Free" (arXiv 2609.26758) | https://arxiv.org/abs/2609.26758 | 2026-10-03 | |
| jev-verify | https://github.com/stillmarcus24/jev-verify | 2026-10-03 | |
| TypeSafe Python and JS SDKs | https://docs.typesafe.ai/sdk.md | 2026-10-03 | npm `@typesafe-ai/sdk`; PyPI `typesafe-sdk`. |
| Vercel AI SDK provider | https://ai-sdk.dev/providers/ai-sdk-providers/typesafe-ai | 2026-10-03 | npm `@ai-sdk/typesafe-ai`. |
| LangChain provider | https://docs.langchain.com/oss/python/integrations/providers/typesafe | 2026-10-03 | PyPI `langchain-typesafe`. |
| jev-mcp | https://github.com/jkudish/jev-mcp | 2026-10-03 | MIT, twelve tools, TypeSafe or configured gateways. Other repos with this name: https://github.com/blakestone-x/jev-mcp, https://github.com/burnigtm/jev-mcp. |
| typesafe-mcp (Go) | https://github.com/itsmostafa/typesafe-mcp | 2026-10-03 | README is now titled "System One Connector". |
| llm-typesafe | https://github.com/simonw/llm-typesafe | 2026-10-03 | Plugin for the `llm` CLI. |
| pg-jev | https://github.com/realZachi/pg-jev | 2026-10-03 | |
| jevframe | https://github.com/ktaletsk/jevframe | 2026-10-03 | |
| jev-tree | https://github.com/reachjalil/jev-tree | 2026-10-03 | Recursive choice past 255 options. Other repos with this name do different things. |
| DSPy adapter | https://github.com/typesafeainate/dspy-typesafeify | 2026-10-03 | DSPy decorator proof of concept. Companion benchmarks: https://github.com/jmanhype/jev-dspy-lab. |
| LlamaIndex adapter | https://github.com/WiktorB2004/llama-index-jev | 2026-10-03 | Reranker and router. |
| Hume | https://archerhume.com/posts/jevs-architecture-unmasked | 2026-10-03 | Median 57.5 ms at 360 tokens and 218 ms at 29,835; 86.5 ms for 1 question and 610 ms for 1,500; caps about 32,768 and 65,536; −0.28 (−0.36 to −0.19); 16/16 vs 11–12/16; 348/415; 192 tokenizers. |
| O6 "other reports" | https://github.com/pawarbi/jev-bias-audit; https://github.com/RINNECODER/jev-behavior-study; https://github.com/pobooo/jev-dice | 2026-10-03 | +0.37 first option; 88% vs 57% by position; first option on 2,000 of 2,000. Found through awesome-jev-robustness. |
| O7 run-to-run variance | https://github.com/jujumilk3/jev-calibration-audit | 2026-10-03 | 15 distinct answer sets in 50 calls. SD range via awesome-jev-robustness. The JS divergence 0.0058 vs 0.0035 was not found in any source checked: unverified. |
| Confidence field | https://docs.typesafe.ai/confidence.md; https://github.com/scienthoon/jev-ood-calibration | 2026-10-03 | Docs give Choice confidence (K·p_max − 1)/(K − 1). The second repo documents rounding to 0.01. |
| Hidden fair die | https://github.com/KantaHayashiAI/jev-does-not-play-dice | 2026-10-03 | 82.9% on its pick at 19% accuracy. The 0.125 Noul–Choice gap is from jev-calibration-audit. |
| KoBBQ abstention | https://github.com/jujumilk3/jev-calibration-audit | 2026-10-03 | 0.950 → 0.000. "0 of 30 out-of-scope" is from https://github.com/priorbench/jev. Complement range 0.71–1.42 is also from jev-calibration-audit. |
| JevOut | https://github.com/xzx34/JevOut; https://arxiv.org/abs/2609.30243 | 2026-10-03 | 312 of 508 flips (61.4%). |
| Permutation Self-Consistency | https://arxiv.org/abs/2310.07712 | 2026-10-03 | |
| PriDe | https://arxiv.org/abs/2309.03882 | 2026-10-03 | |
| NCC (label length) | https://arxiv.org/abs/2511.14385 | 2026-10-03 | |
| DC-PMI | https://arxiv.org/abs/2104.08315 | 2026-10-03 | |
| RAPS | https://arxiv.org/abs/2009.14193 | 2026-10-03 | |
| Clustered conformal | https://arxiv.org/abs/2306.09335 | 2026-10-03 | |
| Conformal abstention | https://arxiv.org/abs/2405.01563 | 2026-10-03 | |
| Mix-n-Match | https://arxiv.org/abs/2003.07329 | 2026-10-03 | |
| Thermometer | https://arxiv.org/abs/2403.08819 | 2026-10-03 | |
| CORDIAL | https://arxiv.org/abs/2609.29807 | 2026-10-03 | 76 of 80 settings, 5–100 labels, in the abstract. |
| Batch Calibration | https://arxiv.org/abs/2309.17249 | 2026-10-03 | |
| Hidden Calibration | https://arxiv.org/abs/2406.16535 | 2026-10-03 | Arxiv title: "Token-based Decision Criteria Are Suboptimal in In-context Learning". 20–50% in the abstract. |
| Layer by Layer | https://arxiv.org/abs/2502.02013 | 2026-10-03 | "Up to 16%" is in the paper body. |
| Chen et al., model stitching | https://arxiv.org/abs/2506.06609 | 2026-10-03 | 50% cheaper SAE training in the abstract. |
| Linear Representation Transferability | https://arxiv.org/abs/2506.00653 | 2026-10-03 | |
| Latent space translation (Maiorca et al.) | https://arxiv.org/abs/2311.00664 | 2026-10-03 | |
| Relative representations | https://arxiv.org/abs/2209.15430 | 2026-10-03 | |
| FrugalGPT | https://arxiv.org/abs/2305.05176 | 2026-10-03 | |
| Cascade deferral (Gupta et al.) | https://arxiv.org/abs/2404.10136 | 2026-10-03 | |
| Gatekeeper | https://arxiv.org/abs/2502.19335 | 2026-10-03 | |
| Learning to defer | https://arxiv.org/abs/2006.01862 | 2026-10-03 | |
| Proxy-tuning | https://arxiv.org/abs/2401.08565 | 2026-10-03 | |
| AMA | https://arxiv.org/abs/2210.02441 | 2026-10-03 | |
| Focal loss | https://arxiv.org/abs/2002.09437 | 2026-10-03 | |
| Venn-Abers | https://arxiv.org/abs/1211.0025 | 2026-10-03 | No ID in the text; Vovk & Petej. |
| PABEE | https://arxiv.org/abs/2006.04152 | 2026-10-03 | No ID in the text. |
| LayerSkip | https://arxiv.org/abs/2404.16710 | 2026-10-03 | No ID in the text. |
| SetFit | https://arxiv.org/abs/2209.11055 | 2026-10-03 | No ID in the text. |
