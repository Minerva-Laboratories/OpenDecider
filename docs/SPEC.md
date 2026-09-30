# OpenDecider specification: testing a hypothesized Jev-like architecture

## 0. Purpose and framing (read first)

This repo builds and evaluates an **open, speculative reconstruction** of a "System One" decision model
in the style of TypeSafe's Jev: unstructured state + typed questions in, calibrated probabilities over
caller-defined options out, no text generation.

**This is a hypothesis test, not a claim about Jev's internals.** TypeSafe has not published Jev's
architecture. The goal is to show whether a design assembled entirely from published, pre-2026
components and open pretrained weights can reproduce Jev's *observable* behavior (interface, latency
scaling, option-order sensitivity, calibration) at low cost. Any write-up must describe results
as "consistent with" or "inconsistent with" public observations, never as "this is how Jev works."

Hardware available: one NVIDIA DGX Spark (training) and one Jetson AGX (edge latency target; confirm
whether it is AGX Orin or AGX Thor, see §8).

---

## 1. Public observations the design must explain

Sources: TypeSafe launch post (2026-09-15), TypeSafe API docs, third-party coverage and user reports.
Items marked (anecdotal) come from informal user testing and must be re-verified if API access exists.

| # | Observation | Source quality |
|---|---|---|
| O1 | Three question types: Choice (≤255 options), Noul (yes/no probability), Score (ordinal levels, expected value) | TypeSafe docs |
| O2 | Choices above 255 handled by a two-stage process (score independently, then choose) | TypeSafe launch post |
| O3 | All questions read the same state, evaluated in parallel and isolated from one another; batching many questions barely changes latency | TypeSafe docs/cookbook |
| O4 | Priced on input tokens only; output "free" → compute dominated by input processing | TypeSafe pricing |
| O5 | 70–500 ms end-to-end latency | TypeSafe (self-measured) |
| O6 | Option order significantly changes probabilities | Anecdotal |
| O7 | Run-to-run variance on identical calls; TypeSafe mentions a "parallel sampler" | Anecdotal + vague vendor wording |
| O8 | Claimed calibration via "RLCD" (RL for Calibrated Decisions), trained on synthetic data | Vendor claim, unverified |

---

## 2. Verified literature basis

Each component below was checked against the original paper. Keep this section accurate; do not
paraphrase papers beyond what they show.

- **Pointer Networks** (Vinyals, Fortunato, Jaitly, NeurIPS 2015, arXiv:1506.03134). Uses attention as a
  pointer to select a member of the input sequence as output, giving an output distribution whose size
  equals the number of input elements. This is the precedent for a *class-agnostic* output head:
  the "vocabulary" is whatever options the caller supplies.
- **Poly-encoders** (Humeau, Shuster, Lachaux, Weston, ICLR 2020, arXiv:1905.01969). Compares
  cross-encoders (full attention over context+candidate; more accurate, slow), bi-encoders (separate
  encoding; fast, less accurate), and poly-encoders (in between). Also finds pre-training on data similar
  to the downstream task matters most. **Design implication:** late fusion (option vectors scored against
  a pooled context) will likely underperform deep fusion; test both (V1 vs V2 below).
- **Perceiver** (Jaegle et al., ICML 2021, arXiv:2103.03206). Cross-attention from a small learned latent
  array into a large input array, then self-attention in latent space: cost O(M·N + L·N²) instead of
  O(L·M²). **Design implication:** optional resampler to compress long state into a fixed memory.
  (Perceiver IO, arXiv:2107.14795, extends this with output queries for arbitrary output structures.)
- **Flamingo** (Alayrac et al., NeurIPS 2022, arXiv:2204.14198). Frozen pretrained LM with newly
  inserted GATED XATTN-DENSE blocks; queries come from language tokens, keys/values from the other
  context; each block's output is multiplied by tanh(α), α initialized to 0, so the model equals the
  original LM at init. A Perceiver Resampler compresses variable-size features to a fixed number of tokens
  (64 in the paper). Ablations found freezing the LM and the tanh gating important.
  **Design implication:** V2 injects state memory into a frozen backbone this way.
- **iTransformer** (Liu et al., ICLR 2024, arXiv:2310.06625). Embeds each variate's whole series as one
  token; attention models relationships *between* variates; generalizes across variate counts.
  **Design implication:** embed each option as one token and let attention relate options to each other
  and to the question. Without positional information, attention over such tokens is
  permutation-equivariant, so it cannot by itself explain O6.
- **MaskGIT** (Chang et al., CVPR 2022, arXiv:2202.04200). Bidirectional transformer; predicts all masked
  tokens in parallel, keeps the most confident, re-masks the rest, over a small fixed number of steps
  (~8 for 256 image tokens). **Design implication:** *not required* for independent single-choice
  questions. Only relevant if we later add interdependent outputs. Listed so we can state why it is out
  of scope.
- **RLCR** (Damani et al., arXiv:2507.16806; ICLR 2026). RL reward = binary correctness + Brier score;
  proves that bounded proper scoring rules yield accurate and calibrated predictions; improves calibration
  without accuracy loss vs. RLVR. Note: their setup has the model emit one answer plus a verbalized
  confidence, and they argue bounded rules are preferable to unbounded log-likelihood *in that RL setting*.
  **Design implication:** our "RLCD-like" stage uses Brier + correctness; see §5.3 for when RL is
  actually needed versus plain supervised proper-scoring losses.

Other known components used without further claims: LoRA; temperature scaling; ECE/Brier/NLL metrics;
CORAL-style ordinal losses (optional for Score).

---

## 3. Core hypothesis ("SlotPointer")

A frozen pretrained LM encodes the state once into a memory. Each question is processed in isolation:
its options are embedded as one token each (iTransformer-style), tagged with a **learned slot embedding**
(index table of size 256; index 0 reserved), fused with the state memory via cross-attention
(Perceiver/Flamingo-style), and scored by a pointer head (one logit per option token) with a softmax
across options.

Why slot embeddings: they are the single assumption that jointly explains O1/O2 (hard cap of 255 = 256
slots minus one reserved), O6 (order sensitivity), and a fixed-cost output head. This is inference from
public behavior, not confirmed.

Why this reproduces O3/O4: state is encoded once; each question is a short, independent computation
against cached memory; no autoregressive decoding exists, so cost scales with input.

Why O7 might occur: either (a) ordinary GPU nondeterminism (tiny jitter), or (b) a deliberate sampler,
e.g., averaging K stochastic passes of the small decision module over the cached memory. We implement
(b) as an option and measure both.

---

## 4. Model variants

All variants share the API in §6 and the same frozen backbone.

**Backbone (frozen, bf16):** start with Qwen3.5-2B (fast iteration); scale to Qwen3.5-9B if V2 shows
promise. Pin exact revisions in `configs/backbone.yaml`. State features = hidden states from a chosen
layer set (default: last layer; ablate middle layers). Note: causal-LM hidden states are fine as memory
because the decision module cross-attends over *all* positions; later tokens have seen the whole state.

### V0 — LM-with-head baseline (Open-Jev style)
Sequence = `[state][question][opt_1 marker]...[opt_K marker]`, causal attention, prefix KV cache shared
across questions. Readout: small MLP on the hidden state at each option marker → logit → softmax.
Purpose: the "it's just an LLM with a head" control. If V2 ≈ V0, architecture is not the differentiator.

### V1 — Late fusion (poly-encoder-like)
- State → backbone → memory `S` (T×d), optional Perceiver resampler → `S'` (N×d, N=256 default).
- Question text → backbone → pooled vector `q`.
- Each option text → backbone (batched, short) → pooled vector `o_k`; add `slot_emb[k]`.
- Decision module: 4-layer transformer over `[q; o_1..o_K; m learned codes]` with cross-attention to `S'`.
- Pointer head: `logit_k = w·tanh(W1·h_ok + W2·h_q)` (Ptr-Net form) or bilinear `h_q^T W h_ok`.
- Cheapest; expected to lose accuracy per poly-encoder findings.

### V2 — Deep fusion (Flamingo-style) — main hypothesis
- State → backbone once → memory `S` (cached per request).
- Per question: sequence `[question tokens][OPT_1 tokens]...[OPT_K tokens]` runs through the frozen
  backbone with **gated xattn-dense blocks inserted every 4 layers**, cross-attending to `S`
  (tanh gate, α init 0, as in Flamingo).
- Option pooling: mean-pool each option's span → option token; add `slot_emb[k]`.
- Small option-interaction block (2 layers, iTransformer-style attention across option tokens + question).
- Pointer head as in V1.
- Trainable: gated xattn blocks, slot table, option-interaction block, head (+ optional LoRA on backbone).

### Toggles (all variants)
- `slot_emb: {learned, none, random_per_call}` — `none` should make outputs permutation-invariant; used to
  test the O6 mechanism.
- `sampler: {off, mc_dropout_k, gaussian_noise_k}` with K ∈ {1,4,8}; returns mean probs + std.
- `resampler: {off, N}`.

### Question types
- **Choice:** softmax over K ≤ 255 option logits.
- **Noul:** Choice with two fixed options ("yes", "no"); return P(yes). (Also try sigmoid on q; compare.)
- **Score:** Choice over ordinal level options; return distribution and expected index. Optional ordinal
  auxiliary loss.
- **>255 options (two-stage, mirrors O2):** stage 1 scores options independently in chunks of ≤255
  (per-option sigmoid relevance), stage 2 runs Choice over the top ≤255.

---

## 5. Training

### 5.1 Data (all reformatted into the §6 schema)
- Public classification / NLI / intent / MCQ / reading comprehension sets with permissive licenses
  (e.g., NLI, BoolQ-style yes/no, intent classification with 77–150 classes, topic classification,
  multiple-choice QA). Record license per dataset in `data/MANIFEST.md`; exclude anything non-redistributable
  from released artifacts.
- **Synthetic tasks** (TypeSafe says Jev is trained exclusively on synthetic data): generate states,
  questions, and option sets with an open-weight teacher LLM. Prefer tasks with *programmatically
  verifiable* answers (extraction from generated structured records, arithmetic over tables, rule-based
  routing, game/simulator states) so labels are ground truth, not teacher opinion.
- Augmentations (mandatory): paraphrase option wording; vary option count 2–255; shuffle option order
  per epoch; add plausible distractors; long/short label variants with equal meaning; multiple questions
  per state.

### 5.2 Stage 1 — supervised proper-scoring training
Loss = cross-entropy (log score) on the option softmax + λ·Brier. Both are proper scoring rules; with
ground-truth labels this directly optimizes calibrated probabilities.

### 5.3 Stage 2 — "RLCD-like" (only where it adds something)
With full distributions and known labels, expected Brier/log reward under the policy is equivalent to the
supervised loss, so RL adds nothing there. Use RL only where the signal is outcome-only:
- multi-step environments (e.g., wiki-link navigation, simple game states) where reward arrives after
  a sequence of decisions;
- verifier-scored tasks without per-option labels.
Reward = correctness + Brier on the chosen option's probability (RLCR form). Algorithm: REINFORCE with
baseline or GRPO over sampled choices. Document clearly that this is *our* guess at RLCD.

### 5.4 Post-hoc calibration
Fit a single temperature on a held-out calibration split; report metrics with and without it.

### 5.5 Budget (rough estimates, verify empirically)
Frozen 2B backbone forward ≈ 4 GFLOP/token. At ~40–60 TFLOPs effective bf16 on Spark (reviews measured
~100 TFLOPs bf16 peak via MAMF), throughput ≈ 10–15k tokens/s → ~200–500M training tokens in roughly
5–14 hours for V1/V2 with a frozen backbone. 9B backbone ≈ 4.5× slower. Spark memory bandwidth
(~273 GB/s) limits large-batch throughput; prefer larger compute-bound batches, gradient checkpointing
in trainable blocks only.

---

## 6. API contract (mirror of the public Jev interface shape; not TypeSafe's code)

```json
POST /v1/decide
{
  "state": "string or JSON-serializable object",
  "questions": {
    "route":   {"type": "choice", "prompt": "...", "options": ["billing", "technical", "refund", "other"]},
    "urgent":  {"type": "noul",   "prompt": "..."},
    "severity":{"type": "score",  "prompt": "...", "levels": ["low","medium","high","critical"]}
  },
  "sampler": {"k": 1}
}
→ {"answers": {"route": {"value": "refund", "probs": {...}, "std": {...}}, ...},
   "timing_ms": {...}, "input_tokens": 0}
```
`/v1/decide` never generates free text. Invalid schema → 400. Probabilities must sum to 1 within 1e-6.
Explanations are a separate, opt-in endpoint (`POST /v1/explain`, same frozen backbone, maintainer-approved
2026-09-30); it is never called on the decision path and its latency/cost are reported separately.

---

## 7. Evaluation

### 7.1 Held-out design
Split by **task family**, not by example: at least 30% of task families unseen in training. Report
in-distribution and OOD separately.

### 7.2 Metrics
Accuracy, NLL, Brier, ECE (15 bins, equal-mass), AUROC of confidence for correctness, selective accuracy
at coverage {50, 80, 95}%, latency p50/p95.

### 7.3 Baselines
1. Frozen backbone, constrained label-token scoring (renormalized over option first tokens; and full
   sequence log-prob with length normalization).
2. V0 (LM-with-head).
3. A classic zero-shot NLI classifier (e.g., a BART-large-MNLI-class model).
4. V1, V2 with toggles.
5. Jev API, if access is obtained (log exact requests; respect ToS).

### 7.4 Behavioral probe battery (run on every model, and on Jev if available)
- **Permutation:** 20 random orderings per item; report mean total-variation distance between
  distributions. Predict: `slot_emb=learned` → nonzero; `none` → ~0.
- **Dummy option:** add an irrelevant option; check whether other options rescale proportionally.
- **Duplicate option:** add an exact duplicate; check probability split.
- **Label length:** same meaning, short vs. long wording; measure shift.
- **Repeat calls:** 50 identical calls; separate float jitter (<1e-4) from sampling variance.
- **Latency scaling:** vary K ∈ {2, 8, 32, 128, 255} and questions ∈ {1, 4, 16, 64}; plot.
The probe signatures, not accuracy alone, are what let us say "consistent/inconsistent with Jev."

---

## 8. Hardware and deployment

- **DGX Spark (training):** GB10, 128 GB unified memory, ~273 GB/s bandwidth, ~1 PFLOP FP4 sparse
  (marketing); treat bf16 ≈ 100 TFLOPs peak (independent measurement) as the planning number.
  ARM64: use NVIDIA's aarch64 PyTorch containers; verify flash-attention / custom kernels build on ARM.
- **Jetson AGX (latency target).** Confirm model:
  - AGX Thor: 128 GB, Blackwell GPU, FP8/FP4 via Transformer Engine; can host the 9B variant.
  - AGX Orin: up to 64 GB, Ampere, no FP4; use FP16/INT8; 2B variant recommended.
  Export path: PyTorch → ONNX → TensorRT; cache state memory per request; batch all questions and options
  into one engine call. Report on-device latency separately from any network latency.
- Latency comparisons with Jev must note that TypeSafe's figures include network and were measured by
  TypeSafe near its servers.

---

## 9. Repo layout

```
configs/            backbone.yaml, v0.yaml, v1.yaml, v2.yaml, train_*.yaml
data/               builders/, synthetic/, MANIFEST.md (licenses)
src/opendecider/
  backbone.py       frozen LM wrapper, feature extraction, KV/memory cache
  fusion.py         gated xattn-dense (tanh gate), perceiver resampler
  options.py        option span pooling, slot embeddings
  heads.py          pointer head, noul, score, two-stage >255
  sampler.py        mc-dropout / noise ensembles
  losses.py         CE, Brier, ordinal aux
  rl.py             outcome-reward stage (REINFORCE/GRPO)
  api.py            /v1/decide server
eval/               metrics.py, probes.py, baselines.py, report.py
deploy/             onnx_export.py, trt_build.py, jetson_bench.py
tests/              schema, prob-sum, permutation-invariance (slot_emb=none), cache equivalence
```

---

## 10. Milestones and success criteria

1. **M1 — Plumbing (V0 + probes):** API, schema tests, probe battery running on V0 and baseline 1.
2. **M2 — V2 on 2B:** matches or beats V0 accuracy on OOD tasks; ECE ≤ 0.05 after temperature scaling.
3. **M3 — Mechanism check:** permutation probe shows slot embeddings produce order sensitivity and
   `none` removes it; latency flat-ish in K.
4. **M4 — Edge:** 2B V2 on AGX with p50 < 150 ms for ~1k-token state, 16 questions, K ≤ 32 (target;
   adjust after first measurement).
5. **M5 — Report:** side-by-side probe signatures (ours vs. Jev if available), clearly labeled as
   hypothesis testing.

**Kill/pivot rules:** if V2 does not beat V0 on OOD accuracy or calibration, report that architecture is
not the differentiator and that data/training likely are. That is a valid, publishable result.

---

## 11. Project rules

- Do not describe any result as revealing Jev's architecture. Use "consistent with public observations."
- Never train on or redistribute data whose license forbids it; update `data/MANIFEST.md` for every source.
- Pin every model and dataset revision; log configs and seeds with every run.
- Keep the backbone frozen unless a config explicitly enables LoRA.
- Every new component needs a unit test; `slot_emb=none` must pass a permutation-invariance test.
- Report all metrics with confidence intervals (bootstrap, 1,000 resamples).
- Do not add text generation paths to the decision API (`/v1/decide`). Generation lives only in `/v1/explain`.
