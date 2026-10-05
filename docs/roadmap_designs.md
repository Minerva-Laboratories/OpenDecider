# Roadmap designs: `none` sink, few-label recalibration, state reuse, explanations

Design notes for four roadmap items from `docs/LANDSCAPE.md` §5. Each section gives the problem, the design, the
training data it needs, the deciding experiment and the current status.

## Status

| Item | Code | Status |
|---|---|---|
| 1. `none` option | `src/opendecider/decider.py` (explicit text option, default), `src/opendecider/v3.py` (`v3_none`, learned vector), `scripts/train_none.py`, `tests/test_none_sink.py` | Implemented. The explicit text option is the default; results below. |
| 2. Few-label recalibration | `src/opendecider/calibrators.py`, `src/opendecider/calibration.py`, `eval/calib_curve.py` | Implemented. `method: "auto"` is the default in `/v1/calibrate`. |
| 3. State reuse, Tier 1 (prefix cache) | `src/opendecider/chunking.py`, `src/opendecider/state_cache.py`, `scripts/bench_state_cache.py` | Implemented and used by `Decider._encode_state`. Benchmark below. |
| 3. State reuse, Tier 2 (store) | `src/opendecider/store.py`, `src/opendecider/embed.py`, `eval/store_recall.py` | Library and recall evaluation implemented. Not yet wired into `/v1/decide`. |
| 3. State reuse, Tier 3 (chunk-local features) | none | Research; not started. |
| 4. Explanations | `src/opendecider/explain.py`, `/v1/explain` in `src/opendecider/api.py`, `eval/explain_eval.py` | Implemented and evaluated; results below. |

---

## 1. Sink option (`none`)

### Problem

Jev and most reproductions have no native way to say "no listed option fits" or "the state does not say".
- Third parties removed the explicit "unknown" option on KoBBQ. Jev's accuracy dropped from 0.95 to 0.00.
- Verdict's built-in abstain option falls to 18% recall on hard negatives (`LANDSCAPE.md` §2).

### Design

Two variants were built. The explicit text option is the default because it works without training (results below).

Explicit text option (default, `Decider`):
- Every Choice, Score and Noul question gets one extra option with the text `none of the above`
  (`OPENDECIDER_NONE_TEXT`; empty turns it off).
- The backbone reads it in context like any other option. The trunk scores it in the same softmax.
- Calibration profiles are fitted on the real options. At inference the `none` column is set aside, the real options
  are calibrated, and P(none) is put back.
- Options interact in the trunk, so adding the option changes the real-option probabilities slightly.

Learned vector (`v3_none`, experimental):
- `none` has no text. Its representation is a learned vector.
  - The V3 trunk receives options as token-embedding spans (`v3_option_repr="embed"`). `none` is a length-1 span.
    Its feature is a trainable parameter (`none_x`).
  - Its LM log-prob feature (`lpX`) is a learned scalar (`none_lp`, initialized to -3.0), because there is no text
    to score.
- `none` has no slot and no position. The permutation-invariance guarantee still holds (`tests/test_none_sink.py`).
- `none` works as a probability sink. Mass flows there when no listed option is supported, as attention sinks absorb
  attention with no useful target.

API (implemented in `Decider`):
- Each answer reports `"none": P(none)` (explicit option, or the learned vector on `v3_none` models).
- `probs` are renormalized over the caller's options.
- `value` is `"none"` when P(none) exceeds every listed option.
- `"none": false` on a question turns it off. Use it for closed-world questions where an option always fits.

Planned interactions with other components:
- Conformal sets (roadmap §5.2) may contain `none`.
- `abstain` means P(none) > τ, or a set that is too large.
- The two-stage path for more than 255 options keeps `none` in both stages.
- Calibration profiles fit the `none` base rate per tenant.

Optional split into two sinks:
- `unanswerable`: the state lacks the information.
- `no_fit`: the information is present, but no listed option matches.

The caller acts differently on each ("fetch more context" vs "extend the taxonomy"). Start with one sink. Split it
only if the probes show the two cases are separable.

### Training data

All targets are programmatic; no teacher.
- Option dropout: remove the gold option, keep the distractors, target `none`. Works on any labeled dataset.
- Hard negatives: remove the gold option but keep the closest distractor (a near-miss paraphrase, or a sibling intent
  in Banking77/CLINC). This is where Verdict's abstain fails.
- Unknowable states: `data/builders/uncertainty_data.py` pairs questions with unrelated states. The target changes
  from uniform over the options to `none`.
- Ambiguous evidence keeps the uniform soft target. Rule: absent evidence → `none`; ambiguous evidence → spread.
- Mix: about 10–15% `none` targets, so the prior stays small. The per-tenant profile corrects the base rate.

`scripts/train_none.py` trains only `none_x` and `none_lp` on the frozen trained model (`runs/x2b/model.pt`). It
uses three item kinds: answerable (target gold), gold removed (target `none`) and unknowable (unrelated state from
`data/synthetic/uncertainty_train.jsonl`, target `none`). Questions with more than 16 options keep the gold plus 15
random distractors.

### Experiments

- AUROC of P(none) on held-out unanswerable and gold-removed items.
- Accuracy cost on answerable items. Target: ≤ 0.5 points.
- KoBBQ-style "remove the correct option" probe.
- Hard-negative recall, compared with Verdict's 18%.
- Permutation TV stays 0.000.

### Results

2B checkpoint (`runs/x2b`), held-out val questions from clinc, massive, dbpedia, arc, arc_challenge and snli
(questions with more than 16 options keep the gold option plus 15 random distractors), plus 150 held-out unrelated-state
questions. Mean P(none) per case and AUROC of P(none) against answerable questions:

| Variant | Training | P(none) answerable | P(none) gold removed | P(none) unrelated state | AUROC gold removed | AUROC unrelated state | Accuracy (answerable) |
|---|---|---|---|---|---|---|---|
| Learned vector, random init | none | 0.318 | 0.697 | 0.349 | 0.834 | 0.576 | 0.892 |
| Learned vector, 2 parameters trained (300 steps) | sink only | 0.295 | 0.684 | 0.284 | 0.839 | 0.537 | 0.892 |
| Explicit text `none of the above` | none | 0.097 | 0.430 | 0.151 | 0.845 | 0.795 | 0.892 |

- The explicit option is better on every measure and needs no training. The backbone already understands the phrase.
- Training only the learned vector does not help. A useful learned sink needs the trunk trained with `none` targets.
- Even a random sink separates "gold removed" well (AUROC 0.83): when the right option is missing, the others score
  lower, so any extra option gains mass.

Sources: `runs/none-explicit/none_eval.json`, `runs/x2b-none/none_eval.json`, `runs/train_none.log`.

---

## 2. Few-label recalibration

### Measurements

`eval/calib_curve.py` (runs `calib-*` and `calib2-*`) measures per-model calibrators against the number of labels,
from 25 upward. Results are in `docs/RESULTS.md` §6–7. Calibrators tested:
- temperature
- ensemble temperature scaling (ETS)
- isotonic (per option, class-agnostic)
- isotonic shrunk toward temperature (`iso_shrunk`)
- top-label isotonic (argmax-preserving)
- histogram binning
- beta calibration
- plain, residual and forest trees

### Findings

1. Temperature fitted on 25 labels can hurt an already calibrated model (9B on OpenBookQA: 0.542 vs 0.319 raw).
   Small-n fits must shrink toward "no change", or toward a global or predicted temperature.
2. Beta calibration (3 parameters: a, b, c on [log p, −log(1−p)]) is the steadiest choice with few labels. It is
   never meaningfully worse than temperature, and much better where temperature overfits.
3. Isotonic helps the 9B on typed-decisions (log loss 1.084 → 1.069) from about 50 labels. Its accuracy loss is a
   tie artifact: flat steps tie options. `iso_shrunk` (blend with the temperature model, weight n/(n+100)) breaks
   the ties and keeps the argmax.
4. Histogram binning wins on the skewed binary prompt-injection task. ETS wins on some QA sets. Trees need 250+
   labels (`docs/RESULTS.md` §6).

### Policy design

Choose by cross-validation (CV) on the tenant's labels. Fall back to the simpler method when the CV difference is
within noise.

| Labels | Default | Candidates checked by CV |
|---|---|---|
| 0 | global profile (later: Thermometer-predicted temperature) | none |
| < 50 | beta, shrunk toward identity (L2 on a=b=1, c=0) | temperature (shrunk), ETS |
| 50–250 | iso_shrunk | beta, ETS, histogram (binary schemas) |
| > 250 | iso_shrunk or residual GBDT | + Dirichlet or vector scaling for fixed option sets |
| Score questions | CORDIAL (arXiv:2609.29807) at any size, once replicated | the rest |

### Implementation

`src/opendecider/calibrators.py` and `src/opendecider/calibration.py`, `method: "auto"` (the default):
- K-fold CV log loss with 5 folds; leave-one-out under 25 labels.
- Candidates: identity, temperature (L2 on log T, strength 2/n), beta (monotone, L2 toward a=b=1, c=0 with strength
  2/n), ETS, vector, histogram (binary schemas only) and iso_shrunk.
- Rule: keep the simplest candidate within one standard error (SE) of the best CV log loss. Simplicity order:
  identity < temperature < beta < ets < vector < histogram < iso_shrunk.
- The profile stores the chosen method, its parameters and the CV table. It is keyed by option name, so it does not
  depend on the caller's option order.

### Not yet in the grid

- Venn-Abers (inductive; gives calibrated probability intervals, which suits Noul).
- CORDIAL for Score questions.
- Hierarchical per-tenant temperature (empirical-Bayes shrinkage toward the global value).
- Thermometer (zero-label temperature predicted from features, trained across our task families).

---

## 3. Reusing states across requests

### Baseline

- The state is prefilled once per request.
- Question rows continue from that cache (`pc.unpack_rows`). Attention KV is stored int8. The Gated DeltaNet (GDN)
  layers carry recurrent and conv state.
- Option and text features are cached by content (`TokenCache`).
- States are serialized as text or JSON (`formatting.state_text`), so record boundaries are explicit.

### Constraint

Qwen3.5 is hybrid: 3 of every 4 layers are GDN, which are recurrent.
- A GDN layer's state after a chunk depends on every earlier token, in order.
- Chunk KV splicing methods built for pure-attention models (CacheBlend, APE, block attention) are therefore not
  exact here. Independently encoded chunks cannot be concatenated.
- Prefix reuse is exact: resume from a saved (attention KV, GDN state) snapshot and prefill only the rest.

The design has three tiers.

### Tier 1: exact prefix cache over chunk boundaries (implemented)

Design:
- Split the serialized state at record boundaries into chunks identified by content hashes.
- Keep a tree of snapshots keyed by chunk hashes, as in SGLang's RadixAttention.
- A request resumes from its deepest matching snapshot and prefills only the remainder.
- The GDN snapshot has a fixed size per layer (heads × d_k × d_v), independent of length.

Implementation:
- `chunking.state_chunks` splits a state into single records: a JSON member or list item of a container larger than
  `max_chars` (default 256), or a text line. Chunks are never groups, because a group boundary moves when records
  are appended. The chunks concatenate exactly to `formatting.state_text(state)`.
- Trailing closers (`]`, `}` and the final newline) are returned separately. The snapshot is taken before them. A
  growing state (events appended to a list, lines appended to a log) therefore shares its whole cached prefix with
  the previous request.
- `state_cache.PrefixCache` keys each snapshot by the cumulative content hash of its chunks. A snapshot stores the
  int8 attention K/V of its own tokens only (a delta over its parent) and the GDN recurrent and conv states at its
  end.
- A request prefills its uncached content in one pass (snapshot taken there), then the closers in a second short
  pass that is never cached.
- Eviction is LRU over leaves only, so every snapshot's ancestors stay present.
- `Decider._encode_state` uses the cache. It is on for models whose question rows continue the state's LLM cache
  (`v3_cross="question"`). Size: `OPENDECIDER_STATE_CACHE_MB` (default 2048; 0 disables).
- Responses report `state_cached_tokens`.

Benchmark (`scripts/bench_state_cache.py --ckpt runs/x2b/model.pt`, `runs/state-cache/bench.json`):
- State: a customer record plus N events. Questions: one Choice, one Noul, one Score.
- Cold, repeat and append run on the same cached `Decider`. Append adds one event to the repeated state.
- Uncached is a separate `Decider` with the cache disabled, run on the appended state.
- Differences are the maximum absolute probability difference, append vs uncached.

| Events | Input tokens | Uncached total (ms) | Cold total (ms) | Repeat total (ms) | Append total (ms) | State phase cold / repeat / append (ms) | Cached tokens on append | Max abs diff | Same argmax |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 883 | 981 | 984 | 601 | 831 | 617 / 235 / 471 | 803 | 0.017 | yes |
| 100 | 3869 | 1865 | 1889 | 753 | 984 | 1377 / 252 / 489 | 3789 | 0.026 | yes |
| 400 | 15177 | 6013 | 5943 | 1441 | 1688 | 4798 / 319 / 547 | 15097 | 0.007 | yes |

Reading:
- Append cost is almost flat in state length: 831–1688 ms total vs 981–6013 ms uncached.
- A cold request costs the same as the uncached path.
- The cached path is not bit-identical to the uncached path. Chunks are tokenized separately, so token boundaries
  can differ. The measured difference is up to 0.026; the argmax matches in all three cases.
- `tests/test_state_cache.py` checks agreement within 5e-2. The design target of 1e-3 against a full re-encode is
  not met.

Not yet implemented:
- Canonical serialization (`state_text(..., canonical=True)`): stable key order, then a pinned head (schema, entity
  profile, policies), then the volatile tail. Today `state_text` keeps the caller's key order, which limits hits.
- API handles: `/v1/decide` returning `state_id`, and later calls sending `state_ref: {id, append: [...]}`.
- Disk spill with the `guards.py` limits (shared GPU flock, ≥4 GB disk floor). The cache is in memory only.

### Tier 2: retrieval-built states for large stores (implemented as a library)

Use case: a knowledge base or history that does not fit the token budget. The state is built per request from a
store.

Store (`src/opendecider/store.py`, one SQLite file, no service):
- `chunks`: one row per chunk (id, source, ts, key path such as `orders[3].status`, text, token count, pinned flag).
- `chunks_fts`: FTS5 index over id, key path and text, ranked with `bm25()`. Key paths and IDs are indexed as text.
  Lexical search handles them much better than dense search. If FTS5 is missing, a vectorized numpy BM25 is used.
- `vecs`: unit-normalized float16 vectors, searched with a numpy matmul (fine up to about 1M chunks on the Orin).
- Embeddings come from a pluggable `embed_fn`. `src/opendecider/embed.py` provides `BackboneEmbedder`: mean-pooled
  middle-layer features of the frozen backbone (strong per Layer by Layer; Skean et al., ICML 2025), so no second
  model is needed. `hash_embedder` is a placeholder for tests.

Retrieval:
- Queries: the request's question prompts, option texts and instructions. Options are discriminative queries:
  "refund" or "chargeback" find the deciding record directly.
- Retrieve once per request, over the union of questions, so all questions share one state.
- Fuse BM25 and dense rankings per query with reciprocal rank fusion (RRF, k0=60). Combine across queries by max
  (default) or sum.
- Selection: pinned chunks first, then greedy MMR over fused candidates until the token budget is spent.

Assembly: `[pinned head] + [retrieved chunks sorted by a stable key (ts, then id), not by score]`. The stable order:
- prevents rank-order leakage (retrieval rank never becomes a position signal);
- lets consecutive requests share longer prefixes, so Tier 1 hits more often.

Robustness: the uncertainty data (buried states) trains the model to ignore unrelated records. The sink option
(§1) catches retrieval misses ("the evidence is not in what was retrieved").

Cost: only the retrieved tail is prefilled. The pinned head comes from Tier 1.

Recall evaluation (`eval/store_recall.py`, `runs/store-recall/recall_hash_seed0.json`):
- Synthetic order records with one buried deciding record per trial; 45 trials per cell; 95% bootstrap CIs.
- Question kinds: `by_id` (order ID), `by_name` (rare surname), `by_event` (an inflected rare word, which exact-token
  BM25 misses).
- Dense vectors come from the hash placeholder embedder. Dense and hybrid numbers therefore do not measure semantic
  retrieval.

Recall@budget, union by max (union by sum in parentheses where it differs):

| Chunks | Method | 512 tokens | 2048 tokens | 8192 tokens |
|---|---|---|---|---|
| 1,000 | BM25 | 0.800 (0.667) | 1.000 (0.711) | 1.000 |
| 1,000 | dense | 0.289 (0.178) | 0.489 (0.378) | 0.689 (0.756) |
| 1,000 | hybrid | 0.467 (0.556) | 0.622 (0.778) | 0.867 (0.978) |
| 10,000 | BM25 | 0.622 (0.667) | 0.667 | 0.978 |
| 10,000 | dense | 0.089 (0.067) | 0.133 (0.156) | 0.356 |
| 10,000 | hybrid | 0.089 (0.133) | 0.400 (0.467) | 0.956 (0.978) |
| 100,000 | BM25 | 0.622 | 0.667 | 0.667 |
| 100,000 | dense | 0.022 | 0.022 | 0.044 |
| 100,000 | hybrid | 0.267 | 0.622 | 0.689 |

Reading:
- BM25 finds `by_id` and `by_name` records at all sizes (per-kind recall 0.867–1.0). It misses `by_event` at 100,000 chunks
  (0.0 at every budget).
- With the placeholder embedder, dense recall falls with store size. At small budgets it dilutes the RRF list, so
  hybrid is below BM25.
- Example CI: BM25 at 100,000 chunks and 8192 tokens is 0.667 [0.511, 0.800].
- Selection p50 at 100,000 chunks: 156–167 ms for BM25, 205–240 ms for hybrid. Index build: 0.62 s for 10,000
  chunks, 6.65 s for 100,000.
- With real embeddings from the 2B backbone (`--embedder opendecider.embed:x2b_embedder`, mean-pooled middle layer),
  union by max, 45 trials:

  | Records | Budget | BM25 | Dense | Hybrid |
  |---|---|---|---|---|
  | 1,000 | 512 | 0.733 [0.600, 0.844] | 0.244 | 0.356 |
  | 1,000 | 2048 | 1.000 | 0.467 | 0.467 |
  | 1,000 | 8192 | 1.000 | 0.600 | 1.000 |
  | 10,000 | 512 | 0.667 [0.511, 0.800] | 0.222 | 0.267 |
  | 10,000 | 2048 | 0.667 | 0.311 | 0.711 [0.578, 0.844] |
  | 10,000 | 8192 | 0.867 [0.756, 0.956] | 0.333 | 0.844 |

  Mean-pooled backbone states are a weak retriever. Dense search finds none of the `by_id` records, but it helps on
  paraphrased questions (`by_event` at 10,000 records and 2048 tokens: dense 0.53, BM25 0.00). At small budgets it
  dilutes the fused list. BM25 stays the default; a dedicated embedding model is future work. Dense search with the
  backbone embedder costs about 180–230 ms p50 per selection. Source: `runs/store-recall/`.

Still to measure: accuracy vs budget, and p50 latency on the Orin for cold, Tier 1 hit, append and Tier 2.

### Tier 3 (research): chunk-local features read by the trunk

- Encode each chunk once, independently, with a short `Record from <source>:` header. Cache its token features by
  content in `TokenCache`.
- The trunk cross-attends over the concatenated chunk features (Fusion-in-Decoder style). The backbone's question
  row sees only the pinned head.
- This removes per-request prefill of retrieved chunks. Chunks are encoded once and shared by all requests and
  tenants.
- It is approximate: chunks do not see each other inside the LLM.
- It needs the trunk trained with its state path enabled (`v3_cross="all"` over chunk features), on the training
  machine.
- Decide it against Tier 2 on accuracy vs latency.

### Evaluation plan

The deciding record for each answer is known, so every quantity is measurable:
- Bury the deciding record in 10k–1M tokens of synthetic records.
- Recall@budget of that record for BM25, dense and hybrid (done with the placeholder embedder, above).
- Accuracy vs budget.
- p50 latency on the Orin for cold, Tier 1 hit, append and Tier 2.

---

## 4. Explaining a decision with the same backbone

### Scope

`/v1/decide` stays generation-free. `docs/SPEC.md` §6 and §11 now allow generation only in a separate, opt-in
endpoint, `POST /v1/explain`. It is never called on the decision path and is timed separately.

### Design (no fine-tuning)

- Use the same frozen backbone. The Qwen3.5 checkpoints are instruction-tuned.
- Prompt: the state, then the question and options, then the decision and runner-up with their probabilities, then
  the evidence records, then an instruction to explain why.
- Generate greedily, or sample N candidates.
- Target cost: explanation tokens only, by reusing the prefilled state cache. The current code re-encodes the state
  with the prompt.

Implementation (`src/opendecider/explain.py`, `Explainer`):
1. Evidence: leave-one-record-out occlusion. Delete one JSON member, list item or text line, and re-decide in one
   batched call. A record's weight is the drop in P(chosen). Records are returned as character offsets into the
   state. (The original design used attention rollout or gradient×input, roadmap §5.10.)
2. Generation: greedy or sampled continuation of the backbone. The tied LM head gives next-token logits.
3. Faithfulness check, no teacher: the decision model reads only the explanation as its state, with option names
   masked. `faithfulness` = P(chosen | masked explanation). With N > 1 samples the most faithful one is returned.
   Below `min_faithfulness`, the response has `explanation: null`.

Request fields (`ExplainRequest`): `state`, one atomic `question`, `samples` (1–8), `min_faithfulness` (0–1),
`evidence` (0–10 records). Composite questions return 400.

Checks from the original design, now in `eval/explain_eval.py`:
- Sufficiency is covered by the simulatability check above. It is compared with a mismatched explanation (another
  case, same question) and with an empty state ("No information.").
- Necessity: the drop in P(decision) when the top evidence record is removed.

### Optional fine-tuning (later)

- A small LoRA on the generation path only, trained by rejection sampling on the model's own explanations that pass
  the faithfulness check (self-training).
- No external teacher, so the no-distillation rule holds. Decision-path weights stay unchanged.

### Experiments

- Faithfulness-score distribution.
- Deletion test: removing cited spans vs random spans of equal length.
- Human spot-check of 50 explanations.
- Latency: explanation tokens only, reported separately from decision latency.

### Results

`eval/explain_eval.py --n 24 --samples 4` on `runs/x2b` (24 typed-decisions test cases, 6 per workflow, one choice
question each). Faithfulness reads only the explanation, with option names masked.

| Metric | Value |
|---|---|
| P(decision), greedy explanation | 0.672 |
| P(decision), best of 4 sampled | 0.876 |
| P(decision), mismatched explanation (other case) | 0.297 |
| P(decision), empty state | 0.281 |
| Decision recovered (argmax), greedy / mismatched | 0.833 / 0.292 |
| Drop in P(decision) when the top evidence record is removed | 0.255 (mean P(decision) 0.506) |
| Latency, greedy, median | 16.5 s |

- Generation blocks the `<think>` and `</think>` tokens. Without that the instruction-tuned backbone opens a
  reasoning block and the first paragraph is empty.
- Explanations can contain small factual slips (one sample said "532 seconds (over 9 minutes)"). A human spot-check
  is still needed.
- Latency is eager decoding on the Orin; the state is re-encoded instead of reusing the prefix cache.

Source: `runs/explain/results.json`.

## Hypothesis (not scheduled): isolate instructions before deciding on injection

Noted 2026-10-04. Our clean models rank injection attempts well (AUROC about 0.96 on deepset/prompt-injections) but are
conservative zero-shot, and detection over raw text has to separate content from instructions in one step. The
hypothesis: decision models alone will not detect injections robustly in practice, though they may improve on current
methods. What should work better is a two-stage design:

1. A separate model, in a separate context, rewrites or extracts what the instruction to the assistant actually is,
   isolating it from untrusted content (documents, emails, tool output).
2. The main model and the decision models then judge the isolated instruction: is it the user's, is it allowed, does
   it conflict with the system instruction. These are easier, cleaner decisions than detection over mixed text.

Experiments, when this resumes: decision on raw text vs on the rewritten instruction, with the rewriter in a separate
context; robustness to injected instructions inside a state (not only detection); a held-out injection set not used for
any earlier choice.
