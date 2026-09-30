# V3 dense late fusion: design note

Proposed 2026-09-22. Implemented in `src/opendecider/v3.py` with the defaults below.

V3 is a candidate variant next to V0/V1/V2 (docs/SPEC.md §4). It follows the same reporting rule: results are
"consistent / inconsistent with public observations".

## Motivation

- V1 pools the question and each option into one vector before any interaction. This discards token-level detail.
  V1 is the weakest variant, as the poly-encoder findings predict.
- V2 keeps detail but needs backprop through the frozen backbone. This is slow on the Orin (about 3.3 s/step).
- Proposal: keep dense token features for state, question and options, straight from the frozen backbone. As in
  V1, there are no backbone gradients. All fusion happens in a small trainable stack. Each option is reduced to a
  single token only at the end, for classification.

## Architecture (as proposed)

Inputs are frozen-backbone features (no grad; cacheable):
- Context C = [state tokens ; question tokens]. Dense per-token features with positional encoding (PE) and a
  segment embedding (state / question).
  - The state part is encoded once per request and cached as int8.
  - The question part is per question and small.
  - So C = cached state K/V + question K/V. The state is still encoded once (O3/O4).
- Options O_k, k = 1..K. Each is the dense token sequence of that option's text, with PE within the option, a
  segment embedding (option) and a slot tag. Slot tags carry identity only, not content (see the slot notes below).

Trainable stack of N layers. Each layer has:
1. Across-option attention (iTransformer-like). Option representations attend to each other bidirectionally.
   Without slot tags it is permutation-equivariant over options.
2. Cross-attention between options and context C. Together with (1), this is the only mixing between the three
   parts (state / question / options).
3. An FFN applied per option (iTransformer style). It operates on one option at a time and never mixes options.

Readout:
- Use the final layer's features, or a learned mix of several layers.
- Produce one CLS token per option, for example with a learned query that attends over that option's tokens.
- The pointer or linear head gives one logit per option. A softmax runs across options.

## Design questions from the proposal

| # | Question | Resolution |
|---|---|---|
| 1 | Cross-attention direction: options→context (options query C), context→options, or both (co-attention)? If only options query C, the question and state interact only through the options. Multi-hop reasoning may need context self-attention (question↔state) or bidirectional cross-attention. | Options → context (see Decisions). Context self-attention added later as `v3_context_layers`. |
| 2 | Granularity of across-option attention: all option tokens jointly (K·L_o tokens), or per-option summary tokens (one per option, iTransformer-exact) that broadcast back to the tokens? | Open. |
| 3 | "FFN per option": a standard token-wise FFN restricted to one option, or an option-level mixing FFN over its tokens (needs a fixed length or pooling)? | Open. |
| 4 | Within-option token mixing: should an option's tokens attend to each other, or rely only on the causal backbone features (the last token has already seen the whole option)? | Open. |
| 5 | Feature layers: final layer vs a learned mix of intermediate layers (decoder middle layers often give better features). | Open. |
| 6 | Slot tags: `learned` (can carry an order prior, the O6 hypothesis), `random_per_call` (identity only), or `none` (exactly permutation-invariant). The M3 permutation probe should run on all three. | See Decisions. |

## Cost sketch (0.8B backbone, d = 1024 → d_model 512)

- Backbone: unchanged. The state runs once per request, plus the question and option texts. Option and question
  dense features are cacheable by text (int8 per token). This extends V1's pooled-text cache to token level.
- Stack: across-option attention over K·L_o tokens (for example 255 × 5 ≈ 1.3k), and cross-attention from K·L_o
  queries to T_state + T_q keys. Both are small next to one backbone pass.
- Training: no backbone gradients, so speed is V1-like (~2 s/step or better on the Orin with warm caches).

## Relation to the hypothesis tests

- O1/O2: slot table of 256, 255 cap.
- O3/O4: state encoded once, questions isolated.
- O6: mechanism test through slot toggles and permutation-equivariant across-option attention.
- O7: sampler on the small stack.
- Compare with V0/V1/V2 on OOD accuracy and calibration. The §10 kill/pivot rule applies.

## Decisions (2026-09-22)

- Cross-attention direction: options → context (O queries C). This is the intended design. C stays static (frozen
  features + PE + segment), so the state part is shared across questions. It is the default
  (`v3_context_update: false`).
- Slots: V3 does not need them. Option membership comes from a per-head same/other-option attention bias.
  `slot_emb` stays only as the O6 mechanism toggle: train with `none`, probe `learned` / `random_per_call`.
- Norms: ScaleNorm everywhere (pre-norm). FFNs: first-order xATGLU.

## Later options (not now)

- Flamingo-style joint self-attention: concatenate [context ; options] into one sequence for self-attention. The
  context then also models its own relationships (state↔question↔options), with gated cross-attention as in
  Flamingo. This is more expressive, but C becomes question-specific, so the state is no longer shared across
  questions.
- `v3_context_update: true` (C also queries O). Already implemented as a toggle.

## Question-aware context: literature (verified against arXiv on 2026-09-22)

| Work | Reference | Relevance |
|---|---|---|
| BiDAF | Seo, Kembhavi, Farhadi, arXiv:1611.01603 (ICLR 2017) | Bi-directional attention flow gives "a query-aware context representation without early summarization". Context and question are encoded separately, then interact. Closest to `v3_context_layers`. |
| Dynamic Coattention Networks | Xiong, Zhong, Socher, arXiv:1611.01604 | Coattention between question and document. |
| QANet | Yu, Dohan, Luong et al., arXiv:1804.09541 | Convolution + self-attention reader with context-query attention. |
| BLIP-2 | Li, Li, Savarese, Hoi, arXiv:2301.12597 | The Q-Former's learned queries extract from a frozen encoder. |
| InstructBLIP | Dai, Li, Li et al., arXiv:2305.06500 | "Instruction-aware Query Transformer, which extracts informative features tailored to the given instruction". Analogy: frozen image features = our state, instruction = our question. Candidate follow-up: question-aware extraction queries (fewer tokens read per option). |
| End-To-End Memory Networks | Sukhbaatar, Szlam, Weston, Fergus, arXiv:1503.08895 | Multi-hop attention over a memory with a query. |
| ColBERT | Khattab, Zaharia, arXiv:2004.12832 | Cached token embeddings + late interaction. Relevant to caching, but the interaction is too shallow for binding. |

## Implemented (2026-09-22)

- `v3_context_layers: N`: joint self-attention over [state ; question] per question. Pre-norm residual, ScaleNorm,
  xATGLU, RoPE with continuous positions. Only the small stack does per-question work. The backbone state encoding
  stays shared.
- Diagnostic runs on record_lookup only (question→record→field binding): V0 vs V3+context(2) vs V3.

## Lookup diagnostic (record_lookup only, ~1.5k train states, 522 val questions; Orin; 2026-09-22)

| model | val acc @300 | @600 | @900 | @1200 | notes |
|---|---|---|---|---|---|
| LLM zero-shot (label scoring) | 0.67 | | | | no training |
| V0 (frozen LLM prefix cache + last-token head) | 0.774 | 0.814 | 0.826 | 0.833 | T=1.04, well calibrated |
| V3 isolated features + 2 context layers | 0.245 | 0.249 | 0.257 | 0.264 | never learns binding; T=6.7 |
| V3 conditioned features (`v3_features: conditioned`) | 0.799 | (running) | | | 4.5 s/step, needs optimizing |

Conclusion:
- Question→state binding must come from the frozen backbone's own attention.
- A from-scratch stack on isolated per-text features does not learn binding at this data scale, even with
  question-aware context layers.
- The V3 stack works well on top of backbone-conditioned dense features.

### `v3_features: conditioned` (a V0/V3 hybrid; not one of the spec's V0/V1/V2)

- States: one left-padded batched prefix pass. It equals encoding each state alone exactly (tested). Its hidden
  states form the state part of the context. Its KV and GDN states form the prefix cache.
- One row per question: [question ; opt_1 ; ... ; opt_K], continuing from its state's cache (int8 KV). Dense
  question and option tokens from this pass feed the V3 stack. No pooling until the per-option CLS.
- Trade-off: the joint causal row makes option order matter at the backbone level (like V0 / V2-joint).
- Branching each option from a [state; question] cache would be permutation-invariant. But each branched row
  carries its own copy of the GDN recurrent states (18 layers x 16 heads x 128x128 fp32 ≈ 18 MB per row; about
  4.6 GB for K=255). This was deferred at the time (see `docs/branched_shared_prefix_math.md`).

### Question-only conditioning (`v3_features: question_conditioned`): dropped 2026-09-22, ceiling too low

- Option final hidden states vs the question's final state: chance (0.264, zero-shot). Hidden states predict the
  next token.
- Option token embeddings (tied LM head space): 0.414 zero-shot. A trained bilinear readout plateaus at ~0.45.
- Compare: options inside the backbone row (conditioned) 0.826; zero-shot LM with options listed in the prompt 0.67.
- So the backbone must see the candidate options while it reads.
- A trained V3 trunk also stayed at chance on these features, below the 0.45 linear ceiling. This is an open
  optimization issue for cross-attention on real features. The toy retrieval task learns fine with attention + FFN
  blocks.
- Next step at the time: branched options. Rows [question ; option_k] continue from the state cache, so each option
  is seen in context with no cross-option order effects. Rows are chunked to bound the ~18 MB-per-row GDN state
  copies.

### Results update (2026-09-23)

- Branched V3 (`v3_features: branched`, `v3_cross: question`, lookups): val acc 0.833 / **0.881** / 0.879 at
  steps 300/600/900, CE 0.29, T=1.43. It beats V0 (0.833) and conditioned V3 (0.826) and is exactly
  permutation-invariant. About 6.5 s/step on the Orin (one backbone row per option).
- All families, conditioned V3 vs V0 (1500 steps): 0.504 vs 0.508. A tie.
- V0 test (500 states per split, 95% bootstrap CI):
  - ID acc 0.560 [0.536, 0.584], ECE 0.029
  - OOD acc 0.494 [0.471, 0.518], ECE 0.052
