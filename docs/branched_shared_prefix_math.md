# Cheap branched rows: one shared state prefix for many option rows (derivation, 2026-09-23)

## Problem

Branched V3 runs one backbone row `[question ; option_k]` per option. Each row continues from the cached state. The
current code copies the prefix cache into every row:

| Layer type | Per-row copy | Size |
|---|---|---|
| GDN layers (18) | recurrent state 16 heads × 128 × 128 fp32 ≈ 1 MB per layer | ≈ 18 MB per row |
| Attention layers (6) | prefix K/V, T × 2 kv-heads × 256 × 2 × bf16 ≈ 2 KB per token per layer | ≈ 9 MB per row at T = 768 |

At K = 255 options this is several GB per question. It dominates the branched training step (6.5 s/step vs 1.8 s
for V0).

## 1. Gated DeltaNet layers: affine state recurrence

Source: `torch_recurrent_gated_delta_rule` (transformers qwen3_5). Per head, q and k are l2-normalized, q is scaled
by 1/√d_k, and S ∈ ℝ^{d_k × d_v}:

    S_t = A_t S_{t-1} + β_t k_t v_tᵀ,      A_t = e^{g_t} (I − β_t k_t k_tᵀ)     (A_t symmetric)
    o_t = S_tᵀ q_t

A branch that starts from the shared prefix state S₀ satisfies S_t = P_t S₀ + Z_t. Here P_t = A_t A_{t−1} ⋯ A_1,
and Z_t is the state reached from a zero initial state. Hence

    o_t = S₀ᵀ u_t + o_t⁽⁰⁾,       u_t = P_tᵀ q_t = A_1 (A_2 ( ⋯ (A_t q_t)))

Terms:
- o_t⁽⁰⁾: the ordinary (fla) kernel run on the row from a zero state. It needs no copy of S₀.
- u_t: t rank-1 updates, A_i x = e^{g_i}(x − β_i k_i (k_iᵀ x)). Cost is O(L²·d_k) per head per row. This is
  negligible for branch rows of L ≈ 10–25 tokens. The WY/chunkwise form of DeltaNet computes the same products.
- S₀ᵀ u_t: one matmul per state (rows grouped by state), not per row.
- Short causal conv (kernel 4): each row needs the last 3 pre-conv inputs of its prefix. That is ≈ 3 × 6144 values
  per layer, so a per-row copy is cheap.

Memory per row drops from ≈ 18 MB to O(L·d) activations.

## 2. Full-attention layers: shared-prefix attention (Hydragen, arXiv:2402.05099)

A row's query attends over [prefix ; own suffix]. Split this into two partial softmaxes and merge them exactly
through their log-sum-exp:

    out = (e^{m_p} Z_p · o_p + e^{m_s} Z_s · o_s) / (e^{m_p} Z_p + e^{m_s} Z_s)

- Prefix part: the queries of all rows that share a state are batched against the one prefix K/V. This is a dense
  matmul with no copy.
- Suffix part: ordinary causal attention inside the row.

Hydragen reports up to 32× throughput with shared prefixes.

## Expected effect

- Branched cost per option ≈ the cost of its own ~20 tokens. With no per-row cache copies, the branched step should
  approach V0 speed. Inference with K = 255 options stays cheap (latency O5, batching O3).
- Exactness: both decompositions are algebraically exact, up to float error. Test them against the current
  per-row-copy path (`tests/test_v3.py::test_v3_branched_features_exact_and_order_free`).

## Implementation sketch

A custom forward for Qwen3.5 decoder layers in "branched" mode (hooks or a small wrapper):
- GDN: projections → conv with a per-row 3-token conv state → fla chunk kernel with a zero initial state (o⁽⁰⁾)
  and the u_t recurrence (torch, O(L²)) → grouped S₀ᵀu.
- Attention: projections → RoPE → prefix attention batched per state (return out + lse) → suffix causal attention
  (out + lse) → lse merge.

## Prior art (literature check 2026-09-23): the GDN decomposition is published

The check covered chunkwise DeltaNet and prefix caching for hybrid/SSM models in serving systems.

- SpecLA (Wang et al., arXiv:2607.16673, July 2026): speculative decoding for Gated DeltaNet. It verifies a tree of
  candidates from one committed state S0, with no per-branch state snapshots:
  O_GDN = Q̂S0ᵀ − Attn(W S0ᵀ) + Attn·U, with the tree mask inside the delta-rule factorization. Tree verification
  is 1.80–7.11× faster. The paper mentions Qwen3.5. This is our derivation in chunk (WY) form. Credit it and do not
  claim novelty.
- The same result follows from the chunk formula of DeltaNet (Yang et al. 2024, arXiv:2406.06484):
  O = Q Sᵀ + (QKᵀ⊙M)(U − W Sᵀ). Gated DeltaNet (arXiv:2412.06464) adds decay through a UT transform.
- Production engines (SGLang hybrid radix cache; Marconi, arXiv:2411.19379) still copy recurrent states per
  branch.

Caveats:
- Rows longer than one chunk need a zero-init per-row state. It is computed, not copied.
- The causal-conv tail (3 tokens) is shared.
- S0 is shareable only if the state pass has no adapters. Keep any VeRA/LoRA on the row pass only.
