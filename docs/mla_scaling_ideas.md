# Breaking MLA's scaling: latent RoPE, query back-projection, online compression

Working notes, 2026-10-07. Notation: token `t` has hidden state `x_t`; one attention layer has `H` query heads in `G` groups,
latent width `r`, head dim `D`. MLA caches `c_t = W_c x_t` (r numbers); keys/values are linear in `c`. `R_θ(n)` is the RoPE
rotation for position `n` (block-diagonal 2x2 rotations, one angle per frequency), so `R(m)^T R(n) = R(n − m)`.
Mercurius today (per attention layer): r = 390–656, H = 16, G = 4, D = 256, decoupled 64-dim RoPE key, 4-bit TurboQuant cache.

---

## 1. RoPE on the latent instead of a decoupled RoPE key

### Why MLA decouples RoPE
With RoPE on the expanded key, `k_j = R(j) W_k c_j`, the score is
`s_tj = q_t^T R(t)^T R(j) W_k c_j = q_t^T R(j−t) W_k c_j`.
Absorption needs one fixed matrix between `q_t` and `c_j`; here it is `R(j−t) W_k`, which changes with every pair. So the query
cannot be folded into an `r`-dim vector once per query, and DeepSeek added a separate, shared 64-dim RoPE key instead.

### Claim 1: RoPE applied to the latent coordinates keeps absorption exact
Pick a pairing of the latent coordinates and frequencies `θ_p` (some pairs may have `θ_p = 0`, i.e. no rotation). Define the
score as `s_tj = a_t^T R_lat(j−t) c_j` with `a_t = W_k^T q_t` (the back-projected query, r numbers).
Proof: `R_lat(j−t) = R_lat(t)^T R_lat(j)`, so `s_tj = (R_lat(t) a_t)^T (R_lat(j) c_j)`. The left factor depends only on the
query, the right only on the key and its position. Both are elementwise 2x2 rotations, O(r) work. ∎

So per query: one back-projection plus one O(r) rotation. Per key: one O(r) rotation, shared by **all heads and all queries**
that read that key (in a flash-attention tile it is done once per tile). The cache can keep `c_j` unrotated: values use
`c_j` as is, and keys rotate on the fly. No 64x64 un-rotation and no second codebook.

### Claim 2: at equal cache size, latent RoPE is at least as expressive as decoupled RoPE
Decoupled RoPE caches `[c_j ; k^rope_j]` (r + 64 numbers) and scores
`s = q_n^T W_k c_j + q_r^T R(j−t) k^rope_j`.
Take a latent of width r + 64 whose last 64 coordinates are `k^rope_j = W_r x_j`, rotated with the same frequencies, and the
first r unrotated (`θ = 0`). Then latent RoPE reproduces the decoupled score exactly. Decoupled RoPE is the special case
"the rotating subspace is a separate projection". Latent RoPE can also learn *which* directions rotate and share them with
content. ∎

### Caveats
- **It is a different model.** It needs training or distillation; Mercurius's weights assume the decoupled key.
- **Quantization basis.** The pairing must be defined in the basis the cache decodes to. Mercurius decodes the latent in
  TurboQuant's rotated basis, so the rotary pairs should live in that rotated basis. Otherwise the per-key rotation becomes a
  dense r×r product again.
- **Frequencies.** Allocate a spectrum over r/2 pairs, possibly partial (only some pairs rotate), as in partial-RoPE work.
- **Prior art to read first:** "Tucker Attention" (Klein et al., arXiv 2603.30033, 2026) reportedly applies RoPE in MLA's latent
  space. Mehta et al. 2025 (arXiv 2506.09342) study MLA rank and RoPE placement for small models.

Gain for Mercurius: drops the rope codes (~34 B of ~300 B per cached token per layer, ~11%) and the per-key 64x64 un-rotation.
Decode attention then works on one homogeneous latent.

---

## 2. Back-projecting Q (absorption) vs expanding the compressed KV

"Back-project Q into the latent" is MLA's absorption. Which side to expand is a cost question with a clean answer.

### Claim 3: absorption costs more per (query, key) pair whenever r exceeds the per-head key+value width
Per (query, key, head), in multiply-adds:
- absorbed: score `(rd + r)` + value accumulation `r` → `rd + 2r`
- expanded: `d_k + d_v` (Mercurius: 448 + 256 = 704)

Expanding costs `r · F` per key, where `F = n_keyfeat + G·d_v` is the total expanded width (Mercurius: 960 + 1024 = 1984). This
cost is paid once per key per prefill batch. Back-projecting the query costs about the same as expanding it, so ignore it.

For a batch of N queries against M keys:
`cost_abs = N·M·H·(rd + 2r)` and `cost_exp = N·M·H·(d_k + d_v) + M·r·F`.

Absorption wins iff `N < N* = r·F / (H·(rd + 2r − d_k − d_v))`.

Mercurius, r = 579: `N* = 579·1984 / (16·(64 + 1158 − 704)) = 1.149M / 8288 ≈ 139` queries. ∎

So: **absorbed for decode and short batches (N < ~140), expanded for prefill.** This is what vLLM and SGLang do for MLA,
and what we are building now (llama.cpp flash attention with a new 448/256 head size for prefill, the fused absorbed kernel
for decode). With r = 579, absorption per pair is 1.7x the expanded arithmetic. That is the whole prefill gap we measured.

### Design corollary
The absorbed per-pair cost is set by r, the expanded one by the head dims. A model meant for cheap long prefill *and* cheap
decode wants r close to `d_k + d_v` per group, or a **grouped latent** (each group has its own latent of width r/G).
Grouped latent cuts absorbed cost to `rd + 2r/G` per pair at the same cache size. That is GLA (Zadouri, Strauss, Dao,
"Hardware-Efficient Attention for Fast Decoding", 2025), which matches MLA quality and shards better.

---

## 3. Online latent projection: Oja-style encoder, Hebbian decoder, softmax unchanged

The idea: keep MLA (a cache of one short code per token, softmax over all of them), but instead of one global fixed `W_c`,
**learn the projection online for the current sequence**. The encoder `E_t` (r×d) is updated per token by an Oja-like rule.
Decompression is either a decoder `D_t` (d×r) learned by a Hebbian/delta rule, or no decoder at all, back-projecting the
query into the latent. Encoder and decoder updates are outer-product accumulations, so computationally each is one
linear-attention step per token. But they hold the *projection*, not the content: the content stays in the n cached codes,
and softmax still selects among all of them.

### 3.1 What the projection should optimize
The cache matters only through scores `q^T k` (and values). For encoder `E`, decoder `D`, queries and keys with
covariances `Σ_q`, `Σ_k`:
`L(D, E) = E[(q^T k − q^T D E k)^2] = ‖Σ_q^{1/2} (I − DE) Σ_k^{1/2}‖_F^2`.

**Claim 4 (optimal projection).** With `A = Σ_q^{1/2}`, `B = Σ_k^{1/2}` invertible and `AB = U S V^T`, the minimum over rank-r
`DE` is `Σ_{i>r} s_i^2`, attained at `D = A^{-1} U_r`, `E = S_r V_r^T B^{-1}`.
Proof: `A(I − DE)B = AB − (AD)(EB)`, and `(AD)(EB)` ranges over all rank-≤r matrices; apply Eckart–Young. ∎

- With isotropic queries this is key PCA, which is exactly what Oja's subspace rule `U ← U + η (k − U U^T k)(U^T k)^T`
  converges to: the top-r eigenspace of `E[k k^T]` (Oja 1982, 1989).
- In general the target is **query-weighted** PCA. The online version is Oja on query-whitened keys `A k`, with `A` from a
  running estimate of the query covariance. This spends capacity only on key directions queries actually read.

### 3.2 Reading scores without a decoder: back-project q
With an orthonormal encoder `U` (codes `c_j = U^T k_j`), the natural decoder is `U` itself:
`q^T U c_j = (U^T q)^T c_j`, so the query is projected once (O(d r)) and each key costs O(r), as in MLA absorption.

**Claim 5 (exactness and drift).** If the code was written with `U_j` and read with `U_t`:
`q^T U_t U_j^T k_j = q^T P_j k_j + q^T (U_t − U_j) U_j^T k_j`, so the error is at most `‖q‖ ‖U_t − U_j‖ ‖c_j‖`.
It is zero when the basis has not moved and grows with how far it moved since the write. ∎

Two points make this manageable:
- **Use an ordered rule.** Oja's *subspace* rule only fixes the subspace, so the basis can keep rotating inside it even after
  the subspace is learned, which makes `U_t − U_j` large for no benefit. Sanger's GHA (generalized Hebbian algorithm)
  converges to the individual eigenvectors, so once the subspace is learned the basis stops moving and old codes stay valid.
  The drift is then concentrated in the warm-up and in genuine topic shifts.
- **Re-base when it does move.** Re-encode old codes lazily (`c_j ← U_t^T U_j c_j`, an r×r map per block of codes written
  under one basis), or keep a few block bases and back-project q once per block.

### 3.3 With a decoder: Hebbian / delta rule
Learn `D_t` online from pairs (code, key) with the delta rule `D ← D + β (k_j − D c_j) c_j^T`. With forgetting, this tracks
the least-squares decoder
`D_t = argmin_D Σ_j λ^{t−j} ‖k_j − D c_j‖²`.

**Claim 6 (what the decoder buys).**
1. If all codes come from one orthonormal `U` and keys lie in its span, `D_t → U`: the decoder adds nothing over §3.2.
2. If codes come from a drifting `U_j`, `D_t` is the best single linear readout of all of them, weighted toward recent
   writes. Scores `q^T D_t c_j = (D_t^T q)^T c_j` are still one back-projection per query, so the cost is the same.
   Its residual is exactly the inconsistency between bases: zero when the bases agree, graceful when they drift slowly.
3. The plain Hebbian version `D = Σ λ^{t−j} k_j c_j^T` gives `D = C U = U Λ` for an eigenbasis `U`. It needs a per-component
   `Λ^{-1}` normalization (which Oja/GHA already track) to become the projector. The delta rule normalizes itself.

So: back-projection with the GHA encoder's transpose is the simplest exact form. The delta-rule decoder is the robust form
when the basis moves, at the cost of one d×r state update per token.

### 3.4 Costs and where it would pay
Per token per layer: encoder update O(d r) and, optionally, decoder update O(d r), comparable to one linear-attention step.
Per query: one O(d r) back-projection. Per (query, key): O(r), the same as MLA. State: one or two d×r matrices per sequence
per layer, fixed size.

Evidence that per-sequence bases beat a global one:
- ShadowKV (Sun et al., ICML 2025): pre-RoPE keys are extremely low-rank within a sequence; a sequence and its continuation
  share subspaces, different sequences do not.
- Loki (Singhania et al., NeurIPS 2024): attention scores computed in a low-dimensional key subspace lose little.

Both compute their bases offline or at prefill time. The online rule gives the streaming version, which also adapts during
generation. Two places to try it:
- On a standard model's pre-RoPE keys (Qwen), calibration-free. RoPE goes on the rope part or into the latent (§1).
- On Mercurius's latent, compressing it further per sequence (r = 579 → r' ≈ 128–256), cutting both the cache and the
  absorbed per-pair cost (§2).

Open questions to test first:
- How fast GHA converges relative to context length. The warm-up tokens could be cached uncompressed, or re-encoded once
  the basis settles.
- Whether query whitening is worth its cost versus plain key PCA.
- Interaction with 4-bit quantization: the quantizer's codebook assumes a fixed per-coordinate distribution, and PCA
  coordinates have very unequal variances, so allocate bits per component.

## 4. Other ways through the current MLA scaling

1. **Absorbed/expanded switch at N\*** (Claim 3). Being implemented now for prefill.
2. **Grouped latent (GLA)** to cut absorbed decode cost by ~G at the same cache size and make decode tensor-core friendly.
3. **Sparse attention over the compressed codes.** Score all keys approximately straight from the 4-bit codes, then run
   exact attention on the top-k. Approximate scores can come from per-query lookup tables over the 16 codebook levels, as
   in product quantization, or from a low-rank subspace as in Loki; DeepSeek-V3.2's lightning indexer is the same idea.
   This turns O(n) per query into O(n·cheap + k·exact) and breaks the quadratic prefill. This llama.cpp fork already
   carries a lightning-indexer op to reuse.
4. **8-bit attention arithmetic.** SageAttention (ICLR 2025) runs QKᵀ in INT8 accurately. On Ampere/Orin INT8 tensor-core
   throughput is 2x FP16, so it would roughly halve the score half of attention. Our 4-bit codes map cleanly to int8 levels.
5. **Latent width rules for kernels.** r a multiple of 64, and ≤ 512 with a 64-dim RoPE part, lets existing MLA kernels
   (FlashMLA, llama.cpp's 576/512 path) run unmodified. Mercurius's 390–656 ranks forced custom kernels throughout.
6. **Cross-layer latent sharing** (Brandon et al., 2024, cross-layer attention): adjacent attention layers read one cache,
   up to 2x less KV memory. Cheap to test by distillation from the current model.

### Suggested order of experiments
1. Finish expanded prefill (in progress) and measure.
2. Sparse top-k over codes (no retraining; measures recall vs k on long-context tasks).
3. Online latent projection (GHA encoder, back-projected queries, delta-rule decoder as fallback) on Qwen's pre-RoPE keys,
   then on Mercurius's latent (§3).
4. Latent RoPE + grouped latent in the next distillation run (needs training; read Tucker Attention first).

## References
- Oja, "Simplified neuron model as a principal component analyzer", J. Math. Biol. 1982; Oja, "Neural networks, principal
  components, and subspaces", Int. J. Neural Systems 1989.
- Eckart & Young, "The approximation of one matrix by another of lower rank", Psychometrika 1936.
- Zadouri, Strauss, Dao, "Hardware-Efficient Attention for Fast Decoding" (GTA/GLA), arXiv 2505.21487, 2025.
- Sun et al., "ShadowKV", arXiv 2410.21465 (ICML 2025). Singhania et al., "Loki", arXiv 2406.02542 (NeurIPS 2024).
  Chang et al., "Palu", arXiv 2407.21118 (ICLR 2025).
- Sanger, "Optimal unsupervised learning in a single-layer linear feedforward neural network" (GHA), Neural Networks 1989.
- Zhang et al., "SageAttention", arXiv 2410.02367 (ICLR 2025).
- Klein et al., "Tucker Attention", arXiv 2603.30033 (2026). Mehta et al., arXiv 2506.09342 (2025).
- Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention", 2024.
