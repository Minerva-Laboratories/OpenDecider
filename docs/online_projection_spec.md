# Spec: Online evolving latent projection for the attention cache

Revision 2, 2026-10-07. Revision 1 (user draft) proposed an online compressive memory; this revision keeps its core
mechanism, the encoder learned online by an Oja-type rule, and uses it for one job: **a per-sequence, online-learned
replacement for MLA's static down-projection**. Softmax attention and the per-token cache stay as they are. The recurrent
memory-head variant is kept as a separate track in Appendix A, corrected after reproducing its toy.

## 0. Status
- Revision 1's toy reproduced (Appendix A). The memory head matches a *tuned* delta rule's denoised recall at 2–4× less
  state, but does not beat it, and stores no one-off associations. The draft's baseline (β = 0.3) was untuned.
- **Phase 1, first run (2026-10-07):** Qwen3.5-2B, 8,192 tokens of real prose, all 6 attention layers x 2 KV groups,
  exact causal attention; script `scripts/online_projection_phase1.py`, results `runs/online_projection_phase1*.json`.
  Mean relative error of the attention output (lower is better):

  | Keys, 192 position-free dims -> r | static PCA (code text) | online, per-chunk basis | online, remap (b) | online QW | oracle | no correction (d) |
  |---|---|---|---|---|---|---|
  | r = 32 | 0.397 | 0.294 | 0.320 | 0.284 | 0.228 | 0.521 |
  | r = 64 | 0.269 | 0.178 | 0.188 | 0.169 | 0.147 | 0.447 |
  | r = 96 | 0.189 | 0.118 | 0.123 | 0.112 | 0.110 | 0.412 |

  | Values, 256 dims -> r | static | online | oracle |
  |---|---|---|---|
  | r = 32 | 0.573 | 0.288 | 0.301 |
  | r = 64 | 0.485 | 0.211 | 0.217 |
  | r = 128 | 0.343 | 0.130 | 0.112 |

  Readings: the online basis beats a fixed one at every rank (keys about -34%, values about -55% at r = 64), and for values
  at r <= 64 it beats the whole-sequence oracle (adapting to the text helps). Query weighting helps slightly. Drift
  correction is required: (d) is far worse, while the cheap remap (b) is close to per-chunk bases.
  Caveats: the static basis was fitted on code (different domain) and is PCA, not a trained MLA projection; one text,
  one model; the errors are output errors, not perplexity (Phase 2).

- **Phase 2, first run (2026-10-08):** the compression inside Qwen3.5-2B's forward pass (all 6 attention layers), mean
  NLL increase per token vs the uncompressed model (prose / code; baseline NLL 1.535 / 1.8024; 4 x 8k prose and
  2 x 8k code tokens; 95% bootstrap CIs over 512-token blocks in `runs/online_projection_phase2.json`). Static basis now
  fitted on held-out MIXED code + prose. Script `scripts/online_projection_phase2.py`.

  | | static | online | qw | oracle (non-causal) |
  |---|---|---|---|---|
  | keys r = 32 | +0.458 / +0.711 | +0.360 / +0.561 | +0.365 / +0.555 | +0.275 / +0.449 |
  | keys r = 64 | +0.169 / +0.257 | +0.149 / +0.208 | +0.159 / +0.217 | +0.085 / +0.151 |
  | keys r = 96 | +0.070 / +0.114 | +0.068 / +0.085 | +0.075 / +0.094 | +0.041 / +0.069 |

  | | static | online | oracle |
  |---|---|---|---|
  | values r = 64 | +0.177 / +0.249 | +0.125 / +0.174 | +0.055 / +0.094 |
  | values r = 128 | +0.041 / +0.059 | +0.031 / +0.044 | +0.004 / +0.019 |
  | keys 64 + values 128 | +0.214 / +0.324 | +0.183 / +0.261 | +0.088 / +0.162 |

  Readings: online beats a static (MLA-like) basis by 10-30% of the NLL cost for code keys and for values (CIs
  separated); for prose keys the gain is small (r = 64: -12%, CIs overlap; r = 96: a tie). Query weighting does not
  help in the model (slightly worse than plain online). The oracle shows much more headroom (values r = 128 nearly
  free: +0.004 prose). In the switch test the forgetting rate barely matters (gamma 0.8-0.97 within noise), so the
  gap to the oracle looks like the cost of a causal estimate (basis from past chunks only), not of forgetting. Topic switch (code <-> prose at 4k): online keys lose ~0.03 nats more than static right after a code -> prose
  switch and gain ~0.03 after prose -> code, CIs overlapping: re-tracking is fast, no lasting damage.
  Next: a basis fitted on the whole PROMPT (legal at inference: the prompt is known before generation) plus online
  updates during generation, measured on the continuation only; lazy re-encoding (option c) of recent chunks.

## 1. Idea
MLA caches `c_t = W_c x_t` with one learned, fixed `W_c` for every input. Within one sequence, keys occupy a much smaller
subspace than across sequences: ShadowKV measured pre-RoPE keys as extremely low-rank within a sequence, with subspaces
shared by a sequence and its continuation but not across sequences. So a projection learned **for the current sequence,
online, while it is read** should reach a lower rank at the same accuracy than a global one, and it needs no calibration.

Per layer and KV head:
- `E_t` (r×d): the current encoder, updated online.
- Cache: per token, the code `c_j = E_{b(j)} k̃_j`, written with the encoder of its chunk `b(j)`.
- Read: scores `q̃_t^T E_{b(j)}^T c_j = (E_{b(j)} q̃_t)^T c_j`, i.e. the query is back-projected once per chunk basis. No
  decoder is needed for scores.
- Values: the same with their own encoder `F_t`; `o_t = Σ_j p_tj F_{b(j)}^T c^v_j`, so the up-projection is applied once
  per chunk group after the weighted sum.

`k̃`, `q̃` are the position-free parts. Qwen3.5 rotates only the first 64 of 256 head dims (partial RoPE), so its other
192 dims are compressed and the 64 RoPE dims are cached exactly. That is the decoupled-RoPE layout of MLA, already
present in the model. For models with full RoPE, compress pre-RoPE keys and apply RoPE in the latent (see
`docs/mla_scaling_ideas.md` §1).

## 2. Objective
The cache only has to preserve scores. For encoder E and decoder D (both rank r), queries and keys with covariances
`Σ_q`, `Σ_k`:
`L = E[(q^T k − q^T D E k)^2] = ‖Σ_q^{1/2} (I − DE) Σ_k^{1/2}‖_F^2`,
minimized by the SVD of `Σ_q^{1/2} Σ_k^{1/2}` (two-sided Eckart–Young, proof in `docs/mla_scaling_ideas.md` §3.1).
- With `Σ_q ∝ I`: key PCA (variant **K**).
- In general: query-weighted PCA (variant **QW**), i.e. PCA of keys whitened by a running query covariance.
- Values only pass through softmax weights, so their target is plain PCA of values (variant **V**).

## 3. Online update (chunked)
Per chunk of C tokens (C = 64 to start):
1. Codes for the chunk with the current basis: `c = E k̃` (one matmul). Cache them with chunk id b.
2. Update the running second moment `M ← γ M + (1 − γ) K̃_b^T K̃_b / C` (K, or whitened for QW). `γ` sets how fast the basis
   forgets; it can be a learned or data-dependent gate later.
3. Subspace iteration: `E ← orth_rows(E (I + η M))`, one QR of an r×d matrix. Unlike Oja's per-token subspace rule, QR
   iteration converges to *ordered* eigenvectors (as Sanger's GHA does), so once the subspace is learned the basis stops
   rotating, and old codes stay consistent with the current basis.
4. Warm-up: the first `W` tokens (e.g. one chunk) are cached exactly (or re-encoded once the first basis exists).

Reading old chunks needs their basis. Options, in order of cost:
- **(a) Per-chunk bases.** Keep `E_b` for every chunk: exact, but r×d per chunk per head.
- **(b) Re-map.** Keep only the current E, plus for each chunk the r×r map `Q_b = E_t E_b^T` (updated by multiplying in
  each step's rotation). Codes are read as `Q_b c`. Exact if the subspace did not change, otherwise the error is the
  part of the old code outside the current subspace.
- **(c) Lazy re-encoding.** When `‖E_t − E_b‖` exceeds a threshold, rewrite chunk b's codes as `Q_b c_b` and drop `Q_b`.
- **(d) Single basis, no correction.** Only for measuring drift error.

## 4. Costs
Per layer, per KV head, d = 192 position-free dims (Qwen3.5), rank r:
- Cache per token: r (+ 64 RoPE dims) instead of 192 + 64 for keys, and r instead of 256 for values.
- Per chunk: one d×d moment update (C·d² MACs), one r×d·d product and a QR: negligible next to attention.
- Per query: one back-projection per *distinct* basis; with (b)/(c) there is one. Per (query, key): r MACs instead of d.

## 5. Relation to existing work
- ShadowKV, Loki, Palu compute low-rank bases offline or once at prefill (batch SVD/PCA).
- Here the basis is streaming: it adapts during generation and to topic shifts, with a score-preserving objective (QW).
- MLA learns a global basis in training. This is per-sequence and needs no training for a first test.

## 6. Plan
**Phase 0, synthetic.** Toy with (i) a topic switch mid-stream, (ii) key-only high-variance nuisance directions,
(iii) heavy-tailed noise. Measure attention-output error and re-tracking speed for K vs QW, and reading options (a)–(d).

**Phase 1, real activations, offline (first).** Qwen3.5 attention layers on long real text: capture position-free
queries/keys and values per KV head. Measure attention fidelity, i.e. relative output error and the overlap of top-k
attended tokens, for:
- static global PCA (fit on other text: the MLA-like baseline);
- per-sequence batch PCA, non-causal (the oracle upper bound);
- online K and QW with reading options (b) and (d),

at r ∈ {16, 32, 64, 96} out of 192 for keys, and r ∈ {32, 64, 128} out of 256 for values.

**Phase 2, in the model.** Hook the compressed attention into the forward pass: perplexity and downstream accuracy
(OpenDecider's benchmarks) vs r.

**Phase 3, kernel.** Fold the code read into the decode attention kernel: scores from codes with a back-projected query,
the same shape as MLA absorption.

**Phase 4 (optional).** Train with the online projection in the loop (differentiable QR), or apply it to Mercurius's latent
to compress it further per sequence.

## 7. Risks
- The basis needs enough tokens to settle. Short contexts gain little, and the warm-up chunk stays uncompressed.
- Long, heterogeneous contexts may need a larger r or faster forgetting. That trade-off is what Phase 0/1 measure.
- 4-bit quantization of codes: PCA coordinates have very unequal variances, so allocate bits per component.

---

## Appendix A: recurrent compressive memory head (revision 1), corrected

Kept as a separate track: a recurrent memory head next to GDN, not an attention variant.

Reproduced toy (rank-8 latent, d = 128, T = 4000, noise 1.0, 500 held-out probe keys; target = clean v):

| | State (floats) | Denoised recall | Episodic recall |
|---|---|---|---|
| Memory head, spec settings | 10,240 (8,192 effective) | 0.951 | 0.01 |
| Delta rule β = 0.3 (draft's baseline) | 16,384 | 0.870 | 0.62 |
| Delta rule β = 0.03 | 16,384 | 0.956 | 0.13 |
| Delta rule β = 0.01 | 16,384 | 0.960 | 0.03 |
| Batch reduced-rank regression (ceiling) | | 0.951 | |

At d = 256: memory head 0.968 at 20,480 floats vs tuned delta 0.977 at 65,536.

Corrections to the draft:
1. **Claim.** "Same denoised recall as a tuned delta rule at 2–4× less state, no episodic memory", not "better than
   delta". Compare against tuned, matched-state baselines (β sweep; GDN with a smaller head; a rank-r factored delta rule).
2. **State.** The key half of D is never read (readout uses `D[d:]`, Oja ignores D), so drop it: state is 4rd, not 5rd.
   `D_v R` is a rank-r linear map learned online, so frame it as a low-rank delta rule with a denoised latent.
3. **Subspace.** Oja on `[k; v]` keeps joint variance. Retrieval needs the predictive subspace (reduced-rank regression /
   CCA); they coincide only when one latent drives both. Test with key-only nuisance directions.
4. **Decay.** The additive `−βλR` differs from GDN's multiplicative α by O(β²λ). Use GDN's form in the reference so the
   chunked/recurrent equivalence test can be exact.
5. **Normalization.** §3 updated D with unnormalized h and §4 with normalized h; the readout query must match. Use NLMS
   (`β_t = η/(‖h_t‖² + ε)`, unnormalized h) in both.
6. **R step.** The gradient uses `k̂` while the input is `k̂/√2`, so the effective step is √2·β; stability needs β < 2√2.
7. **Value scale.** Normalizing v makes outputs direction-only; the head needs a learned output norm/scale.
8. **Protocol.** State whether probes are held-out keys or the current token. The table above uses held-out probes.
