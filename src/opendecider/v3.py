"""V3: dense late fusion (user design, docs/v3_dense_design.md).

Frozen backbone -> dense token features (no backbone gradients, cacheable by text):
    context  C = [state tokens ; question tokens]   (part-specific input projections; RoPE, continuous positions)
    optional context layers: joint self-attention over C per question (question-aware context)
    options  O = all option tokens of a question     (option projection; RoPE positions restart in each option;
                                                        optional slot tags)
Every block is pre-norm residual, x = x + f(ScaleNorm(x)); standard (PyTorch default) initialisation.
Trainable stack, each layer:
    1. option self-attention over ALL option tokens (bidirectional, iTransformer-like "between options");
       a learned per-head same-option / other-option bias tells a token which tokens form its own option
       (membership without any per-slot embedding -> stays permutation-equivariant)
    2. cross-attention option tokens -> C   (optionally also C -> options: `context_update`)
    3. xATGLU feed-forward, token-wise (never mixes options)
Readout: one CLS query per option attends only to that option's tokens (final layer, or a learned mix of
all layers' outputs) -> ScaleNorm -> linear -> one logit per option -> softmax across options.

References (checked): ScaleNorm = g * x/||x||, one scalar g init sqrt(d) (Nguyen & Salazar 2019,
arXiv:1910.05895). xATGLU (first order) = g_a(x) * y with g_a(x) = ((atan(x)+pi/2)/pi)(1+2a) - a, one
trainable scalar a per MLP block init 0, MLP ratio 8/3 (Huang 2024, arXiv:2405.20768).
"""
from __future__ import annotations

import contextlib
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from .backbone import PackedCache, StateMemory
from .batching import Packed, Question
from .formatting import answer_option_text, answer_row_prefix, option_text, question_answer_text, question_text
from .heads import NEG
from .models import DecisionModel, DecisionOutput, Memory, VARIANTS
from .cache import TokenCache, encode_missing, segment_layout
from .prefix_attn import shared_prefix_attention
from .quant import kv_mode, materialize, maybe_quantize


class ScaleNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(math.sqrt(d)))
        self.eps = eps

    def forward(self, x):
        return self.g * x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)


class XATGLU(nn.Module):
    """First-order expanded ArcTan GLU: W_o( g_a(W_g x) * W_v x )."""

    def __init__(self, d: int, ratio: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        h = int(round(d * ratio / 8)) * 8
        self.wg = nn.Linear(d, h)
        self.wv = nn.Linear(d, h)
        self.wo = nn.Linear(h, d)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate = (torch.atan(self.wg(x)) + math.pi / 2) / math.pi * (1 + 2 * self.alpha) - self.alpha
        return self.wo(self.drop(gate * self.wv(x)))


def rope(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    """Rotary position embedding (Su et al. 2021). x: (N,h,L,dh); pos: (N,L) long or None (= position 0)."""
    if pos is None:
        return x
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.to(x.device).float()[:, None, :, None] * inv                   # (N,1,L,dh/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class MHA(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x, ctx, bias, q_pos=None, k_pos=None):
        """x: (N,P,d) queries; ctx: (N,T,d); bias: (N,h|1,P,T) additive mask; *_pos: RoPE positions or None."""
        N, P, d = x.shape
        T = ctx.shape[1]
        q = rope(self.q(x).view(N, P, self.h, self.dh).transpose(1, 2), q_pos)
        k = rope(self.k(ctx).view(N, T, self.h, self.dh).transpose(1, 2), k_pos)
        v = self.v(ctx).view(N, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        return self.o(o.transpose(1, 2).reshape(N, P, d))


class V3Layer(nn.Module):
    def __init__(self, d, n_heads, dropout, context_update: bool, cross: str = "full"):
        super().__init__()
        self.use_cross = cross != "none"
        self.n1, self.self_attn = ScaleNorm(d), MHA(d, n_heads, dropout)
        self.n2, self.n2c, self.cross = ScaleNorm(d), ScaleNorm(d), MHA(d, n_heads, dropout)
        self.n3, self.ff = ScaleNorm(d), XATGLU(d, dropout=dropout)
        self.same_bias = nn.Parameter(torch.zeros(n_heads))     # per-head bias for same-option keys
        self.context_update = context_update
        if context_update:
            self.nc1, self.nc2, self.ctx_attn = ScaleNorm(d), ScaleNorm(d), MHA(d, n_heads, dropout)
            self.nc3, self.ctx_ff = ScaleNorm(d), XATGLU(d, dropout=dropout)

    def forward(self, x, c, same, x_pad, c_pad, opos=None, cpos=None):
        """x: option tokens (N,P,d); c: context (N,T,d); same: (N,P,P) bool same-option; *_pad: True = pad."""
        h = self.self_attn.h
        b_self = self.same_bias.view(1, h, 1, 1) * same[:, None].float()
        b_self = b_self.masked_fill(x_pad[:, None, None, :], NEG)
        hx = self.n1(x)
        x = x + self.self_attn(hx, hx, b_self, opos, opos)                  # RoPE within each option
        if self.use_cross:
            b_cross = torch.zeros(c_pad.shape, device=x.device).masked_fill(c_pad, NEG)[:, None, None, :]
            x = x + self.cross(self.n2(x), self.n2c(c), b_cross, None, cpos)   # keys rotated by context position
        x = x + self.ff(self.n3(x))
        if self.context_update:
            b_c = torch.zeros(x_pad.shape, device=x.device).masked_fill(x_pad, NEG)[:, None, None, :]
            c = c + self.ctx_attn(self.nc1(c), self.nc2(x), b_c, cpos, opos)
            c = c + self.ctx_ff(self.nc3(c))
        return x, c


class DepthAttn(nn.Module):
    """Per-token cross-attention over DEPTH: a token's hidden states from several layers -> one vector.
    Query = learned bias + projection of the token's last listed layer; keys = projections of every (ScaleNormed) layer
    state + a learned per-layer embedding; values = the ScaleNormed layer states. Standard (PyTorch default) init."""

    def __init__(self, d: int, n_layers: int, d_k: int = 256):
        super().__init__()
        self.norm = ScaleNorm(d)
        self.q = nn.Linear(d, d_k)
        self.k = nn.Linear(d, d_k)
        self.layer_emb = nn.Embedding(n_layers, d_k)
        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (m, L, d)
        xn = self.norm(x.float())
        q = self.q(xn[:, -1])                                     # (m, d_k)
        k = self.k(xn) + self.layer_emb.weight[None]              # (m, L, d_k)
        a = torch.softmax(torch.einsum("mk,mlk->ml", q, k).float() * self.scale, -1)
        return torch.einsum("ml,mld->md", a, xn.float()).float()      # fp32 out even under autocast


class ContextLayer(nn.Module):
    """Joint self-attention over [state ; question] (per question), pre-norm residual + xATGLU.
    Makes the context question-aware (BiDAF / instruction-aware Q-Former idea) before options read it.
    Only the small stack does per-question work; the backbone state encoding stays shared."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.n1, self.attn = ScaleNorm(d), MHA(d, n_heads, dropout)
        self.n2, self.ff = ScaleNorm(d), XATGLU(d, dropout=dropout)

    def forward(self, c, c_pad, cpos):
        b = torch.zeros(c_pad.shape, device=c.device).masked_fill(c_pad, NEG)[:, None, None, :]
        h = self.n1(c)
        c = c + self.attn(h, h, b, cpos, cpos)
        return c + self.ff(self.n2(c))


class V3(DecisionModel):
    pack_mode = "separate"

    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        self.conditioned = cfg.v3_features in ("conditioned", "question_conditioned", "branched")
        if cfg.v3_vera_layers:
            assert self.conditioned, "VeRA needs conditioned/branched features"
            from .vera import VeRA
            n = backbone.num_layers if cfg.v3_vera_scope == "all" else cfg.v3_vera_layers
            self.vera = VeRA(backbone, n_layers=n, rank=cfg.v3_vera_rank)
        if cfg.v3_layer_combine == "attn" and len(backbone.cfg.feature_layers) > 1:
            self.depth_attn = DepthAttn(self.d, len(backbone.cfg.feature_layers))
        # width of the per-token features entering the trunk: the backbone's, or a stitched target space
        self.df = cfg.v3_stitch_dim if cfg.v3_layer_combine == "stitch" else self.d
        if cfg.v3_layer_combine == "stitch":
            # model stitching: a trunk trained on backbone A reads backbone B through one linear map, fitted by
            # ridge regression on paired token features (same tokenizer) - see scripts/stitch_backbone.py
            self.stitch = nn.Linear(len(backbone.cfg.feature_layers) * self.d, self.df)
        self.branched = cfg.v3_features == "branched"
        self.q_only = cfg.v3_features == "question_conditioned"
        # the trunk never reads state tokens when options cross-attend to the question only: skip state features
        self.state_free = self.conditioned and cfg.v3_cross == "question"
        if self.conditioned:
            # micro-batch sizing: one row per question, or one row per option when branched
            self.pack_mode = "independent" if self.branched else "joint"
        d, dm = self.df, cfg.d_model
        self.in_norm = ScaleNorm(d)                  # frozen features are large (L2 ~120): normalise first
        # part identity via separate input projections (no additive segment vectors); PyTorch default inits
        self.proj_state, self.proj_question, self.proj_option = nn.Linear(d, dm), nn.Linear(d, dm), nn.Linear(d, dm)
        self.context_layers = nn.ModuleList(ContextLayer(dm, cfg.n_heads, cfg.dropout)
                                            for _ in range(cfg.v3_context_layers))
        self.layers = nn.ModuleList(V3Layer(dm, cfg.n_heads, cfg.dropout, cfg.v3_context_update, cfg.v3_cross)
                                    for _ in range(cfg.decision_layers))
        self.readout_mix = nn.Parameter(torch.zeros(cfg.decision_layers)) if cfg.v3_readout == "mix" else None
        # CLS readout = one pre-norm residual block per option (attention restricted to the option's own
        # tokens, keys RoPE-rotated by position in the option, then xATGLU); final norm; scoring head.
        self.cls = nn.Embedding(1, dm)
        self.cls_nq, self.cls_nkv, self.cls_attn = ScaleNorm(dm), ScaleNorm(dm), MHA(dm, cfg.n_heads, 0.0)
        self.cls_nf, self.cls_ff = ScaleNorm(dm), XATGLU(dm, dropout=cfg.dropout)
        self.final_norm = ScaleNorm(dm)
        if cfg.v3_lm_feature:
            assert cfg.v3_features == "branched", "v3_lm_feature is implemented for branched rows"
            self.lp_proj = nn.Linear(1, dm)                        # token-level LM log-prob feature (no pooling)
            self.lm_skip = nn.Parameter(torch.zeros(()))           # gated skip of the option log-likelihood, init 0
        if cfg.v3_none:
            # `none` sink: one option token whose feature is LEARNED (no text, no LLM row, no slot/position), so the
            # trunk scores "no listed option is supported" like any other option; always the LAST logit column
            self.none_x = nn.Parameter(torch.randn(self.df) * 0.02)
            self.none_lp = nn.Parameter(torch.tensor(-3.0))       # stands in for the option's LM log-prob feature
        self.score = nn.Linear(dm, 1)
        self.rel = nn.Linear(dm, 1)

    # ------------------------------------------------------------------ memory (state)
    def prepare_memory(self, sm: StateMemory, cache: bool, noise_std: float = 0.0) -> Memory:
        f = self._noise(self._mix(sm.get(torch.float32)), noise_std).float()
        # the dense state tokens ARE the memory; store int8 when caching (inference)
        keep = maybe_quantize(f, cache and kv_mode(self.backbone.cfg))
        m = Memory(keep, sm.mask, [], n_tokens=sm.n_tokens, kv_mask=sm.mask)
        m.lens = getattr(sm, "lens", None)
        m.raw_feats = sm.feats if cache else None
        return m

    def memory_from_ids(self, ids, mask, noise_std: float = 0.0):
        if not self.conditioned:
            return super().memory_from_ids(ids, mask, noise_std)
        lens = mask.sum(1).cpu()
        m = Memory(None, mask, n_tokens=int(lens.sum()))
        m.extra_prefix = [ids[i, : int(lens[i])].tolist() for i in range(ids.shape[0])]
        return m

    def encode_states(self, state_texts, noise_std: float = 0.0):
        if not self.conditioned:
            return super().encode_states(state_texts, noise_std)
        seqs = self.backbone.tokenize(state_texts)
        m = Memory(None, torch.zeros(0), n_tokens=sum(map(len, seqs)))
        m.extra_prefix = seqs
        return m

    @contextlib.contextmanager
    def _row_pass(self):
        """Question/option rows continuing from the state cache. With VeRA: adapters ON and grad enabled when
        training (only the adapted top layers build a graph; nothing below requires grad). Without VeRA: no_grad."""
        vera = getattr(self, "vera", None)
        if vera is None:
            with torch.no_grad():
                yield
            return
        vera.active = True
        try:
            with torch.set_grad_enabled(self.training and torch.is_grad_enabled()):
                yield
        finally:
            vera.active = False

    def _rows_forward(self, ids, mask, rows, pc, smask, position_ids=None):
        """One chunk of rows continuing from their states' caches -> wide features. With `v3_row_checkpoint` and a
        graph being built, the chunk is recomputed in backward: the function unpacks a FRESH cache and switches VeRA
        on itself, so the recompute sees the same inputs and adapters as the forward (peak = state graph + 1 chunk)."""
        bb, vera = self.backbone, getattr(self, "vera", None)

        shared = self.cfg.v3_shared_prefix

        def run(ids, mask):
            prev = vera.active if vera is not None else None
            if vera is not None:
                vera.active = True
            try:
                with shared_prefix_attention(bb, pc, rows, smask) if shared else contextlib.nullcontext():
                    f, _ = bb(ids, mask, layers=bb.cfg.feature_layers,
                              past_key_values=pc.unpack_rows(rows, attn_placeholder=shared),
                              use_cache=True, past_mask=smask.index_select(0, rows), position_ids=position_ids)
                return self._wide(f)
            finally:
                if vera is not None:
                    vera.active = prev

        if self.cfg.v3_row_checkpoint and torch.is_grad_enabled() and self.training:
            return torch.utils.checkpoint.checkpoint(run, ids, mask, use_reentrant=False)
        return run(ids, mask)

    def _state_prefix(self, mem, grad: bool | None = None, quantize: bool | None = None):
        """Batched left-padded state pass. With VeRA scope 'all' the adapters are ON here too and, when training,
        the pass keeps its graph (the cache is left unquantised) so the state pass is trained end to end.
        grad/quantize override the defaults (the trainer re-runs this pass for a single deferred backward)."""
        vera = getattr(self, "vera", None)
        full = vera is not None and self.cfg.v3_vera_scope == "all"
        if grad is None:
            grad = full and self.training and torch.is_grad_enabled()
        if full:
            vera.active = True
        try:
            out = self.backbone.prefix_cache_batch(mem.extra_prefix, return_hidden=not self.state_free, grad=grad,
                                                   quantize=quantize)
            return (*out, None) if self.state_free else out
        finally:
            if full:
                vera.active = False

    def _wide(self, f: torch.Tensor) -> torch.Tensor:
        """(L,B,T,d) multi-layer stack -> (B,T,L*d) so token indexing is layer-agnostic; (B,T,d) unchanged."""
        return f.float() if f.dim() == 3 else f.float().permute(1, 2, 0, 3).flatten(2)

    def _mix_wide(self, x: torch.Tensor) -> torch.Tensor:
        """Learned softmax mix over layers, applied OUTSIDE the frozen no-grad pass so it trains."""
        n = len(self.backbone.cfg.feature_layers)
        if self.cfg.v3_layer_combine == "stitch":
            xs = x.float().view(x.shape[0], n, self.d)
            xs = xs / xs.norm(dim=-1, keepdim=True).clamp_min(1e-6)       # unit-normalise each layer
            return self.stitch(xs.flatten(1)).float()
        if n == 1 or x.shape[-1] == self.d:          # single layer, or embedding-space options (already width d)
            return x
        if self.cfg.v3_layer_combine == "attn":
            return self.depth_attn(x.float().view(x.shape[0], n, self.d))
        w = torch.softmax(self.layer_mix.float(), 0)
        # fp32 explicitly: under CUDA autocast einsum would return bf16 and break the fp32 index_put downstream
        return torch.einsum("l,mld->md", w, x.float().view(x.shape[0], n, self.d)).float()

    def _conditioned_feats(self, questions, mem):
        s_flat, s_lens, q_flat, q_lens, o_flat, o_lens = self._conditioned_feats_raw(questions, mem)
        s_mix = None if s_flat is None else self._mix_wide(s_flat)
        return s_mix, s_lens, self._mix_wide(q_flat), q_lens, self._mix_wide(o_flat), o_lens

    def _conditioned_feats_raw(self, questions, mem):
        """Frozen LLM pass, no grad: states once (left-padded batched prefix), then ONE row per question
        [question ; opt_1 ; ... ; opt_K] continuing from its state's cache. Returns flat dense features and
        CPU lengths laid out exactly like the isolated path: (s_flat, s_lens, q_flat, q_lens, o_flat, o_lens)."""
        bb, dev = self.backbone, self.backbone.device
        if getattr(mem, "prefix_batch", None) is None:
            mem.prefix_batch = self._state_prefix(mem)
        pc, smask, sh = mem.prefix_batch
        if self.q_only:
            return self._question_conditioned_feats(questions, mem, pc, smask, sh)
        if self.branched:
            return self._branched_feats(questions, mem, pc, smask, sh)
        n_opt = [len(q.options) for q in questions]
        texts = []
        for q in questions:
            texts.append(question_text(q.prompt))
            texts.extend(option_text(o) for o in q.options)
        toks = bb.tokenize(texts)
        lens = torch.tensor([len(t) for t in toks])
        n = torch.tensor(n_opt)
        first = torch.cumsum(n + 1, 0) - (n + 1)
        is_q = torch.zeros(len(toks), dtype=torch.bool); is_q[first] = True
        rows = []
        for i, k in zip(first.tolist(), n_opt):                  # concatenate token lists (metadata only)
            r = []
            for t in toks[i:i + 1 + k]:
                r.extend(t)
            rows.append(r)
        ids, mask = bb.pad(rows)
        sidx = torch.tensor([q.state_idx for q in questions])
        with self._row_pass():
            f = self._rows_forward(ids, mask, sidx.to(dev), pc, smask)
            # every piece's tokens in row order: row = question index, col = running offset in that row
            owner = torch.repeat_interleave(torch.arange(len(questions)), n + 1)
            r, c, seg, _, _ = segment_layout(lens, owner, len(questions))
            flat = f[(r.to(dev), c.to(dev))]
            tok_is_q = is_q[seg].to(dev)
            q_flat, o_flat = flat[tok_is_q], flat[~tok_is_q]
            s_flat = None if sh is None else self._wide(sh)[smask]
        s_lens = smask.sum(1).cpu()
        return s_flat, s_lens, q_flat, lens[is_q], o_flat, lens[~is_q]

    def _branched_feats(self, questions, mem, pc, smask, sh):
        """One LLM row per OPTION: [question ; option_k] continuing from its state's cache. Every option is read
        in context of the state and question, but never sees the other options (no causal order effects between
        options; comparison is left to the trunk). Rows are built by scatter from flat token-id tensors and run in
        chunks of `v3_branch_chunk` rows to bound the per-row GDN state copies."""
        if self.cfg.v3_question_cache and self.cfg.v3_row_format == "answer" and \
                sum(len(q.options) for q in questions) >= self.cfg.v3_qcache_min_rows:
            return self._branched_feats_qcache(questions, mem, pc, smask, sh)
        bb, dev = self.backbone, self.backbone.device
        N = len(questions)
        if self.cfg.v3_row_format == "answer":
            q_tok = bb.tokenize([answer_row_prefix(q.prompt, q.options, self.cfg.v3_list_cap) for q in questions])
            o_tok = bb.tokenize([answer_option_text(o) for q in questions for o in q.options])
        else:
            q_tok = bb.tokenize([question_text(q.prompt) for q in questions])
            o_tok = bb.tokenize([option_text(o) for q in questions for o in q.options])
        lq = torch.tensor([len(t) for t in q_tok]); lo = torch.tensor([len(t) for t in o_tok])
        q_ids = torch.tensor([i for t in q_tok for i in t]); o_ids = torch.tensor([i for t in o_tok for i in t])
        n_opt = torch.tensor([len(q.options) for q in questions])
        row_q = torch.repeat_interleave(torch.arange(N), n_opt)                    # question of each row
        M = row_q.numel()
        lqr = lq[row_q]
        L = int((lqr + lo).max())
        ids = torch.full((M, L), bb.pad_id, dtype=torch.long)
        q_start = torch.cumsum(lq, 0) - lq
        r, c, _, pos, _ = segment_layout(lqr, torch.arange(M), M)                 # question part of every row
        ids[r, c] = q_ids[q_start[row_q][r] + pos]
        o_start = torch.cumsum(lo, 0) - lo
        r2, c2, _, pos2, _ = segment_layout(lo, torch.arange(M), M)               # option part
        ids[r2, lqr[r2] + c2] = o_ids[o_start[r2] + pos2]
        mask = torch.arange(L)[None, :] < (lqr + lo)[:, None]
        sidx = torch.tensor([q.state_idx for q in questions])[row_q]
        nL = len(bb.cfg.feature_layers)
        F = torch.empty(M, L, self.d * nL, device=dev, dtype=torch.bfloat16 if nL > 1 else torch.float32)
        B = self.cfg.v3_branch_chunk
        with self._row_pass():
            for a in range(0, M, B):                                              # bounded chunks (memory)
                sl = slice(a, min(a + B, M))
                F[sl] = self._rows_forward(ids[sl].to(dev), mask[sl].to(dev), sidx[sl].to(dev), pc, smask)
            first = torch.cumsum(n_opt, 0) - n_opt                                  # question tokens: from 1st row
            qr, qc, _, _, _ = segment_layout(lq, torch.arange(N), N)
            q_flat = F[(first[qr].to(dev), qc.to(dev))]
            oc = (lqr[r2] + c2).to(dev)
            o_flat = F[(r2.to(dev), oc)]
            if self.cfg.v3_lm_feature:
                # LM log-prob of every option token given state+question (+ earlier option tokens): the frozen LM's own
                # zero-shot judgement, via the tied LM head on the FINAL-layer state of the preceding position.
                assert bb.cfg.feature_layers[-1] == "final", "v3_lm_feature needs 'final' as the last feature layer"
                h_prev = F[(r2.to(dev), oc - 1)][:, -self.d:]
                self._lm_lp = self._token_logprob(h_prev, ids[(r2, lqr[r2] + c2)].to(dev))
            s_flat = None if sh is None else self._wide(sh)[smask]
        return s_flat, smask.sum(1).cpu(), q_flat, lq, o_flat, lo

    def _branched_feats_qcache(self, questions, mem, pc, smask, sh):
        """Two-level shared prefix: state cache -> ONE pass per question over [question + canonical candidate listing
        + "Answer:"] -> packed question cache -> every option row carries only its own answer tokens. Same features
        as repeating the question in each row (tested), but the listing is paid once per question, so long listings
        (77- or 255-way) are affordable. A question's cache must end at its last real token: with
        `v3_question_batch` all questions share ONE right-padded pass whose pads leave the GDN state untouched
        (Backbone.padded_continuation; pad keys masked, option rows positioned after their own question), else
        questions are grouped by exact token length, one unpadded pass per group. Option rows then run sorted by
        length in chunks of `v3_branch_chunk`, each chunk as wide as its longest option."""
        bb, dev = self.backbone, self.backbone.device
        N = len(questions)
        q_tok = bb.tokenize([answer_row_prefix(q.prompt, q.options, self.cfg.v3_list_cap) for q in questions])
        o_tok = bb.tokenize([answer_option_text(o) for q in questions for o in q.options])
        lq = torch.tensor([len(t) for t in q_tok]); lo = torch.tensor([len(t) for t in o_tok])
        n_opt = torch.tensor([len(q.options) for q in questions])
        row_q = torch.repeat_interleave(torch.arange(N), n_opt)                    # question of each option row
        M = row_q.numel()
        o_ids = torch.tensor([i for t in o_tok for i in t])
        # >= 2 columns: a 1-token continuation of a cache takes the GDN decode kernel, which has no backward
        # (a trailing right pad never affects the real positions)
        Lo = max(2, int(lo.max()))
        O = torch.full((M, Lo), bb.pad_id, dtype=torch.long)
        r2, c2, _, pos2, _ = segment_layout(lo, torch.arange(M), M)
        O[r2, c2] = o_ids[(torch.cumsum(lo, 0) - lo)[r2] + pos2]
        omask = torch.arange(Lo)[None, :] < lo[:, None]
        Lq = int(lq.max())
        Qids = torch.full((N, Lq), bb.pad_id, dtype=torch.long)
        rq, cq, _, posq, _ = segment_layout(lq, torch.arange(N), N)
        Qids[rq, cq] = torch.tensor([i for t in q_tok for i in t])[(torch.cumsum(lq, 0) - lq)[rq] + posq]
        qmask = torch.arange(Lq)[None, :] < lq[:, None]
        sidx = torch.tensor([q.state_idx for q in questions])
        opt_start = torch.cumsum(n_opt, 0) - n_opt
        nL = len(bb.cfg.feature_layers)
        fdt = torch.bfloat16 if nL > 1 else torch.float32
        Q = torch.zeros(N, Lq, self.d * nL, device=dev, dtype=fdt)
        F = torch.zeros(M, Lo, self.d * nL, device=dev, dtype=fdt)
        B = self.cfg.v3_branch_chunk
        grad = self.training and torch.is_grad_enabled() and getattr(self, "vera", None) is not None
        groups = [torch.arange(N)] if self.cfg.v3_question_batch else \
            [(lq == Lg).nonzero().squeeze(1) for Lg in torch.unique(lq).tolist()]
        with self._row_pass():
            for gi in groups:
                sg = sidx[gi].to(dev)
                Lg = int(lq[gi].max())
                gmask = qmask[gi, :Lg].to(dev)
                cache = pc.unpack_rows(sg)
                padded = bool((lq[gi] < Lg).any())
                with bb.padded_continuation(gmask, cache) if padded else contextlib.nullcontext():
                    fq, cache = bb(Qids[gi, :Lg].to(dev), gmask, layers=bb.cfg.feature_layers, past_key_values=cache,
                                   use_cache=True, past_mask=smask.index_select(0, sg))
                Q[gi.to(dev), :Lg] = self._wide(fq).to(fdt)
                pcq = PackedCache.pack(cache, kv_int8=(not grad) and kv_mode(bb.cfg),
                                       state_int8=bb.cfg.quantize_linear_state and not grad)
                pmq = torch.cat([smask.index_select(0, sg), gmask], 1)
                ng = n_opt[gi]
                local = torch.repeat_interleave(torch.arange(len(gi)), ng)                 # question-in-group per row
                rows = opt_start[gi][local] + torch.arange(int(ng.sum())) - (torch.cumsum(ng, 0) - ng)[local]
                # rows sorted by option length, each chunk only as wide as its longest option (no pad compute)
                order = torch.argsort(lo[rows], stable=True)
                rows, local = rows[order], local[order]
                # option tokens continue right after their OWN question's last token (not after the pads)
                pos0 = (smask.shape[1] + lq[gi])[local] if padded else None
                for a in range(0, rows.numel(), B):                                         # bounded chunks
                    sl = slice(a, a + B)
                    w = max(2, int(lo[rows[sl]].max()))
                    pos = None if pos0 is None else (pos0[sl, None] + torch.arange(w)).to(dev)
                    F[rows[sl].to(dev), :w] = self._rows_forward(O[rows[sl], :w].to(dev), omask[rows[sl], :w].to(dev),
                                                                 local[sl].to(dev), pcq, pmq, pos).to(fdt)
            qm, om = qmask.to(dev), omask.to(dev)
            q_flat, o_flat = Q[qm], F[om]
            if self.cfg.v3_lm_feature:
                # previous position of an option's FIRST token is its question's last token (from the question pass)
                assert bb.cfg.feature_layers[-1] == "final", "v3_lm_feature needs 'final' as the last feature layer"
                q_last = Q[torch.arange(N, device=dev), (lq - 1).to(dev), -self.d:]
                h_prev = torch.cat([q_last[row_q.to(dev)][:, None], F[:, :-1, -self.d:]], 1)[om]
                self._lm_lp = self._token_logprob(h_prev, O.to(dev)[om])
            s_flat = None if sh is None else self._wide(sh)[smask]
        return s_flat, smask.sum(1).cpu(), q_flat, lq, o_flat, lo

    def _token_logprob(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """Log-prob of tok under the frozen LM head (tied embeddings or the untied head). Fixed feature."""
        return self.backbone.token_logprob(h, tok)

    def _question_conditioned_feats(self, questions, mem, pc, smask, sh):
        """Rows are ONLY [question ... Answer:] continuing from the state cache (binding happens in the LLM);
        options are encoded alone (cached by text, no causal order effects between options)."""
        bb, dev = self.backbone, self.backbone.device
        toks = bb.tokenize([question_answer_text(q.prompt) for q in questions])
        ids, mask = bb.pad(toks)
        sidx = torch.tensor([q.state_idx for q in questions]).to(dev)
        with self._row_pass():
            f, _ = bb(ids, mask, layers=bb.cfg.feature_layers, past_key_values=pc.unpack_rows(sidx), use_cache=True,
                      past_mask=smask.index_select(0, sidx))
            q_flat = self._wide(f)[mask]
            s_flat = None if sh is None else self._wide(sh)[smask]
        q_lens = torch.tensor([len(t) for t in toks])
        if self.cfg.v3_option_repr == "embed":
            # The question's final state predicts the answer's tokens; with tied embeddings those live in the
            # INPUT-embedding space, so options are represented by their token embeddings (checked: hidden-state
            # matching is at chance on lookups, embedding matching recovers signal with no training).
            otoks = bb.tokenize([answer_option_text(o) for q in questions for o in q.options])
            o_lens = torch.tensor([len(t) for t in otoks])
            flat_ids = torch.tensor([i for t in otoks for i in t], device=dev)
            with torch.no_grad():
                o_flat = bb.embed(flat_ids).float()
        else:
            o_flat, o_lens = self.text_feats([option_text(o) for q in questions for o in q.options])
        return s_flat, smask.sum(1).cpu(), q_flat, q_lens, o_flat, o_lens

    def with_noise(self, mem, std):
        if std <= 0 or getattr(mem, "raw_feats", None) is None:
            return mem
        f = self._noise(materialize(mem.raw_feats, torch.float32), std)
        return self.prepare_memory(StateMemory(f, mem.mask, mem.n_tokens), cache=False)

    # ------------------------------------------------------------------ dense text features (cached)
    def text_cache(self) -> TokenCache:
        c = self.__dict__.get("_tok_cache")
        if c is None:
            c = TokenCache(self.d, self.backbone.device, max_tokens=self.cfg.text_cache * 16)
            self.__dict__["_tok_cache"] = c
        return c

    def text_feats(self, texts: Sequence[str]):
        """Frozen token features for texts encoded alone -> (flat (sum L, d) float32, lens CPU)."""
        assert len(self.backbone.cfg.feature_layers) == 1, "V3 caches single-layer features"
        cache = self.text_cache()
        encode_missing(cache, self.backbone, texts)
        return cache.lookup(texts)

    # ------------------------------------------------------------------ forward
    def run(self, questions: Sequence[Question], mem: Memory):
        p = self._pack_light(questions)
        return p, self.forward(p, mem, questions=questions)

    def _pack_light(self, questions) -> Packed:
        dev = self.backbone.device
        k = torch.tensor([len(q.options) for q in questions])
        K = int(k.max())
        ar = torch.arange(K)[None, :]
        opt_mask = ar < k[:, None]
        slots = torch.where(opt_mask, ar + 1, torch.zeros_like(ar))
        labs = [q.label for q in questions]
        labels = torch.tensor(labs).to(dev) if all(l is not None for l in labs) else None
        z = torch.zeros(0, dtype=torch.long)
        return Packed(z.view(0, 0), z.view(0, 0).bool(), z, z.view(0, 3), z.view(0, 3), z, z,
                      opt_mask.to(dev), slots.to(dev),
                      torch.tensor([q.state_idx for q in questions]).to(dev), labels)

    def forward(self, p: Packed, mem: Memory, questions: Sequence[Question] = None, feats=None) -> DecisionOutput:
        dev = self.backbone.device
        N, K = p.opt_mask.shape
        dm = self.cfg.d_model
        # ---- CPU metadata (no GPU syncs)
        n_opt = torch.tensor([len(q.options) for q in questions])
        opt_q = torch.repeat_interleave(torch.arange(N), n_opt)                        # (M,) question of option
        opt_k = torch.arange(int(n_opt.sum())) - torch.repeat_interleave(torch.cumsum(n_opt, 0) - n_opt, n_opt)
        sidx = torch.tensor([q.state_idx for q in questions])
        if self.conditioned:
            s_flat, s_lens, q_flat, q_lens, o_flat, o_lens = self._conditioned_feats(questions, mem)
            S = None
            if s_flat is not None:
                srow, scol, _, _, _ = segment_layout(s_lens, torch.arange(len(s_lens)), len(s_lens))
                S = torch.zeros(len(s_lens), int(s_lens.max()), self.df, device=dev)
                S[(srow.to(dev), scol.to(dev))] = s_flat            # right-aligned state features
        else:
            q_flat, q_lens = self.text_feats([question_text(q.prompt) for q in questions])
            o_flat, o_lens = self.text_feats([option_text(o) for q in questions for o in q.options])
            s_lens = mem.lens if getattr(mem, "lens", None) is not None else mem.mask.sum(1).cpu()
            S = materialize(mem.feats, torch.float32)
        # ---- options -> (N, P, d)
        row, col, seg, pos, row_len = segment_layout(o_lens, opt_q, N)
        none = self.cfg.v3_none
        P = int(row_len.max()) + int(none)
        rc = (row.to(dev), col.to(dev))
        X = torch.zeros(N, P, self.df, device=dev)
        X[rc] = o_flat
        opt_id = torch.full((N, P), -1, dtype=torch.long, device=dev)
        opt_id[rc] = opt_k[seg].to(dev)
        opos = torch.zeros(N, P, dtype=torch.long, device=dev)
        opos[rc] = pos.to(dev)
        opt_mask = p.opt_mask
        if none:                                            # one sink token right after each question's options
            nrc = (torch.arange(N, device=dev), row_len.to(dev))
            X = X.index_put(nrc, self.none_x.float().expand(N, -1))
            opt_id[nrc] = K
            opt_mask = torch.cat([opt_mask, torch.ones(N, 1, dtype=torch.bool, device=dev)], 1)
        x_pad = opt_id < 0
        # ---- context [state ; question] -> (N, Tc, d)
        ts = s_lens[sidx]                                                               # (N,) CPU
        pos0 = torch.zeros_like(ts)                                                     # RoPE offset of column 0
        if S is None:                   # state-free: context = question tokens only, positions still after the state
            pos0, ts = ts, torch.zeros_like(ts)
        Tc = int((ts + q_lens).max())
        C = torch.zeros(N, Tc, self.df, device=dev)
        if S is not None:
            T = min(S.shape[1], Tc)
            C[:, :T] = S.index_select(0, sidx.to(dev))[:, :T]
        qrow, qcol, _, qpos, _ = segment_layout(q_lens, torch.arange(N), N)
        C[(qrow.to(dev), (qcol + ts[qrow]).to(dev))] = q_flat                            # overwrite pads after state
        t = torch.arange(Tc)[None, :]
        in_state = t < ts[:, None]
        in_q = (t >= ts[:, None]) & (t < (ts + q_lens)[:, None])
        cpos = t + pos0[:, None]                                                      # state 0..Ts-1, question Ts..
        c_pad = ~(in_state | in_q)
        C = C * (~c_pad).to(dev)[..., None]                                             # zero state padding
        # ---- stack inputs: part-specific projections, positions via RoPE inside attention
        xin = self.in_norm(X)
        x = self.proj_option(xin)
        lpX = None
        if self.cfg.v3_lm_feature:
            lpX = torch.zeros(N, P, device=dev)
            lpX[rc] = self._lm_lp.float()
            if none:
                lpX = lpX.index_put(nrc, self.none_lp.float().expand(N))
            x = x + self.lp_proj(lpX[..., None])
        slots = torch.cat([p.slots, torch.zeros_like(p.slots[:, :1])], 1) if none else p.slots   # sink: no slot
        slot_ids = torch.where(x_pad, torch.zeros_like(opt_id), slots.gather(1, opt_id.clamp_min(0)))
        x = x + self.slots(slot_ids)                                                     # zeros when slot_emb == none
        cin = self.in_norm(C)
        in_q_d = in_q.to(dev)[..., None]
        c = torch.where(in_q_d, self.proj_question(cin), self.proj_state(cin))
        if self.cfg.v3_cross == "question":                                              # hide state tokens
            c_pad = c_pad | in_state
        c_pad = c_pad.to(dev)
        cpos = cpos.to(dev)
        same = (opt_id[:, :, None] == opt_id[:, None, :]) & ~x_pad[:, :, None]
        for cl in self.context_layers:
            c = cl(c, c_pad, cpos)
        outs = []
        for layer in self.layers:
            x, c = layer(x, c, same, x_pad, c_pad, opos, cpos)
            outs.append(x)
        if self.readout_mix is not None:
            w = torch.softmax(self.readout_mix, 0)
            x = torch.einsum("l,lnpd->npd", w, torch.stack(outs))
        # ---- CLS per option: attends only to its own option's tokens
        Ko = K + int(none)
        own = opt_id[:, None, :] == torch.arange(Ko, device=dev)[None, :, None]         # (N,Ko,P)
        b = torch.zeros(own.shape, device=dev).masked_fill(~own, NEG)[:, None]         # (N,1,K,P)
        h = self.cls.weight.expand(N, Ko, dm)
        h = h + self.cls_attn(self.cls_nq(h), self.cls_nkv(x), b, None, opos)   # residual
        h = h + self.cls_ff(self.cls_nf(h))                             # residual
        h = self.final_norm(h)
        logits = self.score(h).squeeze(-1).float()
        if lpX is not None:
            # length-normalised log-likelihood of each option under the frozen LM (arithmetic at the output)
            own_f = own.float()
            ll = (own_f * lpX[:, None, :]).sum(-1) / own_f.sum(-1).clamp_min(1)
            logits = logits + self.lm_skip * ll
        logits = logits.masked_fill(~opt_mask, NEG)
        rel = self.rel(h).squeeze(-1).float().masked_fill(~opt_mask, NEG)
        return DecisionOutput(logits, opt_mask, rel, None)


VARIANTS["v3"] = V3
