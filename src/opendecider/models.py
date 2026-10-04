"""V0 (LM-with-head), V1 (late fusion), V2 (deep fusion, main hypothesis). See docs/SPEC.md section 4.

All variants share one call pattern:
    mem = model.encode_states(state_texts)     # once per request (cached, int8 when kv_quant=int8)
    out = model(packed, mem)                   # every question, isolated from the others
    out.logits: (N, Kmax) with NEG at padding; out.opt_mask; out.relevance (stage-1 scores)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import yaml

from .backbone import Backbone, PackedCache, StateMemory
from .batching import Packed, Question, pack
from .formatting import option_text, question_text
from .fusion import CrossAttention, GatedXAttnDense, PerceiverResampler
from .heads import PointerHead
from .options import SlotEmbedding, span_pool
from .quant import Int8Tensor, kv_mode, materialize, maybe_quantize


@dataclass
class ModelConfig:
    variant: str = "v2"                  # v0 | v1 | v2
    d_model: int = 512
    n_heads: int = 8
    decision_layers: int = 4             # V1 decision transformer depth
    interaction_layers: int = 2          # V2 option-interaction depth
    n_codes: int = 8                     # V1 learned codes (m)
    slot_emb: str = "learned"            # learned | none | random_per_call
    pointer: str = "ptr"                 # ptr | bilinear
    resampler: int = 0                   # 0 = off, else N latents
    option_encoding: str = "joint"       # V2: joint | independent
    xattn_layers: str | list = "full_attention"   # V2: after every full-attention layer (every 4th) or explicit list
    xattn_ff_mult: int = 2
    noul_mode: str = "choice"            # choice | sigmoid
    option_pool: str = "mean"            # V1/V2: mean | last | weighted (causal backbone: see options.span_pool)
    dropout: float = 0.1
    v0_prefix_cache: bool = True
    text_cache: int = 200_000            # V1/V3: max cached question/option text encodings (0 = off)
    v3_context_update: bool = False      # V3: context tokens also attend to options (per question)
    v3_readout: str = "last"             # V3: CLS reads the last stack layer, or "mix" of all layers
    v3_context_layers: int = 0           # V3: joint self-attention layers over [state;question] (question-aware)
    v3_features: str = "isolated"        # V3: isolated (texts alone) | conditioned (LLM row state->question->options)
                                         #     | question_conditioned (LLM row state->question; options alone)
                                         #     | branched (one LLM row [question ; option_k] per option, from the state cache)
    v3_row_format: str = "list"          # branched rows: list ("Options:\n- opt") | answer (sorted candidates + "Answer:" + " opt")
    v3_list_cap: int = 16                # answer format: list candidates only when K <= cap
    v3_lm_feature: bool = False          # branched: frozen-LM log-prob of option tokens as trunk feature + gated skip
    v3_vera_layers: int = 0              # VeRA adapters on the top-N decoder layers, ROW pass only (0 = frozen)
    v3_vera_rank: int = 256
    v3_vera_scope: str = "rows_top"      # rows_top: top-N layers, option rows only | all: every layer, state + rows
    v3_question_cache: bool = False      # branched answer rows: state -> question(+listing) cache -> option suffixes
    v3_row_checkpoint: bool = False      # recompute each option-row chunk in backward (VeRA training memory)
    v3_layer_combine: str = "mix"        # multi-layer features -> one per token: mix (softmax weights) | attn (DepthAttn)
                                         # | stitch (linear map of unit-normalised layers into another backbone's space)
    v3_stitch_dim: int = 0               # stitch: width of the feature space the trunk was trained on (e.g. 2048)
    v3_none: bool = False                # learned `none` sink option appended to every question (last logit column)
    v3_shared_prefix: bool = True        # branched rows: one shared state K/V per state, no per-row copies
    v3_branch_chunk: int = 64            # branched: rows per backbone call (each row copies ~18 MB of GDN state)
    v3_cross: str = "full"               # V3: options cross-attend to full [state;question] | question | none
    v3_option_repr: str = "embed"        # question_conditioned: option tokens as input EMBEDDINGS (the space the
                                         # question's final state predicts into; tied LM head) | hidden

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d.get("model", d))


@dataclass
class DecisionOutput:
    logits: torch.Tensor
    opt_mask: torch.Tensor
    relevance: torch.Tensor | None = None
    noul_logit: torch.Tensor | None = None


@dataclass
class Memory:
    feats: torch.Tensor | Int8Tensor | None        # (S,T,d) raw state features (kept for the noise sampler)
    mask: torch.Tensor                             # (S,T) mask of `feats`
    kv: list = field(default_factory=list)         # per fusion block: (S,2,h,T',dh), int8 when cached
    prefix: PackedCache | None = None              # V0 only
    n_tokens: int = 0
    kv_mask: torch.Tensor | None = None            # (S,T') mask of kv (T' = N latents with a resampler)


def _scatter_options(h_opt: torch.Tensor, p: Packed) -> torch.Tensor:
    N, K = p.opt_mask.shape
    out = torch.zeros(N, K, h_opt.shape[-1], dtype=h_opt.dtype, device=h_opt.device)
    out[p.opt_n, p.opt_k] = h_opt
    return out


class DecisionModel(nn.Module):
    pack_mode = "joint"

    @property
    def row_format(self) -> str:
        """How this model's question rows are serialised (in-context examples are rendered the same way)."""
        return getattr(self.cfg, "v3_row_format", "list") if getattr(self, "branched", False) else "list"

    def __init__(self, backbone: Backbone, cfg: ModelConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.d = backbone.hidden_size
        self.slots = SlotEmbedding(cfg.d_model, cfg.slot_emb)
        self.head = PointerHead(cfg.d_model, cfg.pointer)
        nl = backbone.cfg.feature_layers
        self.layer_mix = nn.Parameter(torch.zeros(len(nl))) if len(nl) > 1 else None
        # Frozen decoder features are large (L2 ~120, a few dims ~20x the median). Normalise them
        # wherever they enter trainable layers, or cross-attention logits saturate at init.
        self.mem_norm = nn.LayerNorm(self.d)
        self.feat_norm = nn.LayerNorm(self.d)

    # --------------------------------------------------------------- packing
    def pack(self, questions: Sequence[Question], state_prefix=None) -> Packed:
        return pack(self.backbone.tokenize, questions, self.pack_mode, self.backbone.pad_id,
                    state_prefix).to(self.backbone.device)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("backbone.")]

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    # --------------------------------------------------------------- memory
    def _mix(self, f: torch.Tensor) -> torch.Tensor:
        if self.layer_mix is None:
            return f
        w = torch.softmax(self.layer_mix, 0).to(f.dtype)
        return (w[:, None, None, None] * f).sum(0)

    def encode_states(self, state_texts: Sequence[str], noise_std: float = 0.0) -> Memory:
        """Inference: encode once, cache (int8 if configured)."""
        with torch.no_grad():
            sm = self.backbone.encode_state(state_texts)
            m = self.prepare_memory(sm, cache=True, noise_std=noise_std)
            m.lens = sm.lens
            return m

    def memory_from_ids(self, ids, mask, noise_std: float = 0.0) -> Memory:
        """Training: backbone frozen (no grad), trainable memory processing keeps grad."""
        with torch.no_grad():
            f = self.backbone.encode_state_grad(ids, mask)
        return self.prepare_memory(StateMemory(f, mask, int(mask.sum())), cache=False, noise_std=noise_std)

    def prepare_memory(self, sm: StateMemory, cache: bool, noise_std: float = 0.0) -> Memory:
        raise NotImplementedError

    @staticmethod
    def _noise(f: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return f
        rms = f.float().pow(2).mean(-1, keepdim=True).sqrt()
        return f + (torch.randn_like(f.float()) * rms * std).to(f.dtype)

    def with_noise(self, mem: Memory, std: float) -> Memory:
        """Sampler helper: perturb the cached memory (dequantised) without re-running the backbone."""
        if std <= 0 or mem.feats is None:
            return mem
        f = self._noise(materialize(mem.feats, torch.float32), std)
        return self.prepare_memory(StateMemory(f, mem.mask, mem.n_tokens), cache=False, noise_std=0.0) \
            if not isinstance(self, V0) else mem

    def _finish(self, h_q, h_o, p: Packed) -> DecisionOutput:
        logits = self.head(h_q, h_o, p.opt_mask)
        rel = self.head.relevance_logits(h_o, p.opt_mask)
        noul = self.head.noul(h_q).squeeze(-1).float() if self.cfg.noul_mode == "sigmoid" else None
        return DecisionOutput(logits, p.opt_mask, rel, noul)


# ============================================================================ V0
class V0(DecisionModel):
    """Causal LM over [state][question][opt_1]..[opt_K]; MLP on the last token of each option span."""
    pack_mode = "joint"

    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        self.proj = nn.Linear(self.d, cfg.d_model)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.head = PointerHead(cfg.d_model, cfg.pointer)

    def prepare_memory(self, sm, cache, noise_std=0.0):
        return Memory(None, sm.mask, n_tokens=sm.n_tokens)

    def encode_states(self, state_texts, noise_std: float = 0.0) -> Memory:
        seqs = self.backbone.tokenize(state_texts)
        if self.cfg.v0_prefix_cache and len(seqs) == 1:
            ids = torch.tensor(seqs, device=self.backbone.device)
            pc = self.backbone.prefix_cache(ids)
            return Memory(None, torch.ones_like(ids, dtype=torch.bool), prefix=pc, n_tokens=len(seqs[0]))
        m = Memory(None, torch.zeros(0), n_tokens=sum(map(len, seqs)))
        m.extra_prefix = seqs
        return m

    def memory_from_ids(self, ids, mask, noise_std=0.0):
        m = Memory(None, mask, n_tokens=int(mask.sum()))
        m.extra_prefix = [ids[i, : int(mask[i].sum())].tolist() for i in range(ids.shape[0])]
        return m

    def run(self, questions: Sequence[Question], mem: Memory) -> tuple[Packed, DecisionOutput]:
        """Each state is read ONCE (prefix cache); all its questions continue from it (O3/O4).
        Many states: one left-padded batched prefix pass + one continuation pass for all questions."""
        p = self.pack(questions)
        if mem.prefix is not None:                                   # inference: single cached state
            cache = mem.prefix.unpack(p.ids.shape[0])
            past_mask = None
        else:
            if getattr(mem, "prefix_batch", None) is None:
                mem.prefix_batch = self.backbone.prefix_cache_batch(mem.extra_prefix)
            pc, smask = mem.prefix_batch
            rows = p.row_state
            cache = pc.unpack_rows(rows)
            past_mask = smask.index_select(0, rows)
        with torch.no_grad():
            f, _ = self.backbone(p.ids, p.mask, past_key_values=cache, use_cache=True, past_mask=past_mask)
        return p, self.forward(p, mem, feats=f)

    def forward(self, p: Packed, mem: Memory, feats=None) -> DecisionOutput:
        f = feats.float()
        last = f[p.opt_spans[:, 0], p.opt_spans[:, 2] - 1]
        qlast = f[p.q_spans[:, 0], p.q_spans[:, 2] - 1]
        h_o = _scatter_options(self.proj(self.feat_norm(last)), p) + self.slots(p.slots)
        h_o = self.mlp(h_o)
        h_q = self.mlp(self.proj(self.feat_norm(qlast)))
        return self._finish(h_q, h_o, p)


# ============================================================================ V1
class DecisionLayer(nn.Module):
    def __init__(self, d, d_mem, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.xa = CrossAttention(d, d_mem, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, pad, kv, mem_mask, row_state):
        h = self.ln1(x)
        x = x + self.sa(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        x = x + self.xa(self.ln2(x), kv, mem_mask, row_state)
        return x + self.ff(self.ln3(x))


class V1(DecisionModel):
    """Late fusion: pooled question and option vectors (each encoded alone) + decision transformer
    with cross-attention to the (optionally resampled) state memory."""
    pack_mode = "separate"

    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        d, dm = self.d, cfg.d_model
        self.resampler = PerceiverResampler(d, cfg.resampler) if cfg.resampler else None
        self.proj_q = nn.Linear(d, dm)
        self.proj_o = nn.Linear(d, dm)
        self.type_emb = nn.Embedding(3, dm)   # 0 question, 1 option, 2 code
        self.codes = nn.Parameter(torch.randn(cfg.n_codes, dm) * 0.02)
        self.layers = nn.ModuleList(DecisionLayer(dm, d, cfg.n_heads, cfg.dropout) for _ in range(cfg.decision_layers))
        self.ln = nn.LayerNorm(dm)

    def prepare_memory(self, sm, cache, noise_std=0.0):
        f = self.mem_norm(self._noise(self._mix(sm.get(torch.float32)), noise_std).float())
        mask = sm.mask
        if self.resampler is not None:
            f, mask = self.resampler(f, mask)
        q8 = cache and kv_mode(self.backbone.cfg)
        kv = [maybe_quantize(l.xa.project_kv(f), q8) for l in self.layers]
        return Memory(sm.feats if cache else None, mask if self.resampler is None else sm.mask, kv,
                      n_tokens=sm.n_tokens, kv_mask=mask)

    def run(self, questions, mem):
        p = self.pack(questions)
        if self.cfg.text_cache:
            texts = []
            for q in questions:
                texts.append(question_text(q.prompt))
                texts.extend(option_text(o) for o in q.options)
            v = self.pooled_texts(texts)
            is_q = torch.zeros(len(texts), dtype=torch.bool)
            i = 0
            for q in questions:
                is_q[i] = True
                i += 1 + len(q.options)
            return p, self.forward(p, mem, pooled=(v[is_q.to(v.device)], v[~is_q.to(v.device)]))
        return p, self.forward(p, mem)

    @torch.no_grad()
    def pooled_texts(self, texts: Sequence[str], budget: int = 8192) -> torch.Tensor:
        """Pooled frozen-backbone vectors for texts encoded ALONE (V1 never conditions them on
        the state), cached by text. Deterministic, so caching only changes speed."""
        cache = self.__dict__.setdefault("_text_cache", {})
        todo = sorted({t for t in texts if t not in cache}, key=len)
        if todo:
            if len(cache) + len(todo) > self.cfg.text_cache:
                cache.clear()
            toks = self.backbone.tokenize(todo)
            order = sorted(range(len(todo)), key=lambda i: len(toks[i]))
            start = 0
            while start < len(order):                      # length-sorted chunks, little padding
                end, L = start, 0
                while end < len(order) and (end - start + 1) * max(L, len(toks[order[end]])) <= budget:
                    L = max(L, len(toks[order[end]])); end += 1
                end = max(end, start + 1)
                idx = order[start:end]
                ids, mask = self.backbone.pad([toks[i] for i in idx])
                f, _ = self.backbone(ids, mask)
                lens = mask.sum(1)
                spans = torch.stack([torch.arange(len(idx), device=ids.device), torch.zeros_like(lens), lens], 1)
                v = span_pool(f, spans, self.cfg.option_pool)
                for j, i in enumerate(idx):
                    cache[todo[i]] = v[j]
                start = end
        return torch.stack([cache[t] for t in texts]).float()

    def forward(self, p: Packed, mem: Memory, feats=None, pooled=None) -> DecisionOutput:
        N, K = p.opt_mask.shape
        pm = self.cfg.option_pool
        if pooled is not None:
            vq, vo = pooled
        else:
            if feats is None:
                with torch.no_grad():
                    feats, _ = self.backbone(p.ids, p.mask)
            f = feats.float()
            vq, vo = span_pool(f, p.q_spans, pm).float(), span_pool(f, p.opt_spans, pm).float()
        h_q = self.proj_q(self.feat_norm(vq)) + self.type_emb.weight[0]
        h_o = _scatter_options(self.proj_o(self.feat_norm(vo)), p) + self.slots(p.slots) + self.type_emb.weight[1]
        codes = (self.codes + self.type_emb.weight[2])[None].expand(N, -1, -1)
        x = torch.cat([h_q[:, None], h_o, codes], 1)
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), ~p.opt_mask,
                         torch.zeros(N, codes.shape[1], dtype=torch.bool, device=x.device)], 1)
        for l, kv in zip(self.layers, mem.kv):
            x = l(x, pad, kv, mem.kv_mask, p.q_state)
        x = self.ln(x)
        return self._finish(x[:, 0], x[:, 1:1 + K], p)


# ============================================================================ V2
class V2(DecisionModel):
    """Deep fusion: question/option tokens run through the frozen LM with gated xattn-dense blocks
    (tanh gate, alpha=0 init) inserted after every full-attention layer, cross-attending to the
    state memory; option spans mean-pooled -> option tokens + slot emb -> interaction -> pointer."""

    def __init__(self, backbone, cfg):
        super().__init__(backbone, cfg)
        self.pack_mode = cfg.option_encoding
        d, dm = self.d, cfg.d_model
        if cfg.xattn_layers == "full_attention":
            idx = [i for i, t in enumerate(backbone.layer_types) if t == "full_attention"]
        elif cfg.xattn_layers == "every4":
            idx = list(range(3, backbone.num_layers, 4))
        else:
            idx = list(cfg.xattn_layers)
        self.xattn_idx = idx
        self.resampler = PerceiverResampler(d, cfg.resampler) if cfg.resampler else None
        self.blocks = nn.ModuleList(GatedXAttnDense(d, d, cfg.n_heads, cfg.xattn_ff_mult, cfg.dropout) for _ in idx)
        self.proj_q = nn.Linear(d, dm)
        self.proj_o = nn.Linear(d, dm)
        enc = nn.TransformerEncoderLayer(dm, cfg.n_heads, 4 * dm, cfg.dropout, batch_first=True, norm_first=True)
        self.interact = nn.TransformerEncoder(enc, cfg.interaction_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dm)

    def prepare_memory(self, sm, cache, noise_std=0.0):
        f = self.mem_norm(self._noise(self._mix(sm.get(torch.float32)), noise_std).float())
        mask = sm.mask
        if self.resampler is not None:
            f, mask = self.resampler(f, mask)
        q8 = cache and kv_mode(self.backbone.cfg)
        kv = [maybe_quantize(b.project_kv(f), q8) for b in self.blocks]
        return Memory(sm.feats if cache else None, mask if self.resampler is None else sm.mask, kv,
                      n_tokens=sm.n_tokens, kv_mask=mask)

    def run(self, questions, mem):
        p = self.pack(questions)
        return p, self.forward(p, mem)

    def forward(self, p: Packed, mem: Memory, feats=None) -> DecisionOutput:
        row_state = p.row_state
        hooks = {li: (lambda h, b=b, kv=kv: b(h, kv, mem.kv_mask, row_state))
                 for li, b, kv in zip(self.xattn_idx, self.blocks, mem.kv)}
        f, _ = self.backbone(p.ids, p.mask, hooks=hooks)
        f = f.float()
        N, K = p.opt_mask.shape
        pm = self.cfg.option_pool
        h_q = self.proj_q(self.feat_norm(span_pool(f, p.q_spans, pm)))
        h_o = _scatter_options(self.proj_o(self.feat_norm(span_pool(f, p.opt_spans, pm))), p) + self.slots(p.slots)
        x = torch.cat([h_q[:, None], h_o], 1)
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), ~p.opt_mask], 1)
        x = self.ln(self.interact(x, src_key_padding_mask=pad))
        return self._finish(x[:, 0], x[:, 1:], p)


VARIANTS = {"v0": V0, "v1": V1, "v2": V2}


def build_model(backbone: Backbone, cfg: ModelConfig) -> DecisionModel:
    if cfg.variant == "v3":
        from . import v3  # noqa: F401  (registers V3)
    return VARIANTS[cfg.variant](backbone, cfg)
