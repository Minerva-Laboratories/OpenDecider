"""Frozen hybrid-attention LM wrapper: feature extraction, layer hooks, int8 caches.

Default backbone: Qwen/Qwen3.5-0.8B (post-trained/instruct; Qwen3.5 has no "-Instruct"
suffix, the -Base repo is the pretrained one). 24 layers, 3 Gated-DeltaNet : 1 gated
full-attention. Loaded text-only via Qwen3_5ForCausalLM (vision tower and MTP head skipped).
"""
from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn
import yaml

from .quant import Int8Embedding, Int8Tensor, kv_mode, materialize, maybe_quantize, quantize_int8_


@dataclass
class BackboneConfig:
    path: str = "models/qwen3.5-0.8b"      # local copy; falls back to repo_id@revision when missing
    repo_id: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    dtype: str = "bfloat16"
    weight_quant: str = "int8"          # none | int8 | nf4 (bitsandbytes 4-bit; embeddings stay int8)
    kv_quant: str = "int8"              # none | int8 | int4  (all cached K/V and state memory)
    quantize_linear_state: bool = False  # GDN recurrent/conv state (not a KV cache)
    feature_layers: list = field(default_factory=lambda: ["final"])  # "final" = post-norm last layer, or ints 1..L
    prefill_chunk: int = 0              # state pass in chunks of this many tokens; 0 = one pass (measured: no peak
                                        # memory gain on the Orin, +34% latency at 15k tokens)
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "BackboneConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


@dataclass
class StateMemory:
    """Encoded state, cached once per request and shared by every question."""
    feats: torch.Tensor | Int8Tensor   # (B, T, d)  or (L, B, T, d) for multi-layer mixes
    mask: torch.Tensor                 # (B, T) bool, True = real token
    n_tokens: int

    def get(self, dtype) -> torch.Tensor:
        return materialize(self.feats, dtype)

    def nbytes(self) -> int:
        f = self.feats
        return f.nbytes() if isinstance(f, Int8Tensor) else f.numel() * f.element_size()


class Backbone(nn.Module):
    def __init__(self, lm: nn.Module, tokenizer, cfg: BackboneConfig):
        super().__init__()
        self.lm = lm                      # the *text model* (embed_tokens, layers, norm)
        self.tok = tokenizer
        self.cfg = cfg
        self.lm.requires_grad_(False)
        self.lm.eval()
        c = lm.config
        self.hidden_size = c.hidden_size
        self.num_layers = c.num_hidden_layers
        self.layer_types = list(getattr(c, "layer_types", ["full_attention"] * self.num_layers))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, cfg: BackboneConfig) -> "Backbone":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        dtype = getattr(torch, cfg.dtype)
        kw = {}
        if cfg.weight_quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        src, rev = (cfg.path, None) if os.path.isdir(cfg.path) else (cfg.repo_id, cfg.revision)
        tok = AutoTokenizer.from_pretrained(src, revision=rev)
        full = AutoModelForCausalLM.from_pretrained(src, revision=rev, dtype=dtype, device_map=cfg.device, **kw)
        lm = full.model
        del full.lm_head      # tied to embeddings; we never produce tokens
        if cfg.weight_quant == "int8":
            quantize_int8_(lm.layers)
        if cfg.weight_quant in ("int8", "nf4"):   # bitsandbytes has no 4-bit embedding: the table is int8 either way
            lm.embed_tokens = Int8Embedding(lm.embed_tokens)
            torch.cuda.empty_cache()
        return cls(lm, tok, cfg)

    # ------------------------------------------------------------------ tokenization
    def tokenize(self, texts: Sequence[str]) -> list[list[int]]:
        """One batched (Rust) tokenizer call; token ids identical to per-text calls."""
        if not texts:
            return []
        return self.tok(list(texts), add_special_tokens=False)["input_ids"]

    def pad(self, seqs: Sequence[Sequence[int]], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad in one tensor construction. Right padding is safe: causal layers never let pads
        affect real tokens."""
        device = device or self.device
        lens = [len(s) for s in seqs]
        T = max(1, max(lens))
        ids = torch.tensor([list(s) + [self.pad_id] * (T - len(s)) for s in seqs], dtype=torch.long)
        mask = torch.arange(T)[None, :] < torch.tensor(lens)[:, None]
        return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)

    @property
    def device(self):
        return self.lm.embed_tokens.qweight.device if hasattr(self.lm.embed_tokens, "qweight") \
            else self.lm.embed_tokens.weight.device

    @property
    def dtype(self):
        return getattr(torch, self.cfg.dtype)

    # ------------------------------------------------------------------ forward
    @contextlib.contextmanager
    def layer_hooks(self, hooks: dict[int, Callable[[torch.Tensor], torch.Tensor]]):
        """Apply `hooks[i](hidden)` to the output of decoder layer i (0-based) during forward."""
        handles = []
        for i, fn in hooks.items():
            def _h(mod, args, out, fn=fn):
                if isinstance(out, tuple):
                    return (fn(out[0]),) + tuple(out[1:])
                return fn(out)
            handles.append(self.lm.layers[i].register_forward_hook(_h))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, layers: Sequence = ("final",),
                hooks: dict | None = None, past_key_values=None, use_cache: bool = False,
                past_mask: torch.Tensor | None = None):
        """Run the frozen LM. Returns (features, cache). features: (B,T,d) for one layer, else (L,B,T,d).

        Gradients flow through activations (needed for V2's inserted blocks) but never into
        backbone weights (requires_grad=False).
        """
        want = [l for l in layers if l != "final"]
        captured: dict[int, torch.Tensor] = {}
        all_hooks = dict(hooks or {})
        for l in want:
            idx = int(l) - 1
            prev = all_hooks.get(idx)
            def cap(h, idx=idx, prev=prev):
                h = prev(h) if prev is not None else h
                captured[idx + 1] = h
                return h
            all_hooks[idx] = cap
        attn_mask = mask.long()
        if past_key_values is not None:
            if past_mask is None:
                past = past_key_values.get_seq_length()
                past_mask = torch.ones(mask.shape[0], past, dtype=torch.bool, device=mask.device)
            attn_mask = torch.cat([past_mask.long(), attn_mask], 1)
        with self.layer_hooks(all_hooks):
            out = self.lm(input_ids=ids, attention_mask=attn_mask, past_key_values=past_key_values,
                          use_cache=use_cache)
        feats = [out.last_hidden_state if l == "final" else captured[int(l)] for l in layers]
        f = feats[0] if len(feats) == 1 else torch.stack(feats, 0)
        return f, (out.past_key_values if use_cache else None)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.embed_tokens(ids)

    # ------------------------------------------------------------------ state memory
    @torch.no_grad()
    def encode_state(self, state_texts: Sequence[str]) -> StateMemory:
        seqs = self.tokenize(state_texts)
        ids, mask = self.pad(seqs)
        f, _ = self.forward(ids, mask, self.cfg.feature_layers)
        sm = StateMemory(maybe_quantize(f, kv_mode(self.cfg)), mask, sum(map(len, seqs)))
        sm.lens = torch.tensor([len(x) for x in seqs])          # CPU lengths: no GPU sync downstream
        return sm

    def encode_state_grad(self, ids, mask) -> torch.Tensor:
        """Training path: no caching/quantization, activations kept for autograd if needed."""
        f, _ = self.forward(ids, mask, self.cfg.feature_layers)
        return f

    def prefill(self, ids, mask, layers=("final",), cache=None, keep_hidden: bool = True):
        """Forward over (B, T) in chunks of cfg.prefill_chunk tokens, each continuing the cache: same result as one
        pass (exact up to float error), but activation memory is bounded by the chunk, not by T.
        Returns (hidden features over all T, or None if not keep_hidden; cache)."""
        C = self.cfg.prefill_chunk or ids.shape[1]
        feats = []
        past = cache.get_seq_length() if cache is not None else 0
        pmask = torch.ones(ids.shape[0], past, dtype=torch.bool, device=mask.device) if past else None
        for a in range(0, ids.shape[1], C):
            b = min(a + C, ids.shape[1])
            f, cache = self.forward(ids[:, a:b], mask[:, a:b], layers, past_key_values=cache, use_cache=True,
                                    past_mask=pmask)
            pmask = mask[:, :b] if pmask is None else torch.cat([pmask, mask[:, a:b]], 1)
            if keep_hidden:
                feats.append(f)
        return (torch.cat(feats, -2) if keep_hidden else None), cache

    # ------------------------------------------------------------------ prefix cache (V0)
    def prefix_cache_batch(self, seqs: Sequence[Sequence[int]], return_hidden: bool = False, grad: bool = False,
                           quantize: bool | None = None):
        """ONE backbone pass for many states, LEFT-padded so every continuation directly follows its
        state's last real token. Leading pads leave the GDN recurrent/conv state at zero and RoPE is
        relative, so this equals encoding each state alone (tested). Returns (cache, left_mask (S,T))."""
        lens = [len(x) for x in seqs]
        T = max(lens)
        ids = torch.tensor([[self.pad_id] * (T - len(x)) + list(x) for x in seqs], dtype=torch.long)
        mask = torch.arange(T)[None, :] >= (T - torch.tensor(lens))[:, None]
        ids, mask = ids.to(self.device), mask.to(self.device)
        # grad=True (training with adapters on the state pass): keep the graph and do NOT quantise the cache, so
        # gradients flow from the question/option rows back through the cached K/V and GDN states.
        with torch.set_grad_enabled(grad):
            layers = self.cfg.feature_layers if return_hidden else ("final",)
            if grad:                                            # training graph: one pass (chunking saves nothing)
                f, cache = self.forward(ids, mask, layers, use_cache=True)
            else:
                f, cache = self.prefill(ids, mask, layers, keep_hidden=return_hidden)
        q = (not grad) if quantize is None else quantize          # quantize=False: exact values without a graph
        pc = PackedCache.pack(cache, kv_int8=q and kv_mode(self.cfg),
                              state_int8=self.cfg.quantize_linear_state and q)
        return (pc, mask, f) if return_hidden else (pc, mask)


    @torch.no_grad()
    def prefix_cache(self, ids: torch.Tensor) -> "PackedCache":
        mask = torch.ones_like(ids, dtype=torch.bool)
        _, cache = self.forward(ids, mask, ("final",), use_cache=True)
        return PackedCache.pack(cache, kv_int8=kv_mode(self.cfg),
                                state_int8=self.cfg.quantize_linear_state)


# ---------------------------------------------------------------------- packed cache
_KV_ATTRS = ("keys", "values")
_STATE_ATTRS = ("conv_states", "recurrent_states")


def _assign_recurrent(self, recurrent_states, state_idx: int = 0, **kw):
    if not self.is_recurrent_states_initialized[state_idx]:
        self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
    self.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


def _assign_conv(self, conv_states, state_idx: int = 0, conv_kernel_size=None, **kw):
    """Same as transformers' LinearAttentionLayer.update_conv_state (non-record_past path), but assigns."""
    if not self.is_conv_states_initialized[state_idx]:
        self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
    if not self.has_previous_state[state_idx]:
        full = conv_states
        self.has_previous_state[state_idx] = True
        k = self.conv_kernel_size[state_idx]
        if full.shape[-1] < k:
            full = torch.nn.functional.pad(full, (k - full.shape[-1], 0), value=0)
    else:
        full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
    self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx]:]
    return full


_ASSIGNING: dict = {}


def _assigning_class(cls):
    if cls not in _ASSIGNING:
        _ASSIGNING[cls] = type("Assigning" + cls.__name__, (cls,),
                               {"update_recurrent_state": _assign_recurrent, "update_conv_state": _assign_conv})
    return _ASSIGNING[cls]


class PackedCache:
    """A transformers DynamicCache with tensors stored as int8 (K/V) between uses.

    Works on the hybrid Qwen3.5 cache: full-attention layers carry keys/values; Gated-DeltaNet
    layers carry conv_states / recurrent_states dicts. `unpack(batch)` rebuilds a fresh bf16
    DynamicCache repeated along batch, so many questions can continue from one state prefix.
    """

    def __init__(self, template, tensors: dict, seq_len: int):
        self.template = template
        self.tensors = tensors
        self.seq_len = seq_len

    @classmethod
    def pack(cls, cache, kv_int8: bool = True, state_int8: bool = False) -> "PackedCache":
        tensors = {}
        template = copy.copy(cache)
        template.layers = []
        for li, layer in enumerate(cache.layers):
            shell = copy.copy(layer)
            for a in _KV_ATTRS:
                t = getattr(layer, a, None)
                if isinstance(t, torch.Tensor) and t.numel():
                    tensors[(li, a, None)] = maybe_quantize(t, kv_int8)
                    setattr(shell, a, None)
            for a in _STATE_ATTRS:
                d = getattr(layer, a, None)
                if isinstance(d, dict):
                    newd = {}
                    for k, t in d.items():
                        if isinstance(t, torch.Tensor):
                            tensors[(li, a, k)] = maybe_quantize(t, state_int8)
                            newd[k] = None
                        else:
                            newd[k] = t
                    setattr(shell, a, newd)
                elif isinstance(d, torch.Tensor):
                    tensors[(li, a, None)] = maybe_quantize(d, state_int8)
                    setattr(shell, a, None)
            template.layers.append(shell)
        return cls(template, tensors, cache.get_seq_length())

    def unpack(self, batch: int = 1):
        cache = copy.copy(self.template)
        cache.layers = [copy.copy(l) for l in self.template.layers]
        for l in cache.layers:
            for a in _STATE_ATTRS:
                if isinstance(getattr(l, a, None), dict):
                    setattr(l, a, dict(getattr(l, a)))
        for (li, a, k), t in self.tensors.items():
            x = materialize(t)
            if batch > 1:
                x = x.repeat_interleave(batch, dim=0)
            x = x.contiguous()
            if k is None:
                setattr(cache.layers[li], a, x)
            else:
                getattr(cache.layers[li], a)[k] = x
        return cache

    def unpack_rows(self, rows: torch.Tensor, autograd_safe: bool = True, attn_placeholder: bool = False):
        """Fresh cache whose batch row i is the packed row `rows[i]` (one gather per tensor).

        autograd_safe: transformers updates GDN conv/recurrent states IN PLACE (copy_), which breaks backward when
        gradients flow through the row pass (e.g. VeRA adapters). Rows never reuse their cache, so updates are made
        by assignment instead; the values are identical."""
        cache = copy.copy(self.template)
        cache.layers = [copy.copy(l) for l in self.template.layers]
        if autograd_safe:
            # swap in a subclass (NOT bound methods on the instance: those create reference cycles that keep every
            # row cache alive until the cycle GC runs, which blew past the memory cap in branched eval)
            for l in cache.layers:
                if hasattr(l, "update_recurrent_state"):
                    l.__class__ = _assigning_class(type(l))
        for l in cache.layers:
            for a in _STATE_ATTRS:
                if isinstance(getattr(l, a, None), dict):
                    setattr(l, a, dict(getattr(l, a)))
        for (li, a, k), t in self.tensors.items():
            if attn_placeholder and a in _KV_ATTRS:           # shared-prefix attention reads K/V itself (prefix_attn)
                sh = (rows.numel(), *t.shape[1:])
                dev = t.q.device if isinstance(t, Int8Tensor) else t.device
                setattr(cache.layers[li], a, torch.zeros((), dtype=t.dtype, device=dev).expand(sh))
                continue
            if isinstance(t, Int8Tensor):                      # select rows on int8 FIRST, then dequantise
                x = t.index_select(0, rows.to(t.q.device)).dequantize()
            else:
                x = t.index_select(0, rows.to(t.device))
            if k is None:
                setattr(cache.layers[li], a, x.contiguous())
            else:
                getattr(cache.layers[li], a)[k] = x.contiguous()
        return cache

    def nbytes(self) -> int:
        n = 0
        for t in self.tensors.values():
            n += t.nbytes() if isinstance(t, Int8Tensor) else t.numel() * t.element_size()
        return n


# ---------------------------------------------------------------------- kernel selection
_KERNEL_FUNCS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update")
_saved_kernels: dict = {}


def use_reference_kernels(enable: bool = True) -> None:
    """Swap the Qwen3.5 fla / causal-conv1d kernels for transformers' pure-torch references.

    Needed on CPU (unit tests) and for export; the CUDA fast path is fla (Triton) +
    causal-conv1d, both verified to run on Orin sm_87.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    for name in _KERNEL_FUNCS:
        if enable:
            f = _saved_kernels.setdefault(name, getattr(mq, name))
            ref = f
            while hasattr(ref, "__wrapped__"):
                ref = ref.__wrapped__
            setattr(mq, name, ref)
        elif name in _saved_kernels:
            setattr(mq, name, _saved_kernels[name])


def checkpoint_backbone_layers(bb: "Backbone") -> None:
    """Activation checkpointing for frozen decoder layers (training V2 through the backbone).

    Wraps each layer's forward in torch.utils.checkpoint; forward hooks (the V2 fusion blocks)
    run outside the checkpointed region, so only the frozen layers are recomputed in backward.
    No-op when grad is disabled."""
    import torch.utils.checkpoint as cp
    for layer in bb.lm.layers:
        if getattr(layer, "_od_ckpt", False):
            continue
        f = layer.forward
        def fwd(*args, _f=f, **kw):
            # never checkpoint a cached continuation: the recompute would update the KV cache a second time
            if kw.get("past_key_values") is not None:
                return _f(*args, **kw)
            if torch.is_grad_enabled() and any(a.requires_grad for a in args if isinstance(a, torch.Tensor)):
                return cp.checkpoint(_f, *args, use_reentrant=False, **kw)
            return _f(*args, **kw)
        layer.forward = fwd
        layer._od_ckpt = True
