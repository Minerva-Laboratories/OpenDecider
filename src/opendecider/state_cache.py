"""Tier 1 state reuse across requests: an exact prefix cache over chunk boundaries (docs/roadmap_designs.md §3).

Qwen3.5 is hybrid: Gated-DeltaNet layers carry a recurrent state that depends on every earlier token in order, so
independently encoded chunks cannot be spliced. What is exact is resuming from a snapshot of the whole cache at a
prefix boundary. A snapshot stores
  - the attention K/V of ITS OWN tokens only (a delta over its parent snapshot; int8), and
  - the GDN recurrent/conv states at its end (fixed size, independent of length).
A request resumes from its deepest cached content prefix, prefills the rest of its content in one pass (snapshot
taken there), then the closers (`]}\n`) in a second short pass that is never cached, so a growing state (events
appended to a list) shares everything up to its previous end. Only leaves are evicted (LRU), so every snapshot's
ancestors are present.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from .backbone import PackedCache
from .quant import maybe_quantize, materialize

_KV = ("keys", "values")
_ST = ("conv_states", "recurrent_states")


@dataclass
class Snapshot:
    key: str
    parent: "Snapshot | None"
    n_tokens: int                                  # total tokens up to this boundary
    kv: dict                                       # (layer, attr) -> int8 K/V slice of this snapshot's own tokens
    states: dict                                   # (layer, attr, k) -> GDN state at the boundary (bf16, cloned)
    children: int = 0
    last_used: float = field(default_factory=time.monotonic)

    def nbytes(self) -> int:
        n = 0
        for t in list(self.kv.values()) + list(self.states.values()):
            n += t.nbytes() if hasattr(t, "q") else t.numel() * t.element_size()
        return n


def chunk_keys(chunks: list[str]) -> list[str]:
    """Cumulative content hashes: keys[i] identifies chunks[:i+1]."""
    keys, h = [], hashlib.sha1()
    for c in chunks:
        h.update(hashlib.sha1(c.encode()).digest())
        keys.append(h.copy().hexdigest())
    return keys


class PrefixCache:
    def __init__(self, backbone, max_bytes: int = 2 << 30):
        self.bb, self.max_bytes = backbone, max_bytes
        self.snaps: OrderedDict[str, Snapshot] = OrderedDict()
        self.bytes = 0
        self.template = None                        # cache structure (layer shells), taken from the first pass
        self.stats = {"hit_tokens": 0, "new_tokens": 0, "requests": 0}

    # ------------------------------------------------------------------ snapshots
    def _take(self, cache, key, parent, n_tokens):
        n0 = parent.n_tokens if parent else 0
        kv, st = {}, {}
        for li, layer in enumerate(cache.layers):
            for a in _KV:
                t = getattr(layer, a, None)
                if isinstance(t, torch.Tensor) and t.numel():
                    kv[(li, a)] = maybe_quantize(t[..., n0:n_tokens, :].contiguous(), self.bb.cfg.kv_quant == "int8")
            for a in _ST:
                d = getattr(layer, a, None)
                if isinstance(d, dict):
                    for k, t in d.items():
                        if isinstance(t, torch.Tensor):
                            st[(li, a, k)] = t.detach().clone()
                elif isinstance(d, torch.Tensor):
                    st[(li, a, None)] = d.detach().clone()
        s = Snapshot(key, parent, n_tokens, kv, st)
        if parent is not None:
            parent.children += 1
        self.snaps[key] = s
        self.bytes += s.nbytes()
        self._evict()
        return s

    def _evict(self):
        while self.bytes > self.max_bytes:
            leaf = next((s for s in self.snaps.values() if s.children == 0), None)   # OrderedDict: LRU first
            if leaf is None:
                return
            del self.snaps[leaf.key]
            self.bytes -= leaf.nbytes()
            if leaf.parent is not None:
                leaf.parent.children -= 1

    def _restore(self, snap: Snapshot) -> PackedCache:
        """PackedCache holding the full prefix up to `snap` (K/V deltas concatenated along the path)."""
        path = []
        s = snap
        while s is not None:
            path.append(s)
            s = s.parent
        path.reverse()
        tensors = {}
        for (li, a) in snap.kv:
            tensors[(li, a, None)] = torch.cat([materialize(p.kv[(li, a)]) for p in path], dim=-2)
        for key, t in snap.states.items():
            tensors[key] = t.clone()
        for p in path:                              # touch the whole path (LRU)
            self.snaps.move_to_end(p.key)
            p.last_used = time.monotonic()
        return PackedCache(self.template, tensors, snap.n_tokens)

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, content: list[str], closers: list[str], pin: int = 0, ctx=contextlib.nullcontext):
        """Encode one state given as content chunks + closers. Returns (PackedCache, mask (1,T), n_cached_tokens).
        pin: also make chunk boundary `pin` resumable (e.g. the pinned head of a retrieval-built state).
        ctx: context manager wrapping every backbone pass (the model's adapter state for its state pass)."""
        bb, dev = self.bb, self.bb.device
        keys = chunk_keys(content)
        toks = bb.tokenize(content + closers)
        ends = torch.cumsum(torch.tensor([len(t) for t in toks[:len(content)]]), 0).tolist()
        hit = next((i for i in range(len(keys) - 1, -1, -1) if keys[i] in self.snaps), -1)
        snap = self.snaps[keys[hit]] if hit >= 0 else None
        cache, n_cached = (self._restore(snap).unpack(1), snap.n_tokens) if snap is not None else (None, 0)
        stops = sorted({p for p in (pin - 1, len(content) - 1) if hit < p})       # chunk indices to snapshot at
        pos = hit + 1
        for stop in stops:                                                        # one pass per snapshot boundary
            ids = [t for c in toks[pos:stop + 1] for t in c]
            cache = self._run(ids, cache, ctx)
            if self.template is None:
                self._set_template(cache)
            snap = self._take(cache, keys[stop], snap, ends[stop])
            pos = stop + 1
        tail = [t for c in toks[len(content):] for t in c]
        if tail:                                                                  # closers: never cached
            cache = self._run(tail, cache, ctx)
        if self.template is None:
            self._set_template(cache)
        T = ends[-1] + len(tail) if content else len(tail)
        pc = PackedCache.pack(cache, kv_int8=bb.cfg.kv_quant == "int8", state_int8=bb.cfg.quantize_linear_state)
        self.stats["requests"] += 1
        self.stats["hit_tokens"] += n_cached
        self.stats["new_tokens"] += T - n_cached
        return pc, torch.ones(1, T, dtype=torch.bool, device=dev), n_cached

    def _run(self, ids, cache, ctx):
        x = torch.tensor([ids], dtype=torch.long, device=self.bb.device)
        with ctx():
            _, cache = self.bb(x, torch.ones_like(x, dtype=torch.bool), ("final",), past_key_values=cache,
                               use_cache=True)
        return cache

    def _set_template(self, cache):
        self.template = PackedCache.pack(cache, kv_int8=False).template
