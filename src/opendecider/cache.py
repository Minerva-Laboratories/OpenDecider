"""Flat int8 token-feature cache + vectorised segment layout helpers.

Everything per-item is done with tensor ops: one gather to fetch all tokens of many cached texts,
one scatter to lay segments out into a padded (N, L, d) batch. Lengths live on the CPU (known from
cache metadata), so assembling a batch needs no GPU->CPU synchronisation.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import torch


def segment_layout(lens: torch.Tensor, group: torch.Tensor, n_groups: int):
    """Lay out variable-length segments, grouped (consecutively) into rows.

    lens:  (M,) CPU long, tokens per segment;  group: (M,) CPU long, row of each segment (non-decreasing).
    Returns (row, col, seg, pos, row_len) for every token: row/col = destination, seg = segment id,
    pos = position inside its segment, row_len = (n_groups,) total tokens per row.
    """
    M = lens.numel()
    seg = torch.repeat_interleave(torch.arange(M), lens)
    start = torch.cumsum(lens, 0) - lens                                   # global start of each segment
    pos = torch.arange(int(lens.sum())) - start[seg]
    row_len = torch.zeros(n_groups, dtype=torch.long).index_add_(0, group, lens)
    row_start = torch.cumsum(row_len, 0) - row_len
    col = start[seg] - row_start[group[seg]] + pos
    return group[seg], col, seg, pos, row_len


class TokenCache:
    """key -> contiguous int8 token rows in one growable buffer (per-token absmax scales)."""

    def __init__(self, d: int, device, max_tokens: int = 50_000_000):
        self.d, self.device, self.max_tokens = d, device, max_tokens
        self.q = torch.empty(0, d, dtype=torch.int8, device=device)
        self.s = torch.empty(0, 1, dtype=torch.float32, device=device)
        self.n = 0
        self.index: dict[Hashable, tuple[int, int]] = {}

    def __contains__(self, k):
        return k in self.index

    def __len__(self):
        return len(self.index)

    def clear(self):
        self.index.clear()
        self.n = 0

    def _reserve(self, extra: int):
        need = self.n + extra
        if need > self.q.shape[0]:
            cap = max(need, 2 * self.q.shape[0], 4096)
            q = torch.empty(cap, self.d, dtype=torch.int8, device=self.device)
            s = torch.empty(cap, 1, dtype=torch.float32, device=self.device)
            q[: self.n] = self.q[: self.n]
            s[: self.n] = self.s[: self.n]
            self.q, self.s = q, s

    def add(self, keys: Sequence[Hashable], feats: torch.Tensor, mask: torch.Tensor, lens: Sequence[int]):
        """feats (B,L,d) padded, mask (B,L) valid tokens, lens python ints per row (same order as keys)."""
        tok = feats[mask].float()                                            # row-major: row 0 tokens, row 1, ...
        if self.n + tok.shape[0] > self.max_tokens:
            self.clear()
        self._reserve(tok.shape[0])
        scale = tok.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
        self.q[self.n:self.n + tok.shape[0]] = torch.round(tok / scale).clamp_(-127, 127).to(torch.int8)
        self.s[self.n:self.n + tok.shape[0]] = scale
        off = self.n
        for k, L in zip(keys, lens):
            self.index[k] = (off, L)
            off += L
        self.n = off

    def lookup(self, keys: Sequence[Hashable]):
        """-> (flat features (sum L, d) float32 on device, lens (M,) CPU long)."""
        meta = torch.tensor([self.index[k] for k in keys], dtype=torch.long)
        offs, lens = meta[:, 0], meta[:, 1]
        seg = torch.repeat_interleave(torch.arange(len(keys)), lens)
        src = (offs[seg] + torch.arange(int(lens.sum())) - (torch.cumsum(lens, 0) - lens)[seg]).to(self.device)
        return self.q[src].float() * self.s[src], lens


def encode_missing(cache: TokenCache, backbone, keys: Sequence[str], texts: Sequence[str] | None = None,
                   layers=None, budget: int = 8192, post=None) -> None:
    """Encode texts whose keys are missing from `cache` with the frozen backbone and add them.
    One batched tokenizer call; length-sorted chunks under a padded-token budget (little padding).
    `post(feats)` optionally transforms features (e.g. a layer mix) before caching."""
    texts = list(texts) if texts is not None else list(keys)
    miss = {}
    for k, t in zip(keys, texts):
        if k not in cache and k not in miss:
            miss[k] = t
    if not miss:
        return
    mk, mt = list(miss), list(miss.values())
    toks = backbone.tokenize(mt)
    order = sorted(range(len(mk)), key=lambda i: len(toks[i]))
    s = 0
    while s < len(order):
        e, L = s + 1, len(toks[order[s]])
        while e < len(order) and (e - s + 1) * max(L, len(toks[order[e]])) <= budget:
            L = max(L, len(toks[order[e]]))
            e += 1
        idx = order[s:e]
        ids, mask = backbone.pad([toks[i] for i in idx])
        with torch.no_grad():
            f, _ = backbone(ids, mask, layers=layers or backbone.cfg.feature_layers)
            if post is not None:
                f = post(f)
        cache.add([mk[i] for i in idx], f, mask, [len(toks[i]) for i in idx])
        s = e
