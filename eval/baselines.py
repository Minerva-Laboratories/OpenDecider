"""Baselines from docs/SPEC.md section 7.3.

Baseline 1 -- frozen backbone, constrained label scoring with the *causal LM head*:
  (a) `first`: logit of each option's first continuation token after "\nAnswer:", renormalised
      over the options (softmax over K first-token logits);
  (b) `seq`:   full option continuation log-prob, length-normalised (mean per-token log-prob),
      then softmax over options.

`Backbone.load` deletes `lm_head`; Qwen3.5 ties it to `embed_tokens`, so logits are
`h @ E[tok].T`. We only ever gather the embedding rows we need (and, for the `seq` log-softmax
normaliser, stream over the vocabulary in chunks) instead of materialising the full
(positions x vocab) projection or the dequantised int8 table.

The scorer exposes the same `encode_states(texts)` / `run(questions, mem)` interface as the
decision models (logits (N,Kmax) with NEG padding, opt_mask), so eval/probes.py and
eval/evaluate.py treat it like any other model. No text is generated: we only score the
caller's options.

Baseline 3 (plan only, NOT downloaded here): zero-shot NLI classifier, e.g.
`facebook/bart-large-mnli` (pin a revision; MIT licence). For each option build the hypothesis
"The answer to '<prompt>' is <option>." with premise = state text, take the entailment logit
per option and softmax over options (the standard zero-shot-classification recipe). It should
implement the same encode_states/run interface as `LabelScorer`. Needs ~1.6 GB of disk: check
`opendecider.guards.require_free_gb` before downloading, and record the model in data/MANIFEST.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from opendecider.backbone import Backbone, PackedCache
from opendecider.batching import Question
from opendecider.formatting import option_text, question_text
from opendecider.heads import NEG
from opendecider.models import DecisionOutput

ANSWER_TEXT = "\nAnswer:"


@dataclass
class _Cfg:
    """Minimal stand-in for ModelConfig fields read by generic adapters."""
    variant: str
    noul_mode: str = "choice"
    slot_emb: str = "n/a"


@dataclass
class LabelMemory:
    state_tokens: list[list[int]]
    n_tokens: int = 0
    extra: dict = field(default_factory=dict)


def _emb_rows(emb, idx) -> torch.Tensor:
    """Rows of the (possibly int8) embedding table as float32, without dequantising the rest."""
    if hasattr(emb, "qweight"):
        return emb.qweight[idx].float() * emb.scale[idx].float()
    return emb.weight[idx].float()


class LabelScorer(torch.nn.Module):
    """Baseline 1: frozen causal LM, constrained scoring of the caller's options."""

    def __init__(self, backbone: Backbone, mode: str = "first", answer_text: str = ANSWER_TEXT,
                 vocab_chunk: int = 32768, max_rows: int = 32):
        super().__init__()
        if mode not in ("first", "seq"):
            raise ValueError(mode)
        tie = getattr(backbone.lm.config, "tie_word_embeddings", True)
        if not tie:
            raise ValueError("LabelScorer needs tied embeddings (lm_head was deleted by Backbone.load)")
        self.backbone = backbone
        self.mode = mode
        self.answer_text = answer_text
        self.vocab_chunk = vocab_chunk
        self.max_rows = max_rows
        self.cfg = _Cfg(variant=f"label_{mode}")

    @property
    def emb(self):
        return self.backbone.lm.embed_tokens

    # ------------------------------------------------------------------ interface
    def encode_states(self, state_texts: Sequence[str], noise_std: float = 0.0) -> LabelMemory:
        toks = self.backbone.tokenize(state_texts)
        return LabelMemory(toks, sum(map(len, toks)))

    def with_noise(self, mem, std):
        return mem

    def _prefix(self, q: Question, mem: LabelMemory) -> list[int]:
        pieces = [question_text(q.prompt)] + [option_text(o) for o in q.options] + [self.answer_text]
        return list(mem.state_tokens[q.state_idx]) + [t for p in self.backbone.tokenize(pieces) for t in p]

    def _continuations(self, q: Question) -> list[list[int]]:
        conts = self.backbone.tokenize([" " + o.strip() for o in q.options])
        return [c if c else [self.backbone.pad_id] for c in conts]

    @torch.no_grad()
    def run(self, questions: Sequence[Question], mem: LabelMemory):
        N, K = len(questions), max(len(q.options) for q in questions)
        dev = self.backbone.device
        logits = torch.full((N, K), NEG, dtype=torch.float32, device=dev)
        opt_mask = torch.zeros(N, K, dtype=torch.bool, device=dev)
        for n, q in enumerate(questions):
            opt_mask[n, : len(q.options)] = True
        if self.mode == "first":
            self._run_first(questions, mem, logits)
        else:
            for n, q in enumerate(questions):
                logits[n, : len(q.options)] = self._score_seq(q, mem)
        return None, DecisionOutput(logits, opt_mask)

    def forward(self, questions, mem):
        return self.run(questions, mem)[1]

    # ------------------------------------------------------------------ (a) first token
    def _run_first(self, questions, mem, logits):
        bb = self.backbone
        for s in range(0, len(questions), self.max_rows):
            chunk = questions[s:s + self.max_rows]
            seqs = [self._prefix(q, mem) for q in chunk]
            ids, mask = bb.pad(seqs)
            f, _ = bb(ids, mask)
            last = mask.long().sum(1) - 1
            h = f[torch.arange(len(chunk), device=f.device), last].float()      # (B,d)
            for b, q in enumerate(chunk):
                first = torch.tensor([c[0] for c in self._continuations(q)], device=f.device)
                logits[s + b, : len(q.options)] = _emb_rows(self.emb, first) @ h[b]

    # ------------------------------------------------------------------ (b) full sequence
    def _logsumexp_vocab(self, h: torch.Tensor) -> torch.Tensor:
        """log sum_v exp(h . E_v) for h (P,d), streaming over vocab chunks."""
        V = self.emb.num_embeddings
        out = torch.full((h.shape[0],), -float("inf"), device=h.device)
        for s in range(0, V, self.vocab_chunk):
            z = h @ _emb_rows(self.emb, slice(s, min(V, s + self.vocab_chunk))).T
            out = torch.logaddexp(out, torch.logsumexp(z, 1))
        return out

    def _score_seq(self, q: Question, mem: LabelMemory) -> torch.Tensor:
        bb = self.backbone
        prefix = self._prefix(q, mem)
        ids = torch.tensor([prefix], device=bb.device)
        f, cache = bb(ids, torch.ones_like(ids, dtype=torch.bool), use_cache=True)
        h0 = f[0, -1].float()
        packed = PackedCache.pack(cache, kv_int8=False, state_int8=False)
        conts = self._continuations(q)
        scores = torch.empty(len(conts), device=bb.device)
        for s in range(0, len(conts), self.max_rows):
            chunk = conts[s:s + self.max_rows]
            # hidden predicting token j of an option: h0 for j=0, else output at continuation pos j-1
            hs, tok, owner = [], [], []
            multi = [c for c in chunk if len(c) > 1]
            if multi:
                cids, cmask = bb.pad([c[:-1] for c in multi])
                fc, _ = bb(cids, cmask, past_key_values=packed.unpack(len(multi)), use_cache=True)
            mi = 0
            for b, c in enumerate(chunk):
                hs.append(h0[None]); tok.append(c[0]); owner.append(b)
                if len(c) > 1:
                    hs.append(fc[mi, : len(c) - 1].float()); tok.extend(c[1:]); owner.extend([b] * (len(c) - 1))
                    mi += 1
            H = torch.cat(hs, 0)
            T = torch.tensor(tok, device=H.device)
            lp = (H * _emb_rows(self.emb, T)).sum(-1) - self._logsumexp_vocab(H)
            own = torch.tensor(owner, device=H.device)
            tot = torch.zeros(len(chunk), device=H.device).index_add_(0, own, lp)
            cnt = torch.zeros(len(chunk), device=H.device).index_add_(0, own, torch.ones_like(lp))
            scores[s:s + len(chunk)] = tot / cnt
        return scores


BASELINES = {"label_first": "first", "label_seq": "seq"}


def build_baseline(name: str, backbone: Backbone) -> LabelScorer:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    return LabelScorer(backbone, BASELINES[name])
