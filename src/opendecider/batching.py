"""Question representation and token packing shared by all variants.

Pieces (state, question, each option) are tokenized separately and concatenated, so every
option has an exact token span regardless of tokenizer merges across boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .formatting import YES_NO, option_text, question_text, state_text


@dataclass
class Question:
    prompt: str
    options: list[str]
    type: str = "choice"            # choice | noul | score
    label: int | None = None        # index of the correct option (training/eval)
    state_idx: int = 0              # which state (in the batch) this question reads
    name: str = ""
    family: str = ""                # task family (for OOD splits)

    @classmethod
    def noul(cls, prompt: str, label_yes: bool | None = None, **kw) -> "Question":
        lab = None if label_yes is None else (0 if label_yes else 1)
        return cls(prompt, list(YES_NO), "noul", lab, **kw)

    @classmethod
    def score(cls, prompt: str, levels: Sequence[str], label: int | None = None, **kw) -> "Question":
        return cls(prompt, list(levels), "score", label, **kw)


@dataclass
class Packed:
    ids: torch.Tensor            # (R,L)
    mask: torch.Tensor           # (R,L) bool
    row_question: torch.Tensor   # (R,) question index of each row
    q_spans: torch.Tensor        # (N,3) (row,start,end) span pooled into the question vector
    opt_spans: torch.Tensor      # (M,3)
    opt_n: torch.Tensor          # (M,) question index of each option
    opt_k: torch.Tensor          # (M,) option position 0..K-1
    opt_mask: torch.Tensor       # (N,Kmax) bool
    slots: torch.Tensor          # (N,Kmax) 1..K, 0 = pad
    q_state: torch.Tensor        # (N,) state index
    labels: torch.Tensor | None  # (N,) or None
    n_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def to(self, device) -> "Packed":
        kw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        return Packed(**kw)

    @property
    def row_state(self) -> torch.Tensor:
        return self.q_state[self.row_question]


def pack(tok_fn, questions: Sequence[Question], mode: str, pad_id: int,
         state_prefix: Sequence[Sequence[int]] | None = None) -> Packed:
    """mode: joint (one row per question: [q][o1]..[oK]); independent (one row per option: [q][ok]);
    separate (question rows then one row per option, no shared context; used by V1).
    `state_prefix[s]` token ids are prepended to rows of state s (V0 without a prefix cache)."""
    rows: list[list[int]] = []
    row_q: list[int] = []
    q_spans, opt_spans, opt_n, opt_k = [], [], [], []
    N = len(questions)
    Kmax = max(len(q.options) for q in questions)
    texts = []
    for q in questions:
        texts.append(question_text(q.prompt))
        texts.extend(option_text(o) for o in q.options)
    toks = tok_fn(texts)
    it = iter(toks)
    for n, q in enumerate(questions):
        qt = next(it)
        ots = [next(it) for _ in q.options]
        pre = list(state_prefix[q.state_idx]) if state_prefix is not None else []
        if mode == "joint":
            r = len(rows)
            seq = pre + qt
            q_spans.append((r, len(pre), len(seq)))
            for k, ot in enumerate(ots):
                opt_spans.append((r, len(seq), len(seq) + len(ot))); opt_n.append(n); opt_k.append(k)
                seq = seq + ot
            rows.append(seq); row_q.append(n)
        elif mode == "independent":
            for k, ot in enumerate(ots):
                r = len(rows)
                seq = pre + qt
                if k == 0:
                    q_spans.append((r, len(pre), len(seq)))
                opt_spans.append((r, len(seq), len(seq) + len(ot))); opt_n.append(n); opt_k.append(k)
                rows.append(seq + ot); row_q.append(n)
        elif mode == "separate":
            r = len(rows)
            q_spans.append((r, 0, len(qt)))
            rows.append(qt); row_q.append(n)
            for k, ot in enumerate(ots):
                opt_spans.append((len(rows), 0, len(ot))); opt_n.append(n); opt_k.append(k)
                rows.append(ot); row_q.append(n)
        else:
            raise ValueError(mode)
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = True
    opt_mask = torch.zeros(N, Kmax, dtype=torch.bool)
    slots = torch.zeros(N, Kmax, dtype=torch.long)
    for n, q in enumerate(questions):
        opt_mask[n, : len(q.options)] = True
        slots[n, : len(q.options)] = torch.arange(1, len(q.options) + 1)
    labels = None
    if all(q.label is not None for q in questions):
        labels = torch.tensor([q.label for q in questions])
    return Packed(ids, mask, torch.tensor(row_q), torch.tensor(q_spans), torch.tensor(opt_spans),
                  torch.tensor(opt_n), torch.tensor(opt_k), opt_mask, slots,
                  torch.tensor([q.state_idx for q in questions]), labels, int(mask.sum()))


def state_texts(states: Sequence) -> list[str]:
    return [state_text(s) for s in states]
