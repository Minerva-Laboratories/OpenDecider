"""Text templates shared by every variant, so V0/V1/V2 and baselines see identical wording."""
from __future__ import annotations

import json
from typing import Any

YES_NO = ("yes", "no")


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return "State:\n" + state.strip() + "\n"
    return "State:\n" + json.dumps(state, ensure_ascii=False, separators=(",", ": ")) + "\n"


def question_text(prompt: str) -> str:
    return "\nQuestion: " + prompt.strip() + "\nOptions:"


def option_text(option: str) -> str:
    # Leading newline + dash so each option is a clean span; no index number, so the only
    # order signal a model gets is its position (and, if enabled, the slot embedding).
    return "\n- " + option.strip()


def question_answer_text(prompt: str) -> str:
    """Question row that ends with an answer cue: after reading the state, the final token's hidden state
    is the LM's prediction of the answer (used when options are NOT in the causal row)."""
    return "\nQuestion: " + prompt.strip() + "\nAnswer:"


def answer_option_text(option: str) -> str:
    """Option written as it would follow "Answer:" (leading space), for embedding-space matching."""
    return " " + option.strip()


def answer_row_prefix(prompt: str, options, list_cap: int = 16) -> str:
    """Answer-framed question row for branched reading: candidates listed in a CANONICAL (case-insensitive sorted)
    order, so the row never depends on the caller's option order, then an answer cue. Mirrors the zero-shot prompt
    whose option likelihoods are the LM's actual answer judgement. Listing is skipped above `list_cap` options."""
    head = "\nQuestion: " + prompt.strip()
    if 0 < len(options) <= list_cap:
        uniq = sorted({o.strip() for o in options}, key=lambda x: (x.lower(), x))      # duplicates listed once
        head += "\nOptions:" + "".join("\n- " + o for o in uniq)
    return head + "\nAnswer:"


def question_block(prompt: str, options, fmt: str = "list", list_cap: int = 16) -> str:
    """The question exactly as a model of the given row format reads it, ending at the answer slot."""
    if fmt == "answer":
        return answer_row_prefix(prompt, options, list_cap)
    return question_text(prompt) + "".join(option_text(o) for o in options) + "\nAnswer:"


def record_state_text(rec: dict, fmt: str = "list", list_cap: int = 16) -> str:
    """A dataset record's full state text: its `examples` (if any) rendered in the query schema, then the state.
    Examples answer the record's questions in order (answers[j] belongs to questions[j]; None = left empty)."""
    exs = rec.get("examples") or []
    qs = rec["questions"]
    prefix = solved_cases_prefix([{"state": ex["state"], "qa": [(q["prompt"], q["options"], a)
                                                                 for q, a in zip(qs, ex["answers"])]} for ex in exs],
                                 fmt, list_cap)
    return prefix + state_text(rec["state"])


def solved_cases_prefix(examples, fmt: str = "list", list_cap: int = 16) -> str:
    """In-context examples rendered in the SAME schema as the query: each example is a state followed by its
    question block(s) with the answer slot filled (labelled) or left EMPTY (unlabelled context). Placed before the
    real state, so it is part of the cached state prefix and shared by every question (O3/O4).
    examples: [{"state": ..., "qa": [(prompt, options, answer_or_None), ...]}, ...]"""
    out = []
    for ex in examples:
        block = state_text(ex["state"])
        for prompt, options, answer in ex["qa"]:
            block += question_block(prompt, options, fmt, list_cap) + (f" {answer}" if answer is not None else "") + "\n"
        out.append(block)
    return "".join(out) + "\nCurrent case:\n" if out else ""
