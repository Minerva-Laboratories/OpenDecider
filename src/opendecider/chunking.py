"""Split a state's text into chunks at record boundaries, for prefix caching across requests (Tier 1).

The chunks concatenate to EXACTLY `formatting.state_text(state)`, so the model reads the same text; only where
the tokenizer is called changes (once per chunk). Chunks are single records (a JSON member or list item of a
container larger than max_chars, or a text line), never groups: a group's boundary would move when records are
appended, and cache snapshots are keyed by chunk boundaries. Trailing closers (`]`, `}` and the final newline) are returned
separately: a cache snapshot is taken before them, so a state that GROWS (events appended to a list, lines appended
to a log) still shares its whole cached prefix with the previous request.
"""
from __future__ import annotations

import json
from typing import Any

SEP = (",", ": ")                    # formatting.state_text's separators


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=SEP)


def _pieces(v: Any, max_chars: int) -> tuple[list[str], list[str]]:
    """(content pieces, closers) whose concatenation is _dumps(v). Containers larger than max_chars are split
    between members; the container's closing bracket is a closer, nested closers accumulate innermost first."""
    s = _dumps(v)
    if len(s) <= max_chars or not isinstance(v, (dict, list)) or not v:
        return [s], []
    items = list(v.items()) if isinstance(v, dict) else [(None, x) for x in v]
    out, closers = [], []
    for i, (k, x) in enumerate(items):
        head = ("{" if isinstance(v, dict) else "[") if i == 0 else ","
        if k is not None:
            head += _dumps(k) + SEP[1]
        sub, sub_close = _pieces(x, max_chars)
        if i < len(items) - 1:                 # nested closers of a non-final member are ordinary content
            sub, sub_close = sub[:-1] + [sub[-1] + "".join(sub_close)], []
        out += [head + sub[0]] + sub[1:]
        closers = sub_close
    return out, closers + ["}" if isinstance(v, dict) else "]"]


def text_chunks(text: str) -> list[str]:
    """Plain text split after newlines (one chunk per line); concatenation == text."""
    return text.splitlines(keepends=True)


def state_chunks(state: Any, max_chars: int = 256) -> tuple[list[str], list[str]]:
    """(content chunks, closers) with "".join(content + closers) == formatting.state_text(state)."""
    if isinstance(state, str):
        body = state.strip()
        return text_chunks("State:\n" + body), ["\n"]
    pieces, closers = _pieces(state, max_chars)
    return ["State:\n" + pieces[0]] + pieces[1:], ["".join(closers) + "\n"]
