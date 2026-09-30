import json

import pytest
import torch

from conftest import make
from opendecider.chunking import state_chunks, text_chunks
from opendecider.decider import Decider
from opendecider.formatting import state_text
from tiny import tiny_backbone

STATES = [
    {"customer": {"name": "Ana", "tier": "gold"}, "events": [{"t": i, "msg": f"event {i} " + "x" * 40} for i in range(30)]},
    [{"id": i, "v": "y" * 50} for i in range(40)],
    "line one\nline two\n" + "\n".join(f"log {i}: " + "z" * 30 for i in range(50)),
    {"a": 1, "b": [1, 2, {"c": [3, 4]}], "d": "ñandú"},
    {},
    [],
    "short",
]


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("max_chars", [16, 64, 1024])
def test_chunks_concatenate_to_state_text(state, max_chars):
    content, closers = state_chunks(state, max_chars)
    assert "".join(content + closers) == state_text(state)
    assert all(content)


def test_appended_list_shares_all_content_chunks():
    s1 = {"customer": {"name": "Ana"}, "events": [{"t": i, "m": "e" * 60} for i in range(10)]}
    s2 = json.loads(json.dumps(s1)); s2["events"].append({"t": 10, "m": "new"})
    c1, _ = state_chunks(s1, 128)
    c2, _ = state_chunks(s2, 128)
    assert c2[:len(c1) - 1] == c1[:-1] and "".join(c2).startswith("".join(c1))


def test_text_chunks_roundtrip():
    t = "a\nbb\n\nccc"
    assert "".join(text_chunks(t)) == t


QS = {"route": {"type": "choice", "prompt": "Which team?", "options": ["billing", "technical", "refund"]},
      "urgent": {"type": "noul", "prompt": "Is it urgent?"}}


def _probs(out):
    return torch.tensor([p for a in out["answers"].values() for p in a["probs_list"]])


@pytest.fixture(scope="module")
def model():
    return make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_row_format="answer",
                slot_emb="none", v3_cross="question")


def test_cache_hit_and_append_match_cold_encoding(model, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "64")
    s1 = {"customer": {"name": "Ana", "tier": "gold"}, "events": [{"t": i, "msg": "late delivery " * 3} for i in range(12)]}
    s2 = json.loads(json.dumps(s1)); s2["events"].append({"t": 12, "msg": "asks for a refund"})
    warm = Decider(model, profiles_dir=str(tmp_path))
    a = warm.decide({"state": s1, "questions": QS})
    assert a["state_cached_tokens"] == 0
    b = warm.decide({"state": s1, "questions": QS})                          # exact repeat: all content cached
    c = warm.decide({"state": s2, "questions": QS})                          # appended event: prefix cached
    assert b["state_cached_tokens"] > 0 and c["state_cached_tokens"] >= b["state_cached_tokens"] * 0.5
    cold = Decider(model, profiles_dir=str(tmp_path))                        # fresh cache: cold for s2
    c0 = cold.decide({"state": s2, "questions": QS})
    assert c0["state_cached_tokens"] == 0
    tol = 2e-2                                                               # int8 K/V at different boundaries
    assert (_probs(a) - _probs(b)).abs().max() < tol
    assert (_probs(c) - _probs(c0)).abs().max() < tol


def test_cache_matches_uncached_path(model, monkeypatch, tmp_path):
    s = {"ticket": "charged twice", "history": [{"n": i, "note": "called support " * 2} for i in range(8)]}
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")
    ref = Decider(model, profiles_dir=str(tmp_path)).decide({"state": s, "questions": QS})
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "64")
    got = Decider(model, profiles_dir=str(tmp_path)).decide({"state": s, "questions": QS})
    # same text; only the tokenizer boundaries (per chunk) can differ
    assert (_probs(ref) - _probs(got)).abs().max() < 5e-2


def test_eviction_keeps_ancestors(model, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "1")
    d = Decider(model, profiles_dir=str(tmp_path))
    for n in range(4):
        d.decide({"state": {"events": [{"i": i, "m": "w" * 200} for i in range(20 + 10 * n)]}, "questions": QS})
    pc = d._prefix_cache()
    assert pc.bytes <= pc.max_bytes or all(s.children for s in pc.snaps.values())
    for s in pc.snaps.values():                                              # every parent is still present
        assert s.parent is None or s.parent.key in pc.snaps


def test_backbone_embedder_normalised_and_order_free():
    from opendecider.embed import BackboneEmbedder
    e = BackboneEmbedder(tiny_backbone(layers=("final",)), layer=4)
    texts = ["refund for order 1042", "app crashes on login", "a", "refund for order 1042"]
    v = e(texts)
    assert v.shape == (4, 64) and abs(float((v ** 2).sum(1).mean()) - 1) < 1e-4
    assert abs(float(v[0] @ v[3]) - 1) < 1e-4                            # identical texts, different batch slots
    v2 = e(texts[::-1])
    assert abs(float(v2[::-1][1] @ v[1]) - 1) < 1e-3


def test_int4_cache_end_to_end_close_to_unquantized(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "64")
    s = {"ticket": "charged twice", "history": [{"n": i, "note": "called support " * 2} for i in range(8)]}
    out = {}
    for kv in ("none", "int4"):
        m = make(tiny_backbone(kv_quant=kv, layers=(4, "final")), "v3", v3_features="branched",
                 v3_row_format="answer", slot_emb="none", v3_cross="question")
        d = Decider(m, profiles_dir=str(tmp_path))
        d.decide({"state": s, "questions": QS})
        out[kv] = d.decide({"state": s, "questions": QS})                  # second call: served from the int4 cache
    assert out["int4"]["state_cached_tokens"] > 0
    assert (_probs(out["none"]) - _probs(out["int4"])).abs().max() < 5e-2
