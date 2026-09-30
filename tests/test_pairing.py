"""Each question must see ITS OWN record's state (real data, tiny backbone, cached path)."""
import json
import random

import torch

from opendecider.backbone import StateMemory
from opendecider.batching import state_texts
from opendecider.formatting import question_text
from opendecider.train import DEFAULT_TRAIN, state_features, to_questions
import opendecider.train as T
from tests.conftest import make


def test_questions_see_their_own_state_in_v3_context(bb):
    recs = [json.loads(l) for _, l in zip(range(6), open("data/synthetic/train.jsonl"))]
    random.Random(0).shuffle(recs)
    T._STATE_CACHE.clear()
    t = dict(DEFAULT_TRAIN, max_state_tokens=128)
    seqs = [s[:128] for s in bb.tokenize(state_texts([r["state"] for r in recs]))]
    ids, mask = bb.pad(seqs)
    state_features(bb, recs, seqs, ids, mask, t)                  # fill cache
    feats = state_features(bb, recs, seqs, ids, mask, t)          # served from cache
    qs = [q for i, r in enumerate(recs) for q in to_questions(r, i, None, 99)]
    m = make(bb, "v3")
    sm = StateMemory(feats, mask, 0); sm.lens = torch.tensor([len(s) for s in seqs])
    mem = m.prepare_memory(sm, cache=False)
    captured = {}
    orig = m.layers[0].forward
    m.layers[0].forward = lambda x, c, *a: (captured.setdefault("c_in", c), orig(x, c, *a))[1]
    m.run(qs, mem)
    # rebuild each question's expected context from a FRESH encoding of its own record
    for n, q in enumerate(qs):
        own = seqs[q.state_idx]
        with torch.no_grad():
            f, _ = bb(*bb.pad([own]))
        qf, _ = bb(*bb.pad(bb.tokenize([question_text(q.prompt)])))
        exp = torch.cat([f[0], qf[0]]).float()
        L = exp.shape[0]
        got_raw = torch.cat([feats[q.state_idx, :len(own)], m.text_feats([question_text(q.prompt)])[0]]).float()
        assert torch.allclose(got_raw, exp, atol=0.05 * exp.abs().max()), f"question {n} paired with wrong state"
        # and questions from different records must differ in context
    ctx = captured["c_in"]
    a = [n for n, q in enumerate(qs) if q.state_idx == 0][0]
    b = [n for n, q in enumerate(qs) if q.state_idx == 1][0]
    assert not torch.allclose(ctx[a, :8], ctx[b, :8])
