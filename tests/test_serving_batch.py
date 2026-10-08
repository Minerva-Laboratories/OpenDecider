"""Batched serving: Decider.decide_many equals decide() per request, and concurrent HTTP requests share batches."""
import threading

import pytest
from fastapi.testclient import TestClient

from opendecider.api import create_app
from opendecider.decider import Decider
from tests.conftest import make
from tests.tiny import tiny_backbone

QS = [
    {"team": {"type": "choice", "prompt": "Which team?", "options": ["billing", "tech", "sales"]},
     "followup": {"type": "noul", "prompt": "Should {team} reply within a day?"},
     "risky": {"type": "composite", "prompt": "Risky?",
               "parts": {"money": {"type": "noul", "prompt": "Is money involved?"},
                         "sev": {"type": "score", "prompt": "Severity?", "levels": ["low", "high"]}},
               "combine": {"weights": {"money": 2.0, "sev=high": 1.5}, "bias": -1.0}}},
    {"sev": {"type": "score", "prompt": "How bad?", "levels": ["low", "mid", "high", "critical"]}},
    {"many": {"type": "choice", "prompt": "Which code?", "options": [f"code {i}" for i in range(300)]}},   # two-stage
]
STATES = ["I was charged twice and want a refund.", {"ticket": "app crashes on login", "tier": "gold"},
          "short", "Another much longer state about an invoice dispute that went on for several weeks " * 3]


def _models():
    yield "v1", make(tiny_backbone(), "v1")
    yield "v3-served", make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched",
                            v3_row_format="answer", slot_emb="none", v3_cross="question")


@pytest.mark.parametrize("name,model", list(_models()))
def test_decide_many_matches_decide(name, model, monkeypatch):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")         # compare like with like (no cache in either)
    d = Decider(model)
    reqs = [{"state": s, "questions": QS[i % len(QS)]} for i, s in enumerate(STATES)]
    many = d.decide_many(reqs)
    for req, got in zip(reqs, many):
        ref = d.decide(req)
        assert list(got["answers"]) == list(ref["answers"])
        for q, a in ref["answers"].items():
            b = got["answers"][q]
            assert b["value"] == a["value"]
            for k, p in a["probs"].items():
                assert abs(b["probs"][k] - p) < 2e-4, (name, q, k)
            assert abs(sum(b["probs"].values()) - 1) < 1e-6
        assert got["timing_ms"]["batch"] == len(reqs)


def test_decide_many_rejects_mixed_settings():
    d = Decider(make(tiny_backbone(), "v1"))
    with pytest.raises(ValueError):
        d.decide_many([{"state": "a", "questions": QS[1]},
                       {"state": "b", "questions": QS[1], "sampler": {"k": 2}}])


def test_concurrent_requests_are_batched_and_isolated():
    dec = Decider(make(tiny_backbone(), "v1"))
    gate, real = threading.Event(), dec.decide

    def slow_first(req):                     # hold the GPU so the next requests queue up behind it
        gate.wait(5)
        return real(req)
    dec.decide = slow_first
    c = TestClient(create_app(dec, max_batch=8))
    out = {}

    def call(i, body):
        out[i] = c.post("/v1/decide", json=body)
    bodies = [{"state": f"state {i}", "questions": QS[1]} for i in range(6)]
    bodies[3] = {"state": "x", "questions": {"t": {"type": "choice", "prompt": "p", "options": ["a", "b"],
                                                    "calibration": "no-such-profile"}}}   # fails alone
    ts = [threading.Thread(target=call, args=(0, bodies[0]))]
    ts[0].start()
    import time
    time.sleep(0.3)
    ts += [threading.Thread(target=call, args=(i, bodies[i])) for i in range(1, 6)]
    for t in ts[1:]:
        t.start()
    time.sleep(0.5)
    gate.set()
    for t in ts:
        t.join(30)
    assert out[3].status_code == 400                           # the bad request fails by itself...
    for i in (0, 1, 2, 4, 5):                                  # ...and the others still answer
        assert out[i].status_code == 200
        assert abs(sum(out[i].json()["answers"]["sev"]["probs"].values()) - 1) < 1e-6
    info = c.get("/v1/info").json()["batching"]
    assert info["batches"] >= 1 and info["largest_batch"] >= 2
