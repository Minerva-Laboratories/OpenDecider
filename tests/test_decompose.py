import math

import pytest

from opendecider.decider import Decider
from opendecider.decompose import fit_logistic
from opendecider.schema import DecideRequest
from tests.conftest import make
from tests.test_models import OPTS, STATE


def test_composite_and_chaining(bb):
    d = Decider(make(bb, "v1"))
    res = d.decide({"state": STATE, "questions": {
        "team": {"type": "choice", "prompt": "Which team?", "options": OPTS},
        "followup": {"type": "noul", "prompt": "Should {team} reply within a day?"},
        "risky": {"type": "composite", "prompt": "Is this a risky ticket?",
                  "parts": {"money": {"type": "noul", "prompt": "Is money involved?"},
                            "sev": {"type": "score", "prompt": "Severity?", "levels": ["low", "high"]}},
                  "combine": {"weights": {"money": 2.0, "sev=high": 1.5}, "bias": -1.0}}}})
    a = res["answers"]
    assert list(a) == ["team", "followup", "risky"]
    assert res["timing_ms"]["waves"] == 2                     # followup depends on team
    r = a["risky"]
    z = -1.0 + 2.0 * r["parts"]["money"]["p_yes"] + 1.5 * r["parts"]["sev"]["probs"]["high"]
    assert abs(r["p_yes"] - 1 / (1 + math.exp(-z))) < 1e-12
    assert abs(sum(r["probs"].values()) - 1) < 1e-9


@pytest.mark.parametrize("qs", [
    {"a": {"type": "noul", "prompt": "Is {b} ok?"}, "b": {"type": "noul", "prompt": "Is {a} ok?"}},     # cycle
    {"a": {"type": "noul", "prompt": "Is {zzz} ok?"}},                                                   # unknown
    {"a": {"type": "composite", "prompt": "x", "parts": {"p": {"type": "noul", "prompt": "y"}},
           "combine": {"weights": {"q": 1.0}}}},                                                         # bad feature
])
def test_invalid_references_rejected(qs):
    with pytest.raises(Exception):
        DecideRequest.model_validate({"state": "s", "questions": qs})


def test_fit_logistic_recovers_rule():
    import random
    rng = random.Random(0)
    parts, labels = [], []
    for _ in range(400):
        a, b = rng.random(), rng.random()
        parts.append({"a": {"p_yes": a, "probs": {"yes": a, "no": 1 - a}}, "b": {"p_yes": b, "probs": {"yes": b, "no": 1 - b}}})
        labels.append(int(a + b > 1.0))
    spec = fit_logistic(parts, labels, ["a", "b"])
    assert spec["weights"]["a"] > 2 and spec["weights"]["b"] > 2 and spec["bias"] < -1


def test_marginal_chaining_mixes_over_referenced_answers(bb):
    """Default chaining: P(followup) = sum_t P(team = t) P(followup | team = t); "top" substitutes the top answer."""
    d = Decider(make(bb, "v1"))
    team = {"type": "choice", "prompt": "Which team?", "options": OPTS}
    res = d.decide({"state": STATE, "questions": {
        "team": team, "followup": {"type": "noul", "prompt": "Should {team} reply within a day?"},
        "top": {"type": "noul", "prompt": "Should {team} reply within a day?", "chain": "top"}}})
    a = res["answers"]
    pt = a["team"]["probs"]
    alone = {t: d.decide({"state": STATE, "questions": {"q": {"type": "noul", "prompt": f"Should {t} reply within a day?"}}})
             ["answers"]["q"]["p_yes"] for t in pt}
    want = sum(pt[t] * alone[t] for t in pt)
    f = a["followup"]
    assert len(f["chain"]["branches"]) == len(OPTS) and abs(f["chain"]["covered"] - 1) < 1e-9
    assert abs(f["p_yes"] - want) < 1e-4 and abs(sum(f["probs"].values()) - 1) < 1e-9
    assert "chain" not in a["top"] and abs(a["top"]["p_yes"] - alone[a["team"]["value"]]) < 1e-4
    many = d.decide_many([{"state": STATE, "questions": {"team": team, "followup": {"type": "noul",
                                                                                    "prompt": "Should {team} reply within a day?"}}}] * 2)
    assert all(abs(m["answers"]["followup"]["p_yes"] - want) < 1e-4 for m in many)


def test_chain_branches_cap_and_order():
    from opendecider import decider as D
    ans = {"a": {"probs": {"x": 0.7, "y": 0.29999, "z": 0.00001}}, "b": {"probs": {"u": 0.5, "v": 0.5}}}
    br = D.chain_branches(["a", "b"], ans)
    assert [w for _, w in br] == sorted([w for _, w in br], reverse=True)
    assert all(g["a"] != "z" for g, _ in br) and len(br) == 4          # z * 0.5 < CHAIN_MIN_WEIGHT
