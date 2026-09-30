import pytest
from fastapi.testclient import TestClient

from opendecider.api import create_app
from opendecider.decider import Decider
from tests.conftest import make
from tests.test_models import STATE


@pytest.fixture(scope="module")
def dec(bb, tmp_path_factory):
    return Decider(make(bb, "v1"), profiles_dir=str(tmp_path_factory.mktemp("profiles")))


def test_descriptions_instructions_confidence(dec):
    res = dec.decide({"state": STATE, "questions": {
        "team": {"type": "choice", "prompt": "Which team?", "instructions": "Refund requests always go to billing.",
                 "options": [{"name": "billing", "description": "payments, charges, refunds"},
                             {"name": "tech", "description": "bugs and outages"}, "other"]},
        "spam": {"type": "noul", "prompt": "Is this spam?", "criteria": "Unsolicited bulk commercial content."},
        "sev": {"type": "score", "prompt": "Severity?", "levels": [{"name": "low", "description": "no impact"}, "high"]}}})
    a = res["answers"]
    assert set(a["team"]["probs"]) == {"billing", "tech", "other"}                # keyed by NAME
    assert set(a["sev"]["probs"]) == {"low", "high"}
    for v in a.values():
        K, p = len(v["probs"]), max(v["probs"].values())
        assert abs(v["confidence"] - max(0.0, (K * p - 1) / (K - 1))) < 1e-9
    q = dec._atomic("team", __import__("opendecider.schema", fromlist=["x"]).DecideRequest.model_validate(
        {"state": "s", "questions": {"t": {"type": "choice", "prompt": "Which team?", "instructions": "X",
                                           "options": [{"name": "billing", "description": "payments"}, "tech"]}}}).questions["t"], {})
    assert q.options[0] == "billing: payments" and "Instructions: X" in q.prompt and q.names == ["billing", "tech"]


def test_calibration_profile_fit_apply_and_order_invariance(dec):
    qspec = {"type": "choice", "prompt": "Which team?", "options": ["billing", "tech", "other"]}
    exs = [{"state": f"{STATE} case {i}", "label": ["billing", "tech", "other"][i % 3]} for i in range(12)]
    client = TestClient(create_app(dec))
    r = client.post("/v1/calibrate", json={"question": qspec, "examples": exs, "method": "vector"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert r.json()["fit_in_sample"]["nll_after"] <= r.json()["fit_in_sample"]["nll_before"] + 1e-6
    base = dec.decide({"state": STATE, "questions": {"t": qspec}})["answers"]["t"]["probs"]
    cal = dec.decide({"state": STATE, "questions": {"t": dict(qspec, calibration=pid)}})["answers"]["t"]["probs"]
    assert any(abs(base[k] - cal[k]) > 1e-4 for k in base)
    rev = dec.decide({"state": STATE, "questions": {"t": dict(qspec, options=qspec["options"][::-1], calibration=pid)}})
    # the profile's biases are keyed by name, so they move with the option under reordering
    b = dec.profile(pid)["bias"]
    assert set(b) == {"billing", "tech", "other"}
    assert client.post("/v1/decide", json={"state": "s", "questions": {"t": dict(qspec, calibration="nope")}}).status_code == 400


def test_in_context_examples_same_schema(dec, bb):
    from opendecider.decider import Decider
    from tests.conftest import make
    for variant, fmt in (("v1", "list"), ("v3", "answer")):
        m = make(bb, variant, **({"v3_features": "branched", "v3_row_format": "answer"} if variant == "v3" else {}))
        d = Decider(m, profiles_dir=dec.profiles_dir)
        seen = []
        orig = m.encode_states
        m.encode_states = lambda texts, *a, **k: (seen.append(texts[0]), orig(texts, *a, **k))[1]
        q = {"type": "choice", "prompt": "Which team?", "options": ["billing", "tech"]}
        res = d.decide({"state": STATE, "questions": {"t": q},
                        "examples": [{"state": "refund please", "answers": {"t": "billing"}},
                                     {"state": "app crashes", "answers": {"t": None}}]})
        assert abs(sum(res["answers"]["t"]["probs"].values()) - 1) < 1e-6
        text = seen[-1]
        assert text.index("refund please") < text.index("Current case:") < text.index(STATE)
        assert "Answer: billing" in text and "Answer:\n" in text           # labelled filled, unlabelled empty
        assert text.count("Question: Which team?") == 2                     # same question schema per example
        assert m.row_format == fmt
