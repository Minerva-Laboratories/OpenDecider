import pytest
from fastapi.testclient import TestClient

from opendecider.api import create_app
from opendecider.decider import Decider
from tests.conftest import make


@pytest.fixture(scope="module")
def client(bb):
    return TestClient(create_app(Decider(make(bb, "v1"))))


def test_ok(client):
    r = client.post("/v1/decide", json={"state": "hello", "questions": {
        "a": {"type": "noul", "prompt": "Is this a greeting?"}}})
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["answers"]["a"]["probs"].values()) - 1) < 1e-6
    assert set(body) == {"answers", "timing_ms", "input_tokens", "state_cached_tokens"}


@pytest.mark.parametrize("bad", [
    {"questions": {"a": {"type": "noul", "prompt": "x"}}},                                  # no state
    {"state": "s", "questions": {}},                                                        # no questions
    {"state": "s", "questions": {"a": {"type": "choice", "prompt": "x", "options": ["one"]}}},
    {"state": "s", "questions": {"a": {"type": "generate", "prompt": "x"}}},                # no text generation
    {"state": "s", "questions": {"a": {"type": "noul", "prompt": "x", "options": ["y"]}}},  # extra field
    {"state": "s", "questions": {"a": {"type": "score", "prompt": "x", "levels": ["only"]}}},
    {"state": "", "questions": {"a": {"type": "noul", "prompt": "x"}}},
    {"state": "s", "questions": {"a": {"type": "noul", "prompt": "x"}}, "sampler": {"k": 0}},
])
def test_invalid_schema_is_400(client, bad):
    assert client.post("/v1/decide", json=bad).status_code == 400


def test_auth_logging_feedback_export(bb, tmp_path):
    c = TestClient(create_app(Decider(make(bb, "v1")), api_key="k" * 20, log_dir=str(tmp_path)))
    q = {"t": {"type": "choice", "prompt": "Topic?", "options": ["a", "b", "c"]}}
    assert c.get("/healthz").status_code == 200 and "OpenDecider" in c.get("/").text   # open without a key
    assert c.post("/v1/decide", json={"state": "s", "questions": q}).status_code == 401
    assert c.post("/v1/decide", json={"state": "s", "questions": q},
                  headers={"authorization": "Bearer wrong"}).status_code == 401
    h = {"authorization": "Bearer " + "k" * 20}
    ids = []
    for i in range(12):
        r = c.post("/v1/decide", json={"state": f"state {i}", "questions": q}, headers=h)
        assert r.status_code == 200
        assert set(r.json()) == {"answers", "timing_ms", "input_tokens", "state_cached_tokens"}   # body unchanged
        ids.append(r.headers["x-request-id"])
    fb = lambda **kw: c.post("/v1/feedback", json={"question": "t", **kw}, headers={"x-api-key": "k" * 20})
    for i, rid in enumerate(ids[:11]):
        assert fb(request_id=rid, label="abc"[i % 3]).status_code == 200
    assert fb(request_id=ids[0], label="b").status_code == 200                  # a later correction wins
    assert fb(request_id=ids[11], missing="d").status_code == 200               # right answer was not an option
    assert fb(request_id=ids[11]).status_code == 400                             # neither label nor missing
    assert fb(request_id="0" * 16, label="a").status_code == 404
    assert c.post("/v1/feedback", json={"request_id": ids[0], "question": "nope", "label": "a"},
                  headers=h).status_code == 400
    ex = c.get("/v1/feedback/export", params={"question": "t"}, headers=h).json()
    assert ex["n_examples"] == 11 and ex["n_answer_not_listed"] == 1 and ex["ready"]
    assert ex["calibrate_request"]["examples"][0] == {"state": "state 0", "label": "b"}
    assert ex["calibrate_request"]["question"] == q["t"]
    info = c.get("/v1/info", headers=h).json()
    assert info["requests"]["decide"] == 12 and info["auth"] and info["logging"]


def test_feedback_needs_log(client):
    assert client.post("/v1/feedback", json={"request_id": "x", "question": "t", "label": "a"}).status_code == 400
