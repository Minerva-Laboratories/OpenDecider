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
