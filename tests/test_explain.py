from fastapi.testclient import TestClient

from conftest import make
from opendecider.api import create_app
from opendecider.decider import Decider
from opendecider.explain import _records, mask_options
from opendecider.formatting import state_text
from tiny import tiny_backbone


def test_records_remove_exactly_one_record():
    s = {"task": "rotate cert", "trace": {"steps": 11, "errors": 0}, "log": ["a", "b"]}
    recs = _records(s)
    paths = [r[0] for r in recs]
    assert ("task",) in paths and ("trace", "steps") in paths and ("log", 1) in paths
    for path, removed, txt in recs:
        assert txt in state_text(s) and len(state_text(removed)) < len(state_text(s))


def test_mask_options_word_boundaries():
    assert mask_options("Stop now; nonstop human review", ["stop", "human_review"]) == "[option] now; nonstop [option]"


def test_explain_endpoint_tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")
    m = make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_row_format="answer",
             slot_emb="none", v3_cross="question")
    c = TestClient(create_app(Decider(m, profiles_dir=str(tmp_path))))
    r = c.post("/v1/explain", json={"state": {"msg": "charged twice", "tier": "gold"},
                                    "question": {"type": "choice", "prompt": "Which team?", "options": ["billing", "tech"]},
                                    "samples": 2})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["decision"] in ("billing", "tech") and 0 <= out["faithfulness"] <= 1 and isinstance(out["evidence"], list)
    assert c.post("/v1/decide", json={"state": "x", "questions": {"a": {"type": "noul", "prompt": "ok?"}}}).status_code == 200
