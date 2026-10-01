import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendecider.api import create_app
from opendecider.conformal import coverage, pack, prediction_set, scores, threshold
from opendecider.decider import Decider
from tests.conftest import make


def _sample(rng, n, K=5):
    """Calibrated synthetic model: y is drawn from p itself, so p is exactly calibrated."""
    P = rng.dirichlet(np.ones(K) * 0.5, n)
    y = np.array([rng.choice(K, p=p) for p in P])
    return P, y


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_split_conformal_covers(method):
    rng = np.random.default_rng(0)
    Pc, yc = _sample(rng, 2000)
    Pt, yt = _sample(rng, 4000)
    for alpha in (0.05, 0.1, 0.2):
        r = coverage(Pt, yt, scores(Pc, yc, method), alpha, method)
        assert r["coverage"] >= 1 - alpha - 0.02, (alpha, r)          # marginal guarantee (+ sampling slack)
        if method == "lac":                                             # deterministic APS over-covers by design
            assert r["coverage"] <= 1 - alpha + 0.04
        assert 1 <= r["mean_size"] + r["empty"] and r["mean_size"] < 5


def test_threshold_finite_sample_and_sets():
    assert threshold([0.1, 0.2, 0.3], 0.1) == float("inf")              # n too small for 90%: every option
    assert threshold(np.linspace(0, 1, 99), 0.1) == pytest.approx(np.linspace(0, 1, 99)[89])
    p = [0.6, 0.3, 0.1]
    assert prediction_set(p, 0.45, "lac") == [0]                        # 1 - p <= q
    assert prediction_set(p, 0.75, "lac") == [0, 1]
    assert prediction_set(p, 0.05, "lac") == []                         # LAC can be empty
    assert prediction_set(p, 0.85, "aps") == [0, 1]                     # mass 0.6 < 0.85 <= 0.9
    assert prediction_set(p, 0.01, "aps") == [0]                        # APS is never empty


def test_pack_groups_by_type_and_pads():
    out = pack([np.array([0.7, 0.3]), np.array([0.2, 0.5, 0.3]), np.array([0.9, 0.1])], [0, 1, 1], ["noul", "choice", "noul"])
    assert out["lac"]["noul"] == [0.3, 0.9] and out["lac"]["choice"] == [0.5]
    assert out["aps"]["choice"] == [0.5]



def test_decider_returns_sets_and_abstains(bb, tmp_path):
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question")
    big = {"choice": list(np.linspace(0.0, 1.0, 200))}
    dec = Decider(m, profiles_dir=str(tmp_path), conformal={"lac": big, "aps": big})
    c = TestClient(create_app(dec))
    req = {"state": "charged twice", "questions": {"r": {"type": "choice", "prompt": "Team?", "options": ["billing", "tech", "sales"]}}}
    a = c.post("/v1/decide", json={**req, "conformal": {"alpha": 0.1}}).json()["answers"]["r"]
    assert set(a["set"]) <= {"billing", "tech", "sales"} and isinstance(a["abstain"], bool)
    assert a["abstain"] == (len(a["set"]) != 1 or a["value"] == "none")
    assert "set" not in c.post("/v1/decide", json=req).json()["answers"]["r"]       # off unless requested
    # no stored scores for this question type -> 400
    r = c.post("/v1/decide", json={"state": "x", "questions": {"u": {"type": "noul", "prompt": "ok?"}}, "conformal": {}})
    assert r.status_code == 400 and "conformal" in r.text


def test_profile_stores_conformal_scores(bb, tmp_path):
    from opendecider.calibration import fit_profile
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question")
    dec = Decider(m, profiles_dir=str(tmp_path))
    rng = random.Random(0)
    q = {"type": "choice", "prompt": "Team?", "options": ["billing", "tech"]}
    ex = [{"state": f"ticket {i} about {'money' if i % 2 else 'a crash'}", "label": "billing" if i % 2 else "tech"}
          for i in range(12)]
    rng.shuffle(ex)
    prof = fit_profile(dec, {"question": q, "examples": ex, "method": "temperature"}, str(tmp_path))
    assert len(prof["conformal"]["lac"]["choice"]) == 12
    out = dec.decide({"state": "refund please", "questions": {"t": {**q, "calibration": prof["id"]}},
                      "conformal": {"alpha": 0.2, "method": "aps"}})
    assert out["answers"]["t"]["set"]
