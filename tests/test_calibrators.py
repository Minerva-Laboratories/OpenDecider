import json

import numpy as np
import pytest
import torch

from opendecider import calibrators as C
from opendecider.decider import Decider
from tests.conftest import make
from tests.test_models import STATE


def _data(n, K=3, scale=1.0, seed=0, pad=False):
    """Logits z; labels drawn from softmax(z) (calibrated), then logits multiplied by `scale` (scale > 1: overconfident).
    pad: random K per question in 2..K (padding NEG)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1.5, (n, K))
    M = np.ones((n, K), bool)
    if pad:
        M = np.arange(K)[None] < rng.integers(2, K + 1, n)[:, None]
    P = C.softmax_masked(Z, M)
    y = (P.cumsum(1) < rng.random((n, 1))).sum(1).clip(max=M.sum(1) - 1)
    L = np.where(M, np.log(np.clip(C.softmax_masked(Z * scale, M), 1e-300, 1)), C.NEG)
    return L, np.eye(K)[y], M


@pytest.mark.parametrize("kind", C.SIMPLICITY)
def test_each_calibrator_returns_distributions(kind):
    L, T, M = _data(120, K=4, scale=3.0, pad=kind != "vector")
    cal = C.fit_calibrator(kind, L, T, M)
    Lt, _, Mt = _data(50, K=4, scale=3.0, seed=1, pad=kind != "vector")
    P = C.apply_calibrator(cal, Lt, Mt)
    assert np.isfinite(P).all() and (P >= 0).all() and (P[~Mt] == 0).all()
    np.testing.assert_allclose(P.sum(1), 1, atol=1e-6)
    # round-trip through JSON (isotonic knots / histogram edges as lists) applies identically
    cal2 = json.loads(json.dumps(cal))
    np.testing.assert_allclose(C.apply_calibrator(cal2, Lt, Mt), P, atol=1e-12)


@pytest.mark.parametrize("kind", ["beta", "iso_shrunk"])
def test_monotone_per_option(kind):
    for scale in (0.3, 3.0):
        L, T, M = _data(150, scale=scale, seed=3)
        cal = C.fit_calibrator(kind, L, T, M)
        x = np.linspace(0, 1, 501)
        q = C.option_map(cal, x)
        if kind == "iso_shrunk":                      # the blend toward p_T keeps the map monotone
            q = cal["lam"] * q + (1 - cal["lam"]) * x
        assert (np.diff(q) >= -1e-12).all(), (kind, scale)


def test_selection_identity_when_calibrated_small_n():
    picks = [C.select_calibrator(*_data(20, scale=1.0, seed=s))[0]["kind"] for s in range(3)]
    assert picks == ["identity"] * 3, picks


def test_selection_nonidentity_when_miscalibrated():
    cal, table = C.select_calibrator(*_data(200, scale=4.0, seed=0))
    assert cal["kind"] != "identity"
    assert table["identity"]["log_loss"] > table[cal["kind"]]["log_loss"] + table[cal["kind"]]["se"]
    assert set(table) == set(C.SIMPLICITY) and all(v["se"] > 0 for v in table.values())


def test_shrunk_temperature_moves_toward_one_with_few_labels():
    L, T, M = _data(10, scale=4.0, seed=1)
    free, shrunk = C.fit_temp(L, T, M), C.fit_temp(L, T, M, prior=C.PRIOR)
    assert abs(np.log(shrunk)) < abs(np.log(free))


def test_calibrate_rows_matches_numpy():
    L, T, M = _data(80, K=4, scale=3.0, pad=True)
    for kind in ("beta", "ets", "iso_shrunk", "histogram"):
        cal = C.fit_calibrator(kind, L, T, M)
        z = torch.tensor(L)                        # float64: histogram is a step map, float32 can flip a bin
        m = torch.tensor(M)
        p = torch.softmax(z / cal["T"], -1) * m
        got = C.calibrate_rows(cal, p, z, m)
        np.testing.assert_allclose(got.numpy(), C.apply_calibrator(cal, L, M), atol=1e-5)
        np.testing.assert_allclose(got.sum(1).numpy(), 1, atol=1e-6)


@pytest.fixture(scope="module")
def dec(bb, tmp_path_factory):
    d = Decider(make(bb, "v1"), profiles_dir=str(tmp_path_factory.mktemp("profiles")))
    d.none_text = ""          # exact comparisons against raw logits: no explicit `none` option (options interact)
    return d


QSPEC = {"type": "choice", "prompt": "Which team?", "options": ["billing", "tech", "other"]}


def _raw_logits(dec, qspec):
    from opendecider.batching import state_texts
    from opendecider.schema import DecideRequest
    spec = DecideRequest.model_validate({"state": STATE, "questions": {"t": qspec}}).questions["t"]
    q = dec._atomic("t", spec, {})
    with torch.no_grad():
        _, out = dec.model.run([q], dec.model.encode_states(state_texts([STATE])))
    return out.logits[0, :3].double()


def test_old_profile_file_loads_and_applies_identically(dec):
    # a profile written by the previous /v1/calibrate (no calibrator / cv / requested fields)
    old = {"temperature": 1.7, "bias": {"billing": 0.4, "tech": -0.2, "other": 0.1}, "method": "vector",
           "type": "choice", "names": ["billing", "tech", "other"], "n": 30, "created": "2026-09-01 00:00:00",
           "fit_in_sample": {}, "id": "oldprofile01"}
    json.dump(old, open(f"{dec.profiles_dir}/oldprofile01.json", "w"))
    got = dec.decide({"state": STATE, "questions": {"t": dict(QSPEC, calibration="oldprofile01")}})["answers"]["t"]["probs"]
    want = torch.softmax(_raw_logits(dec, QSPEC) / 1.7 + torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64), -1)
    np.testing.assert_allclose([got[k] for k in QSPEC["options"]], want.numpy(), atol=1e-5)


def test_fit_profile_auto_end_to_end(dec):
    from opendecider.calibration import fit_profile
    exs = [{"state": f"{STATE} case {i}", "label": ["billing", "tech", "other"][i % 3]} for i in range(12)]
    prof = fit_profile(dec, {"question": QSPEC, "examples": exs}, dec.profiles_dir)       # method defaults to auto
    assert prof["requested"] == "auto" and prof["method"] in C.SIMPLICITY and prof["n"] == 12
    assert "histogram" not in prof["cv"] and set(prof["cv"]) == set(C.SIMPLICITY) - {"histogram"}
    saved = json.load(open(f"{dec.profiles_dir}/{prof['id']}.json"))
    assert saved["method"] == prof["method"] and saved["cv"] == prof["cv"]
    dec._profiles.clear()                                                                  # reload from disk
    a = dec.decide({"state": STATE, "questions": {"t": dict(QSPEC, calibration=prof["id"]), "u": QSPEC}})["answers"]
    assert abs(sum(a["t"]["probs"].values()) - 1) < 1e-6
    # every probability-level method applies through decide exactly as the library map does
    for kind in ("beta", "ets", "iso_shrunk"):
        p = fit_profile(dec, {"question": QSPEC, "examples": exs, "method": kind}, dec.profiles_dir)
        assert p["method"] == kind and p["calibrator"]["kind"] == kind
        got = dec.decide({"state": STATE, "questions": {"t": dict(QSPEC, calibration=p["id"])}})["answers"]["t"]["probs"]
        z = _raw_logits(dec, QSPEC).numpy()[None]
        want = C.apply_calibrator(p["calibrator"], z, np.ones_like(z, bool))[0]
        np.testing.assert_allclose([got[k] for k in QSPEC["options"]], want, atol=1e-5)
        assert abs(sum(got.values()) - 1) < 1e-6
    with pytest.raises(ValueError):
        fit_profile(dec, {"question": QSPEC, "examples": exs, "method": "histogram"}, dec.profiles_dir)
    nq = {"type": "noul", "prompt": "Is this a refund request?"}
    p = fit_profile(dec, {"question": nq, "examples": [{"state": f"{STATE} {i}", "label": i % 2 == 0} for i in range(12)]},
                    dec.profiles_dir)
    assert "histogram" in p["cv"]                                                          # binary schema
