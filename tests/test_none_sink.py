import torch

from conftest import make
from opendecider.batching import Question
from opendecider.decider import Decider
from tiny import tiny_backbone


def _model():
    return make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_row_format="answer",
                slot_emb="none", v3_cross="question", v3_none=True)


def test_sink_is_last_column_and_order_invariant():
    m = _model()
    opts = ["billing", "technical", "refund", "shipping"]
    mem = m.encode_states(["State:\n{\"msg\": \"charged twice\"}\n"])
    _, out = m.run([Question("Which team?", opts, "choice"), Question("Urgent?", ["yes", "no"], "noul")], mem)
    assert out.logits.shape == (2, len(opts) + 1) and out.opt_mask[:, -1].all()
    assert out.logits[1, 2:4].max() < -1e3                                 # padding before the sink is masked
    perm = [2, 0, 3, 1]
    _, out2 = m.run([Question("Which team?", [opts[i] for i in perm], "choice")], mem)
    p1, p2 = torch.softmax(out.logits[0], -1), torch.softmax(out2.logits[0], -1)
    assert torch.allclose(p1[perm], p2[:4], atol=1e-4) and torch.allclose(p1[-1], p2[-1], atol=1e-4)


def test_api_reports_none_and_closed_world_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")
    d = Decider(_model(), profiles_dir=str(tmp_path))
    out = d.decide({"state": {"msg": "hello"}, "questions": {
        "a": {"type": "choice", "prompt": "Which?", "options": ["x", "y", "z"]},
        "b": {"type": "choice", "prompt": "Which?", "options": ["x", "y", "z"], "none": False}}})
    a, b = out["answers"]["a"], out["answers"]["b"]
    assert 0 <= a["none"] <= 1 and "none" not in b
    assert abs(sum(a["probs"].values()) - 1) < 1e-6 and abs(sum(b["probs"].values()) - 1) < 1e-6
    assert b["value"] in ("x", "y", "z")


def test_explicit_none_text_option(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")
    m = make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_row_format="answer",
             slot_emb="none", v3_cross="question")
    d = Decider(m, profiles_dir=str(tmp_path))
    q = {"a": {"type": "choice", "prompt": "Which?", "options": ["x", "y", "z"]},
         "n": {"type": "noul", "prompt": "Ok?"},
         "b": {"type": "choice", "prompt": "Which?", "options": ["x", "y", "z"], "none": False}}
    out = d.decide({"state": {"msg": "hello"}, "questions": q})["answers"]
    for k in ("a", "n"):
        assert 0 <= out[k]["none"] <= 1 and abs(sum(out[k]["probs"].values()) - 1) < 1e-6
        assert "none of the above" not in out[k]["probs"]
    assert "none" not in out["b"]
    monkeypatch.setattr(d, "none_text", "")                                 # disabled -> no none field
    assert "none" not in d.decide({"state": "x", "questions": {"a": q["a"]}})["answers"]["a"]


def test_explicit_none_with_calibration_profile(tmp_path, monkeypatch):
    from opendecider.calibration import fit_profile
    monkeypatch.setenv("OPENDECIDER_STATE_CACHE_MB", "0")
    m = make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_row_format="answer",
             slot_emb="none", v3_cross="question")
    d = Decider(m, profiles_dir=str(tmp_path))
    spec = {"type": "choice", "prompt": "Which?", "options": ["x", "y", "z"]}
    ex = [{"state": {"v": i}, "label": ["x", "y", "z"][i % 3]} for i in range(12)]
    prof = fit_profile(d, {"question": spec, "examples": ex, "method": "beta"}, str(tmp_path))
    out = d.decide({"state": {"v": 1}, "questions": {"a": {**spec, "calibration": prof["id"]}}})["answers"]["a"]
    assert abs(sum(out["probs"].values()) - 1) < 1e-6 and 0 <= out["none"] <= 1
