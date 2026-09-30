import json
import math

import numpy as np
import pytest
import torch

from eval import metrics as M
from eval.baselines import LabelScorer, _emb_rows
from eval.probes import (dummy_option_probe, duplicate_option_probe, label_length_probe, latency_probe,
                         model_decide_fn, model_request_fn, permutation_probe, repeat_probe, run_battery)
from opendecider.batching import Question, state_texts
from opendecider.decider import Decider
from tests.conftest import make

STATE = "Ticket: I was charged twice for order 4411 and want my money back. Plan: Pro annual."
OPTS = ["billing", "technical support", "refund", "other", "sales"]
ITEM = {"state": STATE, "prompt": "Which team should handle this?", "options": OPTS,
        "long_options": ["the billing department", "technical support engineers", "the refunds desk",
                         "some other team", "the sales team"]}


# ------------------------------------------------------------------------ metrics
def test_basic_metrics_hand_computed():
    probs = [np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.6]), np.array([0.25, 0.25, 0.5])]
    labels = [0, 0, 2]
    assert M.accuracy(probs, labels) == pytest.approx(2 / 3)
    assert M.nll(probs, labels) == pytest.approx(-(math.log(0.7) + math.log(0.4) + math.log(0.5)) / 3)
    b = [(0.3**2 + 0.2**2 + 0.1**2), (0.6**2 + 0.6**2), (0.25**2 * 2 + 0.5**2)]
    assert M.brier(probs, labels) == pytest.approx(np.mean(b))


def test_ece_calibrated_and_overconfident():
    rng = np.random.default_rng(0)
    # perfectly calibrated by construction: 15 bins of 100, each bin's accuracy == its confidence
    probs, labels = [], []
    for c in np.linspace(0.55, 0.95, 15):
        n_ok = int(round(c * 100))
        for i in range(100):
            probs.append(np.array([c, 1 - c]))
            labels.append(0 if i < n_ok else 1)
    assert M.ece(probs, labels) < 0.01
    # overconfident: always 0.9 confident, right half the time -> ECE = 0.4
    probs = [np.array([0.9, 0.1])] * 200
    labels = [0] * 100 + [1] * 100
    assert M.ece(probs, labels) == pytest.approx(0.4)
    # equal-mass: 15 bins over 45 examples -> 3 each; sanity vs. a random set
    probs = [rng.dirichlet(np.ones(4)) for _ in range(45)]
    labels = list(rng.integers(0, 4, 45))
    assert 0 <= M.ece(probs, labels) <= 1


def test_auroc_and_selective():
    probs = [np.array([0.9, 0.1]), np.array([0.8, 0.2]), np.array([0.6, 0.4]), np.array([0.55, 0.45])]
    labels = [0, 0, 1, 1]      # the confident ones are right, the others wrong
    assert M.auroc(probs, labels) == 1.0
    assert M.auroc(probs, [1, 1, 0, 0]) == 0.0
    assert M.auroc([np.array([0.7, 0.3])] * 4, labels) == 0.5      # ties -> 0.5
    assert math.isnan(M.auroc(probs[:2], [0, 0]))
    assert M.selective_accuracy(probs, labels, 0.5) == 1.0
    assert M.selective_accuracy(probs, labels, 0.95) == 0.5


def test_bootstrap_ci_contains_point_and_summarize():
    rng = np.random.default_rng(1)
    probs = [rng.dirichlet(np.ones(k)) for k in rng.integers(2, 9, 300)]
    labels = [int(rng.integers(0, len(p))) for p in probs]
    for fn in (M.accuracy, M.nll, M.ece):
        pt, lo, hi = M.bootstrap_ci(fn, probs, labels, n=200)
        assert lo <= pt <= hi and lo < hi
    # generic (non-fast-path) metric
    pt, lo, hi = M.bootstrap_ci(lambda p, y: float(np.mean([len(x) for x in p])), probs, labels, n=100)
    assert lo <= pt <= hi
    s = M.summarize(probs, labels, n_boot=200)
    assert s["n"] == 300
    for k in ["accuracy", "nll", "brier", "ece", "auroc", "selective_acc@50", "selective_acc@80", "selective_acc@95"]:
        assert s[k]["lo"] <= s[k]["value"] <= s[k]["hi"], k
    json.dumps(s)


def test_fit_temperature_recovers_known_T():
    rng = np.random.default_rng(0)
    T_true = 2.5
    logits, labels = [], []
    for _ in range(4000):
        K = int(rng.integers(2, 10))
        z = rng.normal(0, 3, K)
        p = M.softmax(z)                         # true distribution
        labels.append(int(rng.choice(K, p=p)))
        logits.append(z * T_true)                # model is overconfident by T_true
    T = M.fit_temperature(logits, labels)
    assert T == pytest.approx(T_true, rel=0.08)
    before = M.nll(M.apply_temperature(logits, 1.0), labels)
    after = M.nll(M.apply_temperature(logits, T), labels)
    assert after < before


# ------------------------------------------------------------------------ probes
# The reference (pure-torch) Gated-DeltaNet kernel makes each tiny-model call ~1 s on CPU, so the
# end-to-end probe tests use 3 options and few repetitions; probe *logic* is checked on a fast
# synthetic decide_fn below.
SMALL = {"state": STATE, "prompt": "Which team should handle this?", "options": ["billing", "refund", "sales"],
         "long_options": ["the billing department", "the refunds desk", "the sales team"]}


def luce_fn(state, qd):
    """Order-free Luce model: p_k proportional to a per-name weight."""
    w = np.array([1.0 + len(o) for o in qd["options"]])
    return list(w / w.sum())


def positional_fn(state, qd):
    """Order-sensitive toy: the first slot gets a bonus."""
    w = np.array([1.0 + len(o) for o in qd["options"]])
    w[0] *= 3
    return list(w / w.sum())


def test_probe_logic_on_synthetic_models():
    item = {**ITEM}
    assert permutation_probe(luce_fn, item, n_perm=5)["max_tv"] < 1e-12
    pp = permutation_probe(positional_fn, item, n_perm=5)
    assert pp["mean_tv"] > 0.05 and pp["position_mass"][0] > 1 / len(OPTS)
    d = dummy_option_probe(luce_fn, item)
    assert d["tv_renormalised"] < 1e-12 and d["dummy_mass"] > 0          # proportional rescaling
    dup = duplicate_option_probe(luce_fn, item)
    assert dup["split_ratio_first"] == pytest.approx(0.5) and dup["pair_gain"] > 0
    assert dup["others_renorm_tv"] < 1e-12
    ll = label_length_probe(luce_fn, item)
    assert ll["mean_one_at_a_time_shift"] > 0                              # luce_fn favours long names
    assert repeat_probe(luce_fn, item, n=5)["class"] == "deterministic"
    rng = np.random.default_rng(0)
    noisy = lambda s, q: list(np.array(luce_fn(s, q)) + rng.normal(0, 1e-6, len(q["options"])))
    assert repeat_probe(noisy, item, n=5)["class"] == "float_jitter"
    res = run_battery(positional_fn, [item, {**item, "prompt": "Who?"}], None, n_perm=4, n_repeat=3, n_boot=50)
    a = res["aggregate"]
    assert a["permutation_mean_tv"]["lo"] <= a["permutation_mean_tv"]["mean"] <= a["permutation_mean_tv"]["hi"]
    assert a["first_position_excess_mass"]["mean"] > 0 and a["repeat_class"] == "deterministic"
    json.dumps(res)


def test_permutation_probe_v1_no_slots_is_invariant(bb):
    fn = model_decide_fn(make(bb, "v1", slot_emb="none"))
    r = permutation_probe(fn, SMALL, n_perm=3)
    assert r["max_tv"] < 1e-4
    assert abs(sum(r["position_mass"]) - 1) < 1e-6


def test_permutation_probe_learned_slots_perturbed(bb):
    m = make(bb, "v1", slot_emb="learned")
    with torch.no_grad():
        m.slots.table.weight.normal_(0, 1.0)
        m.slots.table.weight[0].zero_()
    r = permutation_probe(model_decide_fn(m), SMALL, n_perm=3)
    assert r["mean_tv"] > 1e-3


def test_battery_end_to_end_on_tiny_model(bb):
    m = make(bb, "v1", slot_emb="none")
    res = run_battery(model_decide_fn(m), [SMALL], model_request_fn(m), n_perm=2, n_repeat=2,
                      latency_k=(2, 4), latency_q=(1,), latency_reps=1, n_boot=50)
    a = res["aggregate"]
    assert a["permutation_mean_tv"]["mean"] < 1e-4
    assert 0 < a["dummy_mass"]["mean"] < 1
    # permutation-invariant model: two identical copies get identical probability
    dup = res["items"]["duplicate"][0]
    assert dup["resolvable"] and dup["p_copy_first"] == pytest.approx(dup["p_copy_second"], abs=1e-5)
    assert len(res["items"]["label_length"][0]["shifts"]) == 3
    assert a["repeat_class"] in ("deterministic", "float_jitter")
    assert len(res["latency"]["cells"]) == 2 and all(c["server_ms_p50"] > 0 for c in res["latency"]["cells"])
    json.dumps(res)


def test_sampler_gives_sampling_variance(bb):
    ms = make(bb, "v2", dropout=0.3)
    assert repeat_probe(model_decide_fn(ms, sampler_k=2), SMALL, n=2)["class"] == "sampling_variance"


def test_duplicate_probe_with_dict_api_is_marked_unresolvable(bb):
    d = Decider(make(bb, "v1"))

    def dict_fn(state, qd):
        return d.decide({"state": state, "questions": {"q": qd}})["answers"]["q"]["probs"]

    r = duplicate_option_probe(dict_fn, SMALL)
    assert r["resolvable"] is False and 0 <= r["p_name_before"] <= 1


# ------------------------------------------------------------------------ baselines
@pytest.mark.parametrize("mode", ["first", "seq"])
def test_baselines_valid_distributions(bb, mode):
    sc = LabelScorer(bb, mode, max_rows=2)
    mem = sc.encode_states(state_texts([STATE]))
    qs = [Question("Which team?", OPTS), Question.noul("Is it urgent?"), Question("Pick", ["a", "b c d"])]
    _, out = sc.run(qs, mem)
    p = torch.softmax(out.logits, -1) * out.opt_mask
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-6)
    assert (p[~out.opt_mask] == 0).all()
    fn = model_decide_fn(sc)
    probs = fn(STATE, {"type": "choice", "prompt": "Which team?", "options": OPTS})
    assert abs(sum(probs) - 1) < 1e-6 and len(probs) == len(OPTS)


def test_baseline_seq_matches_full_forward(bb):
    """Prefix-cached, length-normalised scoring == scoring each full sequence from scratch."""
    sc = LabelScorer(bb, "seq")
    mem = sc.encode_states(state_texts([STATE]))
    q = Question("Which team?", ["billing department", "refund"])
    _, out = sc.run([q], mem)
    prefix = sc._prefix(q, mem)
    E = _emb_rows(bb.lm.embed_tokens, slice(None))
    ref = []
    for c in sc._continuations(q):
        ids = torch.tensor([prefix + c])
        f, _ = bb(ids, torch.ones_like(ids, dtype=torch.bool))
        lp = torch.log_softmax(f[0].float() @ E.T, -1)
        P = len(prefix)
        ref.append(np.mean([lp[P - 1 + j, t].item() for j, t in enumerate(c)]))
    assert np.allclose(out.logits[0, :2].numpy(), ref, atol=1e-3)


def test_baseline_first_matches_full_vocab(bb):
    sc = LabelScorer(bb, "first")
    mem = sc.encode_states(state_texts([STATE]))
    q = Question("Which team?", OPTS)
    _, out = sc.run([q], mem)
    ids = torch.tensor([sc._prefix(q, mem)])
    f, _ = bb(ids, torch.ones_like(ids, dtype=torch.bool))
    full = f[0, -1].float() @ _emb_rows(bb.lm.embed_tokens, slice(None)).T
    first = [c[0] for c in sc._continuations(q)]
    assert torch.allclose(out.logits[0, :5], full[first], atol=1e-4)


# ------------------------------------------------------------------------ evaluate + report
def test_evaluate_collect_and_report(bb, tmp_path):
    from eval.evaluate import collect, metrics_for, to_question
    from eval.report import build_report
    m = make(bb, "v2")
    recs = [{"id": "a", "family": "f1", "split": "test_ood", "state": STATE, "questions": [
                {"type": "choice", "prompt": "Which team?", "options": OPTS, "label": 2},
                {"type": "noul", "prompt": "Is it urgent?", "label": True},
                {"type": "score", "prompt": "Severity?", "options": ["low", "high"], "label": "high"}]},
            {"id": "b", "family": "f2", "split": "test_ood", "state": {"x": 1},
             "questions": [{"type": "choice", "prompt": "Pick", "options": ["a", "b", "c"], "label": 0}]}]
    assert to_question(recs[0]["questions"][1]).label == 0
    rows = collect(m, recs)
    assert len(rows) == 4 and all(len(r["logits"]) == r["K"] for r in rows)
    res = metrics_for(rows, 1.7, n_boot=50, seed=0)
    assert set(res["by_family"]) == {"f1", "f2"} and "scaled" in res
    res["meta"] = {"name": "tiny"}
    res["split"] = "test_ood"
    (tmp_path / "tiny").mkdir()
    (tmp_path / "tiny" / "eval_test_ood.json").write_text(json.dumps(res))
    fn = model_decide_fn(m)
    pr = run_battery(fn, [ITEM], None, probes=("permutation", "repeat"), n_perm=2, n_repeat=2, n_boot=20)
    pr["meta"] = {"name": "tiny", "variant": "v2", "slot_emb": "learned"}
    (tmp_path / "tiny" / "probes.json").write_text(json.dumps(pr))
    text = build_report(str(tmp_path))
    assert "consistent" in text and "tiny" in text
    low = text.lower()
    assert "reveal" not in low.replace("do not reveal", "") and "jev uses" not in low
