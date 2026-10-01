import random

import torch

from opendecider.backbone import checkpoint_backbone_layers
from opendecider.train import microbatches, step_loss, DEFAULT_TRAIN
from opendecider.batching import Question
from tests.conftest import make

REC = {"family": "t", "state": "Order 44 was charged twice; customer wants money back.",
       "questions": [{"type": "choice", "prompt": "Route?", "options": ["billing", "refund", "tech"], "label": 1},
                     {"type": "noul", "prompt": "Charged twice?", "options": ["yes", "no"], "label": 0},
                     {"type": "score", "prompt": "Severity?", "options": ["low", "mid", "high"], "label": 1}]}


def test_microbatches_respect_budget(bb):
    m = make(bb, "v2")
    qs = [Question("Pick one", [f"option number {i}" for i in range(k)]) for k in (2, 3, 200, 5)]
    groups = microbatches(m, qs, budget=300)
    assert sorted(len(q.options) for g in groups for q in g) == [2, 3, 5, 200]
    assert any(len(g) == 1 and len(g[0].options) == 200 for g in groups)


def test_step_loss_accumulates_equivalently(bb):
    """Micro-batched gradient == single-batch gradient; checkpointing does not change it."""
    m = make(bb, "v2").train()
    for mm in m.modules():
        mm.training = False if "Dropout" in type(mm).__name__ else mm.training
    grads = []
    for budget, ckpt in ((10**6, False), (8, False), (8, True)):
        if ckpt:
            checkpoint_backbone_layers(bb)
        m.zero_grad()
        t = dict(DEFAULT_TRAIN, tokens_per_microbatch=budget)
        step_loss(m, [REC], random.Random(0), t)
        grads.append(torch.cat([p.grad.flatten() for p in m.trainable_parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5, rtol=1e-4)
    assert torch.allclose(grads[1], grads[2], atol=1e-5, rtol=1e-4)


def test_state_feature_cache_matches_int8_encoding(bb):
    import opendecider.train as T
    from opendecider.batching import state_texts
    from opendecider.quant import Int8Tensor
    T._STATE_CACHE.clear()
    recs = [dict(REC, id="a"), dict(REC, id="b", state="Short state.")]
    seqs = bb.tokenize(state_texts([r["state"] for r in recs]))
    ids, mask = bb.pad(seqs)
    t = dict(DEFAULT_TRAIN, cache_state_features=True)
    f1 = T.state_features(bb, recs, seqs, ids, mask, t)
    f2 = T.state_features(bb, recs, seqs, ids, mask, t)          # served from cache
    assert len(T._STATE_CACHE["c"]) == 2 and torch.equal(f1, f2)
    with torch.no_grad():
        ref = bb.encode_state_grad(ids, mask)
    assert torch.allclose(f1[1, :len(seqs[1])], Int8Tensor.quantize(ref[1, :len(seqs[1])]).dequantize(), atol=1e-6)
    assert f1[1, len(seqs[1]):].abs().sum() == 0


def test_option_wrap_augmentation_keeps_labels():
    import opendecider.train as T
    rec = {"family": "t", "state": "s", "questions": [{"type": "choice", "prompt": "p",
                                                        "options": ["alpha", "beta", "gamma", "delta"], "label": 2}]}
    T._WRAP_P["p"] = 1.0
    try:
        wrapped = 0
        for seed in range(50):
            q = T.to_questions(rec, 0, random.Random(seed), 9)[0]
            assert "gamma" in q.options[q.label]
            wrapped += any(o not in ("alpha", "beta", "gamma", "delta") for o in q.options)
        assert wrapped == 50
        q = T.to_questions(rec, 0, None, 9)[0]                       # eval path (rng None): never wrapped
        assert q.options == ["alpha", "beta", "gamma", "delta"]
    finally:
        T._WRAP_P["p"] = 0.0


def test_microbatch_cost_counts_answer_row_listing(bb):
    from opendecider.train import question_costs
    q = Question("Pick one", [f"option number {i}" for i in range(12)])
    m_list = make(bb, "v3", v3_features="branched")
    m_ans = make(bb, "v3", v3_features="branched", v3_row_format="answer")
    (r1,), (l1,), _ = question_costs(m_list, [q])
    (r2,), (l2,), _ = question_costs(m_ans, [q])
    assert r1 == r2 == 12 and l2 > l1 + 12 * 3            # each answer row carries the whole listing


def test_soft_targets_follow_option_shuffle():
    import opendecider.train as T
    rec = {"family": "u", "state": "s", "questions": [{"type": "choice", "prompt": "p", "options": ["a", "b", "c"],
                                                        "label": 0, "soft": [0.7, 0.2, 0.1]}]}
    for seed in range(20):
        q = T.to_questions(rec, 0, random.Random(seed), 9)[0]
        assert abs(q.soft[q.options.index("a")] - 0.7) < 1e-9 and abs(q.soft[q.options.index("c")] - 0.1) < 1e-9


def test_unknowable_and_consistency_steps(bb):
    m = make(bb, "v3", v3_features="branched").train()
    unk = {"family": "unknowable_x", "state": "unrelated", "questions": [
        {"type": "choice", "prompt": "Which city?", "options": ["Oslo", "Lima", "Pune"], "label": 1, "soft": [1 / 3] * 3}]}
    batch = {"family": "batch_clinc", "context": "batch", "state": {"items": {"item_1": "book a flight", "item_2": "what's my balance"}},
             "questions": [{"type": "choice", "prompt": "Regarding item_2: What is the intent?", "options": ["travel", "banking"], "label": 1}]}
    t = dict(DEFAULT_TRAIN, consistency_weight=1.0)
    m.zero_grad()
    info = step_loss(m, [unk, batch, REC], random.Random(0), t)
    assert "consistency" in info and info["consistency"] >= 0
    assert sum(p.grad.abs().sum().item() for p in m.trainable_parameters() if p.grad is not None) > 0


def test_training_candidate_sampling_keeps_answer():
    import opendecider.train as T
    opts = [f"o{i}" for i in range(100)]
    rec = {"family": "t", "state": "s", "questions": [{"type": "choice", "prompt": "p", "options": opts, "label": 57}]}
    T._WRAP_P["max_opts"] = 32
    try:
        for seed in range(20):
            q = T.to_questions(rec, 0, random.Random(seed), 9)[0]
            assert len(q.options) == 32 and q.options[q.label] == "o57"
        assert len(T.to_questions(rec, 0, None, 9)[0].options) == 100        # eval path untouched
    finally:
        T._WRAP_P["max_opts"] = 0


def test_schema_examples_render_train_and_consistency(bb):
    from opendecider.formatting import record_state_text
    from opendecider.train import record_tokens
    q = {"type": "choice", "prompt": "What is the intent?", "options": ["travel", "banking"], "label": 1}
    lab = {"family": "schema_x", "context": "schema_icl", "state": {"user_message": "what's my balance"},
           "examples": [{"state": {"user_message": "book a flight"}, "answers": ["travel"]}], "questions": [q]}
    unl = dict(lab, context="schema_batch", examples=[{"state": {"user_message": "book a flight"}, "answers": [None]}])
    txt = record_state_text(lab, "answer", 255)
    assert "book a flight" in txt and txt.index("Answer: travel") < txt.index("Current case:") < txt.index("what's my balance")
    assert "Answer: travel" not in record_state_text(unl, "answer", 255)
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", v3_list_cap=255, v3_question_cache=True).train()
    full = record_tokens(m, [lab], 10**6)[0]
    assert record_tokens(m, [lab], 8)[0] == full[-8:]                       # examples: keep the TAIL (current case)
    t = dict(DEFAULT_TRAIN, consistency_weight=1.0, option_wrap_p=1.0)
    m.zero_grad()
    info = step_loss(m, [lab, unl, REC], random.Random(0), t)
    assert info["consistency"] > 0                                          # schema_batch pair: with vs without examples
    assert sum(p.grad.abs().sum().item() for p in m.trainable_parameters() if p.grad is not None) > 0


def test_state_backward_once_matches_retained_graph():
    """Cutting the shared state pass into leaves (one state backward per step) gives the same gradients as
    backpropagating through the retained state graph from every micro-batch."""
    import torch
    from tests.tiny import tiny_backbone
    bb = tiny_backbone(seed=5, layers=(4, "final"))
    rec2 = {"family": "t2", "state": "Ticket 9: printer jams, urgent.", "questions": [
        {"type": "choice", "prompt": "Which team?", "options": ["IT", "HR", "Finance"], "label": 0},
        {"type": "noul", "prompt": "Is it urgent?", "options": ["yes", "no"], "label": 0}]}
    grads = []
    for once in (False, True):
        m = make(bb, "v3", v3_features="branched", v3_vera_layers=1, v3_vera_scope="all", v3_vera_rank=8,
                 v3_layer_combine="attn", v3_row_format="answer", v3_question_cache=True, v3_list_cap=255)
        with torch.no_grad():
            for p in m.vera.b.values():
                p.normal_(0, 0.1)
        m.train(); m.zero_grad()
        step_loss(m, [REC, rec2], random.Random(0), dict(DEFAULT_TRAIN, tokens_per_microbatch=8, state_backward_once=once))
        grads.append([p.grad.clone() if p.grad is not None else None for p in m.trainable_parameters()])
    assert any(g is not None and g.abs().sum() > 0 for g in grads[1])
    for a, b in zip(*grads):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_none_augmentation_targets():
    """Explicit `none` in training: appended last; unknowable (uniform) -> none; gold removed -> none; else gold kept."""
    import opendecider.train as T
    rec = {"family": "t", "state": "s", "questions": [
        {"type": "choice", "prompt": "p", "options": ["a", "b", "c", "d"], "label": 2},
        {"type": "choice", "prompt": "u", "options": ["a", "b", "c"], "label": 0, "soft": [1 / 3] * 3}]}
    T._WRAP_P.update(p=0.0, max_opts=0, none_p=1.0, none_drop_p=1.0, eval_none=False)
    try:
        q, u = T.to_questions(rec, 0, random.Random(0), 6)
        assert q.options[-1] == T.NONE_TEXT and q.label == len(q.options) - 1 and "c" not in q.options
        assert u.options[-1] == T.NONE_TEXT and u.label == 3 and getattr(u, "soft", None) is None
        T._WRAP_P.update(none_drop_p=0.0)
        q, _ = T.to_questions(rec, 0, random.Random(0), 6)
        assert q.options[q.label] == "c" and q.none_col == 4
        T._WRAP_P.update(none_p=0.0, eval_none=True)                  # evaluation mirrors inference
        q, _ = T.to_questions(rec, 0, None, 6)
        assert q.options == ["a", "b", "c", "d", T.NONE_TEXT] and q.label == 2
    finally:
        T._WRAP_P.update(none_p=0.0, none_drop_p=0.0, eval_none=False)


def test_warm_start_copies_matching_tensors(bb, tmp_path):
    from opendecider.checkpoint import save
    from opendecider.train import warm_start
    src = make(bb, "v2")
    path = str(tmp_path / "m.pt")
    save(src, path, {})
    dst = make(bb, "v2")
    with torch.no_grad():
        for p in dst.trainable_parameters():
            p.add_(1.0)
    info = warm_start(dst, path)
    assert info["loaded"] > 0 and not info["new"] and not info["skipped"]
    for (k, a), b in zip(src.trainable_state_dict().items(), dst.trainable_state_dict().values()):
        assert torch.equal(a, b), k
