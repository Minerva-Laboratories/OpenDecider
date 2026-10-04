import random

import pytest
import torch

from opendecider.batching import Question
from opendecider.decider import Decider
from opendecider.losses import choice_loss
from tests.conftest import make

STATE = "Ticket: I was charged twice for order 4411 and want my money back. Plan: Pro annual."
OPTS = ["billing", "technical support", "refund", "other", "sales"]


def probs(m, opts, state=STATE, prompt="Which team should handle this?"):
    mem = m.encode_states([state])
    _, out = m.run([Question(prompt, list(opts))], mem)
    return torch.softmax(out.logits[0, : len(opts)], -1)


def tv_under_permutation(m, n=6, seed=0):
    rng = random.Random(seed)
    base = probs(m, OPTS)
    tvs = []
    for _ in range(n):
        perm = list(range(len(OPTS)))
        rng.shuffle(perm)
        p = probs(m, [OPTS[i] for i in perm])
        back = torch.empty_like(p)
        back[torch.tensor(perm)] = p
        tvs.append(0.5 * (back - base).abs().sum().item())
    return max(tvs)


@pytest.mark.parametrize("variant,kw", [("v1", {}), ("v2", {"option_encoding": "independent"})])
def test_permutation_invariant_without_slots(bb, variant, kw):
    m = make(bb, variant, slot_emb="none", **kw)
    assert tv_under_permutation(m) < 1e-4


@pytest.mark.parametrize("variant,kw", [("v1", {}), ("v2", {"option_encoding": "independent"})])
def test_slots_make_order_matter(bb, variant, kw):
    m = make(bb, variant, slot_emb="learned", **kw)
    with torch.no_grad():
        m.slots.table.weight.normal_(0, 1.0)
        m.slots.table.weight[0].zero_()
    assert tv_under_permutation(m) > 1e-3


@pytest.mark.parametrize("variant", ["v0", "v2"])
def test_joint_sequence_is_order_sensitive_even_without_slots(bb, variant):
    """Mechanism caveat: a causal joint sequence encodes order by itself, so O6 alone cannot
    single out slot embeddings for V0 / V2-joint."""
    m = make(bb, variant, slot_emb="none")
    assert tv_under_permutation(m) > 1e-4


@pytest.mark.parametrize("variant,kw", [("v0", {}), ("v1", {}), ("v2", {}), ("v2", {"option_encoding": "independent"}),
                                        ("v1", {"resampler": 16}), ("v2", {"pointer": "bilinear"})])
def test_decide_probabilities_sum_to_one(bb, variant, kw):
    d = Decider(make(bb, variant, **kw))
    res = d.decide({"state": {"ticket": STATE, "tier": "pro"}, "questions": {
        "route": {"type": "choice", "prompt": "Which team?", "options": OPTS},
        "urgent": {"type": "noul", "prompt": "Is it urgent?"},
        "sev": {"type": "score", "prompt": "Severity?", "levels": ["low", "medium", "high", "critical"]}}})
    a = res["answers"]
    assert list(a) == ["route", "urgent", "sev"]
    for v in a.values():
        assert abs(sum(v["probs"].values()) - 1) < 1e-6
        assert all(p >= 0 for p in v["probs"].values())
    assert set(a["urgent"]["probs"]) == {"yes", "no"} and 0 <= a["urgent"]["p_yes"] <= 1
    assert 0 <= a["sev"]["expected_index"] <= 3
    assert res["input_tokens"] > 0


def test_questions_are_isolated(bb):
    """A question's answer must not depend on which other questions are in the batch (O3)."""
    m = make(bb, "v2")
    mem = m.encode_states([STATE])
    q1 = Question("Which team?", OPTS)
    q2 = Question("Is it about money?", ["yes", "no"])
    _, a = m.run([q1], mem)
    _, b = m.run([q2, q1], mem)
    assert torch.allclose(torch.softmax(a.logits[0, :5], -1), torch.softmax(b.logits[1, :5], -1), atol=1e-5)


def test_v0_prefix_cache_matches_full_sequence(bb):
    m = make(bb, "v0")
    q = [Question("Which team?", OPTS), Question("Urgent?", ["yes", "no"])]
    bb.cfg.kv_quant = "none"
    try:
        _, a = m.run(q, m.encode_states([STATE]))
        m.cfg.v0_prefix_cache = False
        _, b = m.run(q, m.encode_states([STATE]))
    finally:
        bb.cfg.kv_quant = "int8"; m.cfg.v0_prefix_cache = True
    assert torch.allclose(a.logits.masked_fill(~a.opt_mask, 0), b.logits.masked_fill(~b.opt_mask, 0), atol=1e-4)


@pytest.mark.parametrize("variant", ["v0", "v1", "v2"])
def test_int8_cache_close_to_full_precision(bb, variant):
    m = make(bb, variant)
    q = [Question("Which team?", OPTS)]
    _, a = m.run(q, m.encode_states([STATE]))
    bb.cfg.kv_quant = "none"
    try:
        _, b = m.run(q, m.encode_states([STATE]))
    finally:
        bb.cfg.kv_quant = "int8"
    pa, pb = torch.softmax(a.logits[0, :5], -1), torch.softmax(b.logits[0, :5], -1)
    assert 0.5 * (pa - pb).abs().sum() < 0.01


def test_v2_gates_start_closed_so_backbone_is_unchanged(bb):
    from opendecider.models import ModelConfig, build_model
    m = build_model(bb, ModelConfig(variant="v2", d_model=32, n_heads=4, interaction_layers=1)).eval()
    p = m.pack([Question("Which team?", OPTS)])
    mem = m.encode_states([STATE])
    hooks = {li: (lambda h, b=b, kv=kv: b(h, kv, mem.kv_mask, p.row_state)) for li, b, kv in zip(m.xattn_idx, m.blocks, mem.kv)}
    f0, _ = bb(p.ids, p.mask)
    f1, _ = bb(p.ids, p.mask, hooks=hooks)
    assert torch.equal(f0, f1)
    assert m.xattn_idx == [3, 7]        # after each full-attention layer (every 4th)


def test_two_stage_over_255(bb):
    d = Decider(make(bb, "v1"))
    opts = [f"product {i}" for i in range(300)]
    res = d.decide({"state": STATE, "questions": {"sku": {"type": "choice", "prompt": "Which product?", "options": opts}}})
    a = res["answers"]["sku"]
    assert a["stage1_kept"] == 255 and len(a["probs"]) == 255
    assert abs(sum(a["probs"].values()) - 1) < 1e-6


def test_sampler_returns_std(bb):
    d = Decider(make(bb, "v2", dropout=0.3))
    q = {"r": {"type": "choice", "prompt": "Which team?", "options": OPTS}}
    r1 = d.decide({"state": STATE, "questions": q, "sampler": {"k": 4}})
    assert max(r1["answers"]["r"]["std"].values()) > 0
    r0 = d.decide({"state": STATE, "questions": q})
    assert max(r0["answers"]["r"]["std"].values()) == 0
    rn = d.decide({"state": STATE, "questions": q, "sampler": {"k": 4, "mode": "gaussian_noise"}})
    assert max(rn["answers"]["r"]["std"].values()) > 0


@pytest.mark.parametrize("variant", ["v0", "v1", "v2"])
def test_training_step_only_touches_decision_module(bb, variant):
    m = make(bb, variant).train()
    ids, mask = bb.pad(bb.tokenize([STATE]))
    mem = m.memory_from_ids(ids, mask)
    q = [Question("Which team?", OPTS, label=2)]
    if variant == "v0":
        p = m.pack(q, state_prefix=mem.extra_prefix)
        with torch.no_grad():
            f, _ = bb(p.ids, p.mask)
        out = m(p, mem, feats=f)
    else:
        p = m.pack(q)
        out = m(p, mem)
    loss = choice_loss(out.logits, out.opt_mask, p.labels)["loss"]
    loss.backward()
    assert all(pp.grad is None for pp in bb.parameters())
    assert sum(pp.grad.abs().sum().item() for pp in m.trainable_parameters() if pp.grad is not None) > 0
    if variant == "v2":
        assert m.blocks[0].alpha_xattn.grad is not None


def test_v1_text_cache_matches_direct_encoding(bb):
    q = [Question("Which team?", OPTS), Question("Urgent?", ["yes", "no"])]
    m = make(bb, "v1")
    mem = m.encode_states([STATE])
    _, a = m.run(q, mem)
    _, a2 = m.run(q, mem)                        # served from cache
    m.cfg.text_cache = 0
    try:
        _, b = m.run(q, mem)
    finally:
        m.cfg.text_cache = 200_000
    assert torch.allclose(a.logits, a2.logits)
    mask = a.opt_mask
    assert torch.allclose(a.logits[mask], b.logits[mask], atol=1e-4)


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_memory_is_scale_normalised(bb, variant):
    """Frozen decoder features have L2 ~120; memory K/V must not scale with feature magnitude."""
    from opendecider.backbone import StateMemory
    m = make(bb, variant)
    f = torch.randn(1, 7, bb.hidden_size)
    mask = torch.ones(1, 7, dtype=torch.bool)
    a = m.prepare_memory(StateMemory(f, mask, 7), cache=False)
    b = m.prepare_memory(StateMemory(f * 100, mask, 7), cache=False)
    assert torch.allclose(a.kv[0], b.kv[0], atol=1e-3, rtol=1e-3)


def test_v0_batched_left_padded_prefix_matches_per_state(bb):
    """Left-padded batched state prefixes must give the same answers as encoding each state alone."""
    m = make(bb, "v0")
    bb.cfg.kv_quant = "none"
    try:
        states = [STATE, "Short note: printer on floor 2 is jammed again.", "x " * 30]
        qs = [Question("Which team?", OPTS, state_idx=0), Question("Urgent?", ["yes", "no"], state_idx=1),
              Question("Which team?", OPTS[:3], state_idx=2), Question("Urgent?", ["yes", "no"], state_idx=0)]
        seqs = bb.tokenize(__import__("opendecider.batching", fromlist=["x"]).state_texts(states))
        ids, mask = bb.pad(seqs)
        _, batched = m.run(qs, m.memory_from_ids(ids, mask))
        for i, q in enumerate(qs):
            from opendecider.batching import state_texts
            single = m.encode_states(state_texts([states[q.state_idx]]))
            q1 = Question(q.prompt, q.options, state_idx=0)
            _, o = m.run([q1], single)
            k = len(q.options)
            assert torch.allclose(batched.logits[i, :k], o.logits[0, :k], atol=1e-4), i
    finally:
        bb.cfg.kv_quant = "int8"
