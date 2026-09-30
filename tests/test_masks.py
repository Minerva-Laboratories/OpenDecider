"""Attention-mask / isolation audit for the branched V3 path (the configuration we train)."""
import random

import torch

from opendecider.batching import Question, state_texts
from opendecider.train import DEFAULT_TRAIN, step_loss
from tests.conftest import make
from tests.test_models import OPTS, STATE
from tests.test_train import REC
from tests.tiny import tiny_backbone

OTHER = "Printer on floor 2 jams every morning; the user asks whether IT already knows."


def _mem(m, bb, states):
    seqs = bb.tokenize(state_texts(states))
    ids, mask = bb.pad(seqs)
    return m.memory_from_ids(ids, mask)


def test_batch_and_padding_invariance_branched():
    """A question's answer must not change when batched with another question on another (shorter) state."""
    bb = tiny_backbone(seed=7, kv_quant="none")
    m = make(bb, "v3", v3_features="branched", slot_emb="none")
    qa = Question("Which team?", OPTS, state_idx=0)
    qb = Question("Is IT aware of the printer issue already?", ["yes", "no"], state_idx=1)
    _, alone = m.run([Question(qa.prompt, qa.options, state_idx=0)], _mem(m, bb, [STATE]))
    _, both = m.run([qa, qb], _mem(m, bb, [STATE, OTHER]))
    assert torch.allclose(torch.softmax(alone.logits[0, :5], -1), torch.softmax(both.logits[0, :5], -1), atol=1e-5)


def test_state_isolation_between_requests_in_a_batch():
    bb = tiny_backbone(seed=7, kv_quant="none")
    m = make(bb, "v3", v3_features="branched", slot_emb="none")
    q0 = Question("Which team?", OPTS, state_idx=0)
    _, a = m.run([q0, Question("Which team?", OPTS, state_idx=1)], _mem(m, bb, [STATE, OTHER]))
    _, b = m.run([q0, Question("Which team?", OPTS, state_idx=1)], _mem(m, bb, [STATE, STATE + " Extra text."]))
    assert torch.allclose(a.logits[0, :5], b.logits[0, :5], atol=1e-5)        # q0 never sees state 1


def test_branched_rows_do_not_see_other_options():
    bb = tiny_backbone(seed=7, kv_quant="none")
    m = make(bb, "v3", v3_features="branched", slot_emb="none")               # list-format rows
    mem = _mem(m, bb, [STATE])
    f1 = m._conditioned_feats([Question("Which team?", ["billing", "refund"])], mem)
    f2 = m._conditioned_feats([Question("Which team?", ["billing", "a completely different option"])], mem)
    n_first = int(f1[5][0])                                                    # tokens of option "billing"
    assert torch.allclose(f1[4][:n_first], f2[4][:n_first], atol=1e-5)


def test_full_model_vera_trains_state_pass_and_depth_attn():
    bb = tiny_backbone(seed=8, layers=(4, "final"))
    m = make(bb, "v3", v3_features="branched", v3_vera_layers=1, v3_vera_scope="all", v3_vera_rank=8,
             v3_layer_combine="attn")
    assert len(m.vera.targets) > 0 and all(t[1].startswith("L") for t in m.vera.targets)
    assert {int(t[1].split(".")[0][1:]) for t in m.vera.targets} == set(range(bb.num_layers))   # every layer
    with torch.no_grad():                                                      # make adapters non-trivial
        for p in m.vera.b.values():
            p.normal_(0, 0.1)
    m.train()
    step_loss(m, [REC], random.Random(0), dict(DEFAULT_TRAIN, tokens_per_microbatch=8))   # several micro-batches
    assert all(p.grad is None for p in bb.parameters())
    assert m.depth_attn.q.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.vera.b.values())
    # the state pass itself must carry a graph back to the adapters (training), and none at inference
    mem = _mem(m, bb, [STATE])
    s_flat = m._conditioned_feats([Question("Which team?", OPTS)], mem)[0]
    assert s_flat.requires_grad
    gb = torch.autograd.grad(s_flat.sum(), [m.vera.b[k] for k in m.vera.b], allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gb)
    m.eval()
    with torch.no_grad():
        assert not m._conditioned_feats([Question("Which team?", OPTS)], _mem(m, bb, [STATE]))[0].requires_grad


def test_row_checkpoint_matches_plain_gradients():
    for feats, extra in (("branched", {}), ("conditioned", {}),
                         ("branched", dict(v3_row_format="answer", v3_question_cache=True, v3_list_cap=255))):
        bb = tiny_backbone(seed=9, layers=(4, "final"))
        m = make(bb, "v3", v3_features=feats, v3_vera_layers=1, v3_vera_scope="all", v3_vera_rank=8,
                 v3_layer_combine="attn", v3_branch_chunk=2, **extra)
        with torch.no_grad():
            for p in m.vera.b.values():
                p.normal_(0, 0.1)
        m.train()
        params = [p for p in m.parameters() if p.requires_grad]
        grads = []
        for ckpt in (False, True):
            m.cfg.v3_row_checkpoint = ckpt
            out = m._conditioned_feats([Question("Which team?", OPTS), Question("Is it urgent?", ["yes", "no"])],
                                       _mem(m, bb, [STATE]))
            loss = sum((t.float() ** 2).sum() for t in out[::2])
            grads.append(torch.autograd.grad(loss, params, allow_unused=True))
        for a, b in zip(*grads):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        assert any(g is not None and g.abs().sum() > 0 for g in grads[1])
