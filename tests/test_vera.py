import random

import torch

from opendecider.batching import Question, state_texts
from opendecider.train import DEFAULT_TRAIN, step_loss
from tests.conftest import make
from tests.test_models import OPTS, STATE
from tests.test_train import REC
from tests.tiny import tiny_backbone


def _pair(**kw):
    bb = tiny_backbone(seed=3)                      # own backbone: VeRA installs hooks on its modules
    m = make(bb, "v3", v3_features="branched", v3_vera_layers=2, v3_vera_rank=16, **kw)
    return bb, m


def test_vera_identity_at_init_and_row_only():
    bb, m = _pair()
    assert m.vera.n_trainable() > 0 and len(m.vera.targets) > 0
    q = [Question("Which team?", OPTS)]
    mem = m.encode_states(state_texts([STATE]))
    _, a = m.run(q, mem)
    m.vera.remove(); _, ref = m.run(q, m.encode_states(state_texts([STATE])))
    assert torch.allclose(a.logits, ref.logits, atol=1e-6)            # b = 0 at init -> identical to frozen


def test_vera_changes_rows_not_state_pass():
    bb, m = _pair()
    with torch.no_grad():
        for p in m.vera.b.values():
            p.normal_(0, 0.5)
    ids, mask = bb.pad(bb.tokenize(state_texts([STATE])))
    with torch.no_grad():
        f_state, _ = bb(ids, mask)                                     # vera inactive outside the row pass
    m.vera.active = True
    with torch.no_grad():
        f_on, _ = bb(ids, mask)
    m.vera.active = False
    assert not torch.allclose(f_state, f_on)                           # adapters do change activations when on
    pc, sm, sh = bb.prefix_cache_batch(bb.tokenize(state_texts([STATE])), return_hidden=True)
    assert torch.allclose(sh[0][sm[0]], f_state[0], atol=1e-5)         # state pass = frozen model


def test_vera_gradients_only_to_adapters_and_trunk():
    bb, m = _pair()
    m.train()
    step_loss(m, [REC], random.Random(0), DEFAULT_TRAIN)
    assert all(p.grad is None for p in bb.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.vera.b.values())


def test_row_caches_are_freed_without_gc():
    """Row caches must be released by refcount (no reference cycles)."""
    import gc
    import weakref
    bb = tiny_backbone(seed=4)
    pc, sm = bb.prefix_cache_batch(bb.tokenize(state_texts([STATE])))
    gc.disable()
    try:
        c = pc.unpack_rows(torch.tensor([0, 0]))
        ref = weakref.ref(c.layers[0])
        del c
        assert ref() is None
    finally:
        gc.enable()


def test_vera_with_backbone_checkpointing():
    from opendecider.backbone import checkpoint_backbone_layers
    bb, m = _pair()
    checkpoint_backbone_layers(bb)
    m.train()
    step_loss(m, [REC], random.Random(0), DEFAULT_TRAIN)
    assert any(p.grad is not None for p in m.vera.b.values())
