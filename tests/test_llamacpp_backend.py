"""llama.cpp backend layout (opendecider.llamacpp.branched_feats) with a fake libod_llama: per-question and per-option
features and log-probs land where V3's trunk expects them (the PyTorch branched path's layout)."""
import threading
from types import SimpleNamespace

import numpy as np
import torch

from opendecider.batching import Question
from opendecider.llamacpp import branched_feats

D = 4


class FakeLlama:
    """Feature of a token = (state id, token id, position in its piece, layer) so the test can check the layout."""

    def __init__(self):
        self.lock = threading.Lock()
        self.n_embd = D
        self.cur = None
        self.calls = []

    def state(self, ids):
        self.cur = list(ids)
        return False

    def rows_qcache(self, questions, options):
        self.calls.append((len(self.cur), len(questions)))
        f = lambda t, i: np.array([[len(self.cur), t, i, l] for l in range(4)], np.float16).reshape(-1)
        qf = np.stack([f(t, i) for q in questions for i, t in enumerate(q)])
        of = np.stack([f(t, i) for os_ in options for o in os_ for i, t in enumerate(o)])
        lp = np.array([-t / 1000 for os_ in options for o in os_ for t in o], np.float32)
        return qf, of, lp


class FakeTok:
    """Token ids below 256: features pass through bfloat16 (exact integers up to 256)."""
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[1 + (sum(map(ord, t)) + k) % 250 for k in range(1 + len(t) % 3)] for t in texts]}


def test_branched_feats_layout():
    from opendecider.llamacpp import LlamaCppBackbone
    bb = LlamaCppBackbone.__new__(LlamaCppBackbone)
    bb.tok, bb.llama, bb._device = FakeTok(), FakeLlama(), torch.device("cpu")
    model = SimpleNamespace(backbone=bb, cfg=SimpleNamespace(v3_list_cap=16))
    qs = [Question("Which team?", ["IT", "HR", "Finance"], "choice"), Question("Urgent?", ["yes", "no"], "noul"),
          Question("Which?", ["a", "b"], "choice")]
    qs[2].state_idx = 1
    mem = SimpleNamespace(extra_prefix=[[1, 2, 3], [4, 5, 6, 7, 8]])
    s_flat, s_lens, q_flat, lq, o_flat, lo = branched_feats(model, qs, mem)
    assert s_flat is None and s_lens.tolist() == [3, 5]
    assert bb.llama.calls == [(3, 2), (5, 1)]                       # one call per state, its questions only
    assert q_flat.shape == (int(lq.sum()), 4 * D) and o_flat.shape == (int(lo.sum()), 4 * D)
    assert len(lq) == 3 and len(lo) == 7 and model._lm_lp.shape == (int(lo.sum()),)
    # question tokens: per question in order, from its own state; positions restart per question
    q = q_flat.float().view(-1, 4, D)[:, 0]
    st = torch.cat([torch.full((int(n),), float(s)) for n, s in zip(lq, [3, 3, 5])])
    assert torch.equal(q[:, 0], st)
    assert torch.equal(q[:, 2], torch.cat([torch.arange(int(n)).float() for n in lq]))
    o = o_flat.float().view(-1, 4, D)[:, 0]
    assert torch.equal(o[:, 2], torch.cat([torch.arange(int(n)).float() for n in lo]))
    assert torch.allclose(model._lm_lp, -o[:, 1] / 1000, atol=1e-3)  # log-probs follow their option tokens
