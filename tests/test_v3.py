import math
import random

import pytest
import torch

from opendecider.batching import Question
from opendecider.decider import Decider
from opendecider.v3 import ScaleNorm, XATGLU
from tests.conftest import make
from tests.test_models import OPTS, STATE, tv_under_permutation


def test_scalenorm_and_xatglu_definitions():
    n = ScaleNorm(16)
    x = torch.randn(3, 16) * 50
    assert torch.allclose(n(x).norm(dim=-1), torch.full((3,), math.sqrt(16)), atol=1e-4)
    g = XATGLU(16)
    assert g.alpha.item() == 0.0                              # init 0 -> plain ATGLU gate in (0,1)
    with torch.no_grad():
        g.alpha.fill_(0.5)
        z = torch.tensor([-1e4, 1e4])
        gate = (torch.atan(z) + math.pi / 2) / math.pi * (1 + 2 * g.alpha) - g.alpha
    assert torch.allclose(gate, torch.tensor([-0.5, 1.5]), atol=1e-3)   # range (-a, 1+a)


def test_v3_permutation_invariant_without_slots(bb):
    assert tv_under_permutation(make(bb, "v3", slot_emb="none"), n=3) < 1e-4


def test_v3_slots_make_order_matter(bb):
    m = make(bb, "v3", slot_emb="learned")
    with torch.no_grad():
        m.slots.table.weight.normal_(0, 1.0)
        m.slots.table.weight[0].zero_()
    assert tv_under_permutation(m, n=3) > 1e-3


@pytest.mark.parametrize("kw", [{}, {"v3_context_update": True, "v3_readout": "mix"}])
def test_v3_decide_and_train_step(bb, kw):
    m = make(bb, "v3", **kw)
    res = Decider(m).decide({"state": STATE, "questions": {
        "route": {"type": "choice", "prompt": "Which team?", "options": OPTS},
        "urgent": {"type": "noul", "prompt": "Is it urgent?"}}})
    for a in res["answers"].values():
        assert abs(sum(a["probs"].values()) - 1) < 1e-6
    from opendecider.train import DEFAULT_TRAIN, step_loss
    from tests.test_train import REC
    m.train()
    step_loss(m, [REC], random.Random(0), DEFAULT_TRAIN)
    assert all(p.grad is None for p in bb.parameters())
    g = sum(p.grad.abs().sum().item() for p in m.trainable_parameters() if p.grad is not None)
    assert g > 0
    assert m.layers[0].ff.alpha.grad is not None and m.layers[0].same_bias.grad is not None


def test_v3_duplicate_options_split_evenly_without_slots(bb):
    m = make(bb, "v3", slot_emb="none")
    mem = m.encode_states([STATE])
    _, out = m.run([Question("Which team?", ["billing", "refund", "refund"])], mem)
    p = torch.softmax(out.logits[0, :3], -1)
    assert torch.allclose(p[1], p[2], atol=1e-6)


def test_rope_relative_property():
    from opendecider.v3 import rope
    q = torch.randn(1, 2, 1, 16); k = torch.randn(1, 2, 1, 16)
    s1 = (rope(q, torch.tensor([[3]])) * rope(k, torch.tensor([[1]]))).sum()
    s2 = (rope(q, torch.tensor([[7]])) * rope(k, torch.tensor([[5]]))).sum()
    assert torch.allclose(s1, s2, atol=1e-4)            # depends only on the offset


def test_v3_all_blocks_prenorm_residual(bb):
    """Zeroing every sublayer's output projection must leave the residual stream = its input."""
    m = make(bb, "v3", slot_emb="none")
    for l in m.layers:
        for mod in (l.self_attn.o, l.cross.o, l.ff.wo):
            torch.nn.init.zeros_(mod.weight)
            if mod.bias is not None:
                torch.nn.init.zeros_(mod.bias)
    captured = []
    for l in m.layers:
        l.register_forward_hook(lambda mod, a, out: captured.append((a[0], out[0])))
    m.run([Question("Which team?", OPTS)], m.encode_states([STATE]))
    for x_in, x_out in captured:
        assert torch.equal(x_in, x_out)


def test_v3_context_layers_make_state_question_aware(bb):
    """With context layers, changing only the question changes how the state is represented to options
    even when the question tokens themselves are masked from the options' view."""
    m = make(bb, "v3", slot_emb="none", v3_context_layers=2)
    mem = m.encode_states([STATE])
    _, a = m.run([Question("Which team?", OPTS)], mem)
    _, b = m.run([Question("Is it urgent?", OPTS)], mem)
    assert not torch.allclose(a.logits[0, :5], b.logits[0, :5])
    assert tv_under_permutation(m, n=2) < 1e-4                    # still permutation-invariant


def test_v3_conditioned_features_equal_full_sequence(bb):
    """Batched left-padded state prefixes + one row per question must reproduce, token for token, the
    hidden states of running [state ; question ; options] as one ordinary sequence."""
    from opendecider.batching import state_texts
    from opendecider.formatting import option_text, question_text
    m = make(bb, "v3", v3_features="conditioned")
    bb.cfg.kv_quant = "none"
    try:
        states = [STATE, "Short note: printer jammed."]
        qs = [Question("Which team?", OPTS[:3], state_idx=0), Question("Urgent?", ["yes", "no"], state_idx=1),
              Question("Who?", ["a", "bb"], state_idx=0)]
        seqs = bb.tokenize(state_texts(states))
        ids, mask = bb.pad(seqs)
        mem = m.memory_from_ids(ids, mask)
        s_flat, s_lens, q_flat, q_lens, o_flat, o_lens = m._conditioned_feats(qs, mem)
        qi = oi = 0
        for q in qs:
            pieces = [seqs[q.state_idx]] + bb.tokenize([question_text(q.prompt)] + [option_text(o) for o in q.options])
            full, _ = bb(*bb.pad([sum(pieces, [])]))
            full = full[0].float()
            Ls = len(pieces[0]); Lq = len(pieces[1])
            assert torch.allclose(q_flat[qi:qi + Lq], full[Ls:Ls + Lq], atol=1e-4)
            qi += Lq
            Lo = sum(len(p) for p in pieces[2:])
            assert torch.allclose(o_flat[oi:oi + Lo], full[Ls + Lq:Ls + Lq + Lo], atol=1e-4)
            oi += Lo
        assert s_lens.tolist() == [len(x) for x in seqs]
        # a training step works and never touches backbone weights
        from opendecider.train import DEFAULT_TRAIN, step_loss
        from tests.test_train import REC
        m.train()
        step_loss(m, [REC], random.Random(0), DEFAULT_TRAIN)
        assert all(p.grad is None for p in bb.parameters())
    finally:
        bb.cfg.kv_quant = "int8"


def test_v3_question_conditioned(bb):
    """Question tokens equal a full [state ; question...Answer:] forward; options are order-free."""
    from opendecider.batching import state_texts
    from opendecider.formatting import question_answer_text
    m = make(bb, "v3", v3_features="question_conditioned", slot_emb="none")
    bb.cfg.kv_quant = "none"
    try:
        seqs = bb.tokenize(state_texts([STATE, "Short note."]))
        ids, mask = bb.pad(seqs)
        mem = m.memory_from_ids(ids, mask)
        qs = [Question("Which team?", OPTS, state_idx=1), Question("Urgent?", ["yes", "no"], state_idx=0)]
        _, _, q_flat, q_lens, _, _ = m._conditioned_feats(qs, mem)
        full, _ = bb(*bb.pad([seqs[1] + bb.tokenize([question_answer_text("Which team?")])[0]]))
        L = int(q_lens[0])
        assert torch.allclose(q_flat[:L], full[0, len(seqs[1]):].float(), atol=1e-4)
    finally:
        bb.cfg.kv_quant = "int8"
    # exact permutation invariance through the whole model (options never enter the causal row)
    m.eval()
    from opendecider.batching import state_texts as st
    def pr(opts):
        mm = m.encode_states(st([STATE]))
        _, o = m.run([Question("Which team?", list(opts))], mm)
        return torch.softmax(o.logits[0, :len(opts)], -1)
    a = pr(OPTS); perm = [3, 0, 4, 1, 2]
    b = pr([OPTS[i] for i in perm])
    back = torch.empty_like(b); back[torch.tensor(perm)] = b
    assert torch.allclose(a, back, atol=1e-5)


@pytest.mark.parametrize("cross", ["question", "none"])
def test_v3_cross_ablations_run_and_state_matters_accordingly(bb, cross):
    m = make(bb, "v3", v3_features="question_conditioned", v3_cross=cross, slot_emb="none")
    from opendecider.batching import state_texts as st
    q = [Question("Which team?", OPTS)]
    _, a = m.run(q, m.encode_states(st([STATE])))
    _, b = m.run(q, m.encode_states(st(["Completely different state text here."])))
    # state still reaches the options through the conditioned question tokens (unless cross == none)
    if cross == "none":
        assert torch.allclose(a.logits[0, :5], b.logits[0, :5])     # options never see question/state
    else:
        assert not torch.allclose(a.logits[0, :5], b.logits[0, :5])


def test_v3_branched_features_exact_and_order_free(bb):
    from opendecider.batching import state_texts
    from opendecider.formatting import option_text, question_text
    m = make(bb, "v3", v3_features="branched", v3_branch_chunk=2, slot_emb="none")
    bb.cfg.kv_quant = "none"
    try:
        seqs = bb.tokenize(state_texts([STATE, "Short note."]))
        ids, mask = bb.pad(seqs)
        mem = m.memory_from_ids(ids, mask)
        qs = [Question("Which team?", OPTS[:3], state_idx=1), Question("Urgent?", ["yes", "no"], state_idx=0)]
        _, _, q_flat, q_lens, o_flat, o_lens = m._conditioned_feats(qs, mem)
        oi = 0
        for q in qs:
            qt = bb.tokenize([question_text(q.prompt)])[0]
            for k, o in enumerate(q.options):
                ot = bb.tokenize([option_text(o)])[0]
                full, _ = bb(*bb.pad([seqs[q.state_idx] + qt + ot]))
                Ls = len(seqs[q.state_idx])
                assert torch.allclose(o_flat[oi:oi + len(ot)], full[0, Ls + len(qt):].float(), atol=1e-4)
                oi += len(ot)
    finally:
        bb.cfg.kv_quant = "int8"
    from opendecider.batching import state_texts as st
    m.eval()
    def pr(opts):
        _, o = m.run([Question("Which team?", list(opts))], m.encode_states(st([STATE])))
        return torch.softmax(o.logits[0, :len(opts)], -1)
    a = pr(OPTS); perm = [2, 4, 0, 3, 1]
    b = pr([OPTS[i] for i in perm]); back = torch.empty_like(b); back[torch.tensor(perm)] = b
    assert torch.allclose(a, back, atol=1e-5)


@pytest.mark.parametrize("feat", ["branched", "conditioned", "question_conditioned"])
def test_v3_multilayer_mix_gets_gradient(feat):
    """Learned layer mix must train in the conditioned paths (it used to sit inside the no-grad pass)."""
    from tests.tiny import tiny_backbone
    from opendecider.train import DEFAULT_TRAIN, step_loss
    from tests.test_train import REC
    bb2 = tiny_backbone(layers=(4, "final"))
    m = make(bb2, "v3", v3_features=feat).train()
    step_loss(m, [REC], random.Random(0), DEFAULT_TRAIN)
    assert m.layer_mix is not None and m.layer_mix.grad is not None and m.layer_mix.grad.abs().sum() > 0


def test_mix_wide_stays_fp32_under_autocast():
    from tests.tiny import tiny_backbone
    m = make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched")
    x = torch.randn(5, 2 * m.d)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m._mix_wide(x)
    assert y.dtype == torch.float32 and y.shape == (5, m.d)


def test_lm_feature_matches_full_lm_logprobs():
    """Option-token log-probs from the branched rows == log_softmax of the tied LM head on a full sequence."""
    from opendecider.batching import state_texts
    from opendecider.formatting import option_text, question_text
    from tests.tiny import tiny_backbone
    bb2 = tiny_backbone(seed=5, kv_quant="none")
    m = make(bb2, "v3", v3_features="branched", v3_lm_feature=True, slot_emb="none")
    seqs = bb2.tokenize(state_texts([STATE]))
    ids, mask = bb2.pad(seqs)
    mem = m.memory_from_ids(ids, mask)
    q = Question("Which team?", OPTS[:2], state_idx=0)
    m._conditioned_feats([q], mem)
    lp = m._lm_lp
    qt = bb2.tokenize([question_text(q.prompt)])[0]
    E = bb2.lm.embed_tokens.weight.float()
    ref = []
    for o in q.options:
        ot = bb2.tokenize([option_text(o)])[0]
        full, _ = bb2(*bb2.pad([seqs[0] + qt + ot]))
        logp = torch.log_softmax(full[0].float() @ E.t(), -1)
        start = len(seqs[0]) + len(qt)
        ref += [logp[start + j - 1, t] for j, t in enumerate(ot)]
    assert torch.allclose(lp, torch.stack(ref), atol=2e-2)
    _, out = m.run([q], mem)                                  # end-to-end runs; skip starts at 0
    assert torch.isfinite(out.logits[0, :2]).all() and m.lm_skip.item() == 0.0


def test_answer_rows_are_order_free_and_exact():
    from opendecider.formatting import answer_row_prefix
    a = answer_row_prefix("Which team?", ["b", "A", "c"]); b = answer_row_prefix("Which team?", ["c", "b", "A"])
    assert a == b and a.endswith("\nAnswer:") and a.index("- A") < a.index("- b") < a.index("- c")
    assert "Options" not in answer_row_prefix("q", [str(i) for i in range(20)], list_cap=16)
    m = make(__import__("tests.tiny", fromlist=["x"]).tiny_backbone(seed=6), "v3", v3_features="branched",
             v3_row_format="answer", v3_lm_feature=True, slot_emb="none")
    tv = tv_under_permutation(m, n=2)
    assert tv < 1e-4


def test_answer_listing_dedupes():
    from opendecider.formatting import answer_row_prefix
    assert answer_row_prefix("q", ["b", "a", "b"]).count("- b") == 1


def test_depth_attn_fp32_under_autocast():
    from tests.tiny import tiny_backbone
    m = make(tiny_backbone(layers=(4, "final")), "v3", v3_features="branched", v3_layer_combine="attn")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m._mix_wide(torch.randn(3, 2 * m.d))
    assert y.dtype == torch.float32


def test_question_cache_matches_repeated_question_rows():
    """Two-level prefix (state -> question+listing -> option suffix) == question repeated in every row."""
    import torch
    from opendecider.batching import Question, state_texts
    from tests.conftest import make
    from tests.test_models import OPTS, STATE
    from tests.tiny import tiny_backbone
    bb = tiny_backbone(kv_quant="none", seed=3, layers=(4, "final"))
    qs = [Question("Which team?", OPTS), Question("Is it urgent?", ["yes", "no"]), Question("Pick", ["a", "b"]),
          Question("Which team handles it?", OPTS[::-1] + ["other"]), Question("Is it urgent?", ["no", "yes"])]
    qs[4].state_idx = 1
    outs = []
    for qc in (False, True):
        m = make(bb, "v3", v3_features="branched", v3_row_format="answer", v3_lm_feature=True, v3_list_cap=255,
                 v3_question_cache=qc, v3_branch_chunk=3, v3_shared_prefix=False).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            seqs = bb.tokenize(state_texts([STATE, "Short other state."]))
            ids, mask = bb.pad(seqs)
            f = m._conditioned_feats(qs, m.memory_from_ids(ids, mask))
            outs.append([t.float() for t in f[::2]] + [m._lm_lp.float()])
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_shared_prefix_attention_matches_per_row_copies():
    """Branched rows with one shared state K/V (prefix_attn) == the per-row-copy path, for several states."""
    import torch
    from opendecider.batching import Question
    from tiny import tiny_backbone
    bb = tiny_backbone(layers=(4, "final"), kv_quant="none")
    qs = [Question("Which team?", ["billing", "technical support", "refund"], "choice", state_idx=0),
          Question("Urgent?", ["yes", "no"], "noul", state_idx=1),
          Question("Severity?", ["low", "medium", "high", "critical"], "score", state_idx=0)]
    states = ["State:\n{\"msg\": \"charged twice for my order\"}\n", "State:\n{\"msg\": \"hi\", \"n\": [1, 2, 3, 4, 5]}\n"]
    import opendecider.prefix_attn as pa
    out = {}
    for shared, block in ((False, 0), (True, 1 << 25), (True, 1)):         # 1: one row per SDPA call
        pa.MASK_ELEMS = block or pa.MASK_ELEMS
        m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question",
                 v3_shared_prefix=shared, v3_branch_chunk=3)
        with torch.no_grad():
            _, o = m.run(qs, m.encode_states(states))
        out[(shared, block)] = o.logits
    pa.MASK_ELEMS = 1 << 25
    ref = out[(False, 0)]
    for key in ((True, 1 << 25), (True, 1)):
        assert torch.allclose(ref, out[key], atol=1e-4), (key, (ref - out[key]).abs().max())


def test_state_free_trunk_matches_full_context():
    """v3_cross='question' masks state tokens, so skipping state features (state_free) gives the same logits."""
    import torch
    from opendecider.batching import Question
    from tiny import tiny_backbone
    bb = tiny_backbone(layers=(4, "final"), kv_quant="none")
    qs = [Question("Which team?", ["billing", "technical support", "refund"], "choice", state_idx=0),
          Question("Urgent?", ["yes", "no"], "noul", state_idx=1)]
    states = ["State:\n{\"msg\": \"charged twice for my order\"}\n", "State:\n{\"msg\": \"hi\", \"n\": [1, 2, 3]}\n"]
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question",
             v3_context_layers=1)
    assert m.state_free
    with torch.no_grad():
        _, a = m.run(qs, m.encode_states(states))
        m.state_free = False                                        # old path: state features built, then masked
        _, b = m.run(qs, m.encode_states(states))
    assert torch.allclose(a.logits, b.logits, atol=1e-5), (a.logits - b.logits).abs().max()


@pytest.mark.gpu
def test_shared_prefix_fused_kernel_on_gpu():
    """On CUDA the shared-prefix path must run on the fused memory-efficient SDPA kernel (never the math kernel)
    and match the per-row-copy path."""
    import torch
    from opendecider.batching import Question
    from tiny import tiny_backbone
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from opendecider.guards import gpu_lock
    with gpu_lock("test_shared_prefix_fused"):
        bb = tiny_backbone(layers=(4, "final"), kv_quant="none", dtype="bfloat16")
        bb.lm.to("cuda"); bb.cfg.device = "cuda"
        qs = [Question("Which team?", ["billing", "technical support", "refund"], "choice", state_idx=0),
              Question("Urgent?", ["yes", "no"], "noul", state_idx=1)]
        states = ["State:\n{\"msg\": \"charged twice for my order\"}\n" * 20, "State:\n{\"msg\": \"hi\"}\n"]
        out = {}
        for shared in (False, True):
            m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question",
                     v3_shared_prefix=shared).to("cuda")
            with torch.no_grad():
                _, o = m.run(qs, m.encode_states(states))
            out[shared] = torch.softmax(o.logits.float(), -1)
        assert (out[False] - out[True]).abs().max() < 2e-2                  # bf16 kernels differ in rounding only


@pytest.mark.gpu
def test_graph_engine_matches_eager_on_gpu():
    """CUDA-graph engine (bucketed state and row passes) gives the same decisions as eager."""
    import os
    import torch
    from opendecider.decider import Decider
    from tiny import tiny_backbone
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from opendecider.deploy import GraphEngine
    from opendecider.guards import gpu_lock
    os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
    with gpu_lock("test_graph_engine"):
        bb = tiny_backbone(layers=(4, "final"), kv_quant="int8", dtype="bfloat16")
        bb.lm.to("cuda"); bb.cfg.device = "cuda"
        m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none",
                 v3_cross="question").to("cuda")
        d = Decider(m)
        reqs = [{"state": {"msg": "charged twice " * k}, "questions": {
            "a": {"type": "choice", "prompt": "Which team?", "options": ["billing", "technical", "refund"]},
            "u": {"type": "noul", "prompt": "Urgent?"}}} for k in (1, 30)]
        eager = [d.decide(r) for r in reqs]
        eng = GraphEngine(m).install()
        graph = [d.decide(r) for r in reqs]
        assert eng.stats["state_graph"] == 2 and eng.stats["rows_graph"] >= 2
        for a, b in zip(eager, graph):
            for k in a["answers"]:
                pa, pb = torch.tensor(a["answers"][k]["probs_list"]), torch.tensor(b["answers"][k]["probs_list"])
                assert (pa - pb).abs().max() < 3e-2
