import torch
import torch.nn as nn

from opendecider.quant import Int8Linear, materialize_int8_


def test_materialized_int8_is_bit_identical():
    """Training expands the int8 weights once; outputs and input gradients must equal the per-call int8 path."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 48).to(torch.bfloat16)
    ref, mat = Int8Linear(lin), Int8Linear(lin)
    m = nn.Sequential(mat)
    assert materialize_int8_(m, torch.bfloat16) == 1 and mat.qweight.numel() == 0
    x = torch.randn(5, 64, dtype=torch.bfloat16, requires_grad=True)
    y1 = ref(x); g1, = torch.autograd.grad(y1.sum(), x)
    y2 = mat(x); g2, = torch.autograd.grad(y2.sum(), x)
    assert torch.equal(y1, y2) and torch.equal(g1, g2)
    assert torch.equal(mat.weight, ref.weight)


def test_materialize_respects_budget():
    m = nn.Sequential(*[Int8Linear(nn.Linear(1024, 1024)) for _ in range(4)])   # 1 MiB extra each in bf16
    assert materialize_int8_(m, torch.bfloat16, budget_gb=2.5 / 1024) == 2
    assert sum(l.__dict__.get("_dq") is not None for l in m) == 2


def test_token_logprob_row_blocks_match_one_block():
    """Bounding the logits block (rows processed in blocks) must not change the result."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tiny import tiny_backbone
    bb = tiny_backbone(kv_quant="none")
    torch.manual_seed(0)
    h = torch.randn(37, bb.hidden_size)
    tok = torch.randint(0, 1000, (37,))
    a = bb.token_logprob(h, tok)
    b = bb.token_logprob(h, tok, chunk=4096, max_block_bytes=4 * 4096 * 5)       # 5 rows per block
    assert torch.allclose(a, b, atol=1e-5)
