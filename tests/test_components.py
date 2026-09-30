import pytest
import torch

from opendecider.guards import require_free_gb
from opendecider.losses import choice_loss, ordinal_emd_loss
from opendecider.options import SlotEmbedding, span_mean
from opendecider.quant import Int8Linear, Int8Tensor
from opendecider.heads import PointerHead, two_stage_select


def test_int8_tensor_roundtrip():
    x = torch.randn(3, 5, 64)
    q = Int8Tensor.quantize(x)
    assert q.q.dtype == torch.int8
    err = (q.dequantize() - x).abs().max()
    assert err <= x.abs().amax() / 127 + 1e-6
    assert q.nbytes() < x.numel() * 4 / 3


def test_int8_linear_close():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    q = Int8Linear(lin)
    x = torch.randn(10, 64)
    rel = (q(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01


def test_span_mean():
    h = torch.arange(24.).view(1, 6, 4)
    out = span_mean(h, torch.tensor([[0, 1, 3], [0, 5, 6]]))
    assert torch.allclose(out[0], h[0, 1:3].mean(0)) and torch.allclose(out[1], h[0, 5])


def test_slot_modes():
    s = torch.tensor([[1, 2, 3, 0]])
    assert SlotEmbedding(8, "none")(s).abs().sum() == 0
    e = SlotEmbedding(8, "learned")(s)
    assert e[0, 3].abs().sum() == 0 and e[0, 0].abs().sum() > 0     # slot 0 reserved = zero


def test_losses_proper_and_masked():
    logits = torch.tensor([[2.0, 0.0, -1.0, 99.0]])
    mask = torch.tensor([[True, True, True, False]])
    out = choice_loss(logits, mask, torch.tensor([0]))
    p = torch.softmax(logits[:, :3], -1)
    assert torch.isclose(out["ce"], -p[0, 0].log(), atol=1e-5)
    assert torch.isclose(out["brier"], ((p - torch.tensor([1., 0, 0])) ** 2).sum(), atol=1e-5)
    assert ordinal_emd_loss(logits, mask, torch.tensor([2])) > ordinal_emd_loss(logits, mask, torch.tensor([0]))


def test_pointer_masks_padding():
    h = PointerHead(8)
    z = h(torch.randn(2, 8), torch.randn(2, 3, 8), torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    assert z[0, 2] < -1e8


def test_two_stage_keeps_order():
    r = torch.tensor([0.1, 5.0, 0.2, 3.0])
    assert two_stage_select(r, keep=2).tolist() == [1, 3]


def test_disk_guard():
    with pytest.raises(RuntimeError):
        require_free_gb(expected_write_gb=10**6)
    assert require_free_gb(0.0) > 4.0


def test_span_pool_modes():
    from opendecider.options import span_pool
    h = torch.arange(12.).view(1, 6, 2)
    sp = torch.tensor([[0, 1, 4]])                       # tokens 1,2,3
    assert torch.allclose(span_pool(h, sp, "last")[0], h[0, 3])
    w = torch.tensor([1., 2., 3.]) / 6
    assert torch.allclose(span_pool(h, sp, "weighted")[0], (w[:, None] * h[0, 1:4]).sum(0))


def test_int4_tensor_roundtrip_and_rows():
    import torch
    from opendecider.quant import Int4Tensor, Int8Tensor, maybe_quantize
    torch.manual_seed(0)
    x = torch.randn(3, 4, 50, 72)                                   # last dim not a multiple of the group size
    q = maybe_quantize(x, "int4")
    assert isinstance(q, Int4Tensor) and isinstance(q, Int8Tensor) and q.shape == x.shape
    err = (q.dequantize() - x).abs() / x.abs().amax()
    assert err.max() < 0.08 and q.nbytes() < x.numel() * 2 * 0.5    # under half of bf16 (incl. scales, padding)
    rows = torch.tensor([2, 0])
    assert torch.equal(q.index_select(0, rows).dequantize(), q.dequantize().index_select(0, rows))
    assert isinstance(maybe_quantize(x, "int8"), Int8Tensor) and maybe_quantize(x, False) is x
