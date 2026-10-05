import torch
import torch.nn as nn

from opendecider.quant import Q4Table, quantize_table


def _q4_0_values(n=50, k=64, seed=0):
    """Values exactly representable in Q4_0 (what a Q4_0 GGUF tensor dequantizes to)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (n, k // 32, 32), generator=g)
    q[..., 5] = 0                                          # each block reaches -8 d, as ggml's quantizer guarantees
    d = (torch.rand(n, k // 32, 1, generator=g) * 0.02 + 1e-3).half().float()
    return ((q - 8).float() * d).view(n, k)


def test_q4_0_values_round_trip_exactly():
    w = _q4_0_values()
    t = Q4Table(w)
    assert torch.equal(t.dequant_rows(slice(None), torch.float32), w)


def test_q4_error_bound_and_lookup():
    torch.manual_seed(0)
    emb = nn.Embedding(40, 96)
    t = quantize_table(emb, "q4")
    w = emb.weight.data
    err = (t.dequant_rows(slice(None), torch.float32) - w).abs().view(40, 3, 32)
    d = w.view(40, 3, 32).abs().amax(-1, keepdim=True) / 8
    # Q4_0 maps the block's largest magnitude to -8, so the other side reaches only +7: within one step (ggml too)
    assert (err <= d + 1e-6).all() and err.mean() < (d * 0.3).mean()
    ids = torch.tensor([[3, 7], [39, 0]])
    assert torch.equal(t(ids), t.dequant_rows(ids.reshape(-1), torch.float32).view(2, 2, 96))
    assert torch.equal(t.dequant_rows(torch.tensor([5, 9]), torch.float32),
                       t.dequant_rows(slice(None), torch.float32)[[5, 9]])
    assert t.qs.numel() + 2 * t.d.numel() == 40 * 96 // 2 + 40 * 3 * 2     # 4.5 bits per value
