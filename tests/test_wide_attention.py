import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")


def _ref(q, k, v, scale, kvalid=None):
    B, H, T, _ = q.shape
    S = k.shape[2]
    m = torch.arange(S, device=q.device)[None] <= torch.arange(T, device=q.device)[:, None] + (S - T)
    m = m[None, None].expand(B, 1, T, S)
    if kvalid is not None:
        m = m & kvalid[:, None, None, :]
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=m, scale=scale,
                                          enable_gqa=True).nan_to_num(0)


@pytest.mark.parametrize("B,H,G,T,S,DK,DV", [(1, 16, 4, 100, 100, 448, 256), (2, 16, 4, 37, 300, 448, 256),
                                             (1, 16, 4, 1, 77, 448, 256), (2, 8, 2, 129, 129, 200, 128)])
@pytest.mark.parametrize("padded", [False, True])
def test_matches_sdpa(B, H, G, T, S, DK, DV, padded):
    from opendecider.wide_attention import wide_attention
    torch.manual_seed(0)
    q = torch.randn(B, H, T, DK, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, G, S, DK, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, G, S, DV, device="cuda", dtype=torch.bfloat16)
    kv = None
    if padded:
        kv = torch.ones(B, S, dtype=torch.bool, device="cuda")
        kv[-1, :S // 3] = False
    out = wide_attention(q, k, v, DK ** -0.5, kv).float()
    assert (out - _ref(q, k, v, DK ** -0.5, kv)).abs().max().item() < 2e-2


def test_key_valid_from_hf_mask():
    from opendecider.mercurius_cache import _key_valid
    pad = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    m = causal[None, None] & pad[:, None, None, :]
    assert torch.equal(_key_valid(m, 4), pad)
    add = torch.zeros(2, 1, 4, 4).masked_fill(~m, torch.finfo(torch.float32).min)
    assert torch.equal(_key_valid(add, 4), pad)
    assert _key_valid(causal[None, None].expand(2, 1, 4, 4), 4) is None


def test_noncausal_lse():
    from opendecider.wide_attention import wide_attention
    torch.manual_seed(1)
    B, H, G, T, S, DK = 2, 16, 4, 50, 333, 448
    q = torch.randn(B, H, T, DK, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, G, S, DK, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, G, S, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(B, S, dtype=torch.bool, device="cuda")
    kv[0, :100] = False
    out, lse = wide_attention(q, k, v, DK ** -0.5, kv, causal=False, return_lse=True)
    kk = k.float().repeat_interleave(H // G, 1)
    s = torch.einsum("bhtd,bhsd->bhts", q.float(), kk) * DK ** -0.5
    s = s.masked_fill(~kv[:, None, None], float("-inf"))
    ref = torch.softmax(s, -1) @ v.float().repeat_interleave(H // G, 1)
    assert (out.float() - ref).abs().max().item() < 2e-2
    assert (lse - torch.logsumexp(s, -1)).abs().max().item() < 1e-2
