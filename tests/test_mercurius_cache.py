"""Mercurius's compressed KV cache (src/opendecider/mercurius_cache.py): codes decode to exactly Mercurius's own
TurboQuant round trip, the cache layer appends / expands / selects rows, and PackedCache carries the codes as they are.
Needs the Mercurius source (MERCURIUS_CODE); skipped without it."""
import os
import sys

import pytest
import torch

CODE = os.environ.get("MERCURIUS_CODE", "/mnt/ext/opendecider/code/mercurius")
if not os.path.isdir(os.path.join(CODE, "src", "mercurius")):
    pytest.skip("Mercurius source not available", allow_module_level=True)
sys.path.insert(0, os.path.join(CODE, "src"))
from mercurius.models.qat import tq_roundtrip  # noqa: E402

from opendecider.mercurius_cache import CODE_ATTRS, LatentCodesLayer, tq_decode, tq_encode  # noqa: E402


@pytest.mark.parametrize("r", [64, 289, 512])
def test_codes_equal_mercurius_roundtrip(r):
    torch.manual_seed(r)
    x = (torch.randn(2, 7, r) * 3).to(torch.bfloat16)
    codes, n = tq_encode(x, 4)
    assert codes.dtype == torch.uint8 and codes.shape[-1] == (r + 1) // 2 and n.dtype == torch.float16
    assert torch.equal(tq_decode(codes, n, r, torch.bfloat16, 4), tq_roundtrip(x, 4))


def _parts(B, T, seed):
    g = torch.Generator().manual_seed(seed)
    return {"lat_codes": torch.randint(0, 255, (B, T, 5), generator=g, dtype=torch.uint8),
            "lat_norm": torch.rand(B, T, generator=g).half(), "rope_codes": torch.randint(0, 255, (B, T, 3), generator=g,
                                                                                          dtype=torch.uint8),
            "rope_norm": torch.rand(B, T, generator=g).half(), "inv_r": torch.rand(B, T, 4, generator=g),
            "pos": torch.arange(T, dtype=torch.int32).expand(B, T) + 0}


def test_layer_append_expand_select_and_packed_cache():
    from transformers import DynamicCache
    from opendecider.backbone import PackedCache
    lay = LatentCodesLayer()
    a, b = _parts(2, 5, 0), _parts(2, 3, 1)
    lay.append(**a); lay.append(**b)
    assert lay.get_seq_length() == 8 and lay.get_mask_sizes(4) == (12, 0)
    for k in CODE_ATTRS:
        assert torch.equal(getattr(lay, k), torch.cat([a[k], b[k]], 1))
    cache = DynamicCache()
    cache.layers = [lay]
    pc = PackedCache.pack(cache, kv_int8=True)
    rows = pc.unpack_rows(torch.tensor([1, 0, 1]))
    for k in CODE_ATTRS:
        assert torch.equal(getattr(rows.layers[0], k), getattr(lay, k)[[1, 0, 1]])   # codes untouched by int8
    rep = pc.unpack(batch=2)
    assert torch.equal(rep.layers[0].lat_codes, lay.lat_codes.repeat_interleave(2, 0))


def test_codebook_cached_per_device():
    """Built once per (r, bits, device): no host-to-device copy per call (not allowed inside CUDA-graph capture)."""
    from opendecider.mercurius_cache import _codebook
    a, b = _codebook(64, 4, torch.device("cpu")), _codebook(64, 4, torch.device("cpu"))
    assert a[0] is b[0] and a[1] is b[1]
