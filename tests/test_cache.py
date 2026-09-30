import torch

from opendecider.cache import TokenCache, segment_layout


def test_segment_layout_matches_loop():
    lens = torch.tensor([2, 3, 1, 4, 2])
    group = torch.tensor([0, 0, 1, 2, 2])
    row, col, seg, pos, row_len = segment_layout(lens, group, 3)
    exp = []
    for g in range(3):
        c = 0
        for m in range(5):
            if group[m] == g:
                for p in range(int(lens[m])):
                    exp.append((g, c, m, p)); c += 1
    assert list(zip(row.tolist(), col.tolist(), seg.tolist(), pos.tolist())) == exp
    assert row_len.tolist() == [5, 1, 6]


def test_token_cache_roundtrip_and_growth():
    c = TokenCache(8, "cpu")
    f = torch.randn(3, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    c.add(["a", "b", "c"], f, mask, [3, 5, 1])
    for i in range(3000):                                   # force buffer growth
        c.add([f"x{i}"], f[1:2], mask[1:2], [5])
    flat, lens = c.lookup(["c", "a", "b", "a"])
    assert lens.tolist() == [1, 3, 5, 3]
    ref = torch.cat([f[2, :1], f[0, :3], f[1, :5], f[0, :3]])
    assert (flat - ref).abs().max() <= ref.abs().amax() / 127 + 1e-6
