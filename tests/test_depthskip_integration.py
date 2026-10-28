"""DepthSkip (the skipped state pass, an optional dependency) on OpenDecider's Backbone: the cache-only upper layers write
the same cache as the full layers when the predicted input is exact, through Backbone.prefix_cache_batch / prefill (the
OPENDECIDER_SKIPKV path) and DepthSkip's segment / depth / part-skip forms. Skipped when depthskip is not installed."""
import pytest
import torch

pytest.importorskip("depthskip")

from opendecider.batching import state_texts  # noqa: E402
from depthskip import SkipMaps, fit_skip_maps, skipped_share  # noqa: E402
from tiny import tiny_backbone  # noqa: E402


def _passthrough_above(bb, cut):
    """Layers >= cut add nothing to the residual stream (output projections and MLP zeroed): every upper layer's input
    equals the cut's input, so the identity map predicts it exactly."""
    for layer in bb.lm.layers[cut:]:
        mixer = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.self_attn.o_proj
        mixer.weight.data.zero_()
        layer.mlp.down_proj.weight.data.zero_()


@pytest.mark.parametrize("cut", [3, 6])                     # layer 3 is attention, layer 6 a DeltaNet layer
@pytest.mark.parametrize("chunk", [0, 7])                   # whole pass, and continuation chunks (incl. 1 token)
def test_cache_only_layers_match_full_layers(cut, chunk):
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, cut)
    bb.cfg.prefill_chunk = chunk
    seqs = bb.tokenize(state_texts(["a longer state " * 10 + "x", "short state"]))
    exact, _ = bb.prefix_cache_batch(seqs, quantize=False)
    D, U = bb.hidden_size, len(bb.lm.layers) - cut
    bb.skip_state = SkipMaps(cut, "input", torch.eye(D).repeat(U, 1, 1), torch.zeros(U, D))
    try:
        skip, _ = bb.prefix_cache_batch(seqs, quantize=False)
    finally:
        bb.skip_state = None
    assert set(exact.tensors) == set(skip.tensors)
    for k, t in exact.tensors.items():
        assert torch.allclose(t, skip.tensors[k], atol=1e-5), k
    assert all(type(l).__name__ != "CacheOnlyLayer" for l in bb.lm.layers)    # layers restored


def test_fit_recovers_passthrough_and_share():
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, 4)
    texts = state_texts([f"state number {i} " * 6 for i in range(12)])
    for mode in ("input", "normed"):
        maps = fit_skip_maps(bb, texts[:10], texts[10:], 4, mode, lams=(1e-6,), log=lambda *_: None)
        assert maps.W.shape == (4, bb.hidden_size, bb.hidden_size)
        assert max(maps.meta["held_rel_err"].values()) < 0.05, maps.meta
    s = skipped_share(bb.lm.config, 4)
    assert 0 < s["skipped"] < 1


def test_generic_mixer_path_matches():
    """Mixers with a swapped forward (as Mercurius's ported ones) run whole on the predicted input: same cache."""
    import types
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, 3)
    for layer in bb.lm.layers[3:]:
        m = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        m.forward = types.MethodType(type(m).forward, m)
    seqs = bb.tokenize(state_texts(["a longer state " * 10 + "x", "short state"]))
    exact, _ = bb.prefix_cache_batch(seqs, quantize=False)
    D, U = bb.hidden_size, len(bb.lm.layers) - 3
    bb.skip_state = SkipMaps(3, "input", torch.eye(D).repeat(U, 1, 1), torch.zeros(U, D))
    try:
        skip, _ = bb.prefix_cache_batch(seqs, quantize=False)
    finally:
        bb.skip_state = None
    for k, t in exact.tensors.items():
        assert torch.allclose(t, skip.tensors[k], atol=1e-5), k


def test_ema_scan_matches_loop():
    from depthskip.maps import ema_scan
    x = torch.randn(2, 300, 4)
    y, c = ema_scan(x, 0.9, block=64)
    ref, prev = torch.zeros_like(x), torch.zeros(2, 4)
    for t in range(300):
        prev = 0.9 * prev + 0.1 * x[:, t]
        ref[:, t] = prev
    assert torch.allclose(y, ref, atol=1e-5) and torch.allclose(c, ref[:, -1], atol=1e-5)
    y2, _ = ema_scan(x[:, 100:], 0.9, ref[:, 99], block=64)          # carry continues a split sequence
    assert torch.allclose(y2, ref[:, 100:], atol=1e-5)


@pytest.mark.parametrize("chunk", [0, 7])
def test_tail_and_context_features_keep_exactness(chunk):
    """Passthrough model, identity on the cut's input and zero weight on its EMAs, last 5 tokens through every layer:
    still the exact cache (left padding, continuation chunks)."""
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, 3)
    bb.cfg.prefill_chunk = chunk
    seqs = bb.tokenize(state_texts(["a longer state " * 10 + "x", "short state"]))
    exact, _ = bb.prefix_cache_batch(seqs, quantize=False)
    D, U = bb.hidden_size, len(bb.lm.layers) - 3
    W = torch.cat([torch.eye(D), torch.zeros(2 * D, D)]).repeat(U, 1, 1)
    bb.skip_state = SkipMaps(3, "input", W, torch.zeros(U, D), alphas=(0.5, 0.9), tail=5)
    try:
        skip, _ = bb.prefix_cache_batch(seqs, quantize=False)
    finally:
        bb.skip_state = None
    for k, t in exact.tensors.items():
        assert torch.allclose(t, skip.tensors[k], atol=1e-5), k


def test_fit_with_context_features():
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, 4)
    texts = state_texts([f"state number {i} " * 6 for i in range(12)])
    maps = fit_skip_maps(bb, texts[:10], texts[10:], 4, "normed", lams=(1e-6,), log=lambda *_: None, alphas=(0.9,))
    assert maps.W.shape == (4, 2 * bb.hidden_size, bb.hidden_size) and maps.alphas == (0.9,)
    assert max(maps.meta["held_rel_err"].values()) < 0.05, maps.meta


@pytest.mark.parametrize("tail", [0, 4])
def test_mlp_surrogates_exact_when_mlp_is_linear(tail):
    """Upper MLPs that output zero (down_proj zeroed): the zero surrogate is exact, mixers run as before."""
    from depthskip.maps import MLPSurrogates, fit_mlp_surrogates
    bb = tiny_backbone(kv_quant="none")
    for layer in bb.lm.layers[3:]:
        layer.mlp.down_proj.weight.data.zero_()
    seqs = bb.tokenize(state_texts(["a longer state " * 10 + "x", "short state"]))
    exact, _ = bb.prefix_cache_batch(seqs, quantize=False)
    D, U = bb.hidden_size, len(bb.lm.layers) - 3
    bb.skip_state = MLPSurrogates(3, torch.zeros(U, D, D), torch.zeros(U, D), tail=tail)
    try:
        skip, _ = bb.prefix_cache_batch(seqs, quantize=False)
    finally:
        bb.skip_state = None
    for k, t in exact.tensors.items():
        assert torch.allclose(t, skip.tensors[k], atol=1e-5), k
    assert all(type(l.mlp).__name__ != "_LinearMLP" for l in bb.lm.layers)
    texts = state_texts([f"state number {i} " * 6 for i in range(8)])
    sur = fit_mlp_surrogates(bb, texts[:6], texts[6:], 3, lams=(1e-6,), log=lambda *_: None)
    assert sur.W.shape == (U, D, D) and float(sur.W.float().abs().max()) < 1e-3


def test_seg_lengths_head_and_tail_windows():
    from depthskip.depth import seg_lengths
    m = object()
    assert seg_lengths(1000, [(-1, m), (480, m), (32, None)]) == [488, 480, 32]
    assert seg_lengths(1000, [(64, None), (-1, m), (480, m), (32, None)]) == [64, 424, 480, 32]
    assert seg_lengths(300, [(64, None), (-1, m), (480, m), (32, None)]) == [0, 0, 268, 32]   # tail windows first
    assert seg_lengths(100, [(64, None), (-1, m), (32, None)]) == [64, 4, 32]
    assert sum(seg_lengths(777, [(4, None), (-1, m), (480, m), (32, None)])) == 777


def _cache_tensors(cache):
    """every tensor a DynamicCache holds: attention keys / values, DeltaNet conv / recurrent states."""
    out = {}
    for i, layer in enumerate(cache.layers):
        for k, v in vars(layer).items():
            if torch.is_tensor(v):
                out[(i, k)] = v.float()
            elif isinstance(v, dict):
                out.update({(i, k, kk): t.float() for kk, t in v.items() if torch.is_tensor(t)})
    return out


def test_segment_cache_matches_per_token_depth():
    """Recency plans: the fast segment form writes the cache of the per-token-depth reference (depth_cache)."""
    from depthskip.depth import DepthPlan, depth_cache, segment_cache
    bb = tiny_backbone(kv_quant="none")
    D, L = bb.hidden_size, len(bb.lm.layers)
    g = torch.Generator().manual_seed(1)
    maps = {c: SkipMaps(c, "normed", torch.eye(D).repeat(L - c, 1, 1) + 0.05 * torch.randn(L - c, D, D, generator=g),
                        0.01 * torch.randn(L - c, D, generator=g)) for c in (4, 6)}
    ids = bb.tokenize(["some state text number one "])[0] * 3
    ref = _cache_tensors(depth_cache(bb, ids, DepthPlan(maps, L, [(10, 6), (5, None)])))
    seg = _cache_tensors(segment_cache(bb, ids, [(-1, maps[4]), (5, maps[6]), (5, None)]))
    exact = _cache_tensors(segment_cache(bb, ids, [(-1, None)]))
    assert set(ref) == set(seg) and any(k[0] == 7 for k in ref)
    for k in ref:
        assert torch.allclose(ref[k], seg[k], atol=1e-4), k
    assert torch.allclose(seg[(3, "keys")], exact[(3, "keys")], atol=1e-5)   # below the cut: exact
    assert not torch.allclose(seg[(7, "keys")], exact[(7, "keys")], atol=1e-3)   # above: predicted inputs


@pytest.mark.parametrize("mlp,attn", [(True, False), (False, True), (True, True)])
def test_part_skip_exact_when_parts_are_zero(mlp, attn):
    """Upper MLPs and attention outputs zeroed: zero surrogates are exact, and the skipped attention still writes its
    cache from the true input, so every cache tensor equals the exact pass's."""
    from depthskip.depth import segment_cache
    from depthskip.parts import PartSkip
    bb = tiny_backbone(kv_quant="none")
    _passthrough_above(bb, 4)
    D, L = bb.hidden_size, len(bb.lm.layers)
    ps = PartSkip(4, 4, (torch.zeros(L - 4, D, D), torch.zeros(L - 4, D)) if mlp else None,
                  {7: (torch.zeros(D, D), torch.zeros(D))} if attn else None)
    ids = bb.tokenize(["some state text number one "])[0] * 3
    exact = _cache_tensors(segment_cache(bb, ids, [(-1, None)]))
    skip = _cache_tensors(segment_cache(bb, ids, [(-1, ps)]))
    assert set(exact) == set(skip)
    for k in exact:
        assert torch.allclose(exact[k], skip[k], atol=1e-5), k
    assert all(type(l.mlp).__name__ != "_LinearMLP" for l in bb.lm.layers)
    assert all(type(getattr(l, "self_attn", None)).__name__ != "_AttnSurrogate" for l in bb.lm.layers)
