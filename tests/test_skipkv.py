"""skipkv: the cache-only upper layers write the same cache as the full layers when the predicted input is exact."""
import pytest
import torch

from opendecider.batching import state_texts
from opendecider.skipkv import SkipMaps, fit_skip_maps, skipped_share
from tiny import tiny_backbone


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
    from opendecider.skipkv import ema_scan
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
