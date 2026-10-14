"""State pass that skips the upper layers' work for state tokens (SwiftKV SingleInputKV, arXiv 2410.03960; the causal
encoder-decoder of DeepSeek-V4.1-Flash, arXiv 2609.19969), retrofitted to the frozen backbone with closed-form ridge
maps and no backprop (RESULTS 21).

State tokens run layers 0..cut-1 as usual. Every layer j >= cut builds only its cache for them, from a predicted
input: K/V in the full-attention layers (k_proj, v_proj, k_norm, RoPE), conv and recurrent state in the gated DeltaNet
layers (in_proj_qkv, in_proj_a, in_proj_b, conv, delta rule). Their query/gate projections, output projections, the
attention itself and the MLP are not run for state tokens. Question and option rows run every layer as before and read
that cache, so only the state pass changes.

Predicted input of layer j from h = the input of layer `cut`:
  input   h W_j + b_j, then layer j's own input norm (the probe's maps: RESULTS 21)
  normed  n(h) W_j + b_j straight to layer j's normed input (n = RMS normalization without gain); linear up to
          the projections, so a map could later be folded into them
Mercurius's mixers have their own cache-only paths (MLA: the TurboQuant latent / RoPE-key codes; GDN-2: up to the
state update). Any other mixer runs whole on the predicted input (only the MLP is skipped). Fitted in closed form on calibration states (training-distribution records, never a benchmark); ridge strength
picked by held-out error. Serving opt-in: OPENDECIDER_SKIPKV=<maps file> (Backbone.load) or Backbone.skip_state.
"""
from __future__ import annotations

import contextlib
import sys

import torch
from torch import nn

LAMS = (1e-4, 1e-3, 1e-2, 1e-1)


def _rms(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def ema_scan(x: torch.Tensor, a: float, carry: torch.Tensor | None = None, block: int = 128):
    """Causal exponential moving average along dim 1: y_t = a y_{t-1} + (1 - a) x_t, y_{-1} = carry (zeros if None).
    x (B, T, D) float32. Blocked closed form (one (block x block) decay matrix per block, carry between blocks).
    Returns (y, last row as the next carry)."""
    B, T, D = x.shape
    y_prev = torch.zeros(B, D, dtype=x.dtype, device=x.device) if carry is None else carry
    i = torch.arange(block, device=x.device)
    P = (a ** (i[:, None] - i[None, :]).clamp(min=0).to(x.dtype)) * (i[:, None] >= i[None, :])
    pw = a ** (i + 1).to(x.dtype)
    out = []
    for s in range(0, T, block):
        xb = x[:, s: s + block]
        n = xb.shape[1]
        yb = (1 - a) * torch.einsum("ij,bjd->bid", P[:n, :n], xb) + pw[:n, None] * y_prev[:, None, :]
        y_prev = yb[:, -1]
        out.append(yb)
    return torch.cat(out, 1), y_prev


def context_features(h: torch.Tensor, mode: str, alphas=(), mask: torch.Tensor | None = None, carry=None):
    """Map inputs from the cut's input h (B, T, D): [x, EMA_a(x) for a in alphas], x = n(h) (normed) or h. The EMAs
    are a cheap linear stand-in for the context mixing the skipped layers would do (DeltaNet layers are decay-weighted
    sums themselves). mask (B, T): padding contributes zeros. Returns ((B, T, (1 + len(alphas)) D) float32, carries)."""
    x = _rms(h) if mode == "normed" else h.float()
    feats, new = [x], []
    if alphas:
        xm = x if mask is None else x * mask[..., None].to(x.dtype)
        for k, a in enumerate(alphas):
            e, c = ema_scan(xm, a, None if carry is None else carry[k])
            feats.append(e)
            new.append(c)
    return torch.cat(feats, -1), new


class SkipMaps:
    """One ridge map per layer j >= cut: W (U, F D, D), b (U, D) in bf16 (F = 1 + len(alphas): the cut's input and its
    causal EMAs). tail: the last `tail` tokens of every state pass run all layers (on top of the skipped cache). A plain
    object, not a module: attaching it to the backbone adds nothing to its state dict."""

    def __init__(self, cut: int, mode: str, W: torch.Tensor, b: torch.Tensor, lam: float = 0.0, alphas=(),
                 tail: int = 0):
        assert mode in ("input", "normed")
        self.cut, self.mode, self.lam, self.W, self.b = cut, mode, lam, W, b
        self.alphas, self.tail = tuple(alphas), tail
        self.meta: dict = {}
        self.reset()

    def reset(self):
        """Start of a state pass: no EMA carry, no cached features (skipping() calls it)."""
        self._carry, self._fh, self._feat, self._mask = None, None, None, None

    def predict(self, j: int, h: torch.Tensor) -> torch.Tensor:
        if self._fh is not h:                         # once per chunk: every upper layer receives the same tensor
            f, self._carry = context_features(h, self.mode, self.alphas, self._mask, self._carry or None)
            self._fh, self._feat = h, f.to(h.dtype).reshape(-1, f.shape[-1])
        i = j - self.cut
        return torch.addmm(self.b[i].to(h.dtype), self._feat, self.W[i].to(h.dtype)).view(h.shape)

    def save(self, path: str, meta: dict | None = None):
        torch.save({"cut": self.cut, "mode": self.mode, "lam": self.lam, "W": self.W.cpu(), "b": self.b.cpu(),
                    "alphas": list(self.alphas), "meta": meta or self.meta}, path)

    @classmethod
    def load(cls, path: str, device="cuda", tail: int = 0) -> "SkipMaps":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(d["cut"], d["mode"], d["W"].to(device), d["b"].to(device), d.get("lam", 0.0), d.get("alphas", ()),
                tail)
        m.meta = d.get("meta", {})
        return m


def _stock(mixer: nn.Module) -> bool:
    """transformers' own Qwen3.5 attention / gated DeltaNet, unpatched (the cache-only paths below re-implement
    exactly those). Ported or converted mixers (Mercurius: forward swapped per instance, MLA latent k_proj, channel-wise
    decay) take the generic path."""
    return ("forward" not in mixer.__dict__ and type(mixer).__name__ in ("Qwen3_5Attention", "Qwen3_5GatedDeltaNet")
            and not hasattr(getattr(mixer, "k_proj", None), "latent") and not hasattr(mixer, "_decay"))


def _merc(mixer: nn.Module, module: str, name: str) -> bool:
    """The instance's forward is opendecider.<module>.<name> (Mercurius's ported / compressed-cache forwards)."""
    f = getattr(mixer.__dict__.get("forward"), "__func__", None)
    return f is not None and f.__module__ == f"opendecider.{module}" and f.__name__ == name


class CacheOnlyLayer(nn.Module):
    """Stands in for decoder layer j during a state pass: writes layer j's cache from the predicted input and returns
    its input unchanged (so every layer above the cut sees h = the cut's input)."""

    def __init__(self, layer: nn.Module, j: int, maps: SkipMaps):
        super().__init__()
        self.layer, self.j, self.maps = layer, j, maps

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                past_key_values=None, **kw):
        layer = self.layer
        x = self.maps.predict(self.j, hidden_states)
        if self.maps.mode == "input":
            x = layer.input_layernorm(x)
        mq = sys.modules[type(layer).__module__]
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            if _stock(la):
                self._gdn(la, x, attention_mask, past_key_values, mq)
            elif _merc(la, "mercurius_port", "_forward"):
                self._merc_gdn(la, x, attention_mask, past_key_values)
            else:                       # other DeltaNet forms (Mercurius GDN-2 / KDA): the whole mixer, MLP skipped
                la(hidden_states=x, cache_params=past_key_values, attention_mask=attention_mask)
        else:
            att = layer.self_attn
            if _stock(att):
                self._attn(att, x, position_embeddings, past_key_values, mq)
            elif _merc(att, "mercurius_cache", "_compressed_forward") and position_ids is not None:
                self._merc_attn(att, x, position_ids, past_key_values)
            else:                       # other attention (Mercurius MLA): the whole mixer, MLP skipped
                att(hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, position_embeddings=position_embeddings, **kw)
        return hidden_states

    @staticmethod
    def _attn(att, x, position_embeddings, cache, mq):
        shape = (*x.shape[:-1], -1, att.head_dim)
        k = att.k_norm(att.k_proj(x).view(shape)).transpose(1, 2)
        v = att.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        _, k = mq.apply_rotary_pos_emb(k[:, :0], k, cos, sin)       # RoPE on k only (q slice is empty)
        cache.update(k, v, att.layer_idx)

    @staticmethod
    def _merc_attn(sa, x, position_ids, cache):
        """Mercurius MLA with its TurboQuant cache: this call's latent / RoPE-key codes only (mercurius_cache's
        _compressed_forward up to the cache write)."""
        from .mercurius_cache import LatentCodesLayer, _encode_codes
        B, T = x.shape[:2]
        parts = _encode_codes(sa, x)
        parts["pos"] = position_ids.to(torch.int32).expand(B, T)
        layer = cache.layers[sa.layer_idx]
        if not isinstance(layer, LatentCodesLayer):
            if layer.get_seq_length():
                raise ValueError("cache layer already holds uncompressed K/V")
            layer = cache.layers[sa.layer_idx] = LatentCodesLayer()
        layer.append(**parts)

    @staticmethod
    def _merc_gdn(la, x, mask, cache):
        """Mercurius GDN-2 / KDA (mercurius_port._forward) up to the state update: no z, gated norm or out_proj."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
        x = apply_mask_to_padding_states(x, mask)
        B, T, _ = x.shape
        continuing = cache.has_previous_state(la.layer_idx, state_idx=0)
        qkv = la.in_proj_qkv(x).transpose(1, 2)
        g = la._decay(x)
        qkv = cache.update_conv_state(qkv, la.layer_idx, conv_kernel_size=la.conv_kernel_size)
        qkv = la._conv(qkv)[:, :, -T:].transpose(1, 2)
        q, k, v = torch.split(qkv, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        q = q.reshape(B, T, -1, la.head_k_dim)
        k = k.reshape(B, T, -1, la.head_k_dim)
        v = v.reshape(B, T, -1, la.head_v_dim)
        if la.num_v_heads // la.num_k_heads > 1:
            rep = la.num_v_heads // la.num_k_heads
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        init = cache.layers[la.layer_idx].recurrent_states[0] if continuing else None
        if init is not None:
            init = init.float()
        q, k = la._decay_phase(q, k, g)
        _, state = la._run_kernel(q, k, v, g, x, init, True, False, None)
        cache.update_recurrent_state(state, la.layer_idx)

    @staticmethod
    def _gdn(la, x, mask, cache, mq):
        B, T, _ = x.shape
        if cache.has_previous_state(la.layer_idx, state_idx=0) and T == 1:   # one-token continuation: the full layer
            la(x, cache_params=cache, attention_mask=mask)
            return
        x = mq.apply_mask_to_padding_states(x, mask)
        use_prev = cache.has_previous_state(la.layer_idx, state_idx=0)
        qkv = la.in_proj_qkv(x).transpose(1, 2)
        b, a = la.in_proj_b(x), la.in_proj_a(x)
        qkv = cache.update_conv_state(qkv, la.layer_idx, conv_kernel_size=la.conv_kernel_size)
        qkv = mq.causal_conv1d_fn(qkv, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation)
        qkv = qkv[:, :, -T:].transpose(1, 2)
        q, k, v = torch.split(qkv, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        q = q.reshape(B, T, -1, la.head_k_dim)
        k = k.reshape(B, T, -1, la.head_k_dim)
        v = v.reshape(B, T, -1, la.head_v_dim)
        beta = b.sigmoid()
        g = -la.A_log.float().exp() * nn.functional.softplus(a.float() + la.dt_bias)
        if la.num_v_heads // la.num_k_heads > 1:
            q = q.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
            k = k.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
        init = cache.layers[la.layer_idx].recurrent_states[0] if use_prev else None
        _, state = mq.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=init,
                                                   output_final_state=True, use_qk_l2norm_in_kernel=True)
        cache.update_recurrent_state(state, la.layer_idx)


@contextlib.contextmanager
def skipping(lm: nn.Module, maps: SkipMaps):
    """Layers >= maps.cut replaced by CacheOnlyLayer for the duration (state passes only)."""
    saved = {}
    maps.reset()
    try:
        for j in range(maps.cut, len(lm.layers)):
            saved[j] = lm.layers[j]
            lm.layers[j] = CacheOnlyLayer(saved[j], j, maps)
        yield
    finally:
        for j, layer in saved.items():
            lm.layers[j] = layer


# ---------------------------------------------------------------------------------------------------- fitting
@contextlib.contextmanager
def _capture(lm, cut: int):
    """Inputs of layers cut..L-1 during exact state passes: {j: (T, D)} (batch of one)."""
    store, hs = {}, []
    for j in range(cut, len(lm.layers)):
        hs.append(lm.layers[j].register_forward_pre_hook(
            lambda m, a, k, j=j: store.__setitem__(j, (a[0] if a else k["hidden_states"])[0].detach()),
            with_kwargs=True))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def _pairs(bb, text: str, cut: int, mode: str, alphas=()):
    """(X (T, F D + 1) float32 with a ones column, {j: target (T, D) float32}) for one state."""
    lm = bb.lm
    ids = bb.tokenize([text])
    with _capture(lm, cut) as st:
        bb.prefix_cache_batch(ids)
    X = context_features(st[cut][None], mode, alphas)[0][0]
    X = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    Y = {j: (lm.layers[j].input_layernorm(st[j]).float() if mode == "normed" else st[j].float())
         for j in range(cut, len(lm.layers))}
    return X, Y


@torch.no_grad()
def fit_skip_maps(bb, texts, held_texts, cut: int, mode: str = "normed", lams=LAMS, log=print, alphas=()) -> SkipMaps:
    """Ridge per layer j >= cut on [x, 1] from the normal equations (float64 sums on the GPU); one ridge strength
    (relative to the mean diagonal) for all layers, by held-out squared error. Calls bb with the skip off."""
    prev, bb.skip_state = getattr(bb, "skip_state", None), None
    try:
        L, D, dev = len(bb.lm.layers), bb.hidden_size, bb.device
        Df = (1 + len(alphas)) * D
        XtX = torch.zeros(Df + 1, Df + 1, device=dev, dtype=torch.float64)
        XtY = {j: torch.zeros(Df + 1, D, device=dev, dtype=torch.float64) for j in range(cut, L)}
        n = 0
        for i, t in enumerate(texts):
            X, Y = _pairs(bb, t, cut, mode, alphas)
            XtX += (X.T @ X).double()
            for j in Y:
                XtY[j] += (X.T @ Y[j]).double()
            n += len(X)
            if (i + 1) % 50 == 0:
                log(f"  fit cut {cut} {mode}: {i + 1}/{len(texts)} states, {n} tokens")
        held = [_pairs(bb, t, cut, mode, alphas) for t in held_texts]
        scale = float(XtX.diagonal()[:-1].mean())
        ev, Q = torch.linalg.eigh(XtX)
        ev, Q = ev.float(), Q.float()
        P = {j: Q.T @ XtY[j].float() for j in XtY}
        del XtY
        err = {}
        for lam in lams:
            e = 0.0
            for j in P:
                W = Q @ (P[j] / (ev + lam * scale)[:, None])
                e += sum(float((X @ W - Y[j]).pow(2).sum()) for X, Y in held)
            err[lam] = e
        lam = min(err, key=err.get)
        Ws = torch.empty(L - cut, Df, D, dtype=torch.bfloat16, device=dev)
        bs = torch.empty(L - cut, D, dtype=torch.bfloat16, device=dev)
        rel = {}
        for j in P:
            W = Q @ (P[j] / (ev + lam * scale)[:, None])
            Ws[j - cut], bs[j - cut] = W[:-1].bfloat16(), W[-1].bfloat16()
            num = sum(float((X @ W - Y[j]).pow(2).sum()) for X, Y in held)
            rel[j] = (num / sum(float(Y[j].pow(2).sum()) for _, Y in held)) ** 0.5
        maps = SkipMaps(cut, mode, Ws, bs, lam, alphas)
        maps.meta = {"n_tokens": n, "n_states": len(texts), "lambda": lam, "held_err": {str(k): v for k, v in err.items()},
                     "held_rel_err": {str(j): v for j, v in rel.items()}}
        return maps
    finally:
        bb.skip_state = prev


def skipped_share(cfg_text, cut: int, whole_mixer: bool = False) -> dict:
    """Share of the state pass's weight-multiply work skipped at `cut` (per token; attention scores not counted), from
    the text config: upper layers keep k/v (attention) or qkv/a/b (DeltaNet) plus the map (D x D); whole_mixer=True
    (non-stock mixers, e.g. Mercurius) keeps the whole mixer and skips only the MLP. Stock Qwen3.5 projection sizes
    (Mercurius's MLA latents are smaller, so its figure is approximate)."""
    c, D = cfg_text, cfg_text.hidden_size
    mlp = 3 * D * c.intermediate_size
    hd = getattr(c, "head_dim", D // c.num_attention_heads)
    attn_full = D * (2 * c.num_attention_heads * hd) + 2 * D * c.num_key_value_heads * hd + c.num_attention_heads * hd * D
    attn_keep = 2 * D * c.num_key_value_heads * hd
    kd, vd = c.linear_num_key_heads * c.linear_key_head_dim, c.linear_num_value_heads * c.linear_value_head_dim
    gdn_full = D * (2 * kd + vd) + 2 * D * vd + 2 * D * c.linear_num_value_heads
    gdn_keep = D * (2 * kd + vd) + 2 * D * c.linear_num_value_heads
    full = keep = 0
    for j, t in enumerate(c.layer_types):
        f = (attn_full if t == "full_attention" else gdn_full) + mlp
        full += f
        if whole_mixer:
            keep += f if j < cut else f - mlp + D * D
        else:
            keep += f if j < cut else (attn_keep if t == "full_attention" else gdn_keep) + D * D
    return {"skipped": 1 - keep / full, "full_per_token": full, "kept_per_token": keep}
