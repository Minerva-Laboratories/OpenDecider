"""Mercurius-1-4B (Minerva Laboratories) as an OpenDecider backbone, in its deployed form.

1. Transformers port. Mercurius's KDA / GDN-2 linear-attention forward targets transformers 5.6; OpenDecider runs on
   5.17, whose linear-attention cache API changed and which continues the recurrence natively. port_linear_attention
   swaps in 5.17's stock Qwen3_5GatedDeltaNet.forward with Mercurius's three changes kept as they are (channel-wise
   `_decay` / `_decay_phase`, its `_run_kernel`, its `_conv` with the kernel 5.6 attached). Nothing in Mercurius's own
   code is modified. Checked: same task accuracy as under 5.6, 95-98% identical answers (the 4-bit KV amplifies
   bf16-level kernel differences).
2. Deployed model (decisions #64, #64.1, #66, #68 of the Mercurius repo): deploy_mercurius + compact_storage_ turn the
   research build into fixed weights without undoing anything: merged weights and MLA latents as bitsandbytes NF4
   (quantized once, == the QAT fake-quant), GDN-2 / KDA gates in their exact factored form (tiled bases + VeRA factors,
   bf16), the NF4 embedding table, MLA with the decoupled RoPE key, TurboQuant 4-bit KV with the recorded rotations.
3. The compressed KV cache itself is opendecider.mercurius_cache (the cache holds the 4-bit codes).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

# The deployed configuration (Mercurius experiments/eval_v3_bench.sh): NF4 weights, TurboQuant 4-bit KV latent and
# RoPE key, bf16 gates, NF4 embedding. Files under the Mercurius working root.
MERCURIUS_QAT = {"kv_bits": 4, "kv_group": 32, "kv_quant": "tq", "kv_rot": "none", "gate_bits": 16, "embed_bits": 4}
MERCURIUS_FILES = {"adapters": "ckpt/adapters-v3-absorb-step3900.pt", "covs": "cache/mla_seqcal_667126e299_covs.pt",
                   "groups": "cache/mla_seqcal_667126e299_groups.json"}
GATES = ("in_proj_a", "in_proj_be", "in_proj_bw")          # bf16 (factored) in Mercurius's deployed form


def _forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
    if getattr(self, "_state_passing", False):
        raise NotImplementedError("Mercurius segment state passing is a training feature; not supported here")
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    continuing = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    g = self._decay(hidden_states)                                   # (B, T, H, Dk), channel-wise

    # always the chunk path (also for one new token): OpenDecider rows continue a state with any length >= 1
    if cache_params is not None:
        mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
    mixed_qkv = self._conv(mixed_qkv)
    if cache_params is not None:
        mixed_qkv = mixed_qkv[:, :, -seq_len:]                        # drop the previous frames

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    if self.num_v_heads // self.num_k_heads > 1:
        rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if continuing else None
    if recurrent_state is not None:
        recurrent_state = recurrent_state.float()                    # the chunk kernels need fp32
    query, key = self._decay_phase(query, key, g)
    # terminal rows (OpenDecider option rows: nothing continues from them) skip the per-row final state
    want_final = cache_params is not None and not getattr(cache_params, "terminal", False)
    core_attn_out, last_recurrent_state = self._run_kernel(
        query, key, value, g, hidden_states, recurrent_state, want_final, False, None)
    if want_final:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def _dao_conv():
    """Dao-AILab causal_conv1d when installed: the kernel transformers 5.6 attached to these layers."""
    try:
        from causal_conv1d import causal_conv1d_fn
        return causal_conv1d_fn
    except ImportError:
        return None


def port_linear_attention(model) -> dict:
    """Swap the forward of every Mercurius KDA / GDN-2 layer in `model` (in place). Returns counts and whether the
    decay phase is active (it is computed per pass, so a continuation restarts it; reported, not changed)."""
    from mercurius.models.kda import Qwen3_5KDAGatedDeltaNet
    trunk = model.model.language_model if hasattr(model.model, "language_model") else model.model
    n, phase = 0, 0
    for layer in trunk.layers:
        la = getattr(layer, "linear_attn", None)
        if isinstance(la, Qwen3_5KDAGatedDeltaNet):
            la.causal_conv1d_fn = _dao_conv()                         # what 5.6 attached (None -> fla / PyTorch)
            la.forward = types.MethodType(_forward, la)
            n += 1
            phase += getattr(la, "pe_c", None) is not None
    return {"layers": n, "decay_phase_layers": phase}


def _tq_latent_forward(self, x):
    """MLA latent with Mercurius's 4-bit TurboQuant in the loop (inference form of its QAT latent forward): the latent
    c = down(x) is rotated by the layer's recorded rotation, quantized (per-vector fp16 norm + Beta Lloyd-Max codebook),
    rotated back, then expanded to K and V. Same values as the research build; no STE, no weight caches."""
    from mercurius.models.qat import tq_roundtrip
    c = self.down(x)
    R = self._tq_rot.to(c.device, c.dtype)
    c = tq_roundtrip(c @ R.T, self._tq_bits) @ R
    return self.up_k(c), self.up_v(c)


# ---------------------------------------------------------------------------------------------- deployed form
def _bnb_dense(lin) -> torch.Tensor:
    """fp32 weight of a Linear / bitsandbytes Linear4bit (the values its forward uses)."""
    import bitsandbytes as bnb
    if isinstance(lin, bnb.nn.Linear4bit):
        return bnb.functional.dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
    return lin.weight.data.float()



def _bnb_nf4(W: torch.Tensor, bias=None):
    """bitsandbytes Linear4bit holding W quantized ONCE (NF4, blocksize 64, double-quantized statistics): the same
    call Mercurius's nf4_roundtrip makes, so the stored weight equals its QAT fake-quant bit for bit (decisions #64)."""
    import bitsandbytes as bnb
    import torch.nn as nn
    q = bnb.nn.Linear4bit(W.shape[1], W.shape[0], bias=bias is not None, compute_dtype=torch.bfloat16,
                          compress_statistics=True, quant_type="nf4", device="meta")
    q.weight = bnb.nn.Params4bit(W.detach().to("cpu", torch.bfloat16).contiguous(), requires_grad=False,
                                 quant_type="nf4", compress_statistics=True, blocksize=64)
    if bias is not None:
        q.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
    return q.to(W.device)


def deploy_mercurius(model) -> dict:
    """The deployed Mercurius-1-4B (decisions #64, #64.1, #68), in place, from the research build with QAT installed:
      - every merged weight (base + VeRA, per-head query maps) and the MLA latent projections as bitsandbytes NF4
        Linear4bit, quantized once from the merged value (== the QAT fake-quant);
      - GDN-2 / KDA gates in their exact factored form: VeRA's own forward (base + diag(b) B diag(d) A), bf16;
      - the MLA latent with 4-bit TurboQuant in the loop (recorded rotations), the decoupled RoPE key with its
        TurboQuant and NF4 rounding as in the trained forward;
      - the NF4-rounded tied embedding (as built).
    No QAT wrappers or weight caches remain; nothing is undone."""
    from mercurius.adapters.lora import VeRALinear
    from mercurius.models.qat import _base_weight, _phq_weight, _vera_delta
    from mercurius.surgery.perhead_q import PerHeadQ
    from mercurius.surgery.transmla import LatentKV
    mods = dict(model.named_modules())
    owned = {id(m.base) for m in mods.values() if isinstance(m, PerHeadQ)}
    swaps, n = [], {"nf4_merged": 0, "nf4_phq": 0, "gates_factored": 0, "latent_tq": 0}
    with torch.no_grad():
        for name, m in mods.items():
            if isinstance(m, PerHeadQ):
                swaps.append((name, _bnb_nf4(_phq_weight(m))))
                n["nf4_phq"] += 1
            elif isinstance(m, VeRALinear) and id(m) not in owned:
                if getattr(m, "_qat_gate", False):
                    for a in ("forward", "_qat", "_qat_gate", "_qat_cache"):
                        m.__dict__.pop(a, None)                       # VeRALinear.forward: the factored form
                    n["gates_factored"] += 1
                else:
                    swaps.append((name, _bnb_nf4(_base_weight(m.base) + _vera_delta(m), m.base.bias)))
                    n["nf4_merged"] += 1
        for name, new in swaps:
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, new)
        for m in model.modules():
            if isinstance(m, LatentKV) and hasattr(m, "_qat"):
                cfg, rot = m._qat, m._qat_rot
                if cfg.get("kv_quant") != "tq" or rot is None:
                    raise ValueError("expected a TurboQuant latent with a recorded rotation")
                for leaf in ("down", "up_k", "up_v"):
                    lin = getattr(m, leaf)
                    setattr(m, leaf, _bnb_nf4(lin.weight, lin.bias))
                for a in ("forward", "_qat", "_qat_rot", "_qat_cache"):
                    m.__dict__.pop(a, None)
                m._tq_rot, m._tq_bits = rot, int(cfg["kv_bits"])
                m.forward = types.MethodType(_tq_latent_forward, m)
                n["latent_tq"] += 1
    for m in model.modules():
        m.__dict__.pop("_qat_cache", None)
    torch.cuda.empty_cache()
    return n


def load_deployed(root: str, code: str | None = None, compressed_cache: bool = True):
    """(text model, tokenizer): Mercurius's build() with its deployed QAT settings, the transformers port, then
    deploy_mercurius. `root` is the Mercurius working root (ckpt/, cache/)."""
    import os
    import sys
    root = os.path.abspath(root)
    code = code or os.environ.get("MERCURIUS_CODE", "/mnt/ext/opendecider/code/mercurius")
    if os.path.join(code, "src") not in sys.path:
        sys.path.insert(0, os.path.join(code, "src"))
    os.environ["MERCURIUS_ROOT"] = root
    from transformers import AutoTokenizer
    from mercurius.eval.retrieval_ab import build
    from mercurius.paths import STAGE_AB
    here = os.getcwd()
    os.chdir(root)
    try:
        m = build(os.path.join(root, MERCURIUS_FILES["adapters"]), 512, os.path.join(root, MERCURIUS_FILES["covs"]),
                  quantize=True, groups=os.path.join(root, MERCURIUS_FILES["groups"]), dial="c0",
                  qat=dict(MERCURIUS_QAT)).eval()
    finally:
        os.chdir(here)
    port_linear_attention(m)
    deploy_mercurius(m)
    trunk = m.model.language_model if hasattr(m.model, "language_model") else m.model
    compact_storage_(trunk, root)
    if compressed_cache:
        from .mercurius_cache import install_compressed_cache
        install_compressed_cache(trunk)
    trunk.requires_grad_(False)
    return trunk, AutoTokenizer.from_pretrained(str(STAGE_AB))


# ---------------------------------------------------------------------------------------------- compact storage
class TiledLinear(torch.nn.Module):
    """A frozen Linear whose rows are U rows each repeated `rep` times consecutively (Mercurius's KDA / GDN-2 gate
    bases: in_proj_*.weight.repeat_interleave(Dk)): stores the U unique rows, y = (x U^T) repeated."""

    def __init__(self, unique: torch.Tensor, rep: int):
        super().__init__()
        self.register_buffer("unique", unique.contiguous())
        self.rep = rep
        self.in_features, self.out_features = unique.shape[1], unique.shape[0] * rep

    @classmethod
    def maybe_from(cls, lin):
        """TiledLinear if lin's weight is exactly row-tiled (no bias), else None."""
        if type(lin) is not torch.nn.Linear or lin.bias is not None:
            return None
        W = lin.weight.data
        O = W.shape[0]
        for U in sorted({d for d in range(1, O + 1) if O % d == 0}):
            if U >= O:
                break
            rep = O // U
            Wv = W.view(U, rep, -1)
            if torch.equal(Wv, Wv[:, :1].expand_as(Wv)):
                return cls(Wv[:, 0].clone(), rep)
        return None

    def forward(self, x):
        return F.linear(x, self.unique.to(x.dtype)).repeat_interleave(self.rep, dim=-1)


class NF4Table(torch.nn.Module):
    """Embedding table stored as NF4 the way Mercurius deploys it: the ORIGINAL table quantized once by bitsandbytes
    (NF4, blocksize 64, double-quantized statistics), kept as 4-bit codes + the dequantized fp32 block scales, decoded
    per row as fp32 code * scale -> table dtype (bitsandbytes' own arithmetic, bit for bit). `expected` (the model's
    NF4-rounded table) is checked to be reproduced exactly over every row. forward(ids) and dequant_rows(idx, dtype)
    like opendecider.quant's tables."""
    BLOCK = 64

    def __init__(self, original: torch.Tensor, expected: torch.Tensor):
        super().__init__()
        import bitsandbytes.functional as bnbf
        V, D = original.shape
        assert D % self.BLOCK == 0 and expected.shape == original.shape
        q, st = bnbf.quantize_4bit(original.to(expected.device, torch.bfloat16).contiguous(), blocksize=self.BLOCK,
                                   quant_type="nf4", compress_statistics=True)
        scale = (bnbf.dequantize_blockwise(st.absmax, st.state2) + st.offset).float().view(V, D // self.BLOCK)
        self.register_buffer("code", st.code.float().to(expected.device))
        self.register_buffer("absmax", scale)
        self.register_buffer("codes", q.view(V, D // 2).clone())        # bitsandbytes packing: element 2i high nibble
        self.num_embeddings, self.embedding_dim, self.dtype = V, D, expected.dtype
        del q, st
        for a in range(0, V, 8192):
            if not torch.equal(self.dequant_rows(slice(a, a + 8192), expected.dtype), expected[a:a + 8192]):
                raise ValueError(f"NF4Table does not reproduce rows {a}..{a + 8192} exactly")

    SLICE = 8192                                       # rows decoded at once: bounds the fp32 temporaries

    def dequant_rows(self, idx, dtype=torch.bfloat16):
        """Rows idx (index tensor or slice) in dtype: code * block scale in fp32 (bit-exact with bnb), then rounded.
        A byte -> (high, low) code-pair table replaces int64 nibble indices; long slices go 8k rows at a time."""
        lut = self.__dict__.get("_pair_lut")
        if lut is None or lut.device != self.code.device:
            b = torch.arange(256, device=self.code.device)
            lut = self.__dict__["_pair_lut"] = torch.stack([self.code[b >> 4], self.code[b & 15]], -1)   # (256, 2)
        if isinstance(idx, slice):
            a, e, _ = idx.indices(self.codes.shape[0])
            if e - a > self.SLICE:
                out = torch.empty(e - a, self.codes.shape[1] * 2, device=self.codes.device, dtype=dtype)
                for s in range(a, e, self.SLICE):
                    out[s - a:min(e, s + self.SLICE) - a] = self.dequant_rows(slice(s, min(e, s + self.SLICE)), dtype)
                return out
        p = self.codes[idx]
        v = lut[p.int()].view(*p.shape[:-1], -1, self.BLOCK) * self.absmax[idx].unsqueeze(-1)
        return v.flatten(-2).to(dtype)

    def forward(self, ids):
        return self.dequant_rows(ids, self.dtype)


def _original_embedding(root: str) -> torch.Tensor:
    """The embedding before Mercurius's QAT rounding: the stage A+B checkpoint's (unchanged base) table."""
    import glob
    import os
    from safetensors import safe_open
    for p in glob.glob(os.path.join(root, "ckpt", "*-stageAB", "*.safetensors")):
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return f.get_tensor(k)
    raise FileNotFoundError("no embed_tokens.weight in the stage A+B checkpoint")


def compact_storage_(trunk, root: str) -> dict:
    """Mercurius's deployed storage for what deploy_mercurius left dense: tiled gate bases, NF4 embedding table."""
    n = {"tiled_gate_bases": 0, "dense_left": 0}
    for name, mod in list(trunk.named_modules()):
        for leaf, child in list(mod.named_children()):
            if leaf == "base" and any(name.endswith(g) for g in GATES):
                t = TiledLinear.maybe_from(child)
                if t is None:
                    n["dense_left"] += 1
                else:
                    setattr(mod, leaf, t)
                    n["tiled_gate_bases"] += 1
    orig = _original_embedding(root)
    trunk.embed_tokens = NF4Table(orig, trunk.embed_tokens.weight.data)
    del orig
    torch.cuda.empty_cache()
    return n
