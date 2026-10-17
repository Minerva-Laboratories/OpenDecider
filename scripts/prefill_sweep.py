"""Prompt-pass (prefill) speed from 1k to 1M tokens, of the deployed trunk alone: no decision head, no LM head, no
continuation. The same pass serves the decision model (state memory) and LLM use (prompt cache), so this measures
the engine, not either use.

Per model, arm (exact / cut24 / cut16, the SwiftKV skip of src/opendecider/skipkv.py) and length: wall time of one
prompt pass in 4,096-token slices (median of 3 after a warm-up below 64k, one timing from 64k), tokens/s, peak GPU
memory over the weights and the cache size. Prompts are FineWeb-Edu documents concatenated and cut to length.

Long prompts on the stock attention layers (Qwen3.5-9B): the stock cache grows by torch.cat (a full copy of every
layer's K/V per slice) and the SDPA path builds a (slice x prompt) mask and expands K/V to every query head. Past
~200k tokens that exceeds memory (RESULTS 21.3: 262k stopped at the 12 GB guard). --lean (default on) instead writes
K/V into a buffer allocated once for the whole prompt and attends each slice to its prefix with the wide flash
kernel (wide_attention: GQA without expansion, causal with offset, no mask). Same math; --check compares the two
paths' final hidden states at a short length. Mercurius's MLA layers keep their deployed compressed cache (4-bit
latents, mercurius_cache), which is already small.

A length is skipped when the fit of the measured times (a T + b T^2 over the last two lengths) predicts more than
--max-hours for it, or when cache + weights would leave less than the memory guard.

    python scripts/prefill_sweep.py --ckpt runs/x9b-clean3/model.pt --out /mnt/ext/opendecider/prefill/sweep_9b.json
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.skipkv import SkipMaps  # noqa: E402
from skipkv_lm import DATA  # noqa: E402

GB = 2 ** 30
LENGTHS = [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]


# ---------------------------------------------------------------------------------------------- lean long-prompt path
def _prealloc_cls(kv8=False):
    from transformers.cache_utils import DynamicLayer
    body = {k: v for k, v in vars(_Prealloc).items() if not k.startswith("__") or k == "__init__"}
    if kv8:
        body.update({k: v for k, v in vars(_Prealloc8).items() if not k.startswith("__")})
    return type("PreallocLayer8" if kv8 else "PreallocLayer", (DynamicLayer,), body)


class _Prealloc:
    """Full-attention cache layer with K/V written into one buffer sized for the whole prompt (no per-slice copy).
    Methods grafted onto transformers' DynamicLayer (the Cache only counts CacheLayerMixin layers as attention)."""
    is_croppable = False

    def __init__(self, tmax):
        self.tmax, self.n, self.is_initialized = tmax, 0, False
        self.keys = self.values = None

    def lazy_initialization(self, k, v):
        B, H, _, D = k.shape
        self.keys = k.new_empty(B, H, self.tmax, D)
        self.values = v.new_empty(B, H, self.tmax, v.shape[-1])
        self.dtype, self.device, self.is_initialized = k.dtype, k.device, True

    def update(self, k, v, *a, **kw):
        if not self.is_initialized:
            self.lazy_initialization(k, v)
        t = k.shape[2]
        self.keys[:, :, self.n:self.n + t] = k
        self.values[:, :, self.n:self.n + t] = v
        self.n += t
        return self.keys[:, :, :self.n], self.values[:, :, :self.n]

    def get_seq_length(self):
        return self.n

    def get_mask_sizes(self, query_length):
        return self.n + query_length, 0

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        self.n = 0


def lean_attention(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    """q (B, Hq, t, D) = the slice's queries; k, v (B, Hkv, S, D) = prefix + slice (views of the buffer). Causal with
    offset S - t, GQA without expansion. Returns (B, t, Hq, D) like the stock interfaces."""
    if isinstance(k, Q8) and q.is_cuda:
        o = _wide8()(q, k, v, scaling)
    elif q.is_cuda:
        from opendecider.wide_attention import wide_attention
        o = wide_attention(q, k, v, scaling, None, causal=True)
    else:                                                       # CPU check path: same math, explicit mask
        if isinstance(k, Q8):
            k, v = ((x.x.float() * x.s[..., None]).to(q.dtype) for x in (k, v))
        t, S = q.shape[2], k.shape[2]
        m = torch.arange(S)[None] <= torch.arange(S - t, S)[:, None]
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scaling, enable_gqa=True)
    return o.transpose(1, 2).contiguous(), None


class Q8:
    """int8 K or V view (B, G, S, D) with its per-(token, head) scale (B, G, S) fp32."""
    def __init__(self, x, s):
        self.x, self.s = x, s


def _quant8(x):
    s = x.float().abs().amax(-1).clamp_min(1e-8) / 127.0
    return torch.round(x.float() / s[..., None]).clamp_(-127, 127).to(torch.int8), s


_KERNEL = None


def _wide8():
    """wide_attention's kernel (src/opendecider/wide_attention.py) reading int8 K/V: Q.K on the int8 values (exact in
    bf16), then each key's column scaled by its K scale; P scaled by each key's V scale before P.V."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def fwd(Q, K, V, KS, VS, O, scale_log2, T, S, G, DK, sqb, sqh, sqt, skb, skg, sks, svb, svg, svs, ssb, ssg,
            sob, soh, sot, REP: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
            DKP: tl.constexpr, DV: tl.constexpr):
        BM: tl.constexpr = BT * REP
        pid_m = tl.program_id(0)
        bg = tl.program_id(1)
        b = bg // G
        g = bg % G
        rows = tl.arange(0, BM)
        tok = pid_m * BT + rows // REP
        head = g * REP + rows % REP
        qpos = tok + (S - T)
        offs_d = tl.arange(0, BD)
        offs_v = tl.arange(0, DV)
        q_rows = Q + b * sqb + head * sqh + tok * sqt
        k_base = K + b * skb + g * skg
        v_base = V + b * svb + g * svg
        s_off = b * ssb + g * ssg
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, DV], tl.float32)
        end = tl.minimum(S, (pid_m + 1) * BT + S - T)
        for n0 in range(0, end, BN):
            offs_n = n0 + tl.arange(0, BN)
            nin = offs_n < S
            qk = tl.zeros([BM, BN], tl.float32)
            for d0 in tl.static_range(0, DKP, BD):
                dd = d0 + offs_d
                q = tl.load(q_rows[:, None] + dd[None, :], mask=(tok[:, None] < T) & (dd[None, :] < DK), other=0.0)
                k = tl.load(k_base + offs_n[:, None] * sks + dd[None, :], mask=nin[:, None] & (dd[None, :] < DK),
                            other=0)
                qk += tl.dot(q, tl.trans(k.to(q.dtype)))
            ks = tl.load(KS + s_off + offs_n, mask=nin, other=0.0)
            vs = tl.load(VS + s_off + offs_n, mask=nin, other=0.0)
            ok = nin[None, :] & (offs_n[None, :] <= qpos[:, None])
            qk = tl.where(ok, qk * ks[None, :] * scale_log2, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_base + offs_n[:, None] * svs + offs_v[None, :], mask=nin[:, None], other=0)
            acc = acc * alpha[:, None] + tl.dot((p * vs[None, :]).to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(O + b * sob + head[:, None] * soh + tok[:, None] * sot + offs_v[None, :],
                 out.to(O.dtype.element_ty), mask=tok[:, None] < T)

    def run(q, k8, v8, scale, BM=64, BN=64, BD=32, num_warps=4, num_stages=2):
        B, H, T, DK = q.shape
        k, ks, v, vs = k8.x, k8.s, v8.x, v8.s
        G, S, DV = k.shape[1], k.shape[2], v.shape[-1]
        rep = H // G
        q = q if q.stride(-1) == 1 else q.contiguous()
        o = q.new_empty(B, H, T, DV)
        bt = BM // rep
        grid = (triton.cdiv(T, bt), B * G)
        fwd[grid](q, k, v, ks, vs, o, scale * 1.4426950408889634, T, S, G, DK, q.stride(0), q.stride(1), q.stride(2),
                  k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), ks.stride(0),
                  ks.stride(1), o.stride(0), o.stride(1), o.stride(2), REP=rep, BT=bt, BN=BN, BD=BD,
                  DKP=triton.cdiv(DK, BD) * BD, DV=DV, num_warps=num_warps, num_stages=num_stages)
        return o

    _KERNEL = run
    return run


class _Prealloc8(_Prealloc):
    """_Prealloc storing int8 K/V + per-(token, head) fp32 scales (about half the bytes of bf16)."""

    def lazy_initialization(self, k, v):
        B, H, _, D = k.shape
        self.keys = torch.empty(B, H, self.tmax, D, dtype=torch.int8, device=k.device)
        self.values = torch.empty(B, H, self.tmax, v.shape[-1], dtype=torch.int8, device=k.device)
        self.kscale = torch.empty(B, H, self.tmax, dtype=torch.float32, device=k.device)
        self.vscale = torch.empty_like(self.kscale)
        self.dtype, self.device, self.is_initialized = k.dtype, k.device, True

    def update(self, k, v, *a, **kw):
        if not self.is_initialized:
            self.lazy_initialization(k, v)
        t, n = k.shape[2], self.n
        self.keys[:, :, n:n + t], self.kscale[:, :, n:n + t] = _quant8(k)
        self.values[:, :, n:n + t], self.vscale[:, :, n:n + t] = _quant8(v)
        self.n += t
        return (Q8(self.keys[:, :, :self.n], self.kscale[:, :, :self.n]),
                Q8(self.values[:, :, :self.n], self.vscale[:, :, :self.n]))


def _no_mask(*a, **kw):
    return None


def register_lean():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register("lean_prefill", lean_attention)
    AttentionMaskInterface.register("lean_prefill", _no_mask)


@contextlib.contextmanager
def lean(lm):
    cfg = lm.config
    old = cfg._attn_implementation
    cfg._attn_implementation = "lean_prefill"
    subs = [c for c in (getattr(cfg, "text_config", None),) if c is not None]
    olds = [c._attn_implementation for c in subs]
    for c in subs:
        c._attn_implementation = "lean_prefill"
    try:
        yield
    finally:
        cfg._attn_implementation = old
        for c, o in zip(subs, olds):
            c._attn_implementation = o


def new_cache(bb, tmax, lean_on, kv8=False):
    from transformers import DynamicCache
    cache = DynamicCache(config=bb.lm.config)
    if lean_on:
        for i, t in enumerate(bb.layer_types):
            if t == "full_attention":
                cache.layers[i] = _prealloc_cls(kv8)(tmax)
    return cache


@torch.no_grad()
def prompt_pass(bb, ids, maps, lean_on, keep_last=False, kv8=False):
    """One prompt pass in prefill_chunk slices over `ids` (list), skip maps `maps` (None = exact). Returns
    (cache, last hidden or None)."""
    x = torch.tensor([ids], device=bb.device)
    m = torch.ones_like(x, dtype=torch.bool)
    cache = new_cache(bb, len(ids), lean_on and _stock_attention(bb), kv8)
    C = bb.cfg.prefill_chunk or len(ids)
    prev = getattr(bb, "skip_state", None)
    bb.skip_state = maps
    ctx = lean(bb.lm) if (lean_on and _stock_attention(bb)) else contextlib.nullcontext()
    try:
        with ctx:
            if maps is not None:
                _, cache = bb.prefill(x, m, ("final",), cache, keep_hidden=False)
                return cache, None
            f, cache = bb._prefill(x, m, ("final",), cache, keep_last, C)
            return cache, (f[0, -8:].float() if keep_last else None)
    finally:
        bb.skip_state = prev


def _stock_attention(bb):
    from opendecider.skipkv import _stock
    return any(t == "full_attention" and _stock(l.self_attn) for t, l in zip(bb.layer_types, bb.lm.layers))


def cache_bytes(cache):
    n = 0
    for layer in cache.layers:
        for v in vars(layer).values():
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
            elif isinstance(v, (list, tuple)):
                n += sum(t.numel() * t.element_size() for t in v if torch.is_tensor(t))
    return n


def free_gb():
    with open("/proc/meminfo") as f:
        kv = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return kv["MemAvailable"] / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps-dir", default="/mnt/ext/opendecider/skipkv")
    ap.add_argument("--arms", nargs="*", default=["exact", "cut16", "cut24"])
    ap.add_argument("--lengths", nargs="*", type=int, default=LENGTHS)
    ap.add_argument("--max-hours", type=float, default=7.0, help="skip a length predicted to take longer")
    ap.add_argument("--guard-gb", type=float, default=14.0, help="free memory that must remain (memwatch kills at 12)")
    ap.add_argument("--chunk", type=int, default=4096, help="prefill slice length")
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="compare lean vs stock hidden states at this length, exit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lean_on = not a.no_lean
    register_lean()
    with gpu_lock("prefill_sweep"):
        model, _ = load_model(a.ckpt)
        if not a.check:                         # serving kernels (tuned int8 GEMMs), as deployed
            from opendecider.decider import prepare_inference_
            prepare_inference_(model)
        bb = model.backbone
        bb.skip_state = None
        bb.cfg.prefill_chunk = a.chunk
        name = os.path.basename(os.path.dirname(a.ckpt))
        texts = pq.read_table(DATA, columns=["text"]).column("text").to_pylist()
        need = max(a.lengths + [a.check])
        ids_all, i = [], 3000
        while len(ids_all) < need:
            ids_all += bb.tok(texts[i], add_special_tokens=False)["input_ids"]
            i += 1
        print(f"{need} prompt tokens from {i - 3000} documents", flush=True)
        if a.check:
            _, h0 = prompt_pass(bb, ids_all[:a.check], None, False, keep_last=True)
            _, h1 = prompt_pass(bb, ids_all[:a.check], None, True, keep_last=True)
            rel = float((h1 - h0).norm() / h0.norm())
            print(f"CHECK T={a.check} lean vs stock: rel. diff of last hidden states {rel:.2e}", flush=True)
            if _stock_attention(bb):
                _, h2 = prompt_pass(bb, ids_all[:a.check], None, True, keep_last=True, kv8=True)
                print(f"CHECK T={a.check} lean int8 K/V vs stock: {float((h2 - h0).norm() / h0.norm()):.2e}",
                      flush=True)
            return
        wgb = sum(p.numel() * p.element_size() for p in bb.lm.parameters()) / GB
        res = json.load(open(a.out)) if os.path.exists(a.out) else {}
        res.update({"ckpt": a.ckpt, "weights_gb": wgb, "lean": lean_on, "prefill_chunk": bb.cfg.prefill_chunk,
                    "lengths": a.lengths})
        arms = res.setdefault("arms", {})
        for arm in a.arms:
            maps, base = None, arm.removesuffix("_kv8")
            kv8 = arm.endswith("_kv8")
            if kv8 and not _stock_attention(bb):
                continue                                             # Mercurius: its cache is 4-bit latents already
            if base != "exact":
                maps = SkipMaps.load(os.path.join(a.maps_dir, f"{name}_lm_{base}_normed.pt"), bb.device)
            r = arms.setdefault(arm, {})
            if maps is not None:
                r["cut"] = maps.cut
            for T in a.lengths:
                key = str(T)
                if key in r and "seconds" in r[key]:
                    continue
                done = sorted((int(k), v["seconds"]) for k, v in r.items() if k.isdigit() and "seconds" in v)
                if len(done) >= 2:                                   # a T + b T^2 through the last two points
                    (t1, s1), (t2, s2) = done[-2:]
                    b = (s2 / t2 - s1 / t1) / (t2 - t1)
                    pred = (s2 / t2 + b * (T - t2)) * T
                    if pred > a.max_hours * 3600:
                        r[key] = {"skipped": f"predicted {pred / 3600:.1f} h > {a.max_hours} h"}
                        print(arm, T, r[key], flush=True)
                        continue
                cb = [v.get("cache_gb", 0) / k for k, v in ((int(k), v) for k, v in r.items() if k.isdigit())
                      if v.get("cache_gb")]
                est = (cb[-1] * T if cb else 0.0) * 1.15
                if free_gb() - est < a.guard_gb:
                    r[key] = {"skipped": f"memory: need ~{est:.1f} GB, {free_gb():.1f} GB available"}
                    print(arm, T, r[key], flush=True)
                    continue
                reps = 3 if T < 65536 else 1
                if T < 65536:
                    prompt_pass(bb, ids_all[:T], maps, lean_on, kv8=kv8)        # warm-up
                torch.cuda.synchronize()
                ts, cgb, peak = [], 0.0, 0.0
                for _ in range(reps):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    t0 = time.perf_counter()
                    cache, _ = prompt_pass(bb, ids_all[:T], maps, lean_on, kv8=kv8)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                    cgb = cache_bytes(cache) / GB
                    peak = (torch.cuda.max_memory_allocated() - base) / GB
                    del cache
                s = float(np.median(ts))
                r[key] = {"seconds": s, "tok_per_s": T / s, "cache_gb": cgb, "peak_over_weights_gb": peak,
                          "reps": reps, "free_gb_after": free_gb()}
                print(f"{name} {arm:6s} T={T:8d} {s:9.2f} s  {T / s:8.0f} tok/s  cache {cgb:6.2f} GB  "
                      f"peak {peak:6.2f} GB", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
                torch.cuda.empty_cache()
            if maps is not None:
                del maps
        json.dump(res, open(a.out, "w"), indent=1)
        print("PREFILL_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
