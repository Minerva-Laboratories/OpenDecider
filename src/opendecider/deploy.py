"""CUDA-graph inference engine for V3 branched models (single-state requests).

Measured result (docs/RESULTS.md §12): on the Orin the backbone passes are compute-bound, and this engine is SLOWER
than eager (bucket padding, explicit masks instead of the causal fast path), so it is off by default. It is kept for
GPUs where launch overhead dominates. It captures the two backbone passes as CUDA graphs over bucketed shapes:
  state pass : key T_b (state tokens rounded up to a bucket). The state is LEFT-padded: pads leave the GDN
               recurrent/conv state at zero and are masked in attention, so bucketing is exact.
  option rows: key (T_b, R_b, L_b). The shared prefix K/V (dequantized once), the prefix mask and each row's GDN
               state live in static buffers that are refilled before every replay; shared-prefix attention runs in
               its static-shape mode (pads masked, not sliced). Rows are right-padded.
Anything outside the buckets, multi-state batches and training fall back to the eager path. All graphs share one
memory pool; every output is copied out before the next replay.

    OPENDECIDER_CUDA_GRAPHS=1  (Decider installs the engine on load)
"""
from __future__ import annotations

import bisect
import contextlib
import os

import torch

from .backbone import PackedCache
from .prefix_attn import shared_prefix_attention
from .quant import Int8Tensor, kv_mode, materialize

T_BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 16384)
R_BUCKETS = (4, 8, 16, 32, 64)
L_BUCKETS = (16, 32, 64, 128, 256)
_DEBUG_EAGER = os.environ.get("OPENDECIDER_GRAPHS_DEBUG") == "1"


def _bucket(x, buckets):
    i = bisect.bisect_left(buckets, x)
    return buckets[i] if i < len(buckets) else None


class GraphEngine:
    def __init__(self, model):
        self.m, self.bb = model, model.backbone
        self.pool = torch.cuda.graph_pool_handle()
        self.state_graphs, self.row_graphs = {}, {}
        self.eager_state, self.eager_rows = model._state_prefix, model._rows_forward
        self.stats = {"state_graph": 0, "state_eager": 0, "rows_graph": 0, "rows_eager": 0}

    def install(self):
        self.m._state_prefix = self.state_prefix
        self.m._rows_forward = self.rows_forward
        return self

    # ------------------------------------------------------------------ helpers
    def _vera(self, on_state: bool):
        vera = getattr(self.m, "vera", None)
        if vera is None or (on_state and self.m.cfg.v3_vera_scope != "all"):
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def ctx():
            prev = vera.active
            vera.active = True
            try:
                yield
            finally:
                vera.active = prev
        return ctx()

    def _capture(self, fn):
        """Warm up on a side stream, then capture fn() into a graph in the shared pool; returns (graph, output)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn()
        return g, out

    # ------------------------------------------------------------------ state pass
    def state_prefix(self, mem, grad=None, quantize=None):
        seqs = mem.extra_prefix
        n = len(seqs[0]) if len(seqs) == 1 else 0
        Tb = _bucket(n, T_BUCKETS) if n else None
        if Tb is None or grad or quantize is not None or self.m.training or not self.m.state_free:
            self.stats["state_eager"] += 1
            return self.eager_state(mem, grad=grad, quantize=quantize)
        e = self.state_graphs.get(Tb)
        if e is None:
            bb, dev = self.bb, self.bb.device
            ids = torch.full((1, Tb), bb.pad_id, dtype=torch.long, device=dev)
            mask = torch.ones(1, Tb, dtype=torch.bool, device=dev)
            mask[0, 0] = False                        # capture the padded-mask code path (replays always pad)

            def fn():
                with torch.no_grad(), self._vera(True):
                    return bb(ids, mask, ("final",), use_cache=True)[1]
            g, cache = self._capture(fn)
            e = self.state_graphs[Tb] = (g, ids, mask, cache)
        g, ids, mask, cache = e
        ids.fill_(self.bb.pad_id); mask.fill_(False)
        ids[0, Tb - n:] = torch.tensor(seqs[0], device=ids.device)
        mask[0, Tb - n:] = True
        g.replay()
        pc = PackedCache.pack(cache, kv_int8=kv_mode(self.bb.cfg), state_int8=self.bb.cfg.quantize_linear_state)
        for k, t in pc.tensors.items():               # graph-owned memory is reused by the next replay: copy out
            if not isinstance(t, Int8Tensor):
                pc.tensors[k] = t.clone()
        self.stats["state_graph"] += 1
        return pc, mask.clone(), None

    # ------------------------------------------------------------------ option rows
    def rows_forward(self, ids, mask, rows, pc, smask):
        R, L = ids.shape
        T = smask.shape[1]
        Tb, Rb, Lb = _bucket(T, T_BUCKETS), _bucket(R, R_BUCKETS), _bucket(L, L_BUCKETS)
        single = pc.tensors[next(iter(pc.tensors))].shape[0] == 1
        if None in (Tb, Rb, Lb) or not single or self.m.training or not self.m.cfg.v3_shared_prefix:
            self.stats["rows_eager"] += 1
            return self.eager_rows(ids, mask, rows, pc, smask)
        key = (Tb, Rb, Lb)
        e = self.row_graphs.get(key) or self._capture_rows(pc, Tb, Rb, Lb)
        self.row_graphs[key] = e
        # refill static inputs: row tokens, prefix K/V (left-padded into T_b), prefix mask, per-row GDN states
        e["ids"].fill_(self.bb.pad_id); e["mask"].fill_(True)
        e["ids"][:R, :L] = ids; e["mask"][:R, :L] = mask
        e["pmask"].fill_(False); e["pmask"][0, Tb - T:] = smask[0]
        e["past"].copy_(e["pmask"].expand(Rb, Tb))
        for (li, a), buf in e["kv"].items():
            buf.zero_()
            buf[..., Tb - T:, :] = materialize(pc.tensors[(li, a, None)]).to(buf.dtype)
        for key_, buf in e["state"].items():
            buf.copy_(materialize(pc.tensors[key_]).to(buf.dtype).expand_as(buf))
        if _DEBUG_EAGER:                             # same static buffers, no graph: precise error locations
            e["out"] = e["fn"]()
        else:
            e["g"].replay()
        self.stats["rows_graph"] += 1
        return e["out"][:R, :L]

    def _capture_rows(self, pc, Tb, Rb, Lb):
        bb, m, dev = self.bb, self.m, self.bb.device
        e = {"ids": torch.full((Rb, Lb), bb.pad_id, dtype=torch.long, device=dev),
             "mask": torch.ones(Rb, Lb, dtype=torch.bool, device=dev),
             "pmask": torch.ones(1, Tb, dtype=torch.bool, device=dev),
             "past": torch.ones(Rb, Tb, dtype=torch.bool, device=dev), "kv": {}, "state": {}}
        e["mask"][:, -1] = False; e["pmask"][0, 0] = False; e["past"][:, 0] = False   # padded code paths
        tensors = {}
        for (li, a, k), t in pc.tensors.items():
            if a in ("keys", "values"):
                sh = list(t.shape); sh[-2] = Tb
                buf = e["kv"][(li, a)] = torch.zeros(sh, dtype=bb.dtype, device=dev)
                tensors[(li, a, k)] = buf
            else:
                x = materialize(t)
                buf = e["state"][(li, a, k)] = torch.zeros((Rb, *x.shape[1:]), dtype=x.dtype, device=dev)
                tensors[(li, a, k)] = buf
        static_pc = PackedCache(pc.template, tensors, Tb)
        rows0 = torch.zeros(Rb, dtype=torch.long, device=dev)
        rows_all = torch.arange(Rb, device=dev)

        def fn():
            with torch.no_grad(), self._vera(False):
                # per-row GDN states: static per-row buffers (Rb rows); attention: shared static prefix, pads masked
                cache = PackedCache(pc.template, {**{k: v for k, v in tensors.items() if k[1] in ("keys", "values")},
                                                  **e["state"]}, Tb).unpack_rows(rows_all, attn_placeholder=True)
                with shared_prefix_attention(bb, static_pc, rows0, e["pmask"], static=True):
                    f, _ = bb(e["ids"], e["mask"], layers=bb.cfg.feature_layers, past_key_values=cache,
                              use_cache=True, past_mask=e["past"])
                return m._wide(f)
        e["fn"] = fn
        e["g"], e["out"] = self._capture(fn)
        return e
