"""Upper layers' state cache from an earlier layer (SwiftKV SingleInputKV, arXiv 2410.03960; DeepSeek-V4.1-Flash's causal
encoder-decoder, arXiv 2609.19969, where each decoder layer projects its global KV from layer L/2) retrofitted to a
frozen OpenDecider backbone with no backprop.

The state (record) tokens run layers 1..c as usual. Every layer j > c builds its cache for the state (K/V in the
full-attention layers; conv and recurrent state in the gated DeltaNet layers) from a PREDICTED input instead of its
real one; the rest of those layers would not be computed for state tokens (here they still run and are discarded:
this script measures accuracy, the saving is estimated from the skipped compute). Option rows run every layer as
before and read that cache. Predicted input of layer j:
  identity  h_c, the cut layer's output (SwiftKV without its distillation step)
  ridge     h_c W_j + b_j, one ridge map per upper layer fitted in closed form on calibration states
            (data/synthetic/calib.jsonl: training-distribution records, no benchmark data); ridge strength picked by
            held-out reconstruction error on data/synthetic/val.jsonl records
Evaluated on bench_quant's typed decisions (100 cases), Banking77 8-way and prompt injections, against the exact path;
also agreement of the top answer with the exact path.

    python scripts/skipkv_probe.py --ckpt runs/x9b-clean3/model.pt --cuts 24 20 16 --out runs/skipkv/x9b.json
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from bench_quant import logits  # noqa: E402
from eval.typed_decisions import load_cases, questions_of  # noqa: E402
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402
from opendecider.formatting import record_state_text  # noqa: E402

LAMS = [1e-4, 1e-3, 1e-2, 1e-1]


def _hidden_arg(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _set_hidden(args, kwargs, h):
    if args:
        return (h,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["hidden_states"] = h
    return args, kwargs


@contextlib.contextmanager
def capture_inputs(bb):
    """Record every decoder layer's input hidden states (layer j's input = block j's output, 0-based j) and the
    final layer's output, for state passes."""
    store = {}
    hs = []
    n = len(bb.lm.layers)
    for j, layer in enumerate(bb.lm.layers):
        def pre(mod, args, kwargs, j=j):
            store[j] = _hidden_arg(args, kwargs).detach()
        hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    def post(mod, args, out):
        store[n] = (out[0] if isinstance(out, tuple) else out).detach()
    hs.append(bb.lm.layers[-1].register_forward_hook(post))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@contextlib.contextmanager
def single_input_state(bb, cut, maps=None):
    """During state passes only (Backbone.prefix_cache_batch): layers j >= cut (0-based) read a predicted input
    from h_cut = input of layer `cut` (= output of block `cut`, 1-based), identity or maps[j] = (W, b)."""
    orig = bb.prefix_cache_batch
    cur = {}

    def pre_cut(mod, args, kwargs):
        cur["h"] = _hidden_arg(args, kwargs)

    def pre_up(mod, args, kwargs, j):
        h = cur["h"]
        if maps is not None:
            W, b = maps[j]
            h = (h.float() @ W + b).to(h.dtype)
        return _set_hidden(args, kwargs, h)

    def patched(*a, **k):
        hs = [bb.lm.layers[cut].register_forward_pre_hook(pre_cut, with_kwargs=True)]
        for j in range(cut, len(bb.lm.layers)):
            hs.append(bb.lm.layers[j].register_forward_pre_hook(lambda m, a_, k_, j=j: pre_up(m, a_, k_, j),
                                                                 with_kwargs=True))
        try:
            return orig(*a, **k)
        finally:
            for h in hs:
                h.remove()
    bb.prefix_cache_batch = patched
    try:
        yield
    finally:
        bb.prefix_cache_batch = orig


def state_texts(model, path, n, seed):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    return [record_state_text({"state": rows[i]["state"], "questions": []}, fmt, cap) for i in idx]


@torch.no_grad()
def collect(model, texts, cuts, fit=True, acc=None):
    """Normal-equation sums per cut (X = [h_cut, 1]) and per upper layer target; or (fit=False) held-out pairs."""
    bb = model.backbone
    L = len(bb.lm.layers)
    D = bb.hidden_size
    if acc is None:
        acc = {c: {"XtX": torch.zeros(D + 1, D + 1, device=bb.device, dtype=torch.float64),
                   "XtY": {j: torch.zeros(D + 1, D, device=bb.device, dtype=torch.float64) for j in range(c, L)},
                   "n": 0} for c in cuts}
    held = {c: [] for c in cuts}
    for t in texts:
        ids = bb.tokenize([t])
        with capture_inputs(bb) as st:
            bb.prefix_cache_batch(ids)
        for c in cuts:
            X = st[c][0].float()
            X1 = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
            if fit:
                acc[c]["XtX"] += (X1.T @ X1).double()
                for j in range(c, L):
                    acc[c]["XtY"][j] += (X1.T @ st[j][0].float()).double()
                acc[c]["n"] += len(X)
            else:
                held[c].append({j: st[j][0].half().cpu() for j in range(c, L)} | {"X": X.half().cpu()})
    return acc if fit else held


def solve(acc, held, cut, L):
    """Ridge per upper layer on [h_cut, 1] (the bias column is shrunk too, negligibly), from one eigendecomposition of
    X'X; strength (relative to the mean diagonal) by held-out squared error summed over layers and records."""
    XtX = acc["XtX"]
    scale = float(XtX.diagonal()[:-1].mean())
    ev, Q = torch.linalg.eigh(XtX)
    ev, Q = ev.float(), Q.float()
    P = {j: Q.T @ acc["XtY"][j].float() for j in range(cut, L)}
    dev = Q.device
    errs = {}
    for lam in LAMS:
        e = 0.0
        for j in range(cut, L):
            W = Q @ (P[j] / (ev + lam * scale)[:, None])
            for h in held:
                X1 = torch.cat([h["X"].float().to(dev), torch.ones(len(h["X"]), 1, device=dev)], 1)
                e += float((X1 @ W - h[j].float().to(dev)).norm() ** 2)
        errs[lam] = e
    lam = min(errs, key=errs.get)
    maps, rel = {}, {}
    for j in range(cut, L):
        W = Q @ (P[j] / (ev + lam * scale)[:, None])
        maps[j] = (W[:-1].contiguous(), W[-1].contiguous())
        e_r = e_i = den = 0.0
        for h in held:
            X = h["X"].float().to(dev); Y = h[j].float().to(dev)
            e_r += float((X @ maps[j][0] + maps[j][1] - Y).norm() ** 2); e_i += float((X - Y).norm() ** 2)
            den += float(Y.norm() ** 2)
        rel[j] = {"ridge": (e_r / den) ** 0.5, "identity": (e_i / den) ** 0.5}
    return maps, lam, rel


def evaluate(model, cases, b77, inj):
    """Per-question logits on the three sets (bench_quant's paths)."""
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    out = {"typed": [], "b77": [], "inj": []}
    for r in cases:
        items = questions_of(r)
        qs = [Question(instr, texts, "choice", lab) for _, instr, _, texts, _, lab in items]
        lg = logits(model, record_state_text({"state": r["state"], "questions": []}, fmt, cap), qs)
        for (_, _, names, _, gp, _), l in zip(items, lg):
            out["typed"].append((np.array(l[: len(names)]), gp))
    for key, rows in (("b77", b77), ("inj", inj)):
        for r in rows:
            q = r["questions"][0]
            l = logits(model, record_state_text({"state": r["state"], "questions": []}, fmt, cap),
                       [Question(q["prompt"], q["options"], "choice")])[0]
            out[key].append((np.array(l[: len(q["options"])]), q["label"]))
    return out


def score(ev, ref=None):
    res = {}
    for k, v in ev.items():
        if k == "typed":
            acc = np.mean([lg.argmax() == gp.argmax() for lg, gp in v])
        else:
            acc = np.mean([lg.argmax() == y for lg, y in v])
        res[k] = float(acc)
        if ref is not None:
            res[k + "_agree"] = float(np.mean([a[0].argmax() == b[0].argmax() for a, b in zip(v, ref[k])]))
            sm = lambda z: np.exp(z - np.logaddexp.reduce(z))
            res[k + "_tv"] = float(np.mean([0.5 * np.abs(sm(a[0]) - sm(b[0])).sum() for a, b in zip(v, ref[k])]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cuts", nargs="+", type=int, default=[24, 20, 16])
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-held", type=int, default=10)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--n-public", type=int, default=None, help="first N items of each public set (smoke runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
    model, _ = load_model(a.ckpt)
    model.eval()
    bb = model.backbone
    L = len(bb.lm.layers)
    cases = load_cases("test", a.cases)
    b77 = [json.loads(l) for l in open("data/public/banking77_8way.jsonl")][: a.n_public]
    inj = [json.loads(l) for l in open("data/public/prompt_injections.jsonl")][: a.n_public]
    res = {"ckpt": a.ckpt, "n_calib": a.n_calib, "n_held": a.n_held, "layers": L}
    t = time.time()
    ref = evaluate(model, cases, b77, inj)
    res["exact"] = score(ref) | {"seconds": time.time() - t}
    print("exact", res["exact"], flush=True)
    t = time.time()
    acc = collect(model, state_texts(model, "data/synthetic/calib.jsonl", a.n_calib, 0), a.cuts)
    held = collect(model, state_texts(model, "data/synthetic/val.jsonl", a.n_held, 1), a.cuts, fit=False)
    res["fit_seconds"] = time.time() - t
    res["calib_tokens"] = acc[a.cuts[0]]["n"]
    for c in a.cuts:
        maps, lam, rel = solve(acc[c], held[c], c, L)
        res[f"cut{c}"] = {"lambda": lam, "held_rel_err": {str(j): v for j, v in rel.items()}}
        for kind, m in (("identity", None), ("ridge", maps)):
            t = time.time()
            with single_input_state(bb, c, m):
                ev = evaluate(model, cases, b77, inj)
            res[f"cut{c}"][kind] = score(ev, ref) | {"seconds": time.time() - t}
            print(f"cut {c} {kind}", res[f"cut{c}"][kind], flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
        del maps
        acc.pop(c); torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
