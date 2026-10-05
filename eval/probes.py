"""Behavioural probe battery (docs/SPEC.md section 7.4).

Every probe takes a `decide_fn(state, question_dict) -> probs` where question_dict is
{"type", "prompt", "options"} and probs is either a list aligned with `options` or a
{option: p} dict (the public API shape). A dict cannot tell exact duplicates apart; the
duplicate probe reports that as `resolvable: false` rather than guessing. So the same battery
runs on our models (`model_decide_fn`), on baselines, and later on a Jev API client.

Probe signatures (not accuracy) are what we compare with public observations O3, O6, O7.
Results are "consistent / inconsistent with public observations"; they say nothing about how
Jev is built.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Callable, Sequence

import numpy as np

from .metrics import bootstrap_mean_ci

DecideFn = Callable[[object, dict], object]

DEFAULT_DUMMY = "a purple giraffe playing the tuba"
JITTER_TOL = 1e-4
LATENCY_K = (2, 8, 32, 128, 255)
LATENCY_Q = (1, 4, 16, 64)


# ---------------------------------------------------------------------------- helpers
def _as_list(res, options: Sequence[str]) -> list[float]:
    if isinstance(res, dict):
        if "probs" in res and isinstance(res["probs"], dict):
            res = res["probs"]
        return [float(res[o]) for o in options]
    return [float(x) for x in res]


def _tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def _q(item: dict, options: Sequence[str]) -> dict:
    return {"type": item.get("type", "choice"), "prompt": item["prompt"], "options": list(options)}


def _call(decide_fn: DecideFn, item: dict, options: Sequence[str]) -> list[float]:
    return _as_list(decide_fn(item["state"], _q(item, options)), options)


# ---------------------------------------------------------------------------- probes
def permutation_probe(decide_fn: DecideFn, item: dict, n_perm: int = 20, seed: int = 0) -> dict:
    """TV distance between the identity-order distribution and each shuffled order (un-permuted
    back to the original option indices). Also mean probability by *position*, a position-bias
    signature."""
    opts = list(item["options"])
    K = len(opts)
    rng = random.Random(seed)
    base = np.asarray(_call(decide_fn, item, opts))
    tvs, pos_mass = [], np.zeros(K)
    pos_mass += base
    for _ in range(n_perm):
        perm = list(range(K))
        rng.shuffle(perm)
        p = np.asarray(_call(decide_fn, item, [opts[i] for i in perm]))
        pos_mass += p
        back = np.empty(K)
        back[perm] = p
        tvs.append(_tv(back, base))
    return {"probe": "permutation", "K": K, "n_perm": n_perm, "mean_tv": float(np.mean(tvs)),
            "max_tv": float(np.max(tvs)), "position_mass": (pos_mass / (n_perm + 1)).tolist(),
            "base_probs": base.tolist()}


def dummy_option_probe(decide_fn: DecideFn, item: dict, dummy: str | None = None, position: str = "end") -> dict:
    """Add an irrelevant option. If options rescale proportionally (IIA-like), the renormalised
    original options match the base distribution (TV ~ 0)."""
    opts = list(item["options"])
    dummy = dummy or item.get("dummy") or DEFAULT_DUMMY
    base = np.asarray(_call(decide_fn, item, opts))
    at = len(opts) if position == "end" else 0
    new = opts[:at] + [dummy] + opts[at:]
    p = np.asarray(_call(decide_fn, item, new))
    p_dummy = float(p[at])
    rest = np.delete(p, at)
    renorm = rest / max(rest.sum(), 1e-12)
    return {"probe": "dummy_option", "K": len(opts), "position": position, "dummy": dummy,
            "dummy_mass": p_dummy, "tv_renormalised": _tv(renorm, base), "tv_raw": _tv(rest, base),
            "argmax_changed": bool(int(np.argmax(renorm)) != int(np.argmax(base)))}


def duplicate_option_probe(decide_fn: DecideFn, item: dict, index: int = 0) -> dict:
    """Append an exact duplicate of option `index`. Reports the split between the two copies and
    the change to every other option."""
    opts = list(item["options"])
    base = np.asarray(_call(decide_fn, item, opts))
    new = opts + [opts[index]]
    res = decide_fn(item["state"], _q(item, new))
    if isinstance(res, dict):
        # dict-shaped API: identical names collide, the split is not observable
        pr = res.get("probs", res)
        return {"probe": "duplicate_option", "K": len(opts), "resolvable": False,
                "p_name_before": float(base[index]), "p_name_after": float(pr.get(opts[index], float("nan")))}
    p = np.asarray([float(x) for x in res])
    a, b = float(p[index]), float(p[-1])
    others = [k for k in range(len(opts)) if k != index]
    tot = a + b
    return {"probe": "duplicate_option", "K": len(opts), "resolvable": True,
            "p_original_before": float(base[index]), "p_copy_first": a, "p_copy_second": b,
            "p_pair_total": tot, "split_ratio_first": a / tot if tot > 0 else float("nan"),
            "pair_gain": tot - float(base[index]),
            "others_tv": _tv(p[others], base[others]),
            "others_renorm_tv": _tv(p[others] / max(p[others].sum(), 1e-12),
                                    base[others] / max(base[others].sum(), 1e-12)) if others else 0.0}


def label_length_probe(decide_fn: DecideFn, item: dict) -> dict:
    """Needs `item["long_options"]` aligned with `item["options"]` (same meaning, longer wording).
    Reports TV between all-short and all-long, plus the one-at-a-time lengthening shift
    delta_k = p_k(option k long, rest short) - p_k(all short); positive mean -> favours longer labels."""
    short, long_ = list(item["options"]), list(item["long_options"])
    if len(short) != len(long_):
        raise ValueError("long_options must align with options")
    base = np.asarray(_call(decide_fn, item, short))
    all_long = np.asarray(_call(decide_fn, item, long_))
    deltas = []
    for k in range(len(short)):
        opts = short[:k] + [long_[k]] + short[k + 1:]
        deltas.append(float(_call(decide_fn, item, opts)[k] - base[k]))
    return {"probe": "label_length", "K": len(short), "tv_all_long": _tv(all_long, base),
            "mean_one_at_a_time_shift": float(np.mean(deltas)),
            "mean_abs_one_at_a_time_shift": float(np.mean(np.abs(deltas))), "shifts": deltas}


def repeat_probe(decide_fn: DecideFn, item: dict, n: int = 50, tol: float = JITTER_TOL) -> dict:
    """n identical calls. max_abs_diff = max over options of (max - min) across calls.
    Class: deterministic (0), float_jitter (< tol), sampling_variance (>= tol)."""
    opts = list(item["options"])
    P = np.asarray([_call(decide_fn, item, opts) for _ in range(n)])
    d = float((P.max(0) - P.min(0)).max())
    cls = "deterministic" if d == 0.0 else ("float_jitter" if d < tol else "sampling_variance")
    return {"probe": "repeat", "K": len(opts), "n": n, "max_abs_diff": d,
            "mean_std": float(P.std(0).mean()), "class": cls, "tol": tol}


def latency_probe(decide_request_fn: Callable[[dict], dict], state, k_grid: Sequence[int] = LATENCY_K,
                  q_grid: Sequence[int] = LATENCY_Q, reps: int = 5, warmup: int = 1) -> dict:
    """Latency over a (K options x Q questions) grid. Records the server-reported
    timing_ms.total when present and client wall-clock (which, for a remote API, includes network)."""
    cells = []
    for K in k_grid:
        opts = [f"option {i}" for i in range(K)]
        for Q in q_grid:
            req = {"state": state, "questions": {f"q{j}": {"type": "choice", "prompt": f"Which option fits best? ({j})",
                                                          "options": opts} for j in range(Q)}}
            for _ in range(warmup):
                decide_request_fn(req)
            wall, server, toks = [], [], None
            for _ in range(reps):
                t0 = time.perf_counter()
                r = decide_request_fn(req)
                wall.append(1e3 * (time.perf_counter() - t0))
                tm = r.get("timing_ms") or {}
                if "total" in tm:
                    server.append(float(tm["total"]))
                toks = r.get("input_tokens", toks)
            cells.append({"K": K, "Q": Q, "wall_ms_p50": float(np.percentile(wall, 50)),
                          "wall_ms_p95": float(np.percentile(wall, 95)),
                          "server_ms_p50": float(np.percentile(server, 50)) if server else None,
                          "server_ms_p95": float(np.percentile(server, 95)) if server else None,
                          "input_tokens": toks})
    ref = min(cells, key=lambda c: (c["K"], c["Q"]))
    top = max(cells, key=lambda c: (c["K"], c["Q"]))
    return {"probe": "latency", "reps": reps, "cells": cells,
            "ratio_largest_vs_smallest_wall_p50": top["wall_ms_p50"] / max(ref["wall_ms_p50"], 1e-9)}


# ---------------------------------------------------------------------------- battery
def _agg(vals: Sequence[float], n_boot: int, seed: int) -> dict:
    m, lo, hi = bootstrap_mean_ci(vals, n=n_boot, seed=seed)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}


def run_battery(decide_fn: DecideFn, items: Sequence[dict], decide_request_fn: Callable | None = None,
                probes: Sequence[str] = ("permutation", "dummy", "duplicate", "label_length", "repeat", "latency"),
                n_perm: int = 20, n_repeat: int = 50, repeat_items: int = 1,
                latency_k: Sequence[int] = LATENCY_K, latency_q: Sequence[int] = LATENCY_Q,
                latency_reps: int = 5, n_boot: int = 1000, seed: int = 0) -> dict:
    """Run the battery on `items` ({state, prompt, options, [type], [long_options], [dummy]}).
    Returns per-item results and bootstrap-CI aggregates (JSON-serialisable)."""
    per: dict[str, list] = {p: [] for p in probes}
    for i, it in enumerate(items):
        if "permutation" in probes:
            per["permutation"].append(permutation_probe(decide_fn, it, n_perm, seed + i))
        if "dummy" in probes:
            per["dummy"].append(dummy_option_probe(decide_fn, it))
        if "duplicate" in probes:
            per["duplicate"].append(duplicate_option_probe(decide_fn, it))
        if "label_length" in probes and it.get("long_options"):
            per["label_length"].append(label_length_probe(decide_fn, it))
        if "repeat" in probes and i < repeat_items:
            per["repeat"].append(repeat_probe(decide_fn, it, n_repeat))
    agg: dict = {}
    if per.get("permutation"):
        agg["permutation_mean_tv"] = _agg([r["mean_tv"] for r in per["permutation"]], n_boot, seed)
        agg["permutation_max_tv"] = _agg([r["max_tv"] for r in per["permutation"]], n_boot, seed)
        first = [r["position_mass"][0] - 1.0 / r["K"] for r in per["permutation"]]
        agg["first_position_excess_mass"] = _agg(first, n_boot, seed)
    if per.get("dummy"):
        agg["dummy_mass"] = _agg([r["dummy_mass"] for r in per["dummy"]], n_boot, seed)
        agg["dummy_tv_renormalised"] = _agg([r["tv_renormalised"] for r in per["dummy"]], n_boot, seed)
    dup = [r for r in per.get("duplicate", []) if r["resolvable"]]
    if dup:
        agg["duplicate_pair_gain"] = _agg([r["pair_gain"] for r in dup], n_boot, seed)
        agg["duplicate_split_first"] = _agg([r["split_ratio_first"] for r in dup], n_boot, seed)
        agg["duplicate_others_renorm_tv"] = _agg([r["others_renorm_tv"] for r in dup], n_boot, seed)
    if per.get("label_length"):
        agg["label_length_tv_all_long"] = _agg([r["tv_all_long"] for r in per["label_length"]], n_boot, seed)
        agg["label_length_mean_shift"] = _agg([r["mean_one_at_a_time_shift"] for r in per["label_length"]], n_boot, seed)
    if per.get("repeat"):
        agg["repeat_max_abs_diff"] = max(r["max_abs_diff"] for r in per["repeat"])
        classes = [r["class"] for r in per["repeat"]]
        agg["repeat_class"] = ("sampling_variance" if "sampling_variance" in classes else
                               "float_jitter" if "float_jitter" in classes else "deterministic")
    out = {"n_items": len(items), "aggregate": agg, "items": per}
    if "latency" in probes and decide_request_fn is not None and items:
        out["latency"] = latency_probe(decide_request_fn, items[0]["state"], latency_k, latency_q, latency_reps)
    return out


# ---------------------------------------------------------------------------- adapters
def model_decide_fn(model, temperature: float = 1.0, sampler_k: int = 1, sampler_mode: str = "mc_dropout") -> DecideFn:
    """decide_fn over our DecisionModel, a Decider, or a baseline scorer (anything with
    encode_states / run). Returns a list aligned with the options (so duplicates are resolvable).
    Sampler (O7 hypothesis b) runs through Decider when sampler_k > 1."""
    import torch
    from opendecider.batching import Question, state_texts
    from opendecider.decider import Decider
    from opendecider.formatting import YES_NO
    from opendecider.sampler import SamplerConfig

    if isinstance(model, Decider):
        temperature = model.temperature if temperature == 1.0 else temperature
        model = model.model
    decider = Decider(model, temperature) if hasattr(model, "backbone") and hasattr(model, "slots") else None

    def fn(state, qd: dict) -> list[float]:
        opts = list(qd.get("options") or qd.get("levels") or YES_NO)
        q = Question(qd["prompt"], opts, qd.get("type", "choice"))
        with torch.no_grad():
            mem = model.encode_states(state_texts([state]))
            if decider is not None:
                mode = sampler_mode if sampler_k > 1 else "off"
                mean, _ = decider._probs([q], mem, SamplerConfig(mode=mode, k=sampler_k))
                p = mean[0, : len(opts)].double()
            else:
                _, out = model.run([q], mem)
                p = torch.softmax(out.logits[0, : len(opts)].double() / temperature, -1)
        p = p.clamp_min(0)
        return (p / p.sum()).tolist()

    return fn


def model_request_fn(model) -> Callable[[dict], dict]:
    """decide_request_fn for the latency probe: our full Decider path (timing_ms included)."""
    from opendecider.decider import Decider
    d = model if isinstance(model, Decider) else Decider(model)
    return d.decide


# ---------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the section 7.4 probe battery.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--baseline", choices=["label_first", "label_seq"])
    ap.add_argument("--items", required=True, help="JSONL of {state, prompt, options, [long_options], [dummy]}")
    ap.add_argument("--backbone-config", default="configs/backbone.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-repeat", type=int, default=50)
    ap.add_argument("--sampler-k", type=int, default=1)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--probes", nargs="+", help="subset of: permutation dummy duplicate label_length repeat latency")
    ap.add_argument("--serving", action="store_true", help="the serving path (decider.prepare_inference_)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from .evaluate import load_scorer, gpu_context
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    name = a.name or (os.path.splitext(os.path.basename(a.ckpt))[0] if a.ckpt else a.baseline)
    with gpu_context(a.device, "probes"):
        model, temp = load_scorer(a.ckpt, a.baseline, a.backbone_config, a.device)
        if a.serving and a.ckpt:
            from opendecider.decider import prepare_inference_
            prepare_inference_(model)
        fn = model_decide_fn(model, temp, a.sampler_k)
        req_fn = None if (a.no_latency or a.baseline) else model_request_fn(model)
        kw = {"probes": tuple(a.probes)} if a.probes else {}
        res = run_battery(fn, items, req_fn, n_perm=a.n_perm, n_repeat=a.n_repeat, seed=a.seed, **kw)
    res["meta"] = {"name": name, "ckpt": a.ckpt, "baseline": a.baseline, "items": a.items, "seed": a.seed,
                   "sampler_k": a.sampler_k, "device": a.device, "serving": a.serving, "probes": a.probes,
                   "slot_emb": getattr(getattr(model, "cfg", None), "slot_emb", None),
                   "variant": getattr(getattr(model, "cfg", None), "variant", None)}
    from opendecider.guards import require_free_gb
    require_free_gb(0.01)
    os.makedirs(f"runs/{name}", exist_ok=True)
    path = f"runs/{name}/probes.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["aggregate"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
