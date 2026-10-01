"""Per-deployment calibration profiles: the small "decision head" fitted on a few labelled examples.

The Jev ecosystem documents this pattern (a caller-fitted logistic head over the model's answers; one write-up reports
ECE 0.151 -> 0.030 from re-weighting with ~140 labels). Here: for ONE question, run the model on labelled states, then fit
    auto (default)     CV-select a calibrator (src/opendecider/calibrators.py): identity, shrunk temperature, beta,
                       ets, vector, histogram (binary), iso_shrunk; simplest within one SE of the best CV log loss
    temperature        p = softmax(z / T)
    vector             p = softmax(z / T + b),  one bias per option name (L2-regularised)
    identity | beta | ets | histogram | iso_shrunk   that calibrator alone (no CV)
Logit-level methods store (temperature, bias); probability-level ones also store `calibrator` (serialisable params,
applied after the softmax). The profile is keyed by option NAME, so it is invariant to the caller's option order.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

from . import calibrators as C
from .batching import state_texts
from .calibrators import fit_logit_profile
from .schema import CalibrateRequest, opt_name


def _label_index(spec, names, label) -> int:
    if spec.type == "noul":
        return 0 if (label is True or str(label).lower() in ("yes", "true", "1")) else 1
    if spec.type == "score" and isinstance(label, int) and not isinstance(label, bool):
        return int(label)
    return names.index(str(label))


def _ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    order = np.argsort(conf)
    e, n = 0.0, len(y)
    for chunk in np.array_split(order, min(bins, n)):
        if len(chunk):
            e += len(chunk) / n * abs((pred[chunk] == y[chunk]).mean() - conf[chunk].mean())
    return float(e)


def fit_profile(decider, req: CalibrateRequest | dict, profiles_dir: str = "runs/profiles", l2: float = 1e-2) -> dict:
    if isinstance(req, dict):
        req = CalibrateRequest.model_validate(req)
    spec = req.question
    if spec.type == "composite":
        raise ValueError("calibrate the atomic parts of a composite, not the composite itself")
    q = decider._atomic("q", spec, {})
    names = q.names
    zs, ys = [], []
    with torch.no_grad():
        for ex in req.examples:                       # one state per example; the state is read once
            mem = decider.model.encode_states(state_texts([ex.state]))
            _, out = decider.model.run([q], mem)
            zs.append(out.logits[0, : len(names)].float().cpu())
            ys.append(_label_index(spec, names, ex.label))
    Z, y = torch.stack(zs), torch.tensor(ys)
    cal, cv = None, None
    if req.method in ("temperature", "vector"):              # plain logit-level fits, as before
        T, bias = fit_logit_profile(Z, y, req.method, l2)
        method = req.method
    else:
        L, Y, M = torch.log_softmax(Z.double(), -1).numpy(), np.eye(len(names))[y.numpy()], np.ones(Z.shape, bool)
        if req.method == "auto":
            cands = [c for c in C.SIMPLICITY if c != "histogram" or len(names) == 2]
            cal, cv = C.select_calibrator(L, Y, M, cands)
        elif req.method == "histogram" and len(names) != 2:
            raise ValueError("histogram calibration needs a binary question (noul or 2 options)")
        else:
            cal = C.fit_calibrator(req.method, L, Y, M)
        method, T, bias = cal["kind"], cal["T"], cal["bias"] or [0.0] * len(names)
        if method in C.LOGIT_LEVEL:
            cal = None
    b = torch.tensor(bias)
    with torch.no_grad():
        p0 = torch.softmax(Z, -1).numpy()
        p1 = torch.softmax(Z / T + b, -1).numpy()
    if cal is not None:
        p1 = C.calibrate_probs(cal, p1.astype(np.float64), p0.astype(np.float64), np.ones(p1.shape, bool))
    yn = y.numpy()
    nll = lambda p: float(-np.log(np.clip(p[np.arange(len(yn)), yn], 1e-12, 1)).mean())
    from .conformal import pack
    prof = {"temperature": T, "bias": dict(zip(names, b.detach().tolist())), "method": method, "requested": req.method,
            "calibrator": cal, "cv": cv,
            # conformal scores on the caller's labels, after this profile's calibration (in-sample for the calibration
            # fit itself: with flexible calibrators and few labels, coverage can come out slightly below 1 - alpha)
            "conformal": pack(p1, yn, [spec.type] * len(yn)),
            "type": spec.type, "names": names, "n": len(yn), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fit_in_sample": {"nll_before": nll(p0), "nll_after": nll(p1), "ece_before": _ece(p0, yn), "ece_after": _ece(p1, yn),
                              "acc_before": float((p0.argmax(1) == yn).mean()), "acc_after": float((p1.argmax(1) == yn).mean())}}
    pid = hashlib.sha1(json.dumps([spec.model_dump(), req.method, prof["created"]], sort_keys=True, default=str).encode()).hexdigest()[:12]
    prof["id"] = pid
    os.makedirs(profiles_dir, exist_ok=True)
    json.dump(prof, open(os.path.join(profiles_dir, f"{pid}.json"), "w"), indent=1)
    decider._profiles[pid] = prof
    return prof
