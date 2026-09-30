"""Few-label calibrators for one question's per-option probabilities, and the CV selection used by /v1/calibrate.

Measured in eval/calib_curve.py (docs/RESULTS.md §6-7); policy in docs/roadmap_designs.md §2. Every probability-level
calibrator is class-agnostic: one map p -> q applied to each option of a question, then renormalised over its options.
Arrays: L (N,Km) log-probs or logits (padding NEG), T (N,Km) targets (one-hot or soft), M (N,Km) bool mask.

  identity      no change                                                   (logit level)
  temperature   softmax(z / T), L2 penalty PRIOR/n on log T (shrinks to T=1)  (logit level)
  vector        softmax(z / T + b), one bias per option position (fixed option names only)  (logit level)
  beta          beta calibration on p_T, penalty PRIOR/n toward a=b=1, c=0; a, b >= 0 (monotone)
  ets           simplex mix of p_T, p and uniform (Mix-n-Match)
  histogram     10 equal-mass bins over p_T, Laplace-smoothed (binary schemas only)
  iso_shrunk    isotonic on p_T blended with p_T by n/(n+N0)
Selection: K-fold CV log loss (leave-one-out when n < 25); the simplest candidate within one SE of the best wins.
"""
from __future__ import annotations

import numpy as np
import torch

NEG = -1e4
PRIOR = 2.0            # shrinkage strength: penalty PRIOR/n (n = labelled questions); a N(0, 0.5^2) prior on log T
N0 = 100.0             # iso_shrunk trusts isotonic with weight n/(n+N0)
SIMPLICITY = ("identity", "temperature", "beta", "ets", "vector", "histogram", "iso_shrunk")
LOGIT_LEVEL = ("identity", "temperature", "vector")


def softmax_masked(Z, M):
    Z = np.where(M, Z, -np.inf)
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def scatter(v, M):
    out = np.zeros(M.shape); out[M] = v
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def per_option(fn, P2, M2):
    """Apply a per-option map p -> q (class-agnostic) and renormalise over each question's options."""
    q = scatter(np.clip(fn(P2[M2]), 1e-6, None), M2)
    return q / q.sum(1, keepdims=True)


def fit_temp(Z, T, M, prior=0.0):
    """One temperature over masked logits Z (N,Km) against soft targets T (N,Km); optional penalty prior/N * log^2 T."""
    z = torch.tensor(np.where(M, Z, NEG)); t = torch.tensor(T)
    logT = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(t * torch.log_softmax(z / torch.exp(logT), 1)).sum(1).mean()
        if prior:
            loss = loss + prior / len(Z) * logT ** 2
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logT).detach())


def fit_logit_profile(Z: torch.Tensor, y: torch.Tensor, method: str = "vector", l2: float = 1e-2,
                      mask: torch.Tensor | None = None) -> tuple[float, list[float]]:
    """NLL-optimal (T, per-option bias) on logits Z (n, K) with labels y (indices, or soft targets). method:
    temperature | vector. mask (n, K) marks real options when rows are padded (padded logits are ignored)."""
    Z = Z.float()
    if mask is not None:
        Z = Z.masked_fill(~mask, -1e9)
    logT = torch.zeros((), requires_grad=True)
    b = torch.zeros(Z.shape[1], requires_grad=method == "vector")
    params = [logT] + ([b] if method == "vector" else [])
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z / logT.exp() + b, y) + l2 * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp()), b.detach().tolist()


def iso_knots(x, y):
    """Monotone (non-decreasing) PAV fit on per-option rows -> knots (x, value); applied by linear interpolation."""
    from scipy.optimize import isotonic_regression
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], isotonic_regression(y[o]).x
    ux, first = np.unique(xs, return_index=True)       # tied x -> one knot (the pooled value)
    return ux, ys[first]


def fit_isotonic(x, y):
    kx, ky = iso_knots(x, y)
    return lambda z: np.interp(z, kx, ky)


def hist_bins(x, y, bins=10):
    """Histogram binning: equal-mass bins over p -> (inner edges, Laplace-smoothed mean outcome per bin)."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(edges, x)
    cnt = np.bincount(b, minlength=bins); tot = np.bincount(b, weights=y, minlength=bins)
    return edges, (tot + y.mean()) / (cnt + 1)


def fit_hist(x, y, bins=10):
    edges, val = hist_bins(x, y, bins)
    return lambda z: val[np.searchsorted(edges, z)]


def beta_params(x, y, l2=1e-3, prior=0.0, monotone=False):
    """Beta calibration (Kull et al. 2017): logistic regression on [log p, -log(1-p)] + bias, per option row -> (a, b, c).
    prior: extra L2 toward the identity (a=b=1, c=0). monotone: a, b = exp(u) >= 0 so the map is non-decreasing."""
    xc = np.clip(x, 1e-6, 1 - 1e-6)
    F = torch.tensor(np.stack([np.log(xc), -np.log1p(-xc)], 1)); t = torch.tensor(y)
    w = torch.tensor([0.0, 0.0, 0.0] if monotone else [1.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    ab = (lambda: w[:2].exp()) if monotone else (lambda: w[:2])

    def closure():
        opt.zero_grad()
        a = ab()
        z = F @ a + w[2]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t) + l2 * (a - 1).pow(2).sum()
        if prior:
            loss = loss + prior * ((a - 1).pow(2).sum() + w[2] ** 2)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return np.concatenate([ab().numpy(), w[2:].numpy()])


def beta_map(wd, z):
    zc = np.clip(z, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(wd[0] * np.log(zc) - wd[1] * np.log1p(-zc) + wd[2])))


def fit_beta(x, y, l2=1e-3):
    wd = beta_params(x, y, l2)
    return lambda z: beta_map(wd, z)


def ets_weights(P1, P0, T, M):
    """Ensemble temperature scaling (Mix-n-Match, Zhang et al. 2020): simplex weights of (p_T, p, uniform)."""
    P1, P0 = torch.tensor(P1), torch.tensor(P0)
    U = torch.tensor(M / M.sum(1, keepdims=True)); t = torch.tensor(T)
    a = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = torch.softmax(a, 0)
        q = w[0] * P1 + w[1] * P0 + w[2] * U
        loss = -(t * torch.log(q.clamp_min(1e-12))).sum(1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return torch.softmax(a, 0).detach().numpy()


def ets_mix(w, P1, P0, M):
    return w[0] * P1 + w[1] * P0 + w[2] * M / M.sum(1, keepdims=True)


def fit_ets(L, T, M, Tt):
    w = ets_weights(softmax_masked(L / Tt, M), softmax_masked(L, M), T, M)
    return lambda L2, M2: ets_mix(w, softmax_masked(L2 / Tt, M2), softmax_masked(L2, M2), M2)


def top_label(fn_conf, L2, M2, Tt):
    """Top-label calibration (Gupta & Ramdas 2022): recalibrate only the argmax's probability, rescale the others
    proportionally. The argmax never changes, so accuracy is preserved by construction."""
    P = softmax_masked(L2 / Tt, M2)
    top = P.argmax(1); pm = P.max(1)
    c = np.clip(fn_conf(pm), 1e-4, 1 - 1e-4)
    p2 = np.sort(np.where(M2, P, -1.0), 1)[:, -2]       # runner-up probability
    c = np.maximum(c, p2 / (1 - pm + p2) + 1e-6)        # just above the c at which the runner-up would tie
    rest = (1 - c) / np.clip(1 - pm, 1e-9, None)
    Q = P * rest[:, None]
    Q[np.arange(len(P)), top] = c
    return Q


# ---- serialisable calibrators: {"kind", "T", "bias" (list | None), + kind-specific params (lists / floats)} ----

def fit_calibrator(kind: str, L, T, M) -> dict:
    """Fit one candidate on n questions. Probability-level kinds act on p_T with a shrunk temperature."""
    n = len(L)
    if kind == "identity":
        return {"kind": kind, "T": 1.0, "bias": None}
    if kind == "vector":
        Tv, b = fit_logit_profile(torch.tensor(L), torch.tensor(T, dtype=torch.float32), "vector", mask=torch.tensor(M))
        return {"kind": kind, "T": Tv, "bias": b}
    Tt = fit_temp(L, T, M, prior=PRIOR)
    cal = {"kind": kind, "T": Tt, "bias": None}
    if kind == "temperature":
        return cal
    PT = softmax_masked(L / Tt, M)
    x, y = PT[M], T[M]
    if kind == "beta":
        cal["w"] = beta_params(x, y, prior=PRIOR / n, monotone=True).tolist()
    elif kind == "iso_shrunk":
        kx, ky = iso_knots(x, y)
        cal.update(knots_x=kx.tolist(), knots_y=ky.tolist(), lam=n / (n + N0))
    elif kind == "histogram":
        edges, val = hist_bins(x, y)
        cal.update(edges=edges.tolist(), values=val.tolist())
    elif kind == "ets":
        cal["w"] = ets_weights(PT, softmax_masked(L, M), T, M).tolist()
    else:
        raise ValueError(f"unknown calibrator {kind!r}")
    return cal


def option_map(cal: dict, x):
    """The per-option map p_T -> q (before renormalisation) of a probability-level calibrator."""
    k = cal["kind"]
    if k == "beta":
        return beta_map(np.asarray(cal["w"]), x)
    if k == "iso_shrunk":
        return np.interp(x, cal["knots_x"], cal["knots_y"])
    if k == "histogram":
        return np.asarray(cal["values"])[np.searchsorted(np.asarray(cal["edges"]), x)]
    raise ValueError(f"{k!r} has no per-option map")


def calibrate_probs(cal: dict, PT, P0, M):
    """Probability stage: PT = softmax(z/T + b) (N,Km), P0 = softmax(z) (only read by ets), M mask -> (N,Km)."""
    k = cal["kind"]
    if k in LOGIT_LEVEL:
        return PT
    if k == "ets":
        return ets_mix(np.asarray(cal["w"]), PT, P0, M)
    Q = per_option(lambda z: option_map(cal, z), PT, M)
    if k == "iso_shrunk":
        return cal["lam"] * Q + (1 - cal["lam"]) * PT
    return Q


def apply_calibrator(cal: dict, L, M):
    """Full map on logits / log-probs L (N,Km) -> calibrated probabilities (N,Km), zero on padding."""
    Z = L / cal["T"] + (np.asarray(cal["bias"])[None, : L.shape[1]] if cal.get("bias") is not None else 0.0)
    return calibrate_probs(cal, softmax_masked(Z, M), softmax_masked(L, M), M)


def select_calibrator(L, T, M, candidates=SIMPLICITY, folds=5, seed=0) -> tuple[dict, dict]:
    """CV log loss per candidate (K folds; leave-one-out when n < 25), then the simplest within one SE of the best.
    Returns (calibrator refitted on all n, {kind: {"log_loss", "se"}})."""
    n = len(L)
    k = n if n < 25 else folds
    fold = np.random.default_rng(seed).permutation(n) % k
    cands = [c for c in SIMPLICITY if c in candidates]
    loss = np.zeros((len(cands), n))
    for f in range(k):                                     # offline fitting: loops over folds and candidates
        tr, te = fold != f, fold == f
        for ci, c in enumerate(cands):
            P = apply_calibrator(fit_calibrator(c, L[tr], T[tr], M[tr]), L[te], M[te])
            loss[ci, te] = -(T[te] * np.log(np.clip(P, 1e-12, 1))).sum(1)
    mean, se = loss.mean(1), loss.std(1, ddof=1) / np.sqrt(n)
    best = int(mean.argmin())
    pick = next(c for ci, c in enumerate(cands) if mean[ci] <= mean[best] + se[best])
    table = {c: {"log_loss": float(mean[ci]), "se": float(se[ci])} for ci, c in enumerate(cands)}
    return fit_calibrator(pick, L, T, M), table


def calibrate_rows(cal: dict, p: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inference: probability stage for a batch of questions sharing one profile. p = softmax(z/T + b) (R,Km) on any
    device, logits the raw z (read by ets only), mask the option mask. Returns calibrated (R,Km) like p."""
    M = mask.bool().cpu().numpy()
    P0 = torch.softmax(logits.double(), -1).cpu().numpy() * M if cal["kind"] == "ets" else None
    return torch.from_numpy(calibrate_probs(cal, p.double().cpu().numpy(), P0, M)).to(p)
