"""Fit composite-question weights (caller-side decomposition), e.g. the phishing recipe: several narrow
sub-questions answered by the model, combined by a logistic regression fitted on labelled examples."""
from __future__ import annotations

from typing import Sequence

import torch

from .decider import composite_feature


def fit_logistic(part_answers: Sequence[dict], labels: Sequence[int], features: Sequence[str],
                 l2: float = 1e-3, steps: int = 500) -> dict:
    """part_answers[i] = {part_name: answer dict from /v1/decide}; labels[i] in {0,1} (1 = yes).
    Returns a CombineSpec-compatible dict {"kind","weights","bias"}."""
    X = torch.tensor([[composite_feature(f, pa) for f in features] for pa in part_answers], dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    w = torch.zeros(len(features), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"kind": "logistic", "weights": dict(zip(features, w.detach().tolist())), "bias": float(b)}
