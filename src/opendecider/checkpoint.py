"""Save/load only the trainable decision module (the frozen backbone is referenced by pinned revision)."""
from __future__ import annotations

import dataclasses
import json
import os

import torch

from .backbone import Backbone, BackboneConfig
from .guards import require_free_gb
from .models import ModelConfig, build_model


def save(model, path: str, extra: dict | None = None) -> None:
    sd = {k: v.detach().cpu() for k, v in model.trainable_state_dict().items()}
    size_gb = sum(v.numel() * v.element_size() for v in sd.values()) / 2**30
    require_free_gb(size_gb * 1.1 + 0.05)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"state_dict": sd, "model_cfg": dataclasses.asdict(model.cfg),
                "backbone_cfg": dataclasses.asdict(model.backbone.cfg), "extra": extra or {}}, tmp)
    os.replace(tmp, path)


def load_model(path: str, backbone: Backbone | None = None, device: str | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bcfg = BackboneConfig(**ck["backbone_cfg"])
    if device:
        bcfg.device = device
    bb = backbone or Backbone.load(bcfg)
    model = build_model(bb, ModelConfig(**ck["model_cfg"]))
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("backbone.")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
    return model.to(bb.device).eval(), ck.get("extra", {})


def load_decider(path: str):
    from .decider import Decider
    model, extra = load_model(path)
    return Decider(model, temperature=extra.get("temperature", 1.0),
                   temperature_by_type=extra.get("temperature_by_type"))
