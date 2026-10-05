"""Shareable OpenDecider checkpoints: the trained head as safetensors, connected to a pinned public backbone.

A checkpoint directory (or Hugging Face repo) holds only what we trained (trunk, heads, adapters, stitch map):
    config.json                     model config, temperatures, backbone variants, provenance
    model-0000k-of-0000n.safetensors  fp16 shards, VeRA vectors fp32 (< 50 MB each: fits a git repo without LFS)
    README.md                       model card
The frozen backbone is downloaded from its own repo at a pinned revision the first time it is needed. Variants
(the same head on differently quantized backbones) are listed in config.json:

    from opendecider.hub import load_decider
    dec = load_decider("checkpoints/opendecider-2b")                  # default variant (int8)
    dec = load_decider("checkpoints/opendecider-2b", backbone="awq")  # 4-bit AWQ backbone
    dec.decide({"state": ..., "questions": {...}})

Backbones are cached under $OPENDECIDER_MODELS (default ./models). Every download checks free disk first.
"""
from __future__ import annotations

import json
import os

import torch

FORMAT = 1
TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "special_tokens_map.json", "generation_config.json"]


# ------------------------------------------------------------------ export
def export(ckpt: str, out_dir: str, backbones: dict, default: str, card: str = "", shard_mb: float = 45,
           info: dict | None = None):
    """runs/<x>/model.pt -> out_dir/{config.json, model-*.safetensors, README.md}."""
    from safetensors.torch import save_file
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # fp16 storage, except the VeRA scaling vectors: they multiply fixed random projections, so fp16 rounding there
    # moves output probabilities by up to 0.017 (fp32 costs < 1 MB)
    sd = {k: v.detach().to(torch.float32 if k.startswith("vera.") else torch.float16).contiguous()
          for k, v in ck["state_dict"].items() if not k.startswith("backbone.")}
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                                   # replace old shards
        if f.endswith(".safetensors"):
            os.remove(os.path.join(out_dir, f))
    limit = int(shard_mb * (1 << 20))
    shards, cur, size = [], {}, 0
    for k in sorted(sd, key=lambda k: -sd[k].numel()):
        n = sd[k].numel() * sd[k].element_size()
        if n > limit and sd[k].dim() > 1:                           # split one big matrix by rows
            rows = max(1, limit // (sd[k][0].numel() * sd[k].element_size()))
            for i, a in enumerate(range(0, sd[k].shape[0], rows)):
                shards.append({f"{k}::part{i}": sd[k][a:a + rows].contiguous()})
            continue
        if size + n > limit and cur:
            shards.append(cur); cur, size = {}, 0
        cur[k] = sd[k]; size += n
    if cur:
        shards.append(cur)
    index = {}
    for i, s in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, os.path.join(out_dir, name), metadata={"format": "pt"})
        index.update({k: name for k in s})
    extra = {k: v for k, v in ck.get("extra", {}).items() if isinstance(v, (int, float, str, dict, list))}
    cfg = {"format": FORMAT, "model_cfg": ck["model_cfg"], "extra": extra, "backbones": backbones,
           "default_backbone": default, "weight_map": index, **(info or {})}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    if card:
        open(os.path.join(out_dir, "README.md"), "w").write(card)
    return cfg


# ------------------------------------------------------------------ load
def _local(path_or_repo: str, revision: str | None = None) -> str:
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, revision=revision)


def _cache_dir() -> str:
    return os.environ.get("OPENDECIDER_MODELS", "models")


def _fetch(repo: str, revision: str, patterns=None, filename: str | None = None) -> str:
    """Download (once) a pinned backbone repo, or one file of it, into the model cache; returns the local dir."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from .guards import require_free_gb
    d = os.path.join(_cache_dir(), f"{repo.replace('/', '__')}@{revision[:12]}")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    want = [s for s in info.siblings if (filename and s.rfilename == filename) or
            (not filename and (patterns is None or any(s.rfilename.endswith(p.lstrip("*")) for p in patterns)))]
    missing = [s for s in want if not os.path.exists(os.path.join(d, s.rfilename))]
    if missing:
        require_free_gb(sum((s.size or 0) for s in missing) / 2 ** 30 + 0.2, path=d)
        if filename:
            hf_hub_download(repo, filename, revision=revision, local_dir=d)
        else:
            snapshot_download(repo, revision=revision, local_dir=d, allow_patterns=patterns)
    return d


def resolve_backbone(spec: dict) -> dict:
    """Backbone variant spec (from config.json) -> BackboneConfig kwargs with a local path."""
    spec = dict(spec)
    src = spec.pop("source")
    if src.get("gguf_file"):                                         # GGUF weights + config/tokenizer from the base
        g = _fetch(src["repo_id"], src["revision"], filename=src["gguf_file"])
        base = _fetch(src["config_repo"], src["config_revision"], patterns=TOKENIZER_FILES)
        d = os.path.join(_cache_dir(), f"{src['repo_id'].replace('/', '__')}@{src['revision'][:12]}-gguf")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(base):                                   # one directory the GGUF loader can read
            if f in TOKENIZER_FILES and not os.path.exists(os.path.join(d, f)):
                os.symlink(os.path.abspath(os.path.join(base, f)), os.path.join(d, f))
        link = os.path.join(d, src["gguf_file"])
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(g, src["gguf_file"])), link)
        spec["path"] = link
    else:
        spec["path"] = _fetch(src["repo_id"], src["revision"],
                              patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    spec["repo_id"], spec["revision"] = src["repo_id"], src["revision"]
    return spec


def load_model(path_or_repo: str, backbone: str | None = None, device: str | None = None, kv_quant: str | None = None,
               revision: str | None = None):
    """-> (model, extra). device: cuda if available, else cpu (reference kernels; slow but works)."""
    from safetensors import safe_open
    from .backbone import Backbone, BackboneConfig, use_reference_kernels
    from .models import ModelConfig, build_model
    d = _local(path_or_repo, revision)
    cfg = json.load(open(os.path.join(d, "config.json")))
    name = backbone or cfg["default_backbone"]
    if name not in cfg["backbones"]:
        raise ValueError(f"unknown backbone variant {name!r}; available: {sorted(cfg['backbones'])}")
    bspec = resolve_backbone(cfg["backbones"][name])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":                       # torch reference kernels; GemLite/AWQ/NF4 need CUDA
        if bspec["weight_quant"] not in ("int8", "none"):
            raise ValueError(f"backbone variant {name!r} ({bspec['weight_quant']}) needs a CUDA GPU; on CPU use the "
                             f"int8 variant")
        use_reference_kernels(True)
        bspec["dtype"] = "float32"
    bspec["device"] = device
    if kv_quant:
        bspec["kv_quant"] = kv_quant
    bb = Backbone.load(BackboneConfig(**bspec))
    model = build_model(bb, ModelConfig(**cfg["model_cfg"]))
    sd, parts = {}, {}
    for f in sorted(set(cfg["weight_map"].values())):
        with safe_open(os.path.join(d, f), "pt") as h:
            for k in h.keys():
                t = h.get_tensor(k).float()
                if "::part" in k:
                    parts.setdefault(k.split("::")[0], []).append((int(k.split("::part")[1]), t))
                else:
                    sd[k] = t
    for k, ps in parts.items():
        sd[k] = torch.cat([t for _, t in sorted(ps)], 0)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("backbone.")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
    return model.to(bb.device).eval(), cfg.get("extra", {})


def load_decider(path_or_repo: str, backbone: str | None = None, device: str | None = None,
                 kv_quant: str | None = None, revision: str | None = None):
    from .decider import Decider, prepare_inference_
    model, extra = load_model(path_or_repo, backbone, device, kv_quant, revision)
    prepare_inference_(model)
    return Decider(model, temperature=extra.get("temperature", 1.0),
                   temperature_by_type=extra.get("temperature_by_type"), conformal=extra.get("conformal"))
