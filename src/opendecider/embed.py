"""Dense text embeddings from the frozen backbone itself (no second model), for the Tier 2 store.

Mean-pooled hidden states of a MIDDLE layer (middle layers transfer better than the last one for similarity; Skean
et al., ICML 2025), L2-normalised. Texts are batched by length to bound padding.
"""
from __future__ import annotations

import numpy as np
import torch


class BackboneEmbedder:
    def __init__(self, backbone, layer: int | None = None, batch_tokens: int = 16384, max_tokens: int = 512):
        self.bb = backbone
        self.layer = layer or backbone.num_layers // 2
        self.batch_tokens, self.max_tokens = batch_tokens, max_tokens

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> np.ndarray:
        toks = [t[: self.max_tokens] or [self.bb.pad_id] for t in self.bb.tokenize(list(texts))]
        order = np.argsort([len(t) for t in toks], kind="stable")
        out = np.zeros((len(toks), self.bb.hidden_size), dtype=np.float32)
        a = 0
        while a < len(order):                                   # length-sorted batches under a token budget
            b = a + 1
            while b < len(order) and (b - a + 1) * len(toks[order[b]]) <= self.batch_tokens:
                b += 1
            idx = order[a:b]
            ids, mask = self.bb.pad([toks[i] for i in idx])
            f, _ = self.bb(ids, mask, (self.layer,))
            m = mask[..., None].to(f.dtype)
            v = (f * m).sum(1) / m.sum(1).clamp_min(1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out[idx] = v.cpu().numpy()
            a = b
        return out


def x2b_embedder(ckpt: str = "runs/x2b/model.pt"):
    """embed_fn from the backbone a checkpoint was trained on (factory for eval/store_recall.py --embedder)."""
    from .backbone import Backbone, BackboneConfig
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return BackboneEmbedder(Backbone.load(BackboneConfig(**ck["backbone_cfg"])))
