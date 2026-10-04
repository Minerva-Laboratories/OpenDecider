import json

import torch

from conftest import make
from opendecider import hub
from opendecider.batching import Question
from tiny import tiny_backbone


def test_export_then_load_roundtrip(tmp_path, monkeypatch):
    bb = tiny_backbone(layers=(4, "final"))
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question")
    ck = tmp_path / "model.pt"
    torch.save({"state_dict": {k: v for k, v in m.state_dict().items() if not k.startswith("backbone.")},
                "model_cfg": m.cfg.__dict__, "backbone_cfg": bb.cfg.__dict__, "extra": {"temperature": 1.3}}, ck)
    out = tmp_path / "release"
    spec = {"int8": {"dtype": "float32", "kv_quant": "int8", "feature_layers": [4, "final"], "weight_quant": "none",
                     "source": {"repo_id": "local/tiny", "revision": "0" * 40}}}
    cfg = hub.export(str(ck), str(out), spec, "int8", card="# tiny", shard_mb=0.005)
    assert len(set(cfg["weight_map"].values())) > 2                     # sharded
    assert any("::part" in k for k in cfg["weight_map"])                # one matrix split across shards
    # backbone resolution is network-bound: hand the loader the same tiny backbone
    monkeypatch.setattr(hub, "resolve_backbone", lambda s: {k: v for k, v in s.items() if k != "source"})
    import opendecider.backbone as B
    monkeypatch.setattr(B.Backbone, "load", classmethod(lambda cls, c: bb))
    m2, extra = hub.load_model(str(out), device="cpu")
    assert extra["temperature"] == 1.3 and json.load(open(out / "config.json"))["default_backbone"] == "int8"
    qs = [Question("Which team?", ["billing", "technical", "refund"], "choice")]
    with torch.no_grad():
        a = m.run(qs, m.encode_states(["State:\n{\"msg\": \"charged twice\"}\n"]))[1].logits
        b = m2.run(qs, m2.encode_states(["State:\n{\"msg\": \"charged twice\"}\n"]))[1].logits
    assert torch.allclose(a, b, atol=2e-2)                              # fp16 storage
