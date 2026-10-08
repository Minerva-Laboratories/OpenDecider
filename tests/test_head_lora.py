"""LoRA on the head: starts at the warm start, trains only LoRA (+ tiny Linears), never touches the backbone, and the
merged checkpoint reproduces the LoRA model."""
import torch

from opendecider.batching import Question
from opendecider.checkpoint import load_model, save
from opendecider.head_lora import add_head_lora_
from tests.conftest import make
from tests.tiny import tiny_backbone


def _logits(m, qs, states):
    with torch.no_grad():
        _, out = m.run(qs, m.encode_states(states))
    return out.logits


def test_head_lora(tmp_path):
    bb = tiny_backbone(layers=(4, "final"))
    m = make(bb, "v3", v3_features="branched", v3_row_format="answer", slot_emb="none", v3_cross="question")
    qs = [Question("Which team?", ["billing", "tech", "sales"], state_idx=0), Question("Urgent?", ["yes", "no"], state_idx=1)]
    states = ["State:\nI was charged twice.\n", "State:\nThe app crashes.\n"]
    ref = _logits(m, qs, states)
    lora = add_head_lora_(m, rank=4)
    assert torch.allclose(_logits(m, qs, states), ref)                       # B = 0: exactly the warm start
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("head_lora.") or any(n.startswith(f + ".") for f in lora.full) for n in names)
    assert not any(n.startswith("backbone.") for n in names)
    with torch.no_grad():
        for p in lora.B.values():
            p.normal_(0, 0.05)
    tuned = _logits(m, qs, states)
    assert not torch.allclose(tuned, ref)
    save(m, str(tmp_path / "model.pt"))
    m2, _ = load_model(str(tmp_path / "model.pt"), backbone=bb)
    assert not any(k.startswith("head_lora") for k in m2.state_dict())
    assert torch.allclose(_logits(m2, qs, states), tuned, atol=1e-5)          # merged weights == LoRA model
