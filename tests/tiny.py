"""Tiny random Qwen3.5-architecture backbone for fast CPU unit tests (real tokenizer, random weights)."""
import functools
import os
import torch

from opendecider.backbone import Backbone, BackboneConfig, use_reference_kernels

use_reference_kernels(True)

# Qwen3.5 tokenizer: local copy if present, else the pinned Hub revision
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen3.5-0.8b")
TOKENIZER = _LOCAL if os.path.isdir(_LOCAL) else "Qwen/Qwen3.5-0.8B"
REVISION = None if TOKENIZER == _LOCAL else "2fc06364715b967f1860aea9cf38778875588b17"


@functools.lru_cache(maxsize=None)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION)


def tiny_backbone(kv_quant="int8", weight_quant="none", layers=("final",), seed=0, dtype="float32"):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from opendecider.quant import Int8Embedding, quantize_int8_
    torch.manual_seed(seed)
    tok = tokenizer()
    c = Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=True, pad_token_id=tok.pad_token_id)
    m = Qwen3_5ForCausalLM(c).to(getattr(torch, dtype)).eval()
    cfg = BackboneConfig(path=TOKENIZER, dtype=dtype, weight_quant=weight_quant, kv_quant=kv_quant,
                         feature_layers=list(layers), device="cpu")
    lm = m.model
    if weight_quant == "int8":
        quantize_int8_(lm.layers)
        lm.embed_tokens = Int8Embedding(lm.embed_tokens)
    return Backbone(lm, tok, cfg)
