"""llama.cpp backend: the frozen backbone runs in llama.cpp (a GGUF file), the trained head in PyTorch.

Only for branched-row checkpoints whose trunk does not read state tokens (the 9B: v3_features=branched,
v3_cross=question, v3_lm_feature): the head needs, per option row [question ; option] continuing from the state, the
features after blocks 8 / 16 / 24 and after the final norm, and the LM log-probability of every option token.
deploy/llamacpp/od_llama.cpp computes exactly those (feasibility: RESULTS 20.2, 0.869 vs 0.875 on Banking77 with the
same Q4_0 file). The 9B's VeRA adapters are left out: they are below bf16 resolution at inference (RESULTS 20.2).

    OPENDECIDER_LLAMACPP_GGUF=models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_0.gguf OPENDECIDER_CKPT=runs/x9b-clean3/model.pt \\
        python -m opendecider.api

Library: deploy/llamacpp/build.sh (OPENDECIDER_LLAMACPP_LIB overrides the path). Tunables: OPENDECIDER_LLAMACPP_CTX
(default 16384 tokens: state + one batch of rows), _SEQ (rows per batch + 1, default 17), _UBATCH (tokens per batch,
default 2048), _NGL (layers on the GPU, default 99).
"""
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np
import torch

from .backbone import BackboneConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(_HERE, "..", "..", "deploy", "llamacpp", "build", "libod_llama.so")
FEATURE_LAYERS = [8, 16, 24, "final"]          # what od_llama.cpp captures (l_out-7/15/23, h_nextn)


class LlamaRows:
    """ctypes wrapper of libod_llama.so: one llama.cpp context (one state at a time)."""

    def __init__(self, gguf: str, lib: str | None = None, n_ctx: int = 16384, n_seq: int = 17, n_ubatch: int = 2048,
                 ngl: int = 99, n_threads: int | None = None):
        L = ctypes.CDLL(os.path.abspath(lib or os.environ.get("OPENDECIDER_LLAMACPP_LIB", DEFAULT_LIB)))
        L.odl_init.restype = ctypes.c_void_p
        L.odl_init.argtypes = [ctypes.c_char_p] + [ctypes.c_int] * 5
        L.odl_n_embd.argtypes = [ctypes.c_void_p]
        L.odl_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.odl_rows.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 5
        L.odl_rows_qcache.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 8
        L.odl_free.argtypes = [ctypes.c_void_p]
        self.L = L
        self.h = L.odl_init(gguf.encode(), n_ctx, n_seq, n_ubatch, ngl, n_threads or min(8, os.cpu_count() or 4))
        if not self.h:
            raise RuntimeError(f"llama.cpp could not load {gguf}")
        self.n_embd = L.odl_n_embd(self.h)
        self.n_ctx, self.n_ubatch = n_ctx, n_ubatch
        self.lock = threading.Lock()                 # one context: calls are serialized

    def state(self, ids) -> bool:
        """Run the state (sequence 0); True when it was already there (same tokens as the last call)."""
        a = np.ascontiguousarray(ids, np.int32)
        rc = self.L.odl_state(self.h, a.ctypes.data, len(a))
        if rc < 0:
            raise RuntimeError(f"llama.cpp state pass failed ({rc}); state of {len(a)} tokens, n_ctx {self.n_ctx}")
        return rc == 1

    def rows(self, rows: list[list[int]], nq: list[int]):
        """Rows continuing from the current state -> (features (sum len, 4 n_embd) fp16, option-token log-probs)."""
        lens = np.asarray([len(r) for r in rows], np.int32)
        nqs = np.asarray(nq, np.int32)
        toks = np.ascontiguousarray(np.concatenate([np.asarray(r, np.int32) for r in rows]))
        feats = np.empty((int(lens.sum()), 4 * self.n_embd), np.float16)
        lp = np.empty(int((lens - nqs).sum()), np.float32)
        rc = self.L.odl_rows(self.h, len(rows), toks.ctypes.data, lens.ctypes.data, nqs.ctypes.data,
                             feats.ctypes.data, lp.ctypes.data)
        if rc < 0:
            raise RuntimeError(f"llama.cpp row pass failed ({rc})")
        return feats, lp

    def rows_qcache(self, questions: list[list[int]], options: list[list[list[int]]]):
        """Question cache: each question once from the state, its options forked from it. -> (question features
        (sum q len, 4 n_embd), option features (sum option len, 4 n_embd)) fp16 and option-token log-probs."""
        i32 = lambda x: np.ascontiguousarray(x, np.int32)
        ql = i32([len(q) for q in questions]); no = i32([len(o) for o in options])
        flat_o = [t for os_ in options for t in os_]
        ol = i32([len(t) for t in flat_o])
        qt = i32(np.concatenate([np.asarray(q, np.int32) for q in questions]))
        ot = i32(np.concatenate([np.asarray(t, np.int32) for t in flat_o]))
        qf = np.empty((int(ql.sum()), 4 * self.n_embd), np.float16)
        of = np.empty((int(ol.sum()), 4 * self.n_embd), np.float16)
        lp = np.empty(int(ol.sum()), np.float32)
        rc = self.L.odl_rows_qcache(self.h, len(questions), qt.ctypes.data, ql.ctypes.data, no.ctypes.data,
                                    ot.ctypes.data, ol.ctypes.data, qf.ctypes.data, of.ctypes.data, lp.ctypes.data)
        if rc < 0:
            raise RuntimeError(f"llama.cpp row pass failed ({rc})")
        return qf, of, lp

    def close(self):
        if self.h:
            self.L.odl_free(self.h)
            self.h = None


class LlamaCppBackbone:
    """Stands in for opendecider.backbone.Backbone where the head needs it (tokenizer, sizes, device); the frozen LM
    itself is the GGUF in llama.cpp. Not an nn.Module: no backbone weights in PyTorch."""
    is_llamacpp = True
    supports_shared_prefix = False

    def __init__(self, cfg: BackboneConfig, gguf: str, device: str | None = None, **kw):
        from transformers import AutoTokenizer
        self.cfg = cfg
        if list(cfg.feature_layers) != FEATURE_LAYERS:
            raise ValueError(f"the llama.cpp backend captures layers {FEATURE_LAYERS}, the checkpoint reads "
                             f"{cfg.feature_layers}")
        self.tok = AutoTokenizer.from_pretrained(cfg.repo_id, revision=cfg.revision)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        env = lambda k, d: int(os.environ.get(f"OPENDECIDER_LLAMACPP_{k}", d))
        self.llama = LlamaRows(gguf, n_ctx=env("CTX", 16384), n_seq=env("SEQ", 17), n_ubatch=env("UBATCH", 2048),
                               ngl=env("NGL", 99), **kw)
        self.hidden_size = self.llama.n_embd
        self.num_layers = None
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def device(self):
        return self._device

    def tokenize(self, texts):
        if not texts:
            return []
        return self.tok(list(texts), add_special_tokens=False)["input_ids"]

    def pad(self, seqs, device=None):
        lens = [len(s) for s in seqs]
        T = max(1, max(lens))
        ids = torch.tensor([list(s) + [self.pad_id] * (T - len(s)) for s in seqs], dtype=torch.long)
        mask = torch.arange(T)[None, :] < torch.tensor(lens)[:, None]
        return ids.to(device or self.device), mask.to(device or self.device)


def branched_feats(model, questions, mem):
    """V3._branched_feats on llama.cpp: same token layout (answer-row format), same outputs
    (s_flat=None, s_lens, q_flat, q_lens, o_flat, o_lens) and model._lm_lp."""
    from .formatting import answer_option_text, answer_row_prefix
    bb = model.backbone
    cap = model.cfg.v3_list_cap
    q_tok = bb.tokenize([answer_row_prefix(q.prompt, q.options, cap) for q in questions])
    o_tok = bb.tokenize([answer_option_text(o) for q in questions for o in q.options])
    states = mem.extra_prefix
    n_opt = [len(q.options) for q in questions]
    first = np.cumsum([0] + n_opt[:-1])
    qf, of, lps = [None] * len(questions), [None] * len(questions), [None] * len(questions)
    with bb.llama.lock:
        for s in sorted({q.state_idx for q in questions}):           # one state at a time
            bb.llama.state(states[s])
            qi = [i for i, q in enumerate(questions) if q.state_idx == s]
            Fq, Fo, lp = bb.llama.rows_qcache([q_tok[i] for i in qi],
                                              [o_tok[first[i]:first[i] + n_opt[i]] for i in qi])
            a = b = 0
            for i in qi:
                nq, no = len(q_tok[i]), sum(len(t) for t in o_tok[first[i]:first[i] + n_opt[i]])
                qf[i], of[i], lps[i] = Fq[a:a + nq], Fo[b:b + no], lp[b:b + no]
                a += nq; b += no
    dev = bb.device
    q_flat = torch.from_numpy(np.concatenate(qf)).to(dev).bfloat16()
    o_flat = torch.from_numpy(np.concatenate(of)).to(dev).bfloat16()
    model._lm_lp = torch.from_numpy(np.concatenate(lps)).to(dev).float()
    lq = torch.tensor([len(t) for t in q_tok]); lo = torch.tensor([len(t) for t in o_tok])
    s_lens = torch.tensor([len(s) for s in states])
    return None, s_lens, q_flat, lq, o_flat, lo


def build(backbone_cfg: dict, model_cfg: dict, state_dict: dict, gguf: str, device: str | None = None):
    """A branched-row model with its backbone in llama.cpp (VeRA left out) from its configs and head weights."""
    from .models import ModelConfig, build_model
    bcfg = BackboneConfig(**backbone_cfg)
    mcfg = ModelConfig(**{**model_cfg, "v3_vera_layers": 0, "v3_question_cache": False})
    if not (mcfg.variant == "v3" and mcfg.v3_features == "branched" and mcfg.v3_row_format == "answer"
            and mcfg.v3_cross == "question" and mcfg.v3_lm_feature):
        raise ValueError("the llama.cpp backend serves branched answer-row checkpoints with a state-free trunk "
                         "and the LM feature (the 9B)")
    bb = LlamaCppBackbone(bcfg, gguf, device)
    model = build_model(bb, mcfg)
    sd = {k: v for k, v in state_dict.items() if not k.startswith(("backbone.", "vera."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(bb.device).eval()


def load_model(path: str, gguf: str, device: str | None = None):
    """A training checkpoint (model.pt) on the llama.cpp backend. -> (model, extra)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return build(ck["backbone_cfg"], ck["model_cfg"], ck["state_dict"], gguf, device), ck.get("extra", {})
