"""End-to-end inference: one state, many isolated questions, probabilities out. No text generation.

State is encoded once (cached memory, int8). All questions with <=255 options go through one
batched decision call; >255-option Choice questions use the two-stage path (O2).
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time

import torch

from .batching import Question, state_texts
from .formatting import YES_NO, solved_cases_prefix
from .heads import two_stage_select
from .models import DecisionModel
from .options import MAX_OPTIONS
from .sampler import SamplerConfig, run_sampler
from .schema import REF, DecideRequest, opt_name, opt_text, waves


def _normalize(p: torch.Tensor) -> list[float]:
    p = p.double().clamp_min(0)
    p = p / p.sum()
    return p.tolist()


class Decider:
    def __init__(self, model: DecisionModel, temperature: float = 1.0, temperature_by_type: dict | None = None,
                 profiles_dir: str = "runs/profiles"):
        self.model = model.eval()
        self.temperature = temperature
        self.temperature_by_type = temperature_by_type or {}
        self.profiles_dir = profiles_dir
        self._profiles: dict = {}
        # explicit `none` option: appended as TEXT to every question, so the backbone reads it in context like any
        # other option (no training). "" turns it off; models with the learned sink (v3_none) use that instead.
        self.none_text = os.environ.get("OPENDECIDER_NONE_TEXT", "none of the above")

    def profile(self, pid: str | None):
        if not pid:
            return None
        if pid not in self._profiles:
            path = os.path.join(self.profiles_dir, f"{pid}.json")
            if not os.path.exists(path):
                raise ValueError(f"unknown calibration profile {pid!r}")
            self._profiles[pid] = json.load(open(path))
        return self._profiles[pid]

    @staticmethod
    def _atomic(name: str, spec, answers: dict) -> Question:
        """Atomic spec -> Question. Substitutes {other_question} with its chosen value (chaining); appends
        instructions / Noul criteria to what the model reads; options are read as `name: description`."""
        sub = lambda t: REF.sub(lambda m: str(answers[m.group(1)]["value"]), t)
        prompt = sub(spec.prompt)
        if spec.instructions:
            prompt += "\nInstructions: " + sub(spec.instructions)
        if spec.type == "noul" and spec.criteria:
            prompt += "\nCriteria for yes: " + sub(spec.criteria)
        if spec.type == "choice":
            q = Question(prompt, [opt_text(o) for o in spec.options], "choice", name=name)
            q.names = [opt_name(o) for o in spec.options]
        elif spec.type == "noul":
            q = Question(prompt, list(YES_NO), "noul", name=name)
            q.names = list(YES_NO)
        else:
            q = Question(prompt, [opt_text(o) for o in spec.levels], "score", name=name)
            q.names = [opt_name(o) for o in spec.levels]
        q.calibration = spec.calibration
        q.none = spec.none
        return q

    def _context_prefix(self, req) -> str:
        """Render request-level examples in the model's own question schema (answer filled or empty)."""
        if not req.examples:
            return ""
        exs = []
        for ex in req.examples:
            qa = []
            for name, spec in req.questions.items():
                if spec.type == "composite" or name not in ex.answers:
                    continue
                q = self._atomic(name, spec, {})
                a = ex.answers[name]
                if isinstance(a, bool):
                    a = "yes" if a else "no"
                elif isinstance(a, int) and spec.type == "score":
                    a = q.names[a]
                qa.append((q.prompt, q.options, None if a is None else str(a)))
            exs.append({"state": ex.state, "qa": qa})
        cap = getattr(self.model.cfg, "v3_list_cap", 16)
        return solved_cases_prefix(exs, self.model.row_format, cap)

    @staticmethod
    def _combine(spec, parts: dict) -> dict:
        """Composite answer: logistic combination of sub-question probabilities (caller-fitted weights)."""
        z = spec.combine.bias
        for f, w in spec.combine.weights.items():
            z += w * composite_feature(f, parts)
        p = 1.0 / (1.0 + math.exp(-z))
        return {"value": "yes" if p >= 0.5 else "no", "p_yes": p, "probs": {"yes": p, "no": 1.0 - p},
                "probs_list": [p, 1.0 - p], "parts": parts}

    @torch.no_grad()
    def _probs(self, questions: list[Question], mem, sampler: SamplerConfig):
        from .calibrators import calibrate_rows
        m = self.model

        def once(noise):
            mm = m.with_noise(mem, noise) if noise > 0 else mem
            _, out = m.run(questions, mm)
            Ts, biases, groups = [], torch.zeros_like(out.logits), {}
            for i, q in enumerate(questions):
                prof = self.profile(getattr(q, "calibration", None))
                if prof is None:
                    Ts.append(self.temperature_by_type.get(q.type, self.temperature))
                else:                                     # per-question fitted head: temperature + per-option bias
                    Ts.append(prof["temperature"])
                    names = getattr(q, "names", None) or q.options
                    b = prof.get("bias", {})
                    biases[i, : len(names)] = torch.tensor([b.get(n, 0.0) for n in names], device=biases.device)
                    if prof.get("calibrator"):            # probability-level stage, batched per profile below
                        groups.setdefault(q.calibration, []).append(i)
            T = torch.tensor(Ts, device=out.logits.device, dtype=out.logits.dtype)[:, None]
            logits = out.logits / T + biases
            p = torch.softmax(logits, -1) * out.opt_mask
            if groups:
                # profiles are fitted on the real options only: set the explicit `none` column aside, calibrate the
                # renormalised real options, then put P(none) back
                col = torch.tensor([getattr(q, "none_col", -1) for q in questions], device=p.device)
                has = col >= 0
                rows_n = has.nonzero().squeeze(1)
                pn = torch.zeros(len(questions), device=p.device, dtype=p.dtype)
                mask = out.opt_mask.clone()
                if len(rows_n):
                    pn[rows_n] = p[rows_n, col[rows_n]]
                    p[rows_n, col[rows_n]] = 0
                    mask[rows_n, col[rows_n]] = False
                    p[rows_n] = p[rows_n] / p[rows_n].sum(1, keepdim=True).clamp_min(1e-12)
                for pid, rows in groups.items():
                    r = torch.tensor(rows, device=p.device)
                    p[r] = calibrate_rows(self._profiles[pid]["calibrator"], p[r], out.logits[r], mask[r])
                if len(rows_n):
                    p[rows_n] = p[rows_n] * (1 - pn[rows_n])[:, None]
                    p[rows_n, col[rows_n]] = pn[rows_n]
            if m.cfg.noul_mode == "sigmoid" and out.noul_logit is not None:
                for i, q in enumerate(questions):
                    if q.type == "noul":
                        py = torch.sigmoid(out.noul_logit[i] / self.temperature)
                        p[i, 0], p[i, 1] = py, 1 - py
            return p

        return run_sampler(once, sampler, m)

    @torch.no_grad()
    def _relevance(self, q: Question, mem) -> torch.Tensor:
        """Stage 1 (O2): independent sigmoid relevance in chunks of <=255."""
        scores = []
        for s in range(0, len(q.options), MAX_OPTIONS):
            chunk = Question(q.prompt, q.options[s:s + MAX_OPTIONS], q.type, name=q.name)
            _, out = self.model.run([chunk], mem)
            scores.append(out.relevance[0, : len(chunk.options)])
        return torch.cat(scores)

    def _with_none(self, q: Question) -> Question:
        """Copy of q with the explicit `none` option appended (skipped when opted out or at the 255-option cap)."""
        if getattr(q, "none", None) is False or len(q.options) >= MAX_OPTIONS:
            return q
        q2 = copy.copy(q)
        q2.options = list(q.options) + [self.none_text]
        q2.none_col = len(q.options)
        return q2

    def _answer_batch(self, qs: list[Question], mem, sampler) -> tuple[dict, int]:
        """Answer isolated questions in one batch (<=255 options) plus the two-stage path (>255)."""
        small = [q for q in qs if len(q.options) <= MAX_OPTIONS]
        large = [q for q in qs if len(q.options) > MAX_OPTIONS]
        answers, n_tok = {}, 0
        for q in large:                                     # two-stage (O2)
            keep = two_stage_select(self._relevance(q, mem)).tolist()
            sub = Question(q.prompt, [q.options[i] for i in keep], q.type, name=q.name)
            sub.orig_index = keep
            small.append(sub)
            n_tok += sum(len(t) for t in self.model.backbone.tokenize([q.prompt] + q.options))
        if small:
            sink = getattr(self.model.cfg, "v3_none", False)            # learned sink = last column
            if self.none_text and not sink:                             # explicit text option, after the real ones
                small = [self._with_none(q) for q in small]
            mean, std = self._probs(small, mem, sampler)
            for i, q in enumerate(small):
                K = getattr(q, "none_col", len(q.options))
                probs = _normalize(mean[i, :K])                      # renormalised over the caller's options
                sd = std[i, :K].double().tolist()
                names = getattr(q, "names", None) or q.options
                pmax = max(probs)
                a = {"value": names[max(range(K), key=probs.__getitem__)],
                     # Jev-style confidence: top probability rescaled so that 1/K -> 0 and 1 -> 1
                     "confidence": max(0.0, min(1.0, (K * pmax - 1) / (K - 1))) if K > 1 else 1.0,
                     "probs": dict(zip(names, probs)), "std": dict(zip(names, sd)),
                     # ordered list: keeps duplicate option strings distinguishable (probe battery)
                     "probs_list": probs}
                if (sink and getattr(q, "none", None) is not False) or hasattr(q, "none_col"):
                    p_none = float(mean[i, q.none_col] if hasattr(q, "none_col") else mean[i, -1])
                    a["none"] = p_none                               # P(no listed option is supported)
                    if p_none > float(mean[i, :K].max()):
                        a["value"] = "none"
                if hasattr(q, "orig_index"):
                    a["stage1_kept"] = len(q.options)
                if q.type == "noul":
                    a["p_yes"] = probs[0]
                if q.type == "score":
                    a["expected_index"] = sum(k * p for k, p in enumerate(probs))
                answers[q.name] = a
            n_tok += sum(len(t) for q in small if not hasattr(q, "orig_index")
                         for t in self.model.backbone.tokenize([q.prompt] + q.options))
        return answers, n_tok

    def _prefix_cache(self):
        """Cross-request state cache (Tier 1), on for models whose question rows continue the state's LLM cache.
        Size from OPENDECIDER_STATE_CACHE_MB (default 2048; 0 disables)."""
        pc = self.__dict__.get("_pcache", False)
        if pc is False:
            mb = int(os.environ.get("OPENDECIDER_STATE_CACHE_MB", "2048"))
            pc = None
            cfg = self.model.cfg
            # state-token features are only read by the trunk when it cross-attends to the state; with
            # v3_cross='question' the row pass needs the LLM cache alone, which is what gets cached
            if mb > 0 and getattr(self.model, "state_free", False):
                from .state_cache import PrefixCache
                pc = PrefixCache(self.model.backbone, max_bytes=mb << 20)
            self.__dict__["_pcache"] = pc
        return pc

    @contextlib.contextmanager
    def _state_pass_ctx(self):
        """Same adapter state as the model's own state pass (VeRA scope 'all' adapts the state tokens too)."""
        m = self.model
        vera = getattr(m, "vera", None)
        on = vera is not None and m.cfg.v3_vera_scope == "all"
        if on:
            vera.active = True
        try:
            yield
        finally:
            if on:
                vera.active = False

    def _encode_state(self, req):
        """Encode the request's state once. With the prefix cache, the state is chunked at record boundaries and
        resumed from the deepest cached prefix (exact); returns (memory, number of tokens served from cache)."""
        prefix = self._context_prefix(req)
        pcache = self._prefix_cache()
        if pcache is None:
            return self.model.encode_states([prefix + state_texts([req.state])[0]]), 0
        from .chunking import state_chunks, text_chunks
        content, closers = state_chunks(req.state)
        content = text_chunks(prefix) + content
        mem = self.model.encode_states(["".join(content + closers)])
        pc, smask, n_cached = pcache.encode(content, closers, ctx=self._state_pass_ctx)
        mem.prefix_batch = (pc, smask, None)          # state-free trunk: no state-token features needed
        mem.n_tokens = smask.shape[1]
        return mem, n_cached

    def decide(self, req: DecideRequest | dict) -> dict:
        """State is encoded ONCE. Questions run in dependency waves (chaining); each wave is one isolated batch.
        Composite questions answer all their parts in the same batch and combine the probabilities."""
        if isinstance(req, dict):
            req = DecideRequest.model_validate(req)
        t0 = time.perf_counter()
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        sampler = SamplerConfig(mode=req.sampler.mode if req.sampler.k > 1 else "off", k=req.sampler.k)
        mem, n_cached = self._encode_state(req)
        sync(); t_state = time.perf_counter()
        answers, n_tok = {}, 0
        plan = waves(req.questions)
        for wave in plan:
            batch = []
            for name in wave:
                spec = req.questions[name]
                if spec.type == "composite":
                    batch += [self._atomic(f"{name}.{pn}", ps, answers) for pn, ps in spec.parts.items()]
                else:
                    batch.append(self._atomic(name, spec, answers))
            got, nt = self._answer_batch(batch, mem, sampler)
            n_tok += nt
            for name in wave:
                spec = req.questions[name]
                if spec.type == "composite":
                    answers[name] = self._combine(spec, {pn: got[f"{name}.{pn}"] for pn in spec.parts})
                else:
                    answers[name] = got[name]
        sync(); t_end = time.perf_counter()
        return {"answers": {k: answers[k] for k in req.questions},
                "timing_ms": {"state": 1e3 * (t_state - t0), "questions": 1e3 * (t_end - t_state),
                              "total": 1e3 * (t_end - t0), "waves": len(plan)},
                "state_cached_tokens": n_cached,
                "input_tokens": mem.n_tokens + n_tok}


def composite_feature(f: str, parts: dict) -> float:
    """Feature value for a combine weight key: "part" (noul P(yes)), "part=option", "part.expected"."""
    if "=" in f:
        part, opt = f.split("=", 1)
        return float(parts[part]["probs"].get(opt, 0.0))
    if f.endswith(".expected"):
        return float(parts[f[: -len(".expected")]]["expected_index"])
    a = parts[f]
    return float(a["p_yes"]) if "p_yes" in a else float(max(a["probs"].values()))
