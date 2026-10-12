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
from .schema import REF, DecideRequest, opt_name, opt_text, refs_of, waves


def _normalize(p: torch.Tensor) -> list[float]:
    p = p.double().clamp_min(0)
    p = p / p.sum()
    return p.tolist()


# marginal chaining: joint assignments of the referenced answers asked per question (heaviest first), and the weight
# below which an assignment is dropped (the heaviest is always kept); the mixture is renormalized over those kept
CHAIN_MAX_BRANCHES = 32
CHAIN_MIN_WEIGHT = 1e-4


def chain_branches(refs: list[str], answers: dict) -> list[tuple[dict, float]]:
    """[(given {ref: option name}, weight)] for marginal chaining: weight = product of the referenced answers'
    probabilities (upstream answers treated as independent given the state), heaviest first."""
    out = [({}, 1.0)]
    for r in refs:
        probs = answers[r]["probs"]
        out = sorted(((dict(g, **{r: v}), w * float(p)) for g, w in out for v, p in probs.items()),
                     key=lambda t: -t[1])[:CHAIN_MAX_BRANCHES]
    return out[:1] + [b for b in out[1:] if b[1] >= CHAIN_MIN_WEIGHT]


def mix_answers(branch_answers: list[dict], weights: list[float], givens: list[dict]) -> dict:
    """One answer from the same question asked under several substitutions: probabilities mixed by the branch
    weights (renormalized). `chain` lists every branch; `std` is the within-branch sampler spread, sqrt(sum w sd^2)."""
    tot = sum(weights)
    w = [x / tot for x in weights]
    a0 = branch_answers[0]
    names = list(a0["probs"])
    probs = [sum(wk * a["probs_list"][j] for wk, a in zip(w, branch_answers)) for j in range(len(names))]
    s = sum(probs); probs = [p / s for p in probs]
    K = len(names); pmax = max(probs)
    a = {"value": names[max(range(K), key=probs.__getitem__)],
         "confidence": max(0.0, min(1.0, (K * pmax - 1) / (K - 1))) if K > 1 else 1.0,
         "probs": dict(zip(names, probs)), "probs_list": probs}
    if "std" in a0:
        a["std"] = {n: math.sqrt(sum(wk * b["std"][n] ** 2 for wk, b in zip(w, branch_answers))) for n in names}
    if "none" in a0:
        a["none"] = sum(wk * b["none"] for wk, b in zip(w, branch_answers))
        if a["none"] > pmax:
            a["value"] = "none"
    if "p_yes" in a0:
        a["p_yes"] = probs[0] if "parts" not in a0 else a["probs"]["yes"]
    if "expected_index" in a0:
        a["expected_index"] = sum(k * p for k, p in enumerate(probs))
    if "parts" in a0:                                   # composite: the parts of the heaviest branch
        a["parts"] = a0["parts"]
    a["chain"] = {"mode": "marginal", "covered": tot,
                  "branches": [{"given": g, "weight": wk, "probs_list": b["probs_list"]}
                               for g, wk, b in zip(givens, w, branch_answers)]}
    return a


# VeRA updates smaller than this fraction of the frozen weight (Frobenius norm) are below bf16 resolution (8-bit
# significand: relative spacing 2^-8) and are left out at serving. The 9B's is 0.0019 (on activations 0.04-0.14%,
# RESULTS 20.2); dropping it gives typed decisions 0.560 vs 0.552 and the same Banking77 / injection accuracy
VERA_INERT = 2 ** -8

# option rows per call from which the question cache is faster (scripts/bench_qcache_rows.py, Orin, 50 W, ~0.9k-token
# state): about 64 on the 2B, about 12 on the 9B, where every repeated token costs ~4x more
QCACHE_MIN_ROWS = {2048: 64, 4096: 12}


def prepare_inference_(model) -> None:
    """Speedups for serving that keep the trained function (both tested equal to the training-time path):
    - branched answer rows read each question's prefix once (question cache) instead of once per option row, when a
      call has at least QCACHE_MIN_ROWS option rows for this backbone (below that the extra question pass costs more
      than it saves); OPENDECIDER_QCACHE=0 turns it off, OPENDECIDER_QCACHE_MIN_ROWS overrides the threshold;
    - int8 decoder layers on tuned fused GEMMs when this GPU has been tuned (quant.fast_int8_kernels_);
    - VeRA adapters left out when their updates are below bf16 resolution (VeRA.relative_size < VERA_INERT; the 9B:
      RESULTS 20.2); OPENDECIDER_VERA=1 keeps them, =0 always drops them. Not exact: a change below the rounding of
      the activations it is added to."""
    from .quant import fast_int8_kernels_
    cfg = model.cfg
    vera = getattr(model, "vera", None)
    mode = os.environ.get("OPENDECIDER_VERA", "auto")
    if vera is not None and mode != "1":
        r = vera.relative_size()
        model.vera_relative_size = r
        if mode == "0" or r < VERA_INERT:
            vera.remove()
            del model.vera
            model.vera_dropped = True
    if os.environ.get("OPENDECIDER_QCACHE", "1") != "0" and getattr(cfg, "v3_features", None) == "branched" \
            and getattr(cfg, "v3_row_format", None) == "answer":
        cfg.v3_question_cache = True
        default = QCACHE_MIN_ROWS.get(model.backbone.hidden_size, 32)
        cfg.v3_qcache_min_rows = int(os.environ.get("OPENDECIDER_QCACHE_MIN_ROWS", default))
    fast_int8_kernels_(model.backbone)


class Decider:
    def __init__(self, model: DecisionModel, temperature: float = 1.0, temperature_by_type: dict | None = None,
                 profiles_dir: str = "runs/profiles", conformal: dict | None = None):
        self.model = model.eval()
        if os.environ.get("OPENDECIDER_CUDA_GRAPHS") == "1" and torch.cuda.is_available() and \
                getattr(model, "state_free", False):
            from .deploy import GraphEngine
            self.engine = GraphEngine(model).install()
        self.temperature = temperature
        self.temperature_by_type = temperature_by_type or {}
        self.conformal = conformal or {}          # {method: {question type: sorted calibration scores}}
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

    def _expand(self, name: str, spec, answers: dict, tag: str) -> tuple[list[Question], list]:
        """Questions for one top-level question in its wave. Chained questions in marginal mode get one copy per
        joint assignment of the referenced answers (chain_branches). -> (questions, branches), branches =
        [(given or None, weight, tag of that branch)]."""
        refs = sorted(refs_of(spec))
        if refs and spec.chain == "marginal":
            br = [(g, w, f"{tag}\x1e{k}") for k, (g, w) in enumerate(chain_branches(refs, answers))]
        else:
            br = [(None, 1.0, tag)]
        qs = []
        for g, _, t in br:
            ans = answers if g is None else {**answers, **{r: {"value": v} for r, v in g.items()}}
            if spec.type == "composite":
                qs += [self._atomic(f"{t}.{pn}", ps, ans) for pn, ps in spec.parts.items()]
            else:
                qs.append(self._atomic(t, spec, ans))
        return qs, br

    def _collect(self, spec, branches, got: dict, conformal=None, qmap=None) -> dict:
        """The answer of one top-level question from the batch results of its branches."""
        outs = [self._combine(spec, {pn: got[f"{t}.{pn}"] for pn in spec.parts}) if spec.type == "composite"
                else got[t] for _, _, t in branches]
        if branches[0][0] is None:
            return outs[0]
        a = mix_answers(outs, [w for _, w, _ in branches], [g for g, _, _ in branches])
        if conformal is not None and spec.type != "composite" and qmap is not None:
            from .conformal import prediction_set, threshold
            q = qmap[branches[0][2]]
            thr = threshold(self._conformal_scores(q, conformal.method), conformal.alpha)
            names = list(a["probs"])
            a["set"] = [names[j] for j in prediction_set(a["probs_list"], thr, conformal.method)]
            a["abstain"] = len(a["set"]) != 1 or a["value"] == "none"
        return a

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
            chunk = Question(q.prompt, q.options[s:s + MAX_OPTIONS], q.type, name=q.name, state_idx=q.state_idx)
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

    def _conformal_scores(self, q: Question, method: str):
        prof = self.profile(getattr(q, "calibration", None))
        src = prof.get("conformal") if prof else self.conformal
        s = (src or {}).get(method, {}).get(q.type)
        if not s:
            where = f"profile {q.calibration!r}" if prof else "this checkpoint"
            raise ValueError(f"no conformal calibration scores for {q.type} questions in {where}")
        return s

    def _answer_batch(self, qs: list[Question], mem, sampler, conformal=None) -> tuple[dict, int]:
        """Answer isolated questions in one batch (<=255 options) plus the two-stage path (>255)."""
        small = [q for q in qs if len(q.options) <= MAX_OPTIONS]
        large = [q for q in qs if len(q.options) > MAX_OPTIONS]
        answers, n_tok = {}, 0
        for q in large:                                     # two-stage (O2)
            keep = two_stage_select(self._relevance(q, mem)).tolist()
            sub = Question(q.prompt, [q.options[i] for i in keep], q.type, name=q.name, state_idx=q.state_idx)
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
                if conformal is not None:
                    from .conformal import prediction_set, threshold
                    thr = threshold(self._conformal_scores(q, conformal.method), conformal.alpha)
                    a["set"] = [names[j] for j in prediction_set(probs, thr, conformal.method)]
                    a["abstain"] = len(a["set"]) != 1 or a["value"] == "none"
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
            # state-token features are only read by the trunk when it cross-attends to the state; with
            # v3_cross='question' the row pass needs the LLM cache alone, which is what gets cached
            if mb > 0 and getattr(self.model, "state_free", False) and \
                    not getattr(self.model.backbone, "is_llamacpp", False):     # llama.cpp keeps its own state
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
            batch, branches = [], {}
            for name in wave:
                qs, branches[name] = self._expand(name, req.questions[name], answers, name)
                batch += qs
            got, nt = self._answer_batch(batch, mem, sampler, req.conformal)
            n_tok += nt
            qmap = {q.name: q for q in batch}
            for name in wave:
                answers[name] = self._collect(req.questions[name], branches[name], got, req.conformal, qmap)
        sync(); t_end = time.perf_counter()
        return {"answers": {k: answers[k] for k in req.questions},
                "timing_ms": {"state": 1e3 * (t_state - t0), "questions": 1e3 * (t_end - t_state),
                              "total": 1e3 * (t_end - t0), "waves": len(plan)},
                "state_cached_tokens": n_cached,
                "input_tokens": mem.n_tokens + n_tok}

    def decide_many(self, reqs: list) -> list[dict]:
        """Several independent requests in one batch: ONE left-padded state pass over all their states (the
        cross-request prefix cache is not used here) and, per dependency wave, ONE row pass over all their
        questions, each question reading its own request's state (state_idx). Requests must share the sampler and
        conformal settings (the server groups them); each still gets only its own answers. Equals decide() per
        request up to batched-kernel rounding."""
        reqs = [DecideRequest.model_validate(r) if isinstance(r, dict) else r for r in reqs]
        if len(reqs) == 1:
            return [self.decide(reqs[0])]
        key = {(r.sampler.mode, r.sampler.k, None if r.conformal is None else (r.conformal.alpha, r.conformal.method))
               for r in reqs}
        if len(key) > 1:
            raise ValueError("decide_many: requests must share sampler and conformal settings")
        t0 = time.perf_counter()
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        r0 = reqs[0]
        sampler = SamplerConfig(mode=r0.sampler.mode if r0.sampler.k > 1 else "off", k=r0.sampler.k)
        texts = [self._context_prefix(r) + state_texts([r.state])[0] for r in reqs]
        mem = self.model.encode_states(texts)
        n_state = [len(t) for t in self.model.backbone.tokenize(texts)]
        sync(); t_state = time.perf_counter()
        answers = [{} for _ in reqs]
        n_tok = [0] * len(reqs)
        plans = [waves(r.questions) for r in reqs]
        for w in range(max(map(len, plans))):
            batch, owner, branches = [], [], {}
            for i, (r, plan) in enumerate(zip(reqs, plans)):
                for name in (plan[w] if w < len(plan) else []):
                    qs, branches[i, name] = self._expand(name, r.questions[name], answers[i], f"{i}\x1f{name}")
                    for q in qs:
                        q.state_idx = i
                        batch.append(q)
                        owner.append(i)
            if not batch:
                continue
            got, _ = self._answer_batch(batch, mem, sampler, r0.conformal)
            for q, i in zip(batch, owner):
                n_tok[i] += sum(len(t) for t in self.model.backbone.tokenize([q.prompt] + q.options))
            qmap = {q.name: q for q in batch}
            for i, (r, plan) in enumerate(zip(reqs, plans)):
                for name in (plan[w] if w < len(plan) else []):
                    answers[i][name] = self._collect(r.questions[name], branches[i, name], got, r0.conformal, qmap)
        sync(); t_end = time.perf_counter()
        timing = {"state": 1e3 * (t_state - t0), "questions": 1e3 * (t_end - t_state), "total": 1e3 * (t_end - t0),
                  "waves": max(map(len, plans)), "batch": len(reqs)}
        return [{"answers": {k: answers[i][k] for k in r.questions}, "timing_ms": dict(timing),
                 "state_cached_tokens": 0, "input_tokens": n_state[i] + n_tok[i]} for i, r in enumerate(reqs)]


def composite_feature(f: str, parts: dict) -> float:
    """Feature value for a combine weight key: "part" (noul P(yes)), "part=option", "part.expected"."""
    if "=" in f:
        part, opt = f.split("=", 1)
        return float(parts[part]["probs"].get(opt, 0.0))
    if f.endswith(".expected"):
        return float(parts[f[: -len(".expected")]]["expected_index"])
    a = parts[f]
    return float(a["p_yes"]) if "p_yes" in a else float(max(a["probs"].values()))
