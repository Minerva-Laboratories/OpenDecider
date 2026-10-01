"""Stage 1: supervised proper-scoring training (CE + lambda*Brier [+ ordinal] [+ stage-1 relevance]).

    python -m opendecider.train --config configs/train_v2.yaml [--set train.max_steps=200 ...]

Backbone frozen (unless a config enables LoRA - not implemented yet). Logs config, seed and
pinned revisions with every run. Takes the shared GPU lock; refuses to write checkpoints if
that would leave < 4 GB free.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import torch
import yaml

from .backbone import Backbone, BackboneConfig, StateMemory, checkpoint_backbone_layers
from .batching import Question, state_texts
from .formatting import answer_row_prefix, record_state_text
from .cache import TokenCache, segment_layout
from .checkpoint import save
from .guards import gpu_lock, limit_gpu_memory, require_free_gb
from .losses import choice_loss, ordinal_emd_loss, relevance_loss
from .models import ModelConfig, build_model
from .options import MAX_OPTIONS

DEFAULT_TRAIN = dict(
    data="data/synthetic/train.jsonl", val="data/synthetic/val.jsonl", calib="data/synthetic/calib.jsonl",
    out_dir="runs", name=None, seed=0, max_steps=2000, lr=3e-4, weight_decay=0.01, warmup=100,
    states_per_step=8, max_questions_per_state=6, max_state_tokens=768, grad_clip=1.0,
    brier_weight=1.0, ordinal_weight=0.25, relevance_weight=0.1, shuffle_options=True,
    eval_every=250, eval_states=200, save_every=1000, noise_std_train=0.0,
    max_gpu_gb=24, tokens_per_microbatch=3000, backbone_checkpointing=True,
    backbone_weight_quant="none",
    autocast_bf16=True,             # trainable stack in bf16 autocast (losses stay fp32)
    cache_state_features=True,      # frozen backbone => encode each training state once (int8, in RAM)
    families=None,                  # optional list: train/val/calib only on these task families (diagnostics)
    option_wrap_p=0.0,              # prob. a question gets meaning-preserving option rewordings (verbosity-bias fix)
    data_weights=None,              # optional per-file sampling weights aligned with `data` (else uniform over records)
    consistency_weight=0.0,         # batch-invariance KL for comparative multi-item states
    max_train_options=0,            # >0: training-only candidate sampling (keep the answer + random distractors)
    grad_accum=1,                   # optimizer step every `grad_accum` batches of `states_per_step` (grads averaged)
    init_from=None,                 # warm start: load every trunk/head tensor whose name and shape match
    none_p=0.0,                     # prob. a question gets the explicit "none of the above" option (as at inference)
    none_drop_p=0.0,                # given `none`, prob. the gold option is removed and `none` becomes the target
    conformal=True)                 # store split-conformal scores of the calib split in the checkpoint

# Training-only wrappers. The probe battery's long-wording templates ("the answer is X", "the customer's request is
# about X") are deliberately NOT in this pool, so probe results measure invariance rather than memorised templates.
OPTION_WRAPS = ["It is {o}", "My choice: {o}", "{o}, according to the text", "The option {o}", "I would say {o}",
                "Clearly {o}", "{o} (this one)", "Answer: {o}", "Most likely {o}", "Going with {o}"]


def load_jsonl(path, limit=None):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


_WRAP_P: dict = {}      # set from the train config in main(); read by to_questions (training rng only)
NONE_TEXT = "none of the above"     # same default text as Decider.none_text


def to_questions(rec, state_idx, rng: random.Random | None, max_q: int):
    qs = []
    items = rec["questions"]
    if rng is not None and len(items) > max_q:
        items = rng.sample(items, max_q)
    wrap_p = _WRAP_P.get("p", 0.0)
    for q in items[:max_q]:
        opts, lab = list(q["options"]), q["label"]
        if len(opts) > MAX_OPTIONS:
            continue
        if rng is not None and wrap_p and q["type"] == "choice" and not rec.get("examples") and rng.random() < wrap_p:
            # reword ALL options (length change, no relative signal) or a random SUBSET (forces invariance to
            # one option being wordier than the others). Labels are unchanged.
            k = len(opts) if rng.random() < 0.3 else rng.randint(1, max(1, len(opts) - 1))
            for i in rng.sample(range(len(opts)), k):
                opts[i] = rng.choice(OPTION_WRAPS).format(o=opts[i])
        soft0 = q.get("soft")
        cap = _WRAP_P.get("max_opts", 0)
        if rng is not None and cap and q["type"] == "choice" and len(opts) > cap:
            # candidate sampling: the correct option + random distractors (branched rows cost memory per option)
            keep = sorted([lab] + rng.sample([i for i in range(len(opts)) if i != lab], cap - 1))
            if soft0 is not None:
                sub = [soft0[i] for i in keep]
                tot = sum(sub) or 1.0
                soft0 = [x / tot for x in sub]
            opts, lab = [opts[i] for i in keep], keep.index(lab)
        perm = None
        if rng is not None and q["type"] == "choice":        # re-shuffle every time it is seen (§5.1)
            perm = list(range(len(opts)))
            rng.shuffle(perm)
            opts, lab = [opts[i] for i in perm], perm.index(lab)
        soft = None
        if soft0 is not None:                                # soft target (e.g. uniform for unknowable items)
            soft = [soft0[i] for i in perm] if perm is not None else list(soft0)
        has_none = False
        if (rng.random() < _WRAP_P.get("none_p", 0.0) if rng is not None else _WRAP_P.get("eval_none", False)) \
                and len(opts) < MAX_OPTIONS and not rec.get("examples"):
            # explicit `none` option, appended last as at inference. Programmatic targets, no teacher:
            #   unknowable item (uniform soft target) -> none; gold removed -> none; otherwise the gold stays
            uniform = soft is not None and max(soft) - min(soft) < 1e-9
            if uniform:
                soft, lab = None, len(opts)
            elif rng is not None and q["type"] == "choice" and soft is None and len(opts) >= 3 \
                    and rng.random() < _WRAP_P.get("none_drop_p", 0.0):
                opts, lab = [o for i, o in enumerate(opts) if i != lab], len(opts) - 1
            elif soft is not None:
                soft = soft + [0.0]
            opts = opts + [NONE_TEXT]
            has_none = True
        qq = Question(q["prompt"], opts, q["type"], lab, state_idx, family=rec.get("family", ""))
        if has_none:
            qq.none_col = len(opts) - 1
        if soft is not None:
            qq.soft = soft
        qs.append(qq)
    return qs


def question_costs(model, qs) -> tuple[list[int], list[int]]:
    """(rows, row_len) per question from ONE batched tokenizer call. Cost of a group = rows x max row_len."""
    texts, n = [], []
    cfg = getattr(model, "cfg", None)
    answer_rows = getattr(cfg, "v3_row_format", "list") == "answer"
    for q in qs:
        # answer-format branched rows repeat the (sorted) candidate list in EVERY row: measure the real prefix
        texts.append(answer_row_prefix(q.prompt, q.options, cfg.v3_list_cap) if answer_rows else q.prompt)
        texts.extend(q.options)
        n.append(len(q.options))
    lens = torch.tensor([len(t) for t in model.backbone.tokenize(texts)])
    n = torch.tensor(n)
    first = torch.cumsum(n + 1, 0) - (n + 1)                          # index of each prompt
    qlen = lens[first]
    opt = torch.ones(len(lens), dtype=torch.bool); opt[first] = False
    owner = torch.repeat_interleave(torch.arange(len(qs)), n)
    olen = lens[opt]
    tot = torch.zeros(len(qs), dtype=torch.long).index_add_(0, owner, olen)
    omax = torch.zeros(len(qs), dtype=torch.long).scatter_reduce_(0, owner, olen, "amax", include_self=True)
    if model.pack_mode == "independent":
        if getattr(cfg, "v3_question_cache", False):
            # question (+ full listing) encoded ONCE with its own graph, then n option rows: count the whole
            # question as one unit of qlen + n * omax tokens (the listing pass costs as much as a state of that length)
            return [1] * len(qs), (qlen + n * omax).tolist(), tot.tolist()
        return n.tolist(), (qlen + omax).tolist(), tot.tolist()
    return [1] * len(qs), (qlen + tot).tolist(), tot.tolist()


def microbatches(model, qs, budget, attn_budget: float = 1.2e7):
    """Greedy length-sorted groups under two budgets:
    - backbone: padded tokens (rows x max row length) <= budget
    - trunk (V3): across-option attention elements (questions x max option-tokens^2) <= attn_budget,
      since V3 attends over ALL option tokens of a question (a 250-option question is ~1.5k tokens)."""
    rows, L, P = question_costs(model, qs)
    trunk = model.cfg.variant == "v3"
    order = sorted(range(len(qs)), key=lambda i: (P[i], rows[i] * L[i]))
    groups, cur, cur_rows, cur_len, cur_p = [], [], 0, 0, 0
    for i in order:
        over_tok = (cur_rows + rows[i]) * max(cur_len, L[i]) > budget
        over_att = trunk and (len(cur) + 1) * max(cur_p, P[i]) ** 2 > attn_budget
        if cur and (over_tok or over_att):
            groups.append(cur); cur, cur_rows, cur_len, cur_p = [], 0, 0, 0
        cur.append(qs[i]); cur_rows += rows[i]; cur_len = max(cur_len, L[i]); cur_p = max(cur_p, P[i])
    if cur:
        groups.append(cur)
    return groups


_STATE_CACHE: dict = {}


def state_features(bb, recs, seqs, ids, mask, tcfg):
    """Frozen-backbone state features (S,T,d). With cache_state_features each state is encoded once
    per run and kept int8 in one flat buffer (same per-token absmax as the inference cache); a batch
    is assembled with one gather + one scatter. Pure function of the frozen backbone."""
    if not tcfg.get("cache_state_features"):
        with torch.no_grad():
            return bb.encode_state_grad(ids, mask)
    cache = _STATE_CACHE.get("c")
    if cache is None or cache.device != ids.device:
        cache = _STATE_CACHE["c"] = TokenCache(bb.hidden_size, ids.device)
    keys = [r.get("id") or hash(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])) for r in recs]
    miss = [i for i, k in enumerate(keys) if k not in cache]
    if miss:
        mids, mmask = bb.pad([seqs[i] for i in miss])
        with torch.no_grad():
            f = bb.encode_state_grad(mids, mmask)
        cache.add([keys[i] for i in miss], f, mmask, [len(seqs[i]) for i in miss])
    flat, lens = cache.lookup(keys)
    row, col, _, _, _ = segment_layout(lens, torch.arange(len(keys)), len(keys))
    out = torch.zeros(len(keys), ids.shape[1], bb.hidden_size, dtype=bb.dtype, device=ids.device)
    out[(row.to(ids.device), col.to(ids.device))] = flat.to(bb.dtype)
    return out


_MEMDEBUG = bool(os.environ.get("OPENDECIDER_MEMDEBUG"))


def record_tokens(model, recs, max_tokens):
    """Token ids of each record's full state text (schema examples + state). Plain states keep their HEAD; records
    with examples keep their TAIL, so the current case is never cut and only the oldest examples are dropped."""
    fmt, cap = getattr(model, "row_format", "list"), getattr(getattr(model, "cfg", None), "v3_list_cap", 16)
    toks = model.backbone.tokenize([record_state_text(r, fmt, cap) for r in recs])
    return [t[-max_tokens:] if r.get("examples") else t[:max_tokens] for t, r in zip(toks, recs)]


def step_loss(model, recs, rng, tcfg, noise_std=0.0, backward=True):
    """One optimisation step's loss; backward is done per token-budgeted micro-batch
    (gradient accumulation), so a few 255-option questions cannot blow up activation memory."""
    # training-time augmentation settings come from the config of THIS call (not only from main())
    _WRAP_P["p"] = float(tcfg.get("option_wrap_p") or 0.0)
    _WRAP_P["max_opts"] = int(tcfg.get("max_train_options") or 0)
    bb = model.backbone
    seqs = record_tokens(model, recs, tcfg["max_state_tokens"])
    ids, mask = bb.pad(seqs)
    feats = None
    needs_prefix = model.cfg.variant == "v0" or getattr(model, "conditioned", False)
    if not needs_prefix:
        feats = state_features(bb, recs, seqs, ids, mask, tcfg)     # frozen: reused per micro-batch
    qs = []
    for i, r in enumerate(recs):
        qs.extend(to_questions(r, i, rng, tcfg["max_questions_per_state"]))
    tot = {"loss": 0.0, "ce": 0.0, "brier": 0.0, "acc": 0.0}
    n = len(qs)
    n_score = max(1, sum(q.type == "score" for q in qs))
    v0_mem = model.memory_from_ids(ids, mask) if needs_prefix else None   # state prefixes shared by all groups
    groups = microbatches(model, qs, tcfg["tokens_per_microbatch"])
    shared_graph = getattr(getattr(model, "cfg", None), "v3_vera_scope", "") == "all"   # state pass has a graph
    roots = leaves = None
    if shared_graph and backward and v0_mem is not None and tcfg.get("state_backward_once", True) \
            and hasattr(model, "_state_prefix") and model.training and torch.is_grad_enabled():
        # The state pass is shared by every micro-batch. Run it WITHOUT a graph (exact, unquantised values) and hand
        # the micro-batches leaf copies: each backprops only to the leaves (gradients add up there). At the end the
        # state pass is re-run WITH its graph and backpropagated ONCE from the summed leaf gradients (a manual
        # checkpoint), so no micro-batch ever shares memory with the state-pass graph.
        with torch.no_grad():
            pc, smask_, sh = model._state_prefix(v0_mem, grad=False, quantize=False)
        leaves = {}
        for key, t in list(pc.tensors.items()):
            if torch.is_tensor(t):
                pc.tensors[key] = leaves[key] = t.detach().requires_grad_()
        if sh is not None:                                  # None on state-free models (v3_cross='question')
            sh = leaves["hidden"] = sh.detach().requires_grad_()
        v0_mem.prefix_batch = (pc, smask_, sh)
        roots = True
    for gi, group in enumerate(groups):
        if v0_mem is not None:
            mem = v0_mem
        else:
            sm = StateMemory(feats, mask, sum(map(len, seqs)))
            sm.lens = torch.tensor([len(x) for x in seqs])
            mem = model.prepare_memory(sm, cache=False, noise_std=noise_std)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.get("autocast_bf16", False)
                            and torch.cuda.is_available()):
            _, out = model.run(group, mem)
        labels = torch.tensor([q.label for q in group]).to(out.logits.device, non_blocking=True)
        is_soft = torch.tensor([getattr(q, "soft", None) is not None for q in group], device=labels.device)
        if is_soft.any():
            K = out.logits.shape[1]
            tgt = torch.zeros(len(group), K, device=labels.device)
            for i, q in enumerate(group):
                if getattr(q, "soft", None) is not None:
                    tgt[i, : len(q.soft)] = torch.tensor(q.soft, device=labels.device)
                else:
                    tgt[i, q.label] = 1.0
            d = choice_loss(out.logits.float(), out.opt_mask, tgt, tcfg["brier_weight"])
        else:
            d = choice_loss(out.logits.float(), out.opt_mask, labels, tcfg["brier_weight"])
        w = len(group) / n                     # every term is a per-question mean -> exact accumulation
        loss = d["loss"] * w
        hard = ~is_soft
        if tcfg["relevance_weight"] and hard.any():
            loss = loss + tcfg["relevance_weight"] * w * (hard.sum() / len(group)) * relevance_loss(
                out.relevance[hard], out.opt_mask[hard], labels[hard])
        # ordinal loss over the real levels only: the `none` column is not a level, and `none` targets are skipped
        ncol = torch.tensor([getattr(q, "none_col", -1) for q in group], device=labels.device)
        score = torch.tensor([q.type == "score" for q in group], device=labels.device) & hard & (labels != ncol)
        if tcfg["ordinal_weight"] and score.any():
            om = out.opt_mask[score].clone()
            nc = ncol[score]
            om[(nc >= 0).nonzero().squeeze(1), nc[nc >= 0]] = False
            loss = loss + tcfg["ordinal_weight"] * (int(score.sum()) / n_score) * ordinal_emd_loss(
                out.logits[score], om, labels[score])
        if _MEMDEBUG:
            print(f"  [mem] group {gi}/{len(groups)} q={len(group)} fwd alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        if backward:
            loss.backward(retain_graph=shared_graph and roots is None and gi < len(groups) - 1)
        tot["loss"] += loss.item(); tot["ce"] += d["ce"].item() * w; tot["brier"] += d["brier"].item() * w
        tot["acc"] += (out.logits.argmax(-1) == labels).float().sum().item() / n
    if roots:
        v0_mem.prefix_batch = None
        if any(l.grad is not None for l in leaves.values()):
            pc2, _, sh2 = model._state_prefix(v0_mem, grad=True)                 # same pass, now with its graph
            outs = {**pc2.tensors, **({"hidden": sh2} if sh2 is not None else {})}
            pairs = [(outs[k], l.grad) for k, l in leaves.items() if l.grad is not None and outs[k].requires_grad]
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
            del pc2, sh2, outs, pairs
        del leaves
    if backward and tcfg.get("consistency_weight") and model.cfg.variant == "v3":
        tot["consistency"] = consistency_step(model, recs, rng, tcfg)
        if _MEMDEBUG:
            print(f"  [mem] after consistency alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    tot["n_q"] = n
    return tot


_ITEM_REF = re.compile(r"^Regarding (item_\d+): ")


def consistency_step(model, recs, rng, tcfg, max_pairs: int = 2) -> float:
    """Batch-invariance for i.i.d. items: KL(p_alone || p_in_batch) with p_alone held fixed. The batch may serve as a
    reference scale, but should not move a calibrated answer for an independent item."""
    pairs = []
    for r in recs:
        if r.get("context") == "schema_batch" and r.get("examples"):
            q = rng.choice(r["questions"])
            pairs.append((r, {"state": r["state"], "questions": [q]}, q))
            continue
        items = r["state"].get("items") if isinstance(r["state"], dict) else None
        if not items or len(items) < 2 or r.get("context") != "batch":
            continue
        q = rng.choice(r["questions"])
        m = _ITEM_REF.match(q["prompt"])
        if m and m.group(1) in items:
            pairs.append(({"state": r["state"], "questions": [q]}, {"state": {"items": {m.group(1): items[m.group(1)]}},
                                                                    "questions": [q]}, q))
    if not pairs:
        return 0.0
    pairs = rng.sample(pairs, min(max_pairs, len(pairs)))
    bb = model.backbone
    seqs = record_tokens(model, [x for b, a, _ in pairs for x in (b, a)], tcfg["max_state_tokens"])
    ids, mask = bb.pad(seqs)
    mem = model.memory_from_ids(ids, mask)
    qs = []
    cap = int(tcfg.get("max_train_options") or 0)
    for i, (_, _, q) in enumerate(pairs):
        opts, lab = list(q["options"]), q["label"]
        if cap and len(opts) > cap:          # same candidate subset for both members of the pair (bounded rows)
            keep = sorted([lab] + rng.sample([j for j in range(len(opts)) if j != lab], cap - 1))
            opts, lab = [opts[j] for j in keep], keep.index(lab)
        qs.append(Question(q["prompt"], opts, q["type"], lab, 2 * i))
        qs.append(Question(q["prompt"], opts, q["type"], lab, 2 * i + 1))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.get("autocast_bf16", False) and torch.cuda.is_available()):
        _, out = model.run(qs, mem)
    lp = torch.log_softmax(out.logits.float().masked_fill(~out.opt_mask, -1e9), -1)
    lp_b, lp_a = lp[0::2], lp[1::2].detach()
    m = out.opt_mask[0::2]
    kl = ((lp_a.exp() * (lp_a - lp_b)) * m).sum(-1).mean()
    (tcfg["consistency_weight"] * kl).backward()
    return float(kl)


@torch.no_grad()
def evaluate(model, recs, tcfg, return_logits=False, return_types=False, return_none=False):
    model.eval()
    ce = br = acc = n = 0.0
    all_logits, all_labels, all_types, all_none = [], [], [], []
    for i in range(0, len(recs), tcfg["states_per_step"]):
        chunk = recs[i:i + tcfg["states_per_step"]]
        bb = model.backbone
        seqs = record_tokens(model, chunk, tcfg["max_state_tokens"])
        ids, mask = bb.pad(seqs)
        mem = model.memory_from_ids(ids, mask)
        qs = [q for j, r in enumerate(chunk) for q in to_questions(r, j, None, 10**6)]
        for group in microbatches(model, qs, 4 * tcfg["tokens_per_microbatch"]):
            _, out = model.run(group, mem)
            labels = torch.tensor([q.label for q in group], device=out.logits.device)
            d = choice_loss(out.logits, out.opt_mask, labels)
            k = len(group)
            ce += d["ce"].item() * k; br += d["brier"].item() * k; n += k
            acc += (out.logits.argmax(-1) == labels).float().sum().item()
            if return_logits:
                for j in range(k):
                    all_logits.append(out.logits[j, out.opt_mask[j]].float().cpu())
                all_labels.extend(labels.tolist())
                all_types.extend(q.type for q in group)
                all_none.extend(hasattr(q, "none_col") for q in group)
    model.train()
    res = {"val_ce": ce / n, "val_brier": br / n, "val_acc": acc / n, "val_n": int(n)}
    if return_logits:
        if return_none:
            return res, all_logits, all_labels, all_types, all_none
        return (res, all_logits, all_labels, all_types) if return_types else (res, all_logits, all_labels)
    return res


def warm_start(model, path: str) -> dict:
    """Copy every non-backbone tensor of a checkpoint whose name and shape match; new modules keep their init
    (e.g. VeRA vectors start with zero update). Returns what was loaded and what was not."""
    src = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    own = model.state_dict()
    load = {k: v for k, v in src.items() if not k.startswith("backbone.") and k in own and own[k].shape == v.shape}
    model.load_state_dict(load, strict=False)
    info = {"path": path, "loaded": len(load),
            "skipped": sorted(k for k in src if not k.startswith("backbone.") and k not in load),
            "new": sorted(k for k in own if not k.startswith("backbone.") and k not in load)}
    print(f"[train] warm start from {path}: {info['loaded']} tensors loaded, {len(info['skipped'])} skipped, "
          f"{len(info['new'])} new", flush=True)
    return info


def fit_temperature(logits, labels) -> float:
    """Single temperature minimising NLL on the calibration split (§5.4)."""
    logT = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    lab = torch.tensor(labels)

    def closure():
        opt.zero_grad()
        T = logT.exp()
        l = torch.stack([torch.nn.functional.cross_entropy((z / T)[None], lab[i:i + 1])
                         for i, z in enumerate(logits)]).mean()
        l.backward()
        return l
    opt.step(closure)
    return float(logT.exp())


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def load_config(path, sets=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bcfg = cfg.get("backbone", "configs/backbone.yaml")
    if isinstance(bcfg, str):
        with open(bcfg) as f:
            bcfg = yaml.safe_load(f)
    cfg["backbone"] = bcfg
    cfg["train"] = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    for k, v in parse_sets(sets).items():
        sec, key = k.split(".", 1)
        cfg[sec][key] = v
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="section.key=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    t = cfg["train"]
    name = t["name"] or f"{cfg['model']['variant']}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = os.path.join(t["out_dir"], name)
    require_free_gb(0.5)
    os.makedirs(out, exist_ok=True)
    random.seed(t["seed"]); torch.manual_seed(t["seed"])
    with gpu_lock(f"opendecider-train-{name}"):
        limit_gpu_memory(t["max_gpu_gb"])
        bcfg = BackboneConfig(**cfg["backbone"])
        deploy_quant = bcfg.weight_quant
        if t["backbone_weight_quant"] is not None:
            bcfg.weight_quant = t["backbone_weight_quant"]
        bb = Backbone.load(bcfg)
        bb.cfg.weight_quant = deploy_quant       # checkpoint records the deployment quantisation
        if t["backbone_checkpointing"]:
            checkpoint_backbone_layers(bb)
        model = build_model(bb, ModelConfig(**cfg["model"])).to(bb.device)
        init = warm_start(model, t["init_from"]) if t.get("init_from") else None
        n_train = sum(p.numel() for p in model.trainable_parameters())
        meta = {"config": cfg, "git": git_rev(), "trainable_params": n_train, "torch": torch.__version__, "init": init,
                "argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)
        print(f"[train] {name}: {n_train/1e6:.2f}M trainable params", flush=True)

        fam = set(t["families"]) if t["families"] else None
        keep = (lambda rs: [r for r in rs if r.get("family") in fam]) if fam else (lambda rs: rs)
        many = lambda x: [r for f in ([x] if isinstance(x, str) else x) for r in load_jsonl(f)]
        _WRAP_P["p"] = float(t.get("option_wrap_p") or 0.0)
        _WRAP_P["max_opts"] = int(t.get("max_train_options") or 0)
        _WRAP_P["none_p"], _WRAP_P["none_drop_p"] = float(t.get("none_p") or 0.0), float(t.get("none_drop_p") or 0.0)
        _WRAP_P["eval_none"] = _WRAP_P["none_p"] > 0          # val/calib questions carry `none` as at inference
        files = [t["data"]] if isinstance(t["data"], str) else list(t["data"])
        pools = [keep(load_jsonl(f)) for f in files]
        train = [r for pool in pools for r in pool]
        weights = t.get("data_weights")
        if weights:
            assert len(weights) == len(files), "data_weights must align with data files"
            print("[train] per-file sampling weights: " + ", ".join(f"{os.path.basename(f)}={w}" for f, w in zip(files, weights)), flush=True)
        vals = [t["val"]] if isinstance(t["val"], str) else (t["val"] or [])
        per = max(1, t["eval_states"] // max(1, len(vals)))              # equal share per validation file
        val = [r for f in vals if os.path.exists(f) for r in keep(load_jsonl(f))[:per]]
        print(f"[train] {len(train)} training states" + (f" (families {sorted(fam)})" if fam else ""), flush=True)
        params = model.trainable_parameters()
        opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"],
                                fused=torch.cuda.is_available())    # no per-param .item() syncs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup"]) *
                                                  0.5 * (1 + math.cos(math.pi * min(1, s / t["max_steps"]))))
        rng = random.Random(t["seed"])
        log = open(os.path.join(out, "log.jsonl"), "a")
        model.train()
        t0 = time.time()
        for step in range(1, t["max_steps"] + 1):
            opt.zero_grad(set_to_none=True)
            k = max(1, int(t.get("grad_accum") or 1))
            infos = []
            for _ in range(k):
                if weights:                                # choose a file by weight, then a record uniformly
                    recs = [rng.choice(pools[i]) for i in rng.choices(range(len(pools)), weights=weights, k=t["states_per_step"])]
                else:
                    recs = rng.sample(train, t["states_per_step"])
                infos.append(step_loss(model, recs, rng, t, t["noise_std_train"]))
            info = {key: (sum(x[key] for x in infos) / k if isinstance(infos[0][key], float) else infos[-1][key])
                    for key in infos[0]}
            if k > 1:
                torch._foreach_div_([p.grad for p in params if p.grad is not None], float(k))
            gn = torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            opt.step(); sched.step()
            info.update(step=step, gn=float(gn), lr=sched.get_last_lr()[0],
                        s_per_step=(time.time() - t0) / step)
            if step % 10 == 0 or step == 1:
                log.write(json.dumps(info) + "\n"); log.flush()
                print(f"[train] step {step} loss {info['loss']:.4f} ce {info['ce']:.4f} acc {info['acc']:.3f} "
                      f"gn {info['gn']:.2f} {info['s_per_step']:.2f}s/step", flush=True)
            if val and (step % t["eval_every"] == 0 or step == t["max_steps"]):
                ev = evaluate(model, val, t); ev["step"] = step
                log.write(json.dumps(ev) + "\n"); log.flush()
                print(f"[eval] step {step} {ev}", flush=True)
            if step % t["save_every"] == 0 or step == t["max_steps"]:
                save(model, os.path.join(out, "model.pt"), {"step": step})
        extra = {"step": t["max_steps"], "temperature": 1.0}
        calibs = [t["calib"]] if isinstance(t["calib"], str) else (t["calib"] or [])
        calib = [r for f in calibs if os.path.exists(f) for r in keep(load_jsonl(f))[: max(1, t["eval_states"] // max(1, len(calibs)))]]
        if calib:
            _, lg, lb, ty, hn = evaluate(model, calib, t, return_logits=True, return_types=True, return_none=True)
            extra["temperature"] = fit_temperature(lg, lb)
            by = {}
            for typ in sorted(set(ty)):
                idx = [i for i, x in enumerate(ty) if x == typ]
                if len(idx) >= 30:
                    by[typ] = fit_temperature([lg[i] for i in idx], [lb[i] for i in idx])
            extra["temperature_by_type"] = by
            print(f"[calib] temperature {extra['temperature']:.3f} by type {by}", flush=True)
            if t.get("conformal", True):
                # probabilities exactly as the decider reports them: per-type temperature, renormalised over the
                # real options (the `none` column set aside)
                from .conformal import pack
                P = []
                for z, typ, none in zip(lg, ty, hn):
                    p = torch.softmax(z.float() / by.get(typ, extra["temperature"]), -1)
                    p = p[:-1] / p[:-1].sum() if none else p
                    P.append(p.numpy())
                extra["conformal"] = pack(P, lb, ty)
                print("[calib] conformal scores: " + ", ".join(f"{k}={len(v)}" for k, v in extra["conformal"]["lac"].items()), flush=True)
        if init:
            extra["init_from"] = init["path"]
        save(model, os.path.join(out, "model.pt"), extra)
        print(f"[train] done -> {out}/model.pt", flush=True)


if __name__ == "__main__":
    main()
