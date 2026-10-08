"""Phase 2 of docs/online_projection_spec.md: the compressed cache inside the model, measured as next-token NLL.

Qwen3.5-2B, all attention layers, one forward pass per sequence with hooks:
  keys    after k_norm (pre-RoPE): the 192 position-free dims of every KV group are replaced by their rank-r
          reconstruction (the 64 RoPE dims stay exact);
  values  after v_proj: the 256 dims of every KV group are replaced by their rank-r reconstruction.
Reconstructing a key/value with the basis of its chunk is exactly what attention sees when the cache holds the codes
c_j = E_b(j) x_j and queries are back-projected per chunk basis (spec §1, reading option (a)); chunk 0 (warm-up) is
exact. Variants: static (PCA fitted once on held-out mixed code + prose: the MLA-like global projection), online
(chunked subspace iteration, spec §3), qw (keys only: query-weighted, spec §2), oracle (PCA of the sequence itself,
non-causal: an upper bound for any per-sequence basis).

Reported: mean NLL difference vs the uncompressed model on held-out prose and code (paired per token, bootstrap 95%
CI over 512-token blocks, 1,000 resamples), and a topic-switch test (code -> prose and prose -> code at the midpoint:
NLL difference on the 1,024 tokens after the switch, for several forgetting rates gamma).
    PYTHONPATH=src .venv/bin/python scripts/online_projection_phase2.py --out runs/online_projection_phase2.json
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from opendecider.guards import gpu_lock
from online_projection_phase1 import ROT, capture, online_bases, pca_basis, project_chunks

C = 64            # chunk size of the online update


def qw_project(k_nope, q_nope, r, gamma):
    """Query-weighted online projection (as in phase 1): keys whitened by the running query second moment A, subspace
    iteration on A k, oblique reconstruction A^-1 E^T E A k with the basis of the previous chunks; chunk 0 exact."""
    T, d = k_nope.shape
    out = k_nope.clone()
    Mq = torch.zeros(d, d, device=k_nope.device)
    Mk = torch.zeros_like(Mq)
    E = A_prev = Ai_prev = None
    eye = torch.eye(d, device=k_nope.device)
    for b0 in range(0, T, C):
        qb = q_nope[b0:b0 + C].reshape(-1, d)
        Mq = gamma * Mq + (1 - gamma) * qb.T @ qb / qb.shape[0]
        w, U = torch.linalg.eigh(Mq + 1e-4 * Mq.diagonal().mean() * eye)
        A = (U * w.clamp_min(1e-8).sqrt()) @ U.T
        Ai = (U * w.clamp_min(1e-8).rsqrt()) @ U.T
        if E is not None:
            out[b0:b0 + C] = ((k_nope[b0:b0 + C] @ A_prev) @ E.T @ E) @ Ai_prev
        kb = k_nope[b0:b0 + C] @ A
        Mk = gamma * Mk + (1 - gamma) * kb.T @ kb / kb.shape[0]
        if E is None:
            w2, U2 = torch.linalg.eigh(Mk)
            E = U2[:, -r:].flip(-1).T.contiguous()
        else:
            Qm, _ = torch.linalg.qr((E @ Mk).T)
            E = Qm.T.contiguous()
        A_prev, Ai_prev = A, Ai
    return out


def reconstruct(X, r, mode, gamma, static_basis=None, q=None):
    """X (T, d) fp32 -> rank-r reconstruction as attention would read it (see module docstring)."""
    if mode == "static":
        return (X @ static_basis.T) @ static_basis
    if mode == "oracle":
        E = pca_basis(X, r)
        return (X @ E.T) @ E
    if mode == "online":
        return project_chunks(X, online_bases(X, r, C=C, gamma=gamma), C=C)
    if mode == "qw":
        return qw_project(X, q, r, gamma)
    raise ValueError(mode)


class Compressor:
    """Forward hooks on every attention layer. cfg: {"k": (mode, r) | None, "v": (mode, r) | None, "gamma": g}."""

    def __init__(self, model, static):
        self.trunk = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.static, self.cfg, self.q = static, None, {}
        self.hooks = []
        for i, layer in enumerate(self.trunk.layers):
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            self.hooks.append(sa.q_norm.register_forward_hook(lambda m, a, o, i=i: self._q(i, o)))
            self.hooks.append(sa.k_norm.register_forward_hook(lambda m, a, o, i=i: self._k(i, o)))
            self.hooks.append(sa.v_proj.register_forward_hook(lambda m, a, o, i=i, sa=sa: self._v(i, o, sa)))

    def _q(self, i, o):
        if self.cfg and self.cfg.get("k") and self.cfg["k"][0] == "qw":
            self.q[i] = o[0].float()                                     # (T, Hq, D)

    def _k(self, i, o):
        if not self.cfg or not self.cfg.get("k"):
            return None
        mode, r = self.cfg["k"]
        out = o.clone()
        T, Hk, D = o.shape[1:]
        for g in range(Hk):
            x = o[0, :, g, ROT:].float()
            q = None
            if mode == "qw":
                Hq = self.q[i].shape[1]
                rep = Hq // Hk
                q = self.q[i][:, g * rep:(g + 1) * rep, ROT:]
            sb = self.static[i]["k"][g][r] if mode == "static" else None
            out[0, :, g, ROT:] = reconstruct(x, r, mode, self.cfg["gamma"], sb, q).to(o.dtype)
        return out

    def _v(self, i, o, sa):
        if not self.cfg or not self.cfg.get("v"):
            return None
        mode, r = self.cfg["v"]
        D = sa.head_dim
        v = o.view(o.shape[0], o.shape[1], -1, D)
        out = v.clone()
        for g in range(v.shape[2]):
            sb = self.static[i]["v"][g][r] if mode == "static" else None
            out[0, :, g] = reconstruct(v[0, :, g].float(), r, mode, self.cfg["gamma"], sb).to(o.dtype)
        return out.view_as(o)


def token_nll(model, trunk, ids, chunk=1024):
    """Per-token NLL (T-1,) without materializing the full (T, vocab) logits."""
    with torch.no_grad():
        h = trunk(input_ids=ids, use_cache=False).last_hidden_state[0]
        out = []
        for a in range(0, h.shape[0] - 1, chunk):
            b = min(h.shape[0] - 1, a + chunk)
            logits = model.lm_head(h[a:b]).float()
            out.append(torch.nn.functional.cross_entropy(logits, ids[0, a + 1:b + 1], reduction="none"))
        return torch.cat(out)


def boot_ci(d, block=512, n=1000, seed=0):
    """Mean of d with a 95% bootstrap CI over contiguous blocks (tokens within a block are correlated)."""
    nb = d.numel() // block
    blocks = d[:nb * block].view(nb, block).mean(1).cpu()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, nb, (n, nb), generator=g)
    m = blocks[idx].mean(1)
    return [round(d.mean().item(), 4), round(m.quantile(0.025).item(), 4), round(m.quantile(0.975).item(), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/ext/opendecider/models/Qwen__Qwen3.5-2B@15852e8c1636")
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--ranks_k", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--ranks_v", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--switch_gammas", type=float, nargs="+", default=[0.8, 0.9, 0.97])
    ap.add_argument("--out", default="runs/online_projection_phase2.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = os.path.join(os.path.dirname(__file__), "..")
    rd = lambda *p: "".join(open(os.path.join(root, x)).read() for x in p)
    # held-out evaluation text and a disjoint mixed calibration set for the static basis
    texts = {"prose": rd("docs/RESULTS.md"), "code": rd("src/opendecider/v3.py", "src/opendecider/train.py")}
    calib = {"prose": rd("docs/LANDSCAPE.md", "README.md"),
             "code": rd("src/opendecider/backbone.py", "src/opendecider/store.py", "src/opendecider/decider.py")}
    switch = {"code": rd("src/opendecider/models.py", "src/opendecider/quant.py", "src/opendecider/calibrators.py"),
              "prose": rd("docs/roadmap_designs.md", "docs/SPEC.md")}
    res = {"model": a.model, "T": a.T, "gamma": a.gamma, "chunk": C}
    with gpu_lock("online_projection_phase2"):
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
        # static bases: PCA of the pooled calibration activations (prose + code), per layer, group and rank
        caps = [capture(model, tok, calib[k], a.T)[0] for k in ("prose", "code")]
        static = {}
        for li in caps[0]:
            Hk = caps[0][li]["k"].shape[1]
            static[li] = {"k": {}, "v": {}}
            for g in range(Hk):
                kc = torch.cat([c[li]["k"][:, g, ROT:] for c in caps])
                vc = torch.cat([c[li]["v"][:, g] for c in caps])
                static[li]["k"][g] = {r: pca_basis(kc, r) for r in a.ranks_k}
                static[li]["v"][g] = {r: pca_basis(vc, r) for r in a.ranks_v}
        del caps
        torch.cuda.empty_cache()
        comp = Compressor(model, static)
        trunk = comp.trunk

        def nll(ids, cfg):
            comp.cfg = cfg
            try:
                return token_nll(model, trunk, ids)
            finally:
                comp.cfg, comp.q = None, {}

        seqs = {}
        for name, t in texts.items():
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"]
            n = ids.shape[1] // a.T
            seqs[name] = [ids[:, i * a.T:(i + 1) * a.T].cuda() for i in range(n)]
        res["sequences"] = {k: len(v) for k, v in seqs.items()}
        configs = {"baseline": None}
        for r in a.ranks_k:
            for m in ("static", "online", "qw", "oracle"):
                configs[f"k_{m}_r{r}"] = {"k": (m, r), "v": None, "gamma": a.gamma}
        for r in a.ranks_v:
            for m in ("static", "online", "oracle"):
                configs[f"v_{m}_r{r}"] = {"k": None, "v": (m, r), "gamma": a.gamma}
        for m in ("static", "online", "oracle"):
            configs[f"kv_{m}_k64_v128"] = {"k": (m, 64), "v": (m, 128), "gamma": a.gamma}
        configs["kv_qw+online_k64_v128"] = {"k": ("qw", 64), "v": ("online", 128), "gamma": a.gamma}
        t0 = time.time()
        base = {d: [nll(s, None) for s in ss] for d, ss in seqs.items()}
        res["baseline_nll"] = {d: round(torch.cat(v).mean().item(), 4) for d, v in base.items()}
        res["delta_nll"] = {}
        for name, cfg in configs.items():
            if cfg is None:
                continue
            res["delta_nll"][name] = {}
            for d, ss in seqs.items():
                diff = torch.cat([nll(s, cfg) - b for s, b in zip(ss, base[d])])[C:]
                res["delta_nll"][name][d] = boot_ci(diff)
            print(f"{name:26s} {res['delta_nll'][name]}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)

        # topic switch: first half one domain, second half the other; NLL difference on the 1,024 tokens after it
        half = a.T // 2
        ids_sw = {k: tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :half] for k, t in switch.items()}
        res["switch"] = {}
        for order in (("code", "prose"), ("prose", "code")):
            ids = torch.cat([ids_sw[order[0]], ids_sw[order[1]]], 1).cuda()
            b = nll(ids, None)
            key = f"{order[0]}->{order[1]}"
            res["switch"][key] = {}
            cfgs = {"k_static_r64": {"k": ("static", 64), "v": None, "gamma": a.gamma}}
            for g in a.switch_gammas:
                cfgs[f"k_online_r64_g{g}"] = {"k": ("online", 64), "v": None, "gamma": g}
                cfgs[f"v_online_r128_g{g}"] = {"k": None, "v": ("online", 128), "gamma": g}
            for name, cfg in cfgs.items():
                diff = nll(ids, cfg) - b
                after = diff[half - 1:half - 1 + 1024]                  # predictions of tokens half .. half+1023
                before = diff[half - 1 - 1024:half - 1]
                res["switch"][key][name] = {"before": boot_ci(before, block=128), "after": boot_ci(after, block=128),
                                            "late": boot_ci(diff[-1024:], block=128)}
                print(key, name, res["switch"][key][name], flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        for h in comp.hooks:
            h.remove()
    print("done", flush=True)


if __name__ == "__main__":
    main()
