"""Deployed Mercurius-1-4B: weight memory by module kind, and the compressed KV cache (opendecider.mercurius_cache)
against the stock cache on the same model: full pass, cache continuation and a left-padded batch continued by rows,
plus cache bytes per token.
    MERCURIUS_ROOT=... PYTHONPATH=<mercurius>/src:src .venv/bin/python scripts/mercurius_cache_check.py"""
import collections, os, sys, types, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opendecider.guards import gpu_lock
from opendecider import mercurius_port as mp, mercurius_cache as mc

TEXTS = ["State:\nThe customer was charged twice for the March invoice and asks for a refund. They have been a customer "
         "for six years and threaten to leave.\nQuestion: Which team? billing",
         "In linear attention, the delta rule updates a fixed-size state matrix. Cuando el modelo pierde la codificacion."]


def rel(a, b):
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


def run(trunk, ids, mask=None, cache=None, pos=None):
    o = trunk(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True, position_ids=pos)
    return o.last_hidden_state.float(), o.past_key_values


def main():
    with gpu_lock("mercurius_cache_check"):
        trunk, tok = mp.load_deployed(os.environ["MERCURIUS_ROOT"], compressed_cache=False)
        kinds = collections.Counter()
        for name, mod in trunk.named_modules():
            for p in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
                kinds[type(mod).__name__ + ("/gate" if any(g in name for g in mp.GATES) else "")] += p.numel() * p.element_size()
        print("weight memory by module kind (GiB):", {k: round(v / 2**30, 3) for k, v in kinds.most_common(10)}, flush=True)
        stock = types.SimpleNamespace(fw={})
        attn = [l.self_attn for l in trunk.layers if hasattr(l, "self_attn")]
        stock_fw = [a.forward for a in attn]
        res = {}
        with torch.no_grad():
            for i, t in enumerate(TEXTS):
                ids = tok(t, return_tensors="pt")["input_ids"].cuda()
                n = ids.shape[1] // 2
                pos = torch.arange(ids.shape[1], device="cuda")[None]
                for a, f in zip(attn, stock_fw):
                    a.forward = f
                hs, _ = run(trunk, ids, pos=pos)
                _, c = run(trunk, ids[:, :n], pos=pos[:, :n])
                hs2, _ = run(trunk, ids[:, n:], cache=c, pos=pos[:, n:])
                mc.install_compressed_cache(trunk)
                hc, _ = run(trunk, ids, pos=pos)
                _, cc = run(trunk, ids[:, :n], pos=pos[:, :n])
                hc2, cc = run(trunk, ids[:, n:], cache=cc, pos=pos[:, n:])
                lay = [l for l in cc.layers if isinstance(l, mc.LatentCodesLayer)]
                res[i] = {"full": rel(hs, hc), "continuation": rel(hs2, hc2),
                          "compressed_layers": len(lay), "bytes_per_token_per_layer": lay[0].nbytes() / ids.shape[1],
                          "stock_bytes_per_token_per_layer": sum(t.numel() * t.element_size() for t in (c.layers[3].keys, c.layers[3].values)) / n}
                print(i, res[i], flush=True)
        print("compressed cache per token (all", res[0]["compressed_layers"], "attention layers):",
              round(res[0]["bytes_per_token_per_layer"] * res[0]["compressed_layers"]), "bytes; stock bf16 K/V:",
              round(res[0]["stock_bytes_per_token_per_layer"] * res[0]["compressed_layers"]), "bytes", flush=True)


if __name__ == "__main__":
    main()
