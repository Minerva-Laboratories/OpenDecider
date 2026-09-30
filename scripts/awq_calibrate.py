"""AWQ calibration for a checkpoint's backbone (src/opendecider/awq.py). Inputs are recorded while the full model
(bf16 backbone, adapters active as in inference) answers OpenDecider requests built from the training data.

    .venv/bin/python scripts/awq_calibrate.py --ckpt runs/x2b/model.pt --out runs/awq/x2b.pt --n-states 64
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from stitch_backbone import calls, sample_records  # noqa: E402
from opendecider.awq import ActivationRecorder, calibrate  # noqa: E402
from opendecider.backbone import Backbone, BackboneConfig  # noqa: E402
from opendecider.checkpoint import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--out", default="runs/awq/x2b.pt")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    wq = "none" if not ck["backbone_cfg"]["path"].endswith(".gguf") else "none"
    bb = Backbone.load(BackboneConfig(**{**ck["backbone_cfg"], "weight_quant": wq}))
    model, _ = load_model(a.ckpt, backbone=bb)
    model.eval()
    rec = ActivationRecorder(bb.lm.layers, max_rows=a.max_rows, seed=a.seed)
    t0 = time.time()
    with torch.no_grad():
        for k, (qs, mem) in enumerate(calls(model, sample_records(a.n_states, a.seed))):
            model.run(qs, mem)
    rec.remove()
    print(f"recorded {len(rec.x)} layers from {a.n_states} states in {time.time() - t0:.0f}s", flush=True)
    params = calibrate(bb.lm.layers, rec.x)
    import numpy as np
    ratio = np.array([p["err"] / max(p["err_rtn"], 1e-12) for p in params.values()])
    print(f"AWQ / round-to-nearest output error: mean {ratio.mean():.3f}, median {np.median(ratio):.3f}, "
          f"worst {ratio.max():.3f} over {len(params)} layers", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"params": params, "ckpt": a.ckpt, "backbone": ck["backbone_cfg"], "n_states": a.n_states,
                "bits": 4, "group": 64}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
