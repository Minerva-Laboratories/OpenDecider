"""Released checkpoints (checkpoints/*) vs the original run checkpoints: same decisions?

    .venv/bin/python scripts/check_release.py
"""
import os
import sys

import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from bench_state_cache import QS, state  # noqa: E402
from opendecider.checkpoint import load_decider as load_run  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.hub import load_decider  # noqa: E402

os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
reqs = [{"state": state(n, n), "questions": QS} for n in (3, 30)]
P = lambda outs: torch.tensor([p for o in outs for a in o["answers"].values() for p in a["probs_list"]])
with gpu_lock("check_release"):
    for rel, variant, run in (("checkpoints/opendecider-2b", "int8", "runs/x2b-clean3/model.pt"),
                              ("checkpoints/opendecider-2b", "awq", None),
                              ("checkpoints/opendecider-9b", "gguf-q4_0", "runs/x9b-clean3/model.pt"),
                              ("checkpoints/opendecider-9b-awq", "awq", "runs/x2b-clean3-stitch9b-awq/model.pt")):
        d = load_decider(rel, backbone=variant)
        got = [d.decide(r) for r in reqs]
        msg = f"{rel} [{variant}]: answers {[a['value'] for o in got for a in o['answers'].values()]}"
        assert all(abs(sum(a["probs_list"]) - 1) < 1e-6 for o in got for a in o["answers"].values()), "probs must sum to 1"
        if run:
            del d; torch.cuda.empty_cache()
            ref = [load_run(run).decide(r) for r in reqs]
            msg += f"; max prob diff vs {run}: {float((P(got) - P(ref)).abs().max()):.2e}"
        print(msg, flush=True)
        torch.cuda.empty_cache()
