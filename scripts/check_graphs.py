"""CUDA-graph engine vs eager on the real model: same decide() outputs for several state sizes and question sets."""
import os
import sys

import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from bench_state_cache import QS, state  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/x2b/model.pt"
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
dec = load_decider(ckpt)
from opendecider.deploy import GraphEngine  # noqa: E402
reqs = [{"state": state(n, n), "questions": QS} for n in (3, 20, 60)] + \
       [{"state": {"ticket": "charged twice"}, "questions": {"a": {"type": "choice", "prompt": "Which team?",
                                                                    "options": ["billing", "tech"]}}}]
eager = [dec.decide(r) for r in reqs]
eng = GraphEngine(dec.model).install()
for _ in range(2):
    graph = [dec.decide(r) for r in reqs]
worst = 0.0
for r, a, b in zip(reqs, eager, graph):
    for k in a["answers"]:
        pa, pb = torch.tensor(a["answers"][k]["probs_list"]), torch.tensor(b["answers"][k]["probs_list"])
        worst = max(worst, float((pa - pb).abs().max()))
        assert a["answers"][k]["value"] == b["answers"][k]["value"] or float((pa - pb).abs().max()) < 0.05, k
print("max prob diff graph vs eager:", f"{worst:.2e}", "stats:", eng.stats)
