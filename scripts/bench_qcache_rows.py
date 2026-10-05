"""Where the question cache starts to pay off: decide() latency with and without it, by number of option rows.

    .venv/bin/python scripts/bench_qcache_rows.py --ckpt runs/x9b-clean3/model.pt --out runs/latency/qcache_rows_9b.json

Serving path otherwise (tuned int8 kernels if present); state of ~0.9k tokens (bench_state_cache.state(25, 25)).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ["OPENDECIDER_STATE_CACHE_MB"] = "0"
from bench_state_cache import state  # noqa: E402
from profile_latency import workload  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.guards import gpu_lock  # noqa: E402

SHAPES = [(1, 4), (3, 4), (2, 16), (4, 16), (2, 32), (8, 16), (4, 32), (16, 8), (8, 32), (16, 16), (16, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--events", type=int, default=25)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    with gpu_lock("bench_qcache_rows"):
        dec = load_decider(a.ckpt)
        cfg = dec.model.cfg
        st = state(a.events, a.events)
        for n, k in SHAPES:
            req = {"state": st, "questions": workload(n, k)}
            rows = sum(len(q.get("options") or q.get("levels") or [0, 0]) for q in req["questions"].values())
            row = {"questions": n, "options": k, "rows": rows}
            for label, on in (("with_qcache_s", True), ("without_qcache_s", False)):
                cfg.v3_question_cache, cfg.v3_qcache_min_rows = on, 0
                dec.decide(req)
                lat = []
                for _ in range(a.repeats):
                    torch.cuda.synchronize(); t = time.perf_counter(); dec.decide(req); torch.cuda.synchronize()
                    lat.append(time.perf_counter() - t)
                row[label] = float(np.median(lat))
            print(json.dumps(row), flush=True)
            res.append(row)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
