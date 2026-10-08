"""Concurrent-request throughput of the server's model worker, batching off (--max-batch 1) vs on.

    .venv/bin/python scripts/bench_serving.py --model checkpoints/opendecider-9b --clients 1 4 8 16 --out runs/serving

Each client sends requests back to back (a different support ticket each time, 3 questions) for --seconds.
Reports requests/s and p50/p95 latency per (max_batch, clients), and checks that batched answers match the
unbatched ones on the same requests (max |dp|).
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opendecider.api import ModelWorker                     # noqa: E402
from opendecider.guards import gpu_lock                     # noqa: E402
from opendecider.hub import load_decider                    # noqa: E402
from opendecider.schema import DecideRequest                # noqa: E402

ISSUES = ["charged twice for my March invoice", "the app crashes when I log in", "I want to cancel my plan",
          "my card was declined abroad", "please change the email on my account", "the export to CSV is empty",
          "I never received the refund you promised", "how do I add a second user", "the invoice has the wrong VAT",
          "your website is down for me"]
QS = {"route": {"type": "choice", "prompt": "Which team handles this?",
                "options": ["billing", "technical", "account", "refund", "other"]},
      "urgent": {"type": "noul", "prompt": "Does this need a reply within the hour?"},
      "severity": {"type": "score", "prompt": "How severe is it?", "levels": ["low", "medium", "high", "critical"]}}


def make_req(rng):
    words = " ".join(rng.choice(ISSUES) for _ in range(rng.randint(3, 12)))
    return DecideRequest.model_validate({"state": {"ticket": f"Hello, {words}. Thanks.", "tier": rng.choice(["free", "pro"])},
                                         "questions": QS})


def run(worker, clients, seconds, seed):
    lat, stop = [], time.time() + seconds
    lock = threading.Lock()

    def client(i):
        rng = random.Random(seed * 1000 + i)
        while time.time() < stop:
            req = make_req(rng)
            t = time.perf_counter()
            worker.submit_decide(req)
            with lock:
                lat.append(time.perf_counter() - t)
    t0 = time.time()
    ts = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    lat.sort()
    return {"clients": clients, "requests": len(lat), "req_per_s": len(lat) / wall,
            "p50_ms": 1e3 * statistics.median(lat), "p95_ms": 1e3 * lat[int(0.95 * (len(lat) - 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/opendecider-9b")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default="runs/serving")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with gpu_lock("bench_serving"):
        dec = load_decider(a.model, backbone=a.backbone)
        reqs = [make_req(random.Random(i)) for i in range(16)]
        dec.decide(reqs[0])                                                   # warm-up
        ref = [dec.decide(r) for r in reqs]
        many = dec.decide_many(reqs)
        maxdiff = max(abs(m["answers"][q]["probs"][k] - p) for r, m in zip(ref, many)
                      for q in r["answers"] for k, p in r["answers"][q]["probs"].items())
        flips = sum(m["answers"][q]["value"] != r["answers"][q]["value"] for r, m in zip(ref, many) for q in r["answers"])
        print(f"batched vs single on 16 requests: max |dp| {maxdiff:.2e}, value flips {flips}/48", flush=True)
        rows = []
        for mb in (1, 16):
            w = ModelWorker(dec, max_batch=mb)
            for c in a.clients:
                r = {"max_batch": mb, **run(w, c, a.seconds, seed=c)}
                print(json.dumps(r), flush=True)
                rows.append(r)
        res = {"model": a.model, "backbone": a.backbone, "seconds": a.seconds, "max_abs_dp_batched": maxdiff,
               "value_flips_batched": flips, "rows": rows}
        json.dump(res, open(f"{a.out}/bench_{os.path.basename(os.path.normpath(a.model))}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
