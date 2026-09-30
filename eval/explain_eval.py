"""Evaluate backbone-generated explanations on typed-decisions cases (no teacher, no human labels needed).

    .venv/bin/python -m eval.explain_eval --ckpt runs/x2b/model.pt --n 24 --samples 4

Per decision: evidence by occlusion, greedy explanation and best-of-N sampled explanation.
  simulatability: the decision model reads ONLY the explanation (option names masked) as its state;
                  P(decision) and whether its argmax recovers the decision.
  baselines     : the same check with a MISMATCHED explanation (another case, same question) and with an empty
                  state ("No information."), so gains over them are evidence carried by the explanation.
  necessity     : drop in P(decision) when the top evidence record is removed from the state.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
from opendecider.batching import Question  # noqa: E402
from opendecider.checkpoint import load_decider  # noqa: E402
from opendecider.explain import Explainer, mask_options  # noqa: E402
from opendecider.guards import gpu_lock, limit_gpu_memory  # noqa: E402

from .typed_decisions import load_cases, questions_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/x2b/model.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--name", default="explain")
    a = ap.parse_args()
    limit_gpu_memory(float(os.environ.get("OPENDECIDER_EVAL_GPU_GB", "24")))
    rows = load_cases("test")
    rng = random.Random(0)
    by_wf = {}
    for r in rows:
        by_wf.setdefault(r["workflow"], []).append(r)
    pick = [r for wf in sorted(by_wf) for r in rng.sample(by_wf[wf], a.n // len(by_wf))]
    res = []
    with gpu_lock("explain_eval"):
        dec = load_decider(a.ckpt)
        ex = Explainer(dec)
        for r in pick:
            name, instr, names, texts, gp, lab = next(x for x in questions_of(r) if len(x[2]) >= 3)
            state = json.loads(r["state"])
            q = Question(instr, texts, "choice")
            t = time.perf_counter()
            greedy = ex.explain(state, q, names, n_samples=1)
            t_greedy = time.perf_counter() - t
            best = ex.explain(state, q, names, n_samples=a.samples)
            c = names.index(greedy["decision"])
            res.append({"id": r["id"], "workflow": r["workflow"], "question": name, "names": names, "c": c,
                        "gold": gp.tolist(), "p": greedy["p"], "greedy": greedy, "best": best,
                        "latency_greedy_s": t_greedy, "state": state, "q": [instr, texts]})
            print(f"{r['id']}:{name} -> {greedy['decision']} ({greedy['p']:.2f})  faith greedy "
                  f"{greedy['faithfulness']:.2f} best {best['faithfulness']:.2f}  {t_greedy:.1f}s", flush=True)
        # baselines: mismatched explanation (same question name, other case) and empty state
        for i, x in enumerate(res):
            q = Question(*x["q"], "choice")
            others = [y for j, y in enumerate(res) if j != i and y["question"] == x["question"]] or \
                     [y for j, y in enumerate(res) if j != i]
            mis = mask_options(rng.choice(others)["greedy"]["explanation"] or "", x["names"])
            P = ex.probs([mask_options(x["greedy"]["explanation"] or "", x["names"]), mis, "No information."], q)
            x["sim_greedy"], x["sim_mismatch"], x["sim_empty"] = (float(v) for v in P[:, x["c"]])
            x["recover_greedy"] = int(P[0].argmax()) == x["c"]
            x["recover_mismatch"] = int(P[1].argmax()) == x["c"]
            x["necessity"] = x["greedy"]["evidence"][0]["weight"] if x["greedy"]["evidence"] else 0.0
    m = lambda k: float(np.mean([x[k] for x in res]))
    summary = {"n": len(res), "sim_greedy": m("sim_greedy"), "sim_best_of_n": float(np.mean([x["best"]["faithfulness"] for x in res])),
               "sim_mismatch": m("sim_mismatch"), "sim_empty": m("sim_empty"),
               "recover_greedy": m("recover_greedy"), "recover_mismatch": m("recover_mismatch"),
               "necessity_top_record": m("necessity"), "p_decision": m("p"),
               "latency_greedy_s_median": float(np.median([x["latency_greedy_s"] for x in res]))}
    print(json.dumps(summary, indent=1))
    os.makedirs(f"runs/{a.name}", exist_ok=True)
    json.dump({"summary": summary, "items": res}, open(f"runs/{a.name}/results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
