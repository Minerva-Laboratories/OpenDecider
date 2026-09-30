"""Collect every number used in the findings report from run artifacts into reports/data.json.

    .venv/bin/python reports/collect.py
Nothing is typed by hand: public/synthetic evals come from runs/*/eval_*.json, probes from runs/probes-*/probes.json,
validation curves and diagnostics from the training logs.
"""
import ast
import json
import os
import re

R = "runs"
BENCH = ["banking77_77way", "banking77_8way", "pubmedqa", "openbookqa", "commonsenseqa", "prompt_injections"]


def ev(run, split):
    p = f"{R}/{run}/eval_{split}.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    a, e = j["raw"]["accuracy"], j["raw"]["ece"]
    return {"acc": a["value"], "lo": a["lo"], "hi": a["hi"], "ece": e["value"], "n": j["raw"]["n"],
            "by_type": {k: {"acc": v["accuracy"]["value"], "ece": v["ece"]["value"]} for k, v in j.get("by_type", {}).items()},
            "by_family": {k: v["accuracy"]["value"] for k, v in j.get("by_family", {}).items()}}


def evals_from_log(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"\[eval\] step (\d+) (\{.*\})", line.strip())
        if m:
            d = ast.literal_eval(m.group(2))
            out.append({"step": d["step"], "acc": d["val_acc"], "ce": d["val_ce"], "brier": d["val_brier"]})
    return out


def calib_from_log(path):
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("[calib]"):
            return line.strip()
    return None


data = {
    "public": {run: {b: ev(f"public-{run}", b) for b in BENCH} for run in
               ["all-v0", "all-v3branch", "mix-v0", "mix-v3branch", "mix2-v3branch", "mix2-v3branch-vera"]},
    "synthetic_test": {run: {s: ev(run, s) for s in ("test_id", "test_ood")} for run in
                       ["all-v0", "all-v0-1500", "all-v3branch"]},
    "lookup_val": {n: evals_from_log(f"{R}/{n}.log") for n in
                   ["lk-v0", "lk-v3ctx", "lk-v3cond", "lk-v3branch", "lk-v3qemb-q", "lk-v3qcond-hidden"]},
    "v1_ln_val": evals_from_log(f"{R}/v1-ln-600.log"),
    "v3_isolated_val": evals_from_log(f"{R}/v3-fix-final.log"),
    "curves": {n: evals_from_log(f"{R}/{n}.log") for n in ["mix2-v3branch", "mix2-v3branch-vera", "mix-v0", "mix-v3branch"]},
    "calib": {n: calib_from_log(f"{R}/{n}.log") for n in ["mix2-v3branch", "mix2-v3branch-vera"]},
    "probes": {n: json.load(open(f"{R}/probes-{n}/probes.json"))["aggregate"]
               for n in ["mix-v0", "mix-v3branch"] if os.path.exists(f"{R}/probes-{n}/probes.json")},
    "baseline_zeroshot_val": ev("baseline-labelseq-val", "val"),
}
# the lk-v3qcond run log was renamed when it was stopped; keep whatever exists
for alt in ["lk-v3qcond-hidden", "lk-v3qemb-fullcross"]:
    if not data["lookup_val"].get(alt):
        data["lookup_val"][alt] = evals_from_log(f"{R}/{alt}.log")
os.makedirs("reports", exist_ok=True)
json.dump(data, open("reports/data.json", "w"), indent=1)
print("wrote reports/data.json")
for k, v in data["lookup_val"].items():
    print(k, [(e["step"], round(e["acc"], 3)) for e in v])
