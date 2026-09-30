"""Collect runs/*/eval_*.json and runs/*/probes.json into a markdown report.

Wording rule (docs/SPEC.md section 11): results are described as "consistent with" or "inconsistent
with" the public observations in section 1. Nothing here is a statement about how Jev is built.

    python -m eval.report --runs runs --out runs/report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

METRIC_COLS = ["accuracy", "nll", "brier", "ece", "auroc", "selective_acc@80"]
HEADER = ("> This report tests a *hypothesised* architecture assembled from published components. "
          "Probe signatures are compared with public observations of Jev (docs/SPEC.md section 1) and are "
          "described only as consistent or inconsistent with those observations; they do not reveal "
          "or confirm Jev's architecture.\n")

# thresholds for the verdict column (heuristic; state them in the report)
ORDER_TV = 0.01          # permutation mean TV above this counts as order-sensitive (O6)
JITTER = 1e-4            # repeat-call spread below this counts as float jitter (O7)
FLAT_RATIO = 3.0         # largest/smallest latency cell ratio below this counts as "flat-ish" (O3)


def _fmt(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "-"
    v, lo, hi = m["value"], m.get("lo"), m.get("hi")
    if lo is None or hi is None or lo != lo:
        return f"{v:.3f}"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _agg(a: dict | None) -> str:
    if not a:
        return "-"
    return f"{a['mean']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}]"


def load(runs: str):
    evals, probes = [], []
    for p in sorted(glob.glob(os.path.join(runs, "*", "eval_*.json"))):
        with open(p) as f:
            evals.append((p, json.load(f)))
    for p in sorted(glob.glob(os.path.join(runs, "*", "probes*.json"))):
        with open(p) as f:
            probes.append((p, json.load(f)))
    return evals, probes


def metrics_table(evals) -> list[str]:
    lines = ["## Accuracy and calibration (95% percentile-bootstrap CIs; resamples: " + ", ".join(sorted({str(e.get("meta", {}).get("n_boot", "?")) for _, e in evals})) + ")", "",
             "| run | split | n | " + " | ".join(METRIC_COLS) + " | T | ECE after T |",
             "|" + "---|" * (len(METRIC_COLS) + 5)]
    for _, e in evals:
        name = e.get("meta", {}).get("name", "?")
        raw = e.get("raw", {})
        sc = e.get("scaled")
        cells = [_fmt(raw.get(c)) for c in METRIC_COLS]
        T = f"{e['temperature']:.3f}" if "temperature" in e else "-"
        lines.append(f"| {name} | {e.get('split', '?')} | {raw.get('n', 0)} | " + " | ".join(cells)
                     + f" | {T} | {_fmt(sc.get('ece')) if sc else '-'} |")
    return lines + [""]


def family_table(evals) -> list[str]:
    lines = ["## Accuracy by task family", "", "| run | split | family | n | accuracy | ECE |", "|---|---|---|---|---|---|"]
    for _, e in evals:
        name = e.get("meta", {}).get("name", "?")
        for fam, m in e.get("by_family", {}).items():
            lines.append(f"| {name} | {e.get('split', '?')} | {fam or '-'} | {m.get('n', 0)} | "
                         f"{_fmt(m.get('accuracy'))} | {_fmt(m.get('ece'))} |")
    return lines + [""]


def _verdicts(pr: dict) -> dict[str, tuple[str, str]]:
    """{observation: (consistent | inconsistent | inconclusive, detail)}."""
    agg = pr.get("aggregate", {})
    v = {}
    pm = agg.get("permutation_mean_tv")
    if pm:
        word = "consistent" if pm["lo"] > ORDER_TV else "inconsistent" if pm["hi"] <= ORDER_TV else "inconclusive"
        v["O6 order sensitivity"] = (word, f"mean TV {_agg(pm)}")
    rc = agg.get("repeat_class")
    if rc:
        word = "consistent" if rc == "sampling_variance" else "inconsistent"
        v["O7 run-to-run variance"] = (word, f"{rc}, max diff {agg.get('repeat_max_abs_diff'):.2e}")
    lat = pr.get("latency")
    if lat:
        r = lat["ratio_largest_vs_smallest_wall_p50"]
        v["O3 batching barely changes latency"] = ("consistent" if r < FLAT_RATIO else "inconsistent",
                                                   f"largest/smallest cell p50 ratio {r:.2f}")
    return v


def probe_table(probes) -> list[str]:
    lines = ["## Behavioural probe signatures (section 7.4)", "",
             "| run | variant | slot_emb | perm. mean TV | first-position excess | dummy mass | dummy TV (renorm.) "
             "| duplicate pair gain | label-length shift | repeat class |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, p in probes:
        m, a = p.get("meta", {}), p.get("aggregate", {})
        lines.append(f"| {m.get('name', '?')} | {m.get('variant', '-')} | {m.get('slot_emb', '-')} | "
                     f"{_agg(a.get('permutation_mean_tv'))} | {_agg(a.get('first_position_excess_mass'))} | "
                     f"{_agg(a.get('dummy_mass'))} | {_agg(a.get('dummy_tv_renormalised'))} | "
                     f"{_agg(a.get('duplicate_pair_gain'))} | {_agg(a.get('label_length_mean_shift'))} | "
                     f"{a.get('repeat_class', '-')} |")
    lines += ["", "### Comparison with public observations", "",
              f"Heuristic thresholds: order-sensitive if the permutation-TV CI lies above {ORDER_TV}; "
              f"float jitter if repeat spread < {JITTER:g}; latency flat-ish if the largest/smallest grid "
              f"cell ratio < {FLAT_RATIO}. O6 and O7 are anecdotal observations (section 1).", ""]
    for _, p in probes:
        name = p.get("meta", {}).get("name", "?")
        for obs, (word, detail) in _verdicts(p).items():
            lines.append(f"- **{name}** / {obs}: {word} with public observations ({detail}).")
    lat_lines = []
    for _, p in probes:
        lat = p.get("latency")
        if not lat:
            continue
        name = p.get("meta", {}).get("name", "?")
        lat_lines += [f"#### {name}", "", "| K | questions | wall p50 ms | wall p95 ms | server p50 ms |", "|---|---|---|---|---|"]
        for c in lat["cells"]:
            s = f"{c['server_ms_p50']:.1f}" if c.get("server_ms_p50") is not None else "-"
            lat_lines.append(f"| {c['K']} | {c['Q']} | {c['wall_ms_p50']:.1f} | {c['wall_ms_p95']:.1f} | {s} |")
        lat_lines.append("")
    if lat_lines:
        lines += ["", "### Latency scaling", "",
                  "On-device figures; Jev's published 70-500 ms includes network and was measured by TypeSafe.", ""] + lat_lines
    return lines + [""]


def build_report(runs: str = "runs") -> str:
    evals, probes = load(runs)
    out = ["# OpenDecider evaluation report", "", HEADER]
    if evals:
        out += metrics_table(evals) + family_table(evals)
    if probes:
        out += probe_table(probes)
    if not evals and not probes:
        out.append("_No eval or probe results found._")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = build_report(a.runs)
    out = a.out or os.path.join(a.runs, "report.md")
    with open(out, "w") as f:
        f.write(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
