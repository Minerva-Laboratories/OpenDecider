"""README figure from the evaluation outputs (runs/, gitignored).   python3 docs/make_figures.py [9B run tag]

Zero-shot accuracy with 95% bootstrap intervals on benchmarks with real labels. Jev markers are third-party numbers
(sources in docs/RESULTS.md, "Sources for numbers not measured here"), not reproduced here.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})

NINE = sys.argv[1] if len(sys.argv) > 1 else "x9b-clean3"
B = [("banking77_8way", "Banking77\n8 intents"), ("prompt_injections", "Prompt-injection\ndetection"),
     ("openbookqa", "OpenBookQA"), ("commonsenseqa", "Commonsense\nQA"), ("pubmedqa", "PubMedQA")]
S = [("public-x2b-clean3", "OpenDecider 2B", BLUE), (f"public-{NINE}", "OpenDecider 9B", AQUA),
     ("zs-9b", "Qwen3.5-9B, letter scores", YELLOW)]
JEV = {"banking77_8way": 0.838, "prompt_injections": 0.870, "openbookqa": 0.942, "commonsenseqa": 0.881}

fig, ax = plt.subplots(figsize=(7.6, 3.3))
w = 0.24
for si, (run, lab, col) in enumerate(S):
    acc = [json.load(open(f"runs/{run}/eval_{b}.json"))["raw"]["accuracy"] for b, _ in B]
    xs = [i + (si - 1) * (w + 0.02) for i in range(len(B))]
    v = [a["value"] for a in acc]
    err = [[a["value"] - a.get("lo", a["value"]) for a in acc], [a.get("hi", a["value"]) - a["value"] for a in acc]]
    ax.bar(xs, v, width=w, color=col, label=lab)
    ax.errorbar(xs, v, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
for i, (b, _) in enumerate(B):
    if b in JEV:
        ax.scatter([i], [JEV[b]], marker="D", s=24, color=INK, zorder=5, edgecolor="white", linewidth=1)
ax.scatter([], [], marker="D", s=24, color=INK, label="Jev, third-party reports")
ax.set_xticks(range(len(B))); ax.set_xticklabels([l for _, l in B], fontsize=8)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("zero-shot accuracy (held-out)")
ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
ax.grid(axis="x", visible=False)
fig.savefig("docs/img/public_benchmarks.png"); plt.close(fig)
print("docs/img/public_benchmarks.png written")
