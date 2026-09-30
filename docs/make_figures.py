"""README figures from committed result files.   python3 docs/make_figures.py   (system python with matplotlib)"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})

# ---- typed-decisions: accuracy vs KL (lower-left is worse on acc; right = better acc, low KL = better probs)
rows = [  # name, accuracy, KL, colour, ours?
    ("meraGPT Decider 1 (proprietary)", 0.768, 0.096, GREY, False),
    ("TypeSafe Jev 1.13 (proprietary)", 0.727, 1.442, GREY, False),
    ("Featherless Simple Jev (35B MoE)", 0.716, 0.488, GREY, False),
    ("ModernBERT-base, fitted per workflow", 0.646, 0.223, GREY, False),
    ("MiniLM-L6, fitted per workflow", 0.587, 0.262, GREY, False),
    ("Prior (ignores input)", 0.470, 0.347, GREY, False),
    ("OpenDecider 2B, zero-shot", 0.577, 0.324, BLUE, True),
    ("OpenDecider 2B→9B stitched, zero-shot", 0.616, 0.279, BLUE, True),
    ("OpenDecider + 9B, linear profile", 0.600, 0.271, AQUA, True),
    ("OpenDecider tree profile (fitted, 2-fold)", 0.675, None, ORANGE, True),
]
fig, ax = plt.subplots(figsize=(7.4, 3.6))
y = list(range(len(rows)))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
for yi, r in zip(y, rows):
    lab = f"{r[1]:.3f}" + (f"   KL {r[2]:.2f}" if r[2] is not None else "")
    ax.text(r[1] + 0.006, yi, lab, va="center", fontsize=7.6, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
ax.set_xlim(0.4, 0.86); ax.set_xlabel("typed-decisions accuracy vs teacher gold (400 cases, 2,000 decisions)")
# reference points from the dataset card (rev f7a2487edd7a): gold = mean of 3 samples from a ~4B teacher
ax.axvline(0.704, color=INK2, lw=1, ls=(0, (1, 2)))
ax.text(0.702, len(rows) - 0.6, "task ceiling 0.704", fontsize=7, color=INK2, ha="right")
ax.axvline(0.735, color=INK2, lw=1, ls=(0, (3, 2)))
ax.text(0.737, len(rows) - 0.6, "teacher self-agreement 0.735", fontsize=7, color=INK2)
ax.grid(axis="y", visible=False)
fig.savefig("docs/img/typed_decisions.png"); plt.close(fig)

# ---- public benchmarks
B = [("banking77_8way", "Banking77\n8 intents"), ("prompt_injections", "Prompt-injection\ndetection"), ("openbookqa", "OpenBookQA"),
     ("commonsenseqa", "Commonsense\nQA"), ("pubmedqa", "PubMedQA")]
S = [("preds-x2b", "OpenDecider 2B", BLUE), ("public-x9b-stitch", "2B→9B stitched (no training)", AQUA),
     ("zs-9b", "Qwen3.5-9B zero-shot", YELLOW)]
JEV = {"banking77_8way": 0.838, "prompt_injections": 0.870, "openbookqa": 0.942, "commonsenseqa": 0.881}
ENS = json.load(open("runs/ens-x2b-9b/ensemble_cv.json"))
fig, ax = plt.subplots(figsize=(7.6, 3.2))
w = 0.2
for si, (run, lab, col) in enumerate(S):
    vals = [json.load(open(f"runs/{run}/eval_{b}.json"))["raw"]["accuracy"]["value"] for b, _ in B]
    ax.bar([i + (si - 1.5) * (w + 0.01) for i in range(len(B))], vals, width=w, color=col, label=lab)
ev = [ENS[b]["ensemble"]["accuracy"]["value"] for b, _ in B]
ax.bar([i + 1.5 * (w + 0.01) for i in range(len(B))], ev, width=w, color=ORANGE, label="OpenDecider 2B + 9B (profile)")
for i, (b, _) in enumerate(B):
    if b in JEV:
        ax.scatter([i], [JEV[b]], marker="D", s=24, color=INK, zorder=5, edgecolor="white", linewidth=1)
    ax.text(i + 1.5 * (w + 0.01), ev[i] + 0.012, f"{ev[i]:.2f}", ha="center", fontsize=7, color=INK)
ax.scatter([], [], marker="D", s=24, color=INK, label="Jev (independent report)")
ax.set_xticks(range(len(B))); ax.set_xticklabels([l for _, l in B], fontsize=8)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("accuracy (held-out)")
ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
ax.grid(axis="x", visible=False)
fig.savefig("docs/img/public_benchmarks.png"); plt.close(fig)
print("figures written")
