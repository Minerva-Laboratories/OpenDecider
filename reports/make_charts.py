"""Static charts for the findings report (reads reports/data.json, writes reports/fig_*.png).

    python3 reports/make_charts.py        # system python (matplotlib)
Palette: validated reference categorical slots (blue, orange, aqua, yellow), fixed order; light surface.
Two slots are below 3:1 contrast on white, so every chart has direct value labels and the PDF carries tables.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = json.load(open("reports/data.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# colour follows the MODEL across all figures: V0 synth-only, V0 mixed, V3 branched, V3 branched + VeRA
V0S, V0M, V3B, V3V = C
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def lookup_chance():
    ks = []
    n = 0
    for line in open("data/synthetic/val.jsonl"):
        r = json.loads(line)
        if r["family"] != "record_lookup":
            continue
        n += 1
        if n > 200:
            break
        ks += [1 / len(q["options"]) for q in r["questions"] if len(q["options"]) <= 255]
    return sum(ks) / len(ks)


# ---------------------------------------------------------------- Fig 1: where must binding happen?
lv = D["lookup_val"]
best = lambda k: max(e["acc"] for e in lv[k]) if lv.get(k) else float("nan")
last = lambda k: lv[k][-1]["acc"]
rows = [("V3 trunk on isolated features\n(+2 question-aware context layers)", last("lk-v3ctx")),
        ("V3 trunk, question-only conditioning\n(options encoded alone)", last("lk-v3qemb-q")),
        ("V0: frozen LLM reads state→question→all options\n(one causal row) + head", last("lk-v0")),
        ("V3 conditioned: same LLM row + dense trunk", last("lk-v3cond")),
        ("V3 branched: LLM reads each option alone\n+ bidirectional dense trunk", last("lk-v3branch"))]
fig, ax = plt.subplots(figsize=(7.2, 3.1))
y = range(len(rows))[::-1]
ax.barh(list(y), [v for _, v in rows], color=C[0], height=0.55)
for yi, (_, v) in zip(y, rows):
    ax.text(v + 0.012, yi, f"{v:.3f}", va="center", color=INK, fontsize=8.5,
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"}, zorder=6)
ax.set_yticks(list(y)); ax.set_yticklabels([r for r, _ in rows], fontsize=7.8)
zs = D["baseline_zeroshot_val"]["by_family"]["record_lookup"]
ch = lookup_chance()
for xv, lab in ((ch, f"chance {ch:.2f}"), (zs, f"frozen LLM zero-shot {zs:.2f}")):
    ax.axvline(xv, color=INK2, lw=1, ls=(0, (3, 2)))
    ax.text(xv + 0.005, len(rows) - 0.45, lab, color=INK2, fontsize=7.5)
ax.set_xlim(0, 1); ax.set_xlabel("validation accuracy, record_lookup (522 questions), final eval")
ax.grid(axis="y", visible=False)
fig.savefig("reports/fig1_lookup.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: public benchmarks
B = [("banking77_77way", "Banking77\n77-way"), ("banking77_8way", "Banking77\n8-way"), ("pubmedqa", "PubMedQA\nyes/no"),
     ("openbookqa", "OpenBookQA"), ("commonsenseqa", "Commonsense\nQA"), ("prompt_injections", "Prompt\ninjections")]
S = [("all-v0", "V0, synthetic data only"), ("mix-v0", "V0, mixed corpus"),
     ("mix2-v3branch", "V3 branched, mixed + injection"), ("mix2-v3branch-vera", "V3 branched + VeRA (top-6, row pass)")]
JEV = {"banking77_77way": 0.788, "banking77_8way": 0.838, "openbookqa": 0.942, "commonsenseqa": 0.881, "prompt_injections": 0.870}
CHANCE = {"banking77_77way": 1 / 77, "banking77_8way": 1 / 8, "pubmedqa": 0.5, "openbookqa": 0.25, "commonsenseqa": 0.2, "prompt_injections": 0.5}
fig, ax = plt.subplots(figsize=(7.6, 3.4))
w = 0.19
for si, (run, lab) in enumerate(S):
    xs, vs, lo, hi = [], [], [], []
    for bi, (b, _) in enumerate(B):
        e = D["public"][run][b]
        xs.append(bi + (si - 1.5) * (w + 0.012)); vs.append(e["acc"]); lo.append(e["acc"] - e["lo"]); hi.append(e["hi"] - e["acc"])
    ax.bar(xs, vs, width=w, color=C[si], label=lab, yerr=[lo, hi], error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 0})
for bi, (b, _) in enumerate(B):
    ax.plot([bi - 0.45, bi + 0.45], [CHANCE[b]] * 2, color=INK2, lw=1, ls=(0, (2, 2)))
    if b in JEV:
        ax.scatter([bi], [JEV[b]], marker="D", s=26, color=INK, zorder=5, edgecolor="white", linewidth=1.2)
ax.scatter([], [], marker="D", s=26, color=INK, label="Jev, independent report")
ax.plot([], [], color=INK2, lw=1, ls=(0, (2, 2)), label="chance")
ax.set_xticks(range(len(B))); ax.set_xticklabels([l for _, l in B], fontsize=7.8)
ax.set_ylim(0, 1); ax.set_ylabel("accuracy (95% bootstrap CI)")
ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
ax.grid(axis="x", visible=False)
fig.savefig("reports/fig2_public.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: behavioural probes
P = D["probes"]
M = [("permutation_mean_tv", "Permutation: mean TV\nacross 20 orderings", "0 = order-invariant"),
     ("dummy_mass", "Dummy option:\nmass taken by an\nirrelevant option", "lower is better"),
     ("duplicate_pair_gain", "Duplicate option:\nextra mass for the\nduplicated answer", "lower is better"),
     ("duplicate_split_first", "Duplicate:\nshare kept by the\noriginal copy", "0.5 = even split"),
     ("label_length_mean_shift", "Label length:\nshift when one option\nis reworded longer", "0 = no verbosity bias")]
fig, axes = plt.subplots(1, len(M), figsize=(8.2, 2.8), sharey=True)
for ax, (k, t, note) in zip(axes, M):
    vals = [P["mix-v0"][k]["mean"], P["mix-v3branch"][k]["mean"]]
    cis = [(P[r][k]["mean"] - P[r][k]["lo"], P[r][k]["hi"] - P[r][k]["mean"]) for r in ("mix-v0", "mix-v3branch")]
    ax.bar([0, 1], vals, color=[V0M, V3B], width=0.6, yerr=list(zip(*cis)), error_kw={"elinewidth": 0.7, "ecolor": INK2})
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_title(t, fontsize=7.4, color=INK); ax.set_xticks([0, 1]); ax.set_xticklabels(["V0\n(mixed)", "V3 branched\n(mixed)"], fontsize=7)
    ax.set_xlabel(note, fontsize=6.8); ax.grid(axis="x", visible=False); ax.set_ylim(0, 0.62)
fig.tight_layout(w_pad=0.6)
fig.savefig("reports/fig3_probes.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: VeRA vs frozen validation curves
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for i, (run, lab) in enumerate((("mix2-v3branch", "frozen backbone"), ("mix2-v3branch-vera", "+ VeRA (top-6, row pass)"))):
    cur = D["curves"][run]
    xs, ys = [e["step"] for e in cur], [e["acc"] for e in cur]
    ax.plot(xs, ys, color=C[i + 2], lw=2, marker="o", ms=4)
    ax.text(xs[-1] + 40, ys[-1], f"{lab} ({ys[-1]:.3f})", color=INK, fontsize=7.5, va="center")
ax.set_xlim(300, 2750); ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy\n(synthetic + public + injection)")
fig.savefig("reports/fig4_vera.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: synthetic ID vs OOD, equal budget
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for i, (run, lab) in enumerate((("all-v0-1500", "V0"), ("all-v3branch", "V3 branched"))):
    xs = [j + (i - 0.5) * 0.36 for j in range(2)]
    es = [D["synthetic_test"][run][s] for s in ("test_id", "test_ood")]
    ax.bar(xs, [e["acc"] for e in es], width=0.34, color=(V0S, V3B)[i], label=lab,
           yerr=[[e["acc"] - e["lo"] for e in es], [e["hi"] - e["acc"] for e in es]], error_kw={"elinewidth": 0.7, "ecolor": INK2})
    for x, e in zip(xs, es):
        ax.text(x, e["hi"] + 0.01, f"{e['acc']:.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["in-distribution\n(13 train families)", "OOD\n(7 unseen families)"], fontsize=8)
ax.set_ylim(0.3, 0.62); ax.set_ylabel("test accuracy (95% CI)")
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
ax.grid(axis="x", visible=False)
fig.savefig("reports/fig5_synthetic.png"); plt.close(fig)
print("charts written")
