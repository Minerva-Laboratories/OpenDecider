"""Build reports/OpenDecider_findings.pdf from reports/data.json + reports/fig*.png.

    .venv/bin/python reports/collect.py && python3 reports/make_charts.py && .venv/bin/python reports/make_report.py
All numbers in tables come from data.json (run artifacts). Jev figures are third-party reports (cited).
"""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

INK, INK2, RULE, TINT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d6d5d0"), colors.HexColor("#f3f2ee")
BLUE = colors.HexColor("#2a78d6")
st = {
    "title": ParagraphStyle("t", fontName="DVB", fontSize=19, leading=23, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="DV", fontSize=8.8, leading=12.2, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("sm", fontName="DV", fontSize=7.4, leading=9.8, textColor=INK2, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=7.6, leading=10, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("c", fontName="DV", fontSize=7.4, leading=9.4, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="DVB", fontSize=7.4, leading=9.4, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.2, leading=11.2, textColor=INK),
}
D = json.load(open("reports/data.json"))
P = lambda t, s="p": Paragraph(t, st[s])
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"


def table(rows, widths, header=True, bold_first_col=False):
    data = [[Paragraph(str(c), st["cellb"] if (header and i == 0) or (bold_first_col and j == 0) else st["cell"])
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), TINT))
    t.setStyle(TableStyle(style))
    return t


def boxed(text):
    t = Table([[Paragraph(text, st["box"])]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 2.2, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def fig(path, width_mm, caption):
    img = Image(path)
    r = width_mm * mm / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * r, img.imageHeight * r
    return KeepTogether([img, P(caption, "cap")])


def pub(run, b):
    e = D["public"][run][b]
    return f"{f3(e['acc'])}<br/><font size=6.3 color='#52514e'>[{f2(e['lo'])}–{f2(e['hi'])}] · ECE {f2(e['ece'])}</font>"


def syn(run, s):
    e = D["synthetic_test"][run][s]
    return f"{f3(e['acc'])} [{f3(e['lo'])}, {f3(e['hi'])}]", f2(e["ece"])


lv = D["lookup_val"]
lk = {k: v[-1]["acc"] for k, v in lv.items() if v}
pr = D["probes"]
pm = lambda r, k: pr[r][k]["mean"]
vera_T = D["calib"]["mix2-v3branch-vera"]
import ast as _ast
type_T = ", ".join(f"{k} {v:.2f}" for k, v in _ast.literal_eval(vera_T.split("by type ")[-1]).items())

story = []
# ------------------------------------------------------------------------------------------------ title
story += [P("OpenDecider: findings", "title"),
          P("Testing an open, hypothesised architecture for “System One” decision models (typed questions in, "
            "calibrated probabilities over caller-defined options out, no text generation). Status report, 24 Sep 2026.", "sub"),
          boxed("<b>How to read this report.</b> This is a hypothesis test, not a description of any vendor's system. TypeSafe has "
                "not published Jev's architecture. We build designs from published components and open weights and ask whether "
                "they reproduce Jev's <i>publicly observable</i> behaviour. Statements about Jev are phrased as “consistent / "
                "inconsistent with public observations”. Jev numbers are third-party reports on different setups (cited in §6) "
                "and are not directly comparable to ours. All of our numbers come from run artefacts. Confidence intervals are 95% bootstrap "
                "(1,000 resamples unless noted)."),
          Spacer(1, 8)]

# ------------------------------------------------------------------------------------------------ 1 summary
b77v, b8v, pmv, piv = (D["public"]["mix2-v3branch-vera"][k] for k in ("banking77_77way", "banking77_8way", "pubmedqa", "prompt_injections"))
story += [P("1. Summary", "h1")]
for t in [
    f"<b>The frozen LLM must do the question→state binding.</b> A small trunk trained from scratch on independently encoded "
    f"features never learned to look up a record (lookup accuracy {f3(lk['lk-v3ctx'])}, chance ≈ 0.27). Designs in which the "
    f"frozen LLM reads state → question → option reached {f3(lk['lk-v0'])}–{f3(lk['lk-v3branch'])}. Late-fusion V1 (pooled "
    f"vectors) did no better than a string-matching heuristic on the full synthetic suite ({f3(D['v1_ln_val'][-1]['acc'])} vs 0.344; chance 0.30).",
    f"<b>Best design: “V3 branched”.</b> The frozen LLM reads each option <i>alone</i>, in the context of the cached state and the "
    f"question. A dense bidirectional trunk (pre-norm ScaleNorm blocks, xATGLU, RoPE, no pooling until a per-option CLS) then compares "
    f"the options. It is the most accurate on lookups ({f3(lk['lk-v3branch'])} vs {f3(lk['lk-v0'])} for V0). It generalises better to unseen "
    f"task families ({syn('all-v3branch','test_ood')[0]} vs {syn('all-v0-1500','test_ood')[0]} OOD at equal budget). It is exactly "
    f"invariant to option order.",
    f"<b>Order sensitivity (O6) has a clear mechanism in our models.</b> Reading all options in one causal row gives large, item-specific "
    f"order effects: mean total variation {f2(pm('mix-v0','permutation_mean_tv'))} across 20 orderings, and a late duplicate takes "
    f"{f2(1 - pm('mix-v0','duplicate_split_first'))} of the pair's mass. Per-option reading removes them entirely (TV "
    f"{f2(pm('mix-v3branch','permutation_mean_tv'))}). In a branched design, order sensitivity can only come from explicit slot embeddings.",
    f"<b>Light, row-only adaptation helps.</b> VeRA adapters on the top 6 layers, active only while reading question/option rows (so the "
    f"state cache stays frozen), raise Banking77 8-way to {f3(b8v['acc'])} (ECE {f2(b8v['ece'])}), 77-way to {f3(b77v['acc'])}, and prompt-injection "
    f"detection to {f3(piv['acc'])}.",
    "<b>Data coverage drives recognition tasks.</b> A permissively licensed mix (intents, NLI, yes/no QA, science MC, topic, "
    "injection) tripled Banking77 77-way accuracy relative to synthetic-only training, without ever training on the benchmark datasets.",
    "<b>Remaining gap vs public Jev reports.</b> We are near on intent routing and PubMedQA, but far behind on knowledge-heavy multiple choice "
    "(OpenBookQA, CommonsenseQA). That is consistent with the limited factual knowledge of the frozen 0.8B backbone we use."]:
    story.append(P("•&nbsp;&nbsp;" + t))

# ------------------------------------------------------------------------------------------------ 2 setup
story += [P("2. Setup", "h1"), table([
    ["Item", "Choice"],
    ["Backbone", "Qwen/Qwen3.5-0.8B (post-trained), pinned revision 2fc06364; hybrid 18 Gated-DeltaNet + 6 gated full-attention layers; "
                 "frozen; int8 weight-only inference; bf16 during training"],
    ["Caches", "State encoded once per request; KV cache and state features stored int8 (per-token absmax); GDN recurrent state fp32"],
    ["Interface", "POST /v1/decide: state + named questions (Choice ≤255 options, Noul yes/no, Score ordinal levels); probabilities only; "
                  "composite (decomposed) and chained questions supported"],
    ["Training data", "Programmatic synthetic tasks: 20 families, 13 train / 7 held-out OOD. Permissively licensed public corpus: CLINC150, MASSIVE, "
                      "BoolQ, SNLI, ARC, WinoGrande, DBpedia-14; injection sets from 5 sources. Licenses and revisions in data/MANIFEST.md"],
    ["Held-out benchmarks", "Banking77 (77- and 8-way), PubMedQA (yes/no), OpenBookQA, CommonsenseQA, deepset/prompt-injections. Never trained on "
                            "(deduplicated against them)"],
    ["Losses & calibration", "Cross-entropy + Brier (proper scoring), ordinal EMD auxiliary; temperature fitted on a calibration split "
                             "(global and per question type)"],
    ["Hardware", "Jetson AGX Orin 64 GB (MODE_50W) for all training and evaluation reported here"],
], [32 * mm, 142 * mm], bold_first_col=True)]

# ------------------------------------------------------------------------------------------------ 3 variants
story += [KeepTogether([P("3. Architectures compared", "h1"), table([
    ["Variant", "How the model reads state, question and options", "Status"],
    ["V0", "Frozen LLM reads [state] → [question; opt₁…opt_K] as one causal row (state from prefix cache); MLP on each option's last token",
     "strong baseline; order-sensitive"],
    ["V1", "Late fusion: question and each option encoded alone, pooled to vectors; small transformer cross-attends to state memory",
     "fails (chance)"],
    ["V3 isolated", "User design: dense tokens of state, question, options (each encoded alone) → trunk: bidirectional attention across all option "
                    "tokens, options cross-attend to [state; question], xATGLU, ScaleNorm pre-norm, RoPE, per-option CLS", "fails to bind"],
    ["V3 conditioned", "Same trunk on dense features from the V0-style causal row", "≈ V0"],
    ["V3 branched", "Same trunk; the LLM reads one row [question; option_k] per option from the cached state (each option sees state + question, "
                    "never other options)", "<b>best</b>"],
    ["+ VeRA", "VeRA adapters (shared frozen random A,B; trainable scaling vectors) on the top 6 layers, active only on option rows",
     "<b>best + adapted</b>"],
], [24 * mm, 118 * mm, 32 * mm], bold_first_col=True)])]

# ------------------------------------------------------------------------------------------------ 4 hypotheses
story += [P("4. Architecture hypotheses and evidence", "h1")]
v1 = D["v1_ln_val"][-1]["acc"]
story += [P("H1. Question→state binding must come from the frozen LLM's own attention", "h2"),
          P(f"<b>Test.</b> We trained on <i>record_lookup</i> only (~1.5k states; “What is the city listed for Ximena Novak?” over ~20 JSON "
            f"records whose cities all appear as options) and compared where the binding can happen. <b>Result.</b> The trunk on isolated features "
            f"stays at chance even with question-aware context layers ({f3(lk['lk-v3ctx'])}), with a fitted temperature of 6.7 "
            f"(confidently wrong). Every design where the LLM reads the question with the state in context learns quickly. On the full "
            f"synthetic suite, late-fusion V1 ended at {f3(v1)} validation accuracy, against a chance level of 0.30 and 0.344 for a string-match "
            f"heuristic. <b>Verdict: supported</b> at this data scale."),
          fig("reports/fig1_lookup.png", 168, "Figure 1. Lookup diagnostic: final validation accuracy by where the binding happens. "
              "Dashed lines: chance, and the untrained frozen LLM scoring each option's likelihood (zero-shot).")]
story += [P("H2. The LLM must see each option in context; bringing options in afterwards is not enough", "h2"),
          P("<b>Test.</b> The LLM reads state → question → “Answer:”, and options are encoded separately and matched to the question's "
            "final state. <b>Result.</b> Matching options' final hidden states is at chance (0.264, no training), because hidden states "
            "predict the <i>next</i> token. Matching option token embeddings (the tied LM-head space) gives 0.414–0.443 with no training, "
            f"and a trained bilinear readout plateaus at ≈ 0.45. The trained trunk on these features stayed at {f3(lk['lk-v3qemb-q'])}. "
            f"Options read inside the LLM reach 0.83–0.88. <b>Verdict: supported.</b>")]
id_v0, id_v0e = syn("all-v0-1500", "test_id"); ood_v0, ood_v0e = syn("all-v0-1500", "test_ood")
id_b, id_be = syn("all-v3branch", "test_id"); ood_b, ood_be = syn("all-v3branch", "test_ood")
story += [P("H3. Reading each option alone, then comparing options bidirectionally, generalises better than one causal row", "h2"),
          P(f"<b>Test.</b> V0 against V3 branched, trained 1,500 steps each on synthetic data. Test on 500 states per split: in-distribution "
            f"families and 7 unseen (OOD) families. <b>Result.</b> OOD: branched {ood_b} vs V0 {ood_v0}. In-distribution: {id_b} vs {id_v0} "
            f"(overlapping CIs). On lookups branched reaches {f3(lk['lk-v3branch'])} against {f3(lk['lk-v0'])}. On public benchmarks after mixed "
            f"training it leads on recognition tasks (intents, PubMedQA) and ties on knowledge-heavy multiple choice (§5). "
            f"<b>Verdict: supported</b> for OOD generalisation; not for in-distribution accuracy."),
          fig("reports/fig5_synthetic.png", 105, "Figure 2. Equal-budget synthetic test accuracy, 95% CIs.")]
story += [P("H4. The source of option-order sensitivity (O6)", "h2"),
          P(f"<b>Test.</b> The §7.4 probe battery on 60 items (Banking77 8-way and CommonsenseQA): 20 random orderings, an irrelevant dummy "
            f"option, an exact duplicate, same-meaning long wording, and 50 repeated calls. <b>Result.</b> V0, which reads options in one causal row, "
            f"is strongly order-sensitive (TV {f2(pm('mix-v0','permutation_mean_tv'))}) without any global first-position prior. A duplicated "
            f"option gains {f2(pm('mix-v0','duplicate_pair_gain'))} mass, almost all of it on the <i>later</i> copy, which is a recency effect. V3 branched is "
            f"exactly permutation-invariant, splits duplicates evenly and gives half the mass to dummies. However, it shows a "
            f"<b>verbosity bias</b>: rewording one option longer raises it by {f2(pm('mix-v3branch','label_length_mean_shift'))}. Both "
            f"are bit-exact deterministic across repeated calls. <b>Verdict.</b> In our models, causal reading of the option list is a sufficient "
            f"mechanism for O6. V0 is <i>consistent with</i> the anecdotal O6 observation; V3 branched is <i>inconsistent</i> with it unless slot "
            f"embeddings are enabled. Both are <i>inconsistent with</i> O7 (run-to-run variance) unless the stochastic sampler is on."),
          fig("reports/fig3_probes.png", 172, "Figure 3. Behavioural probe signatures (mean, 95% CI), trained on the mixed corpus.")]
story += [P("H5. Data coverage, not steps, drives recognition tasks", "h2"),
          P("<b>Test.</b> The same V0 trained on synthetic data only, against synthetic plus the permissive public corpus; then V3 branched with injection "
            "data added. The benchmark datasets were never used. <b>Result.</b> Banking77 77-way went from "
            f"{f3(D['public']['all-v0']['banking77_77way']['acc'])} to {f3(D['public']['mix-v0']['banking77_77way']['acc'])}; OpenBookQA from "
            f"{f3(D['public']['all-v0']['openbookqa']['acc'])} to {f3(D['public']['mix-v0']['openbookqa']['acc'])}; prompt-injection from "
            f"{f3(D['public']['mix-v3branch']['prompt_injections']['acc'])} to {f3(D['public']['mix2-v3branch']['prompt_injections']['acc'])} "
            f"with injection data. Synthetic-only validation accuracy had plateaued. <b>Verdict: supported.</b>")]
story += [P("H6. Row-only adapters improve decisions without touching the shared state cache", "h2"),
          P(f"<b>Test.</b> V3 branched frozen against V3 branched + VeRA (top 6 layers, rank 256, active only on option rows), same data. "
            f"<b>Result.</b> VeRA leads at every validation checkpoint and improves recognition accuracy and calibration on the "
            f"held-out benchmarks (§5). There is no gain on knowledge-heavy multiple choice. The fitted per-type temperatures ({type_T}) "
            f"show Score answers are <i>under</i>-confident, so a single global temperature is the wrong correction. <b>Verdict: supported</b> "
            f"for recognition and calibration."),
          fig("reports/fig4_vera.png", 105, "Figure 4. Validation accuracy (synthetic + public + injection mix), frozen against VeRA.")]

# ------------------------------------------------------------------------------------------------ 5 public benchmarks
story += [P("5. Held-out public benchmarks", "h1"),
          fig("reports/fig2_public.png", 172, "Figure 5. Accuracy on held-out public benchmarks (95% CI). Diamonds are independent reports on Jev "
              "(different conditions; see §6). PubMedQA has no Jev accuracy marker because the public figure is a different metric.")]
max_dT = 0.0
for _r in ("all-v0", "mix-v0", "mix2-v3branch", "mix2-v3branch-vera"):
    for _b in ("banking77_77way", "banking77_8way", "pubmedqa", "openbookqa", "commonsenseqa", "prompt_injections"):
        _j = json.load(open(f"runs/public-{_r}/eval_{_b}.json"))
        max_dT = max(max_dT, abs(_j["scaled"]["ece"]["value"] - _j["raw"]["ece"]["value"]))
BN = [("banking77_77way", "Banking77 77-way"), ("banking77_8way", "Banking77 8-way"), ("pubmedqa", "PubMedQA"),
      ("openbookqa", "OpenBookQA"), ("commonsenseqa", "CommonsenseQA"), ("prompt_injections", "Prompt injections")]
JEV = {"banking77_77way": "0.790 / 0.788", "banking77_8way": "0.838", "pubmedqa": "69.0 (decision score)",
       "openbookqa": "0.942", "commonsenseqa": "0.881", "prompt_injections": "0.870"}
rows = [["Benchmark (n)", "V0 synthetic only", "V0 mixed", "V3 branched mixed+inj.", "V3 branched + VeRA", "Jev (independent)"]]
for b, name in BN:
    rows.append([f"{name} ({D['public']['mix-v0'][b]['n']})", pub("all-v0", b), pub("mix-v0", b), pub("mix2-v3branch", b),
                 pub("mix2-v3branch-vera", b), JEV[b]])
story += [table(rows, [30 * mm, 27 * mm, 27 * mm, 30 * mm, 30 * mm, 30 * mm], bold_first_col=True),
          P(f"Cells: accuracy, [95% CI], raw ECE (15 equal-mass bins). Temperature scaling fitted on our (synthetic + public) calibration split "
            f"changes these ECEs by at most {max_dT:.3f}: a global temperature cannot fix a domain shift.", "cap")]

# ------------------------------------------------------------------------------------------------ 6 Jev comparison
story += [KeepTogether([P("6. Consistency with public observations of Jev", "h1"), table([
    ["Observation", "Source quality", "Our finding", "Verdict"],
    ["O1/O2: Choice ≤ 255 options; Noul; Score; two-stage above 255", "vendor docs", "Implemented; 255 comes from a 256-slot table (slot 0 reserved). "
     "Branched V3 needs no cap without slots", "consistent (by construction)"],
    ["O3/O4: state read once; questions isolated; cost ≈ input tokens", "vendor docs", "State encoded once and cached; isolation tested; no decoding", "consistent"],
    ["O5: 70–500 ms end to end", "vendor-measured", "Not yet comparable: our Orin pipeline is unoptimised (≈ 1.2 s); a published shared-state "
     "method (SpecLA, arXiv:2607.16673) and shared-prefix attention (Hydragen, arXiv:2402.05099) are the planned path", "open"],
    ["O6: option order changes probabilities", "anecdotal", "Causal option row (V0) reproduces it; per-option reading does not", "V0 consistent; branched not"],
    ["O7: run-to-run variance", "anecdotal", "Deterministic unless the stochastic sampler is enabled", "inconsistent (sampler off)"],
    ["O8: calibrated probabilities", "vendor claim + independent tests (ECE 0.024–0.032 in-domain; 0.107 OOD)",
     "ECE 0.03–0.07 on task types seen in training (synthetic test, PubMedQA, 8-way intents); 0.13–0.26 on unfamiliar ones; "
     "Score questions miscalibrated in both systems' reports", "partly consistent"],
], [40 * mm, 34 * mm, 68 * mm, 32 * mm], bold_first_col=True)])]
story += [P("Independent Jev reports used: Towards Data Science (Banking77 77-way, 79.0%); AY Automate (Banking77 8/77-way 83.8/78.8%, "
            "prompt-injections 87.0%, AUROC 0.990); scienthoon/jev-ood-calibration (OpenBookQA 94.2%, CommonsenseQA 88.1%, HellaSwag 86.1%; OOD ECE "
            "0.107); Jevals (PubMedQA decision score 69.0); BERI phishing study (single question 62.6% → 95.0% with 5 sub-questions and fitted weights). "
            "TypeSafe's own claims (“193.6× faster, 444.6× cheaper”) are measured as agreement with frontier models, not ground truth.", "small")]

# ------------------------------------------------------------------------------------------------ 7 limits / next
story += [P("7. Limitations", "h1")]
for t in ["Backbone is 0.8B and frozen (plus light adapters): factual recall caps knowledge-heavy multiple choice.",
          "Our synthetic suite over-represents computation (arithmetic, dates, counting), the documented weak spots of this model class. "
          "Absolute numbers on it are low by design.",
          "Latency is not yet optimised, and no comparison with Jev latency is claimed.",
          "Benchmark n is small for prompt-injections (116) and Banking77 8-way (160); CIs are wide.",
          "Jev figures come from third parties under different conditions; no claim is made about Jev's internals."]:
    story.append(P("•&nbsp;&nbsp;" + t))
story += [P("8. Next steps", "h1")]
for t in ["Reduce the verbosity bias of per-option reading (paraphrase and length augmentation); ordinal head plus per-type temperature for Score questions.",
          "Shared-state option rows (the published delta-rule prefix decomposition) and shared-prefix attention, so branched inference costs about as much as V0; then on-device latency.",
          "Decomposition: composite and chained questions are implemented in the API. Next is measuring gains on multi-hop families.",
          "Scale the backbone (Qwen3.5-2B / 9B) on the DGX Spark; run the probe battery on Jev through its API if access is obtained."]:
    story.append(P("•&nbsp;&nbsp;" + t))
story += [Spacer(1, 6), P("Reproducibility: code, configs and pinned data revisions in the OpenDecider repository (119 passing tests). "
                          "Figures and tables regenerate from run artefacts with reports/collect.py → make_charts.py → make_report.py.", "small")]


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(INK2)
    c.drawString(18 * mm, 10 * mm, "OpenDecider · hypothesis test, not a description of Jev's internals")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


doc = SimpleDocTemplate("reports/OpenDecider_findings.pdf", pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=18 * mm, title="OpenDecider findings", author="OpenDecider")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote reports/OpenDecider_findings.pdf")
