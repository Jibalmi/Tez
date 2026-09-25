"""The probe lab, one chart per experiment on cached hidden states (docs/figures/panels/lab_*.png), and the same charts
on one page (docs/figures/tez_lab.png / .pdf, the paper's figure).

    python experiments/make_lab_figure.py

Reads results/probe_lab_*.json and the GPU follow-ups (multi-question pass, cross-lingual probes, truth probe, pruned
GGUFs, decide-then-bind, the collapsing voice bound). A panel whose file is missing is skipped, never drawn with
placeholder numbers. Each panel is a Panel entry (experiments/figpanels.py); make_dashboard.py draws the rest.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.ticker
import numpy as np

import figpanels as fp
from figpanels import (HATCH, INK, INK2, LAYA_ML, LAYA_TD, NEUTRAL, NEUTRAL2, SURFACE, TEZ, TEZ_DEEP,
                       TEZ_INK, TEZ_LIGHT, J, Panel, bar, dot, f3, grid, hlab, line, patch, pct, refline, strip, vlab)

LAYA_TD_ACC, JEV_TD = 0.766, 0.727         # typed-decisions: laya-typed-decisions (fine-tuned), Jev (published)
WF_NAME = {"agent_trace_observability": "agent-trace\nobservability", "customer_service": "customer\nservice",
           "invoice_processing": "invoice\nprocessing", "security_incidents": "security\nincidents"}
SETUP = ("Frozen Qwen3.5-4B (bf16) unless stated; one cache of every layer's last-token state for the 8,000 "
         "typed-decisions prompts; probes fitted on the train split, scored on the 2,000 test decisions.")
SETUP_SERVED = ("Depth-pruned Qwen3.5-4B Q8_0 GGUFs (the first N of 32 blocks) served by llama.cpp b11100 on the RTX 5080 "
                "laptop; typed-decisions test split, 2,000 decisions.")
SETUP_SUITE = ("Zero-shot letters of the depth-pruned Qwen3.5-4B GGUFs on Laya's public suite: the same rows as "
               "BENCHMARKS.md §1, llama.cpp b11100, prompt caching off.")
SETUP_XL = ("A logistic probe on the frozen Qwen3.5-4B, trained on 2,000 English MASSIVE rows, scored with Laya's "
            "20-option protocol on 100 rows per language (the same rows as BENCHMARKS.md §1).")
SETUP_VOICE = ("220 spoken commands, 16 actions; Gemma 4 12B Q8_0 letters on llama-server, streamed word by word; commit "
               "rules chosen on one half of the utterances, scored on the other.")
SRC = "BENCHMARKS.md §4c"
TASK_NAME = {"ag_news": "AG News", "emotion": "DAIR\nEmotion", "banking77": "Banking77", "sst5": "SST-5", "boolq": "BoolQ",
             "prompt_injections": "prompt-\ninjections", "typed_decisions": "typed-\ndecisions"}


def load():
    D = SimpleNamespace(S=J("results/h2h/summary.json", {}))
    for name, rel in (("lens4", "results/probe_lab_lens_qwen35-4b.json"), ("lens12", "results/probe_lab_lens_gemma4-12b-nf4.json"),
                      ("intr", "results/probe_lab_intrinsic_qwen35-4b.json"), ("sweep", "results/hidden_probe_sweep_qwen35-4b.json"),
                      ("any", "results/probe_lab_anytime_qwen35-4b.json"), ("var", "results/probe_lab_variants_qwen35-4b.json"),
                      ("w2s", "results/probe_lab_w2s_qwen35-4b.json"), ("lm", "results/probe_lab_labelmodel_qwen35-4b.json"),
                      ("stack", "results/probe_lab_stack_qwen35-4b.json"), ("typ", "results/probe_lab_typical_qwen35-4b.json"),
                      ("fs", "results/probe_lab_fewshot_qwen35-4b.json"), ("prior", "results/probe_lab_prior_qwen35-4b.json"),
                      ("cold", "results/probe_lab_online2_qwen35-4b.json"), ("cs", "results/probe_lab_confsel_qwen35-4b.json"),
                      ("uni", "results/probe_lab_universal_qwen35-4b.json"), ("truth", "results/probe_truth_qwen35-4b.json"),
                      ("wfb", "results/probe_lab_workflow_baselines.json"), ("mq", "results/probe_multiq_qwen35-4b.json"),
                      ("xl", "results/probe_crosslingual_qwen35-4b.json"), ("pruned", "results/pruned_server_qwen35-4b.json"),
                      ("lat", "results/latency_breakdown.json"), ("cut24", "results/h2h_q4b_L24/summary.json"),
                      ("cut29", "results/h2h_q4b_L29/summary.json"), ("cut32", "results/h2h_q4b_L32/summary.json"),
                      ("ddm_p", "results/voice_ddm_partial_last.json"), ("ddm_f", "results/voice_ddm_final_last.json"),
                      ("order", "results/probe_order_qwen35-4b.json")):
        setattr(D, name, J(rel))
    return D


def layers(d):
    return sorted(int(k) for k in d)


def refs(ax, laya=True, jev=True, where=0.99, ha="right"):
    if laya:
        refline(ax, LAYA_TD_ACC, f"laya-typed-decisions (fine-tuned) {f3(LAYA_TD_ACC)}", color=LAYA_TD, x=where, ha=ha)
    if jev:
        refline(ax, JEV_TD, f"Jev (published) {f3(JEV_TD)}", x=where, ha=ha, va="top", ls=(0, (1, 2)))


def logx(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])


def finish(ax, xlabel=None, ylabel=None, both=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    grid(ax)
    if both:
        grid(ax, "x")
    strip(ax, keep=("bottom", "left"))


# ---------------------------------------------------------------- a: logit lens
def draw_lens(ctx, D):
    ax = ctx.ax()
    L4 = D.lens4["logit_lens"]
    ax.plot([k / 32 for k in layers(L4)], [L4[str(k)] for k in layers(L4)], "-o", color=TEZ, lw=2, ms=4.5, mec=SURFACE,
            mew=0.8, zorder=3)
    if D.lens12:
        L12 = D.lens12["logit_lens"]
        ax.plot([k / 48 for k in layers(L12)], [L12[str(k)] for k in layers(L12)], "-s", color=TEZ_DEEP, lw=2, ms=4,
                mec=SURFACE, mew=0.8, zorder=3)
    b, top, full = lens_facts(D)
    ax.plot([b / 32], [L4[str(b)]], "o", ms=10, color=TEZ, mec=SURFACE, mew=1.4, zorder=4)
    ax.annotate(f"4B at its best, layer {b}: {f3(L4[str(b)])}", (b / 32, L4[str(b)]), xytext=(0.62, 0.82),
                textcoords="axes fraction", ha="right", va="center", fontsize=8.8, color=TEZ_INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEZ_INK, lw=0.8, shrinkA=4, shrinkB=6))
    ax.annotate(f"top layer: {f3(top)}", (1.0, top), xytext=(-8, -4), textcoords="offset points", ha="right", va="top",
                fontsize=8.8, color=TEZ_INK)
    refline(ax, full, f"12B letters at full depth (llama.cpp) {f3(full)}", color=TEZ_DEEP, x=0.01, ha="left")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(0.2, 0.8)
    finish(ax, "depth read (share of the model's layers)", "zero-shot letter accuracy")
    ctx.legend(ax, [line(color=TEZ, label="Qwen3.5-4B, 32 layers"),
                    line(color=TEZ_DEEP, marker="s", label="Gemma 4 12B (NF4), layers 0–40 of 48")], ncol=2,
               note="the letters read through the final norm and the output head at every layer")


def lens_facts(D):
    L4 = D.lens4["logit_lens"]
    b = max(layers(L4), key=lambda k: L4[str(k)])
    return b, L4[str(max(layers(L4)))], D.S["typed_decisions"]["tez"]["accuracy"]


def finding_lens(D):
    b, top, full = lens_facts(D)
    best = D.lens4["logit_lens"][str(b)]
    return (f"Read through the logit lens, the 4B's letters score best at layer {b} ({f3(best)}), "
            f"{100 * (best - top):.1f} points above its top layer; the 12B's letters rise late and keep improving to full "
            f"depth ({f3(full)})")


# ---------------------------------------------------------------- b: the layer without labels
def id_band(D, tol=0.1):
    I = {int(k): v["twonn_pooled"] for k, v in D.intr["layers"].items() if v["twonn_pooled"] == v["twonn_pooled"]}
    peak = max(I, key=I.get)
    after = {k: v for k, v in I.items() if k > peak}          # the minimum after the peak, not the first layers
    lo = min(after.values())
    band = [k for k, v in after.items() if v <= lo + tol]
    return I, min(after, key=after.get), min(band), max(band)


def plateau(sw, tol=0.004):
    ls = layers(sw["layer_sweep"])
    best = max(sw["layer_sweep"][str(k)]["acc"] for k in ls)
    on = [k for k in ls if sw["layer_sweep"][str(k)]["acc"] >= best - tol]
    return best, min(on), max(on)


def draw_id(ctx, D):
    top, bottom = ctx.stack((1, 1), gap=0.5)
    I, kmin, a, b = id_band(D)
    ks = sorted(I)
    top.plot(ks, [I[k] for k in ks], "-o", color=NEUTRAL, lw=2, ms=4, mec=SURFACE, mew=0.8)
    top.plot([kmin], [I[kmin]], "o", ms=9, color=NEUTRAL, mec=SURFACE, mew=1.2)
    top.annotate(f"minimum at layer {kmin}", (kmin, I[kmin]), xytext=(0, -12), textcoords="offset points", ha="center",
                 va="top", fontsize=8.8, color=INK2)
    SW = D.sweep["layer_sweep"]
    ls = layers(SW)
    bottom.plot(ls, [SW[str(k)]["acc"] for k in ls], "-o", color=TEZ, lw=2, ms=4.5, mec=SURFACE, mew=0.8)
    best, pa, pb = plateau(D.sweep)
    for ax in (top, bottom):
        ax.axvspan(a, b, color=TEZ, alpha=0.08, lw=0)
        finish(ax, both=False)
    bottom.annotate(f"probe plateau, layers {pa}–{pb} ({f3(best)} at best)", ((pa + pb) / 2, 0.45), ha="center",
                    fontsize=8.8, color=TEZ_INK)
    top.set_ylabel("intrinsic dimension", fontsize=9.5)
    bottom.set_ylabel("probe accuracy", fontsize=9.5)
    bottom.set_ylim(0.4, 0.85)
    bottom.set_xlabel("layer (of 32)", fontsize=10)
    ctx.subtitle(top, "intrinsic dimension of unlabelled states (TwoNN): no labels needed")
    ctx.subtitle(bottom, "probe accuracy: needs labels")
    ctx.legend(top, note=f"shaded: layers {a}–{b}, where the intrinsic dimension is within 0.1 of its minimum after its "
                         f"peak")


def finding_id(D):
    I, kmin, a, b = id_band(D)
    best, pa, pb = plateau(D.sweep)
    return (f"The layer can be chosen without labels: past its peak, the intrinsic dimension of unlabelled states bottoms "
            f"out at layers {a}–{b}, inside the probe's accuracy plateau (layers {pa}–{pb})")


# ---------------------------------------------------------------- c: anytime depth
def anytime_facts(D):
    A = D.any
    fixed = {int(k): v for k, v in A["fixed"].items()}
    target = max(fixed.values()) - 0.005
    g = min((v for v in A["gate"].values() if v["acc"] >= target), key=lambda v: v["mean_depth"])
    L = min(fixed, key=lambda k: abs(k - g["mean_depth"]))
    return g, L, fixed[L]


def draw_anytime(ctx, D):
    ax = ctx.ax()
    A = D.any
    g = A["grid"]
    ax.plot(g, [A["fixed"][str(k)] for k in g], "-o", color=NEUTRAL2, lw=2.2, ms=6, mec=SURFACE, mew=1)
    for key, c, mk in (("gate", TEZ, "s"), ("accumulate", TEZ_DEEP, "^")):
        pts = sorted((v["mean_depth"], v["acc"]) for v in A[key].values())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=c, lw=2, ms=6, mec=SURFACE, mew=1)
    ax.set_ylim(0.55, 0.82)
    finish(ax, "mean layers used (of 32)", "accuracy")
    ctx.legend(ax, [line(color=NEUTRAL2, label="fixed depth"), line(color=TEZ, marker="s", label="stop when max p ≥ τ"),
                    line(color=TEZ_DEEP, marker="^", label="stop when the summed log-odds ≥ θ")], ncol=3, fs=8.8,
               note="per-layer probes on gold labels, grid of layers 12–28; each point on an adaptive curve is one threshold")


def finding_anytime(D):
    g, L, fx = anytime_facts(D)
    return (f"An adaptive stop only ties a fixed cut: stopping when the probe is sure reaches {f3(g['acc'])} at "
            f"{g['mean_depth']:.1f} layers on average; a fixed cut at {L} layers reaches {f3(fx)}")


# ---------------------------------------------------------------- d: code size
def draw_code(ctx, D):
    ax = ctx.ax()
    cs = D.var["code_size"]
    ms = layers(cs["pca"])
    ax.plot(ms, [cs["random"][str(m)] for m in ms], "-s", color=NEUTRAL2, lw=2, ms=6, mec=SURFACE, mew=1)
    ax.plot(ms, [cs["pca"][str(m)] for m in ms], "-o", color=TEZ, lw=2.2, ms=7, mec=SURFACE, mew=1)
    ax.annotate(f"64 numbers: {f3(cs['pca']['64'])}", (64, cs["pca"]["64"]), xytext=(6, -14), textcoords="offset points",
                fontsize=8.8, color=TEZ_INK, fontweight="bold")
    refline(ax, cs["full"], f"all {cs['dim']:,} numbers: {f3(cs['full'])}", color=INK2, x=0.01, ha="left", ls="-")
    refline(ax, JEV_TD, f"Jev (published) {f3(JEV_TD)}", x=0.99, va="top", ls=(0, (1, 2)))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_ylim(0.55, 0.82)
    finish(ax, "numbers kept per state", "probe accuracy")
    ctx.legend(ax, [line(color=TEZ, label="one PCA learned from unlabelled states, shared by the 20 questions"),
                    line(color=NEUTRAL2, marker="s", label="random projection")], ncol=1, note="layer 26")


def finding_code(D):
    cs = D.var["code_size"]
    return (f"A decision needs 64 numbers: one PCA learned from unlabelled states keeps {f3(cs['pca']['64'])} of the full "
            f"{f3(cs['full'])} with 64 of the state's {cs['dim']:,} numbers")


# ---------------------------------------------------------------- e: probe families
def draw_families(ctx, D):
    ax = ctx.ax()
    fam = D.var["families"]
    items = sorted(((n, v["acc"]) for n, v in fam.items()), key=lambda t: -t[1])
    best = items[0][1]
    for i, (n, v) in enumerate(items):
        c = TEZ if n.startswith("logreg C=0.5") else TEZ_LIGHT if n.startswith("logreg") else NEUTRAL2
        bar(ax, i, v, 0.66, horizontal=True, color=c)
        hlab(ax, v, i, f3(v), fs=9, color=INK if v >= best - 1e-9 else INK2, weight="bold" if c == TEZ else "normal")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items], fontsize=9.2)
    ax.set_ylim(len(items) - 0.4, -0.6)
    ax.set_xlim(0.6, 0.83)
    finish(ax, "probe accuracy, layer 26", both=False)
    grid(ax, "x")
    ctx.legend(ax, [patch(color=TEZ, label="plain logistic regression (the runtime's probe)"),
                    patch(color=TEZ_LIGHT, label="other logistic variants"), patch(color=NEUTRAL2, label="other families")],
               ncol=2, fs=8.8)


def finding_families(D):
    fam = D.var["families"]
    plain = fam["logreg C=0.5"]["acc"]
    beat = [n for n, v in fam.items() if v["acc"] > plain + 0.001]
    return (f"On layer 26, plain logistic regression ({f3(plain)}) is as good as any of the {len(fam)} probe families "
            f"tried" + ("" if not beat else f"; only {', '.join(beat)} edges past it")
            + f"; nearest centroid scores {f3(fam['nearest centroid (cosine)']['acc'])}")


# ---------------------------------------------------------------- f: no labels
def label_free_items(D):
    W, LM, ST = D.w2s, D.lm, D.stack
    items = [("gold labels (reference)", W["gold"]["26"], dict(color=NEUTRAL), True)]
    for tname, lab in (("gemma12b", "12B"), ("self", "4B, its own")):
        T = W["teachers"].get(tname)
        if T:
            items.append((f"teacher: {lab} letters", T["teacher_test_acc"], dict(color=NEUTRAL2), False))
            best = max(((L, n, v["acc"]) for L, d in T["variants"].items() for n, v in d.items()), key=lambda t: t[2])
            items.append((f"probe taught by the {lab} letters", best[2], dict(color=TEZ, hatch=HATCH), False))
    if W["teachers"].get("agree"):
        v = max(W["teachers"]["agree"]["variants"].values(), key=lambda d: d["agreement-filtered"]["acc"])["agreement-filtered"]["acc"]
        items.append(("probe taught where 4B and 12B agree", v, dict(color=TEZ, hatch=HATCH), False))
    if LM:
        best = max(LM["combos"].values(), key=lambda v: v["label_model_test_acc"])
        items.append(("Dawid–Skene label model over readouts", best["label_model_test_acc"], dict(color=NEUTRAL2), False))
    if ST:
        items.append(("gold probe + 12B letters, stacked", ST["pools"]["probe L26 + 12B letters"]["stacked_logreg"],
                      dict(color=TEZ, hatch=HATCH), True))
    return items


def draw_label_free(ctx, D):
    ax = ctx.ax()
    items = label_free_items(D)
    for i, (lab, v, st, gold) in enumerate(items):
        bar(ax, i, v, 0.66, horizontal=True, **st)
        hlab(ax, v, i, f3(v), fs=9, color=INK if gold else INK2, weight="bold" if gold else "normal")
    refline(ax, JEV_TD, f"Jev (published) {f3(JEV_TD)}", horizontal=False, x=0.99, ls=(0, (1, 2)))
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=9.2)
    ax.set_ylim(len(items) - 0.4, -0.6)
    ax.set_xlim(0.4, 0.86)
    finish(ax, "accuracy", both=False)
    grid(ax, "x")
    ctx.legend(ax, [patch(color=NEUTRAL, label="gold labels"), patch(color=NEUTRAL2, label="teacher or label model"),
                    patch(color=TEZ, hatch=HATCH, label="probe")], ncol=3,
               note="teachers' answers on the train split used as labels in place of gold ones")


def finding_label_free(D):
    W = D.w2s
    T = W["teachers"]["gemma12b"]
    best = max(v["acc"] for d in T["variants"].values() for v in d.values())
    return (f"Without labels the probe only matches its teacher: taught by the 12B's zero-shot letters it scores "
            f"{f3(best)} against the teacher's {f3(T['teacher_test_acc'])}; gold labels give {f3(W['gold']['26'])}")


# ---------------------------------------------------------------- g: few labels
def draw_few(ctx, D):
    ax = ctx.ax()
    TY, FS, PRI = D.typ, D.fs, D.prior
    ns = layers(TY["n"])
    ax.errorbar(ns, [TY["n"][str(n)]["random"]["mean"] for n in ns], yerr=[TY["n"][str(n)]["random"]["std"] for n in ns],
                fmt="-o", ms=6, capsize=3, color=NEUTRAL2, lw=2, mec=SURFACE, mew=1)
    ax.errorbar(ns, [TY["n"][str(n)]["typical"]["mean"] for n in ns], yerr=[TY["n"][str(n)]["typical"]["std"] for n in ns],
                fmt="-s", ms=6, capsize=3, color=TEZ_LIGHT, lw=2, mec=SURFACE, mew=1)
    if FS:
        fn = layers(FS["n"])
        for m, c, mk in (("LDA (unlabelled covariance)", NEUTRAL, "v"), ("PCA-64 (unlabelled) + logreg", NEUTRAL, "^")):
            ax.plot(fn, [FS["n"][str(n)][m]["mean"] for n in fn], "--" + mk, ms=6, color=c, lw=1.6)
    if PRI:
        pn = [n for n in layers(PRI["n"]) if n > 0]
        ys = [PRI["n"][str(n)]["typical"]["mix n0=10"] for n in pn]
        ax.plot(pn, ys, "-D", ms=7, color=TEZ, lw=2.4, mec=SURFACE, mew=1, zorder=4)
        ax.annotate(f3(ys[-1]), (pn[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK, fontweight="bold")
        refline(ax, PRI["zero_shot_12b"], f"12B zero-shot, no labels {f3(PRI['zero_shot_12b'])}", color=TEZ_DEEP, x=0.01,
                ha="left", va="top")
    refline(ax, LAYA_TD_ACC, f"laya-typed-decisions (fine-tuned on all 300) {f3(LAYA_TD_ACC)}", color=LAYA_TD, x=0.01,
            ha="left")
    logx(ax, [5, 10, 25, 50])
    ax.set_ylim(0.4, 0.8)
    finish(ax, "labelled rows per question", "accuracy")
    h = [line(color=TEZ, marker="D", label="typical rows + the 12B's zero-shot answers as a prior"),
         line(color=TEZ_LIGHT, marker="s", label="most typical rows labelled first"),
         line(color=NEUTRAL2, label="random rows labelled"),
         line(color=NEUTRAL, marker="v", ls="--", label="LDA with an unlabelled covariance"),
         line(color=NEUTRAL, marker="^", ls="--", label="PCA-64 from unlabelled states + logistic")]
    ctx.legend(ax, h, ncol=2, fs=8.8, note="layer 26, 3 seeds (± one s.d.)")


def finding_few(D):
    PRI = D.prior
    v50 = PRI["n"]["50"]["typical"]["mix n0=10"]
    return (f"With few labels, label the most typical rows first and keep the 12B's zero-shot answers as a prior: 50 per "
            f"question reach {f3(v50)}, the level of laya-typed-decisions fine-tuned on all 300 ({f3(LAYA_TD_ACC)})")


# ---------------------------------------------------------------- h: cold start
POLICY = {"escalate-only": (NEUTRAL, "o", "learn from escalations only"),
          "escalate+audit": (TEZ, "o", "escalations + a random audit slice"),
          "audit-train": (TEZ_LIGHT, "s", "audit slice only"),
          "typical-seed": (TEZ_DEEP, "D", "typical seed + escalations")}


def draw_cold(ctx, D):
    ax = ctx.ax()
    O2 = D.cold
    for key, r in O2["runs"].items():
        pol = key.split("_")[1]
        c, mk, _ = POLICY.get(pol, (NEUTRAL2, "o", pol))
        ax.plot(r["labels"], r["random_same_budget_test_acc"], "x", color=NEUTRAL2, ms=8, mew=1.8, zorder=2)
        ax.plot(r["labels"], r["test_acc"], mk, color=c, ms=9, mec=SURFACE, mew=1.2, zorder=3)
    refline(ax, O2["zero_shot_test_acc"], f"start: the 12B's zero-shot answers {f3(O2['zero_shot_test_acc'])}",
            color=TEZ_DEEP, x=0.01, ha="left")
    refline(ax, LAYA_TD_ACC, f"laya-typed-decisions {f3(LAYA_TD_ACC)}", color=LAYA_TD, x=0.99)
    logx(ax, [300, 1000, 3000])
    ax.set_ylim(0.69, 0.81)
    finish(ax, "human labels spent", "held-out accuracy of the probe")
    h = [dot(c, lab, marker=mk) for c, mk, lab in POLICY.values()]
    h.append(dot(NEUTRAL2, "the same number of random labels", marker="x"))
    ctx.legend(ax, h, ncol=2, fs=8.8,
               note="a stream of the 6,000 train decisions; escalate below confidence τ (0.6, 0.8, 0.9); layer 26")


def cold_facts(D):
    runs = D.cold["runs"]
    pick = lambda pol, eps: max((r for k, r in runs.items() if k.split("_")[1] == pol and k.endswith(eps)),  # noqa: E731
                                key=lambda r: r["labels"])
    return pick("escalate-only", "eps0.0"), pick("escalate+audit", "eps0.1")


def finding_cold(D):
    e, a = cold_facts(D)
    return (f"Cold start: learning only from escalations trails the same number of random labels "
            f"({f3(e['test_acc'])} against {f3(e['random_same_budget_test_acc'])} at {e['labels']:,} labels); with a random "
            f"audit slice it keeps pace ({f3(a['test_acc'])} against {f3(a['random_same_budget_test_acc'])} at "
            f"{a['labels']:,})")


# ---------------------------------------------------------------- i: conformal selection
CS_STYLE = [(TEZ, "o", "-"), (TEZ_DEEP, "s", "-"), (NEUTRAL2, "^", "--")]


def draw_confsel(ctx, D):
    ax = ctx.ax()
    CS = D.cs
    alphas = [0.05, 0.1, 0.2]
    h = []
    for (name, R), (c, mk, ls) in zip(CS.items(), CS_STYLE):
        ys = [R[f"alpha{a}"]["acted"] * 100 for a in alphas]
        ax.plot([a * 100 for a in alphas], ys, ls + mk, color=c, lw=2, ms=7, mec=SURFACE, mew=1)
        for a, y in zip(alphas, ys):
            ax.annotate(f"wrong {100 * R[f'alpha{a}']['realised_error_among_acted']:.1f} %", (a * 100, y), xytext=(8, -2),
                        textcoords="offset points", va="top", fontsize=8.2, color=INK2)
        h.append(line(color=c, marker=mk, ls=ls, label=name))
    ax.set_xticks([5, 10, 20])
    ax.set_xticklabels(["5 %", "10 %", "20 %"])
    ax.set_xlim(3, 26)
    ax.set_ylim(-3, 105)
    finish(ax, "error bound chosen (share of acted decisions allowed to be wrong)", "% of decisions acted on")
    ctx.legend(ax, h, ncol=1, fs=8.8,
               note="conformal p-values with Benjamini–Hochberg, 50 random half/half calibration splits; labels: the "
                    "realised error among acted decisions")


def finding_confsel(D):
    R = D.cs["4B probe L26 (gold labels)"]["alpha0.05"]
    L = D.cs["12B letters (zero-shot)"]["alpha0.05"]
    return (f"A guaranteed error rate: allowed 5 % wrong, the probe acts on {pct(R['acted'])} of decisions and is wrong on "
            f"{100 * R['realised_error_among_acted']:.1f} % of them; the 12B's zero-shot letters act on {pct(L['acted'])}")


# ---------------------------------------------------------------- j: unseen workflows
def unseen_series(D):
    U, TR, WB = D.uni, D.truth, D.wfb
    wfs = list(U["baselines"])
    out = [("4B letters (zero-shot)", [U["baselines"][f]["letters_4b"] for f in wfs], dict(color=NEUTRAL2))]
    bl = max(U["layers"], key=lambda k: U["layers"][k]["lowo_mean"])
    out.append((f"a shared letter probe (layer {bl})", [U["layers"][bl]["lowo"][f]["acc"] for f in wfs],
                dict(color=TEZ_LIGHT, hatch=HATCH)))
    if TR:
        k = max(TR["lowo"], key=lambda kk: TR["lowo"][kk]["mean"])
        out.append((f"'is this answer right?' probe ({k.replace('L', 'layer ')})", [TR["lowo"][k][f] for f in wfs],
                    dict(color=TEZ, hatch=HATCH)))
    if WB:
        out.append(("12B letters (zero-shot)", [WB["letters_12b_by_workflow"][f] for f in wfs], dict(color=TEZ_DEEP)))
    return wfs, out


def draw_unseen(ctx, D):
    ax = ctx.ax()
    wfs, series = unseen_series(D)
    w = 0.8 / len(series)
    for j, (lab, vals, st) in enumerate(series):
        for i, v in enumerate(vals):
            x = i - 0.4 + w / 2 + j * w
            bar(ax, x, v, w * 0.92, **st)
            if "right" in lab:
                vlab(ax, x, v, f3(v), fs=8.4, color=INK, weight="bold")
    ax.set_xticks(range(len(wfs)))
    ax.set_xticklabels([WF_NAME.get(f, f) for f in wfs], fontsize=9.3)
    ax.set_ylim(0, 1.0)
    finish(ax, None, "accuracy on the held-out workflow", both=False)
    ctx.legend(ax, [patch(label=lab, **st) for lab, _, st in series], ncol=2, fs=8.8,
               note="each workflow scored by a head trained on the other three")


def finding_unseen(D):
    wfs, series = unseen_series(D)
    m = {lab: float(np.mean(v)) for lab, v, _ in series}
    truth = next(v for k, v in m.items() if "right" in k)
    return (f"On a workflow it has never seen, one 'is this answer right?' probe trained on the other three scores "
            f"{f3(truth)} on average, against {f3(m['4B letters (zero-shot)'])} for the 4B's own letters and "
            f"{f3(m['12B letters (zero-shot)'])} for the 12B's")


# ---------------------------------------------------------------- k: many decisions from one pass
MQ = [("per_question_prompt", NEUTRAL2, "o", "one prompt per question (5 passes per row)"),
      ("A_all_questions_one_pass", TEZ, "s", "all questions after one state, one pass"),
      ("B_state_only", TEZ_DEEP, "^", "the state alone, one pass")]


def draw_multiq(ctx, D):
    ax = ctx.ax()
    M = D.mq
    ls = layers(M["probe"])
    for key, c, mk, _ in MQ:
        ax.plot(ls, [M["probe"][str(k)][key] for k in ls], "-" + mk, color=c, lw=2, ms=6, mec=SURFACE, mew=1)
    refs(ax, where=0.01, ha="left")
    t = M["tokens_test"]
    ax.set_xticks(ls)
    ax.set_ylim(0.55, 0.83)
    finish(ax, "layer (of 32)", "probe accuracy")
    ctx.legend(ax, [line(color=c, marker=mk, label=lab) for _, c, mk, lab in MQ], ncol=1, fs=8.8,
               note=f"prompt tokens for the 400 test rows: {t['per_question_prompts']:,} one prompt per question · "
                    f"{t['A_all_questions_one_pass']:,} all in one pass · {t['B_state_only']:,} the state alone")


def finding_multiq(D):
    M = D.mq
    t = M["tokens_test"]
    per = max(v["per_question_prompt"] for v in M["probe"].values())
    one = max(v["A_all_questions_one_pass"] for v in M["probe"].values())
    ratio = t["per_question_prompts"] / t["A_all_questions_one_pass"]
    return (f"Five questions read after one state in one pass cost {ratio:.1f}× fewer tokens than one prompt per "
            f"question, for {100 * (per - one):.1f} points ({f3(one)} against {f3(per)} at the best layer)")


# ---------------------------------------------------------------- l: cross-lingual
def xl_facts(D):
    X = D.xl
    bl = max(X["layers"], key=lambda k: X["layers"][k]["macro_acc20"])
    R = X["layers"][bl]
    langs = [k for k in R if isinstance(R[k], dict)]
    return bl, R, langs


def draw_xl(ctx, D):
    ax = ctx.ax()
    bl, R, langs = xl_facts(D)
    S = D.S
    series = [(f"4B probe trained on English only (layer {bl})", [R[k]["acc20"] for k in langs], dict(color=TEZ, hatch=HATCH)),
              ("12B zero-shot letters", [S.get(f"massive:{k}", {}).get("tez", {}).get("accuracy", np.nan) for k in langs],
               dict(color=TEZ_DEEP)),
              ("laya-multilingual", [S.get(f"massive:{k}", {}).get("laya-ml", {}).get("accuracy", np.nan) for k in langs],
               dict(color=LAYA_ML))]
    w = 0.8 / len(series)
    for j, (_, vals, st) in enumerate(series):
        for i, v in enumerate(vals):
            if v == v:
                bar(ax, i - 0.4 + w / 2 + j * w, v, w * 0.92, **st)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    finish(ax, "MASSIVE language (100 test rows each, 20 options)", "accuracy", both=False)
    ctx.legend(ax, [patch(label=lab, **st) for lab, _, st in series], ncol=2, fs=8.8,
               note="the probe is trained on 2,000 English rows and used unchanged in every language")


def finding_xl(D):
    bl, R, langs = xl_facts(D)
    ml = float(np.mean([D.S[f"massive:{k}"]["laya-ml"]["accuracy"] for k in langs]))
    tz = float(np.mean([D.S[f"massive:{k}"]["tez"]["accuracy"] for k in langs]))
    return (f"Trained on English only, a MASSIVE probe scores {f3(R['macro_acc20'])} macro over {len(langs)} languages, "
            f"against {f3(ml)} for laya-multilingual and {f3(tz)} for the 12B's zero-shot letters")


# ---------------------------------------------------------------- m: pruned GGUFs, served
NM = {"L20": "Qwen3.5-4B, 20 of 32 blocks", "L24": "Qwen3.5-4B, 24 of 32 blocks", "L29": "Qwen3.5-4B, 29 of 32 blocks",
      "L32": "Qwen3.5-4B, all 32 blocks"}
LETTERS_12B = 0.705                     # the 12B's letters in the same latency runs (BENCHMARKS.md §4c)


def pruned_rows(D):
    BD = D.lat or {}
    out = []
    for name, r in D.pruned.items():
        b = BD.get(NM.get(name, ""), {})
        out.append((int(name[1:]), r["probe_acc"], b.get("embedding (probe readout)", {}).get("p50", r["ms_embed_p50"]),
                    r["letters_acc"], b.get("completion, n_probs 200", {}).get("p50", r["ms_letters_p50"])))
    return sorted(out)


def draw_pruned(ctx, D):
    ax = ctx.ax()
    rows = pruned_rows(D)
    ax.plot([r[2] for r in rows], [r[1] for r in rows], "-", color=TEZ, lw=1, alpha=0.5)
    ax.plot([r[4] for r in rows], [r[3] for r in rows], "-", color=NEUTRAL2, lw=1, alpha=0.7)
    for n, pa, pm, la, lm in rows:
        ax.plot(pm, pa, "o", color=TEZ, ms=10, mec=SURFACE, mew=1.2, zorder=3)
        ax.annotate(f"{n}", (pm, pa), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.8,
                    color=TEZ_INK, fontweight="bold" if n == 24 else "normal")
        ax.plot(lm, la, "s", color=NEUTRAL2, ms=9, mec=SURFACE, mew=1.2, zorder=3)
        ax.annotate(f"{n}", (lm, la), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.6, color=INK2)
    g12 = (D.lat or {}).get("Gemma 4 12B Q8")
    if g12:
        x = g12["completion, n_probs 200"]["p50"]
        ax.plot(x, LETTERS_12B, "D", color=TEZ_DEEP, ms=9, mec=SURFACE, mew=1.2)
        ax.annotate(f"12B letters\n{f3(LETTERS_12B)} · {x:.0f} ms", (x, LETTERS_12B), xytext=(-8, 6),
                    textcoords="offset points", ha="right", fontsize=8.6, color=TEZ_DEEP)
    ax.set_ylim(0.2, 0.86)
    ax.set_xlim(40, 220)
    finish(ax, "ms per decision (p50), llama.cpp, dedicated runs, prompt caching off", "accuracy on typed-decisions")
    ctx.legend(ax, [dot(TEZ, "probe on the served state (/embedding)"),
                    dot(NEUTRAL2, "zero-shot letters (the logit lens at the cut)", marker="s"),
                    dot(TEZ_DEEP, "Gemma 4 12B letters", marker="D")], ncol=2, fs=8.8,
               note="labels: blocks kept of the 4B's 32 (experiments/gguf_truncate.py)")


def finding_pruned(D):
    rows = {r[0]: r for r in pruned_rows(D)}
    a, b = rows[24], rows[32]
    return (f"Cut the served 4B where the decision is made: with 24 of its 32 blocks its probe scores {f3(a[1])} in "
            f"{a[2]:.0f} ms, against {f3(b[1])} in {b[2]:.0f} ms for the whole model")


# ---------------------------------------------------------------- n: cut 4B on the public suite
def cut_tasks(D):
    A32, A29 = D.cut32, D.cut29
    return [t for t in A32 if t in A29 and "tez" in A32[t] and not t.startswith(("massive:", "xnli:"))] + ["MASSIVE (11)", "XNLI (10)"]


def cut_val(S, t):
    if t == "MASSIVE (11)":
        return float(np.mean([S[k]["tez"]["accuracy"] for k in S if k.startswith("massive:")]))
    if t == "XNLI (10)":
        return float(np.mean([S[k]["tez"]["accuracy"] for k in S if k.startswith("xnli:")]))
    return S[t]["tez"]["accuracy"]


def draw_cut(ctx, D):
    ax = ctx.ax()
    tasks = cut_tasks(D)
    cols = [("4B, all 32 blocks", D.cut32, NEUTRAL2), ("cut to 29", D.cut29, TEZ_LIGHT)] + \
        ([("cut to 24", D.cut24, TEZ)] if D.cut24 else [])
    w = 0.8 / len(cols)
    for j, (_, Sx, c) in enumerate(cols):
        for i, t in enumerate(tasks):
            bar(ax, i - 0.4 + w / 2 + j * w, cut_val(Sx, t), w * 0.92, color=c)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_NAME.get(t, t.replace(" (", "\n(")) for t in tasks], fontsize=8.8)
    ax.set_ylim(0, 1.0)
    finish(ax, None, "zero-shot letter accuracy", both=False)
    ctx.legend(ax, [patch(color=c, label=lab) for lab, _, c in cols], ncol=3,
               note="Laya's public suite, same rows as §1, prompt caching off · MASSIVE and XNLI: mean over the languages")


def cut_mean(D, S):
    keys = [k for k in D.cut32 if k in S and "tez" in D.cut32[k]]
    return 100 * float(np.mean([S[k]["tez"]["accuracy"] - D.cut32[k]["tez"]["accuracy"] for k in keys])), len(keys)


def finding_cut(D):
    d29, n = cut_mean(D, D.cut29)
    out = f"Cut to 29 blocks, the 4B's zero-shot letters lose {-d29:.1f} points on average over Laya's public suite ({n} entries)"
    if D.cut24:
        d24, _ = cut_mean(D, D.cut24)
        out += f", cut to 24 they lose {-d24:.1f}: cut for probes, validate per task for letters"
    return out


# ---------------------------------------------------------------- o: voice commit bound
DDM = [("default_on_eval", NEUTRAL2, "o", "default (τ 0.9, stability 2)"), ("fixed", TEZ_LIGHT, "s", "best retuned fixed threshold"),
       ("collapsing", TEZ, "D", "best collapsing bound")]


def draw_ddm(ctx, D):
    axes = ctx.row((1, 1), gap=0.5)
    for ax, (R, title) in zip(axes, ((D.ddm_p, "partial-transcript log"), (D.ddm_f, "final-transcript log"))):
        if not R:
            continue
        hs = R["held_out_summary"]
        seen = []
        for key, c, mk, _ in DDM:
            r = hs[key]
            y = r["harmful"] + r["oos_false"]
            x = r["mean_first_word"]
            ax.plot(x, y, mk, color=c, ms=11 if not seen else 9, mec=SURFACE, mew=1.3, zorder=3 + len(seen))
            near = [k for k, (sx, sy) in seen if abs(sx - x) < 0.05 and sy == y]
            if near:                                    # the same outcome as a rule drawn already: say so beside it
                ax.annotate("collapsing = default" if key == "collapsing" else f"{key} = default", (x, y),
                            xytext=(10, -2), textcoords="offset points", ha="left", va="top", fontsize=8.6, color=TEZ_INK)
            else:
                ax.annotate(f"{r['harmful']} + {r['oos_false']}", (x, y), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8.6, color=INK2)
            seen.append((key, (x, y)))
        ax.set_xlim(3.1, 4.0)
        ax.set_ylim(0, 9)
        finish(ax, "mean word of the first action", None)
        ctx.subtitle(ax, title)
    axes[0].set_ylabel("harmful + out-of-scope false actions", fontsize=9.5)
    ctx.legend(axes[0], [dot(c, lab, marker=mk) for _, c, mk, lab in DDM], ncol=3, fs=8.6,
               note="220 utterances; parameters chosen on one half, scored on the other · labels: harmful + out-of-scope")


def finding_ddm(D):
    f = D.ddm_f["held_out_summary"]
    p = D.ddm_p["held_out_summary"]
    same = f["collapsing"]["harmful"] == f["default_on_eval"]["harmful"] and \
        f["collapsing"]["oos_false"] == f["default_on_eval"]["oos_false"]
    return ("A collapsing commit bound buys nothing: on final transcripts it "
            + ("matches the default rule" if same else "differs little from the default rule")
            + f"; on partial ones it fires earlier (word {p['collapsing']['mean_first_word']:.2f} against "
            f"{p['default_on_eval']['mean_first_word']:.2f}) only with {p['collapsing']['harmful']} harmful actions "
            f"instead of {p['default_on_eval']['harmful']}")


# ---------------------------------------------------------------- p: decide, then bind
def order_facts(D):
    P = D.order["probe"]
    ls = layers(P)
    free = [k for k in ls if P[str(k)]["reversed_read_as_content"] >= P[str(k)]["standard"] - 0.002]
    last_free = max(k for k in free if all(j in free for j in ls if j <= k))
    return P, ls, last_free


def draw_order(ctx, D):
    ax = ctx.ax()
    P, ls, last_free = order_facts(D)
    ax.plot(ls, [P[str(k)]["standard"] for k in ls], "-o", color=NEUTRAL2, lw=2, ms=6, mec=SURFACE, mew=1)
    ax.plot(ls, [P[str(k)]["reversed_read_as_content"] for k in ls], "-o", color=TEZ, lw=2.2, ms=7, mec=SURFACE, mew=1)
    ax.axvspan(min(ls), last_free, color=TEZ, alpha=0.08, lw=0)
    ax.annotate("no loss", ((min(ls) + last_free) / 2, 0.83), ha="center", fontsize=8.8, color=TEZ_INK)
    L = D.order["letters"]
    refline(ax, L["standard_order_acc"], f"4B letters {f3(L['standard_order_acc'])}; reversing the options changes the "
            f"pick in {100 * (1 - L['same_option_chosen']):.0f} %", color=NEUTRAL, x=0.99, va="top")
    ax.set_ylim(0.4, 0.85)
    finish(ax, "layer (of 32)", "probe accuracy")
    ctx.legend(ax, [line(color=NEUTRAL2, label="options in the order the probe was trained on"),
                    line(color=TEZ, label="options reversed, same probe")], ncol=2)


def finding_order(D):
    P, ls, last_free = order_facts(D)
    d18 = 100 * (P["18"]["standard"] - P["18"]["reversed_read_as_content"])
    d26 = 100 * (P["26"]["standard"] - P["26"]["reversed_read_as_content"])
    return (f"Decide first, bind later: a probe trained on one option order reads reversed options with no loss up to "
            f"layer {last_free}, {d18:.1f} points down at layer 18, but {d26:.1f} down at layer 26")


# ---------------------------------------------------------------- registry, in reading order (a to p in the grid)
def P_(key, short, finding, what, source, draw, needs, h=3.6, left=0.9, section="The probe lab", setup=SETUP):
    return Panel(key=key, short=short, finding=finding, what=what, setup=setup, source=source, draw=draw, h=h,
                 left=left, needs=tuple(needs), section=section)


PANELS = [
    P_("lab_logit_lens", "Logit lens: where the letter readout is best", finding_lens,
       "Zero-shot letter accuracy read at every layer through the final norm and output head",
       f"{SRC} (Where to read: the zero-shot logit lens; results/probe_lab_lens_*.json); 12B at full depth: BENCHMARKS.md §1.",
       draw_lens, ("results/probe_lab_lens_qwen35-4b.json", "results/h2h/summary.json"), section="Where and what to read"),
    P_("lab_layer_without_labels", "Choosing the layer without labels", finding_id,
       "Intrinsic dimension of the unlabelled states and probe accuracy, layer by layer",
       f"{SRC} (intrinsic dimension; results/probe_lab_intrinsic_qwen35-4b.json, results/hidden_probe_sweep_qwen35-4b.json).",
       draw_id, ("results/probe_lab_intrinsic_qwen35-4b.json", "results/hidden_probe_sweep_qwen35-4b.json"), h=4.2,
       section="Where and what to read"),
    P_("lab_anytime_depth", "Anytime decisions: an adaptive stop ties a fixed cut", finding_anytime,
       "Accuracy against mean layers used, for fixed cuts and two adaptive stopping rules",
       f"{SRC} (Anytime depth; results/probe_lab_anytime_qwen35-4b.json).", draw_anytime,
       ("results/probe_lab_anytime_qwen35-4b.json",), section="Where and what to read"),
    P_("lab_code_size", "64 numbers carry the decision", finding_code,
       "Probe accuracy when each state is cut to its first m principal components, or m random projections",
       f"{SRC} (How many numbers does a decision need?; results/probe_lab_variants_qwen35-4b.json); Jev: third-party published.",
       draw_code, ("results/probe_lab_variants_qwen35-4b.json",), section="Where and what to read"),
    P_("lab_probe_families", "Probe families on layer 26: plain logistic wins", finding_families,
       "Accuracy of each probe family on the layer-26 states", f"{SRC} (Probe families on layer 26; "
       "results/probe_lab_variants_qwen35-4b.json).", draw_families, ("results/probe_lab_variants_qwen35-4b.json",),
       left=2.4, section="Probes and labels"),
    P_("lab_without_labels", "Without labels the probe only matches its teacher", finding_label_free,
       "Probes trained on teachers' answers instead of gold labels, a label model and a stacked readout",
       f"{SRC} (No human labels; Stacking readouts; results/probe_lab_w2s_qwen35-4b.json, "
       "results/probe_lab_labelmodel_qwen35-4b.json, results/probe_lab_stack_qwen35-4b.json); Jev: third-party published.",
       draw_label_free, ("results/probe_lab_w2s_qwen35-4b.json",), left=2.85, section="Probes and labels"),
    P_("lab_few_labels", "With few labels, choose which rows to label", finding_few,
       "Accuracy against labelled rows per question, by which rows are labelled and what they are combined with",
       f"{SRC} (Few labels; results/probe_lab_typical_qwen35-4b.json, results/probe_lab_fewshot_qwen35-4b.json, "
       "results/probe_lab_prior_qwen35-4b.json).", draw_few, ("results/probe_lab_typical_qwen35-4b.json",), h=3.8,
       section="Probes and labels"),
    P_("lab_cold_start", "Cold start: learn from escalations, but audit at random", finding_cold,
       "Held-out accuracy against human labels spent, by labelling policy, beside random labels at the same budget",
       f"{SRC} (Cold start; results/probe_lab_online2_qwen35-4b.json).", draw_cold,
       ("results/probe_lab_online2_qwen35-4b.json",), h=3.8, section="Probes and labels"),
    P_("lab_conformal_selection", "A guaranteed error rate on what it acts on", finding_confsel,
       "Share of decisions acted on for a chosen bound on the error among them", f"{SRC} (Conformal selection; "
       "results/probe_lab_confsel_qwen35-4b.json).", draw_confsel, ("results/probe_lab_confsel_qwen35-4b.json",),
       section="Guarantees and transfer"),
    P_("lab_unseen_workflows", "One head for workflows it has never seen", finding_unseen,
       "Accuracy on each workflow of a head trained on the other three", f"{SRC} (A universal 'is this answer right?' "
       "probe; One head for every question; results/probe_lab_universal_qwen35-4b.json, results/probe_truth_qwen35-4b.json, "
       "results/probe_lab_workflow_baselines.json).", draw_unseen, ("results/probe_lab_universal_qwen35-4b.json",),
       section="Guarantees and transfer"),
    P_("lab_many_decisions", "Many decisions from one forward pass", finding_multiq,
       "Probe accuracy by layer for three prompt layouts, 400 test rows of 5 questions",
       f"{SRC} (Many decisions from one forward pass; results/probe_multiq_qwen35-4b.json); Jev: third-party published.",
       draw_multiq, ("results/probe_multiq_qwen35-4b.json",), section="Guarantees and transfer"),
    P_("lab_crosslingual", "MASSIVE: trained in English only, 11 languages", finding_xl,
       "MASSIVE intent accuracy per language: an English-trained probe, the 12B's letters and laya-multilingual",
       f"{SRC} (MASSIVE: one probe trained on English only; results/probe_crosslingual_qwen35-4b.json); 12B and "
       "laya-multilingual: BENCHMARKS.md §1 (results/h2h/summary.json).", draw_xl,
       ("results/probe_crosslingual_qwen35-4b.json", "results/h2h/summary.json"), section="Guarantees and transfer",
       setup=SETUP_XL),
    P_("lab_pruned_served", "Cut the served 4B: the probe keeps its accuracy, faster", finding_pruned,
       "Accuracy against milliseconds per decision for depth-pruned GGUFs served by llama.cpp",
       f"{SRC} (Depth-pruned Qwen3.5-4B GGUFs; Where a decision's latency goes; results/pruned_server_qwen35-4b.json, "
       "results/latency_breakdown.json).", draw_pruned, ("results/pruned_server_qwen35-4b.json",), h=3.8, setup=SETUP_SERVED,
       section="Serving it"),
    P_("lab_cut_public_suite", "The cut 4B on Laya's public suite (zero-shot)", finding_cut,
       "Zero-shot letter accuracy of the 4B at full depth and cut to 29 or 24 blocks, on Laya's public tasks",
       f"{SRC} (The cut 4B on Laya's public suite; results/h2h_q4b_L24, _L29, _L32/summary.json).", draw_cut,
       ("results/h2h_q4b_L29/summary.json", "results/h2h_q4b_L32/summary.json"), section="Serving it", setup=SETUP_SUITE),
    P_("lab_voice_commit_bound", "Voice: a shrinking commit bound buys nothing", finding_ddm,
       "Mean word of the first action against harmful and out-of-scope false actions, three commit rules",
       f"{SRC} (Streaming voice: a collapsing commit bound; results/voice_ddm_partial_last.json, "
       "results/voice_ddm_final_last.json).", draw_ddm, ("results/voice_ddm_final_last.json",), h=3.2, setup=SETUP_VOICE,
       section="Serving it"),
    P_("lab_decide_then_bind", "Decide, then bind: reversing the options", finding_order,
       "Probe accuracy by layer with the options in the trained order and reversed",
       f"{SRC} (Decide, then bind; results/probe_order_qwen35-4b.json).", draw_order,
       ("results/probe_order_qwen35-4b.json",), section="Serving it"),
]


def main():
    D = load()
    print("panels:")
    for p in PANELS:
        fp.render_panel(p, D)
    print("grid:")
    out = fp.ROOT / "docs" / "figures"
    fp.render_grid(PANELS, D, out / "tez_lab.png", out / "tez_lab.pdf",
                   "Tez probe lab: reading decisions from the middle of a frozen model",
                   "typed-decisions, 2,000 test decisions, unless stated. Each panel is also a chart of its own in "
                   "docs/figures/panels/, titled with its finding.", ncol=4)
    fp.write_index()


if __name__ == "__main__":
    main()
