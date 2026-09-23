"""One figure for the probe-lab round: every experiment on cached hidden states, one panel each.
Reads results/probe_lab_*.json and the GPU follow-ups (multi-question pass, cross-lingual probes, truth
probe, pruned GGUFs); a missing result leaves its panel empty with a note.

  py experiments/make_lab_figure.py        ->  docs/figures/tez_lab.png / .pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COL = {"tez": "#0E7C7B", "dark": "#0B5F5E", "blue": "#4A7FB5", "orange": "#E07A3F", "green": "#3C8D5A", "grey": "#9A9A9A", "red": "#C8475A", "light": "#7FB8B7"}


def J(p):
    f = ROOT / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def refs(ax, x, jev=True, laya=True, fs=6.3):
    if laya:
        ax.axhline(0.766, ls="--", color=COL["green"], lw=.8); ax.text(x, .769, "laya-td 0.766 (fine-tuned)", fontsize=fs, color=COL["green"])
    if jev:
        ax.axhline(0.727, ls=":", color=COL["grey"], lw=.8); ax.text(x, .713, "Jev 0.727", fontsize=fs, color=COL["grey"])


def empty(ax, what):
    ax.text(.5, .5, f"{what}\n(not yet measured)", ha="center", va="center", fontsize=8, color=COL["grey"], transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])


def hbars(ax, items, xlim, ref=None):
    ax.barh(range(len(items)), [v for _, v, _ in items], color=[c for _, _, c in items])
    for i, (_, v, _) in enumerate(items):
        ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=6.3)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _, _ in items], fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(*xlim)
    if ref:
        ax.axvline(ref[0], ls=":", color=COL["grey"], lw=.8); ax.text(ref[0], len(items) - .4, ref[1], fontsize=6, color=COL["grey"])


def main():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.8, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(24, 22)); gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.42, top=0.955)
    fig.suptitle("Tez probe lab: reading decisions from the middle of a frozen model (typed-decisions, 2,000 test decisions, unless stated)", fontsize=14, fontweight="bold", y=0.995)

    # a: logit lens by relative depth
    ax = fig.add_subplot(gs[0, 0])
    for tag, lab, col, n in (("qwen35-4b", "Qwen3.5-4B", COL["tez"], 32), ("gemma4-12b-nf4", "Gemma 4 12B (layers 0-40 of 48)", COL["dark"], 48)):
        R = J(f"results/probe_lab_lens_{tag}.json")
        if R:
            ls = sorted(int(k) for k in R["logit_lens"]); ax.plot([l / n for l in ls], [R["logit_lens"][str(l)] for l in ls], "-o", ms=2.5, color=col, label=lab)
    ax.axhline(0.704, ls="--", color=COL["dark"], lw=.8); ax.text(0.02, .712, "12B at full depth (llama.cpp) 0.704", fontsize=6.3, color=COL["dark"])
    ax.annotate("4B: layer 29 beats\nthe final layer by 10 pts", xy=(29 / 32, .583), xytext=(.45, .66), fontsize=6.3, arrowprops=dict(arrowstyle="->", lw=.7))
    ax.set_xlabel("depth read (fraction of layers)"); ax.set_ylabel("zero-shot letter accuracy"); ax.set_ylim(.2, .8); ax.legend(fontsize=6.3, loc="upper left")
    ax.set_title("a · Logit lens: where the letter readout is best")

    # b: intrinsic dimension vs probe accuracy (label-free layer choice)
    ax = fig.add_subplot(gs[0, 1]); I = J("results/probe_lab_intrinsic_qwen35-4b.json"); SW = J("results/hidden_probe_sweep_qwen35-4b.json")
    if I and SW:
        ls = sorted(int(k) for k in I["layers"] if I["layers"][k]["twonn_pooled"] == I["layers"][k]["twonn_pooled"])
        ax.plot(ls, [I["layers"][str(l)]["twonn_pooled"] for l in ls], "-", color=COL["orange"], lw=1.5, label="intrinsic dimension of unlabelled states (TwoNN)")
        ax.set_ylabel("intrinsic dimension", color=COL["orange"]); ax.set_xlabel("layer (of 32)")
        a2 = ax.twinx(); lp = sorted(int(k) for k in SW["layer_sweep"]); a2.plot(lp, [SW["layer_sweep"][str(l)]["acc"] for l in lp], "-o", ms=2.5, color=COL["tez"], label="probe accuracy (needs labels)")
        a2.set_ylabel("probe accuracy", color=COL["tez"]); a2.set_ylim(.4, .85); a2.spines["right"].set_visible(True)
        ax.axvspan(24, 28, color=COL["orange"], alpha=.1); ax.text(26, ax.get_ylim()[0] + .3, "ID minimum\n= probe plateau", ha="center", fontsize=6.3, color=COL["orange"])
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="upper left")
    else:
        empty(ax, "intrinsic dimension")
    ax.set_title("b · Choosing the layer without labels")

    # c: anytime depth frontier
    ax = fig.add_subplot(gs[0, 2]); A = J("results/probe_lab_anytime_qwen35-4b.json")
    if A:
        g = A["grid"]; ax.plot(g, [A["fixed"][str(l)] for l in g], "-o", color=COL["grey"], ms=3, label="fixed depth")
        for key, col, lab in (("gate", COL["tez"], "stop when max p >= tau"), ("accumulate", COL["orange"], "stop when summed log-odds >= theta")):
            pts = sorted((v["mean_depth"], v["acc"]) for v in A[key].values()); ax.plot([p[0] for p in pts], [p[1] for p in pts], "-s", ms=3, color=col, label=lab)
        ax.set_xlabel("mean layers used (of 32)"); ax.set_ylabel("accuracy"); ax.set_ylim(.55, .82); ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("c · Anytime decisions: an adaptive stop ties a fixed cut")

    # d: decision code size
    ax = fig.add_subplot(gs[0, 3]); V = J("results/probe_lab_variants_qwen35-4b.json")
    if V:
        cs = V["code_size"]; ms = sorted(int(k) for k in cs["pca"])
        ax.plot(ms, [cs["pca"][str(m)] for m in ms], "-o", color=COL["tez"], ms=3, label="one PCA, unlabelled, shared by 20 questions")
        ax.plot(ms, [cs["random"][str(m)] for m in ms], "-s", color=COL["blue"], ms=3, label="random projection")
        ax.axhline(cs["full"], color=COL["dark"], lw=.8); ax.text(2.2, cs["full"] + .006, f"all {cs['dim']} numbers: {cs['full']:.3f}", fontsize=6.3, color=COL["dark"])
        refs(ax, 70); ax.set_xscale("log", base=2); ax.set_xlabel("numbers kept per state"); ax.set_ylabel("probe accuracy"); ax.set_ylim(.55, .82); ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("d · 64 numbers carry the decision")

    # e: probe families
    ax = fig.add_subplot(gs[1, 0])
    if V:
        fam = V["families"]; items = [(n, fam[n]["acc"], COL["tez"] if n.startswith("logreg C=0.5") else COL["blue"]) for n in fam]
        hbars(ax, items, (.6, .83))
    ax.set_title("e · Probe families on layer 26: plain logistic wins")

    # f: label-free probes, label model, stacking
    ax = fig.add_subplot(gs[1, 1]); W = J("results/probe_lab_w2s_qwen35-4b.json"); LM = J("results/probe_lab_labelmodel_qwen35-4b.json"); ST = J("results/probe_lab_stack_qwen35-4b.json")
    items = []
    if W:
        items.append(("probe on gold labels (reference)", W["gold"]["26"], COL["grey"]))
        for tname, lab in (("gemma12b", "12B answers"), ("self", "4B's own answers")):
            T = W["teachers"].get(tname)
            if T:
                items.append((f"teacher: {lab}", T["teacher_test_acc"], COL["orange"]))
                best = max(((L, n, v["acc"]) for L, d in T["variants"].items() for n, v in d.items()), key=lambda t: t[2])
                items.append((f"probe from {lab} ({best[1]}, L{best[0]})", best[2], COL["tez"]))
        if W["teachers"].get("agree"):
            v = max(W["teachers"]["agree"]["variants"].values(), key=lambda d: d["agreement-filtered"]["acc"])["agreement-filtered"]["acc"]
            items.append(("probe from rows where 4B and 12B agree", v, COL["dark"]))
    if LM:
        best = max(LM["combos"].items(), key=lambda kv: kv[1]["label_model_test_acc"])
        items.append((f"Dawid-Skene label model ({best[0]})", best[1]["label_model_test_acc"], COL["red"]))
    if ST:
        items.append(("stacked: probe + 12B letters (gold labels)", ST["pools"]["probe L26 + 12B letters"]["stacked_logreg"], COL["green"]))
    if items:
        hbars(ax, items, (.4, .86), ref=(0.727, "Jev"))
    else:
        empty(ax, "label-free probes")
    ax.set_title("f · Without labels the probe only matches its teacher")

    # g: few labels — labelling order and small-sample probes
    ax = fig.add_subplot(gs[1, 2]); TY = J("results/probe_lab_typical_qwen35-4b.json"); FS = J("results/probe_lab_fewshot_qwen35-4b.json")
    if TY:
        ns = sorted(int(k) for k in TY["n"])
        ax.errorbar(ns, [TY["n"][str(n)]["random"]["mean"] for n in ns], yerr=[TY["n"][str(n)]["random"]["std"] for n in ns], fmt="-o", ms=3, capsize=2, color=COL["grey"], label="random rows labelled")
        ax.errorbar(ns, [TY["n"][str(n)]["typical"]["mean"] for n in ns], yerr=[TY["n"][str(n)]["typical"]["std"] for n in ns], fmt="-s", ms=3, capsize=2, color=COL["tez"], label="most typical rows labelled first")
    if FS:
        ns = sorted(int(k) for k in FS["n"])
        for m, col, mk in (("LDA (unlabelled covariance)", COL["red"], "v"), ("PCA-64 (unlabelled) + logreg", COL["blue"], "^")):
            ax.plot(ns, [FS["n"][str(n)][m]["mean"] for n in ns], "--" + mk, ms=3, color=col, label=m)
    if TY or FS:
        refs(ax, 5.2); ax.set_xscale("log"); ax.set_xlabel("labelled rows per question"); ax.set_ylabel("accuracy"); ax.set_ylim(.4, .82); ax.legend(fontsize=6.2, loc="lower right")
    else:
        empty(ax, "few labels")
    ax.set_title("g · With few labels, choose which rows to label")

    # h: cold start
    ax = fig.add_subplot(gs[1, 3]); O2 = J("results/probe_lab_online2_qwen35-4b.json")
    if O2:
        pol_col = {"escalate-only": COL["red"], "escalate+audit": COL["orange"], "audit-train": COL["blue"], "typical-seed": COL["tez"]}
        for key, r in O2["runs"].items():
            pol = key.split("_")[1]; ax.plot(r["labels"], r["test_acc"], "o", color=pol_col.get(pol, COL["grey"]), ms=5)
            ax.plot(r["labels"], r["random_same_budget_test_acc"], "x", color=COL["grey"], ms=5)
        for pol, col in pol_col.items():
            ax.plot([], [], "o", color=col, label=pol)
        ax.plot([], [], "x", color=COL["grey"], label="same number of random labels")
        ax.axhline(O2["zero_shot_test_acc"], ls="--", color=COL["dark"], lw=.8); ax.text(0.02, O2["zero_shot_test_acc"] + .004, f"12B zero-shot start {O2['zero_shot_test_acc']:.3f}", fontsize=6.2, color=COL["dark"], transform=ax.get_yaxis_transform())
        refs(ax, 60); ax.set_xscale("log"); ax.set_xlabel("human labels spent"); ax.set_ylabel("held-out accuracy"); ax.set_ylim(.6, .82); ax.legend(fontsize=6.2, loc="lower right")
    else:
        empty(ax, "cold start")
    ax.set_title("h · Cold start: learn from escalations, but audit at random")

    # i: conformal selection (bounded error among acted decisions)
    ax = fig.add_subplot(gs[2, 0]); CS = J("results/probe_lab_confsel_qwen35-4b.json")
    if CS:
        alphas = [0.05, 0.1, 0.2]; cols = [COL["tez"], COL["dark"], COL["orange"]]
        for (name, R), col in zip(CS.items(), cols):
            ax.plot([a * 100 for a in alphas], [R[f"alpha{a}"]["acted"] * 100 for a in alphas], "-o", ms=3, color=col, label=f"{name}: acted")
            for a in alphas:
                ax.text(a * 100 + .3, R[f"alpha{a}"]["acted"] * 100 - 3, f"err {R[f'alpha{a}']['realised_error_among_acted'] * 100:.1f}%", fontsize=5.8, color=col)
        ax.set_xlabel("error bound chosen (% of acted decisions allowed to be wrong)"); ax.set_ylabel("% of decisions acted on"); ax.set_ylim(0, 105); ax.legend(fontsize=6.0, loc="upper left")
    else:
        empty(ax, "conformal selection")
    ax.set_title("i · A guaranteed error rate on what it acts on")

    # j: universal heads on unseen workflows
    ax = fig.add_subplot(gs[2, 1]); U = J("results/probe_lab_universal_qwen35-4b.json"); TR = J("results/probe_truth_qwen35-4b.json")
    if U:
        wfs = list(U["baselines"]); x = np.arange(len(wfs)); w = .2
        bestL = max(U["layers"], key=lambda l: U["layers"][l]["lowo_mean"])
        ax.bar(x - 1.5 * w, [U["baselines"][f]["letters_4b"] for f in wfs], w, color=COL["blue"], label="4B letters (zero-shot)")
        ax.bar(x - .5 * w, [U["layers"][bestL]["lowo"][f]["acc"] for f in wfs], w, color=COL["light"], label=f"shared letter probe (L{bestL})")
        if TR:
            k = max(TR["lowo"], key=lambda kk: TR["lowo"][kk]["mean"])
            ax.bar(x + .5 * w, [TR["lowo"][k][f] for f in wfs], w, color=COL["tez"], label=f"'is this answer right?' probe ({k})")
        if U["baselines"][wfs[0]].get("letters_12b") is not None:
            ax.bar(x + 1.5 * w, [U["baselines"][f]["letters_12b"] for f in wfs], w, color=COL["dark"], label="12B letters (zero-shot)")
        ax.set_xticks(x); ax.set_xticklabels([f.replace("_", "\n") for f in wfs], fontsize=6.5); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.0, loc="upper center", ncol=2)
    ax.set_title("j · One head for workflows it has never seen")

    # k: many decisions per pass
    ax = fig.add_subplot(gs[2, 2]); M = J("results/probe_multiq_qwen35-4b.json")
    if M:
        ls = sorted(int(k) for k in M["probe"])
        for key, col, lab in (("per_question_prompt", COL["grey"], "one prompt per question (5 passes per row)"), ("A_all_questions_one_pass", COL["tez"], "all questions in one pass"), ("B_state_only", COL["orange"], "state only, one pass")):
            ax.plot(ls, [M["probe"][str(l)][key] for l in ls], "-o", ms=3, color=col, label=lab)
        t = M["tokens_test"]
        ax.text(0.02, 0.03, f"tokens, 400 test rows: per question {t['per_question_prompts']:,}\none pass {t['A_all_questions_one_pass']:,} · state only {t['B_state_only']:,}", transform=ax.transAxes, fontsize=6.2)
        refs(ax, 12.5); ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy"); ax.set_ylim(.45, .85); ax.legend(fontsize=6.2, loc="center right")
    else:
        empty(ax, "many decisions per pass")
    ax.set_title("k · Many decisions from one forward pass")

    # l: cross-lingual transfer
    ax = fig.add_subplot(gs[2, 3]); X = J("results/probe_crosslingual_qwen35-4b.json"); S = J("results/h2h/summary.json")
    if X:
        bestL = max(X["layers"], key=lambda l: X["layers"][l]["macro_acc20"]); R = X["layers"][bestL]
        langs = [l for l in R if isinstance(R[l], dict)]; x = np.arange(len(langs)); w = .27
        ax.bar(x - w, [R[l]["acc20"] for l in langs], w, color=COL["tez"], label=f"4B probe trained on English only (L{bestL})")
        if S:
            ax.bar(x, [S.get(f"massive:{l}", {}).get("tez", {}).get("accuracy", np.nan) for l in langs], w, color=COL["dark"], label="12B zero-shot letters")
            ax.bar(x + w, [S.get(f"massive:{l}", {}).get("laya-ml", {}).get("accuracy", np.nan) for l in langs], w, color=COL["orange"], label="laya-multilingual")
        ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=6.5); ax.set_ylim(0, 1.15); ax.legend(fontsize=6.0, loc="upper center", ncol=2)
    else:
        empty(ax, "cross-lingual probe")
    ax.set_title(f"l · MASSIVE: trained in English only, 11 languages" + (f" (macro {R['macro_acc20']:.3f})" if X else ""))

    # m: pruned GGUFs served by llama.cpp
    ax = fig.add_subplot(gs[3, 0]); P = J("results/pruned_server_qwen35-4b.json")
    if P:
        for name, r in P.items():
            n = int(name[1:])
            ax.plot(r["ms_embed_p50"], r["probe_acc"], "o", color=COL["tez"], ms=5); ax.text(r["ms_embed_p50"], r["probe_acc"] + .01, f"{n} layers", fontsize=6.3, color=COL["tez"])
            ax.plot(r["ms_letters_p50"], r["letters_acc"], "s", color=COL["blue"], ms=5); ax.text(r["ms_letters_p50"], r["letters_acc"] - .03, f"{n}", fontsize=6.3, color=COL["blue"])
        ax.plot([], [], "o", color=COL["tez"], label="probe on the served state"); ax.plot([], [], "s", color=COL["blue"], label="zero-shot letters (= logit lens at the cut)")
        ax.plot(304, 0.7045, "D", color=COL["dark"], ms=5); ax.text(304, 0.715, "12B letters", fontsize=6.3, color=COL["dark"])
        refs(ax, 5); ax.set_xlabel("ms per decision, llama.cpp, RTX 5080 laptop (p50)"); ax.set_ylabel("accuracy"); ax.set_ylim(.2, .85); ax.legend(fontsize=6.2, loc="lower right")
    else:
        empty(ax, "pruned GGUFs")
    ax.set_title("m · Depth-pruned 4B GGUFs: accuracy vs latency")

    # n: pruned 4B on the public suite
    ax = fig.add_subplot(gs[3, 1]); A29 = J("results/h2h_q4b_L29/summary.json"); A32 = J("results/h2h_q4b_L32/summary.json")
    if A29 and A32:
        tasks = [t for t in A32 if t in A29 and "tez" in A32[t] and not t.startswith(("massive:", "xnli:"))] + ["MASSIVE (11)", "XNLI (10)"]
        def val(Sx, t):
            if t == "MASSIVE (11)":
                return np.mean([Sx[k]["tez"]["accuracy"] for k in Sx if k.startswith("massive:")])
            if t == "XNLI (10)":
                return np.mean([Sx[k]["tez"]["accuracy"] for k in Sx if k.startswith("xnli:")])
            return Sx[t]["tez"]["accuracy"]
        x = np.arange(len(tasks)); w = .38
        ax.bar(x - w / 2, [val(A32, t) for t in tasks], w, color=COL["blue"], label="4B, all 32 layers")
        ax.bar(x + w / 2, [val(A29, t) for t in tasks], w, color=COL["tez"], label="4B cut to 29 layers")
        ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=6.2, rotation=0); ax.set_ylim(0, 1.05); ax.legend(fontsize=6.3, loc="upper right")
    else:
        empty(ax, "pruned 4B on the public suite")
    ax.set_title("n · Removing the top 3 layers, on Laya's public tasks")

    # o: voice commit rule
    ax = fig.add_subplot(gs[3, 2])
    rows = []
    for v, lab in (("partial_last", "partial"), ("final_last", "final")):
        R = J(f"results/voice_ddm_{v}.json")
        if R:
            for key, col in (("default_on_eval", COL["grey"]), ("fixed", COL["blue"]), ("collapsing", COL["tez"])):
                r = R["held_out_summary"][key]; rows.append((lab, key, r, col))
    if rows:
        for lab, key, r, col in rows:
            mk = "o" if lab == "partial" else "s"
            ax.plot(r["mean_first_word"], r["harmful"] + r["oos_false"] + (0.08 if lab == "final" else 0), mk, color=col, ms=7, alpha=.85)
        for key, col, name in (("default_on_eval", COL["grey"], "default (tau 0.9, stability 2)"), ("fixed", COL["blue"], "best retuned fixed threshold"), ("collapsing", COL["tez"], "best collapsing bound")):
            ax.plot([], [], "o", color=col, label=name)
        ax.plot([], [], "o", color="k", mfc="none", label="partial-transcript log"); ax.plot([], [], "s", color="k", mfc="none", label="final-transcript log")
        ax.legend(fontsize=6.0, loc="upper right")
        ax.set_xlabel("mean word at which the first action fires (lower = earlier)"); ax.set_ylabel("harmful + out-of-scope false actions (of 220)")
    else:
        empty(ax, "voice commit rule")
    ax.set_title("o · Voice: a shrinking commit bound buys nothing")

    # p: the recipe
    ax = fig.add_subplot(gs[3, 3]); ax.axis("off")
    ax.text(0, 1, "What carries over to any schema\n\n"
                  "1  read the decision at the layer where the\n   unlabelled states are most compressed\n   (intrinsic-dimension minimum: layers 24-28)\n"
                  "2  cut the model there: fewer layers, same probe\n"
                  "3  label the most typical rows first,\n   then escalate, but keep a random audit slice\n"
                  "4  act only under a conformal error bound\n   (5 %: acts on 40 % of decisions)\n"
                  "5  keep 64 numbers per state: the decision code\n\n"
                  "What did not work: label-free probes beyond their\nteacher, Dawid-Skene over weak readouts,\nuncertainty-only labelling, DoLa, adaptive depth,\none shared letter probe on unseen workflows,\na collapsing voice bound.",
            va="top", fontsize=8.2, family="monospace", transform=ax.transAxes)

    out = ROOT / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "tez_lab.png", dpi=100, bbox_inches="tight"); fig.savefig(out / "tez_lab.pdf", bbox_inches="tight")
    print("wrote", out / "tez_lab.png")


if __name__ == "__main__":
    main()
