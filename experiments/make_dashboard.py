"""One multi-panel dashboard of every measurement in the project (docs/figures/tez_dashboard.png),
in the spirit of Laya's README figure, plus the individual panels as PDFs for the paper.

Reads: results/h2h/summary.json, results/authored144_*.metrics.json, results/perm_analysis_gemma4-12b-q8_0.json,
results/voicefast_final_last_gemma4-12b-q8_0.summary.json, results/stream_analysis_gemma4-12b-q8_0.json,
results/stream_policy_final_last_gemma4-12b-q8_0.json, results/conformal_typed_decisions.json,
results/cascade_authored144.json, results/batch_calibration.json, results/asr_bench_*.summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"tez": "#0E7C7B", "laya-en": "#4A7FB5", "laya-ml": "#E07A3F", "laya-td": "#3C8D5A", "jev": "#9A9A9A", "grey": "#BBBBBB"}
NAME = {"tez": "Tez · Gemma 4 12B, zero-shot", "laya-en": "laya (English)", "laya-ml": "laya-multilingual", "laya-td": "laya-typed-decisions (fine-tuned)", "jev": "Jev 1.13 (published)"}
JEV = {"ag_news": 0.910, "emotion": 0.480, "banking77": 0.870, "typed_decisions": 0.727}


def J(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def routed(S, t):
    lang = t.split(":")[1] if ":" in t else "en"
    return S.get(t, {}).get("laya-en" if lang == "en" else "laya-ml")


def main():
    S = J("results/h2h/summary.json", {})
    out = Path("docs/figures"); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10})
    fig = plt.figure(figsize=(22, 35))
    gs = fig.add_gridspec(7, 4, hspace=0.6, wspace=0.35)
    fig.suptitle("Tez — every measurement, one page.  Zero-shot frozen Gemma 4 12B Q8_0 · one forward pass · RTX 5080 laptop (16 GB)\n"
                 "Laya checkpoints ran on the SAME rows and GPU; Jev figures are third-party published, never measured here.",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.965)

    # ---------------------------------------------------------------- A: public datasets
    ax = fig.add_subplot(gs[0, 0:2])
    tasks = [t for t in ["ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections", "massive:en", "xnli:en", "typed_decisions"] if t in S]
    w = 0.27
    for i, t in enumerate(tasks):
        tz = S[t].get("tez", {}).get("accuracy"); la = routed(S, t)
        if tz is not None: ax.bar(i - w, tz, w, color=COL["tez"]); ax.text(i - w, tz + .01, f"{tz:.3f}", ha="center", fontsize=6.5)
        if la: ax.bar(i, la["accuracy"], w, color=COL["laya-en"]); ax.text(i, la["accuracy"] + .01, f"{la['accuracy']:.3f}", ha="center", fontsize=6.5)
        ref = S[t].get("laya-td", {}).get("accuracy") if t == "typed_decisions" else JEV.get(t)
        if ref: c = COL["laya-td"] if t == "typed_decisions" else COL["jev"]; ax.bar(i + w, ref, w, color=c); ax.text(i + w, ref + .01, f"{ref:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t.replace("_", " ").replace(":en", " (en)") for t in tasks], rotation=15, fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy"); ax.grid(axis="y", alpha=.3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("tez", "laya-en", "jev", "laya-td")],
              labels=[NAME["tez"], "Laya (routed checkpoint)", NAME["jev"], "laya-typed-decisions (fine-tuned on the benchmark)"], fontsize=7, loc="lower left", ncol=2)
    ax.set_title("A · Accuracy on Laya's public datasets — same rows, same machine")

    # ---------------------------------------------------------------- B: every task, every checkpoint (horizontal)
    ax = fig.add_subplot(gs[0:2, 2:4])
    tasks_b = [t for t in ["ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections", "typed_decisions", "massive:en", "xnli:en"] if t in S] + sorted(t for t in S if t.startswith("massive:") and t != "massive:en")
    models = ["tez", "laya-en", "laya-ml", "laya-td"]
    h = 0.8 / len(models)
    for j, m in enumerate(models):
        ys = [i + (j - 1.5) * h for i in range(len(tasks_b))]
        xs = [S[t].get(m, {}).get("accuracy") or 0 for t in tasks_b]
        ax.barh(ys, xs, h, color=COL[m], label=NAME[m])
        for y, x in zip(ys, xs):
            if x: ax.text(x + .005, y, f"{x:.3f}", va="center", fontsize=5.5)
    ax.set_yticks(range(len(tasks_b))); ax.set_yticklabels(tasks_b, fontsize=7); ax.invert_yaxis(); ax.set_xlim(0, 1.08); ax.set_xlabel("accuracy")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="x", alpha=.3)
    ax.set_title("B · Every task, every checkpoint (MASSIVE: 20 options, 100/lang)")

    # ---------------------------------------------------------------- C: languages dot plot
    ax = fig.add_subplot(gs[1, 0:2])
    langs = sorted([t for t in S if t.startswith("massive:")], key=lambda t: -(S[t].get("tez", {}).get("accuracy") or 0))
    for i, t in enumerate(langs):
        for m, mk in (("laya-en", "o"), ("laya-ml", "s"), ("tez", "D")):
            if m in S[t]: ax.plot(S[t][m]["accuracy"], len(langs) - i, mk, color=COL[m], ms=7)
    ax.set_yticks(range(1, len(langs) + 1)); ax.set_yticklabels([t.split(":")[1] for t in reversed(langs)], fontsize=7)
    ax.axvline(0.05, ls="--", color="grey", lw=.8); ax.axvline(0.15, ls=":", color="grey", lw=.8); ax.set_xlim(0, 1)
    ax.text(0.052, len(langs) + .3, "random", fontsize=6, color="grey"); ax.text(0.152, len(langs) + .3, "3× random", fontsize=6, color="grey")
    ax.legend(handles=[plt.Line2D([], [], marker=mk, color=COL[m], ls="") for m, mk in (("tez", "D"), ("laya-en", "o"), ("laya-ml", "s"))], labels=[NAME[m] for m in ("tez", "laya-en", "laya-ml")], fontsize=7, loc="lower right")
    ax.set_xlabel("accuracy · MASSIVE intent"); ax.set_title("C · One frozen decoder vs two trained encoders, per language")

    # ---------------------------------------------------------------- D: typed-decisions
    ax = fig.add_subplot(gs[2, 0])
    T = S.get("typed_decisions", {})
    FS0 = J("results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json")
    bars = [("Tez\nzero-shot", T.get("tez", {}).get("accuracy"), COL["tez"]), ("Tez\n4-shot prefix", FS0["accuracy"] if FS0 else None, "#0B5F5E"), ("laya base\nzero-shot", T.get("laya-en", {}).get("accuracy"), COL["laya-en"]),
            ("laya-td\nfine-tuned", T.get("laya-td", {}).get("accuracy"), COL["laya-td"]), ("Jev\npublished", 0.727, COL["jev"]), ("teacher\nceiling", 0.735, "#DDDDDD"), ("majority", 0.461, "#EEEEEE"), ("random", 0.318, "#F5F5F5")]
    bars = [b for b in bars if b[1] is not None]
    for i, (n, v, c) in enumerate(bars):
        ax.bar(i, v, color=c, edgecolor="#888"); ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=6.5); ax.set_ylim(0, .9); ax.grid(axis="y", alpha=.3)
    ax.set_title("D · typed-decisions (2,000 decisions)")

    # ---------------------------------------------------------------- E: calibration shipped vs refit
    ax = fig.add_subplot(gs[2, 1])
    for j, m in enumerate(models):
        raw = [v[m]["ece"] for v in S.values() if m in v and "ece_refit" in v[m]]; ref = [v[m]["ece_refit"] for v in S.values() if m in v and "ece_refit" in v[m]]
        if raw:
            a, b = np.mean(raw), np.mean(ref)
            ax.bar(j - .18, a, .36, color=COL[m], alpha=.45); ax.bar(j + .18, b, .36, color=COL[m])
            ax.text(j - .18, a + .005, f"{a:.2f}", ha="center", fontsize=6.5); ax.text(j + .18, b + .005, f"{b:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.246, ls="--", color="grey", lw=.8); ax.text(3.4, .25, "Jev 0.246", fontsize=6, color="grey", ha="right")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(["Tez", "laya", "laya-ml", "laya-td"], fontsize=7); ax.set_ylabel("mean ECE-15")
    ax.set_title("E · Calibration: shipped (light) → per-task refit (dark)")

    # ---------------------------------------------------------------- F: order flip
    ax = fig.add_subplot(gs[2, 2])
    ft = [t for t in ["ag_news", "emotion", "massive:en", "xnli:en"] if t in S]
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        ax.bar([i + (j - 1.5) * w for i in range(len(ft))], [S[t].get(m, {}).get("flip_rate") or 0 for t in ft], w, color=COL[m])
    ax.axhline(0.13, ls="--", color="grey", lw=.8); ax.text(len(ft) - .5, .135, "Jev 0.13", fontsize=6, color="grey", ha="right")
    ax.set_xticks(range(len(ft))); ax.set_xticklabels(ft, fontsize=7); ax.set_ylabel("flip rate, options reversed")
    ax.set_title("F · Option-order robustness (first 100 rows)")

    # ---------------------------------------------------------------- G: speed
    ax = fig.add_subplot(gs[2, 3])
    st = [t for t in ["ag_news", "boolq", "massive:en", "typed_decisions", "banking77"] if t in S]
    for j, m in enumerate(models):
        ax.bar([i + (j - 1.5) * w for i in range(len(st))], [S[t].get(m, {}).get("ms_p50") or 0 for t in st], w, color=COL[m])
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st, fontsize=7, rotation=10); ax.set_ylabel("ms / decision (p50)")
    ax.set_title("G · Latency, same GPU (Tez over HTTP, state uncached)")

    # ---------------------------------------------------------------- H: SemIf scale curve
    ax = fig.add_subplot(gs[3, 0])
    pts = [("Llama 3.2 3B", "llama32-3b-q4km"), ("Gemma 3 4B", "gemma3-4b-q4km"), ("Qwen3-4B", "qwen3-4b-bf16-hf-fresh"), ("Qwen3.5-4B", "qwen35-4b-bf16-hf"), ("Gemma 4 12B Q4", "gemma4-12b-q4km"), ("Gemma 4 12B Q8", "gemma4-12b-q8_0")]
    xs, ys, lo, hi = [], [], [], []
    for n, tag in pts:
        m = J(f"results/authored144_{tag}.metrics.json")
        if m:
            xs.append(n); v = m["mean_family_balanced_accuracy"]; ys.append(v); lo.append(v - m["mfba_ci95"][0]); hi.append(m["mfba_ci95"][1] - v)
    ax.bar(range(len(xs)), ys, color=COL["tez"], alpha=.8, yerr=[lo, hi], capsize=3)
    for i, v in enumerate(ys): ax.text(i, v + .03, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.axhline(0.813, ls="--", color=COL["jev"], lw=.8); ax.text(0, .83, "SemIf's published 4B 0.813", fontsize=6, color="grey")
    ax.axhline(0.958, ls=":", color=COL["jev"], lw=.8); ax.text(0, .965, "SemIf's 27B 0.958", fontsize=6, color="grey")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=20, fontsize=7); ax.set_ylim(0, 1.05); ax.set_ylabel("mean-family balanced acc.")
    ax.set_title("H · Scale-gated symbol binding (SemIf, 95% CI)")

    # ---------------------------------------------------------------- I: SemIf calibration recipes
    ax = fig.add_subplot(gs[3, 1])
    P = J("results/perm_analysis_gemma4-12b-q8_0.json", {})
    rec = [("raw", "orig"), ("perm2", "perm2_mean"), ("perm6", "perm6_mean"), ("perm6+T", "perm6_mean+temperature_oof"), ("Zhao N/A", "zhao_NA_only"), ("Zhao 3-avg", "zhao_3placeholder_avg")]
    names = [n for n, k in rec if k in P]; nll = [P[k]["nll"] for n, k in rec if k in P]; acc = [P[k]["accuracy"] for n, k in rec if k in P]
    ax.bar(range(len(names)), nll, color=[COL["tez"] if "Zhao" not in n else "#D1495B" for n in names])
    for i, (v, a) in enumerate(zip(nll, acc)): ax.text(i, v + .03, f"NLL {v:.2f}\nacc {a:.3f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=15); ax.set_ylabel("NLL (lower better)")
    ax.set_title("I · Calibration recipes, Gemma 4 12B Q8")

    # ---------------------------------------------------------------- J: Batch calibration dumbbells
    ax = fig.add_subplot(gs[3, 2])
    B = J("results/batch_calibration.json", {})
    keys = [k for k in B if k.startswith("h2h/") and ":" not in k] + [k for k in B if k.startswith("h2h/xnli:")][:4]
    for i, k in enumerate(keys):
        a, b = B[k]["raw_nll"], B[k]["bc_oof_nll"]
        ax.plot([a, b], [i, i], color="#BBBBBB", lw=2); ax.plot(a, i, "o", color="#D1495B"); ax.plot(b, i, "o", color=COL["tez"])
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([k.replace("h2h/", "") for k in keys], fontsize=7); ax.invert_yaxis(); ax.set_xlabel("NLL: raw (red) → Batch Calibration, out-of-fold (green)")
    ax.set_title("J · Batch Calibration (Zhou 2023), NLL")

    # ---------------------------------------------------------------- K: conformal
    ax = fig.add_subplot(gs[3, 3])
    C = J("results/conformal_typed_decisions.json", {})
    ms_ = [m for m in ("tez", "laya-td", "laya-en") if m in C]
    for i, m in enumerate(ms_):
        for j, key in enumerate(("alpha=0.1 temp=True", "alpha=0.05 temp=True")):
            v = C[m][key]; x = i + (j - .5) * .38
            ax.bar(x, v["act_rate"], .36, color=COL[m]); ax.bar(x, v["escalate_rate"], .36, bottom=v["act_rate"], color=COL[m], alpha=.35)
            ax.text(x, v["act_rate"] / 2, f"act {v['act_rate']:.0%}\n@{v['acc_when_acting']:.2f}", ha="center", va="center", fontsize=6, color="white")
            ax.text(x, 1.01, "90%" if j == 0 else "95%", ha="center", fontsize=6)
    ax.set_xticks(range(len(ms_))); ax.set_xticklabels(["Tez zero-shot", "laya-td fine-tuned", "laya base"][: len(ms_)], fontsize=7); ax.set_ylim(0, 1.1); ax.set_ylabel("share of decisions")
    ax.set_title("K · Conformal act / escalate (typed-decisions)")

    # ---------------------------------------------------------------- L: voice streaming
    ax = fig.add_subplot(gs[4, 0])
    ax.plot([.1, .3, .5, .7, .9], [0.222, 0.396, 0.567, 0.651, 0.845], "-o", color=COL["tez"], label="argmax = gold intent")
    ax.plot([.1, .3, .5, .7, .9], [0.744, 0.561, 0.356, 0.245, 0.034], "-s", color="#D1495B", label="argmax = none (keep listening)")
    ax.set_xticks([.1, .3, .5, .7, .9]); ax.set_xticklabels(["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"], fontsize=7); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_xlabel("fraction of the utterance heard"); ax.set_title("L · Voice: partial-transcript predictions")

    # ---------------------------------------------------------------- M: voice policy + budget
    ax = fig.add_subplot(gs[4, 1])
    V = J("results/voicefast_final_last_gemma4-12b-q8_0.summary.json", {}); SP = J("results/stream_policy_final_last_gemma4-12b-q8_0.json", {})
    items = [("intent acc\n(full)", V.get("intent_accuracy", 0)), ("acc accepting\nthen/alt", V.get("intent_accuracy_alt_or_then_ok", 0)), ("none\nprecision", V.get("none_precision", 0)),
             ("final action\nconsistent", SP.get("final_action_consistent_rate", 0)), ("1 − harmful", 1 - SP.get("harmful_rate", 0)), ("compound\nresidual→then", 22 / 22)]
    ax.bar(range(len(items)), [v for _, v in items], color=COL["tez"])
    for i, (_, v) in enumerate(items): ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([n for n, _ in items], fontsize=6.5); ax.set_ylim(0, 1.1)
    ax.set_title("M · Voice → action outcomes (220 commands)")

    ax = fig.add_subplot(gs[4, 2])
    stages = [("ASR partial\n(base.en)", 40), ("decision\ncompute", 30), ("HTTP+JSON", 15), ("policy+slots", 1)]
    left = 0
    for n, v in stages:
        ax.barh(0, v, left=left, color=[COL["laya-en"], COL["tez"], COL["jev"], COL["laya-td"]][stages.index((n, v))]); ax.text(left + v / 2, 0, f"{n}\n{v} ms", ha="center", va="center", fontsize=6.5, color="white"); left += v
    ax.barh(1, 205, color="#D1495B", alpha=.6); ax.text(102, 1, "before the cache fix: 205 ms compute per word", ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["now", "before"], fontsize=7); ax.set_xlabel("ms per streamed word"); ax.set_xlim(0, 230)
    ax.set_title("N · Per-word budget (~85 ms word → action)")

    # ---------------------------------------------------------------- O: cascade
    ax = fig.add_subplot(gs[4, 3])
    Cs = J("results/cascade_authored144.json", {})
    if Cs:
        for k, mk, lab in (("small_only", "o", "Gemma 3 4B"), ("small_perm2_only", "o", "4B ×2 orderings"), ("large_only", "D", "Gemma 4 12B"), ("agreement_only", "s", "cascade: 4B, escalate on disagreement")):
            v = Cs[k]; ax.plot(v["ms"], v["acc"], mk, ms=8, label=lab); ax.text(v["ms"] + 2, v["acc"], f"{v['acc']:.3f}", fontsize=6.5)
        sw = Cs["sweeps"]["perm_agreement_gate"]; ax.plot([s["ms"] for s in sw], [s["acc"] for s in sw], "-", color="#BBBBBB", lw=1, label="cascade sweep")
        ax.set_xlabel("mean ms per decision"); ax.set_ylabel("accuracy"); ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("O · Small → large cascade: not worth it")

    # ---------------------------------------------------------------- P: ASR + Q: typed-decisions by workflow
    ax = fig.add_subplot(gs[5, 0])
    for j, (tag, c) in enumerate((("base-en", COL["tez"]), ("small-en", COL["laya-en"]))):
        A = J(f"results/asr_bench_{tag}_gemma4-12b-q8_0.summary.json")
        if A:
            vals = [A["partial_ms_p50"] / 200, A["wer_mean"], 1 - A["exact_rate"], 1 - A["intent_agree_asr_vs_text"]]
            ax.bar([i + (j - .5) * .38 for i in range(4)], vals, .36, color=c, label=tag)
            for i, v in enumerate(vals): ax.text(i + (j - .5) * .38, v + .005, f"{v:.3f}" if i else f"{A['partial_ms_p50']:.0f} ms", ha="center", fontsize=6)
    ax.set_xticks(range(4)); ax.set_xticklabels(["partial latency\n(÷200)", "WER", "1 − exact", "intent disagreement\nvs true text"], fontsize=6.5); ax.legend(fontsize=7)
    ax.set_title("P · ASR stage (faster-whisper, synthetic speech)")

    ax = fig.add_subplot(gs[5, 1])
    if "typed_decisions" in S:
        wfs = sorted(S["typed_decisions"]["tez"]["by_workflow"])
        for j, m in enumerate(("tez", "laya-td", "laya-en")):
            if m in S["typed_decisions"]:
                ax.bar([i + (j - 1) * .27 for i in range(len(wfs))], [S["typed_decisions"][m]["by_workflow"][w]["accuracy"] for w in wfs], .27, color=COL[m], label=NAME[m])
        ax.set_xticks(range(len(wfs))); ax.set_xticklabels([w.replace("_", "\n") for w in wfs], fontsize=6.5); ax.set_ylim(0, 1); ax.legend(fontsize=6.5)
    ax.set_title("Q · typed-decisions by workflow")

    # ---------------------------------------------------------------- R: soft accuracy / Brier vs teacher
    ax = fig.add_subplot(gs[5, 2])
    if "typed_decisions" in S:
        ms2 = [m for m in ("tez", "laya-en", "laya-td") if m in S["typed_decisions"]]
        sa = [S["typed_decisions"][m].get("soft_accuracy", 0) for m in ms2]; br = [S["typed_decisions"][m].get("brier_vs_soft", 0) for m in ms2]
        ax.bar([i - .2 for i in range(len(ms2))], sa, .4, color=[COL[m] for m in ms2]); ax.bar([i + .2 for i in range(len(ms2))], br, .4, color=[COL[m] for m in ms2], alpha=.4)
        for i in range(len(ms2)): ax.text(i - .2, sa[i] + .01, f"soft {sa[i]:.3f}", ha="center", fontsize=6); ax.text(i + .2, br[i] + .01, f"Brier {br[i]:.3f}", ha="center", fontsize=6)
        ax.axhline(0.580, ls="--", color="grey", lw=.8); ax.text(len(ms2) - .5, .59, "Jev soft acc 0.580", fontsize=6, color="grey", ha="right")
        ax.set_xticks(range(len(ms2))); ax.set_xticklabels(["Tez", "laya base", "laya-td"][: len(ms2)], fontsize=7)
    ax.set_title("R · Teacher-distribution match (soft acc ↑, Brier ↓)")

    # ---------------------------------------------------------------- S: hidden-state probes
    HP = J("results/hidden_probe_qwen35-4b.json"); HP12 = J("results/hidden_probe_server_gemma4-12b-q8_0.json")
    if HP:
        axp = fig.add_subplot(gs[6, 0])
        names = ["4B letter" + chr(10) + "logits", "4B probe" + chr(10) + "L-1", "4B probe" + chr(10) + "L-4", "4B probe" + chr(10) + "L-8", "4B probe" + chr(10) + "L-12", "12B probe" + chr(10) + "final (server)"]
        vals = [HP["letter_logits"]["accuracy"], HP["layer-1"]["logreg_acc"], HP["layer-4"]["logreg_acc"], HP["layer-8"]["logreg_acc"], HP["layer-12"]["logreg_acc"], HP12["logreg_acc"] if HP12 else 0]
        axp.bar(range(6), vals, color=[COL["laya-en"]] + [COL["tez"]] * 4 + ["#0B5F5E"])
        for i, v in enumerate(vals): axp.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=7)
        axp.axhline(0.766, ls="--", color=COL["laya-td"], lw=.8); axp.text(5.4, .775, "laya-td 0.766", fontsize=6.5, ha="right", color=COL["laya-td"])
        axp.axhline(0.727, ls=":", color="grey", lw=.8); axp.text(5.4, .70, "Jev 0.727", fontsize=6.5, ha="right", color="grey")
        axp.set_xticks(range(6)); axp.set_xticklabels(names, fontsize=7); axp.set_ylim(0, .9); axp.set_title("S · Probes on frozen hidden states (typed-decisions)")
    # ---------------------------------------------------------------- T/U: probe layer curve + data efficiency
    SW = J("results/hidden_probe_sweep_qwen35-4b.json")
    if SW:
        axt = fig.add_subplot(gs[6, 1])
        ls = sorted(int(k) for k in SW["layer_sweep"]); axt.plot(ls, [SW["layer_sweep"][str(l)]["acc"] for l in ls], "-o", color=COL["tez"], ms=3)
        axt.axhline(SW["letter_logits"]["acc"], ls=":", color=COL["laya-en"], lw=.8); axt.text(0, SW["letter_logits"]["acc"] + .01, "4B letter logits", fontsize=5.5, color=COL["laya-en"])
        axt.axhline(0.766, ls="--", color=COL["laya-td"], lw=.8); axt.text(0, .77, "laya-td 0.766", fontsize=5.5, color=COL["laya-td"])
        axt.set_xlabel("layer (of 32)"); axt.set_ylabel("probe accuracy"); axt.tick_params(labelsize=7); axt.set_ylim(.4, .85)
        axt.set_title("T · Probe accuracy by layer, Qwen3.5-4B")
        axu = fig.add_subplot(gs[6, 2])
        de = SW["data_efficiency"]; xs = sorted(int(k) for k in de); axu.errorbar(xs, [de[str(x)]["mean"] for x in xs], yerr=[de[str(x)]["std"] for x in xs], fmt="-o", color=COL["tez"], ms=3, capsize=2)
        axu.axhline(0.727, ls=":", color="grey", lw=.8); axu.text(10, .73, "Jev 0.727", fontsize=5.5, color="grey")
        axu.axhline(0.766, ls="--", color=COL["laya-td"], lw=.8); axu.text(10, .77, "laya-td 0.766", fontsize=5.5, color=COL["laya-td"])
        axu.set_xscale("log"); axu.set_xlabel("labelled rows per question"); axu.tick_params(labelsize=7); axu.set_ylim(.6, .85)
        axu.set_title("U · Probe data efficiency (layer 26)")
    # ---------------------------------------------------------------- S: headline numbers text
    ax = fig.add_subplot(gs[5, 3]); ax.axis("off")
    txt = ("Headline\n\n"
           f"typed-decisions zero-shot  {S.get('typed_decisions', {}).get('tez', {}).get('accuracy', 0):.3f}\n   (Laya base 0.36 · Jev 0.727 · Laya fine-tuned 0.766)\n"
           f"MASSIVE, 11 languages      {np.mean([S[t]['tez']['accuracy'] for t in S if t.startswith('massive:')]):.3f} macro\n   (laya-multilingual 0.524 · Khmer 0.790 vs 0.000)\n"
           f"Banking77, 77 options      {S.get('banking77', {}).get('tez', {}).get('accuracy', 0):.3f}  (Laya 0.395)\n"
           "SemIf authored144          0.943  (SemIf 4B 0.813)\n"
           "voice: per streamed word   30 ms compute · 1 harmful action / 198\n"
           "order flip @ 20 options    0.07  (Laya 0.15–0.20 · Jev 0.13)\n"
           "conformal 90% coverage     act on 52% @ 0.853 accuracy\n"
           "probe on frozen 4B (L26)  0.794 · 50 rows/question 0.741\n\n"
           "Where others win: Laya on tasks in its training mix\n(AG News, NLI) and on single-question latency;\nJev on Banking77 (0.870).")
    ax.text(0, 1, txt, va="top", fontsize=9, family="monospace")

    fig.savefig(out / "tez_dashboard.png", dpi=110, bbox_inches="tight")
    fig.savefig(out / "tez_dashboard.pdf", bbox_inches="tight")
    print("wrote", out / "tez_dashboard.png")


if __name__ == "__main__":
    main()
