"""Every measurement in the project, one chart per finding (docs/figures/panels/), and the same charts on one page
(docs/figures/tez_dashboard.png / .pdf) for anyone who wants them all at once.

    python experiments/make_dashboard.py

Each panel below is a Panel entry (experiments/figpanels.py): a draw function, the finding it states (its title,
built from the data), what is plotted, the conditions and the source. Charts on their own are 9 inches wide at
200 dpi, for a README column; the grid shows the same panels with short titles.

Reads: results/h2h/summary.json, results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json,
results/calibration/default_temperature.json, results/authored144_*.metrics.json,
results/perm_analysis_gemma4-12b-q8_0.json, results/batch_calibration.json, results/conformal_typed_decisions.json,
results/voicefast_final_last_gemma4-12b-q8_0.summary.json and _stream.jsonl, results/stream_policy_final_last_gemma4-12b-q8_0.json,
results/cascade_authored144.json, results/asr_bench_*.summary.json, results/hidden_probe_*.json,
results/early_exit_qwen35-4b.json, results/retrieval_narrow_banking77_minilm.json,
results/option_elimination_gemma4-12b-q8_0.json, results/speed/summary.json (BENCHMARKS.md §5b).
A panel whose file is missing is skipped, never drawn with placeholder numbers.
"""
from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

import matplotlib
import matplotlib.ticker
import numpy as np

import figpanels as fp
from figpanels import (BAD, BEFORE_LW, COLOR, HATCH, INK, INK2, INK3, JEV, LAYA_EN, LAYA_ML, LAYA_TD, NAME, NEUTRAL,
                       NEUTRAL2, REF, SHIPPED_ALPHA, SURFACE, TEZ, TEZ_DEEP, TEZ_INK, TEZ_LIGHT, J, Panel, Patch, bar,
                       dot, f3, grid, hlab, join_and, line, patch, pct, refline, strip, vlab)

# Jev's published figures (third-party; Jev was never run here) and Laya's published typed-decisions baselines
JEV_ACC = {"ag_news": 0.910, "emotion": 0.480, "banking77": 0.870, "typed_decisions": 0.727}
JEV_ECE, JEV_FLIP, JEV_SOFT = 0.246, 0.13, 0.580
LAYA_TD_PUB = {"teacher ceiling": 0.735, "majority class": 0.461, "random": 0.318}
SEMIF_PUB = {"4B": 0.813, "27B": 0.958}
MODELS = ["tez", "laya-en", "laya-ml", "laya-td"]
TASK_NAME = {"ag_news": "AG News", "emotion": "DAIR Emotion", "banking77": "Banking77", "sst5": "SST-5", "boolq": "BoolQ",
             "prompt_injections": "prompt-injections", "typed_decisions": "typed-decisions",
             "massive:en": "MASSIVE-en", "xnli:en": "XNLI-en"}
IN_LAYA_TRAINING = {"ag_news", "boolq", "xnli:en"}
WF_NAME = {"agent_trace_observability": "agent-trace observability", "customer_service": "customer service",
           "invoice_processing": "invoice processing", "security_incidents": "security incidents"}

H2H_SETUP = ("Tez and Laya's checkpoints answered byte-identical rows on one RTX 5080 laptop GPU. Tez = frozen Gemma 4 "
             "12B Q8_0 on llama.cpp b11100, one forward pass, zero-shot letter readout. Jev's figures are third-party "
             "published; Jev was never run here.")
SEMIF_SETUP = ("SemIf's exact prompt and public fixtures (authored144: 144 three-option decisions), frozen models, "
               "RTX 5080 laptop; mean-family balanced accuracy, classes keyed by option id.")
VOICE_SETUP = ("220 spoken commands, 16 actions; Gemma 4 12B Q8_0 on llama-server with --swa-full, the action list as a "
               "cached prefix and the transcript last.")
PROBE_SETUP = ("Logistic probes on the frozen Qwen3.5-4B's hidden states (bf16; no LLM training), one per question, "
               "fitted on the typed-decisions train split; test split, 2,000 decisions.")
SPEED_SETUP = ("Clean reruns under a GPU lock on the RTX 5080 laptop, VRAM checked before and after each run: Gemma 4 "
               "12B Q8_0 letters on the production llama-server (-c 4096 -b 512 -np 1 --swa-full) or in process "
               "through llama.dll.")


def load():
    S = J("results/h2h/summary.json", {})
    D = SimpleNamespace(S=S)
    D.fewshot = J("results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json")
    D.anchor = (J("results/calibration/default_temperature.json") or {}).get("anchor_28")
    D.semif = {n: J(f"results/authored144_{t}.metrics.json") for n, t in (
        ("Llama 3.2 3B Q4", "llama32-3b-q4km"), ("Gemma 3 4B Q4", "gemma3-4b-q4km"), ("Qwen3-4B", "qwen3-4b-bf16-hf-fresh"),
        ("Qwen3.5-4B BF16", "qwen35-4b-bf16-hf"), ("Gemma 4 12B Q4", "gemma4-12b-q4km"), ("Gemma 4 12B Q8", "gemma4-12b-q8_0"))}
    D.perm = J("results/perm_analysis_gemma4-12b-q8_0.json", {})
    D.bc = J("results/batch_calibration.json", {})
    D.conf = J("results/conformal_typed_decisions.json", {})
    D.voice = J("results/voicefast_final_last_gemma4-12b-q8_0.summary.json", {})
    D.policy = J("results/stream_policy_final_last_gemma4-12b-q8_0.json", {})
    D.cascade = J("results/cascade_authored144.json", {})
    D.asr = {t: J(f"results/asr_bench_{t}_gemma4-12b-q8_0.summary.json") for t in ("base-en", "small-en")}
    D.hp = J("results/hidden_probe_qwen35-4b.json")
    D.hp12 = J("results/hidden_probe_server_gemma4-12b-q8_0.json")
    D.sweep = J("results/hidden_probe_sweep_qwen35-4b.json")
    D.sweep12 = J("results/hidden_probe_sweep_gemma4-12b-nf4_std.json")
    D.early = J("results/early_exit_qwen35-4b.json")
    D.rn = J("results/retrieval_narrow_banking77_minilm.json")
    D.tp = J("results/hidden_probe_tasks_qwen35-4b.json")
    D.oe = J("results/option_elimination_gemma4-12b-q8_0.json")
    D.speed = J("results/speed/summary.json")
    D.stream = _stream_by_fraction("results/voicefast_final_last_gemma4-12b-q8_0_stream.jsonl")
    return D


def _stream_by_fraction(rel):
    """experiments/stream_analysis.py's buckets (actionable utterances, share of words heard in fifths): how often
    the top answer on a partial transcript is the gold action, and how often it is 'none'."""
    p = fp.ROOT / rel
    if not p.exists():
        return None
    gold, none, n = defaultdict(list), defaultdict(list), 0
    for x in p.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        s = json.loads(x)
        if s["gold"] == "none":
            continue
        n += 1
        for t in s["trajectory"]:
            b = min(4, int(t["k"] / s["n_words"] * 5))
            gold[b].append(t["pred"] == s["gold"])
            none[b].append(t["pred"] == "none")
    return {"gold": [round(float(np.mean(gold[b])), 3) for b in range(5)],
            "none": [round(float(np.mean(none[b])), 3) for b in range(5)], "n": n}


def acc(D, t, m):
    return D.S.get(t, {}).get(m, {}).get("accuracy")


# ================================================================== accuracy against Laya and Jev
PUBLIC = ["typed_decisions", "banking77", "massive:en", "prompt_injections", "sst5", "boolq", "emotion", "ag_news", "xnli:en"]


def public_rows(D):
    return [t for t in PUBLIC if t in D.S]


def draw_public(ctx, D):
    ax = ctx.ax()
    rows = public_rows(D)
    for i, t in enumerate(rows):
        vals = [acc(D, t, m) for m in MODELS if acc(D, t, m) is not None] + ([JEV_ACC[t]] if t in JEV_ACC else [])
        ax.plot([min(vals), max(vals)], [i, i], color=fp.GRID, lw=5, solid_capstyle="round", zorder=1)
        for m, mk in (("laya-en", "o"), ("laya-ml", "s"), ("laya-td", "^")):
            v = acc(D, t, m)
            if v is not None:
                ax.plot(v, i, mk, ms=8, color=COLOR[m], mec=SURFACE, mew=1.2, zorder=3)
        if t in JEV_ACC:
            ax.plot(JEV_ACC[t], i, "D", ms=8, mfc="none", mec=JEV, mew=1.6, zorder=3)
        v = acc(D, t, "tez")
        ax.plot(v, i, "o", ms=12, color=TEZ, mec=SURFACE, mew=1.6, zorder=4)
        ax.annotate(f3(v), (v, i), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    color=INK, fontweight="bold", zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TASK_NAME[t] + ("*" if t in IN_LAYA_TRAINING else "") for t in rows], fontsize=9.5)
    ax.set_ylim(len(rows) - 0.4, -0.75)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("accuracy", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, [dot(TEZ, "Tez zero-shot", ms=9), dot(LAYA_EN, "laya"), dot(LAYA_ML, "laya-multilingual", "s"),
                    dot(LAYA_TD, "laya-typed-decisions", "^"), dot(JEV, "Jev (published)", "D", hollow=True)], ncol=5,
               note="* in Laya's training mix · laya-typed-decisions is fine-tuned on typed-decisions' train split",
               fs=8.8)


def finding_public(D):
    rows = public_rows(D)
    lead = [t for t in rows if acc(D, t, "tez") > max(acc(D, t, m) for m in MODELS[1:])]
    jev = [t for t in rows if t in JEV_ACC]
    jev_lead = [t for t in jev if JEV_ACC[t] > acc(D, t, "tez")]
    return (f"Zero-shot Tez leads every Laya checkpoint on {len(lead)} of {len(rows)} public tasks on identical rows; "
            f"Laya leads the other {len(rows) - len(lead)}, and Jev's published figures lead {len(jev_lead)} of the "
            f"{len(jev)} they cover")


def draw_td(ctx, D):
    ax = ctx.ax()
    T = D.S.get("typed_decisions", {})
    items = [("Tez zero-shot", T.get("tez", {}).get("accuracy"), dict(color=TEZ), True),
             ("Tez, 4 worked examples\nin the cached prefix", D.fewshot["accuracy"] if D.fewshot else None,
              dict(color=TEZ_DEEP), True),
             None,
             ("laya-typed-decisions\n(fine-tuned on it)", T.get("laya-td", {}).get("accuracy"), dict(color=LAYA_TD), False),
             ("Jev (published)", JEV_ACC["typed_decisions"], dict(color=JEV), False),
             ("laya", T.get("laya-en", {}).get("accuracy"), dict(color=LAYA_EN), False),
             ("laya-multilingual", T.get("laya-ml", {}).get("accuracy"), dict(color=LAYA_ML), False),
             None] + [(f"{k}*", v, dict(color=REF), False) for k, v in LAYA_TD_PUB.items()]
    y, ys, labs = 0.0, [], []
    for it in items:
        if it is None:
            y += 0.4
            continue
        lab, v, st, strong = it
        if v is None:
            continue
        bar(ax, y, v, 0.72, horizontal=True, **st)
        hlab(ax, v, y, f3(v), fs=9, color=INK if strong else INK2, weight="bold" if strong else "normal")
        ys.append(y)
        labs.append(lab)
        y += 1.0
    ax.set_yticks(ys)
    ax.set_yticklabels(labs, fontsize=9.2)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("accuracy, 2,000 decisions", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, note="* Laya's published baselines for typed-decisions, not our rerun")


def finding_td(D):
    T = D.S["typed_decisions"]
    fs = f" ({f3(D.fewshot['accuracy'])} with 4 worked examples)" if D.fewshot else ""
    return (f"On typed-decisions zero-shot Tez scores {f3(T['tez']['accuracy'])}{fs}, against Jev's published "
            f"{f3(JEV_ACC['typed_decisions'])} and {f3(T['laya-td']['accuracy'])} for laya-typed-decisions, fine-tuned on "
            f"the benchmark; laya's base checkpoints score {f3(T['laya-en']['accuracy'])} and {f3(T['laya-ml']['accuracy'])}")


def draw_td_workflow(ctx, D):
    ax = ctx.ax()
    T = D.S["typed_decisions"]
    wfs = sorted(T["tez"]["by_workflow"])
    ms = [m for m in ("tez", "laya-td", "laya-en") if m in T]
    w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        for i, wf in enumerate(wfs):
            v = T[m]["by_workflow"][wf]["accuracy"]
            x = i - 0.4 + w / 2 + j * w
            bar(ax, x, v, w * 0.92, color=COLOR[m])
            vlab(ax, x, v, f3(v), fs=8.4, color=INK if m == "tez" else INK2, weight="bold" if m == "tez" else "normal")
    ax.set_xticks(range(len(wfs)))
    ax.set_xticklabels([WF_NAME.get(wf, wf) for wf in wfs], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=COLOR[m], label={"tez": "Tez zero-shot", "laya-td": "laya-typed-decisions (fine-tuned on it)"}
                          .get(m, NAME[m])) for m in ms], ncol=3)


def finding_td_workflow(D):
    T = D.S["typed_decisions"]
    gaps = {wf: T["laya-td"]["by_workflow"][wf]["accuracy"] - T["tez"]["by_workflow"][wf]["accuracy"]
            for wf in T["tez"]["by_workflow"]}
    behind = [wf for wf, g in gaps.items() if g > 0]
    worst = max(gaps, key=gaps.get)
    ahead = [WF_NAME.get(wf, wf) for wf, g in gaps.items() if g < 0]
    return (f"By workflow, laya-typed-decisions (fine-tuned on this benchmark) leads zero-shot Tez on {len(behind)} of "
            f"{len(gaps)}, by up to {100 * gaps[worst]:.1f} points on {WF_NAME.get(worst, worst)}"
            + (f"; Tez leads on {join_and(ahead)}" if ahead else ""))


def draw_teacher(ctx, D):
    left, right = ctx.row((1, 1), gap=0.7)
    T = D.S["typed_decisions"]
    ms = [m for m in ("tez", "laya-en", "laya-td") if m in T]
    for ax, key, lab, ref in ((left, "soft_accuracy", "soft accuracy (higher is better)", JEV_SOFT),
                              (right, "brier_vs_soft", "Brier vs the teacher (lower is better)", None)):
        for i, m in enumerate(ms):
            v = T[m].get(key)
            bar(ax, i, v, 0.62, color=COLOR[m])
            vlab(ax, i, v, f3(v), fs=9, color=INK if m == "tez" else INK2, weight="bold" if m == "tez" else "normal")
        if ref is not None:
            refline(ax, ref, f"Jev (published) {f3(ref)}")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([NAME[m].replace("laya-typed-decisions", "laya-typed-\ndecisions") for m in ms], fontsize=9.2)
        ax.set_ylim(0, 0.75)
        grid(ax)
        strip(ax)
        ctx.subtitle(ax, lab)


def finding_teacher(D):
    T = D.S["typed_decisions"]
    return (f"Scored against the teacher's full distribution, zero-shot Tez's soft accuracy is "
            f"{f3(T['tez']['soft_accuracy'])} (Jev published {f3(JEV_SOFT)}); laya-typed-decisions, trained on it, has the "
            f"lowest Brier, {f3(T['laya-td']['brier_vs_soft'])}")


# ================================================================== languages
def massive(D):
    return sorted([t for t in D.S if t.startswith("massive:")], key=lambda t: -(acc(D, t, "tez") or 0))


def draw_languages(ctx, D):
    ax = ctx.ax()
    langs = massive(D)
    for i, t in enumerate(langs):
        vals = [acc(D, t, m) for m in ("tez", "laya-en", "laya-ml") if acc(D, t, m) is not None]
        ax.plot([min(vals), max(vals)], [i, i], color=fp.GRID, lw=4, solid_capstyle="round", zorder=1)
        for m, mk in (("laya-en", "o"), ("laya-ml", "s")):
            if acc(D, t, m) is not None:
                ax.plot(acc(D, t, m), i, mk, ms=8, color=COLOR[m], mec=SURFACE, mew=1.2, zorder=3)
        v = acc(D, t, "tez")
        ax.plot(v, i, "o", ms=11, color=TEZ, mec=SURFACE, mew=1.5, zorder=4)
        ax.annotate(f3(v), (v, i), xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=8.8,
                    color=INK, fontweight="bold")
    ax.axvline(0.05, color=INK3, lw=1, ls=(0, (4, 3)))
    ax.axvline(0.15, color=INK3, lw=1, ls=(0, (1, 2)))
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([t.split(":")[1] for t in langs], fontsize=9.5)
    ax.set_ylim(len(langs) - 0.5, -0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("accuracy, MASSIVE intent (20 options, 100 rows per language)", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, [dot(TEZ, "Tez zero-shot", ms=9), dot(LAYA_EN, "laya (English)"), dot(LAYA_ML, "laya-multilingual", "s")],
               ncol=3, note="dashed line: random (0.05) · dotted line: 3× random (0.15)")


def finding_languages(D):
    langs = massive(D)
    tz = [acc(D, t, "tez") for t in langs]
    lo_en = min(langs, key=lambda t: acc(D, t, "laya-en"))
    lo_ml = min(langs, key=lambda t: acc(D, t, "laya-ml"))
    return (f"One frozen decoder reads all {len(langs)} MASSIVE languages at {f3(min(tz))}–{f3(max(tz))}; Laya's English "
            f"checkpoint falls to {f3(acc(D, lo_en, 'laya-en'))} ({lo_en.split(':')[1]}) and laya-multilingual to "
            f"{f3(acc(D, lo_ml, 'laya-ml'))} ({lo_ml.split(':')[1]})")


# ================================================================== calibration, robustness, abstention
def ece_means(D):
    out = {}
    for m in MODELS:
        rows = [v[m] for v in D.S.values() if m in v and "ece_refit" in v[m]]
        if rows:
            out[m] = (float(np.mean([r["ece"] for r in rows])), float(np.mean([r["ece_refit"] for r in rows])))
    return out


def tez_default(D):
    """Tez as shipped reads unfitted letters at its default temperature, fitted without the task scored
    (results/calibration/default_temperature.json, anchor_28: the same 28 entries as the head-to-head mean)."""
    A = D.anchor
    rows = {t: v["tez"] for t, v in D.S.items() if "tez" in v and "ece_refit" in v["tez"]}
    if A and sorted(rows) == sorted(e["entry"] for e in A["entries"]) and \
            abs(np.mean([r["ece"] for r in rows.values()]) - A["mean"]["raw"]) < 1e-9:
        return A["mean"]["rule_loto"]
    return None


def draw_calibration(ctx, D):
    ax = ctx.ax()
    E, dflt = ece_means(D), tez_default(D)
    w = 0.34
    for j, m in enumerate(MODELS):
        raw, ref = E[m]
        if m == "tez" and dflt is not None:
            bar(ax, -w, raw, w * 0.95, color=SURFACE, edgecolor=TEZ, linewidth=BEFORE_LW)
            vlab(ax, -w, raw, f3(raw), fs=8.8)
            xa, xb, a = 0, w, dflt
        else:
            c = j + 0.3
            xa, xb, a = c - w / 2, c + w / 2, raw
        bar(ax, xa, a, w * 0.95, color=COLOR[m], alpha=SHIPPED_ALPHA)
        bar(ax, xb, ref, w * 0.95, color=COLOR[m])
        vlab(ax, xa, a, f3(a), fs=8.8)
        vlab(ax, xb, ref, f3(ref), fs=8.8, color=INK if m == "tez" else INK2, weight="bold" if m == "tez" else "normal")
    refline(ax, JEV_ECE, f"Jev (published) {f3(JEV_ECE)}")
    ax.set_xticks([0] + [j + 0.3 for j in range(1, len(MODELS))])
    ax.set_xticklabels(["Tez", "laya", "laya-multilingual", "laya-typed-decisions"], fontsize=9.5)
    ax.set_ylim(0, 0.38)
    ax.set_ylabel("mean ECE-15 (lower is better)", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=INK3, alpha=SHIPPED_ALPHA, label="as shipped"),
                    patch(color=INK3, label="one temperature per task"),
                    Patch(facecolor=SURFACE, edgecolor=TEZ, linewidth=BEFORE_LW, label="Tez at T = 1, before its default")],
               ncol=3, note="mean over the head-to-head tasks · Tez as shipped = its default temperature, fitted without the "
                            "task scored")


def finding_calibration(D):
    E, dflt = ece_means(D), tez_default(D)
    lt = [E[m][1] for m in MODELS[1:]]
    ship = (f"Tez ships at a mean ECE-15 of {f3(dflt)} ({f3(E['tez'][0])} at T = 1, before its default temperature)"
            if dflt is not None else f"Tez's mean ECE-15 at T = 1 is {f3(E['tez'][0])}")
    return (f"{ship}; one temperature per task brings Tez to {f3(E['tez'][1])} and Laya's checkpoints to "
            f"{f3(min(lt))}–{f3(max(lt))}")


FLIP_TASKS = ["massive:en", "emotion", "ag_news", "xnli:en"]


def draw_flip(ctx, D):
    ax = ctx.ax()
    ts = [t for t in FLIP_TASKS if t in D.S]
    w = 0.8 / len(MODELS)
    for j, m in enumerate(MODELS):
        for i, t in enumerate(ts):
            v = D.S[t].get(m, {}).get("flip_rate")
            if v is None:
                continue
            x = i - 0.4 + w / 2 + j * w
            bar(ax, x, v, w * 0.92, color=COLOR[m])
            vlab(ax, x, v, f"{v:.2f}", fs=8.4, color=INK if m == "tez" else INK2, weight="bold" if m == "tez" else "normal")
    refline(ax, JEV_FLIP, f"Jev (published, MASSIVE-en only) {JEV_FLIP:.2f}")
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([TASK_NAME[t] for t in ts], fontsize=9.5)
    ax.set_ylim(0, 0.26)
    ax.set_ylabel("share of answers that change", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=COLOR[m], label=NAME[m]) for m in MODELS], ncol=4,
               note="options presented in reverse order, first 100 rows of each task; MASSIVE-en has 20 options")


def finding_flip(D):
    t = D.S["massive:en"]
    lo = min(t[m]["flip_rate"] for m in MODELS[1:])
    hi = max(t[m]["flip_rate"] for m in MODELS[1:])
    return (f"Reversing the 20 options of MASSIVE-en changes {t['tez']['flip_rate']:.2f} of Tez's answers, against "
            f"{lo:.2f}–{hi:.2f} for Laya's checkpoints and {JEV_FLIP:.2f} published for Jev")


def bc_keys(D):
    B = D.bc
    return [k for k in B if k.startswith("h2h/") and ":" not in k] + [k for k in B if k.startswith("h2h/xnli:")][:4]


def draw_bc(ctx, D):
    ax = ctx.ax()
    B = D.bc
    keys = bc_keys(D)
    for i, k in enumerate(keys):
        a, b = B[k]["raw_nll"], B[k]["bc_oof_nll"]
        ax.plot([a, b], [i, i], color=fp.GRID, lw=4, solid_capstyle="round", zorder=1)
        ax.plot(a, i, "o", ms=9, mfc=SURFACE, mec=NEUTRAL, mew=1.6, zorder=3)
        ax.plot(b, i, "o", ms=9, color=TEZ, mec=SURFACE, mew=1.2, zorder=4)
        ax.annotate(f"{a:.2f} → {b:.2f}", (max(a, b), i), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([TASK_NAME.get(k[4:], k[4:]).replace("xnli:", "XNLI-") for k in keys], fontsize=9.3)
    ax.set_ylim(len(keys) - 0.5, -0.6)
    ax.set_xlim(0, max(max(B[k]["raw_nll"], B[k]["bc_oof_nll"]) for k in keys) * 1.25)
    ax.set_xlabel("NLL (lower is better)", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, [dot(NEUTRAL, "raw letters", hollow=True), dot(TEZ, "after Batch Calibration, out of fold")], ncol=2)


def finding_bc(D):
    B = D.bc
    keys = bc_keys(D)
    better = [k for k in keys if B[k]["bc_oof_nll"] < B[k]["raw_nll"]]
    k = max(keys, key=lambda k: B[k]["raw_nll"] - B[k]["bc_oof_nll"])
    return (f"Batch Calibration (Zhou et al. 2023), fitted out of fold, lowers NLL on {len(better)} of the {len(keys)} "
            f"sets shown, most on {TASK_NAME.get(k[4:], k[4:])} ({B[k]['raw_nll']:.2f} → {B[k]['bc_oof_nll']:.2f})")


def draw_conformal(ctx, D):
    ax = ctx.ax()
    C = D.conf
    ms = [m for m in ("tez", "laya-td", "laya-en") if m in C]
    rows, y = [], 0
    for m in ms:
        for key, cov in (("alpha=0.1 temp=True", "90 %"), ("alpha=0.05 temp=True", "95 %")):
            v = C[m][key]
            a = v["act_rate"]
            bar(ax, y, a, 0.7, color=COLOR[m], horizontal=True)
            bar(ax, y, v["escalate_rate"], 0.7, color=COLOR[m], alpha=0.18, horizontal=True, left=a)
            txt = f"acts on {pct(a)} at {f3(v['acc_when_acting'])}"
            inside = a > 0.35
            ax.annotate(txt, (a, y), xytext=(-6 if inside else 5, 0), textcoords="offset points",
                        ha="right" if inside else "left", va="center", fontsize=8.8,
                        color=fp.text_on(COLOR[m]) if inside else INK, fontweight="bold" if m == "tez" else "normal")
            rows.append((y, f"{NAME[m]}, {cov} coverage"))
            y += 1
        y += 0.4
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9.2)
    ax.set_ylim(y - 0.4 - 0.6, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([pct(x) for x in np.arange(0, 1.01, 0.25)])
    ax.set_xlabel("share of the 2,000 typed-decisions decisions", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, [patch(color=INK3, label="acts (a single answer in the conformal set)"),
                    patch(color=INK3, alpha=0.18, label="escalates")], ncol=2,
               note="Mondrian split-conformal sets by question type, with a temperature, calibrated on half the cases")


def finding_conformal(D):
    t, l = D.conf["tez"]["alpha=0.1 temp=True"], D.conf["laya-td"]["alpha=0.1 temp=True"]
    return (f"At 90 % coverage, conformal sets let zero-shot Tez act on {pct(t['act_rate'])} of typed-decisions at "
            f"{f3(t['acc_when_acting'])} accuracy and escalate the rest; laya-typed-decisions, fine-tuned on it, acts on "
            f"{pct(l['act_rate'])} at {f3(l['acc_when_acting'])}")


# ================================================================== speed
LAT_TASKS = ["ag_news", "boolq", "massive:en", "typed_decisions", "banking77"]


def draw_latency(ctx, D):
    ax = ctx.ax()
    ts = [t for t in LAT_TASKS if t in D.S]
    w = 0.8 / len(MODELS)
    for j, m in enumerate(MODELS):
        for i, t in enumerate(ts):
            v = D.S[t].get(m, {}).get("ms_p50")
            if not v:
                continue
            x = i - 0.4 + w / 2 + j * w
            bar(ax, x, v, w * 0.92, color=COLOR[m])
            vlab(ax, x, v, f"{v:.0f}", fs=8.3, color=INK if m == "tez" else INK2, weight="bold" if m == "tez" else "normal")
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([TASK_NAME[t] + ("\n(tournament)" if t == "banking77" else "") for t in ts], fontsize=9.5)
    ax.set_ylabel("p50 ms per decision", fontsize=10)
    ax.set_ylim(0, max(D.S[t][m].get("ms_p50") or 0 for t in ts for m in MODELS if m in D.S[t]) * 1.15)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=COLOR[m], label=NAME[m]) for m in MODELS], ncol=4,
               note="one question per call · Tez over HTTP to llama-server, question first: the state evaluated again "
                    "for every question · Laya in process")


def finding_latency(D):
    ts = [t for t in LAT_TASKS if t in D.S]
    tz = [D.S[t]["tez"]["ms_p50"] for t in ts]
    la = [D.S[t][m]["ms_p50"] for t in ts for m in MODELS[1:] if m in D.S[t]]
    return (f"One question per call, Laya's checkpoints answer in {min(la):.0f}–{max(la):.0f} ms; Tez over HTTP, with "
            f"the question-first layout of §1, took {min(tz):.0f}–{max(tz):.0f} ms per decision on these tasks")


def speed5b(D):
    """BENCHMARKS.md §5b's first table, row labels as it prints them, from results/speed/summary.json."""
    lp = D.speed["exp1_many_questions"]["laya_protocol"]
    rows = [("tez serve, question first (the runtime as deployed when measured)", "laya_prod_unique.json:tez:today", "ship"),
            ("direct /completion per question, question first", "laya_prod_unique.json:http:today", "direct"),
            ("direct /completion per question, state first (state cached once per call)",
             "laya_prod_unique.json:http:statefirst", "direct"),
            ("in process, sequential, state first (KV rollback)", "laya_12b_inproc_unique.json:seq_statefirst", "inproc"),
            ("in process, the state once and all question suffixes in one decode",
             "laya_12b_inproc_unique.json:batch_statefirst", "best")]
    return [(lab, {int(n): v for n, v in lp[k].items()}, kind) for lab, k, kind in rows if k in lp]


def draw_speed5b(ctx, D):
    ax = ctx.ax()
    rows = speed5b(D)
    style = {"ship": dict(color=TEZ, alpha=SHIPPED_ALPHA), "direct": dict(color=NEUTRAL2), "inproc": dict(color=TEZ),
             "best": dict(color=TEZ)}
    top = max(r[1][50]["p50"] for r in rows)
    for i, (lab, v, kind) in enumerate(rows):
        p = v[50]["p50"]
        bar(ax, i, p, 0.62, horizontal=True, **style[kind])
        ax.annotate(f"{p:,.0f} ms\n{v[50]['ms_per_question_p50']:.1f} ms per question", (p, i), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.8, linespacing=1.15,
                    color=INK if kind == "best" else INK2, fontweight="bold" if kind == "best" else "normal")
        ax.annotate(f"{v[1]['p50']:,.0f} ms", (1, i), xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=8.8, color=INK3)
    ax.annotate("1 question", (1, -0.62), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8.6,
                color=INK3, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([fp.wrap(r[0], 2.7, 9) for r in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.45, -0.75)
    ax.set_xlim(0, top * 1.8)
    ax.set_xticks(np.arange(0, top * 1.2, 4000))
    ax.set_xticklabels([f"{x:,.0f}" for x in np.arange(0, top * 1.2, 4000)])
    ax.set_xlabel("p50 ms per call, 50 distinct questions about one state", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, note="Laya's latency protocol with distinct questions, a new ticket every call · tinted: the runtime as "
                        "measured · grey: llama-server called directly · blue: in process")


def finding_speed5b(D):
    rows = {kind: v for _, v, kind in speed5b(D)}
    best, ship = rows["best"][50], rows["ship"][50]
    return (f"50 questions about one state: {best['p50']:,.0f} ms in process with one decode for every question's suffix "
            f"({best['ms_per_question_p50']:.1f} ms each), against {ship['p50']:,.0f} ms for tez serve as deployed when "
            f"measured")


def voice5b(D):
    """BENCHMARKS.md §5b's voice table, row labels as it prints them."""
    V = D.speed["exp3_voice"]
    rows = [("Gemma 4 12B letters, production llama-server (§3's setting)", "voice_12b_http_prod.json", "ref"),
            ("Gemma 4 12B letters, in process", "voice_12b_inproc.json", "best"),
            ("Qwen3.5-4B 24 blocks, probe at the answer position, in process, deferred commit (trained on prefixes, "
             "2-fold)", "voice_q4bL24_probetail_defer.json [prefix]", "probe"),
            ("Qwen3.5-4B 24 blocks letters, llama-server, cache off (the runtime's Qwen setting)",
             "voice_q4bL24_http_nocache.json", "bad")]
    out = []
    for lab, k, kind in rows:
        if k in V:
            r = V[k]
            a = dict(r.get("accuracy", {}), harmful=r.get("policy", {}).get("harmful"))
            out.append((lab, r["latency_incremental_words"]["round_trip_ms"], r["latency_incremental_words"]["compute_ms"],
                        a, kind))
    return out


def draw_voice5b(ctx, D):
    ax = ctx.ax()
    rows = voice5b(D)
    style = {"ref": dict(color=TEZ, alpha=SHIPPED_ALPHA), "best": dict(color=TEZ),
             "probe": dict(color=TEZ, hatch=HATCH), "bad": dict(color=NEUTRAL2)}
    for i, (lab, rt, cp, a, kind) in enumerate(rows):
        bar(ax, i, rt["p50"], 0.66, horizontal=True, **style[kind])
        ax.plot([rt["p50"], rt["p95"]], [i, i], color=INK3, lw=1.2, zorder=4)
        ax.plot([rt["p95"]], [i], "|", color=INK3, ms=9, zorder=4)
        harm = a.get("harmful")
        acc_ = a.get("intent_accuracy")
        extra = f"  ·  intent {acc_:.3f}" if acc_ is not None else ""
        extra += f"  ·  {harm} harmful / 198" if harm is not None else ""
        ax.annotate(f"{rt['p50']:.1f} ms{extra}", (rt["p95"], i), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.8, color=INK if kind == "best" else INK2,
                    fontweight="bold" if kind == "best" else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([fp.wrap(r[0], 2.6, 9) for r in rows], fontsize=8.8)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, max(r[1]["p95"] for r in rows) * 2.1)
    ax.set_xlabel("round trip per streamed word, ms (bar p50, whisker to p95)", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, note="220 commands, 1,158 words; words after the first · hatched = a probe trained on labelled prefixes")


def finding_voice5b(D):
    r = {kind: (rt, cp, a) for _, rt, cp, a, kind in voice5b(D)}
    ref, best, probe = r["ref"], r["best"], r["probe"]
    return (f"Streamed voice in the clean rerun: {ref[0]['p50']:.1f} ms per word on the production server, "
            f"{best[0]['p50']:.1f} ms in process; a 4B probe trained on prefixes takes {probe[0]['p50']:.1f} ms, with "
            f"{probe[2].get('harmful')} harmful actions instead of {ref[2].get('harmful')}")


# ================================================================== voice
FRACTIONS = ["0–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]


def draw_voice_partial(ctx, D):
    ax = ctx.ax()
    st = D.stream
    x = np.arange(5)
    ax.plot(x, st["gold"], "-o", color=TEZ, lw=2.2, ms=8, mec=SURFACE, mew=1.4, zorder=3)
    ax.plot(x, st["none"], "-s", color=NEUTRAL, lw=2.2, ms=8, mec=SURFACE, mew=1.4, zorder=3)
    for xi in (0, 4):
        vlab(ax, xi, st["gold"][xi], f3(st["gold"][xi]), fs=9, color=INK, weight="bold", dy=8)
        vlab(ax, xi, st["none"][xi], f3(st["none"][xi]), fs=9, color=INK2, dy=8)
    ax.set_xticks(x)
    ax.set_xticklabels(FRACTIONS, fontsize=9.5)
    ax.set_xlabel("share of the utterance heard", fontsize=10)
    ax.set_ylabel("share of partial transcripts", fontsize=10)
    ax.set_ylim(0, 1.05)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [line(label="top answer = the gold action", color=TEZ),
                    line(label="top answer = none (keep listening)", color=NEUTRAL, marker="s")], ncol=2,
               note=f"the {D.stream['n']} actionable commands, every word-by-word prefix of the final transcript")


def finding_voice_partial(D):
    st = D.stream
    return (f"On the first fifth of a command the top answer is 'none' ({f3(st['none'][0])} of prefixes) far more often than "
            f"the gold action ({f3(st['gold'][0])}); by the last fifth the gold action leads ({f3(st['gold'][4])}): a "
            f"confident 'none' on a prefix means keep listening")


def voice_items(D):
    V, SP = D.voice, D.policy
    return [("intent accuracy, full command", V.get("intent_accuracy")),
            ("accepting the second action\nof compound commands", V.get("intent_accuracy_alt_or_then_ok")),
            ("out-of-scope → none, precision", V.get("none_precision")),
            ("final action consistent\n(class-aware commit)", SP.get("final_action_consistent_rate")),
            ("1 − harmful rate", 1 - SP.get("harmful_rate", 0)),
            ("compound commands, residual\nread as a second decision", 22 / 22)]


def draw_voice_outcomes(ctx, D):
    ax = ctx.ax()
    items = voice_items(D)
    for i, (lab, v) in enumerate(items):
        bar(ax, i, v, 0.66, color=TEZ, horizontal=True)
        ax.annotate(f3(v), (v, i), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=9.2)
    ax.set_ylim(len(items) - 0.4, -0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("rate", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, note="compound commands: 22 of 22 (BENCHMARKS.md §3)")


def finding_voice_outcomes(D):
    V, SP = D.voice, D.policy
    return (f"Voice to action on 220 commands: intent accuracy {f3(V['intent_accuracy'])} "
            f"({f3(V['intent_accuracy_alt_or_then_ok'])} accepting the second action of compound commands), and "
            f"{SP['harmful']} harmful action in {SP['actionable']} actionable utterances with the class-aware commit")


BUDGET = [("ASR partial\n(base.en)", 40), ("decision\ncompute", 30), ("HTTP\n+ JSON", 15), ("policy\n+ slots", 1)]
BEFORE_SWA = 205


def draw_budget(ctx, D):
    ax = ctx.ax()
    cols = [NEUTRAL2, TEZ, TEZ_LIGHT, NEUTRAL]
    left = 0
    for (n, v), c in zip(BUDGET, cols):
        bar(ax, 0, v, 0.55, color=c, horizontal=True, left=left)
        if v >= 10:
            ax.annotate(f"{n}\n{v} ms", (left + v / 2, 0), ha="center", va="center", fontsize=8.8,
                        color=fp.text_on(c), linespacing=1.15)
        left += v
    ax.annotate("policy + slots < 1 ms", (left, 0), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.6, color=INK3)
    bar(ax, 1, BEFORE_SWA, 0.55, color=NEUTRAL2, horizontal=True, alpha=0.6)
    ax.annotate(f"decision compute before --swa-full: {BEFORE_SWA} ms per word", (BEFORE_SWA, 1), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["now", "before the\ncache fix"], fontsize=9.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xlim(0, 230)
    ax.set_xlabel("ms per streamed word", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, note="BENCHMARKS.md §3's per-word budget (≈ values, as printed there)")


def finding_budget(D):
    tot = sum(v for _, v in BUDGET[:3])
    return (f"About {tot} ms from a spoken word to an action: {BUDGET[0][1]} ms for the ASR partial, {BUDGET[1][1]} ms to "
            f"decide, {BUDGET[2][1]} ms of HTTP and under 1 ms of policy; before --swa-full the decision alone took "
            f"{BEFORE_SWA} ms")


def budget_source(D):
    s = "BENCHMARKS.md §3 (per-word budget; --swa-full)."
    v = (D.speed or {}).get("exp3_voice", {}).get("voice_12b_http_prod.json")
    if v:
        li = v["latency_incremental_words"]
        s += (f" §5b's clean rerun of the same loop measures {li['compute_ms']['p50']:.1f} ms compute and "
              f"{li['round_trip_ms']['p50']:.1f} ms round trip per word (results/speed/summary.json).")
    return s


def draw_asr(ctx, D):
    left, right = ctx.row((1, 2.2), gap=0.9)
    tags = [t for t in ("base-en", "small-en") if D.asr.get(t)]
    cols = {"base-en": TEZ, "small-en": NEUTRAL2}
    for i, t in enumerate(tags):
        v = D.asr[t]["partial_ms_p50"]
        bar(left, i, v, 0.6, color=cols[t])
        vlab(left, i, v, f"{v:.0f} ms", fs=9, color=INK if t == "base-en" else INK2)
    left.set_xticks(range(len(tags)))
    left.set_xticklabels([t.replace("-", ".") for t in tags], fontsize=9.5)
    left.set_ylim(0, 150)
    left.set_ylabel("ms per 200 ms chunk (p50)", fontsize=9.5)
    ctx.subtitle(left, "partial latency")
    mets = [("WER", lambda a: a["wer_mean"]), ("transcript\nnot exact", lambda a: 1 - a["exact_rate"]),
            ("intent differs from\nthe true text's", lambda a: 1 - a["intent_agree_asr_vs_text"])]
    w = 0.36
    for j, t in enumerate(tags):
        for i, (_, f) in enumerate(mets):
            v = f(D.asr[t])
            x = i - 0.2 + j * w * 1.1
            bar(right, x, v, w, color=cols[t])
            vlab(right, x, v, pct(v, 1), fs=8.6, color=INK if t == "base-en" else INK2)
    right.set_xticks(range(len(mets)))
    right.set_xticklabels([m[0] for m in mets], fontsize=9.3)
    right.set_ylim(0, 0.32)
    right.set_yticks(np.arange(0, 0.31, 0.1))
    right.set_yticklabels([pct(x) for x in np.arange(0, 0.31, 0.1)])
    ctx.subtitle(right, "error rates (lower is better)")
    for a in (left, right):
        grid(a)
        strip(a)
    ctx.legend(left, [patch(color=cols[t], label=f"faster-whisper {t.replace('-', '.')}") for t in tags], ncol=2)


def finding_asr(D):
    b, s = D.asr["base-en"], D.asr["small-en"]
    return (f"faster-whisper base.en returns a partial transcript in {b['partial_ms_p50']:.0f} ms (p50) at "
            f"{pct(b['wer_mean'], 1)} WER, and the intent read from it agrees with the true text on "
            f"{pct(b['intent_agree_asr_vs_text'], 1)}; small.en takes {s['partial_ms_p50']:.0f} ms for "
            f"{pct(s['wer_mean'], 1)} WER")


# ================================================================== SemIf
def draw_semif(ctx, D):
    ax = ctx.ax()
    pts = [(n, m) for n, m in D.semif.items() if m]
    for i, (n, m) in enumerate(pts):
        v = m["mean_family_balanced_accuracy"]
        lo, hi = m["mfba_ci95"]
        c = TEZ if n == "Gemma 4 12B Q8" else TEZ_LIGHT if n.startswith("Gemma 4") else NEUTRAL2
        bar(ax, i, v, 0.62, color=c)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK2, capsize=4, lw=1.2, zorder=4)
        ax.annotate(f3(v), (i, 0), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9.5,
                    color=fp.text_on(c), fontweight="bold" if c == TEZ else "normal")
    refline(ax, SEMIF_PUB["4B"], f"SemIf published, Qwen3.5-4B {f3(SEMIF_PUB['4B'])}", x=0.01, ha="left")
    refline(ax, SEMIF_PUB["27B"], f"SemIf published, EXL3 27B {f3(SEMIF_PUB['27B'])}", x=0.01, ha="left", ls=(0, (1, 2)))
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([n.replace(" Q", "\nQ").replace(" BF16", "\nBF16") for n, _ in pts], fontsize=9.3)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean-family balanced accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, note="95 % interval by source-group bootstrap · chance is about 0.33")


def finding_semif(D):
    pts = {n: m["mean_family_balanced_accuracy"] for n, m in D.semif.items() if m}
    lo = min(pts, key=pts.get)
    return (f"Symbol binding is scale-gated: on SemIf's authored144, Gemma 4 12B Q8 scores {f3(pts['Gemma 4 12B Q8'])} "
            f"(SemIf published {f3(SEMIF_PUB['4B'])} for its 4B) and {lo} {f3(pts[lo])}, near chance")


RECIPES = [("raw", "orig"), ("average of\n2 orderings", "perm2_mean"), ("average of\n6 orderings", "perm6_mean"),
           ("6 orderings\n+ temperature", "perm6_mean+temperature_oof"), ("Zhao, N/A\nplaceholder", "zhao_NA_only"),
           ("Zhao, 3\nplaceholders", "zhao_3placeholder_avg")]


def draw_recipes(ctx, D):
    ax = ctx.ax()
    P = D.perm
    rec = [(n, k) for n, k in RECIPES if k in P]
    best = min((P[k]["nll"] for n, k in rec if "zhao" not in k))
    for i, (n, k) in enumerate(rec):
        v, a = P[k]["nll"], P[k]["accuracy"]
        c = BAD if "zhao" in k else TEZ if v == best else NEUTRAL2 if k == "orig" else TEZ_LIGHT
        bar(ax, i, v, 0.62, color=c)
        vlab(ax, i, v, f"NLL {v:.3f}\naccuracy {a:.3f}", fs=8.4, color=INK if v == best else INK2,
             weight="bold" if v == best else "normal")
    ax.set_xticks(range(len(rec)))
    ax.set_xticklabels([n for n, _ in rec], fontsize=9)
    ax.set_ylim(0, max(P[k]["nll"] for _, k in rec) * 1.3)
    ax.set_ylabel("NLL (lower is better)", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=NEUTRAL2, label="raw letters"), patch(color=TEZ_LIGHT, label="permutation averaging"),
                    patch(color=TEZ, label="best recipe"),
                    patch(color=BAD, label="contextual calibration (Zhao et al.): harmful here")], ncol=3,
               note="Gemma 4 12B Q8_0 on authored144, out-of-fold temperature; accuracy is plain accuracy")


def finding_recipes(D):
    P = D.perm
    return (f"On SemIf, averaging six option orderings plus a temperature cuts NLL from {P['orig']['nll']:.3f} to "
            f"{P['perm6_mean+temperature_oof']['nll']:.3f} at the same accuracy; Zhao's contextual calibration drops "
            f"accuracy from {P['orig']['accuracy']:.3f} to {P['zhao_NA_only']['accuracy']:.3f}")


def draw_cascade(ctx, D):
    ax = ctx.ax()
    Cs = D.cascade
    sw = Cs["sweeps"]["perm_agreement_gate"]
    ax.plot([s["ms"] for s in sw], [s["acc"] for s in sw], "-", color=fp.GRID, lw=3, zorder=1)
    pts = [("small_only", "Gemma 3 4B", NEUTRAL2, "o"), ("small_perm2_only", "4B, two orderings", NEUTRAL2, "s"),
           ("agreement_only", "cascade: 4B, escalate to\nthe 12B on disagreement", NEUTRAL, "^"),
           ("large_only", "Gemma 4 12B alone", TEZ, "o")]
    for k, lab, c, mk in pts:
        v = Cs[k]
        ax.plot(v["ms"], v["acc"], mk, ms=11, color=c, mec=SURFACE, mew=1.5, zorder=3)
        left = k in ("agreement_only", "large_only")
        ax.annotate(f"{lab}\n{f3(v['acc'])} · {v['ms']:.0f} ms", (v["ms"], v["acc"]), xytext=(-10 if left else 9, -2),
                    textcoords="offset points", va="top", ha="right" if left else "left", fontsize=8.8,
                    color=INK if c == TEZ else INK2, fontweight="bold" if c == TEZ else "normal", linespacing=1.15)
    ax.set_xlim(0, max(s["ms"] for s in sw) * 1.05)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("mean ms per decision", fontsize=10)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    grid(ax, "x")
    strip(ax, keep=("bottom", "left"))
    ctx.legend(ax, note="SemIf authored144 · grey curve: the cascade over its escalation threshold")


def finding_cascade(D):
    Cs = D.cascade
    a, L = Cs["agreement_only"], Cs["large_only"]
    return (f"A small-to-large cascade is not worth it: escalating when the 4B's two orderings disagree gives "
            f"{f3(a['acc'])} at {a['ms']:.0f} ms, against {f3(L['acc'])} at {L['ms']:.0f} ms for the 12B alone")


# ================================================================== probes
def probe_items(D):
    HP, HP12 = D.hp, D.hp12
    items = [("4B letters\n(its own answer)", HP["letter_logits"]["accuracy"], dict(color=NEUTRAL2), "letters")]
    for k, lab in (("layer-1", "4B probe,\nfinal layer"), ("layer-4", "4B probe,\nlayer −4"), ("layer-8", "4B probe,\nlayer −8"),
                   ("layer-12", "4B probe,\nlayer −12")):
        items.append((lab, HP[k]["logreg_acc"], dict(color=TEZ, hatch=HATCH), k))
    if HP12:
        items.append(("12B probe,\nfinal layer\n(llama-server)", HP12["logreg_acc"], dict(color=TEZ_DEEP, hatch=HATCH), "12b"))
    return items


def draw_probes(ctx, D):
    ax = ctx.ax()
    items = probe_items(D)
    best = max(v for _, v, _, k in items if k != "letters")
    for i, (lab, v, st, _) in enumerate(items):
        bar(ax, i, v, 0.62, horizontal=True, **st)
        ax.annotate(f3(v), (v, i), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9.2,
                    color=INK if v == best else INK2, fontweight="bold" if v == best else "normal", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    refs = [(0.766, LAYA_TD, (0, (4, 3)), "laya-typed-decisions (fine-tuned) 0.766"),
            (JEV_ACC["typed_decisions"], INK3, (0, (1, 2)), f"Jev (published) {f3(JEV_ACC['typed_decisions'])}")]
    for x, c, ls, _ in refs:
        ax.axvline(x, color=c, lw=1.4, ls=ls, zorder=2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0].replace("\n", " ").replace("probe, final layer (llama", "probe, final layer\n(llama")
                        for it in items], fontsize=9.2)
    ax.set_ylim(len(items) - 0.4, -0.6)
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("accuracy, typed-decisions (2,000 decisions)", fontsize=10)
    grid(ax, "x")
    strip(ax)
    ctx.legend(ax, [patch(color=NEUTRAL2, label="the model's own letter readout"),
                    patch(color=TEZ, hatch=HATCH, label="probe on Qwen3.5-4B"),
                    patch(color=TEZ_DEEP, hatch=HATCH, label="probe on Gemma 4 12B")]
               + [line(color=c, marker="", ls=ls, lw=1.4, label=lab) for _, c, ls, lab in refs], ncol=2, fs=8.8)


def finding_probes(D):
    items = probe_items(D)
    lab, v = max(((lab, v) for lab, v, _, k in items if k.startswith("layer")), key=lambda t: t[1])
    return (f"A linear probe on the frozen Qwen3.5-4B reads typed-decisions at {f3(v)} "
            f"({lab.split(',')[1].strip()}), above the fine-tuned laya-typed-decisions (0.766), from a model whose own "
            f"letters score {f3(D.hp['letter_logits']['accuracy'])}")


def plateau(sw, tol=0.004):
    ls = sorted(int(k) for k in sw["layer_sweep"])
    best = max(sw["layer_sweep"][str(k)]["acc"] for k in ls)
    on = [k for k in ls if sw["layer_sweep"][str(k)]["acc"] >= best - tol]
    return best, min(on), max(on)


def draw_depth(ctx, D):
    ax = ctx.ax()
    SW, S12 = D.sweep, D.sweep12
    ls = sorted(int(k) for k in SW["layer_sweep"])
    n4 = 32                                             # Qwen3.5-4B's 32 blocks (layer 0 = the embeddings)
    ax.plot([k / n4 for k in ls], [SW["layer_sweep"][str(k)]["acc"] for k in ls], "-o", color=TEZ, lw=2, ms=5,
            mec=SURFACE, mew=1, label="Qwen3.5-4B (bf16), layer of 32")
    if S12:
        l12 = sorted(int(k) for k in S12["layer_sweep"])
        ax.plot([k / 48 for k in l12], [S12["layer_sweep"][str(k)]["acc"] for k in l12], "-s", color=TEZ_DEEP, lw=2,
                ms=5, mec=SURFACE, mew=1, label="Gemma 4 12B (NF4, z-scored), layers 0–40 of 48")
        ax.plot([1.0], [D.hp12["logreg_acc"]], "D", color=TEZ_DEEP, ms=8, zorder=4)
        ax.annotate(f"12B final layer via\nllama-server {f3(D.hp12['logreg_acc'])}", (1.0, D.hp12["logreg_acc"]),
                    xytext=(-8, -6), textcoords="offset points", ha="right", va="top", fontsize=8.5, color=TEZ_DEEP)
    best, a, b = plateau(SW)
    ax.axvspan(a / n4, b / n4, color=TEZ, alpha=0.07, lw=0)
    ax.annotate(f"4B plateau, layers {a}–{b}", ((a + b) / 2 / n4, 0.42), ha="center", fontsize=8.6, color=TEZ_INK)
    refline(ax, SW["letter_logits"]["acc"], f"4B letters {f3(SW['letter_logits']['acc'])}", x=0.99, ha="right",
            va="top", color=NEUTRAL)
    refline(ax, 0.766, "laya-typed-decisions 0.766", color=LAYA_TD, x=0.01, ha="left")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.4, 0.84)
    ax.set_xlabel("depth read (share of the model's layers)", fontsize=10)
    ax.set_ylabel("probe accuracy", fontsize=10)
    grid(ax)
    strip(ax, keep=("bottom", "left"))
    h = [line(color=TEZ, label="Qwen3.5-4B (bf16), 32 layers"),
         line(color=TEZ_DEEP, marker="s", label="Gemma 4 12B (NF4, z-scored), layers 0–40 of 48")]
    ctx.legend(ax, h, ncol=2)


def finding_depth(D):
    best, a, b = plateau(D.sweep)
    S12 = D.sweep12
    l12 = max(S12["layer_sweep"], key=lambda k: S12["layer_sweep"][k]["acc"])
    return (f"The 4B probe plateaus at {f3(best)} from layer {a} to {b} of 32, well below the top; the 12B peaks at "
            f"{f3(S12['layer_sweep'][l12]['acc'])} at layer {l12} of 48, so the bigger backbone buys no better probe")


def draw_efficiency(ctx, D):
    ax = ctx.ax()
    for sw, c, mk, lab in ((D.sweep, TEZ, "o", "Qwen3.5-4B, layer 26"), (D.sweep12, TEZ_DEEP, "s", "Gemma 4 12B, layer 34")):
        if not sw:
            continue
        de = sw["data_efficiency"]
        xs = sorted(int(k) for k in de)
        ys = [de[str(x)]["mean"] for x in xs]
        ax.errorbar(xs, ys, yerr=[de[str(x)]["std"] for x in xs], fmt="-" + mk, color=c, lw=2, ms=6, capsize=3,
                    mec=SURFACE, mew=1, label=lab)
        if c == TEZ:
            for x, y in zip(xs, ys):
                ax.annotate(f3(y), (x, y), xytext=(0, -12), textcoords="offset points", ha="center", va="top",
                            fontsize=8.8, color=INK, fontweight="bold")
    refline(ax, 0.766, "laya-typed-decisions, fine-tuned on all 300 per question: 0.766", color=LAYA_TD, x=0.01, ha="left")
    refline(ax, JEV_ACC["typed_decisions"], f"Jev (published) {f3(JEV_ACC['typed_decisions'])}", x=0.99, va="top",
            ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([10, 25, 50, 100, 200])
    ax.set_xticklabels(["10", "25", "50", "100", "200"])
    ax.set_ylim(0.62, 0.83)
    ax.set_xlabel("labelled rows per question (3 seeds, ± one s.d.)", fontsize=10)
    ax.set_ylabel("probe accuracy", fontsize=10)
    grid(ax)
    strip(ax, keep=("bottom", "left"))
    ctx.legend(ax, [line(color=TEZ, label="Qwen3.5-4B, layer 26"), line(color=TEZ_DEEP, marker="s", label="Gemma 4 12B, layer 34")],
               ncol=2)


def finding_efficiency(D):
    de = D.sweep["data_efficiency"]
    return (f"With 50 labelled rows per question the 4B probe reaches {f3(de['50']['mean'])}, with 200 "
            f"{f3(de['200']['mean'])}, past laya-typed-decisions' 0.766 after fine-tuning on all 300")


def draw_early(ctx, D):
    top, bottom = ctx.stack((1, 1), gap=0.45)
    SW, EE = D.sweep, D.early
    ls = sorted(int(k) for k in SW["layer_sweep"])
    top.plot(ls, [SW["layer_sweep"][str(k)]["acc"] for k in ls], "-o", color=TEZ, lw=2, ms=4.5, mec=SURFACE, mew=1)
    ds = sorted(int(k.split("_")[1]) for k in EE if k.startswith("depth_"))
    bottom.plot(ds, [EE[f"depth_{d}"]["speedup"] for d in ds], "-s", color=NEUTRAL, lw=2, ms=7, mec=SURFACE, mew=1)
    for d in ds:
        vlab(bottom, d, EE[f"depth_{d}"]["speedup"], f"{EE[f'depth_{d}']['speedup']:.2f}×", fs=8.6, dy=7)
        a = SW["layer_sweep"].get(str(d), {}).get("acc")
        if a is not None:
            top.plot(d, a, "o", ms=9, color=TEZ, mec=SURFACE, mew=1.3, zorder=4)
            vlab(top, d, a, f3(a), fs=8.4, dy=7, color=INK)
    for a in (top, bottom):
        a.axvspan(20, 28, color=TEZ, alpha=0.07, lw=0)
        grid(a)
        strip(a, keep=("bottom", "left"))
    top.set_ylim(0.45, 0.86)
    top.set_ylabel("probe accuracy", fontsize=9.5)
    bottom.set_ylim(0.8, 2.25)
    bottom.set_ylabel("prefill speed-up", fontsize=9.5)
    bottom.set_xlabel("layers kept (of 32)", fontsize=10)
    ctx.subtitle(top, "probe accuracy at that layer")
    ctx.subtitle(bottom, "prefill speed-up when the model is cut there (eager PyTorch, 200 prompts)")
    ctx.legend(top, note="Qwen3.5-4B, typed-decisions")


def finding_early(D):
    EE = D.early
    best, a, b = plateau(D.sweep)
    return (f"Reading early is cheap latency: the 4B's probe accuracy plateaus from layer {a} to {b}, and the model cut to "
            f"24 or 20 of its 32 layers prefills {EE['depth_24']['speedup']:.2f}× or {EE['depth_20']['speedup']:.2f}× "
            f"faster")


def task_rows(D):
    TP = D.tp
    out = []
    for t in TP:
        best = max(v["acc"] for k, v in TP[t].items() if k.startswith("probe_layer"))
        out.append((t, TP[t].get("letter_acc", {}).get("acc") if isinstance(TP[t].get("letter_acc"), dict)
                    else TP[t].get("letter_acc"), best, acc(D, t, "tez"), acc(D, t, "laya-td")))
    return out


def draw_task_probes(ctx, D):
    ax = ctx.ax()
    rows = task_rows(D)
    w = 0.2
    series = [(1, dict(color=NEUTRAL2), "4B letters (zero-shot)"), (2, dict(color=TEZ, hatch=HATCH), "4B probe, best layer"),
              (3, dict(color=TEZ), "12B letters (zero-shot)"), (4, dict(color=LAYA_TD), "laya-typed-decisions (fine-tuned)")]
    for j, (idx, st, _) in enumerate(series):
        for i, r in enumerate(rows):
            v = r[idx]
            if v is None:
                continue
            x = i - 0.3 + j * w
            bar(ax, x, v, w * 0.92, **st)
            if idx == 2:
                vlab(ax, x, v, f3(v), fs=8.2, color=INK, weight="bold", rot=90, dy=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([TASK_NAME.get(r[0], r[0]) for r in rows], fontsize=9.3)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(label=s[2], **s[1]) for s in series], ncol=2,
               note="2,000 labelled train rows per task, same test rows as §1 · the MASSIVE-en probe chooses among all 60 "
                    "intents (the letters among 20) · no 4B letter readout on Banking77 and MASSIVE-en · 12B on Banking77: "
                    "the chunked tournament")


def finding_task_probes(D):
    rows = task_rows(D)
    wins = [r for r in rows if r[4] is not None and r[2] > r[4]]
    b77 = next((r[2] for r in rows if r[0] == "banking77"), None)
    return (f"One probe per task on the frozen 4B beats laya-typed-decisions on {len(wins)} of {len(rows)} of Laya's public "
            f"tasks" + (f" and reaches {f3(b77)} on Banking77, against Jev's published {f3(JEV_ACC['banking77'])}" if b77 else ""))


# ================================================================== other levers
def draw_shortlist(ctx, D):
    ax = ctx.ax()
    RN = D.rn
    ks = sorted(int(k) for k in RN["recall_at"])
    ax.plot(ks, [RN["recall_at"][str(k)] for k in ks], "-", color=NEUTRAL2, lw=2)
    runs = sorted(((v["k"], v["acc"], v["ms_p50"]) for v in RN["runs"].values()), key=lambda t: t[0])
    ax.plot([r[0] for r in runs], [r[1] for r in runs], "-o", color=TEZ, lw=2.2, ms=8, mec=SURFACE, mew=1.3)
    for k, a, ms in runs:
        ax.annotate(f"{f3(a)}\n{ms:.0f} ms", (k, 0.02), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8.8, color=INK, fontweight="bold", linespacing=1.15)
        ax.plot([k, k], [0.585, a], color=TEZ, lw=0.8, ls=(0, (1, 2)), zorder=1)
    refline(ax, RN["tournament_ref_acc"], f"chunked tournament over all 77, 5 passes: {f3(RN['tournament_ref_acc'])}",
            color=TEZ_INK, x=0.01, ha="left")
    refline(ax, RN["retrieval_only_acc"], f"MiniLM top-1 alone {f3(RN['retrieval_only_acc'])}", color=NEUTRAL, x=0.99,
            va="top")
    refline(ax, JEV_ACC["banking77"], f"Jev (published) {f3(JEV_ACC['banking77'])}", x=0.99)
    ax.set_xlim(0, 26)
    ax.set_ylim(0.52, 1.0)
    ax.set_xlabel("shortlist size k", fontsize=10)
    ax.set_ylabel("accuracy / recall", fontsize=10)
    grid(ax)
    strip(ax, keep=("bottom", "left"))
    ctx.legend(ax, [line(color=NEUTRAL2, marker="", label="shortlist recall@k (MiniLM)"),
                    line(color=TEZ, label="Tez on the shortlist, one pass")], ncol=2)


def finding_shortlist(D):
    RN = D.rn
    k20 = max(RN["runs"].values(), key=lambda v: v["k"])
    return (f"On Banking77's 77 intents, a MiniLM shortlist of {k20['k']} lets the 12B reach {f3(k20['acc'])} in one pass "
            f"(the chunked tournament: {f3(RN['tournament_ref_acc'])} in five); Jev's published {f3(JEV_ACC['banking77'])} "
            f"stays ahead")


RULE_NAME = {"single": "one pass", "top4": "keep top 4", "top8": "keep top 8", "above_mean_logp": "above mean log-p",
             "mass95": "95 % of the mass"}


def draw_elimination(ctx, D):
    ax = ctx.ax()
    OE = D.oe
    tasks = list(OE)
    rules = list(OE[tasks[0]]["summary"])
    w = 0.8 / len(rules)
    for j, r in enumerate(rules):
        for i, t in enumerate(tasks):
            v = OE[t]["summary"][r]["acc"]
            x = i - 0.4 + w / 2 + j * w
            bar(ax, x, v, w * 0.92, color=TEZ if r == "single" else NEUTRAL2)
            vlab(ax, x, v, f"{v:.3f}", fs=7.6, rot=90, dy=2, color=INK if r == "single" else INK2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_NAME.get(t, t).replace("typed-decisions", "typed-decisions\nchoice, ≥ 3 options") for t in tasks],
                       fontsize=9.3)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy", fontsize=10)
    grid(ax)
    strip(ax)
    ctx.legend(ax, [patch(color=TEZ, label="one pass (as shipped)"),
                    patch(color=NEUTRAL2, label="eliminate, then rescore the survivors")], ncol=2,
               note="grey bars left to right: " + ", ".join(RULE_NAME.get(r, r) for r in rules[1:]))


def finding_elimination(D):
    OE = D.oe
    d = max(abs(OE[t]["summary"][r]["acc"] - OE[t]["summary"]["single"]["acc"]) for t in OE for r in OE[t]["summary"])
    return (f"Eliminating options and rescoring the survivors moves accuracy by at most {100 * d:.1f} points on any task "
            f"and costs up to a second pass: a negative result")


# ================================================================== registry, in reading order
SRC_H2H = "BENCHMARKS.md §1 (experiments/bench_h2h.py, results/h2h/summary.json)"
H2H = ("results/h2h/summary.json",)

PANELS = [
    Panel("public_tasks", "Accuracy on Laya's public tasks, same rows", finding_public,
          "Accuracy on each public task: zero-shot Tez, Laya's three checkpoints on the same rows, Jev where published",
          H2H_SETUP, f"{SRC_H2H}; Jev: third-party published, as collected there.", draw_public, h=4.6, left=1.45,
          needs=H2H, section="Accuracy against Laya and Jev"),
    Panel("typed_decisions", "typed-decisions (2,000 decisions)", finding_td,
          "typed-decisions test split: 400 cases, 2,000 decisions in four workflows", H2H_SETUP,
          f"{SRC_H2H}; 4 examples: results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json (§4b); teacher ceiling, "
          "majority class and random: Laya's README (published).", draw_td, h=4.3, left=1.95, needs=H2H,
          section="Accuracy against Laya and Jev"),
    Panel("typed_decisions_by_workflow", "typed-decisions by workflow", finding_td_workflow,
          "Accuracy on each of typed-decisions' four workflows", H2H_SETUP,
          "BENCHMARKS.md §1 (typed-decisions by workflow; results/h2h/summary.json).", draw_td_workflow, h=3.4, needs=H2H,
          section="Accuracy against Laya and Jev"),
    Panel("teacher_distribution", "Teacher-distribution match (soft acc ↑, Brier ↓)", finding_teacher,
          "typed-decisions scored against the teacher's probability distribution rather than its top answer", H2H_SETUP,
          "BENCHMARKS.md §1 (soft accuracy and Brier vs teacher distribution; results/h2h/summary.json); Jev: third-party "
          "published.", draw_teacher, h=3.2, needs=H2H, section="Accuracy against Laya and Jev"),
    Panel("languages_massive", "One frozen decoder vs two trained encoders, per language", finding_languages,
          "MASSIVE intent accuracy in each of 11 languages", H2H_SETUP,
          "BENCHMARKS.md §1 Languages (results/h2h/summary.json).", draw_languages, h=4.2, left=0.75, needs=H2H,
          section="Languages"),
    Panel("calibration", "Calibration: T = 1, as shipped and one temperature per task", finding_calibration,
          "Mean ECE-15 over the head-to-head tasks, as shipped and after one temperature per task, with Tez also at "
          "T = 1, before its default temperature", H2H_SETUP,
          "BENCHMARKS.md §1 (Calibration, order robustness, latency), §5c (Tez's default temperature; "
          "results/calibration/default_temperature.json, anchor_28); Jev: third-party published.",
          draw_calibration, h=3.6, needs=H2H, section="Calibration, robustness and abstention"),
    Panel("order_flip", "Option-order robustness (first 100 rows)", finding_flip,
          "Share of answers that change when the options are presented in reverse order", H2H_SETUP,
          "BENCHMARKS.md §1 (option-order flip; results/h2h/summary.json flip_rate); Jev: third-party published.",
          draw_flip, h=3.3, needs=H2H, section="Calibration, robustness and abstention"),
    Panel("conformal", "Conformal act / escalate (typed-decisions)", finding_conformal,
          "Share of decisions each system acts on (conformal set of one answer) and its accuracy there, at 90 % and 95 % "
          "target coverage", H2H_SETUP, "BENCHMARKS.md §4 (experiments/conformal_td.py, "
          "results/conformal_typed_decisions.json).", draw_conformal, h=3.6, left=2.45,
          needs=("results/conformal_typed_decisions.json",), section="Calibration, robustness and abstention"),
    Panel("batch_calibration", "Batch Calibration (Zhou 2023), NLL", finding_bc,
          "NLL of Tez's letters before and after Batch Calibration, fitted out of fold", H2H_SETUP,
          "BENCHMARKS.md §4 (experiments/batch_calibration.py, results/batch_calibration.json).", draw_bc, h=4.0,
          left=1.3, needs=("results/batch_calibration.json",), section="Calibration, robustness and abstention"),
    Panel("latency_per_task", "Latency, same GPU (Tez over HTTP, question first)", finding_latency,
          "p50 milliseconds per decision, one question per call, on five of the head-to-head tasks", H2H_SETUP,
          f"{SRC_H2H}. Measured with the question-first layout; BENCHMARKS.md §5b re-measures the runtime's paths.",
          draw_latency, h=3.4, needs=H2H, section="Speed"),
    Panel("speed_many_questions", "Many questions about one state (§5b)", finding_speed5b,
          "p50 per call with 50 distinct questions about one state; row labels as BENCHMARKS.md §5b prints them",
          SPEED_SETUP + " The state-first rows are the study's direct /completion calls and in-process code; measured "
          "later through tez serve on an idle machine, 50 questions took 2,878 ms over llama-server and 1,147 ms in "
          "process.",
          "BENCHMARKS.md §5b (Many questions about one state; results/speed/summary.json, results/speed/tables.md).",
          draw_speed5b, h=3.9, left=2.95, needs=("results/speed/summary.json",), section="Speed"),
    Panel("voice_partial_transcripts", "Voice: partial-transcript predictions", finding_voice_partial,
          "The top answer on each word-by-word prefix of a command, by the share of the command heard", VOICE_SETUP,
          "BENCHMARKS.md §3 (streaming; experiments/stream_analysis.py's buckets over "
          "results/voicefast_final_last_gemma4-12b-q8_0_stream.jsonl).", draw_voice_partial, h=3.3,
          needs=("results/voicefast_final_last_gemma4-12b-q8_0_stream.jsonl",), section="Voice"),
    Panel("voice_outcomes", "Voice → action outcomes (220 commands)", finding_voice_outcomes,
          "Rates on the 220 commands: intent, out-of-scope, the class-aware commit policy and compound commands",
          VOICE_SETUP, "BENCHMARKS.md §3 (results/voicefast_final_last_gemma4-12b-q8_0.summary.json, "
          "results/stream_policy_final_last_gemma4-12b-q8_0.json).", draw_voice_outcomes, h=3.4, left=2.35,
          needs=("results/voicefast_final_last_gemma4-12b-q8_0.summary.json",
                 "results/stream_policy_final_last_gemma4-12b-q8_0.json"), section="Voice"),
    Panel("voice_budget", "Per-word budget (~85 ms word → action)", finding_budget,
          "Where the time goes between a spoken word and an action, per streamed word", VOICE_SETUP, budget_source,
          draw_budget, h=2.0, left=1.15, section="Voice"),
    Panel("voice_per_word", "Streamed voice, word to action (§5b)", finding_voice5b,
          "Round trip per streamed word; row labels as BENCHMARKS.md §5b prints them", SPEED_SETUP,
          "BENCHMARKS.md §5b (Streamed voice: word to action; results/speed/summary.json).", draw_voice5b, h=3.6,
          left=2.85, needs=("results/speed/summary.json",), section="Voice"),
    Panel("asr", "ASR stage (faster-whisper, synthetic speech)", finding_asr,
          "faster-whisper on the 220 commands synthesised with Windows SAPI, 200 ms chunks", VOICE_SETUP,
          "BENCHMARKS.md §3 (ASR; results/asr_bench_*_gemma4-12b-q8_0.summary.json).", draw_asr, h=3.0,
          needs=("results/asr_bench_base-en_gemma4-12b-q8_0.summary.json",), section="Voice"),
    Panel("semif_scale", "Scale-gated symbol binding (SemIf, 95 % CI)", finding_semif,
          "Mean-family balanced accuracy on SemIf's authored144 by model", SEMIF_SETUP,
          "BENCHMARKS.md §2 (results/authored144_*.metrics.json); SemIf's published figures from its repository.",
          draw_semif, h=3.6, needs=("results/authored144_gemma4-12b-q8_0.metrics.json",), section="SemIf's benchmark"),
    Panel("semif_calibration_recipes", "Calibration recipes, Gemma 4 12B Q8", finding_recipes,
          "NLL and accuracy of the 12B's letters under each calibration recipe", SEMIF_SETUP,
          "BENCHMARKS.md §2 (calibration recipes; results/perm_analysis_gemma4-12b-q8_0.json).", draw_recipes, h=3.6,
          needs=("results/perm_analysis_gemma4-12b-q8_0.json",), section="SemIf's benchmark"),
    Panel("cascade", "Small → large cascade: not worth it", finding_cascade,
          "Accuracy against mean milliseconds per decision on authored144", SEMIF_SETUP,
          "BENCHMARKS.md §4 (experiments/cascade_analysis.py, results/cascade_authored144.json).", draw_cascade, h=3.5,
          needs=("results/cascade_authored144.json",), section="SemIf's benchmark"),
    Panel("probes_typed_decisions", "Probes on frozen hidden states (typed-decisions)", finding_probes,
          "typed-decisions accuracy of the model's own letters and of logistic probes on its hidden state", PROBE_SETUP,
          "BENCHMARKS.md §4b (Hidden-state probe; results/hidden_probe_qwen35-4b.json, "
          "results/hidden_probe_server_gemma4-12b-q8_0.json); Jev: third-party published.", draw_probes, h=3.6, left=2.0,
          needs=("results/hidden_probe_qwen35-4b.json",), section="Probes"),
    Panel("probe_depth", "Probe accuracy by depth, 4B vs 12B", finding_depth,
          "Probe accuracy at each layer, by the share of the model's depth read", PROBE_SETUP,
          "BENCHMARKS.md §4b (layer sweep, probe backbone; results/hidden_probe_sweep_qwen35-4b.json, "
          "results/hidden_probe_sweep_gemma4-12b-nf4_std.json).", draw_depth, h=3.6,
          needs=("results/hidden_probe_sweep_qwen35-4b.json", "results/hidden_probe_sweep_gemma4-12b-nf4_std.json"),
          section="Probes"),
    Panel("probe_data_efficiency", "Probe data efficiency", finding_efficiency,
          "Probe accuracy against labelled rows per question", PROBE_SETUP,
          "BENCHMARKS.md §4b (data efficiency; results/hidden_probe_sweep_*.json); Jev: third-party published.",
          draw_efficiency, h=3.4, needs=("results/hidden_probe_sweep_qwen35-4b.json",), section="Probes"),
    Panel("early_readout", "Early readout: probe accuracy and prefill speed-up (4B)", finding_early,
          "Probe accuracy at a layer, and the prefill speed-up of a model cut to that many layers", PROBE_SETUP,
          "BENCHMARKS.md §4b (early readout; results/early_exit_qwen35-4b.json, "
          "results/hidden_probe_sweep_qwen35-4b.json).", draw_early, h=4.2,
          needs=("results/early_exit_qwen35-4b.json", "results/hidden_probe_sweep_qwen35-4b.json"), section="Probes"),
    Panel("task_probes", "One probe per task, 2,000 rows", finding_task_probes,
          "Accuracy on Laya's public tasks: the 4B's letters and its probe, the 12B's letters and the fine-tuned "
          "laya-typed-decisions", PROBE_SETUP.replace("fitted on the typed-decisions train split; test split, 2,000 "
                                                      "decisions", "2,000 labelled train rows per task"),
          "BENCHMARKS.md §4b (Task probes; results/hidden_probe_tasks_qwen35-4b.json, results/h2h/summary.json); Jev: "
          "third-party published.", draw_task_probes, h=3.6, needs=("results/hidden_probe_tasks_qwen35-4b.json",) + H2H,
          section="Probes"),
    Panel("banking77_shortlist", "Banking77: shortlist, then decide", finding_shortlist,
          "Accuracy of the 12B's letters on a MiniLM shortlist of k intents, and the shortlist's recall", H2H_SETUP,
          "BENCHMARKS.md §4b (Retrieval-narrowed Banking77; results/retrieval_narrow_banking77_minilm.json); Jev: "
          "third-party published.", draw_shortlist, h=3.6, needs=("results/retrieval_narrow_banking77_minilm.json",),
          section="Other levers"),
    Panel("option_elimination", "Option elimination: no gain", finding_elimination,
          "Accuracy of one pass against eliminating options and rescoring the survivors", H2H_SETUP,
          "BENCHMARKS.md §4b (Option elimination; results/option_elimination_gemma4-12b-q8_0.json).", draw_elimination,
          h=3.4, needs=("results/option_elimination_gemma4-12b-q8_0.json",), section="Other levers"),
]


def main():
    D = load()
    print("panels:")
    for p in PANELS:
        fp.render_panel(p, D)
    print("grid:")
    out = fp.ROOT / "docs" / "figures"
    fp.render_grid(PANELS, D, out / "tez_dashboard.png", out / "tez_dashboard.pdf",
                   "Tez: every measurement, one chart per finding",
                   "Zero-shot frozen Gemma 4 12B Q8_0, one forward pass, RTX 5080 laptop (16 GB). Laya's checkpoints ran on "
                   "the same rows and GPU; Jev's figures are third-party published, never measured here. Each panel is "
                   "also a chart of its own in docs/figures/panels/, titled with its finding.", ncol=4)
    fp.write_index()


if __name__ == "__main__":
    main()
