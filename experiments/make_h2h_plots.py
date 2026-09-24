"""Render the head-to-head figures from results/h2h/summary.json (matplotlib, PNG + PDF).

  1. tez_vs_laya_vs_jev.png   accuracy on every public dataset: Tez (zero-shot) vs Laya routed vs Jev (published)
  2. languages.png            MASSIVE intent per language: Tez vs laya-en vs laya-ml (dot plot, Laya's chart)
  3. calibration.png          ECE as shipped vs after temperature refit, every model; Tez as shipped = its default
                              temperature fitted without the task (results/calibration/default_temperature.json),
                              with its temperature-1 ECE outlined beside it
  4. typed_decisions.png      the 2,000-decision benchmark: Tez zero-shot vs Laya base/fine-tuned vs Jev vs ceilings
  5. order_flip.png           option-order flip rate on choice tasks
  6. speed.png                ms per decision on this machine (same GPU) -- note Laya numbers are its own runtime
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JEV = {  # third-party published, never measured here (AbdelStark/jev-benchmarks, nibzard/decision-model-benchmark, Laya README)
    "ag_news": 0.910, "emotion": 0.480, "banking77": 0.870, "typed_decisions": 0.727,
}
COL = {"tez": "#0E7C7B", "laya-en": "#4A7FB5", "laya-ml": "#E07A3F", "laya-td": "#3C8D5A", "jev": "#9A9A9A"}
NAME = {"tez": "Tez (Gemma 4 12B, zero-shot)", "laya-en": "laya (English)", "laya-ml": "laya-multilingual", "laya-td": "laya-typed-decisions", "jev": "Jev 1.13 (published)"}


def routed(summary, task):
    """Laya's Router: English -> laya-en, otherwise laya-multilingual."""
    lang = task.split(":")[1] if ":" in task else "en"
    m = "laya-en" if lang == "en" else "laya-ml"
    return summary.get(task, {}).get(m), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/h2h/summary.json")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    S = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=180); fig.savefig(out / f"{name}.pdf"); plt.close(fig)

    # 1. public datasets
    tasks = [t for t in ["ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections", "massive:en", "xnli:en", "typed_decisions"] if t in S]
    fig, ax = plt.subplots(figsize=(10.5, 4))
    w = 0.26
    for i, t in enumerate(tasks):
        tez = S[t].get("tez", {}).get("accuracy")
        lay, lm = routed(S, t)
        if tez is not None:
            ax.bar(i - w, tez, w, color=COL["tez"]); ax.text(i - w, tez + 0.01, f"{tez:.3f}", ha="center", fontsize=7)
        if lay:
            ax.bar(i, lay["accuracy"], w, color=COL["laya-en"]); ax.text(i, lay["accuracy"] + 0.01, f"{lay['accuracy']:.3f}", ha="center", fontsize=7)
        if t == "typed_decisions" and "laya-td" in S[t]:
            v = S[t]["laya-td"]["accuracy"]; ax.bar(i + w, v, w, color=COL["laya-td"]); ax.text(i + w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        elif t in JEV:
            ax.bar(i + w, JEV[t], w, color=COL["jev"]); ax.text(i + w, JEV[t] + 0.01, f"{JEV[t]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t.replace("_", " ").replace(":en", " (en)") for t in tasks], rotation=15)
    ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy"); ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("tez", "laya-en", "laya-td", "jev")],
              labels=[NAME["tez"], "Laya (routed checkpoint, zero-shot)", "laya-typed-decisions (fine-tuned on this benchmark)", NAME["jev"]], fontsize=7, loc="lower left")
    ax.set_title("Same rows, same machine: Tez vs Laya (Jev bars are third-party published, not measured here)", fontsize=9)
    save(fig, "tez_vs_laya_vs_jev")

    # 2. languages (MASSIVE)
    langs = [t for t in S if t.startswith("massive:")]
    if langs:
        langs.sort(key=lambda t: -(S[t].get("tez", {}).get("accuracy") or 0))
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(langs) + 1.2))
        for i, t in enumerate(langs):
            y = len(langs) - i
            for m, mk in (("laya-en", "o"), ("laya-ml", "s"), ("tez", "D")):
                if m in S[t]:
                    ax.plot(S[t][m]["accuracy"], y, mk, color=COL[m], ms=6)
        ax.set_yticks(range(1, len(langs) + 1)); ax.set_yticklabels([t.split(":")[1] for t in reversed(langs)])
        ax.axvline(0.05, ls="--", color="grey", lw=0.8); ax.text(0.052, len(langs) + 0.4, "random (0.050)", fontsize=7, color="grey")
        ax.axvline(0.15, ls=":", color="grey", lw=0.8); ax.text(0.152, 0.4, "3x random", fontsize=7, color="grey")
        ax.set_xlim(0, 1); ax.set_xlabel("accuracy · MASSIVE intent, 20 options, 100 cases per language"); ax.grid(axis="x", alpha=0.3)
        ax.legend(handles=[plt.Line2D([], [], marker=mk, color=COL[m], ls="") for m, mk in (("tez", "D"), ("laya-en", "o"), ("laya-ml", "s"))],
                  labels=[NAME[m] for m in ("tez", "laya-en", "laya-ml")], fontsize=7, loc="lower right")
        ax.set_title("Every language, one frozen decoder vs two trained encoders", fontsize=9)
        save(fig, "languages")

    # 3. calibration: ECE as shipped vs refit, averaged over tasks with both numbers. Tez reads unfitted letters at its
    #    default temperature (tez/temperature.py), so Tez as shipped is that default, fitted without the task scored
    #    (results/calibration/default_temperature.json, anchor_28: the same entries); its T = 1 ECE is drawn outlined.
    models = [m for m in ("tez", "laya-en", "laya-ml", "laya-td") if any(m in v for v in S.values())]
    tez_default = None
    dt_path = Path("results/calibration/default_temperature.json")
    if dt_path.exists():
        A = json.loads(dt_path.read_text(encoding="utf-8"))["anchor_28"]
        tez_rows = {t: v["tez"] for t, v in S.items() if "tez" in v and "ece_refit" in v["tez"]}
        same = sorted(tez_rows) == sorted(e["entry"] for e in A["entries"])
        if same and abs(sum(r["ece"] for r in tez_rows.values()) / len(tez_rows) - A["mean"]["raw"]) < 1e-9:
            tez_default = A["mean"]["rule_loto"]
        else:
            print(f"WARNING: {dt_path} does not cover the same Tez entries as {args.summary}; Tez's default not drawn")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for j, m in enumerate(models):
        raw = [v[m]["ece"] for v in S.values() if m in v and "ece_refit" in v[m]]
        ref = [v[m]["ece_refit"] for v in S.values() if m in v and "ece_refit" in v[m]]
        if raw:
            a, b = sum(raw) / len(raw), sum(ref) / len(ref)
            if m == "tez" and tez_default is not None:
                ax.bar(j - 0.3, a, 0.28, color="white", edgecolor=COL[m], linewidth=1.2)
                ax.text(j - 0.3, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
                ax.text(j - 0.3, a / 2, "T = 1", ha="center", va="center", fontsize=6.5, color=COL[m], rotation=90)
                a, xa, xb, w = tez_default, j, j + 0.3, 0.28
            else:
                xa, xb, w = j - 0.18, j + 0.18, 0.36
            ax.bar(xa, a, w, color=COL[m], alpha=0.45); ax.bar(xb, b, w, color=COL[m])
            ax.text(xa, a + 0.005, f"{a:.3f}", ha="center", fontsize=7); ax.text(xb, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
    ax.axhline(0.246, ls="--", color="grey", lw=0.8); ax.text(len(models) - 0.5, 0.25, "Jev ECE 0.246 (published)", fontsize=7, color="grey", ha="right")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], fontsize=7)
    title = "Calibration: as shipped (light) vs one temperature per task, fitted out-of-fold (dark)"
    if tez_default is not None:
        title += "\nTez as shipped: its default temperature, fitted without the task; outlined: Tez at T = 1"
    ax.set_ylabel("mean ECE-15 over tasks (lower is better)"); ax.set_title(title, fontsize=9)
    save(fig, "calibration")

    # 4. typed decisions
    if "typed_decisions" in S:
        T = S["typed_decisions"]
        bars = [("Tez\n(Gemma 4 12B, zero-shot)", T.get("tez", {}).get("accuracy"), COL["tez"]), ("laya base\n(zero-shot)", T.get("laya-en", {}).get("accuracy"), COL["laya-en"]),
                ("laya-typed-decisions\n(fine-tuned on train split)", T.get("laya-td", {}).get("accuracy"), COL["laya-td"]), ("Jev 1.13\n(published)", 0.727, COL["jev"]),
                ("teacher\nself-agreement", 0.735, "#DDDDDD"), ("majority\nclass", 0.461, "#EEEEEE"), ("random", 0.318, "#F5F5F5")]
        bars = [b for b in bars if b[1] is not None]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for i, (n, v, c) in enumerate(bars):
            ax.bar(i, v, color=c, edgecolor="#888"); ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=7)
        ax.set_ylim(0, 0.9); ax.set_ylabel("accuracy"); ax.grid(axis="y", alpha=0.3)
        ax.set_title("typed-decisions: 400 cases, 2,000 decisions, four workflows", fontsize=9)
        save(fig, "typed_decisions")

    # 5. order flip
    ft = [t for t in S if any(m in S[t] and S[t][m].get("flip_rate") is not None for m in models) and ":" not in t]
    if ft:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        w = 0.8 / max(1, len(models))
        for j, m in enumerate(models):
            xs = [i + (j - len(models) / 2 + 0.5) * w for i in range(len(ft))]
            ys = [S[t].get(m, {}).get("flip_rate") or 0 for t in ft]
            ax.bar(xs, ys, w, color=COL[m], label=NAME[m])
        ax.axhline(0.13, ls="--", color="grey", lw=0.8); ax.text(len(ft) - 0.5, 0.135, "Jev 0.13 (published)", fontsize=7, color="grey", ha="right")
        ax.set_xticks(range(len(ft))); ax.set_xticklabels(ft, rotation=15); ax.set_ylabel("option-order flip rate"); ax.legend(fontsize=7)
        ax.set_title("How often the answer changes when the options are reversed (first 100 rows)", fontsize=9)
        save(fig, "order_flip")

    # 6. speed
    fig, ax = plt.subplots(figsize=(8, 3.4))
    st = [t for t in ["ag_news", "emotion", "boolq", "sst5", "massive:en", "xnli:en", "typed_decisions", "banking77"] if t in S]
    w = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        xs = [i + (j - len(models) / 2 + 0.5) * w for i in range(len(st))]
        ys = [S[t].get(m, {}).get("ms_p50") or 0 for t in st]
        ax.bar(xs, ys, w, color=COL[m], label=NAME[m])
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st, rotation=15); ax.set_ylabel("ms per decision (p50, wall clock)"); ax.legend(fontsize=7)
    ax.set_title("Latency on the same RTX 5080 laptop GPU (Tez via llama-server HTTP; Laya in-process)", fontsize=9)
    save(fig, "speed")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
