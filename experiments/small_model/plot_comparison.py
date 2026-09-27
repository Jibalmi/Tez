"""Export accuracy/latency trade-offs measured on the same H200 trial."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    root = Path(__file__).resolve().parents[2]
    results = root / "results/pilot-improvement-20260927"
    comparison = json.loads((results / "comparison.json").read_text())
    latency = json.loads((results / "cloud-latency/latency.json").read_text())
    labels = {"a-base-12ep": "Tez encoder, longer run", "c-qwen08-30k": "Tez Qwen student",
              "gliclass-multilang-mini": "GLiClass Multilang Mini",
              "gliclass-instruct-edge-v1.0": "GLiClass Instruct Edge", "kev08-native": "Kev 0.8B",
              "decider08-native": "Decider 0.8B"}
    records = []
    for run in latency.values():
        arm = run["arm"]
        measured = comparison["systems"].get(arm)
        if arm not in labels or not measured or measured["typed_accuracy"] is None:
            continue
        if run["device"] != "cuda" or run["contaminated"] is not False or run["throttled"] is not False:
            continue
        records.append((arm, measured["typed_accuracy"] * 100, run))
    if not records:
        raise SystemExit("No jointly measured clean accuracy/latency records")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharey=True)
    palette = plt.get_cmap("tab10")
    for axis, size, field, title in zip(axes, ("1", "50"), ("p50_ms", "ms_per_question"),
                                       ("One question per request", "50 questions per request")):
        for i, (arm, accuracy, run) in enumerate(records):
            ms = run["sizes"][size][field]
            axis.scatter(ms, accuracy, s=64, color=palette(i), zorder=3)
            # Number labels keep nearby points readable; the legend names each measured model.
            axis.annotate(str(i + 1), (ms, accuracy), xytext=(5, -15 if arm == "kev08-native" else 6),
                          textcoords="offset points", fontsize=9)
        axis.set_xscale("log")
        ticks = [5, 10, 20, 40] if size == "1" else [.25, .5, 1, 2, 4]
        axis.set_xticks(ticks, [f"{v:g}" for v in ticks])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Median milliseconds per question (log scale; lower is better)")
        axis.set_title(title)
        axis.grid(True, alpha=.2)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Typed-decision accuracy (%)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=palette(i), label=f"{i + 1}. {labels[arm]}")
               for i, (arm, _, _) in enumerate(records)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .10), ncol=3, frameon=False)
    fig.suptitle("Small-model trade-offs on the same H200", fontsize=16, x=.06, ha="left")
    fig.text(.06, .04, "Accuracy: 2,000 frozen typed decisions. Latency: PyTorch eager, fresh states; 5 warmups, 40/15 repetitions.\n"
             "Competitor training overlap is unknown. These timings do not measure optimized servers or pass laptop speed gates.",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .22, 1, .95))
    for suffix in ("png", "svg"):
        target = results / ("quality_speed." + suffix)
        fig.savefig(target, dpi=160, facecolor="white")
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n",
                              encoding="utf-8")
    print(json.dumps({"models": len(records), "files": ["quality_speed.png", "quality_speed.svg"]}))


if __name__ == "__main__":
    main()
