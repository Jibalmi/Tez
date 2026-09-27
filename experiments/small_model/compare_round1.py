"""Compare recorded pilot runs on identical tasks; published claims never enter the ranking."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from common import ROOT, MASSIVE11, file_sha256


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keyed = {row["id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate decision IDs in {path}")
    return keyed


def paired_typed_delta(cases, left, right, reps=2000):
    """Resample whole states together, retaining all decisions from each selected state."""
    ids = {c["id"] for c in cases}
    if set(left) != ids or set(right) != ids:
        raise ValueError("Paired comparison requires every frozen decision exactly once")
    groups = defaultdict(list)
    for c in cases:
        a, b = left[c["id"]], right[c["id"]]
        if a["gold"] != c["gold"] or b["gold"] != c["gold"]:
            raise ValueError("Gold labels differ between aligned records")
        groups[c["group"]].append(int(a["pred"] == c["gold"]) - int(b["pred"] == c["gold"]))
    values = [(sum(ds), len(ds)) for ds in groups.values()]
    rng = random.Random(13)
    estimates = []
    for _ in range(reps):
        sample = rng.choices(values, k=len(values))
        estimates.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    estimates.sort()
    return {"accuracy_delta_points": 100 * sum(s for s, _ in values) / len(cases),
            "ci95_points": [100 * estimates[int(reps * .025)], 100 * estimates[int(reps * .975)]],
            "states": len(values), "decisions": len(cases), "bootstrap_reps": reps, "seed": 13,
            "note": "Exploratory paired state-cluster percentile interval; no multiplicity correction."}


def regime(arm):
    if arm == "laya-td":
        return "Trained on typed-decisions training split"
    if arm.startswith(("a-", "c-")):
        return "Tez teacher distillation; whole typed-decisions dataset excluded"
    if arm == "qwen08-frozen":
        return "Original post-trained Qwen; same Tez prompt, no pilot LoRA"
    if arm == "tez-12b":
        return "Teacher reference; upstream training overlap unknown"
    return "Released checkpoint; upstream benchmark overlap not established"


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/pilot-improvement-20260927")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "runs/improvement-artifacts-h200-20260927")
    args = parser.parse_args()
    original = ROOT / "results/pilot-student-h200-20260927"
    systems, provenance, calibrations = {}, {}, {}
    refs = load(original / "baselines.json")
    for task, records in refs["tasks"].items():
        for arm, metrics in records.items():
            systems.setdefault(arm, {})[task] = {"n": metrics["n"], "t1": metrics, "semif": metrics.get("semif", {})}
    for directory in (original, args.results):
        for path in sorted(directory.glob("eval_*.json")):
            report = load(path)
            arm = report["arm"]
            systems[arm] = report["tasks"]
            calibrations[arm] = bool(report.get("temperatures"))
            provenance[arm] = {"file": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
                               "weights_sha256": report.get("weights_sha256"), "model_info": report.get("model_info")}
    summary = {}
    for arm, tasks in systems.items():
        td = tasks.get("typed_decisions", {})
        languages = [tasks[f"massive_{lang}"]["t1"]["accuracy"] for lang in MASSIVE11 if f"massive_{lang}" in tasks]
        summary[arm] = {"training_regime": regime(arm), "recorded_decisions": sum(t["n"] for t in tasks.values()),
                        "typed_accuracy": td.get("t1", {}).get("accuracy"),
                        "typed_ece_raw": td.get("t1", {}).get("ece"),
                        "typed_ece_shipped_or_validation_cal": td.get("cal", {}).get("ece") if calibrations.get(arm) else None,
                        "massive11_macro": statistics.mean(languages) if len(languages) == 11 else None,
                        "massive_languages": len(languages),
                        "apps": {t[4:]: v["t1"]["accuracy"] for t, v in tasks.items() if t.startswith("app_")},
                        "semif": {t: v.get("semif", {}).get("mean_family_balanced_accuracy")
                                  for t, v in tasks.items() if t.startswith("semif_")}}
    cases_path = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    cases = load(cases_path)["typed_decisions"]
    row_roots = [ROOT / "runs/student-artifacts-h200-20260927/results/pilot-student-h200-20260927/rows",
                 args.artifacts / "results/pilot-improvement-20260927/rows"]
    rows = {}
    for directory in row_roots:
        for path in directory.glob("rows_typed_decisions_tez-s-*.jsonl"):
            rows[path.stem.removeprefix("rows_typed_decisions_tez-s-")] = read_rows(path)
    paired = {}
    anchor = "c-qwen08-30k"
    if anchor in rows:
        for arm, predictions in rows.items():
            if arm != anchor and len(predictions) == len(cases):
                paired[arm] = paired_typed_delta(cases, predictions, rows[anchor])
    report = {"metric": "Accuracy on identical frozen rows, except explicitly named secondary metrics",
              "interpretation": "Exploratory; public tests informed follow-up selection. No release claim.",
              "eval_suites_sha256": file_sha256(cases_path), "systems": summary, "provenance": provenance,
              "typed_paired_delta_vs_c_qwen08_30k": paired,
              "excluded_from_ranking": ["Published scores on other suites", "Missing or failed evaluations"]}
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ordered = sorted(summary, key=lambda a: summary[a]["typed_accuracy"] or -1, reverse=True)
    lines = ["# Matched small-model comparison", "", "All values below come from recorded pilot evaluations.", "",
             "Typed accuracy and MASSIVE are percentages; ECE is shown in percentage points (lower is better).",
             "The MASSIVE mean requires all 11 languages. No composite rank is constructed across unlike tasks.", "",
             "| System | Typed accuracy | MASSIVE 11 | Typed ECE, raw | Typed ECE, shipped/validation calibration |",
             "|---|---:|---:|---:|---:|"]
    for arm in ordered:
        v = summary[arm]
        lines.append(f"| {arm} | {pct(v['typed_accuracy'])} | {pct(v['massive11_macro'])} | "
                     f"{pct(v['typed_ece_raw'])} | {pct(v['typed_ece_shipped_or_validation_cal'])} |")
    lines += ["", "Laya-td trained on typed-decisions' training split. Tez students excluded the entire dataset.",
              "Released competitors may have benchmark overlap; unknown overlap is not evidence of a held-out test.",
              "Raw T=1 is primary. Calibration uses only teacher-validation labels or the released temperature.",
              "Evaluation-label refits are excluded. Follow-up gains require fresh independent confirmation.", "",
              "## Application tasks", "", "All seven tasks are shown to preserve strengths and losses.", ""]
    apps = ["model_routing_domain", "guardrails_jailbreak", "moderation_toxicity", "email_spam", "phishing",
            "rag_relevance", "support_triage"]
    lines += ["| System | Routing | Jailbreak | Toxicity | Spam | Phishing | Relevance | Triage |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ordered:
        lines.append("| " + arm + " | " + " | ".join(pct(summary[arm]["apps"].get(t)) for t in apps) + " |")
    lines += ["", "## SemIf generalization", "", "Mean family balanced accuracy, using SemIf's own evaluator.", "",
              "| System | Authored 144 | Perturbations 108 |", "|---|---:|---:|"]
    for arm in ordered:
        values = summary[arm]["semif"]
        if any(v is not None for v in values.values()):
            lines.append(f"| {arm} | {pct(values.get('semif_authored144'))} | {pct(values.get('semif_perturbations108'))} |")
    lines += ["", "## Paired uncertainty on typed decisions", "",
              "Difference from the original Qwen student, in accuracy points. Resampling keeps each state's five",
              "questions together (2,000 draws, seed 13). Intervals are exploratory and unadjusted for multiple comparisons.", "",
              "| Candidate minus original Qwen student | Difference | 95% state-bootstrap interval |",
              "|---|---:|---:|"]
    for arm, stats in paired.items():
        lo, hi = stats["ci95_points"]
        lines.append(f"| {arm} | {stats['accuracy_delta_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## Evidence and limits", "",
              "[Source inventory and training distinctions](RESEARCH.md), [machine-readable comparison](comparison.json).",
              "Published Kev/Decider/JevBench scores use different protocols and do not enter these tables.",
              "Missing evaluations are not zero scores. GPU, CPU, and different hardware timings must remain separate.",
              "The registered speed and quality gates still govern model selection; task wins alone do not pass them.", ""]
    latency_path = args.results / "cloud-latency/latency.json"
    if latency_path.exists():
        latency = load(latency_path)
        lines += ["## H200 timings", "", "Same fresh-state protocol; 5 warmups, 40 repetitions for one question, 15 for 50.",
                  "Kev and Decider use their native eager path. Their optimized serving implementations were not measured.",
                  "Merge/cache variants require their own parity decision; timing alone does not approve them.", "",
                  "| Arm | 1 question p50, ms | 50 questions p50, ms | ms/question at 50 | Clean telemetry |",
                  "|---|---:|---:|---:|---|"]
        for run in latency.values():
            sizes = run["sizes"]
            clean = run.get("contaminated") is False and run.get("throttled") is False
            lines.append(f"| {run['arm']} | {sizes['1']['p50_ms']:.2f} | {sizes['50']['p50_ms']:.2f} | "
                         f"{sizes['50']['ms_per_question']:.3f} | {clean} |")
        lines += ["", "These H200 timings do not satisfy the laptop speed gate.", ""]
    if (args.results / "quality_speed.png").exists():
        lines += ["![Same-trial quality and speed trade-offs](quality_speed.png)", "",
                  "[Download vector figure](quality_speed.svg). Only candidates with separately measured accuracy and clean timings appear.", ""]
    (args.results / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"systems": len(summary), "paired_comparisons": len(paired), "report": str(args.results / 'COMPARISON.md')}))


if __name__ == "__main__":
    main()
