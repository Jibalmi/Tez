"""Explore one fixed equal-probability blend of Qwen and Decider from saved, aligned predictions.

No mixture weight is searched, calibrated or selected. The public evaluations already informed
model selection, so this is exploratory error analysis, not a new held-out certification.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np

from common import ROOT, file_sha256, dump
from compare_round1 import paired_typed_delta, read_rows


def blend(cases, left, right):
    ids = {c["id"] for c in cases}
    if len(ids) != len(cases) or set(left) != ids or set(right) != ids:
        raise ValueError("Every frozen decision must be present exactly once")
    rows = {}
    counts = {"both_correct": 0, "left_only_correct": 0, "right_only_correct": 0, "both_wrong": 0,
              "prediction_disagreement": 0}
    for case in cases:
        a, b = left[case["id"]], right[case["id"]]
        vectors = []
        for row in (a, b):
            p = np.asarray(row["probabilities"], dtype=float)
            if (row["gold"] != case["gold"] or row["qtype"] != case["qtype"]
                    or p.shape != (len(case["options"]),) or not np.isfinite(p).all()
                    or np.any(p < 0) or abs(p.sum() - 1) > 1e-5 or row["pred"] != int(p.argmax())):
                raise ValueError("Input prediction or probability differs from the frozen case")
            vectors.append(p)
        p = (vectors[0] + vectors[1]) / 2
        rows[case["id"]] = {"gold": case["gold"], "pred": int(p.argmax()), "probabilities": p.tolist()}
        ok_a, ok_b = a["pred"] == case["gold"], b["pred"] == case["gold"]
        key = "both_correct" if ok_a and ok_b else "left_only_correct" if ok_a else "right_only_correct" if ok_b else "both_wrong"
        counts[key] += 1
        counts["prediction_disagreement"] += int(a["pred"] != b["pred"])
    def accuracy(data):
        return sum(r["pred"] == r["gold"] for r in data.values()) / len(cases)
    return rows, {"n": len(cases), "qwen_accuracy": accuracy(left), "decider_accuracy": accuracy(right),
                  "equal_blend_accuracy": accuracy(rows), "complementarity_counts": counts,
                  "hindsight_model_selection_accuracy": 1 - counts["both_wrong"] / len(cases)}


def main():
    output = ROOT / "results/pilot-improvement-20260927"
    registration = output / "ensemble_registration.json"
    result = output / "ensemble_diagnostic.json"
    if registration.exists() or result.exists():
        raise ValueError("Recorded diagnostic is immutable")
    dump(registration, {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": file_sha256(__file__),
                        "models": ["c-qwen08-30k", "decider08-native"], "weights": [0.5, 0.5], "temperature": 1,
                        "selection": "One fixed blend, no test-label weight fitting or mixture search",
                        "scope": "Exploratory secondary analysis after inspecting individual public benchmark results",
                        "deployment": "No runtime promotion or speed claim; both models would need inference"})
    original = ROOT / "runs/student-artifacts-h200-20260927/results/pilot-student-h200-20260927/rows"
    newer = ROOT / "runs/improvement-artifacts-h200-20260927/results/pilot-improvement-20260927/rows"
    source = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    suites = json.loads(source.read_text(encoding="utf-8"))
    report = {"frozen_cases_sha256": file_sha256(source), "registration_sha256": file_sha256(registration), "tasks": {}}
    for task, cases in suites.items():
        left_path = original / f"rows_{task}_tez-s-c-qwen08-30k.jsonl"
        right_path = newer / f"rows_{task}_tez-s-decider08-native.jsonl"
        left, right = read_rows(left_path), read_rows(right_path)
        rows, metrics = blend(cases, left, right)
        metrics["sources"] = {str(p.relative_to(ROOT)): file_sha256(p) for p in (left_path, right_path)}
        if task == "typed_decisions":
            metrics["blend_minus_decider_bootstrap"] = paired_typed_delta(cases, rows, right)
        report["tasks"][task] = metrics
    report["massive11_macro"] = {
        key: float(np.mean([v[key] for task, v in report["tasks"].items() if task.startswith("massive_")]))
        for key in ("qwen_accuracy", "decider_accuracy", "equal_blend_accuracy")}
    report["validated_input_predictions"] = sum(t["n"] for t in report["tasks"].values()) * 2
    dump(result, report)
    lines = ["# Fixed Qwen/Decider probability blend", "",
             "This is exploratory analysis of already observed public evaluations. It averages the two raw",
             "probability vectors with fixed equal weights; no mixture search or test-label calibration is used.",
             "No inference was run and no model weights or runtime were changed. Combining models would require",
             "both inference paths; this analysis makes no latency, cost-efficiency or promotion claim.", "",
             "| Task | Qwen | Decider | Equal blend |", "|---|---:|---:|---:|"]
    for task, metrics in report["tasks"].items():
        lines.append(f"| {task} | {metrics['qwen_accuracy']:.2%} | {metrics['decider_accuracy']:.2%} | {metrics['equal_blend_accuracy']:.2%} |")
    macro = report["massive11_macro"]
    lines += [f"| MASSIVE-11 macro | {macro['qwen_accuracy']:.2%} | {macro['decider_accuracy']:.2%} | {macro['equal_blend_accuracy']:.2%} |", ""]
    td = report["tasks"]["typed_decisions"]
    uncertainty = td["blend_minus_decider_bootstrap"]
    lo, hi = uncertainty["ci95_points"]
    lines += [f"On typed decisions, blend minus Decider is {uncertainty['accuracy_delta_points']:+.2f} points;",
              f"the exploratory state-cluster 95% interval is [{lo:+.2f}, {hi:+.2f}]. Whole states stay together",
              "during 2,000 bootstrap resamples; these intervals have no multiplicity correction.", "",
              "Complementarity counts and a hindsight oracle are preserved in the JSON. The oracle chooses",
              "the correct model using the true answer and is not an implementable routing system.", "",
              "Sources: [registration](ensemble_registration.json), [validated measurements](ensemble_diagnostic.json)."]
    (output / "ENSEMBLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"typed": td, "massive11_macro": macro}, indent=2))


if __name__ == "__main__":
    main()
