"""Render verified paired-probe results without replacing the original quality or speed gates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from structure_probe import OPERATIONS, summarize


def main():
    directory = Path(__file__).resolve().parents[2] / "results/pilot-structure-probe-20260927"
    summarize(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    names = {"a-base-30k": "Encoder, original", "a-base-12ep": "Encoder, longer run", "c-qwen08-30k": "Qwen student"}
    lines = ["# Paired structured-record results", "",
             "This English synthetic diagnostic measures frozen models on identical facts in flat and nested JSON.",
             "It does not substitute for the original public accuracy, calibration or laptop speed gates.", "",
             "| Model | Flat accuracy | Nested accuracy | Nested minus flat | State-bootstrap 95% interval | Label flips |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, result in report["models"].items():
        lo, hi = result["paired_state_bootstrap_ci95_points"]
        lines.append(f"| {names[arm]} | {result['accuracy']['flat']['all']['accuracy']:.2%} | "
                     f"{result['accuracy']['nested']['all']['accuracy']:.2%} | {result['nested_minus_flat_points']:+.2f} points | "
                     f"[{lo:+.2f}, {hi:+.2f}] | {result['prediction_flip_fraction']:.2%} |")
    lines += ["", "All five questions for a state stay together across 2,000 bootstrap resamples. These are",
              "exploratory unadjusted intervals. Container keys and punctuation differ between representations;",
              "all leaf keys, values and their traversal order, questions, options and answers are preserved.", "",
              "## Results by operation", "", "Each cell is flat / nested accuracy.", "",
              "| Operation | Uniform random | Schema majority label | " + " | ".join(names[a] for a in report["models"]) + " |",
              "|---|---:|---:|" + "---:|" * len(report["models"])]
    for operation in OPERATIONS:
        questions = [(r["domain"], q) for r in cases for q in r["questions"] if q["operation"] == operation]
        buckets = defaultdict(Counter)
        for domain, q in questions:
            buckets[domain][q["options"][q["gold"]][0]] += 1
        majority = sum(max(counts.values()) for counts in buckets.values()) / len(questions)
        uniform = sum(1 / len(q["options"]) for _, q in questions) / len(questions)
        parts = [f"| {operation} | {uniform:.2%} | {majority:.2%}"]
        for result in report["models"].values():
            parts.append(f"{result['accuracy']['flat'][operation]['accuracy']:.2%} / "
                         f"{result['accuracy']['nested'][operation]['accuracy']:.2%}")
        lines.append(" | ".join(parts) + " |")
    lines += ["", "The majority baseline chooses the most frequent semantic answer for each schema while ignoring",
              "state values. It is a descriptive class-balance baseline, calculated from the diagnostic labels,",
              "not a fitted deployed model. In particular, the Boolean tasks have unequal answer frequencies.", "",
              "## Provenance and limits", "",
              "The generator authored these exact-answer records without reading training or benchmark inputs.",
              "This set stays evaluation-only. It repeats five rule families in four settings and does not prove",
              "generalization to arbitrary JSON workflows, natural documents or other languages. Any later training",
              "candidate requires disjoint validation and fresh confirmation; this probe cannot select it.", "",
              "Completed runs validated every prediction, finite normalized probability vector, gold label and",
              "argmax, plus the exact previously evaluated weight hashes. Input token checks are recorded in",
              "each execution receipt. The laptop thermal governor and shared lock remained enabled; these",
              "elapsed times are not usable latency measurements. No additional RunPod resources were used.", "",
              "See [registration and method](README.md), [registration](registration.json), and [full results](summary.json)."]
    pending = set(report["registration"]["models"]) - report["models"].keys()
    if pending:
        lines += ["", "Incomplete models: " + ", ".join(sorted(pending)) + ". Inspect execution receipts; no result is inferred."]
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
