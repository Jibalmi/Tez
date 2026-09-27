"""Describe class coverage and majority references in the completed authored confirmation run."""
from collections import Counter
import json

from structure_probe import digest, write_json
from train_reasoning_round2 import RESULTS


def main():
    summary = json.loads((RESULTS / "confirmation_summary.json").read_text(encoding="utf-8"))
    models = {}
    for arm, model in summary["models"].items():
        path = RESULTS / f"confirmation_rows_{arm}.jsonl"
        if digest(path) != model["receipt"]["rows_sha256"]:
            raise ValueError("Validated confirmation rows changed")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        operations = {}
        for op in sorted({r["operation"] for r in rows}):
            selected = [r for r in rows if r["operation"] == op]
            gold = Counter(r["gold"] for r in selected)
            pred = Counter(r["pred"] for r in selected)
            recalls = {str(g): sum(r["pred"] == g for r in selected if r["gold"] == g) / n for g, n in gold.items()}
            operations[op] = {"gold": dict(gold), "pred": dict(pred), "class_recall": recalls,
                              "balanced_accuracy": sum(recalls.values()) / len(recalls),
                              "accuracy": sum(r["gold"] == r["pred"] for r in selected) / len(selected),
                              "posthoc_majority_accuracy": max(gold.values()) / len(selected)}
        models[arm] = {"operations": operations, "rows_sha256": digest(path)}
    write_json(RESULTS / "bias_audit.json", {"models": models, "source_sha256": digest(__file__),
               "interpretation": "Post-hoc descriptive diagnosis. Majority answers are chosen from confirmation gold frequencies, not deployable predictions or checkpoint-selection data."})
    lines = ["# Confirmation class-coverage audit", "", "This descriptive audit uses already completed predictions; it changes no model or selection.",
             "The majority reference picks each operation's most frequent confirmation gold answer after the fact.",
             "It is a class-balance check, not a deployable oracle or independent baseline fitted on training data.", "",
             "| Operation | Post-hoc majority | Starting model | Rule supervision | Rule balanced accuracy |", "|---|---:|---:|---:|---:|"]
    original = models["a-base-12ep"]["operations"]
    for op, values in models["a-base-rules150"]["operations"].items():
        numbers = [values["posthoc_majority_accuracy"], original[op]["accuracy"], values["accuracy"], values["balanced_accuracy"]]
        lines.append(f"| {op} | " + " | ".join(f"{100 * n:.2f}%" for n in numbers) + " |")
    lines += ["", "Compare, combined and count improve beyond their majority references. These are the three operations retained in added training.",
              "Policy remains below its 50% majority reference. Largest remains near its majority reference and heavily favors one answer.",
              "Policy and largest were removed from added training by the choice-schema overlap guard; their weak scores do not establish a causal explanation.",
              "Counting is still uneven across classes. The full answer histograms and class recalls are in [bias_audit.json](bias_audit.json).",
              "The overall gain should not be described as general rule mastery or broad competitiveness.", ""]
    (RESULTS / "BIAS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(models["a-base-rules150"]["operations"], indent=2))


if __name__ == "__main__":
    main()
