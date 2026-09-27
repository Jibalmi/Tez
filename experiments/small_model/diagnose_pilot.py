"""Describe pilot errors and training targets without fitting anything to the test set."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math

from common import RESULTS, ROOT, RUNS, dump, file_sha256, read_jsonl
from dataset import load_examples


def metric(rows):
    n = len(rows)
    correct = sum(r["pred"] == r["gold"] for r in rows)
    return {"n": n, "accuracy": correct / n,
            "mean_confidence": sum(max(r["probabilities"]) for r in rows) / n,
            "predicted_option_counts": dict(Counter(r["pred"] for r in rows)),
            "gold_option_counts": dict(Counter(r["gold"] for r in rows))}


def describe_training(examples):
    schemas = defaultdict(list)
    targets = defaultdict(list)
    for e in examples:
        for d in e.decisions:
            key = json.dumps([d.type, d.instructions, d.options], ensure_ascii=False)
            schemas[key].append(int(d.target.argmax()))
            targets[d.type].append(d.target)
    return {"states": len(examples), "unique_schemas": len(schemas),
            "schemas_with_at_least_5_examples": sum(len(v) >= 5 for v in schemas.values()),
            "majority_agreement_within_schema": sum(max(Counter(v).values()) for v in schemas.values()) /
            sum(len(v) for v in schemas.values()),
            "types": {ty: {"n": len(vals), "argmax_index_counts": dict(Counter(int(p.argmax()) for p in vals)),
                            "mean_top_probability": sum(max(p) for p in vals) / len(vals),
                            "mean_normalized_entropy": sum(-sum(float(v) * math.log(max(float(v), 1e-12)) for v in p) /
                                                            math.log(len(p)) for p in vals) / len(vals)}
                      for ty, vals in targets.items()}}


def main():
    train, _, val = load_examples()
    records = {"scope": "Descriptive error analysis only; no test labels used to train, calibrate, or select a checkpoint.",
               "teacher_manifest_sha256": file_sha256(RUNS / "manifest.json"),
               "train": describe_training(train), "validation": describe_training(val), "evaluations": {}}
    artifact_results = ROOT / "runs/student-artifacts-h200-20260927/results/pilot-student-h200-20260927"
    for path in sorted((artifact_results / "rows").glob("rows_typed_decisions_tez-s-*.jsonl")):
        arm = path.stem.split("_tez-s-")[1]
        rows = read_jsonl(path)
        slices = defaultdict(list)
        for row in rows:
            for key in ("type=" + row["qtype"], "workflow=" + row["workflow"], "options=" + str(row["k"])):
                slices[key].append(row)
        records["evaluations"][arm] = {"rows_sha256": file_sha256(path), "overall": metric(rows),
                                        "slices": {k: metric(v) for k, v in sorted(slices.items())}}
    dump(RESULTS / "diagnostics.json", records)
    print(json.dumps({"train": records["train"], "validation": records["validation"],
                      "typed_decisions_by_type": {arm: {k: v for k, v in ev["slices"].items() if k.startswith("type=")}
                                                  for arm, ev in records["evaluations"].items()}}, indent=2))


if __name__ == "__main__":
    main()
