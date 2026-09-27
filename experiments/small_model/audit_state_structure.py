"""Compare training and evaluation state shapes without looking at evaluation labels."""
from __future__ import annotations

from collections import Counter
import json

from common import ROOT, RUNS, RESULTS, dump, file_sha256
from dataset import load_examples


def depth(value):
    if isinstance(value, dict):
        return 1 + max((depth(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((depth(v) for v in value), default=0)
    return 0


def describe(states):
    return {"states": len(states), "container_depth": dict(Counter(depth(s) for s in states)),
            "root_type": dict(Counter(type(s).__name__ for s in states)),
            "top_level_dictionary_fields": dict(Counter(len(s) for s in states if isinstance(s, dict))),
            "nested_container_states": sum(depth(s) >= 2 for s in states)}


def main():
    train, _, val = load_examples()
    source = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    suites = json.loads(source.read_text(encoding="utf-8"))
    evaluation = {}
    for task, cases in suites.items():
        unique = {json.dumps(c["state"], sort_keys=True, ensure_ascii=False): c["state"] for c in cases}
        evaluation[task] = describe(list(unique.values()))
    report = {"scope": "Descriptive distribution comparison; no inference, fitting, or evaluation labels used",
              "teacher_manifest_sha256": file_sha256(RUNS / "manifest.json"), "eval_suites_sha256": file_sha256(source),
              "train": describe([e.state for e in train]), "validation": describe([e.state for e in val]),
              "evaluation": evaluation,
              "interpretation": "A shape mismatch is a transfer hypothesis; it does not establish the cause of accuracy differences"}
    dump(RESULTS / "state_structure_audit.json", report)
    print(json.dumps({"train": report["train"], "typed_decisions": evaluation["typed_decisions"]}, indent=2))


if __name__ == "__main__":
    main()
