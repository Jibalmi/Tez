"""Inspect semantic answer collapse and preserve a rendered ordinal prompt for the recorded probe."""
from collections import Counter
import json
from pathlib import Path

from structure_probe import digest, write_json
from train_c import prompt_for


def main():
    directory = Path(__file__).resolve().parents[2] / "results/pilot-structure-probe-20260927"
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in cases}
    audit = {"source_sha256": digest(__file__), "cases_sha256": digest(directory / "cases.json")}
    for arm, result in summary["models"].items():
        path = directory / f"rows_{arm}.jsonl"
        if digest(path) != result["receipt"]["rows_sha256"]:
            raise ValueError("Prediction hash mismatch")
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        audit[arm] = {}
        for form in ("flat", "nested"):
            audit[arm][form] = {}
            for operation in ("threshold", "conjunction", "band"):
                counts = Counter()
                for row in rows:
                    if row["format"] == form and row["operation"] == operation:
                        q = next(q for q in lookup[row["id"]]["questions"] if q["operation"] == operation)
                        counts[q["options"][row["pred"]][0]] += 1
                audit[arm][form][operation] = dict(counts)
    record = cases[0]
    q = next(q for q in record["questions"] if q["operation"] == "band")
    audit["rendered_score_example"] = {
        "id": record["id"], "state": record["states"]["flat"], "question": q,
        "qwen_prompt": prompt_for(q["qtype"], q["instructions"], q["options"], record["states"]["flat"])}
    write_json(directory / "behavior_audit.json", audit)


if __name__ == "__main__":
    main()
