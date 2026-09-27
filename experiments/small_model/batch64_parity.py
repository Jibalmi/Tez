"""Check a larger inference batch without modifying the trained weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import RESULTS, dump, file_sha256, write_jsonl
from dataset import load_examples
from eval_arms import PredictC, softmax


def main():
    spec_path = Path(__file__).with_name("batch64_registration.json")
    spec = json.loads(spec_path.read_text())
    _, _, examples = load_examples()
    groups = [[{"state": e.state, "qtype": d.type, "instructions": d.instructions, "options": d.options}
               for d in e.decisions] for e in examples]
    pred = PredictC("c-qwen08-30k")
    original_batch_size = 16
    original = [softmax(p) for group in pred.groups(groups) for p in group]
    pred.batch_size = 64
    candidate = [softmax(p) for group in pred.groups(groups) for p in group]
    agreement = float(np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(original, candidate)]))
    delta = max(float(np.max(np.abs(a - b))) for a, b in zip(original, candidate))
    mean_tv = float(np.mean([np.abs(a - b).sum() / 2 for a, b in zip(original, candidate)]))
    rows_path = RESULTS / "batch64_parity_rows.jsonl"
    write_jsonl(rows_path, [{"validation_decision_index": i, "original": a.tolist(), "batch64": b.tolist()}
                            for i, (a, b) in enumerate(zip(original, candidate))])
    record = {"states": len(examples), "decisions": len(original), "weights_sha256": pred.weights,
              "registration_sha256": file_sha256(spec_path), "rows_sha256": file_sha256(rows_path),
              "original_batch_size": original_batch_size, "scope": "teacher-validation split only",
              "argmax_agreement": agreement, "max_probability_delta": delta, "mean_total_variation": mean_tv,
              "passed": agreement >= spec["argmax_agreement_min"] and delta <= spec["max_probability_delta_max"]
                        and mean_tv <= spec["mean_total_variation_max"]}
    dump(RESULTS / "batch64_parity.json", record)
    print(record, flush=True)
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
