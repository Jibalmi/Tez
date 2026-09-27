"""Validate a merged Qwen adapter against its original on teacher-validation states only."""
from __future__ import annotations

import numpy as np

from common import RESULTS, dump
from dataset import load_examples
from eval_arms import PredictC, softmax


def main():
    _, _, examples = load_examples()
    groups = [[{"state": e.state, "qtype": d.type, "instructions": d.instructions, "options": d.options}
               for d in e.decisions] for e in examples]
    pred = PredictC("c-qwen08-30k")
    original = [softmax(p) for group in pred.groups(groups) for p in group]
    pred.model = pred.model.merge_and_unload(safe_merge=True).eval()
    pred.model.config.use_cache = False
    merged = [softmax(p) for group in pred.groups(groups) for p in group]
    agreement = float(np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(original, merged)]))
    delta = max(float(np.max(np.abs(a - b))) for a, b in zip(original, merged))
    mean_tv = float(np.mean([np.abs(a - b).sum() / 2 for a, b in zip(original, merged)]))
    record = {"states": len(examples), "decisions": len(original), "weights_sha256": pred.weights,
              "scope": "teacher-validation split only", "argmax_agreement": agreement,
              "max_probability_delta": delta, "mean_total_variation": mean_tv,
              "thresholds": {"argmax_agreement_min": 0.99, "max_probability_delta_max": 0.04,
                             "mean_total_variation_max": 0.005},
              "passed": agreement >= 0.99 and delta <= 0.04 and mean_tv <= 0.005}
    dump(RESULTS / "merge_parity.json", record)
    print(record, flush=True)
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
