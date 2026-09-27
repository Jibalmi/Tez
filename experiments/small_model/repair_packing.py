"""Complete the skipped encoder packing diagnostic without changing primary predictions."""
from __future__ import annotations

import json
import time

from common import RESULTS, dump, file_sha256
from eval_arms import PredictA, score_task, suites


def main():
    arm = "a-base-12ep"
    path = RESULTS / f"eval_{arm}.json"
    report = json.loads(path.read_text())
    before = file_sha256(path)
    dump(RESULTS / "packing_before_a-base-12ep.json", report)
    pred = PredictA(arm)
    assert pred.weights == report["weights_sha256"]
    cases = suites()["typed_decisions"]
    start = time.monotonic()
    predictions = [group[0] for group in pred.groups([[c] for c in cases])]
    scored, _ = score_task("typed_decisions", cases, predictions, pred.info.get("temperatures", {}), arm)
    typed = report["tasks"]["typed_decisions"]
    typed["one_question_per_row"] = {"t1": scored["t1"], "retention_vs_12b": scored.get("retention_vs_12b")}
    typed["packing_delta_points"] = round(100 * (typed["t1"]["accuracy"] - scored["t1"]["accuracy"]), 2)
    dump(path, report)
    dump(RESULTS / "packing_repair.json", {"arm": arm, "decisions": len(cases), "weights_sha256": pred.weights,
         "before_sha256": before, "after_sha256": file_sha256(path), "seconds": time.monotonic() - start,
         "reason": "Predictor factory imports eval_arms while its command runs as __main__; class identity skipped the diagnostic",
         "primary_predictions_unchanged": True, "packing_delta_points": typed["packing_delta_points"]})
    print(json.dumps(typed["one_question_per_row"]))


if __name__ == "__main__":
    main()
