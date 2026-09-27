"""Confirmation must remain closed until both selections finish, with correct semantic indexing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_reasoning_confirmation as evaluation
from structure_probe import digest


class ConfirmationTests(unittest.TestCase):
    def test_both_training_runs_must_complete_before_confirmation_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "experiments/small_model/reasoning_round2.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"arms": ["control", "candidate"], "steps": 150, "new_data": "data"}))
            results = root / "results"
            results.mkdir()
            for arm, state in (("control", "complete"), ("candidate", "training")):
                receipt = {"status": state, "step": 150, "specification_sha256": digest(config_path)}
                (results / f"train_{arm}.json").write_text(json.dumps(receipt))
            with patch.object(evaluation, "ROOT", root), patch.object(evaluation, "RESULTS", results):
                with self.assertRaisesRegex(ValueError, "Finish both"):
                    evaluation.prerequisites()
            self.assertFalse((root / "data").exists())

    def test_gold_index_follows_semantic_option_order(self):
        record = {"id": "s", "domain": "fixture", "state": {"value": 2}, "questions": [
            {"operation": "count", "qtype": "score", "instructions": "Count", "gold_key": "2",
             "options": [["2", "Both"], ["0", "Neither"], ["1", "One"]]}]}
        case = evaluation.cases_for([record])[0][0]
        self.assertEqual(case["gold"], 0)
        self.assertEqual(case["group"], "s")
        self.assertEqual(case["state"], {"value": 2})


if __name__ == "__main__":
    unittest.main()
