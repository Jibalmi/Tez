"""Cloud snapshot and target-parity checks without any GPU or provider API calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from cloud_teacher import collect, parity


class CloudTeacherTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"one": {"id": "one", "questions": [{"qid": "q", "type": "noul",
            "keys": ["false", "true"], "abstain": False, "z_orig": [-0.1, -3.0], "z_rev": [-3.0, -0.1]}]}}

    def test_same_targets_pass(self):
        result = parity(self.reference, copy.deepcopy(self.reference))
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_probability_delta"], 0)

    def test_changed_option_order_is_rejected(self):
        candidate = copy.deepcopy(self.reference)
        candidate["one"]["questions"][0]["keys"].reverse()
        with self.assertRaisesRegex(ValueError, "metadata"):
            parity(self.reference, candidate)

    def test_missing_states_and_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            parity(self.reference, {})
        candidate = copy.deepcopy(self.reference)
        candidate["one"]["questions"][0]["z_orig"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid logits"):
            parity(self.reference, candidate)

    def test_materially_changed_distribution_fails(self):
        candidate = copy.deepcopy(self.reference)
        q = candidate["one"]["questions"][0]
        q["z_orig"], q["z_rev"] = [-10, 0], [0, -10]
        self.assertFalse(parity(self.reference, candidate)["passed"])

    def test_duplicate_worker_outputs_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            for worker in ("worker-0", "worker-1"):
                directory = Path(folder) / worker / "teacher"
                directory.mkdir(parents=True)
                (directory / "part-0000.jsonl").write_text(json.dumps(self.reference["one"]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect(Path(folder))


if __name__ == "__main__":
    unittest.main()
