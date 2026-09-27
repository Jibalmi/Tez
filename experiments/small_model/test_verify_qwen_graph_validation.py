"""Reject corrupted probability arithmetic, option masks and graph batch membership."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_qwen_graph_validation import verify_rows


class VerifyGraphRowsTests(unittest.TestCase):
    def setUp(self):
        self.case = {"state_id": "state-1", "qid": "q1", "qtype": "choice", "k": 2,
                     "prompt_sha256": hashlib.sha256(b"input").hexdigest(), "encoded_tokens": 3}
        self.plan = [{"indices": [0], "length": 128, "dummy_rows": 15}]
        self.row = {**self.case, "probabilities": [0.5, 0.5], "log_probabilities": [0., 0.] + [-10000.] * 24,
                    "batch_index": 0, "padded_tokens": 128}

    def verify(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return verify_rows(path, [self.case], self.plan)

    def test_valid_only_real_output(self):
        self.assertEqual(self.verify([self.row]), {("state-1", "q1", 2): [0.5, 0.5]})

    def test_reject_wrong_math_mask_and_provenance(self):
        for key, value in (("probabilities", [0.25, 0.75]), ("batch_index", 1), ("padded_tokens", 256),
                            ("prompt_sha256", "wrong"), ("log_probabilities", [0.] * 26)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                row = copy.deepcopy(self.row)
                row[key] = value
                self.verify([row])

    def test_reject_duplicate_dummy_or_missing_output(self):
        for rows in ([self.row, self.row], []):
            with self.assertRaises(ValueError):
                self.verify(rows)


if __name__ == "__main__":
    unittest.main()
