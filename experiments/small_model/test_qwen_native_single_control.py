"""Reject incomplete native-control predictions and inconsistent option probabilities."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from qwen_native_single_control import verify_control_rows


class NativeControlTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"state_id": "s", "qid": "q-long", "k": 2, "qtype": "choice", "prompt": "longer",
                       "prompt_sha256": "long", "encoded_tokens": 3},
                      {"state_id": "s", "qid": "q-short", "k": 2, "qtype": "noul", "prompt": "s",
                       "prompt_sha256": "short", "encoded_tokens": 1}]
        self.rows = []
        for index, case in enumerate(reversed(self.cases)):
            self.rows.append({**{k: v for k, v in case.items() if k != "prompt"}, "native_batch": index,
                              "batch_size": 1, "log_probabilities": [math.log(.3), math.log(.7)],
                              "probabilities": [.3, .7]})

    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return verify_control_rows(path, self.cases)

    def test_complete_native_order_and_probabilities(self):
        values = self.read(self.rows)
        self.assertEqual(len(values), 2)
        self.assertEqual(values[("s", "q-short", 2)], [.3, .7])

    def test_missing_duplicate_reordered_rows_rejected(self):
        for rows in (self.rows[:1], [self.rows[0], self.rows[0]], list(reversed(self.rows))):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read(rows)

    def test_changed_provenance_or_numerical_readout_rejected(self):
        for field, value in (("batch_size", 16), ("native_batch", 1), ("prompt_sha256", "wrong"),
                             ("encoded_tokens", 9), ("qtype", "score"), ("probabilities", [.4, .6]),
                             ("log_probabilities", [0, 0]), ("log_probabilities", [float("nan"), 0]),
                             ("log_probabilities", [math.log(.3)])):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(rows)


if __name__ == "__main__":
    unittest.main()
