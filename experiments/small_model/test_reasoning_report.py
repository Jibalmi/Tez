"""The continuation report must reject incomplete and inconsistent probability records."""
import copy
import unittest

import numpy as np

from report_reasoning_round2 import validate_rows


class ReasoningReportTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "one", "gold": 1, "qtype": "choice", "options": [["x", "X"], ["y", "Y"]]}]
        self.rows = {"one": {"id": "one", "task": "fixture", "gold": 1, "qtype": "choice", "k": 2,
                             "probabilities": [.25, .75], "logits": np.log([.25, .75]).tolist(), "pred": 1}}

    def test_valid_and_missing_records(self):
        self.assertEqual(validate_rows(self.cases, self.rows, "fixture"), 1.)
        with self.assertRaises(ValueError):
            validate_rows(self.cases, {}, "fixture")

    def test_rejects_wrong_semantics_and_probability_mapping(self):
        for field, value in (("gold", 0), ("pred", 0), ("probabilities", [.75, .25]),
                             ("logits", [0., 0.]), ("probabilities", [float("nan"), .75])):
            rows = copy.deepcopy(self.rows)
            rows["one"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_rows(self.cases, rows, "fixture")


if __name__ == "__main__":
    unittest.main()
