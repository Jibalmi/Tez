"""Semantic remapping must preserve meaning across presentation order and reject corrupt rows."""
import copy
import unittest

from ordinal_probe import make_inputs, prediction_row, score
from structure_probe import generate

REGISTRATION = {"conditions": {"question_type": ["score", "choice"],
                               "semantic_option_order": [["0", "1", "2"], ["1", "2", "0"], ["2", "0", "1"]]}}


class OrdinalProbeTests(unittest.TestCase):
    def test_permutations_keep_facts_instructions_and_semantic_gold(self):
        records = generate()[:1]
        inputs = make_inputs(records, REGISTRATION)
        self.assertEqual(len(inputs), 6)
        self.assertTrue(all(c["state"] == records[0]["states"]["nested"] for c in inputs))
        self.assertEqual(len({c["instructions"] for c in inputs}), 1)
        self.assertEqual(len({c["gold_key"] for c in inputs}), 1)
        for case in inputs:
            order = [key for key, _ in case["options"]]
            p = [.1, .2, .7]
            row = prediction_row(case, [p[int(key)] for key in order])
            self.assertEqual(row["canonical_probabilities"], p)
            self.assertEqual(row["pred_key"], "2")

    def test_scoring_rejects_changed_semantics_and_preserves_native_ties(self):
        inputs = make_inputs(generate()[:3], REGISTRATION)
        rows = [prediction_row(c, [float(k == c["gold_key"]) for k, _ in c["options"]]) for c in inputs]
        metrics = score(inputs, rows, REGISTRATION)
        self.assertTrue(all(m["accuracy"] == 1 for m in metrics["conditions"].values()))
        self.assertEqual(metrics["semantic_order_invariance"], {"score": 1, "choice": 1})
        tie = prediction_row(inputs[1], [1/3, 1/3, 1/3])
        self.assertEqual(tie["pred_key"], "1")
        with self.assertRaises(ValueError):
            score(inputs, rows[:-1], REGISTRATION)
        changed = copy.deepcopy(rows)
        changed[0]["canonical_probabilities"] = [1, 0, 0]
        changed[0]["gold_key"] = "corrupt"
        with self.assertRaises(ValueError):
            score(inputs, changed, REGISTRATION)
        with self.assertRaises(ValueError):
            prediction_row(inputs[0], [float("nan"), 0, 1])


if __name__ == "__main__":
    unittest.main()
