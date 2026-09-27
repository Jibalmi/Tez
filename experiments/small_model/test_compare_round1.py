"""Matched uncertainty must preserve state clusters and reject inconsistent labels."""
import unittest

from compare_round1 import paired_typed_delta


class PairedComparisonTests(unittest.TestCase):
    def test_keeps_clustered_decisions_together(self):
        cases = [{"id": str(i), "group": str(i // 2), "gold": 0} for i in range(4)]
        left = {str(i): {"gold": 0, "pred": 0} for i in range(4)}
        right = {str(i): {"gold": 0, "pred": int(i < 2)} for i in range(4)}
        result = paired_typed_delta(cases, left, right, reps=200)
        self.assertEqual(result["accuracy_delta_points"], 50)
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["ci95_points"], [0, 100])

    def test_rejects_misaligned_gold_and_missing_decisions(self):
        cases = [{"id": "a", "group": "state", "gold": 0}]
        with self.assertRaises(ValueError):
            paired_typed_delta(cases, {}, {})
        with self.assertRaises(ValueError):
            paired_typed_delta(cases, {"a": {"gold": 0, "pred": 0}}, {"a": {"gold": 1, "pred": 1}})


if __name__ == "__main__":
    unittest.main()
