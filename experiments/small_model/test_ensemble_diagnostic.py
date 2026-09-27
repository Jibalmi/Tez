import unittest

from ensemble_diagnostic import blend


class EnsembleDiagnosticTests(unittest.TestCase):
    def test_fixed_probabilities_and_corrupted_alignment(self):
        case = {"id": "a", "gold": 1, "qtype": "choice", "options": [("a", "A"), ("b", "B")]}
        left = {"a": {"gold": 1, "qtype": "choice", "pred": 0, "probabilities": [.6, .4]}}
        right = {"a": {"gold": 1, "qtype": "choice", "pred": 1, "probabilities": [.1, .9]}}
        rows, summary = blend([case], left, right)
        self.assertAlmostEqual(rows["a"]["probabilities"][1], .65)
        self.assertEqual(summary["complementarity_counts"]["right_only_correct"], 1)
        self.assertEqual(summary["equal_blend_accuracy"], 1)
        with self.assertRaises(ValueError):
            blend([case], {}, right)
        right["a"]["probabilities"] = [1, 1]
        with self.assertRaises(ValueError):
            blend([case], left, right)


if __name__ == "__main__":
    unittest.main()
