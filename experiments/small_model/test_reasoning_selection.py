"""A new-task gain cannot bypass the registered original-validation guardrails."""
import unittest

from train_reasoning_round2 import selection_key


class ReasoningSelectionTests(unittest.TestCase):
    def test_retention_limits_and_fixed_weight(self):
        baseline = {"agreement": .8, "kl": .15}
        fresh = {"agreement": .9, "kl": .1}
        self.assertIsNone(selection_key({"agreement": .8, "kl": .171}, fresh, baseline, .2))
        self.assertIsNone(selection_key({"agreement": .789, "kl": .15}, fresh, baseline, .2))
        key = selection_key({"agreement": .795, "kl": .16}, fresh, baseline, .2)
        self.assertAlmostEqual(key, .148)


if __name__ == "__main__":
    unittest.main()
