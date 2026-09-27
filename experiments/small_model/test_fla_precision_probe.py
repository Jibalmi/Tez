"""Reject malformed vectors and enforce all independent probability limits without a GPU."""
import math
import unittest

from fla_precision_probe import compare


class ProbabilityLimits(unittest.TestCase):
    def test_small_change_preserves_winner(self):
        self.assertTrue(compare([[.6, .4]], [[.604, .396]], [2])["passed"])

    def test_each_failure_limit_is_independent(self):
        maximum = compare([[.6, .4]] * 100, [[.6, .4]] * 99 + [[.65, .35]], [2] * 100)
        self.assertFalse(maximum["passed"])
        self.assertLess(maximum["mean_total_variation"], .005)
        variation = compare([[.6, .4]], [[.61, .39]], [2])
        self.assertFalse(variation["passed"])
        self.assertLess(variation["max_probability_delta"], .04)
        winner = compare([[.499, .501]], [[.501, .499]], [2])
        self.assertFalse(winner["passed"])
        self.assertLess(winner["mean_total_variation"], .005)

    def test_incomplete_or_invalid_predictions_are_not_scores(self):
        for candidate in ([], [[.5]], [[math.nan, .5]], [[math.inf, .5]], [[-.1, 1.1]], [[.2, .2]]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare([[.6, .4]], candidate, [2])


if __name__ == "__main__":
    unittest.main()
