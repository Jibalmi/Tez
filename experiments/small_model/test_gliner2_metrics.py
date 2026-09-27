"""Check the raw probability measurements without importing a model or fitting calibration."""
import math
import unittest

from eval_gliner2 import raw_metrics


class RawMetricsTest(unittest.TestCase):
    def test_known_distributions(self):
        rows = [{"pred": 1, "gold": 1, "probabilities": [.25, .75], "extra": {"soft": [.2, .8]}},
                {"pred": 0, "gold": 1, "probabilities": [.75, .25]}]
        value = raw_metrics(rows)
        self.assertEqual(value["accuracy"], .5)
        self.assertAlmostEqual(value["ece"], .25)
        self.assertAlmostEqual(value["brier"], .625)
        self.assertAlmostEqual(value["soft_accuracy"], .65)
        self.assertAlmostEqual(value["nll"], -math.log(.75 * .25) / 2)
        self.assertAlmostEqual(value["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
