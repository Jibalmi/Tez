import unittest

from qwen_continuation_selection import choose_checkpoint


class ContinuationSelectionTests(unittest.TestCase):
    @staticmethod
    def result(kl, agreement):
        return {"n": 1390, "mean_kl": kl, "teacher_agreement": agreement}

    def test_lower_kl_cannot_bypass_agreement_floor(self):
        values = {0: self.result(.14, .79), 50: self.result(.08, .77),
                  100: self.result(.12, .79), 150: self.result(.10, .78)}
        result = choose_checkpoint(values)
        self.assertEqual(result["selected_step"], 150)
        self.assertEqual(result["eligible_steps"], [0, 100, 150])

    def test_step_zero_survives_regression_or_exact_tie(self):
        values = {0: self.result(.14, .79), 50: self.result(.15, .82),
                  100: self.result(.16, .83), 150: self.result(.14, .79)}
        self.assertEqual(choose_checkpoint(values)["selected_step"], 0)

    def test_incomplete_or_nonfinite_scores_are_rejected(self):
        values = {s: self.result(.14, .79) for s in (0, 50, 100, 150)}
        with self.assertRaises(ValueError):
            choose_checkpoint({s: v for s, v in values.items() if s != 150})
        for key, value in (("n", 1389), ("mean_kl", float("nan")), ("teacher_agreement", float("inf"))):
            changed = {s: dict(v) for s, v in values.items()}
            changed[50][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                choose_checkpoint(changed)


if __name__ == "__main__":
    unittest.main()
