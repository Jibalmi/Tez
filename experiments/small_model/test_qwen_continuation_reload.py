"""Reject reload of a checkpoint inconsistent with the fixed validation-only selection."""
import copy
import unittest

from qwen_continuation_reload import selected_step
from qwen_continuation_selection import choose_checkpoint


class ReloadSelectionTests(unittest.TestCase):
    @staticmethod
    def selection():
        metrics = {step: {"n": 1390, "mean_kl": kl, "teacher_agreement": agreement}
                   for step, kl, agreement in ((0, .14, .79), (50, .11, .79), (100, .08, .76), (150, .12, .8))}
        return {"status": "complete", "continuation_registration_sha256": "a" * 64,
                "export_registration_sha256": "b" * 64, **choose_checkpoint(metrics),
                "validation_metrics": {str(key): value for key, value in metrics.items()},
                "retained_original_weights": False, "validation_evidence": {str(step): {} for step in metrics}}

    def test_reload_uses_eligible_validation_choice(self):
        selection = self.selection()
        self.assertEqual(selected_step(selection, "a" * 64, "b" * 64), 50)

    def test_reject_different_checkpoint_registration_or_missing_validation(self):
        original = self.selection()
        for key, value in (("selected_step", 100), ("eligible_steps", [0, 50, 100, 150]),
                            ("retained_original_weights", True), ("status", "running"),
                            ("continuation_registration_sha256", "x" * 64),
                            ("export_registration_sha256", "x" * 64)):
            selection = copy.deepcopy(original)
            selection[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                selected_step(selection, "a" * 64, "b" * 64)
        for field in ("validation_metrics", "validation_evidence"):
            selection = copy.deepcopy(original)
            del selection[field]["150"]
            with self.subTest(field=field), self.assertRaises(ValueError):
                selected_step(selection, "a" * 64, "b" * 64)

    def test_unchanged_model_requires_consistent_selection_metadata(self):
        selection = self.selection()
        for result in selection["validation_metrics"].values():
            result.update(mean_kl=.14, teacher_agreement=.79)
        selection.update(choose_checkpoint({int(key): value for key, value in selection["validation_metrics"].items()}))
        selection["retained_original_weights"] = True
        self.assertEqual(selected_step(selection, "a" * 64, "b" * 64), 0)


if __name__ == "__main__":
    unittest.main()
