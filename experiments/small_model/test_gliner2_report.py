"""Semantic class identity must survive independently reordered answer options."""
import unittest

from report_gliner2 import semantic_macro_f1


class SemanticF1Test(unittest.TestCase):
    def test_option_positions_are_not_semantic_classes(self):
        cases = [{"id": str(i), "options": [("a", "A"), ("b", "B")] if i % 2 == 0 else [("b", "B"), ("a", "A")],
                  "gold": i % 2} for i in range(4)]
        rows = {str(i): {"pred": prediction} for i, prediction in enumerate([1, 0, 0, 1])}
        # Gold is always semantic A; two A predictions and two B predictions yield macro-F1 1/3.
        # Treating the alternating positions as classes would incorrectly report 1/2.
        self.assertAlmostEqual(semantic_macro_f1(cases, rows), 1 / 3)
        flipped = [{**case, "options": list(reversed(case["options"])), "gold": 1 - case["gold"]} for case in cases]
        reversed_rows = {key: {"pred": 1 - row["pred"]} for key, row in rows.items()}
        self.assertAlmostEqual(semantic_macro_f1(flipped, reversed_rows), semantic_macro_f1(cases, rows))


if __name__ == "__main__":
    unittest.main()
