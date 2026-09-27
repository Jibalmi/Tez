"""The native competitor adapter preserves question meaning and caller option order."""
import unittest

from competitor_models import kev_request, decider_probabilities


class KevRequestTests(unittest.TestCase):
    def test_decider_native_boolean_order_and_rounded_score_probabilities(self):
        case = {"qtype": "noul", "options": [("true", "yes"), ("false", "no")]}
        self.assertEqual(decider_probabilities(case, {"noul": .75}).tolist(), [.75, .25])
        case = {"qtype": "score", "options": [("low", "no"), ("medium", "some"), ("high", "yes")]}
        values = decider_probabilities(case, {"probabilities": {"0": .3333, "1": .3333, "2": .3333}})
        self.assertAlmostEqual(sum(values), 1)
        self.assertEqual(list(values), [1 / 3] * 3)

    def test_boolean_order_and_score_levels_map_back_without_changing_meaning(self):
        group = [
            {"state": "fixture", "qtype": "noul", "instructions": "Is it true?",
             "options": [("true", "yes"), ("false", "no")]},
            {"state": "fixture", "qtype": "score", "instructions": "Rate it",
             "options": [("level 0", "low"), ("level 1", "high")]},
            {"state": "fixture", "qtype": "choice", "instructions": "Choose",
             "options": [("b", "second"), ("a", "first")]},
        ]
        record, keys = kev_request(group)
        self.assertEqual(keys, [["true", "false"], ["0", "1"], ["b", "a"]])
        self.assertEqual(record["questions"]["1"]["criteria"], ["low", "high"])
        self.assertEqual(record["questions"]["2"]["criteria"], {"b": "second", "a": "first"})
        self.assertEqual(record["state"], "fixture")


if __name__ == "__main__":
    unittest.main()
