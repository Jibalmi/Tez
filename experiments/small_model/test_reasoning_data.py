"""Verify exact supervision, tie policy, negation, split isolation and visible-fact consistency."""
import copy
import unittest

from reasoning_data import generate, labels, validate


class ReasoningDataTests(unittest.TestCase):
    def test_boundaries_negation_and_priority(self):
        facts = {"usage": 8, "cap": 8, "checked": True, "held": False}
        racks = {"rack_elm": 3, "rack_fir": 3, "rack_oak": 1}
        self.assertEqual(labels(facts, racks, False), {"compare": "false", "combined": "true", "policy": "dispatch_cycle",
                                                      "count": "0", "largest": "rack_elm"})
        self.assertEqual(labels(facts, racks, True)["combined"], "false")
        facts.update(usage=9, held=True)
        self.assertEqual(labels(facts, racks, False), {"compare": "true", "combined": "false", "policy": "pause_cycle",
                                                      "count": "2", "largest": "rack_elm"})
        facts["held"] = False
        self.assertEqual(labels(facts, racks, False)["policy"], "buffer_cycle")
        self.assertEqual(labels(facts, racks, False)["count"], "1")

    def test_deterministic_disjoint_data_and_rejection_of_corrupt_facts(self):
        data = generate()
        validate(data)
        self.assertEqual(data, generate())
        changed = copy.deepcopy(data)
        record = changed["train"][0]
        record["state"]["record"][record["field_mapping"]["cap"]] += 1
        with self.assertRaisesRegex(ValueError, "visible facts"):
            validate(changed)
        changed = copy.deepcopy(data)
        changed["train"][0]["questions"][0]["gold_key"] = "bad"
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate(changed)


if __name__ == "__main__":
    unittest.main()
