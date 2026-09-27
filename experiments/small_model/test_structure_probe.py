"""Check paired semantics, rule boundaries and rejection of corrupted measurements."""
import copy
import unittest

from structure_probe import FORMATS, generate, leaves_by_key, oracle, score, validate_records


class StructureProbeTests(unittest.TestCase):
    def test_same_leaf_facts_and_labels_in_both_forms(self):
        cases = generate()
        validate_records(cases)
        self.assertEqual(cases, generate())
        broken = copy.deepcopy(cases)
        broken[0]["states"]["flat"]["lower_limit"] += 1
        with self.assertRaisesRegex(ValueError, "content"):
            validate_records(broken)

    def test_rule_boundaries_and_unknown_values(self):
        spec = {"mode": "mode", "quantity": "value", "flag": "ready", "optional": "site"}
        for value, band, truth in ((9, "0", "false"), (10, "1", "true"), (20, "1", "true"), (21, "2", "true")):
            facts = {"mode": "violet", "value": value, "ready": True, "lower_limit": 10, "upper_limit": 20}
            expected = {"lookup": "violet", "band": band, "threshold": truth, "conjunction": truth, "missing": "__none__"}
            self.assertEqual(oracle({"record": facts}, spec), expected)
            facts["ready"] = False
            facts["site"] = None
            self.assertEqual(oracle(facts, spec)["conjunction"], "false")
            self.assertEqual(oracle(facts, spec)["missing"], "__none__")
            facts["site"] = "south"
            self.assertEqual(oracle(facts, spec)["missing"], "south")

    def test_ambiguous_leaf_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            leaves_by_key({"a": {"x": 1}, "b": {"x": 2}})

    def test_score_pairs_clusters_and_validates_every_row(self):
        cases = generate()[:2]
        rows = []
        for record in cases:
            for form in FORMATS:
                for q in record["questions"]:
                    pred = q["gold"] if form == "flat" else (q["gold"] + 1) % len(q["options"])
                    rows.append({"id": record["id"], "format": form, "operation": q["operation"], "gold": q["gold"],
                                 "pred": pred, "probabilities": [float(i == pred) for i in range(len(q["options"]))]})
        result = score(cases, rows)
        self.assertEqual(result["nested_minus_flat_points"], -100)
        self.assertEqual(result["paired_state_bootstrap_ci95_points"], [-100, -100])
        self.assertEqual(result["prediction_flip_fraction"], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            score(cases, rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            score(cases, rows + rows[:1])
        bad = copy.deepcopy(rows)
        bad[0]["probabilities"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            score(cases, bad)


if __name__ == "__main__":
    unittest.main()
