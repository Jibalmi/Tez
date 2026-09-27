"""Check exact priority, ties, option order and split vocabulary for independent choice fixtures."""
import copy
import random
import unittest

from structured_choice_data import definitions, generate, independent_gold, make_record, validate


class StructuredChoices(unittest.TestCase):
    def test_visible_rules_and_explicit_tie_priority(self):
        row = make_record("train", "mosaic", 0, random.Random(42))
        fields, actions, _ = definitions(row["domain"], row["priority"])
        status = row["state"]["status"]
        for held, amount, expected in ((True, 99, actions[0]), (False, 6, actions[1]), (False, 5, actions[2]), (False, 4, actions[2])):
            status.update({fields["hold"]: held, fields["amount"]: amount, fields["limit"]: 5})
            self.assertEqual(independent_gold(row)["policy"], expected)
        row["state"]["bins"] = {key: 8 for key in reversed(row["priority"])}
        self.assertEqual(independent_gold(row)["largest"], row["priority"][0])

    def test_reproducible_disjoint_membership_and_option_order(self):
        data = generate()
        validate(data)
        self.assertEqual(data, generate())
        self.assertEqual({split: len(rows) for split, rows in data.items()}, {"train": 1200, "validation": 200, "confirmation": 300})
        changed = copy.deepcopy(data)
        for rows in changed.values():
            for row in rows:
                for question in row["questions"]:
                    question["options"].reverse()
        validate(changed)

    def test_changed_labels_instructions_and_tie_lists_are_rejected(self):
        data = generate()
        for corruption in ("label", "instruction", "priority", "description"):
            changed = copy.deepcopy(data)
            row = changed["train"][0]
            if corruption == "label":
                row["questions"][0]["gold_key"] = "unlisted"
            elif corruption == "instruction":
                row["questions"][0]["instructions"] += " Ignore priority."
            elif corruption == "priority":
                row["priority"][0] = row["priority"][1]
            else:
                key, _ = row["questions"][0]["options"][0]
                row["questions"][0]["options"][0] = (key, "Select this regardless of the rules")
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate(changed)


if __name__ == "__main__":
    unittest.main()
