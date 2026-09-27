"""Corrupted request coverage, probabilities and cache histories cannot yield a parity pass."""
import copy
import math
import unittest

from verify_qwen_graph_requests import json_sha, probabilities, summarize, verify_saved_rows


SPEC = {"allowed_shapes": [[1, 128]], "argmax_agreement_min": .99, "max_probability_delta_max": .04,
        "mean_total_variation_max": .005}


def fixture():
    audits, rows = [], []
    batch = {"original_batch": 0, "indices": [0], "batch_size": 1, "native_length": 50, "length": 128}
    for index in range(2):
        audit = {"request": index + 1, "n_questions": 1, "phase": "warmup", "batches": [batch], "cases": [{"k": 2}]}
        vector = [math.log(.3), math.log(.7)]
        row = {"request": index + 1, "n_questions": 1, "phase": "warmup", "audit_sha256": json_sha(audit),
               "batches": [{**batch, "graph_replay": index + 1}], "resident_shapes": [[1, 128]], "captures_total": 1,
               "reference_log_probabilities": [vector], "graph_log_probabilities": [vector],
               "uncaptured_log_probabilities": [vector]}
        audits.append(audit)
        rows.append(row)
    return rows, audits


class VerificationTests(unittest.TestCase):
    def test_reuse_has_one_capture_and_complete_predictions(self):
        rows, audits = fixture()
        full, eager, captures = verify_saved_rows(rows, audits, SPEC)
        self.assertTrue(full["1"]["passed"])
        self.assertEqual(full["1"]["decisions"], 2)
        self.assertEqual(eager["1"]["max_probability_delta"], 0)
        self.assertEqual(captures, [{"shape": [1, 128], "request_index": 0}])

    def test_missing_reordered_or_duplicate_requests_rejected(self):
        rows, audits = fixture()
        for changed in (rows[:1], list(reversed(rows)), [rows[0], rows[0]]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verify_saved_rows(changed, audits, SPEC)

    def test_changed_batch_lifetime_or_input_hash_rejected(self):
        rows, audits = fixture()
        for key, value in (("captures_total", 2), ("resident_shapes", []), ("audit_sha256", "different"),
                           ("batches", [{**rows[1]["batches"][0], "graph_replay": 1}]),
                           ("batches", [{**rows[1]["batches"][0], "indices": [1]}])):
            changed = copy.deepcopy(rows)
            changed[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_saved_rows(changed, audits, SPEC)

    def test_missing_unmasked_or_unnormalized_probabilities_rejected(self):
        for values in ([0, 0], [0], [float("nan"), 0], [True, False], [0, -1, -2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                probabilities(values, 2)

    def test_each_original_parity_limit_remains_required(self):
        self.assertTrue(summarize([([.3, .7], [.3, .7])], SPEC)["passed"])
        for pair in (([.499, .501], [.501, .499]), ([.3, .7], [.31, .69]), ([.3, .7], [.35, .65])):
            self.assertFalse(summarize([pair], SPEC)["passed"])


if __name__ == "__main__":
    unittest.main()
