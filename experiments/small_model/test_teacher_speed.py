"""Validation and leakage checks for the hosted-label speed comparison; no model calls."""
from __future__ import annotations

import unittest
import threading
import time

from teacher_speed import prompt_for, run_batches, teacher_answer, validate_answers


class TeacherSpeedTests(unittest.TestCase):
    def setUp(self):
        self.batch = [{"id": "s1", "input": {"state": "A ticket", "questions": [
            {"instructions": "Urgency", "options": [{"key": "0", "description": "low"},
                                                        {"key": "1", "description": "high"}]}]},
            "references": [{"qid": "score", "type": "score", "teacher": "1", "gold": "0"}]}]

    def test_score_indices_map_to_canonical_keys(self):
        result = validate_answers({"answers": [[0]]}, self.batch)
        self.assertEqual(result["labels"], [{"id": "s1", "qid": "score", "key": "0"}])
        self.assertEqual(result["gold_correct"], 1)
        self.assertEqual(result["teacher_agreed"], 0)

    def test_invalid_or_incomplete_output_is_rejected(self):
        for bad in ({"answers": []}, {"answers": [[0, 0]]}, {"answers": [[True]]},
                    {"answers": [[-1]]}, {"answers": [[2]]}, {"answers": [["0"]]},
                    {"answers": [[0]], "extra": 1}, [], {"answers": [None]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_answers(bad, self.batch)

    def test_prompt_excludes_references_and_ids(self):
        self.batch[0]["references"][0].update({"gold": "SECRET_GOLD", "teacher": "SECRET_TEACHER"})
        self.batch[0]["id"] = "SECRET_STATE_ID"
        prompt = prompt_for(self.batch)
        self.assertNotIn("SECRET", prompt)
        self.assertIn("A ticket", prompt)
        self.assertIn("Urgency", prompt)

    def test_teacher_reverses_second_distribution(self):
        row = {"type": "noul", "keys": ["false", "true"], "z_orig": [0, 1], "z_rev": [10, 0]}
        self.assertEqual(teacher_answer(row), "true")

    def test_stops_new_requests_after_a_failure(self):
        called = []
        def call(index, batch):
            called.append(index)
            return {"error": "usage limit"}
        results = run_batches([[]] * 20, 1, call)
        self.assertEqual(called, [0])
        self.assertEqual(len(results), 1)

    def test_parallelism_is_bounded_and_every_batch_runs_once(self):
        lock = threading.Lock()
        active = peak = 0
        def call(index, batch):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return {"index": index}
        rows = run_batches([[]] * 20, 3, call)
        self.assertEqual(sorted(r["index"] for r in rows), list(range(20)))
        self.assertLessEqual(peak, 3)


if __name__ == "__main__":
    unittest.main()
