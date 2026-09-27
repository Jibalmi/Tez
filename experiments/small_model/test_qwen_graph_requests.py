"""Preserve request membership and expose missing outputs before any GPU comparison."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest  # noqa: E402

from qwen_graph_requests import request_batches, restore_request  # noqa: E402


class RequestTests(unittest.TestCase):
    def test_fifty_questions_keep_native_sixteen_sixteen_sixteen_two(self):
        cases = [{"prompt": "p" * (50 - i), "encoded_tokens": 20 + i} for i in range(50)]
        plan = request_batches(cases, [128, 256])
        expected = list(reversed(range(50)))
        self.assertEqual([batch["batch_size"] for batch in plan], [16, 16, 16, 2])
        self.assertEqual([i for batch in plan for i in batch["indices"]], expected)
        self.assertEqual([batch["native_length"] for batch in plan], [69, 53, 37, 21])
        values = [[f"answer-{i}" for i in batch["indices"]] for batch in plan]
        self.assertEqual(restore_request(plan, values, 50), [f"answer-{i}" for i in range(50)])

    def test_short_requests_are_not_padded_with_duplicate_decisions(self):
        for size in (1, 5, 10):
            plan = request_batches([{"prompt": "same", "encoded_tokens": 128}] * size, [128, 256])
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["indices"], list(range(size)))
            self.assertEqual(plan[0]["batch_size"], size)
            self.assertEqual(plan[0]["length"], 256)

    def test_execution_order_survives_shape_changes(self):
        cases = [{"prompt": "p" * i, "encoded_tokens": 30 if i < 16 else 200} for i in range(33)]
        plan = request_batches(cases, [128, 256])
        self.assertEqual([batch["original_batch"] for batch in plan], [0, 1, 2])
        self.assertEqual([(b["batch_size"], b["length"]) for b in plan], [(16, 128), (16, 256), (1, 256)])

    def test_reject_missing_duplicate_or_invalid_outputs(self):
        for batches, values, size in (([{"indices": [0, 0]}], [[1, 2]], 2),
                                      ([{"indices": [0]}], [[1]], 2),
                                      ([{"indices": [-1]}], [[1]], 1),
                                      ([{"indices": [1]}], [[1]], 1),
                                      ([{"indices": [0, 1]}], [[1]], 2)):
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                restore_request(batches, values, size)


if __name__ == "__main__":
    unittest.main()
