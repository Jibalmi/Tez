"""CPU checks of complete-input preservation, bucket boundaries and masked option rows."""
import os
import unittest
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402 -- Set numerical library limits before importing Torch.
from graph_qwen import batch_plan, pack_inputs  # noqa: E402

torch.set_num_threads(1)


class GraphInputTests(unittest.TestCase):
    def test_boundary_membership_and_final_duplicate(self):
        lengths = [4, 1, 7, 3, 8, 2]
        plan = batch_plan(lengths, 2, [4, 8, 16])
        self.assertEqual(plan, [
            {"length": 4, "indices": [1, 5], "dummy_rows": 0},
            {"length": 4, "indices": [3], "dummy_rows": 1},
            {"length": 8, "indices": [0, 2], "dummy_rows": 0},
            {"length": 16, "indices": [4], "dummy_rows": 1},
        ])
        self.assertEqual(sorted(i for row in plan for i in row["indices"]), list(range(6)))
        ids, mask, allowed = pack_inputs([[8, 9, 10]], [3], 2, 4, 99)
        self.assertEqual(ids.tolist(), [[99, 8, 9, 10]] * 2)
        self.assertEqual(mask.tolist(), [[0, 1, 1, 1]] * 2)
        self.assertEqual(allowed.sum(1).tolist(), [3, 3])

    def test_variable_counts_and_real_pad_token(self):
        ids, mask, allowed = pack_inputs([[0, 8], [9]], [26, 2], 2, 4, 0)
        self.assertEqual(ids.tolist(), [[0, 0, 0, 8], [0, 0, 0, 9]])
        # A real token equal to pad_id must still be attended to.
        self.assertEqual(mask.tolist(), [[0, 0, 1, 1], [0, 0, 0, 1]])
        self.assertEqual(allowed.sum(1).tolist(), [26, 2])
        self.assertTrue(allowed[1, :2].all())
        self.assertFalse(allowed[1, 2:].any())

    def test_reject_clipping_implicit_masks_and_invalid_options(self):
        for lengths, batch, buckets in (([16], 1, [4, 16]), ([0], 1, [4]), ([2], 0, [4]),
                                         ([2], 1, [8, 4]), ([2], 1, [4, 4])):
            with self.assertRaises(ValueError):
                batch_plan(lengths, batch, buckets)
        for rows, counts in (([[1, 2, 3, 4]], [2]), ([[]], [2]), ([[1]], [27]),
                             ([[1]], [True]), ([[1]], []), ([[-1]], [2])):
            with self.assertRaises(ValueError):
                pack_inputs(rows, counts, 1, 4, 0)


if __name__ == "__main__":
    unittest.main()
