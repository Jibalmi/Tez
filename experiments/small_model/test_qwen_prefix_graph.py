"""CPU checks for unchanged native token positions, batch composition and final-token reads."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402
from qwen_prefix_graph import native_batch_plan, native_final_hidden, pack_native_prefix  # noqa: E402

torch.set_num_threads(1)


class NativePrefixTests(unittest.TestCase):
    def test_keep_native_batch_composition_including_short_final_batch(self):
        cases = [{"prompt": "x" * chars, "encoded_tokens": tokens}
                 for chars, tokens in ((3, 20), (1, 50), (5, 1), (2, 3), (4, 2))]
        self.assertEqual(native_batch_plan(cases, 2, [8, 32, 64]), [
            {"original_batch": 1, "indices": [0, 4], "batch_size": 2, "native_length": 20, "length": 32},
            {"original_batch": 0, "indices": [1, 3], "batch_size": 2, "native_length": 50, "length": 64},
            {"original_batch": 2, "indices": [2], "batch_size": 1, "native_length": 1, "length": 8},
        ])

    def test_native_prefix_masks_real_pad_token_and_option_counts(self):
        ids, mask, allowed, end = pack_native_prefix([[99, 4], [5, 6, 7]], [2, 26], 2, 8, 99)
        self.assertEqual(ids.tolist(), [[99, 99, 4, 99, 99, 99, 99, 99], [5, 6, 7, 99, 99, 99, 99, 99]])
        self.assertEqual(mask.tolist(), [[0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
        self.assertEqual(allowed.sum(1).tolist(), [2, 26])
        self.assertEqual(end.tolist(), [2])
        self.assertFalse(allowed[0, 2:].any())

    def test_read_native_end_and_update_index_without_using_padded_end(self):
        hidden = torch.arange(2 * 8 * 3).reshape(2, 8, 3).float()
        end = torch.tensor([2])
        torch.testing.assert_close(native_final_hidden(hidden, end), hidden[:, 2:3, :], rtol=0, atol=0)
        end.copy_(torch.tensor([4]))
        torch.testing.assert_close(native_final_hidden(hidden, end), hidden[:, 4:5, :], rtol=0, atol=0)
        self.assertFalse(torch.equal(native_final_hidden(hidden, end), hidden[:, -1:, :]))

    def test_reject_clipping_filling_or_invalid_positions(self):
        for rows, counts, batch, length in (([[1, 2, 3]], [2], 1, 3), ([[1]], [2], 2, 8),
                                            ([[]], [2], 1, 8), ([[1]], [27], 1, 8)):
            with self.assertRaises(ValueError):
                pack_native_prefix(rows, counts, batch, length, 99)
        with self.assertRaises(ValueError):
            native_final_hidden(torch.zeros(1, 8, 3), torch.tensor([2.]))
        with self.assertRaises(ValueError):
            native_batch_plan([{"prompt": "x", "encoded_tokens": 64}], 1, [8, 32, 64])


if __name__ == "__main__":
    unittest.main()
