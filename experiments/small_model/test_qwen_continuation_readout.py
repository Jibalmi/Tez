"""Batched masks must reproduce separate unpadded decision softmaxes."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402

from qwen_continuation_readout import option_logp  # noqa: E402

torch.set_num_threads(1)


class ContinuationReadoutTests(unittest.TestCase):
    def test_mixed_option_counts_match_independent_decisions(self):
        letters = torch.tensor([11, 7, 13, 2, 23, 4, 17, 5, 19, 1, 21, 8, 27, 29, 9, 31])
        # B != K catches the observed exception; B == K catches silent column-wise masking.
        for counts in ([2, 3, 5, 2] * 4, list(range(2, 17)) + [2], [2]):
            logits = (torch.arange(len(counts) * 32).reshape(len(counts), 32) % 19 / 7).to(torch.bfloat16)
            logp = option_logp(logits, letters, counts)
            self.assertEqual(tuple(logp.shape), (len(counts), max(counts)))
            self.assertEqual(logp.dtype, torch.float32)
            for index, count in enumerate(counts):
                expected = torch.softmax(logits[index, letters[:count]].float(), dim=0)
                torch.testing.assert_close(logp[index, :count].exp(), expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(torch.count_nonzero(logp[index, count:].exp()).item(), 0)

    def test_invalid_row_counts_and_option_counts_are_rejected(self):
        logits = torch.zeros(3, 12)
        letters = torch.arange(8)
        for counts in ([2, 3], [2, 3, 9], [2, 3, 0], [True, 3, 4]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                option_logp(logits, letters, counts)


if __name__ == "__main__":
    unittest.main()
