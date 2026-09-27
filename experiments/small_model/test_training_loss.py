"""Order-sensitive loss remains invariant when score options are shuffled."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from student import ordinal_loss
from train_a import ordered_score_loss


class ScoreOrderTests(unittest.TestCase):
    def test_permuted_levels_restore_loss_and_gradients_with_padding(self):
        p = torch.tensor([[0.1, 0.2, 0.7, 1e-12]], requires_grad=True)
        target = torch.tensor([[0.6, 0.3, 0.1, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        expected = ordinal_loss(p.log(), target, mask)
        expected_grad = torch.autograd.grad(expected.sum(), p, retain_graph=True)[0]
        perm = [2, 0, 1, 3]
        row = SimpleNamespace(meta={"canonical_order": [np.argsort(perm[:3]).tolist()]})
        actual = ordered_score_loss(p[:, perm].log(), target[:, perm], mask[:, perm], [row])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), p)[0], expected_grad)
        self.assertFalse(torch.allclose(ordinal_loss(p[:, perm].log(), target[:, perm], mask[:, perm]), expected))


if __name__ == "__main__":
    unittest.main()
