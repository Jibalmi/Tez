"""Check accumulated single-decision gradients against the original padded batch loss."""
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qwen_continuation_loss import decision_loss  # noqa: E402

torch.set_num_threads(1)


class ContinuationLossTests(unittest.TestCase):
    def test_accumulation_matches_original_batch_loss_and_shared_parameter_gradient(self):
        # Missing gold rows still count in the original loss denominator.
        targets = [torch.tensor(v) for v in ([.6, .4], [0., .5, .5], [.1, .2, .3, .4], [.2] * 5)]
        gold = [1, None, None, 3]
        features = torch.tensor([[.1, .4], [-.8, .5], [.9, -.2], [.6, -.5]])
        weights = torch.tensor([[.2, -.1, .8, .3, -.9], [-.7, .6, .4, -.2, .5]], requires_grad=True)
        sizes = torch.tensor([len(t) for t in targets])
        mask = torch.arange(5)[None, :] < sizes[:, None]
        logp = F.log_softmax(torch.where(mask, features @ weights, torch.full((4, 5), -1e4)), dim=-1)
        padded = torch.stack([F.pad(t, (0, 5 - len(t))) for t in targets])
        dense = (padded * (padded.clamp_min(1e-12).log() - logp) * mask).sum(-1).mean()
        indices = torch.tensor([g if g is not None else -100 for g in gold])
        dense = dense + .5 * F.nll_loss(logp, indices, ignore_index=-100) * (indices >= 0).float().mean()
        dense.backward()
        expected_gradient = weights.grad.clone()
        weights.grad = None
        accumulated = 0.
        for x, target, label in zip(features, targets, gold, strict=True):
            loss = decision_loss((x @ weights)[:len(target)], target, label) / len(targets)
            accumulated += loss.detach().item()
            loss.backward()
        self.assertAlmostEqual(accumulated, dense.detach().item(), places=6)
        torch.testing.assert_close(weights.grad, expected_gradient, rtol=1e-6, atol=1e-7)

    def test_option_permutation_preserves_loss_and_canonical_gradient(self):
        logits = torch.tensor([.5, -.8, .1, .9], requires_grad=True)
        target = torch.tensor([.3, .1, 0., .6])
        canonical = decision_loss(logits, target, 3)
        canonical.backward()
        expected = logits.grad.clone()
        logits.grad = None
        order = [2, 0, 3, 1]
        permuted = decision_loss(logits[order], target[order], order.index(3))
        permuted.backward()
        torch.testing.assert_close(permuted, canonical, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-7)

    def test_invalid_targets_and_gold_are_rejected(self):
        logits = torch.tensor([.2, -.3])
        for target in (torch.tensor([.1, .1]), torch.tensor([-1., 2.]), torch.tensor([float('nan'), .5]),
                       torch.tensor([1, 0]), torch.tensor([.5, .5], requires_grad=True), torch.tensor([1.])):
            with self.subTest(target=target), self.assertRaises(ValueError):
                decision_loss(logits, target)
        for gold in (-1, 2, True, 1.0):
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                decision_loss(logits, torch.tensor([.5, .5]), gold)


if __name__ == "__main__":
    unittest.main()
