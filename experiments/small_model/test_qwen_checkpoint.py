"""Verify optimizer/RNG replay and reject incompatible checkpoints on one CPU thread."""
import os
from pathlib import Path
import tempfile
import unittest

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import torch  # noqa: E402 -- cap CPU libraries before importing Torch

from qwen_checkpoint import load_checkpoint, save_checkpoint  # noqa: E402

torch.set_num_threads(1)
REGISTRATION = "a" * 64


def parameters():
    a = torch.nn.Parameter(torch.randn(3, 2))
    b = torch.nn.Parameter(torch.randn(2))
    return [("layer.lora_A", a), ("layer.lora_B", b)]


def step(named, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    loss = ((x @ named[0][1] + named[1][1] - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


class Replay(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / "runs/cpu-checkpoint-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(root.resolve()))
        self.addCleanup(self.temporary.cleanup)
        torch.manual_seed(31)
        self.named = parameters()
        self.optimizer = torch.optim.AdamW([p for _, p in self.named], lr=.01)

    def test_resume_reproduces_updates_and_random_stream(self):
        step(self.named, self.optimizer)
        path = self.folder / "step1.pt"
        receipt = save_checkpoint(path, self.named, self.optimizer, {"step": 1}, REGISTRATION)
        expected_loss = [step(self.named, self.optimizer) for _ in range(3)]
        expected_weights = [p.detach().clone() for _, p in self.named]
        expected_random = torch.rand(5)
        other = parameters()
        optimizer = torch.optim.AdamW([p for _, p in other], lr=.5)
        state = load_checkpoint(path, receipt["sha256"], other, optimizer, REGISTRATION)
        self.assertEqual(state, {"step": 1})
        self.assertEqual(expected_loss, [step(other, optimizer) for _ in range(3)])
        for expected, (_, actual) in zip(expected_weights, other, strict=True):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_random, torch.rand(5)))
        with self.assertRaises(ValueError):
            save_checkpoint(path, other, optimizer, state, REGISTRATION)

    def test_wrong_registration_or_content_is_rejected_before_parameter_copy(self):
        path = self.folder / "state.pt"
        receipt = save_checkpoint(path, self.named, self.optimizer, {"step": 0}, REGISTRATION)
        before = [p.detach().clone() for _, p in self.named]
        for checksum, registration in (("b" * 64, REGISTRATION), (receipt["sha256"], "c" * 64)):
            with self.subTest(checksum=checksum), self.assertRaises(ValueError):
                load_checkpoint(path, checksum, self.named, self.optimizer, registration)
        for expected, (_, actual) in zip(before, self.named, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_optimizer_reordering_and_nonfinite_weights_are_rejected(self):
        optimizer = torch.optim.AdamW([p for _, p in reversed(self.named)])
        with self.assertRaises(ValueError):
            save_checkpoint(self.folder / "bad_order.pt", self.named, optimizer, {}, REGISTRATION)
        with torch.no_grad():
            self.named[0][1].fill_(float("nan"))
        with self.assertRaises(ValueError):
            save_checkpoint(self.folder / "nan.pt", self.named, self.optimizer, {}, REGISTRATION)


if __name__ == "__main__":
    unittest.main()
