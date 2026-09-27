"""Reject inconsistent checkpoint state independently of the training worker."""
import copy
import os
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402

from verify_qwen_continuation import checkpoint_payload  # noqa: E402

torch.set_num_threads(1)


class CheckpointAuditTests(unittest.TestCase):
    def payload(self, step=0):
        original = {"layer.lora_A.weight": torch.tensor([[.2, -.4]])}
        parameter = torch.nn.Parameter(original["layer.lora_A.weight"].clone())
        optimizer = torch.optim.AdamW([parameter], lr=5e-5, weight_decay=0.)
        for _ in range(step):
            optimizer.zero_grad(set_to_none=True)
            parameter.square().sum().backward()
            optimizer.step()
        name = "layer.lora_A.default.weight"
        saved = {"format": 1, "registration_sha256": "a" * 64,
                 "parameter_names": [name], "adapter": {name: parameter.detach().clone()},
                 "optimizer": optimizer.state_dict(), "state": {"step": step, "decisions": step * 16, "tokens": step * 100},
                 "torch_rng": torch.get_rng_state(), "cuda_rng": [torch.tensor([1, 2], dtype=torch.uint8)]}
        return saved, original

    def test_original_weights_and_real_cpu_adamw_state_are_accepted(self):
        for step in (0, 1, 2):
            saved, original = self.payload(step)
            result = checkpoint_payload(saved, original, "a" * 64, [0, 100, 200])
            self.assertEqual(result["step"], step)
            self.assertEqual(result["changed_from_original"], int(step > 0))

    def test_step_zero_cannot_contain_changed_weights(self):
        saved, original = self.payload()
        saved["adapter"][saved["parameter_names"][0]][0, 0] += 1
        with self.assertRaises(ValueError):
            checkpoint_payload(saved, original, "a" * 64, [0, 100, 200])

    def test_optimizer_progress_and_registration_corruption_are_rejected(self):
        saved, original = self.payload(1)
        for kind in ("lr", "update", "moment", "tokens", "layout", "registration"):
            changed = copy.deepcopy(saved)
            if kind == "lr":
                changed["optimizer"]["param_groups"][0]["lr"] = .001
            elif kind == "update":
                changed["optimizer"]["state"][0]["step"] += 1
            elif kind == "moment":
                changed["optimizer"]["state"][0]["exp_avg"][0, 0] = float("nan")
            elif kind == "tokens":
                changed["state"]["tokens"] += 1
            elif kind == "layout":
                changed["parameter_names"] *= 2
            else:
                changed["registration_sha256"] = "b" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                checkpoint_payload(changed, original, "a" * 64, [0, 100, 200])


if __name__ == "__main__":
    unittest.main()
