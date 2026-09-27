"""CPU export checks using a real tiny PEFT model and corrupted validation fixtures."""
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(USE_TF="0", CUDA_VISIBLE_DEVICES="")

import torch  # noqa: E402
from export_qwen_continuation import canonical_adapter, main, save_adapter, validation_metrics  # noqa: E402

torch.set_num_threads(1)


class AdapterExportTests(unittest.TestCase):
    def test_real_peft_cpu_roundtrip_preserves_outputs_and_its_canonical_keys(self):
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import BertConfig, BertModel
        torch.manual_seed(29)
        config = BertConfig(vocab_size=32, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
                            intermediate_size=24, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        base = BertModel(config)
        initial_base = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"], bias="none"))
        original = {name: tensor.clone() for name, tensor in get_peft_model_state_dict(model).items()}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.copy_(torch.randn_like(parameter) * .1)
        model.eval()
        adapter = {name: value for name, value in model.named_parameters() if "lora_" in name}
        expected = get_peft_model_state_dict(model)
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        with torch.inference_mode():
            expected_output = model(input_ids=inputs).last_hidden_state.clone()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model.peft_config["default"].save_pretrained(root / "config")
            saved = save_adapter(adapter, original, root / "config/adapter_config.json", root / "adapter")
            self.assertEqual(saved["tensors_verified"], len(expected))
            actual = load_file(root / "adapter/adapter_model.safetensors")
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(all(torch.equal(actual[key], value) for key, value in expected.items()))
            restored_base = BertModel(config)
            restored_base.load_state_dict(initial_base)
            restored = PeftModel.from_pretrained(restored_base, root / "adapter").eval()
            with torch.inference_mode():
                actual_output = restored(input_ids=inputs).last_hidden_state
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_adapter(adapter, original, root / "config/adapter_config.json", root / "adapter")

    def test_reject_missing_unexpected_nonfinite_and_changed_layout(self):
        name = "base_model.model.block.lora_A.default.weight"
        original = {name.replace(".default.", "."): torch.ones(2, 3)}
        for tensors in ({}, {name: torch.ones(3, 2)}, {name: torch.ones(2, 3, dtype=torch.float64)},
                        {name: torch.full((2, 3), float("nan"))}, {"unrelated.weight": torch.ones(2, 3)},
                        {name.replace(".default.", ".other."): torch.ones(2, 3)}):
            with self.assertRaises(ValueError):
                canonical_adapter(tensors, original)

    def test_only_remove_the_adapter_segment(self):
        name = "base_model.model.default.block.lora_A.default.weight"
        key = "base_model.model.default.block.lora_A.weight"
        value = torch.tensor([[1., 2.]])
        self.assertEqual(set(canonical_adapter({name: value}, {key: value})), {key})


class ValidationSelectionInputTests(unittest.TestCase):
    @staticmethod
    def rows():
        p, target = [.75, .25], [.5, .5]
        kl = sum(t * math.log(t / q) for t, q in zip(target, p, strict=True))
        return [{"state_id": f"state-{i // 5}", "qid": f"q{i % 5}", "k": 2, "qtype": "choice",
                 "probabilities": p.copy(), "target": target.copy(), "log_probabilities": [math.log(q) for q in p],
                 "gold": None, "encoded_tokens": 123, "prompt_sha256": "a" * 64,
                 "kl": kl, "teacher_argmax": 0, "student_argmax": 0} for i in range(1390)]

    def test_metrics_recomputed_from_the_complete_same_membership(self):
        rows = self.rows()
        metrics, baseline = validation_metrics(rows)
        self.assertAlmostEqual(metrics["mean_kl"], .5 * math.log(4 / 3), places=12)
        self.assertEqual(metrics["teacher_agreement"], 1.)
        self.assertEqual(validation_metrics(list(reversed(rows)), baseline)[0], metrics)

    def test_reject_changed_input_target_arithmetic_and_membership(self):
        rows = self.rows()
        _, baseline = validation_metrics(rows)
        for key, value in (("target", [.25, .75]), ("prompt_sha256", "b" * 64), ("kl", 9.),
                            ("probabilities", [float("nan"), .25]), ("log_probabilities", [0., 0.])):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validation_metrics(changed, baseline)
        for changed in (rows[:-1], [rows[0]] + rows[:-1]):
            with self.assertRaises(ValueError):
                validation_metrics(changed, baseline)

    def test_selection_refuses_a_partial_validation_set_before_export(self):
        with self.assertRaisesRegex(ValueError, "four distinct"):
            main([])


if __name__ == "__main__":
    unittest.main()
