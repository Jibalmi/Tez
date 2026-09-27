"""A time allowance must not silently change numerical acceptance or inference inputs."""
import copy
import unittest

from qwen_prefix_single_validation import extended_spec


class ResourceAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.parent = {"batch_sizes": [1, 16], "plans": {"1": list(range(1390)), "16": ["batches"]},
                       "soft_deadline_seconds": 720, "worker_timeout_seconds": 840,
                       "argmax_agreement_min": .99, "max_probability_delta_max": .04, "mean_total_variation_max": .005,
                       "source": {"frozen": "hash"}, "inputs": {"original": "hash"}}
        self.verified = {"status": "complete", "passed": True, "batch_size": 16, "real_predictions_per_arm": 1390}
        self.failed = {"status": "failed", "error_type": "TimeoutError", "batch_size": 1,
                       "batches": [None] * 1348, "elapsed_seconds": 720.25}

    def test_only_unexecuted_arm_and_time_allocation_change(self):
        original = copy.deepcopy(self.parent)
        result = extended_spec(self.parent, self.verified, self.failed)
        self.assertEqual(self.parent, original)
        self.assertEqual({k for k in result if result[k] != original[k]},
                         {"batch_sizes", "plans", "soft_deadline_seconds", "worker_timeout_seconds"})
        self.assertEqual(result["plans"]["1"], original["plans"]["1"])
        result["source"]["frozen"] = "mutated-copy"
        self.assertEqual(self.parent["source"]["frozen"], "hash")

    def test_require_complete_passing_batch16_prefix_result(self):
        for field, value in (("status", "running"), ("passed", False), ("batch_size", 1), ("real_predictions_per_arm", 1389)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                extended_spec(self.parent, {**self.verified, field: value}, self.failed)

    def test_require_actual_incomplete_terminal_deadline(self):
        for field, value in (("status", "running"), ("error_type", "MemoryError"), ("batch_size", 16),
                             ("batches", []), ("batches", [None] * 1390), ("elapsed_seconds", 700)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                extended_spec(self.parent, self.verified, {**self.failed, field: value})


if __name__ == "__main__":
    unittest.main()
