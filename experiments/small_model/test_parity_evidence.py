"""Contract checks for numerical-parity evidence and actual dispatch accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from parity_evidence import compare_rows, read_rows
from windows_kernel_parity import track_dispatch


class ParityEvidenceTest(unittest.TestCase):
    def test_probability_membership_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {"state_id": "s", "qid": "q", "k": 2, "probabilities": [0.4, 0.6]}
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(read_rows(path, [("s", "q", 2)]), {("s", "q", 2): [0.4, 0.6]})
            with self.assertRaises(ValueError):
                read_rows(path, [("s", "different", 2)])
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(path)
            for bad in ([0.4, 0.7], [-0.1, 1.1], [float("nan"), 0.6], [True, 0]):
                path.write_text(json.dumps({**row, "probabilities": bad}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_rows(path)

    def test_argmax_alone_is_insufficient(self):
        spec = {"argmax_agreement_min": 0.99, "max_probability_delta_max": 0.04, "mean_total_variation_max": 0.005}
        reference = {("s", "q", 2): [0.4, 0.6]}
        changed = {("s", "q", 2): [0.41, 0.59]}
        result = compare_rows(reference, changed, spec)
        self.assertEqual(result["argmax_agreement"], 1)
        self.assertAlmostEqual(result["mean_total_variation"], 0.01)
        self.assertFalse(result["passed"])
        self.assertTrue(compare_rows(reference, reference, spec)["passed"])

    def test_dispatch_counts_selected_implementation(self):
        def implementation(value):
            return value
        implementation.__module__ = "fla.ops.gated_delta_rule.test"
        is_new_implementation = True

        def wrapper(value):
            if is_new_implementation:
                return implementation(value)
            return None

        record = track_dispatch(wrapper, "fla.ops.gated_delta_rule")
        tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype="bf16", shape=(1, 2))
        self.assertIs(wrapper(tensor), tensor)
        self.assertEqual(record["calls"], 1)
        self.assertEqual(record["first_tensors"][0]["shape"], [1, 2])

    def test_dispatch_rejects_reference_fallback(self):
        def implementation(value):
            return value
        is_new_implementation = False

        def wrapper(value):
            return implementation(value) if is_new_implementation else value

        with self.assertRaises(ValueError):
            track_dispatch(wrapper, "fla.ops.gated_delta_rule")


if __name__ == "__main__":
    unittest.main()
