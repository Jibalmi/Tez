"""Report decisions on synthetic fixtures; no models, GPU work or benchmark-number assertions."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report


def evaluation(retention=0.95, accuracy=0.95):
    tasks = {
        name: {"t1": {"accuracy": accuracy, "ece": 0.01}, "cal": {"ece": 0.01},
               "retention_vs_12b": retention, "order_flip_20": 0.0}
        for name in report.G1_TASKS
    }
    tasks["typed_decisions"]["t1"]["soft_accuracy"] = 0.9
    return {"tasks": tasks}


def timing(arm):
    return {"arm": arm, "device": "cuda", "throttled": False, "contaminated": False,
            "sizes": {str(n): {"p50_ms": n * 0.5, "ms_per_question": 0.5} for n in (1, 5, 10, 50)}}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        self.patch = patch.object(report, "RESULTS", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.files = {
            "eval_a-base-30k.json": evaluation(),
            "eval_a-base-10k.json": evaluation(retention=0.8),
            "eval_c-qwen08-30k.json": evaluation(retention=0.9),
            "latency.json": {f"{arm}_cuda": timing(arm) for arm in ("a-base-30k", "c-qwen08-30k")},
        }

    def run_report(self):
        for name, data in self.files.items():
            (self.root / name).write_text(json.dumps(data), encoding="utf-8")
        with contextlib.redirect_stdout(self.output):
            report.main()
        return json.loads((self.root / "pilot.json").read_text(encoding="utf-8"))

    def test_no_results_remains_pending(self):
        self.files = {}
        result = self.run_report()
        self.assertFalse(result["g1_complete"])
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])

    def test_complete_clean_results_can_choose_a(self):
        result = self.run_report()
        self.assertTrue(result["g1_complete"])
        self.assertTrue(result["g1_all_pass"])
        self.assertEqual(result["decision"], "continue with A (go to M2)")

    def test_missing_learning_curve_is_pending_not_pass(self):
        del self.files["eval_a-base-10k.json"]
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])
        self.assertIn("PENDING", self.output.getvalue())

    def test_missing_language_cannot_be_hidden_in_macro_average(self):
        del self.files["eval_a-base-30k.json"]["tasks"][f"massive_{report.MASSIVE11[-1]}"]
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])
        self.assertTrue(result["arms"]["a-base-30k"]["missing_measurements"])

    def test_missing_latency_is_pending_not_pass(self):
        del self.files["latency.json"]
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])

    def test_throttled_a_timing_cannot_pass_speed_gate(self):
        self.files["latency.json"]["a-base-30k_cuda"]["throttled"] = True
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])
        self.assertIn("throttled", (self.root / "tables.md").read_text(encoding="utf-8"))

    def test_contaminated_c_cannot_be_selected(self):
        self.files["eval_c-qwen08-30k.json"] = evaluation(retention=1.1)
        self.files["latency.json"]["c-qwen08-30k_cuda"]["contaminated"] = True
        result = self.run_report()
        self.assertIsNone(result["decision"])
        self.assertIn("contaminated", (self.root / "tables.md").read_text(encoding="utf-8"))

    def test_unknown_thermal_status_is_not_a_clean_timing(self):
        self.files["latency.json"]["a-base-30k_cuda"]["throttled"] = None
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIn("throttled status unknown", (self.root / "tables.md").read_text(encoding="utf-8"))

    def test_missing_c_cannot_trigger_stop_rule(self):
        self.files["eval_a-base-30k.json"] = evaluation(accuracy=0.1)
        del self.files["eval_c-qwen08-30k.json"]
        result = self.run_report()
        self.assertIsNone(result["decision"])

    def test_measured_failures_can_trigger_stop_rule(self):
        self.files["eval_a-base-30k.json"] = evaluation(accuracy=0.1)
        self.files["eval_c-qwen08-30k.json"] = evaluation(accuracy=0.1)
        result = self.run_report()
        self.assertFalse(result["g1_all_pass"])
        self.assertEqual(result["decision"], "stop the zero-shot student; build per-schema students (D3)")

    def test_better_clean_c_can_win_even_if_a_passes(self):
        self.files["eval_c-qwen08-30k.json"] = evaluation(retention=1.1)
        result = self.run_report()
        self.assertTrue(result["g1_all_pass"])
        self.assertEqual(result["decision"], "switch to C")

    def test_nonfinite_metric_is_not_complete_evaluation(self):
        self.files["eval_a-base-30k.json"]["tasks"]["typed_decisions"]["t1"]["soft_accuracy"] = float("nan")
        result = self.run_report()
        self.assertIsNone(result["g1_all_pass"])
        self.assertIsNone(result["decision"])

    def test_missing_timing_size_is_pending_and_table_still_renders(self):
        del self.files["latency.json"]["a-base-30k_cuda"]["sizes"]["50"]
        result = self.run_report()
        self.assertIsNone(result["decision"])
        self.assertTrue((self.root / "tables.md").exists())


if __name__ == "__main__":
    unittest.main()
