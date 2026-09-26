"""Cloud snapshot and target-parity checks without any GPU or provider API calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cloud_teacher
from cloud_teacher import collect, parity


class CloudTeacherTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"one": {"id": "one", "questions": [{"qid": "q", "type": "noul",
            "keys": ["false", "true"], "abstain": False, "z_orig": [-0.1, -3.0], "z_rev": [-3.0, -0.1]}]}}

    def test_same_targets_pass(self):
        result = parity(self.reference, copy.deepcopy(self.reference))
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_probability_delta"], 0)

    def test_changed_option_order_is_rejected(self):
        candidate = copy.deepcopy(self.reference)
        candidate["one"]["questions"][0]["keys"].reverse()
        with self.assertRaisesRegex(ValueError, "metadata"):
            parity(self.reference, candidate)

    def test_missing_states_and_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            parity(self.reference, {})
        candidate = copy.deepcopy(self.reference)
        candidate["one"]["questions"][0]["z_orig"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid logits"):
            parity(self.reference, candidate)

    def test_materially_changed_distribution_fails(self):
        candidate = copy.deepcopy(self.reference)
        q = candidate["one"]["questions"][0]
        q["z_orig"], q["z_rev"] = [-10, 0], [0, -10]
        self.assertFalse(parity(self.reference, candidate)["passed"])

    def test_duplicate_worker_outputs_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            for worker in ("worker-0", "worker-1"):
                directory = Path(folder) / worker / "teacher"
                directory.mkdir(parents=True)
                (directory / "part-0000.jsonl").write_text(json.dumps(self.reference["one"]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect(Path(folder))

    def _run_protocol(self, root, rebuild, unstable=False):
        source = root / "runs/cloud-input"
        source.mkdir(parents=True)
        (source / "canary-reference.jsonl").write_text(json.dumps(self.reference["one"]) + "\n", encoding="utf-8")
        (source / "canary.ids").write_text("one\n", encoding="utf-8")
        (source / "pending.ids").write_text("two\n", encoding="utf-8")
        pilot = root / "runs/pilot"
        pilot.mkdir()
        (pilot / "assign.jsonl").write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
        candidate = copy.deepcopy(self.reference)
        q = candidate["one"]["questions"][0]
        q["z_orig"], q["z_rev"] = [-10, 0], [0, -10]
        calls = []

        def group(base, ids, workers, minutes, deadline, *, fresh_state_cache=False):
            base.mkdir(parents=True)
            selected = ids.read_text(encoding="utf-8").splitlines()
            calls.append((base.name, selected, fresh_state_cache))
            return {"workers": workers, "states": len(selected), "complete": True, "states_per_second": 1.0}

        def reads(base):
            return self.reference if unstable and base.name == "canary-2" else candidate

        args = SimpleNamespace(name="trial", workers=[1, 2], minutes=5, production=True, rebuild_all=rebuild)
        with patch.object(cloud_teacher, "ROOT", root), \
                patch.object(cloud_teacher, "os", SimpleNamespace(name="posix", replace=cloud_teacher.os.replace)), \
                patch.object(cloud_teacher, "verify_inputs", return_value={"pending_states": 1}), \
                patch.object(cloud_teacher.subprocess, "check_output", side_effect=["GPU info", "143771\n"]), \
                patch.object(cloud_teacher, "run_group", side_effect=group), \
                patch.object(cloud_teacher, "collect", side_effect=reads), patch("builtins.print"):
            error = None
            try:
                cloud_teacher.run(args)
            except ValueError as exc:
                error = exc
        report = json.loads((root / "runs/cloud-output/trial/report.json").read_text())
        return report, calls, error

    def test_rebuild_preserves_failed_local_comparison_and_covers_all_states(self):
        with tempfile.TemporaryDirectory() as folder:
            report, calls, error = self._run_protocol(Path(folder), True)
            self.assertIsNone(error)
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["trials"][0]["local_reference_comparison"]["passed"])
            self.assertTrue(report["trials"][1]["parity"]["passed"])
            self.assertEqual(calls[-1][1], ["one", "two"])
            self.assertTrue(all(call[2] for call in calls))
            original = (Path(folder) / "runs/cloud-input/canary-reference.jsonl").read_text()
            self.assertEqual(json.loads(original), self.reference["one"])

    def test_resume_and_unstable_rebuild_both_block_production(self):
        for rebuild, unstable in ((False, False), (True, True)):
            with self.subTest(rebuild=rebuild), tempfile.TemporaryDirectory() as folder:
                report, calls, error = self._run_protocol(Path(folder), rebuild, unstable)
                self.assertIsNotNone(error)
                self.assertEqual(report["status"], "parity_failed")
                self.assertFalse(any(call[0] == "production" for call in calls))


if __name__ == "__main__":
    unittest.main()
