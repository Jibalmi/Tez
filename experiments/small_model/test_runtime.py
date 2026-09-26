"""Offline recovery checks: python -m unittest discover -s experiments/small_model -p test_runtime.py."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import common
import teacher


class ThermalTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        common._THERMAL_LAST[0] = 0.0
        common.THERMAL.update(checks=0, pauses=0, paused_s=0.0, max_c=0)
        self.addCleanup(patch.stopall)
        patch("time.monotonic", side_effect=lambda: self.clock).start()
        patch("time.sleep", side_effect=self.advance).start()

    def advance(self, seconds):
        self.clock += seconds

    def test_cool_gpu_and_debounce(self):
        with patch.object(common, "gpu_temp", return_value=83) as read:
            self.assertEqual(common.thermal_guard(), 0)
            self.assertEqual(common.thermal_guard(), 0)
            self.assertEqual(read.call_count, 1)

    def test_waits_for_cool_target_and_records_peak_during_pause(self):
        with patch.object(common, "gpu_temp", side_effect=[86, 87, 85, 84]):
            self.assertEqual(common.thermal_guard(), 3)
        self.assertEqual(common.THERMAL["max_c"], 87)
        self.assertEqual(common.THERMAL["paused_s"], 3)

    def test_timeout_never_grants_permission_to_resume_hot(self):
        with patch.object(common, "gpu_temp", return_value=88):
            with self.assertRaises(common.ThermalCooldownTimeout):
                common.thermal_guard(max_s=5)
        self.assertEqual(self.clock, 105)
        self.assertEqual(common.THERMAL["paused_s"], 5)

    def test_missing_sensor_stops_work(self):
        for readings in ([None], [86, None]):
            with self.subTest(readings=readings), patch.object(common, "gpu_temp", side_effect=readings):
                with self.assertRaises(common.ThermalCooldownTimeout):
                    common.thermal_guard(force=True)

    def test_force_checks_again_even_immediately_after_safe_read(self):
        with patch.object(common, "gpu_temp", side_effect=[83, 89]):
            common.thermal_guard()
            with self.assertRaises(common.ThermalCooldownTimeout):
                common.thermal_guard(force=True, max_s=0)

    def test_training_saves_before_returning_resumable_exit(self):
        saved = []
        with patch.object(common, "thermal_guard", side_effect=common.ThermalCooldownTimeout("hot")):
            with self.assertRaises(SystemExit) as result:
                common.thermal_guard_or_checkpoint(lambda: saved.append("checkpoint"))
        self.assertEqual(result.exception.code, 3)
        self.assertEqual(saved, ["checkpoint"])

    def test_checkpoint_failure_is_not_reported_as_resumable(self):
        def fail():
            raise OSError("disk write failed")

        with patch.object(common, "thermal_guard", side_effect=common.ThermalCooldownTimeout("hot")):
            with self.assertRaises(OSError):
                common.thermal_guard_or_checkpoint(fail)


class TeacherRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "teacher"
        self.out.mkdir()
        self.results = self.root / "results"
        self.addCleanup(patch.stopall)
        patch.object(teacher, "OUT", self.out).start()
        patch.object(teacher, "RUNS", self.root).start()
        patch.object(teacher, "RESULTS", self.results).start()

    def test_interrupted_publish_is_not_a_visible_checkpoint(self):
        with patch.object(teacher.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                teacher.save_part(0, [{"id": "one"}])
        self.assertEqual(list(self.out.glob("part-*.jsonl")), [])
        teacher.save_part(0, [{"id": "one"}])
        self.assertEqual(common.read_jsonl(self.out / "part-0000.jsonl"), [{"id": "one"}])
        self.assertFalse((self.out / "part-0000.jsonl.tmp").exists())

    def test_cooling_stop_flushes_closes_and_resumes_without_repeating_states(self):
        from tez import inproc

        rows = [{"id": name, "state": {"text": name}} for name in ("one", "two")]
        common.write_jsonl(self.root / "pool.jsonl", rows)
        common.write_jsonl(self.root / "assign.jsonl", [
            {"id": row["id"], "questions": [{"qid": "q", "type": "noul", "instructions": "Is this true?"}]}
            for row in rows
        ])
        instances = []

        class Backend:
            def __init__(self, *args, **kwargs):
                self.calls = 0
                self.closed = False
                instances.append(self)

            def load(self):
                pass

            def close(self):
                self.closed = True

            def read_many(self, prompts, ks):
                self.calls += 1
                return [(SimpleNamespace(logits=[-0.2, -1.7], tokens=8, timings={"batch_prompt_n": 8}),)
                        for _ in prompts]

        patch.object(inproc, "InprocBackend", Backend).start()
        patch.object(inproc, "find_library", return_value="unused").start()
        patch.object(inproc, "load_api", return_value=SimpleNamespace(
            llama_model_default_params=lambda: SimpleNamespace())).start()
        patch("sys.argv", ["teacher.py", "--minutes", "1"]).start()
        with patch.object(teacher, "thermal_guard", side_effect=[0, common.ThermalCooldownTimeout("still hot")]):
            self.assertEqual(teacher.main(), 3)
        self.assertTrue(instances[0].closed)
        self.assertEqual(instances[0].calls, 1)
        self.assertEqual(common.read_jsonl(self.out / "part-0000.jsonl")[0]["id"], "one")
        with patch.object(teacher, "thermal_guard", return_value=0):
            self.assertEqual(teacher.main(), 0)
        self.assertTrue(instances[1].closed)
        self.assertEqual(instances[1].calls, 1)
        self.assertEqual(common.read_jsonl(self.out / "part-0001.jsonl")[0]["id"], "two")
        stats = json.loads((self.results / "teacher_timing.json").read_text(encoding="utf-8"))
        self.assertEqual([run["states"] for run in stats["runs"]], [1, 1])
        self.assertEqual(stats["runs"][0]["stopped_for"], "still hot")


if __name__ == "__main__":
    unittest.main()
