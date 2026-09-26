"""A cloud export must be complete, aligned with its assignments, and independently reproducible."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_teacher import MODEL_SHA, sha
from common import read_jsonl, write_jsonl
import export_cloud_teacher as exporter


class ExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source = self.root / "cloud"
        self.output = self.root / "exported"
        question = {"qid": "q", "type": "noul", "instructions": "Does the state match?"}
        write_jsonl(self.inputs / "assign.jsonl", [{"id": s, "questions": [question]} for s in ("one", "two")])
        write_jsonl(self.inputs / "pool.jsonl", [{"id": s, "state": s} for s in ("one", "two")])
        self.rows = [{"id": s, "questions": [{"qid": "q", "type": "noul", "keys": ["false", "true"],
                      "abstain": False, "z_orig": [-3, -0.1], "z_rev": [-0.1, -3]}]} for s in ("one", "two")]
        self.production = self.source / "production/worker-0/teacher/part-0000.jsonl"
        self.production.parent.mkdir(parents=True)
        write_jsonl(self.production, self.rows)
        reference = self.source / "canary-1/worker-0/teacher/part-0000.jsonl"
        reference.parent.mkdir(parents=True)
        write_jsonl(reference, self.rows[:1])
        report = {"status": "complete", "rebuild_all": True, "production": {"fresh_state_cache": True},
                  "manifest": {"model_sha256": MODEL_SHA, "files": {
                      "runs/pilot/" + name: sha(self.inputs / name) for name in ("pool.jsonl", "assign.jsonl")}},
                  "trials": [{"local_reference_comparison": {"passed": False}}]}
        (self.source / "report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_complete_export_keeps_provenance_and_normalized_probabilities(self):
        with patch.object(exporter, "RUNS", self.inputs):
            manifest = exporter.export(self.source, self.output)
        self.assertEqual((manifest["states"], manifest["decisions"]), (2, 2))
        self.assertFalse(manifest["original_laptop_comparison"]["passed"])
        self.assertTrue(manifest["production_reference_consistency"]["passed"])
        for target in read_jsonl(self.output / "probability-targets.jsonl"):
            self.assertEqual(target["keys"], ["false", "true"])
            self.assertAlmostEqual(sum(target["probabilities"]), 1)
            self.assertGreater(target["probabilities"][1], target["probabilities"][0])

    def test_incomplete_production_cannot_create_an_export(self):
        write_jsonl(self.production, self.rows[:1])
        with patch.object(exporter, "RUNS", self.inputs), self.assertRaisesRegex(ValueError, "states"):
            exporter.export(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_question_order_cannot_create_an_export(self):
        self.rows[1]["questions"][0]["keys"].reverse()
        write_jsonl(self.production, self.rows)
        with patch.object(exporter, "RUNS", self.inputs), self.assertRaisesRegex(ValueError, "metadata"):
            exporter.export(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
