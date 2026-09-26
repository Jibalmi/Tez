"""Provider price and lifecycle protections, using stubbed API calls only."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runpod_trial


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = self.root / "ssh.pub"
        self.key.write_text("ssh-ed25519 example-public-key", encoding="utf-8")
        self.args = SimpleNamespace(receipt=self.root / "receipt.json", gpu_id="test-h100",
                                    max_hourly_usd=4.0, minutes=60, public_key=self.key, key_file=None)

    def test_expensive_quote_never_allocates(self):
        with patch.object(runpod_trial, "gpu_catalog", return_value=[{"id": "test-h100", "securePrice": 5}]), \
                patch.object(runpod_trial, "request") as api:
            with self.assertRaisesRegex(ValueError, "ceiling"):
                runpod_trial.launch(self.args, "test-token")
            api.assert_not_called()

    def test_unexpected_allocated_price_stops_only_created_pod(self):
        calls = []
        def api(key, method, path, body=None):
            calls.append((method, path))
            return {"id": "new-pod", "costPerHr": 6.0} if path == "pods" else {}
        with patch.object(runpod_trial, "gpu_catalog", return_value=[{"id": "test-h100", "securePrice": 3.49}]), \
                patch.object(runpod_trial, "request", side_effect=api):
            with self.assertRaisesRegex(ValueError, "price"):
                runpod_trial.launch(self.args, "test-token")
        self.assertEqual(calls, [("POST", "pods"), ("POST", "pods/new-pod/stop")])

    def test_existing_receipt_prevents_duplicate_allocation(self):
        self.args.receipt.write_text("{}", encoding="utf-8")
        with patch.object(runpod_trial, "gpu_catalog") as api:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                runpod_trial.launch(self.args, "test-token")
            api.assert_not_called()

    def test_deadline_watcher_stops_exact_pod(self):
        runpod_trial.save(self.args.receipt, {"pod_id": "owned-pod", "stop_at_unix": 0})
        with patch.object(runpod_trial, "request", return_value={}) as api:
            runpod_trial.watch("test-token", self.args.receipt)
        api.assert_called_once_with("test-token", "POST", "pods/owned-pod/stop")


if __name__ == "__main__":
    unittest.main()
