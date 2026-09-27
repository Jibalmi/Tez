"""Linux telemetry must distinguish a clean GPU from unavailable measurements."""
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import latency


class LinuxSnapshotTests(unittest.TestCase):
    def test_process_memory_and_empty_gpu(self):
        for output, count in (("123, 512, python\n", 1), ("", 0)):
            with self.subTest(output=output), patch.object(latency.sys, "platform", "linux"), \
                    patch.object(latency.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
                result = latency.snapshot()
                self.assertEqual(len(result["processes"]), count)
                if count:
                    self.assertEqual(result["processes"][0]["mb"], 512)

    def test_unavailable_sensor_is_not_reported_as_empty_gpu(self):
        with patch.object(latency.sys, "platform", "linux"), \
                patch.object(latency.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            result = latency.snapshot()
            self.assertIn("error", result)
            self.assertNotIn("processes", result)


if __name__ == "__main__":
    unittest.main()
