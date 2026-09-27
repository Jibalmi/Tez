"""A clean timing needs known thermal telemetry and case-independent memory labels."""
import unittest

from latency import contamination, throttling


class TimingConditionsTests(unittest.TestCase):
    def test_windows_lowercase_and_linux_memory_labels(self):
        for kind in ("dedicated", "Dedicated compute"):
            snapshot = {"processes": [{"pid": 20, "mb": 350, "kind": kind}]}
            self.assertTrue(contamination(snapshot, 10))
            self.assertFalse(contamination(snapshot, 20))
        self.assertIsNone(contamination({"error": "failed"}, 10))

    def test_unknown_telemetry_is_not_a_clean_run(self):
        clean = {"temp_c": 60, "sw_thermal": False, "hw_thermal": False}
        self.assertFalse(throttling([clean]))
        self.assertIsNone(throttling([clean, {"error": "failed"}]))
        self.assertIsNone(throttling([]))
        self.assertTrue(throttling([{**clean, "temp_c": 75}]))


if __name__ == "__main__":
    unittest.main()
