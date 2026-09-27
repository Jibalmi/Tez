"""Counter corruption and other adapters must not make an ineligible run appear clean."""
import copy
import unittest

from windows_gpu_telemetry import MEMORY_LIMIT, selected_memory, thermal_eligible


def pair(pid=None, luid="0x00000000_0x00001234", node=0, dedicated=0, shared=0):
    instance = (f"pid_{pid}_" if pid is not None else "") + f"luid_{luid}_phys_{node}"
    object_name = "gpu process memory" if pid is not None else "gpu adapter memory"
    return [{"instance": instance, "path": f"\\\\fixture\\{object_name}({instance})\\{kind} usage",
             "object": object_name, "counter": kind + " usage",
             "bytes": value, "status": 0, "name": "fixture" if pid else None}
            for kind, value in (("dedicated", dedicated), ("shared", shared))]


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"luid": "0x00000000_0x00001234", "physical_node": 0}

    def test_other_adapter_or_node_and_own_pid_are_separate(self):
        rows = pair(dedicated=9 * MEMORY_LIMIT) + pair(10, dedicated=8 * MEMORY_LIMIT)
        rows += pair(20, luid="0x00000000_0x00005678", dedicated=7 * MEMORY_LIMIT)
        rows += pair(30, node=1, dedicated=6 * MEMORY_LIMIT) + pair(40, dedicated=MEMORY_LIMIT)
        result = selected_memory({"samples": rows}, self.identity, 10)
        self.assertFalse(result["contaminated"])
        self.assertEqual([p["pid"] for p in result["processes"]], [10, 40])
        rows += pair(50, dedicated=MEMORY_LIMIT + 1)
        result = selected_memory({"samples": rows}, self.identity, 10)
        self.assertTrue(result["contaminated"])
        self.assertEqual([p["pid"] for p in result["foreign_over_limit"]], [50])

    def test_do_not_sum_shared_across_process_allocations(self):
        result = selected_memory({"samples": pair(dedicated=MEMORY_LIMIT) +
                                 pair(20, dedicated=MEMORY_LIMIT) + pair(30, dedicated=MEMORY_LIMIT)}, self.identity, 10)
        self.assertEqual(result["adapter"]["dedicated_bytes"], MEMORY_LIMIT)
        self.assertFalse(result["contaminated"])

    def test_unavailable_target_adapter_or_half_process_reading_rejected(self):
        for rows in ([], pair(10), pair() + pair(20)[:1], pair(luid="0x00000000_0x00005678")):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                selected_memory({"samples": rows}, self.identity, 10)

    def test_corrupted_or_duplicate_counters_rejected(self):
        rows = pair() + pair(20)
        for field, value in (("status", 3221228488), ("status", False), ("bytes", float("nan")), ("bytes", -1),
                             ("bytes", True), ("bytes", .5), ("instance", "unknown"), ("object", "gpu adapter memory")):
            changed = copy.deepcopy(rows)
            changed[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                selected_memory({"samples": changed}, self.identity, 10)
        with self.assertRaises(ValueError):
            selected_memory({"samples": rows + rows[:1]}, self.identity, 10)

    def test_indexed_other_adapter_instances_are_retained_but_never_summed(self):
        other = pair(20, luid="0x00000000_0x00005678", shared=123)
        indexed = copy.deepcopy(other)
        for row in indexed:
            row["path"] = row["path"].replace(")", "#1)")
        result = selected_memory({"samples": pair() + other + indexed}, self.identity, 10)
        self.assertFalse(result["contaminated"])
        self.assertEqual(result["processes"], [])
        target = pair(20)
        for row in target:
            row["path"] = row["path"].replace(")", "#1)")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            selected_memory({"samples": pair() + target}, self.identity, 10)

    def test_thermal_sensors_are_required_and_limits_unchanged(self):
        good = {"temp_c": 70, "sw_thermal": False, "hw_thermal": False}
        self.assertTrue(thermal_eligible(good))
        for update in ({"temp_c": 71}, {"sw_thermal": True}, {"hw_thermal": True}):
            self.assertFalse(thermal_eligible({**good, **update}))
        for bad in ({}, {**good, "temp_c": "60"}, {**good, "sw_thermal": None}, {**good, "temp_c": -1}):
            with self.assertRaises(ValueError):
                thermal_eligible(bad)


if __name__ == "__main__":
    unittest.main()
