"""Keep generic probability checks and add native batch/position provenance checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_verify_qwen_graph_validation as original
from verify_qwen_prefix_validation import verify_rows


class PrefixRowsTests(original.VerifyGraphRowsTests):
    def setUp(self):
        super().setUp()
        self.plan[0].pop("dummy_rows")
        self.plan[0].update(batch_size=1, native_length=3, original_batch=7)
        self.row.update(native_length=3, original_batch=7)

    def verify(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return verify_rows(path, [self.case], self.plan)

    def test_reject_changed_native_endpoint_or_batch(self):
        for key, value in (("original_batch", 8), ("native_length", 4)):
            row = copy.deepcopy(self.row)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify([row])


if __name__ == "__main__":
    unittest.main()
