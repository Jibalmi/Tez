"""Reject corrupted public evidence and test option semantics and label exclusion."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

from qwen_public_evaluation import audit_case, grouped_cases, make_row, presented  # noqa: E402
from report_qwen_public import check_rows  # noqa: E402


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "case", "group": "state", "qtype": "choice", "instructions": "Choose the named item.",
                     "state": {"item": "blue"}, "options": [["red", "Red"], ["blue", "Blue"], ["green", "Green"]],
                     "gold": 1, "extra": {"soft": [.1, .8, .1]}, "lang": "en"}
        self.audit = {"prompt_sha256": "frozen", "token_ids_sha256": "tokens", "encoded_tokens": 30,
                      "prompt_characters": 80, "case_sha256": "case", "option_order": [0, 1, 2]}

    def check(self, row, reverse=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            audit = {**self.audit, "option_order": [2, 1, 0] if reverse else [0, 1, 2]}
            return check_rows(path, "massive_en", [self.case], {"case": audit}, reverse)

    def test_reverse_maps_probabilities_and_preserves_original_gold(self):
        row = make_row("massive_en", self.case, [math.log(.6), math.log(.3), math.log(.1)],
                       {**self.audit, "option_order": [2, 1, 0]}, True)
        self.check(row, True)
        self.assertEqual(row["gold"], 1)
        self.assertEqual(row["pred"], 2)
        for actual, expected in zip(row["probabilities"], [.1, .3, .6], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_reject_changed_metadata_or_raw_readout(self):
        row = make_row("massive_en", self.case, [-2., -1., -3.], self.audit)
        self.check(row)
        corruptions = {"prompt_sha256": "changed", "encoded_tokens": 29, "option_order": [2, 1, 0],
                       "extra": {"soft": [1., 0., 0.]}, "gold": 0, "presented_log_probabilities": [-2., float("nan"), -3.],
                       "presented_probabilities": [.2, .7, .1], "logits": [-3., -1., -2.]}
        for key, value in corruptions.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({**row, key: value})

    def test_unknown_or_missing_duplicate_rows_rejected(self):
        row = make_row("massive_en", self.case, [-2., -1., -3.], self.audit)
        for rows in ([], [row, row], [{**row, "id": "other"}]):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rows.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_rows(path, "massive_en", [self.case], {"case": self.audit})

    def test_group_order_and_no_labels_in_predictor_inputs(self):
        cases = [self.case, {**self.case, "id": "second", "group": "other"}, {**self.case, "id": "third"}]
        groups = grouped_cases("typed_decisions", cases)
        self.assertEqual([[c["id"] for c in group] for group in groups], [["case", "third"], ["second"]])
        saved = copy.deepcopy(self.case)
        safe = presented(self.case, True)
        self.assertEqual(set(safe), {"state", "qtype", "instructions", "options"})
        self.assertEqual(safe["options"], self.case["options"][::-1])
        self.assertEqual(self.case, saved)

    def test_empty_or_overlong_inputs_rejected(self):
        class Tokenizer:
            def __init__(self, n):
                self.n = n

            def encode(self, *_args, **_kwargs):
                return [1] * self.n

        for n in (0, 4097):
            with self.subTest(tokens=n), self.assertRaises(ValueError):
                audit_case(Tokenizer(n), self.case)
        self.assertEqual(audit_case(Tokenizer(4096), self.case)["encoded_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
