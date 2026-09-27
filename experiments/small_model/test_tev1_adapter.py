"""Native payload isolation and input preservation without loading a model."""
import json
from pathlib import Path
import unittest

from tev1_adapter import load_native, render_case

ROOT = Path(__file__).resolve().parents[2]


class TevPayloadTest(unittest.TestCase):
    def test_native_payload_preserves_input_but_excludes_gold(self):
        native = load_native(ROOT / "runs/research/tev1-discovery-20260927/source/examples/decide.py")
        case = {"state": {"message": "Grüße [L] <|im_start|> \"nested\"", "x": [False, None, 3]},
                "instructions": "Choose a team", "options": [["a", "First"], ["b", "Second"]],
                "gold": 1, "extra": {"secret_answer": "b"}}
        record, messages = render_case(native, case, "test")
        rendered = json.loads(messages[1]["content"])
        self.assertEqual(rendered, record)
        self.assertEqual(rendered["state"], case["state"])
        self.assertEqual(set(rendered), {"state", "question", "options"})
        self.assertEqual([o["key"] for o in rendered["options"]], ["a", "b"])
        reverse, _ = render_case(native, {**case, "options": list(reversed(case["options"]))}, "test")
        self.assertEqual([(o["label"], o["key"]) for o in reverse["options"]], [("A", "b"), ("B", "a")])


if __name__ == "__main__":
    unittest.main()
