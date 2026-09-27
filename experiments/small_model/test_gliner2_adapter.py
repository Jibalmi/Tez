"""Input-preservation and native probability contracts without importing GLiNER or a model."""
import math
from types import SimpleNamespace
import unittest

from gliner2_adapter import audit_native_batch, native_probabilities


class NativeAdapterTest(unittest.TestCase):
    def test_full_vector_must_match_native_answer(self):
        values = native_probabilities([0, math.log(3)], ["no", "yes"], {"label": "yes", "confidence": .75})
        self.assertAlmostEqual(values[0], .25)
        self.assertAlmostEqual(values[1], .75)
        for answer in ({"label": "no", "confidence": .75}, {"label": "yes", "confidence": .9},
                       {"label": "yes", "confidence": float("nan")}):
            with self.assertRaises(ValueError):
                native_probabilities([0, math.log(3)], ["no", "yes"], answer)

    def test_reject_silent_truncation_and_label_reordering(self):
        inputs = [{"text": "One two", "instructions": "Which (one)?", "options": {"a": "First", "b": "Second"}}]
        tokens = ["(", "[P]", "decision: Which (one)? [DESCRIPTION] a: First [DESCRIPTION] b: Second",
                  "(", "[L]", "a", "[L]", "b", ")", ")"]
        batch = SimpleNamespace(original_texts=["One two"], text_tokens=[["one", "two"]],
                                task_types=[["classifications"]], schema_tokens_list=[[tokens]])

        def splitter(text, lower=True):
            return [(word.lower() if lower else word, 0, 1) for word in text.split()]

        audit_native_batch(batch, inputs, splitter)
        batch.text_tokens = [["one"]]
        with self.assertRaises(ValueError):
            audit_native_batch(batch, inputs, splitter)
        batch.text_tokens = [["one", "two"]]
        tokens[5], tokens[7] = tokens[7], tokens[5]
        with self.assertRaises(ValueError):
            audit_native_batch(batch, inputs, splitter)


if __name__ == "__main__":
    unittest.main()
