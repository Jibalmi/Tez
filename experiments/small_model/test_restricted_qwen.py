"""Restricted readout must preserve masked backbone inputs, head bias, and option ordering."""
from types import SimpleNamespace
import unittest

import torch

from restricted_qwen import RestrictedQwen


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, input_ids, attention_mask):
        self.seen = (input_ids.clone(), attention_mask.clone())
        # Include a distinct model transformation so a direct embedding-only read cannot pass.
        values = input_ids.float() * attention_mask + 7
        hidden = torch.stack((values, values.square(), values * -2), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


class Tokenizer:
    def __call__(self, prompts, **kwargs):
        if kwargs["padding_side"] != "left" or kwargs["truncation"]:
            raise AssertionError("Wrong padding or truncation contract")
        return {"input_ids": torch.tensor([[0, 2, 3], [4, 5, 6]]), "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}


class RestrictedProjectionTests(unittest.TestCase):
    def reference(self):
        head = torch.nn.Linear(3, 7, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.arange(21).reshape(7, 3) / 100)
            head.bias.copy_(torch.arange(7) / 10)
        body = Body()
        lm = SimpleNamespace(model=body, config=SimpleNamespace(model_type="qwen3_5_text"), get_output_embeddings=lambda: head)
        return SimpleNamespace(model=SimpleNamespace(training=False, get_base_model=lambda: lm), tok=Tokenizer(),
                               device="cpu", letter_ids=torch.tensor([5, 1, 3]), weights={}, info={}, batch_size=16), head, body

    def test_matches_full_head_after_backbone_and_option_mask(self):
        reference, head, body = self.reference()
        candidate = RestrictedQwen(reference)
        actual = candidate.logp(["short", "long"], [2, 3])
        ids, mask = body.seen
        full = head(body(ids, mask).last_hidden_state)[:, -1, :][:, reference.letter_ids]
        full[0, 2] = -1e4
        torch.testing.assert_close(actual, torch.log_softmax(full, -1), atol=1e-5, rtol=1e-5)
        self.assertEqual(mask[0].tolist(), [0, 1, 1])
        self.assertLess(actual[0, 2].item(), -1000)

    def test_rejects_head_wrappers_and_duplicate_letter_mapping(self):
        reference, head, _ = self.reference()
        reference.letter_ids = torch.tensor([1, 1])
        with self.assertRaises(ValueError):
            RestrictedQwen(reference)
        reference, _, _ = self.reference()
        reference.model.get_base_model().get_output_embeddings = lambda: torch.nn.Sequential(head)
        with self.assertRaises(ValueError):
            RestrictedQwen(reference)


if __name__ == "__main__":
    unittest.main()
