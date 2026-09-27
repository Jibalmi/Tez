"""Experimental fixed-shape graphs over the unchanged Qwen backbone and letter head."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_plan(lengths, batch_size, buckets):
    """Group complete inputs; leave at least one pad token so native masks stay explicit."""
    if (not lengths or type(batch_size) is not int or batch_size < 1 or not buckets
            or any(type(n) is not int or n < 2 for n in buckets) or list(buckets) != sorted(set(buckets))):
        raise ValueError("Invalid graph shape plan")
    groups = {n: [] for n in buckets}
    for index, size in enumerate(lengths):
        if type(size) is not int or not 0 < size < buckets[-1]:
            raise ValueError("A complete input must fit with at least one padding token")
        groups[next(n for n in buckets if n > size)].append(index)
    plan = []
    for length, indices in groups.items():
        indices.sort(key=lambda i: (lengths[i], i))
        for start in range(0, len(indices), batch_size):
            real = indices[start:start + batch_size]
            plan.append({"length": length, "indices": real, "dummy_rows": batch_size - len(real)})
    return plan


def pack_inputs(token_ids, counts, batch_size, length, pad_id, letters=26):
    """Left-pad without clipping, duplicating only the final row to fill the graph batch."""
    if (type(batch_size) is not int or batch_size < 1 or type(length) is not int or length < 2
            or type(letters) is not int or letters < 2
            or not token_ids or len(token_ids) != len(counts) or len(token_ids) > batch_size
            or type(pad_id) is not int or pad_id < 0
            or any(type(k) is not int or not 2 <= k <= letters for k in counts)
            or any(not row or len(row) >= length or any(type(t) is not int or t < 0 for t in row) for row in token_ids)):
        raise ValueError("Invalid graph input; truncation and implicit masks are forbidden")
    real = len(token_ids)
    expanded = list(token_ids) + [token_ids[-1]] * (batch_size - real)
    expanded_counts = list(counts) + [counts[-1]] * (batch_size - real)
    ids = torch.full((batch_size, length), pad_id, dtype=torch.long)
    padding = torch.zeros_like(ids)
    for i, row in enumerate(expanded):
        ids[i, -len(row):] = torch.tensor(row, dtype=torch.long)
        padding[i, -len(row):] = 1
    allowed = torch.arange(letters)[None, :] < torch.tensor(expanded_counts)[:, None]
    return ids, padding, allowed


class FixedQwenGraph:
    """Own one graph, its static buffers and its private CUDA memory pool.

    Call under inference_mode and BF16 autocast. A new shape needs a new instance;
    callers must account for capture when measuring cold or uncached requests.
    """

    def __init__(self, candidate, batch_size, length, token_ids, counts, guard):
        self.candidate = candidate
        self.batch_size, self.length = batch_size, length
        self.pad_id = candidate.tok.pad_token_id
        self.letters = len(candidate.letters)
        if candidate.model.training or not torch.is_inference_mode_enabled():
            raise ValueError("Capture requires frozen inference mode")
        ids, padding, allowed = pack_inputs(token_ids, counts, batch_size, length, self.pad_id, self.letters)
        self.ids, self.padding, self.allowed = (x.to("cuda") for x in (ids, padding, allowed))
        self.positions = torch.arange(length, device="cuda").view(1, 1, -1).expand(4, batch_size, -1)
        self.masks = self._masks(self.ids, self.padding)
        self.replays = 0
        self.graph = None
        self.output = None
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                torch.cuda.synchronize()
                guard(force=True)
                self._forward()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        guard(force=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._forward()
        torch.cuda.synchronize()

    def _masks(self, ids, padding):
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        kwargs = {"config": self.candidate.lm.config,
                  "inputs_embeds": self.candidate.lm.model.embed_tokens(ids),
                  "attention_mask": padding, "past_key_values": None, "position_ids": self.positions[0]}
        masks = {"full_attention": create_causal_mask(**kwargs),
                 "linear_attention": create_recurrent_attention_mask(**kwargs)}
        if any(not isinstance(value, torch.Tensor) for value in masks.values()):
            raise ValueError("Explicit native masks are required for every graph replay")
        return masks

    def _forward(self):
        hidden = self.candidate.lm.model(input_ids=self.ids, attention_mask=self.masks,
                                         position_ids=self.positions, use_cache=False).last_hidden_state[:, -1:, :]
        logits = F.linear(hidden, self.candidate.weight, self.candidate.bias)[:, -1, :].float()
        return F.log_softmax(torch.where(self.allowed, logits, torch.full_like(logits, -1e4)), -1)

    def run(self, token_ids, counts, guard, check_eager=False):
        if self.graph is None:
            raise ValueError("Cannot replay a closed graph")
        ids, padding, allowed = pack_inputs(token_ids, counts, self.batch_size, self.length, self.pad_id, self.letters)
        ids, padding, allowed = (x.to("cuda") for x in (ids, padding, allowed))
        replacement = self._masks(ids, padding)
        self.ids.copy_(ids)
        self.padding.copy_(padding)
        self.allowed.copy_(allowed)
        for key, value in replacement.items():
            if value.shape != self.masks[key].shape or value.dtype != self.masks[key].dtype:
                raise ValueError("Native mask structure changed")
            self.masks[key].copy_(value)
        uncaptured = None
        if check_eager:
            guard(force=True)
            uncaptured = self._forward().clone()
            torch.cuda.synchronize()
        guard(force=True)
        self.graph.replay()
        self.replays += 1
        # Own the returned storage: a subsequent replay must not overwrite a caller's result.
        output = self.output.clone()
        torch.cuda.synchronize()
        return output[:len(token_ids)], None if uncaptured is None else uncaptured[:len(token_ids)]

    def close(self):
        torch.cuda.synchronize()
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.output = self.masks = None
        self.ids = self.padding = self.allowed = self.positions = None
