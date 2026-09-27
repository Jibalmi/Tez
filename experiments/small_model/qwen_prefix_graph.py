"""Preserve each native left-padded batch as an unchanged prefix inside a larger CUDA graph."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from graph_qwen import FixedQwenGraph


def native_batch_plan(cases, batch_size, buckets):
    """Keep original character-sorted batch composition; group whole batches by graph shape."""
    if (not cases or type(batch_size) is not int or batch_size < 1 or not buckets
            or list(buckets) != sorted(set(buckets)) or any(type(n) is not int or n < 2 for n in buckets)):
        raise ValueError("Invalid native batch plan")
    if any(type(c["encoded_tokens"]) is not int or not 0 < c["encoded_tokens"] < buckets[-1] for c in cases):
        raise ValueError("Every complete input must fit with trailing padding")
    order = sorted(range(len(cases)), key=lambda i: len(cases[i]["prompt"]))
    batches = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        native_length = max(cases[i]["encoded_tokens"] for i in indices)
        length = next(n for n in buckets if n > native_length)
        batches.append({"original_batch": start // batch_size, "indices": indices, "batch_size": len(indices),
                        "native_length": native_length, "length": length})
    # No within-batch reordering, filling, duplication or merging. Cache-off batches are independent.
    return sorted(batches, key=lambda row: (-row["batch_size"], row["length"], row["original_batch"]))


def pack_native_prefix(token_ids, counts, batch_size, length, pad_id, letters=26):
    """Keep real tokens at native batch positions and append masked tokens only at the right."""
    if (type(batch_size) is not int or batch_size < 1 or type(length) is not int or length < 2
            or type(letters) is not int or letters < 2 or type(pad_id) is not int or pad_id < 0
            or len(token_ids) != batch_size or len(counts) != batch_size
            or any(type(k) is not int or not 2 <= k <= letters for k in counts)
            or any(not row or len(row) >= length or any(type(t) is not int or t < 0 for t in row) for row in token_ids)):
        raise ValueError("Require complete native batches with supported options and trailing padding")
    native_length = max(map(len, token_ids))
    ids = torch.full((batch_size, length), pad_id, dtype=torch.long)
    padding = torch.zeros_like(ids)
    for index, row in enumerate(token_ids):
        ids[index, native_length - len(row):native_length] = torch.tensor(row, dtype=torch.long)
        padding[index, native_length - len(row):native_length] = 1
    allowed = torch.arange(letters)[None, :] < torch.tensor(counts)[:, None]
    return ids, padding, allowed, torch.tensor([native_length - 1], dtype=torch.long)


def native_final_hidden(hidden, position):
    """Select the native last-token column via a static CUDA index buffer, without a CPU sync."""
    if hidden.ndim != 3 or position.ndim != 1 or position.numel() != 1 or position.dtype != torch.long:
        raise ValueError("Expected one native final position for this left-padded batch")
    return hidden.index_select(1, position)


class NativePrefixGraph(FixedQwenGraph):
    """Use the existing native-mask/lifetime machinery with unchanged prefix positions."""

    def __init__(self, candidate, batch_size, length, token_ids, counts, guard):
        self.candidate = candidate
        self.batch_size, self.length = batch_size, length
        self.pad_id, self.letters = candidate.tok.pad_token_id, len(candidate.letters)
        if candidate.model.training or not torch.is_inference_mode_enabled():
            raise ValueError("Capture requires frozen inference mode")
        packed = pack_native_prefix(token_ids, counts, batch_size, length, self.pad_id, self.letters)
        self.ids, self.padding, self.allowed, self.end = (tensor.to("cuda") for tensor in packed)
        self.positions = torch.arange(length, device="cuda").view(1, 1, -1).expand(4, batch_size, -1)
        self.masks = self._masks(self.ids, self.padding)
        self.replays = 0
        self.graph = self.output = None
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

    def _forward(self):
        output = self.candidate.lm.model(input_ids=self.ids, attention_mask=self.masks,
                                        position_ids=self.positions, use_cache=False).last_hidden_state
        hidden = native_final_hidden(output, self.end)
        logits = F.linear(hidden, self.candidate.weight, self.candidate.bias)[:, 0, :].float()
        return F.log_softmax(torch.where(self.allowed, logits, torch.full_like(logits, -1e4)), -1)

    def run(self, token_ids, counts, guard, check_eager=False):
        if self.graph is None:
            raise ValueError("Cannot replay a closed graph")
        packed = pack_native_prefix(token_ids, counts, self.batch_size, self.length, self.pad_id, self.letters)
        ids, padding, allowed, end = (tensor.to("cuda") for tensor in packed)
        replacement = self._masks(ids, padding)
        self.ids.copy_(ids)
        self.padding.copy_(padding)
        self.allowed.copy_(allowed)
        self.end.copy_(end)
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
        output = self.output.clone()
        torch.cuda.synchronize()
        return output, uncaptured

    def close(self):
        super().close()
        self.end = None
