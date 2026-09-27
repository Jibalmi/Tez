"""Experimental graph cache for complete fresh-state requests; never changes native batch membership."""
from __future__ import annotations

import time


def request_batches(cases, buckets):
    from qwen_prefix_graph import native_batch_plan
    # Preserve native execution order as well as membership. Independent pools keep each shape alive.
    return sorted(native_batch_plan(cases, 16, buckets), key=lambda batch: batch["original_batch"])


def restore_request(batches, values, size):
    """Invert native batch ordering without losing or duplicating a decision."""
    if len(batches) != len(values) or type(size) is not int or size < 1:
        raise ValueError("Incomplete request outputs")
    output = [None] * size
    for batch, vectors in zip(batches, values, strict=True):
        if len(batch["indices"]) != len(vectors):
            raise ValueError("Batch output count changed")
        for index, vector in zip(batch["indices"], vectors, strict=True):
            if type(index) is not int or not 0 <= index < size or output[index] is not None:
                raise ValueError("Duplicate or invalid request index")
            output[index] = vector
    if any(value is None for value in output):
        raise ValueError("Missing request prediction")
    return output


class QwenGraphRequests:
    """Sequential replays, with one private graph/pool for each predeclared shape.

    This adapter is experimental. Full validation of one resident graph does not establish
    this cache's correctness or speed. The caller must validate the exact request workload.
    """

    def __init__(self, reference, buckets, allowed_shapes, guard, minimum_free_bytes, maximum_allocated_bytes):
        from restricted_qwen import RestrictedQwen
        if reference.batch_size != 16 or reference.model.training or str(reference.device) != "cuda":
            raise ValueError("Require the original frozen native batch-16 CUDA reference")
        self.candidate = RestrictedQwen(reference)
        if self.candidate.lm.config._attn_implementation != "sdpa":
            raise ValueError("Preserve native SDPA")
        self.buckets = tuple(buckets)
        self.allowed_shapes = {tuple(shape) for shape in allowed_shapes}
        if (not self.allowed_shapes or len(self.allowed_shapes) != len(allowed_shapes)
                or any(len(shape) != 2 or any(type(x) is not int for x in shape)
                       or not 1 <= shape[0] <= 16 or shape[1] not in self.buckets for shape in self.allowed_shapes)
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or type(maximum_allocated_bytes) is not int or maximum_allocated_bytes <= 0):
            raise ValueError("Require explicit graph shapes and memory bounds")
        self.guard = guard
        self.minimum_free_bytes = minimum_free_bytes
        self.maximum_allocated_bytes = maximum_allocated_bytes
        self.graphs = {}
        self.captures = []
        self.requests = 0
        self.closed = False

    def request(self, cases, check_eager=False):
        """Include prompt creation, tokenization, masks, transfers and CPU-owned output in this call."""
        import torch
        from qwen_prefix_graph import NativePrefixGraph
        from train_c import prompt_for
        if self.closed or not cases:
            raise ValueError("Require an open cache and complete request")
        audit = []
        for case in cases:
            prompt = prompt_for(case["qtype"], case["instructions"], list(case["options"]), case["state"])
            ids = self.candidate.tok.encode(prompt, add_special_tokens=False, truncation=False)
            audit.append({"prompt": prompt, "token_ids": ids, "encoded_tokens": len(ids), "k": len(case["options"])})
        batches = request_batches(audit, self.buckets)
        if any((batch["batch_size"], batch["length"]) not in self.allowed_shapes for batch in batches):
            raise ValueError("Request requires an unregistered graph shape")
        output, eager, trace = [], [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in batches:
                shape = (batch["batch_size"], batch["length"])
                selected = [audit[index] for index in batch["indices"]]
                ids, counts = [c["token_ids"] for c in selected], [c["k"] for c in selected]
                if shape not in self.graphs:
                    self.guard(force=True)
                    free, total = torch.cuda.mem_get_info()
                    if free < self.minimum_free_bytes:
                        raise MemoryError("Insufficient free CUDA memory for a registered graph")
                    started = time.perf_counter()
                    graph = NativePrefixGraph(self.candidate, *shape, ids, counts, self.guard)
                    self.graphs[shape] = graph
                    self.captures.append({"shape": list(shape), "request_index": self.requests,
                                          "warmup_forwards": 3, "free_cuda_bytes_before": free,
                                          "total_cuda_bytes": total,
                                          "operational_seconds": time.perf_counter() - started})
                graph = self.graphs[shape]
                captured, uncaptured = graph.run(ids, counts, self.guard, check_eager=check_eager)
                # Each graph has independent static outputs. Own CPU storage before another replay.
                output.append([vector[:k].clone().numpy() for vector, k in zip(captured.float().cpu(), counts, strict=True)])
                if check_eager:
                    eager.append([vector[:k].clone().numpy() for vector, k in zip(uncaptured.float().cpu(), counts, strict=True)])
                trace.append({**batch, "graph_replay": graph.replays})
                if torch.cuda.max_memory_allocated() > self.maximum_allocated_bytes:
                    raise MemoryError("Registered allocation bound exceeded")
        self.requests += 1
        return {"log_probabilities": restore_request(batches, output, len(cases)),
                "uncaptured_log_probabilities": restore_request(batches, eager, len(cases)) if check_eager else None,
                "batches": trace, "resident_shapes": [list(shape) for shape in self.graphs],
                "captures_total": len(self.captures)}

    def close(self):
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.closed = True
