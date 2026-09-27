"""Native GLiNER classification adapter with complete probability and input-preservation checks."""
from __future__ import annotations

import math


def native_probabilities(logits, labels, answer):
    """Recover the full T=1 vector and verify its public native winner/confidence."""
    if not logits or len(logits) != len(labels) or len(set(labels)) != len(labels) or not all(map(math.isfinite, logits)):
        raise ValueError("Missing, duplicate or invalid native label logits")
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)
    probabilities = [value / total for value in values]
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    if (not math.isfinite(answer["confidence"]) or not 0 <= answer["confidence"] <= 1
            or answer["label"] != labels[index] or abs(answer["confidence"] - probabilities[index]) > 1e-6):
        raise ValueError("Captured probabilities disagree with the native answer")
    return probabilities


def audit_native_batch(batch, inputs, word_splitter):
    """Reject lost input words, changed label order, omitted instructions and descriptions."""
    if len(batch.original_texts) != len(inputs) or batch.original_texts != [case["text"] for case in inputs]:
        raise ValueError("Native preprocessing changed input membership or text")
    for index, case in enumerate(inputs):
        words = [word for word, _, _ in word_splitter(case["text"], lower=True)]
        if batch.text_tokens[index] != words:
            raise ValueError("Native preprocessing dropped or changed text words")
        schemas = batch.schema_tokens_list[index]
        if batch.task_types[index] != ["classifications"] or len(schemas) != 1:
            raise ValueError("Expected one independent classification per record")
        tokens = schemas[0]
        labels = [tokens[j + 1] for j, token in enumerate(tokens[:-1]) if token == "[L]"]
        if labels != list(case["options"]):
            raise ValueError("Native preprocessing changed label membership or order")
        prompt = tokens[2]
        if not prompt.startswith("decision: " + case["instructions"]):
            raise ValueError("Native preprocessing omitted the question")
        for key, description in case["options"].items():
            if description and f"[DESCRIPTION] {key}: {description}" not in prompt:
                raise ValueError("Native preprocessing omitted a label description")


class NativeGLiner:
    """Keep native rendering and FP32 scoring; batch independent records with different schemas."""

    def __init__(self, model_path, batch_size=4):
        from gliner2 import AutoExtractor
        import torch
        self.model = AutoExtractor.from_pretrained(str(model_path), map_location="cuda", use_flashdeberta=False)
        self.model.eval()
        if next(self.model.parameters()).dtype != torch.float32:
            raise ValueError("The registered competitor uses original FP32 weights")
        self.batch_size = batch_size
        self.audit = {"records_checked": 0, "max_padded_tokens": 0, "encoder_calls": 0}
        self._inputs, self._captured = [], []
        processor = self.model.processor
        original_collate = processor.collate_fn_inference

        def collate(rows, **kwargs):
            if kwargs.get("max_len") is not None:
                raise ValueError("Input truncation is disabled")
            batch = original_collate(rows, **kwargs)
            audit_native_batch(batch, self._inputs, processor.word_splitter)
            self.audit["records_checked"] += len(rows)
            self.audit["max_padded_tokens"] = max(self.audit["max_padded_tokens"], int(batch.input_ids.shape[1]))
            return batch

        def head_hook(module, args, output):
            self._captured.append(output.detach().float().cpu().numpy())

        def encoder_hook(module, args, kwargs):
            self.audit["encoder_calls"] += 1

        processor.collate_fn_inference = collate
        self._handles = [self.model.classifier.register_forward_hook(head_hook),
                         self.model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True)]

    def score_inputs(self, inputs):
        import torch
        from common import thermal_guard
        results = []
        for start in range(0, len(inputs), self.batch_size):
            self._inputs = inputs[start:start + self.batch_size]
            self._captured.clear()
            schemas = [self.model._classification_schema({"decision": {"labels": case["options"],
                       "prompt": case["instructions"], "multi_label": False, "class_act": "softmax"}})
                       for case in self._inputs]
            thermal_guard()
            before = self.audit["encoder_calls"]
            with torch.inference_mode():
                answers = self.model.batch_extract([case["text"] for case in self._inputs], schemas,
                           batch_size=self.batch_size, num_workers=0, include_confidence=True, max_len=None)
            if (len(answers) != len(self._inputs) or len(self._captured) != len(self._inputs)
                    or self.audit["encoder_calls"] - before != 1):
                raise ValueError("Unexpected native forward or prediction count")
            for case, raw, answer in zip(self._inputs, self._captured, answers):
                if raw.shape != (len(case["options"]), 1):
                    raise ValueError("Unexpected native head shape")
                logits = raw[:, 0].tolist()
                probabilities = native_probabilities(logits, list(case["options"]), answer["decision"])
                results.append({"logits": logits, "probabilities": probabilities, "native": answer["decision"]})
        return results

    def score_cases(self, cases):
        from tez.prompt import render_state
        inputs = []
        for case in cases:
            options = dict(case["options"])
            if len(options) != len(case["options"]):
                raise ValueError("Duplicate option keys")
            inputs.append({"text": render_state(case["state"]), "instructions": case["instructions"], "options": options})
        return self.score_inputs(inputs)
