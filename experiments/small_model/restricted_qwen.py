"""Experimental option-letter projection over the unchanged Qwen backbone and unmerged adapters."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


class RestrictedQwen:
    """Share a frozen PredictC model; replace only its full-vocabulary output projection."""

    def __init__(self, reference):
        self.reference = reference
        self.model, self.tok = reference.model, reference.tok
        self.device = reference.device
        self.weights = dict(reference.weights)
        self.info = {**reference.info, "readout": "frozen option-letter projection; backbone, cache, LoRA and prompts unchanged"}
        self.lm = self.model.get_base_model()
        if self.model.training or self.lm.config.model_type != "qwen3_5_text":
            raise ValueError("Only a frozen Qwen3.5 text model is supported")
        head = self.lm.get_output_embeddings()
        if type(head) is not torch.nn.Linear:
            raise ValueError("A wrapped or adapted output head must not be bypassed")
        self.letters = reference.letter_ids
        if self.letters.ndim != 1 or len(self.letters) != len(set(self.letters.tolist())):
            raise ValueError("Answer letters must identify unique vocabulary rows")
        self.weight = head.weight.detach().index_select(0, self.letters).clone()
        self.bias = None if head.bias is None else head.bias.detach().index_select(0, self.letters).clone()
        self.info.update(projected_vocabulary_rows=len(self.letters), original_vocabulary_rows=head.out_features)

    @torch.no_grad()
    def logp(self, prompts, option_counts):
        if not prompts or len(prompts) != len(option_counts) or any(k < 2 or k > len(self.letters) for k in option_counts):
            raise ValueError("Every prompt needs a supported option count")
        # No clipping: the registered worker audits the same inputs used by the reference.
        encoded = self.tok(prompts, add_special_tokens=False, padding=True, truncation=False,
                           return_tensors="pt", padding_side="left")
        if encoded["input_ids"].shape[1] > 4096:
            raise ValueError("Input exceeds the unchanged context limit")
        output = self.lm.model(input_ids=encoded["input_ids"].to(self.device),
                               attention_mask=encoded["attention_mask"].to(self.device))
        hidden = output.last_hidden_state[:, -1:, :]
        logits = F.linear(hidden, self.weight, self.bias)[:, -1, :].float()
        k = max(option_counts)
        mask = torch.arange(k, device=self.device)[None, :] < torch.tensor(option_counts, device=self.device)[:, None]
        logits = torch.where(mask, logits[:, :k], torch.full_like(logits[:, :k], -1e4))
        return F.log_softmax(logits, -1)

    @torch.no_grad()
    def groups(self, groups):
        from common import thermal_guard
        from train_c import prompt_for
        flat = [(gi, c) for gi, group in enumerate(groups) for c in group]
        if not flat:
            return [[] for _ in groups]
        prompts = [prompt_for(c["qtype"], c["instructions"], list(c["options"]), c["state"]) for _, c in flat]
        order = sorted(range(len(flat)), key=lambda i: len(prompts[i]))
        results = [None] * len(flat)
        batch = self.reference.batch_size
        for start in range(0, len(order), batch):
            indices = order[start:start + batch]
            if str(self.device).startswith("cuda"):
                thermal_guard()
            context = torch.autocast("cuda", dtype=torch.bfloat16) if str(self.device).startswith("cuda") else nullcontext()
            with context:
                logp = self.logp([prompts[i] for i in indices], [len(flat[i][1]["options"]) for i in indices])
            values = logp.float().cpu().numpy()
            for value, i in zip(values, indices):
                results[i] = value[:len(flat[i][1]["options"])]
        answer = [[] for _ in groups]
        for (gi, _), value in zip(flat, results):
            answer[gi].append(value)
        return answer
