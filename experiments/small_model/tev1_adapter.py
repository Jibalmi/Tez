"""Local Tev1 inference using the publisher's pinned decision payload and complete option logits."""
import hashlib
import importlib.util
import json
import math


def load_native(path):
    spec = importlib.util.spec_from_file_location("tev1_native_request", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def render_case(native, case, repository):
    record = {"state": case["state"], "question": case["instructions"],
              "options": [{"label": chr(65 + i), "key": key, "description": description}
                          for i, (key, description) in enumerate(case["options"])]}
    body = native.payload(record, repository)
    if json.loads(body["messages"][1]["content"]) != record:
        raise ValueError("Native helper changed decision content")
    return record, body["messages"]


class NativeTev:
    def __init__(self, model_path, native_path, repository, context_bound):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.native = load_native(native_path)
        self.repository = repository
        self.context_bound = context_bound
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
        self.audit = {"records_checked": 0, "max_encoded_tokens": 0, "batch_size": 1}

    def score_cases(self, cases):
        import torch
        from common import thermal_guard
        values = []
        for case in cases:
            record, messages = render_case(self.native, case, self.repository)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = [self.tokenizer.encode(option["label"], add_special_tokens=False) for option in record["options"]]
            if any(len(v) != 1 for v in ids) or len({v[0] for v in ids}) != len(ids):
                raise ValueError("Option letters must be unique single tokens")
            token_ids = [value[0] for value in ids]
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).to("cuda")
            count = int(encoded["input_ids"].shape[1])
            if count > self.context_bound:
                raise ValueError("Input exceeds registered context bound; truncation is prohibited")
            thermal_guard()
            with torch.inference_mode():
                full = self.model(**encoded, logits_to_keep=1).logits[0, -1].float()
                logits = full[token_ids].double()
                probabilities = torch.softmax(logits, -1).cpu().tolist()
                mass = float(torch.softmax(full.double(), -1)[token_ids].sum().item())
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-12:
                raise ValueError("Invalid option probabilities")
            self.audit["records_checked"] += 1
            self.audit["max_encoded_tokens"] = max(count, self.audit["max_encoded_tokens"])
            values.append({"logits": logits.cpu().tolist(), "probabilities": probabilities,
                           "unrestricted_option_mass": mass, "encoded_tokens": count, "letter_token_ids": token_ids,
                           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                           "payload_sha256": hashlib.sha256(json.dumps(record, ensure_ascii=False).encode("utf-8")).hexdigest()})
        return values
