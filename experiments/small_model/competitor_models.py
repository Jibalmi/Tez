"""Pinned competitors accessed through their native libraries, with full option probabilities."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from common import ROOT, file_sha256, thermal_guard

REVISIONS = {
    "knowledgator/gliclass-x-base": "4599a431a1b25b4eb076b46ae97c9ba61f82f416",
    "knowledgator/gliclass-multilang-mini": "0bd888b6c3ef9fca5f0a9d407bddfbbc7623486b",
    "knowledgator/gliclass-instruct-base-v1.0": "4f6a108b08a5537f395521d19b5073e197923dd3",
    "knowledgator/gliclass-instruct-edge-v1.0": "727be8a417f6a7718e591b025e07054c146d8139",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "jaredpalmer/kev-0.8b": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
    "Mapika/decider-0.8b": "a0a01d6f8135298f400a8c856b355793012ae971",
}


def model_revision(name):
    return REVISIONS[name]


def kev_request(group):
    questions = {}
    keys = []
    for i, case in enumerate(group):
        q = {"type": case["qtype"], "instructions": case["instructions"]}
        options = case["options"]
        if case["qtype"] == "score":
            q["criteria"] = [description for _, description in options]
            keys.append([str(j) for j in range(len(options))])
        else:
            q["criteria"] = dict(options)
            keys.append([key for key, _ in options])
        # LocalPredictor's materializer requires metadata, but no gold label enters this adapter.
        q["label"] = keys[-1][0] if case["qtype"] == "choice" else 0
        q["src"] = "external_pilot"
        questions[str(i)] = q
    return {"state": group[0]["state"], "questions": questions}, keys


class PredictFrozenQwen:
    def __init__(self, device="cuda"):
        from huggingface_hub import snapshot_download
        from train_c import BASE, load_base
        self.tok, self.model, self.letter_ids = load_base(device=device)
        self.model.eval()
        self.device = device
        path = Path(snapshot_download(BASE, revision=model_revision(BASE), allow_patterns=["config.json"]))
        self.weights = {p.name: file_sha256(p) for p in path.glob("*.safetensors")}
        self.info = {"temperatures": {}, "model": BASE, "revision": model_revision(BASE), "fine_tuned": False}

    def groups(self, groups, batch=16):
        from eval_arms import PredictC
        return PredictC.groups(self, groups, batch=batch)


class PredictKev:
    def __init__(self, device="cuda"):
        sys.path.insert(0, str(ROOT / "runs/research/kev"))
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        name = "jaredpalmer/kev-0.8b"
        self.pred = LocalPredictor(name + "@" + model_revision(name), device,
                                   LoadOptions(dtype=torch.bfloat16, merge=True, temperature=1.0),
                                   context={"max_state": 8192, "max_branch": 4096, "max_packed": 65536})
        self.device = device
        checkpoint = self.pred.checkpoint
        self.weights = {"adapter": checkpoint.weights_sha256(), "head": file_sha256(checkpoint.file("head.pt"))}
        temperature = float(checkpoint.meta.temperature)
        self.info = {"temperatures": {ty: temperature for ty in ("choice", "choice_many", "noul", "score")},
                     "calibration_source": "released checkpoint temperature; raw T=1 also scored",
                     "model": name, "revision": model_revision(name),
                     "dtype": "bf16", "merge": True, "temperature": 1.0,
                     "readout": "native LocalPredictor, no date-fact preprocessing",
                     "code_commit": (ROOT / "runs/research/kev-commit.txt").read_text().strip()}

    @torch.no_grad()
    def groups(self, groups):
        result = []
        for group in groups:
            thermal_guard()
            record, keys = kev_request(group)
            prediction = self.pred(record)
            result.append([np.asarray([prediction["logits"][str(i)][k] for k in ks], dtype=float)
                           for i, ks in enumerate(keys)])
        return result


def decider_probabilities(case, answer):
    if case["qtype"] == "noul":
        values = {"true": answer["noul"], "false": 1 - answer["noul"]}
        probabilities = [values[k] for k, _ in case["options"]]
    elif case["qtype"] == "score":
        probabilities = [answer["probabilities"][str(i)] for i in range(len(case["options"]))]
    else:
        probabilities = [answer["probabilities"][k] for k, _ in case["options"]]
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("Invalid native Decider probability distribution")
    # The public API rounds to four decimals; preserve its distribution apart from normalising this rounding.
    return probabilities / probabilities.sum()


class PredictDecider:
    def __init__(self, device="cuda"):
        from huggingface_hub import snapshot_download
        sys.path.insert(0, str(ROOT / "runs/research/decider"))
        from decider.infer import Decider
        name = "Mapika/decider-0.8b"
        path = Path(snapshot_download(name, revision=model_revision(name),
                                     allow_patterns=["*.json", "*.safetensors", "*.jinja"]))
        self.pred = Decider(str(path), device=device, dtype=torch.bfloat16, temperature=1.0, use_graphs=False)
        self.device = device
        self.weights = {p.name: file_sha256(p) for p in path.glob("*.safetensors")}
        self.info = {"temperatures": {}, "model": name, "revision": model_revision(name),
                     "code_commit": (ROOT / "runs/research/decider-commit.txt").read_text().strip(),
                     "dtype": "bf16", "use_graphs": False, "temperature": 1.0,
                     "isolated_levels": self.pred.isolated_levels, "layout": self.pred.layout,
                     "readout": "Native public system_one; four-decimal probabilities normalised after rounding",
                     "calibration_note": "T=1 only; applying temperature after isolated-level aggregation is not native calibration"}

    @torch.no_grad()
    def groups(self, groups):
        result = []
        for group in groups:
            thermal_guard()
            record, _ = kev_request(group)
            questions = {k: {field: value for field, value in question.items() if field not in ("src", "label")}
                         for k, question in record["questions"].items()}
            prediction = self.pred.system_one(record["state"], questions, independent=True, max_state_tokens=8192)
            result.append([np.log(np.maximum(decider_probabilities(case, prediction["answers"][str(i)]), 1e-30))
                           for i, case in enumerate(group)])
        return result


def predictor(arm, device="cuda"):
    from eval_arms import PredictA, PredictC, PredictGLiClass
    if arm == "qwen08-frozen":
        return PredictFrozenQwen(device)
    if arm == "kev08-native":
        return PredictKev(device)
    if arm == "decider08-native":
        return PredictDecider(device)
    if arm == "c-qwen08-merged":
        return PredictC("c-qwen08-30k", device, merge=True)
    if arm == "c-qwen08-cacheoff":
        return PredictC("c-qwen08-30k", device, disable_cache=True)
    if arm == "c-qwen08-batch64":
        pred = PredictC("c-qwen08-30k", device)
        pred.batch_size = 64
        pred.info = {**pred.info, "inference_batch_size": 64}
        return pred
    if arm.startswith("c-"):
        return PredictC(arm, device)
    if arm.startswith("gliclass"):
        return PredictGLiClass(arm, device)
    return PredictA(arm, device)
