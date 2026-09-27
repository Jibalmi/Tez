"""Bounded validation of a restricted Qwen output projection, without changing trained weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-restricted-qwen-20260927"
REGISTRATION = Path(__file__).with_name("restricted_qwen_registration.json")


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    model_root = ROOT / "runs/student-artifacts-h200-20260927/runs/student-data"
    os.environ.update(TEZ_PILOT_RUNS=str(model_root), TEZ_PILOT_RESULTS=str(DIRECTORY),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    import dataset
    from common import THERMAL, thermal_guard
    from competitor_models import model_revision
    from eval_arms import PredictC, softmax
    from restricted_qwen import RestrictedQwen
    from train_c import BASE, kernel_packages, prompt_for

    path = DIRECTORY / "parity.json"
    if path.exists():
        raise ValueError("Preserve existing parity attempts")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "packages": kernel_packages(),
               "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(), "sources": {}}
    write_json(path, receipt)
    start = time.monotonic()
    try:
        source_dir = DIRECTORY / "executed_source"
        source_dir.mkdir(exist_ok=False)
        for name in ("restricted_qwen.py", "restricted_qwen_parity.py", "restricted_qwen_registration.json",
                     "train_c.py", "train_a.py", "dataset.py", "student.py", "common.py", "eval_arms.py", "competitor_models.py"):
            source = Path(__file__).with_name(name)
            shutil.copy2(source, source_dir / name)
            receipt["sources"][name] = digest(source)
        import peft.peft_model
        from transformers.models.qwen3_5 import modeling_qwen3_5
        for name, module in (("dependency_peft_model.py", peft.peft_model), ("dependency_modeling_qwen3_5.py", modeling_qwen3_5)):
            shutil.copy2(module.__file__, source_dir / name)
            receipt["sources"][name] = digest(module.__file__)
        data = ROOT / spec["teacher_directory"]
        manifest_path = data / "manifest.json"
        if digest(manifest_path) != spec["teacher_manifest_sha256"]:
            raise ValueError("Original teacher manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(data / name) != expected:
                raise ValueError("Original teacher file changed")
        receipt["teacher_manifest_sha256"] = digest(manifest_path)
        receipt["teacher_files_verified"] = len(manifest["files"])
        dataset.RUNS = data
        _, _, examples = dataset.load_examples()
        groups = [[{"state": e.state, "qtype": d.type, "instructions": d.instructions, "options": d.options}
                   for d in e.decisions] for e in examples]
        ids = [(e.id, d.qid, len(d.options)) for e in examples for d in e.decisions]
        if len(examples) != 278 or len(ids) != 1390 or len(set((a, b) for a, b, _ in ids)) != len(ids):
            raise ValueError("Unexpected validation membership")
        thermal_guard(force=True)
        pred = PredictC(spec["reference"])
        if pred.weights != {"adapter": spec["adapter_sha256"]}:
            raise ValueError("Adapter checksum mismatch")
        base = Path(snapshot_download(BASE, revision=model_revision(BASE), allow_patterns=["*.safetensors"], local_files_only=True))
        base_weights = {p.name: digest(p) for p in base.glob("*.safetensors")}
        if list(base_weights.values()) != [spec["base_sha256"]]:
            raise ValueError("Base checkpoint checksum mismatch")
        candidate = RestrictedQwen(pred)
        receipt.update(status="running", weights=pred.weights, base_weights=base_weights, model_info=candidate.info,
                       original_cache_setting=pred.model.config.use_cache, states=len(examples), decisions=len(ids))
        lengths = [len(pred.tok(prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]),
                                add_special_tokens=False)["input_ids"]) for group in groups for c in group]
        receipt["token_audit"] = {"max_tokens": max(lengths), "context_limit": 4096}
        if max(lengths) > 4096:
            raise ValueError("Reference would truncate a validation prompt")
        write_json(path, receipt)
        all_values = {}
        for name, model in (("reference", pred), ("restricted", candidate)):
            thermal_guard(force=True)
            values = [softmax(z) for group in model.groups(groups) for z in group]
            if len(values) != len(ids):
                raise ValueError("Wrong prediction count")
            rows_path = DIRECTORY / f"{name}_rows.jsonl"
            with rows_path.open("x", encoding="utf-8") as output:
                for (state_id, qid, k), p in zip(ids, values):
                    if p.shape != (k,) or not np.isfinite(p).all() or np.any(p < 0) or abs(float(p.sum()) - 1) > 1e-5:
                        raise ValueError("Invalid parity probability vector")
                    output.write(json.dumps({"state_id": state_id, "qid": qid, "k": k, "probabilities": p.tolist()}) + "\n")
            receipt[name + "_rows_sha256"] = digest(rows_path)
            write_json(path, receipt)
            all_values[name] = values
        original, restricted = all_values["reference"], all_values["restricted"]
        agreement = float(np.mean([a.argmax() == b.argmax() for a, b in zip(original, restricted)]))
        maximum = max(float(np.abs(a - b).max()) for a, b in zip(original, restricted))
        mean_tv = float(np.mean([np.abs(a - b).sum() / 2 for a, b in zip(original, restricted)]))
        passed = (agreement >= spec["argmax_agreement_min"] and maximum <= spec["max_probability_delta_max"]
                  and mean_tv <= spec["mean_total_variation_max"])
        if pred.model.config.use_cache != receipt["original_cache_setting"]:
            raise ValueError("Cache setting was changed")
        receipt.update(status="complete", argmax_agreement=agreement, max_probability_delta=maximum,
                       mean_total_variation=mean_tv, passed=passed,
                       promotion="Not promoted; valid improved target-laptop latency remains separately required")
        return 0 if passed else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                       thermal_governor=THERMAL)
        write_json(path, receipt)
        print(json.dumps(receipt), flush=True)


def driver():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": 480}
    write_json(path, receipt)
    with (DIRECTORY / "parity.log").open("x", encoding="utf-8") as log:
        try:
            outcome = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=480, check=False)
            code = outcome.returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt.update(status="complete" if code in (0, 2) else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    raise SystemExit(worker() if args.worker else driver())
