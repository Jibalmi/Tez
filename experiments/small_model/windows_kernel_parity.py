"""Check isolated FLA dispatch and full Qwen probability parity against a frozen laptop reference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import version
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-fla-qwen-20260927"
REGISTRATION = Path(__file__).with_name("windows_kernel_registration.json")


def count_implementation(implementation, record):
    @wraps(implementation)
    def counted(*args, **kwargs):
        record["calls"] += 1
        if record["calls"] == 1:
            tensors = [a for a in args if hasattr(a, "device")]
            tensors += [a for a in kwargs.values() if hasattr(a, "device")]
            record["first_tensors"] = [{"shape": list(a.shape), "device": str(a.device), "dtype": str(a.dtype)}
                                       for a in tensors]
            if not tensors or any(a.device.type != "cuda" for a in tensors):
                raise ValueError("Expected CUDA tensors at FLA dispatch")
        return implementation(*args, **kwargs)
    return counted


def track_dispatch(function, expected_module):
    """Count the selected FLA implementation itself; reject a reference fallback."""
    for _ in range(5):
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        if "implementation" in cells and "is_new_implementation" in cells:
            implementation = cells["implementation"].cell_contents
            if not cells["is_new_implementation"].cell_contents or not implementation.__module__.startswith(expected_module):
                raise ValueError("Optimized FLA dispatch is absent")
            source = inspect.getsourcefile(implementation)
            record = {"implementation": implementation.__module__ + "." + implementation.__name__,
                      "source": source, "source_sha256": digest(source), "calls": 0}

            cells["implementation"].cell_contents = count_implementation(implementation, record)
            return record
        if not hasattr(function, "__wrapped__"):
            break
        function = function.__wrapped__
    raise ValueError("Cannot verify the installed Transformers dispatch closure")


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    path = DIRECTORY / "parity.json"
    if path.exists():
        raise ValueError("Preserve existing parity attempts")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "sources": {}, "dispatch": {}}
    write_json(path, receipt)
    start = time.monotonic()
    thermal = {}
    try:
        preparation_path = ROOT / spec["preparation"]
        if digest(preparation_path) != spec["preparation_sha256"]:
            raise ValueError("Kernel preparation changed")
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        overlay = ROOT / preparation["overlay"]
        for name, expected in preparation["installed_file_hashes"].items():
            if digest(overlay / name) != expected:
                raise ValueError("Staged kernel file changed")
        receipt["overlay_files_verified"] = len(preparation["installed_file_hashes"])
        if "compiler" in spec:
            compiler = ROOT / spec["compiler"]
            if (digest(compiler) != spec["compiler_sha256"] or os.environ.get("CC") != str(compiler)
                    or os.environ.get("CUDA_PATH") != str(ROOT / spec["cuda_directory"])):
                raise ValueError("Registered isolated toolchain is not active")
            receipt["toolchain"] = {"compiler": str(compiler), "compiler_sha256": digest(compiler),
                                    "cuda_directory": os.environ["CUDA_PATH"]}
        receipt["packages"] = {name: version(name) for name in spec["packages"]}
        if receipt["packages"] != spec["packages"]:
            raise ValueError("Kernel/backend versions changed")
        for name in ("fla", "triton", "einops"):
            found = importlib.util.find_spec(name)
            if found is None or not Path(found.origin).resolve().is_relative_to(overlay.resolve()):
                raise ValueError("Kernel module is outside the isolated overlay")

        reference_path = ROOT / spec["reference_receipt"]
        reference_rows = ROOT / spec["reference_rows"]
        if digest(reference_path) != spec["reference_receipt_sha256"] or digest(reference_rows) != spec["reference_rows_sha256"]:
            raise ValueError("Frozen reference changed")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference["status"] != "complete" or reference["reference_rows_sha256"] != digest(reference_rows):
            raise ValueError("Reference was not completed")
        source_dir = DIRECTORY / "executed_source"
        source_dir.mkdir(exist_ok=False)
        for name in ("windows_kernel_parity.py", REGISTRATION.name, "parity_evidence.py", "train_c.py", "train_a.py",
                     "dataset.py", "student.py", "common.py", "eval_arms.py", "competitor_models.py"):
            source = Path(__file__).with_name(name)
            if name in reference["sources"] and digest(source) != reference["sources"][name]:
                raise ValueError("Reference inference source changed")
            shutil.copy2(source, source_dir / name)
            receipt["sources"][name] = digest(source)
        os.environ.update(TEZ_PILOT_RUNS=str(ROOT / "runs/student-artifacts-h200-20260927/runs/student-data"),
                          TEZ_PILOT_RESULTS=str(DIRECTORY), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_HUB_KERNELS="0")
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        import dataset
        from common import THERMAL, thermal_guard
        from competitor_models import model_revision
        from eval_arms import PredictC, softmax
        from parity_evidence import compare_rows, read_rows
        from train_c import BASE, prompt_for
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        import transformers.integrations.hub_kernels as hub
        import peft.peft_model

        thermal = THERMAL
        receipt.update(gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads())
        for name, module in (("dependency_peft_model.py", peft.peft_model), ("dependency_modeling_qwen3_5.py", qwen),
                             ("dependency_hub_kernels.py", hub)):
            if name in reference["sources"] and digest(module.__file__) != reference["sources"][name]:
                raise ValueError("Reference dependency source changed")
            shutil.copy2(module.__file__, source_dir / name)
            receipt["sources"][name] = digest(module.__file__)
        receipt["dispatch"] = {name: track_dispatch(getattr(qwen, name), "fla.ops.gated_delta_rule")
                               for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
        data = ROOT / spec["teacher_directory"]
        manifest_path = data / "manifest.json"
        if digest(manifest_path) != spec["teacher_manifest_sha256"]:
            raise ValueError("Original teacher manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(data / name) != expected:
                raise ValueError("Original teacher file changed")
        receipt["teacher_files_verified"] = len(manifest["files"])
        dataset.RUNS = data
        _, _, examples = dataset.load_examples()
        groups = [[{"state": e.state, "qtype": d.type, "instructions": d.instructions, "options": d.options}
                   for d in e.decisions] for e in examples]
        ids = [(e.id, d.qid, len(d.options)) for e in examples for d in e.decisions]
        if len(examples) != 278 or len(ids) != 1390:
            raise ValueError("Unexpected validation membership")
        baseline = read_rows(reference_rows, expected_ids=ids)
        write_json(DIRECTORY / "validation_inputs.json", {"ids": ids, "groups": groups})
        receipt["validation_inputs_sha256"] = digest(DIRECTORY / "validation_inputs.json")
        thermal_guard(force=True)
        pred = PredictC(spec["reference"])
        if pred.weights != {"adapter": spec["adapter_sha256"]}:
            raise ValueError("Adapter checksum mismatch")
        base = Path(snapshot_download(BASE, revision=model_revision(BASE), allow_patterns=["*.safetensors"], local_files_only=True))
        base_weights = {p.name: digest(p) for p in base.glob("*.safetensors")}
        if list(base_weights.values()) != [spec["base_sha256"]]:
            raise ValueError("Base checkpoint checksum mismatch")
        if pred.model.config.use_cache != reference["original_cache_setting"] or pred.batch_size != 16:
            raise ValueError("Reference cache or batch setting changed")
        lengths = [len(pred.tok(prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]),
                                add_special_tokens=False)["input_ids"]) for group in groups for c in group]
        if max(lengths) > 4096:
            raise ValueError("Reference would truncate a validation prompt")
        receipt.update(status="running", weights=pred.weights, base_weights=base_weights, model_info=pred.info,
                       token_audit={"max_tokens": max(lengths), "context_limit": 4096}, states=len(examples), decisions=len(ids))
        write_json(path, receipt)
        values = [softmax(z) for group in pred.groups(groups) for z in group]
        if len(values) != len(ids):
            raise ValueError("Wrong prediction count")
        rows_path = DIRECTORY / "candidate_rows.jsonl"
        with rows_path.open("x", encoding="utf-8") as output:
            for (state_id, qid, k), p in zip(ids, values):
                if p.shape != (k,) or not np.isfinite(p).all():
                    raise ValueError("Invalid candidate probability shape")
                output.write(json.dumps({"state_id": state_id, "qid": qid, "k": k, "probabilities": p.tolist()}) + "\n")
        if receipt["dispatch"]["torch_chunk_gated_delta_rule"]["calls"] == 0:
            raise ValueError("No optimized attention call was observed")
        metrics = compare_rows(baseline, read_rows(rows_path, expected_ids=ids), spec)
        receipt.update(status="complete", **metrics, candidate_rows_sha256=digest(rows_path),
                       promotion="Not promoted; valid improved target-laptop latency is separately required")
        return 0 if metrics["passed"] else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                       thermal_governor=thermal)
        write_json(path, receipt)
        print(json.dumps(receipt), flush=True)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    preparation = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / preparation["overlay"]) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TRITON_CACHE_DIR"] = str(ROOT / "runs/pilot-windows-kernels-20260927/triton-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if "compiler" in spec:
        compiler = ROOT / spec["compiler"]
        if digest(compiler) != spec["compiler_sha256"]:
            raise ValueError("Registered compiler changed")
        env["CC"] = str(compiler)
        env["CUDA_PATH"] = str(ROOT / spec["cuda_directory"])
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": 720}
    write_json(path, receipt)
    with (DIRECTORY / "parity.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--attempt", str(spec.get("attempt", 1))], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=720)
        except subprocess.TimeoutExpired:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
    receipt.update(status="complete" if code in (0, 2) else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.attempt == 2:
        DIRECTORY = ROOT / "results/pilot-fla-qwen-attempt2-20260927"
        REGISTRATION = Path(__file__).with_name("windows_kernel_registration_attempt2.json")
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    raise SystemExit(worker() if args.worker else driver())
