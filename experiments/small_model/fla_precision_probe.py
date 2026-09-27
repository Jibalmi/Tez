"""Compare unchanged and FP32-input FLA on four non-benchmark inputs; no quality or speed claim."""
from datetime import datetime, timezone
from importlib.metadata import version
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
DIRECTORY = ROOT / "results/pilot-fla-precision-probe-20260927"
REGISTRATION = Path(__file__).with_name("fla_precision_registration.json")


def dispatch_parts(function, require_fla=True):
    """Recover the installed dispatcher without changing its source or another process."""
    for _ in range(5):
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        if {"implementation", "is_new_implementation", "torch_function"} <= cells.keys():
            implementation = cells["implementation"].cell_contents
            fallback = cells["torch_function"].cell_contents
            if require_fla and (not cells["is_new_implementation"].cell_contents
                                or not implementation.__module__.startswith("fla.ops.gated_delta_rule")):
                raise ValueError("The frozen FLA implementation is not selected")
            if not require_fla and (cells["is_new_implementation"].cell_contents or implementation is not fallback):
                raise ValueError("The convolution reference must remain unchanged")
            return implementation, fallback
        if not hasattr(function, "__wrapped__"):
            break
        function = function.__wrapped__
    raise ValueError("The installed dispatcher cannot be inspected")


def wrapper(implementation, record, promote):
    import torch
    applicable = tuple(inspect.signature(implementation).parameters)

    def call(query, key, value, g, beta, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key in applicable}
        tensors = [query, key, value, g, beta]
        if any(x.device.type != "cuda" for x in tensors):
            raise ValueError("Expected CUDA attention inputs")
        record["calls"] += 1
        input_dtype = query.dtype
        if promote:
            tensors = [x.float() for x in tensors]
            if kwargs.get("initial_state") is not None:
                kwargs["initial_state"] = kwargs["initial_state"].float()
        # Disable autocast only inside the promoted attention operator, preserving the rest of the model.
        from contextlib import nullcontext
        context = torch.autocast("cuda", enabled=False) if promote else nullcontext()
        with context:
            output, state = implementation(*tensors[:3], g=tensors[3], beta=tensors[4], **kwargs)
        if record["calls"] == 1:
            record.update(input_dtype=str(input_dtype), called_dtypes=[str(x.dtype) for x in tensors],
                          shapes=[list(x.shape) for x in tensors], implementation_output_dtype=str(output.dtype),
                          state_dtype=None if state is None else str(state.dtype))
        if promote:
            output = output.to(input_dtype)
        return output, state
    return call


def compare(reference, candidate, counts):
    import math
    changes, variations, matches = [], [], []
    if len(reference) != len(counts) or len(candidate) != len(counts):
        raise ValueError("Incomplete predictions")
    for a, b, count in zip(reference, candidate, counts, strict=True):
        a, b = a[:count], b[:count]
        for row in (a, b):
            if len(row) != count or not all(math.isfinite(v) and 0 <= v <= 1 for v in row) or abs(sum(row) - 1) > 1e-5:
                raise ValueError("Invalid probability vector")
        delta = [abs(p - q) for p, q in zip(a, b, strict=True)]
        changes.extend(delta)
        variations.append(sum(delta) / 2)
        matches.append(max(range(count), key=a.__getitem__) == max(range(count), key=b.__getitem__))
    result = {"max_probability_delta": max(changes), "mean_total_variation": sum(variations) / len(variations),
              "argmax_agreement": sum(matches) / len(matches)}
    result["passed"] = result["max_probability_delta"] <= .04 and result["mean_total_variation"] <= .005 and result["argmax_agreement"] >= .99
    return result


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    path = DIRECTORY / "probe.json"
    if path.exists():
        raise ValueError("Preserve previous precision probe")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "sources": {}, "rows": [], "dispatch": {}}
    started = time.monotonic()
    write_json(path, receipt)
    try:
        for name, expected in spec["sources"].items():
            source = ROOT / name
            if digest(source) != expected:
                raise ValueError("Registered source changed")
            destination = DIRECTORY / "executed_source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            receipt["sources"][name] = expected
        for name, expected in spec["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen prerequisite changed")
        preparation = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
        overlay = ROOT / preparation["overlay"]
        for name, expected in preparation["installed_file_hashes"].items():
            if digest(overlay / name) != expected:
                raise ValueError("Frozen kernel dependency changed")
        if {name: version(name) for name in spec["packages"]} != spec["packages"]:
            raise ValueError("Dependency versions changed")
        import torch
        from huggingface_hub import snapshot_download
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        from functools import partial
        import common
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        from eval_arms import PredictC
        from train_c import letter_logp, prompt_for
        if digest(qwen.__file__) != spec["qwen_source_sha256"]:
            raise ValueError("Qwen dependency source changed")
        for name in ("causal_conv1d_fn", "causal_conv1d_update"):
            dispatch_parts(getattr(qwen, name), require_fla=False)
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Base model weights changed")
        functions = {name: dispatch_parts(getattr(qwen, name)) for name in
                     ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
        for implementation, _ in functions.values():
            if not Path(inspect.getsourcefile(inspect.unwrap(implementation))).resolve().is_relative_to(overlay.resolve()):
                raise ValueError("FLA dispatch is outside the isolated overlay")
        thermal_guard(force=True)
        predictor = PredictC("c-qwen08-30k")
        if predictor.weights != {"adapter": spec["adapter_sha256"]}:
            raise ValueError("Adapter weights changed")
        prompts = [prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]) for c in spec["cases"]]
        counts = [len(c["options"]) for c in spec["cases"]]
        lengths = [len(predictor.tok.encode(p, add_special_tokens=False, truncation=False)) for p in prompts]
        if max(lengths) > 4096:
            raise ValueError("Do not truncate probe inputs")
        receipt.update(status="running", gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
                       packages=spec["packages"], weights=predictor.weights, encoded_lengths=lengths,
                       base_sha256=spec["base_sha256"], original_cache_setting=predictor.model.config.use_cache,
                       precision_note="FP32 attention inputs do not imply all-IEEE arithmetic: the unchanged FLA/Triton kernels still have TF32 paths.")
        write_json(path, receipt)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for arm in ("torch_fallback", "fla_native", "fla_fp32_inputs"):
                receipt["dispatch"][arm] = {}
                for name, (implementation, fallback) in functions.items():
                    selected = fallback if arm == "torch_fallback" else implementation
                    record = {"implementation": selected.__module__ + "." + selected.__name__,
                              "unwrapped_source_sha256": digest(inspect.getsourcefile(inspect.unwrap(selected))), "calls": 0}
                    receipt["dispatch"][arm][name] = record
                    setattr(qwen, name, wrapper(selected, record, arm == "fla_fp32_inputs"))
                values = []
                for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
                    thermal_guard(force=True)
                    logp, _ = letter_logp(predictor.model, predictor.tok, predictor.letter_ids,
                                          [prompts[i] for i in order], [counts[i] for i in order], "cuda", max_len=4096)
                    probabilities = logp.exp().double().cpu().tolist()
                    values.append({"order": order, "option_counts": [counts[i] for i in order], "probabilities": probabilities})
                receipt["rows"].append({"arm": arm, "batches": values})
                write_json(path, receipt)
        rows = {row["arm"]: row["batches"] for row in receipt["rows"]}
        metrics = {}
        for arm in ("fla_native", "fla_fp32_inputs"):
            a = [p for batch in rows["torch_fallback"] for p in batch["probabilities"]]
            b = [p for batch in rows[arm] for p in batch["probabilities"]]
            ks = [k for batch in rows[arm] for k in batch["option_counts"]]
            metrics[arm] = compare(a, b, ks)
            if receipt["dispatch"][arm]["torch_chunk_gated_delta_rule"]["calls"] == 0:
                raise ValueError("No optimized attention dispatch observed")
        receipt.update(status="complete", passed=metrics["fla_fp32_inputs"]["passed"], metrics=metrics,
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                       scope="Four fixed non-benchmark inputs in two batch orders. No full-validation parity, quality improvement, speed result, training or runtime promotion.")
        return 0 if receipt["passed"] else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve previous execution")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite changed")
    prep = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / prep["overlay"]), PYTHONUTF8="1", PYTHONIOENCODING="utf-8", USE_TF="0", USE_HUB_KERNELS="0",
               FLA_DISABLE_BACKEND_DISPATCH="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(DIRECTORY),
               TRITON_CACHE_DIR=str(ROOT / "runs/pilot-fla-precision-probe-20260927/triton-cache"),
               CC=str(ROOT / spec["compiler"]), CUDA_PATH=str(ROOT / spec["cuda_directory"]))
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": 480}
    write_json(path, receipt)
    with (DIRECTORY / "probe.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=480)
        except subprocess.TimeoutExpired:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
    receipt.update(status="complete" if code == 0 else "failed", returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if "--worker" in sys.argv:
        raise SystemExit(worker())
    raise SystemExit(driver())
