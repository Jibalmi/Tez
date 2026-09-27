"""Check full native GLiNER probability extraction on fixed non-benchmark inputs."""
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
DIRECTORY = ROOT / "results/pilot-gliner2-probe-20260927"
PREPARATION = ROOT / "results/pilot-gliner2-discovery-20260927/preparation_attempt2.json"
MODELS = ("fastino/GLiNER2.5-Decide", "fastino/GLiNER2.5-multi-Decide")
ATTEMPT = 1
CASES = [
    {"text": '{"message":"Please replace my damaged headphones (order 17)."}',
     "instructions": "Which team should handle this request (choose one)?",
     "options": {"support": "Repairs and replacements", "sales": "New purchases", "billing": "Invoices and payments"}},
    {"text": '{"message":"Meine Kopfhörer sind beschädigt. Bitte senden Sie einen Ersatz."}',
     "instructions": "Which team should handle this request?",
     "options": {"support": "Repairs and replacements", "sales": "New purchases", "billing": "Invoices and payments"}},
    {"text": '{"message":"The door is locked and the key is missing."}',
     "instructions": "Is the door unlocked?", "options": {"true": "yes", "false": "no"}},
    {"text": '{"message":"The service was acceptable; neither excellent nor awful."}',
     "instructions": "Rate the expressed satisfaction.",
     "options": {"0": "Very dissatisfied", "1": "Dissatisfied", "2": "Neutral (mixed)", "3": "Satisfied", "4": "Very satisfied"}},
]


def worker(name):
    import numpy as np
    import torch
    from gliner2 import AutoExtractor
    from common import THERMAL, thermal_guard
    from importlib.metadata import version

    receipt_path = DIRECTORY / (name.split("/")[-1] + ".json")
    if receipt_path.exists():
        raise ValueError("Preserve existing native probes")
    prep = json.loads(PREPARATION.read_text(encoding="utf-8"))
    entry = prep["models"][name]
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(), "model": name,
               "revision": entry["revision"], "preparation_sha256": digest(PREPARATION), "source_sha256": digest(__file__),
               "cases": CASES, "rows": [], "scope": "Interface compatibility only; no accuracy or latency claims",
               "packages": {package: version(package) for package in ("torch", "gliner2", "transformers", "tokenizers", "huggingface-hub")}}
    start = time.monotonic()
    write_json(receipt_path, receipt)
    try:
        if prep["status"] != "staged" or not prep["baseline_unchanged"]:
            raise ValueError("Complete isolated preparation first")
        overlay = ROOT / prep["overlay"]
        for filename, expected in prep["installed_file_hashes"].items():
            if digest(overlay / filename) != expected:
                raise ValueError("Staged package changed")
        model_path = ROOT / entry["directory"]
        for filename, file in entry["files"].items():
            if digest(model_path / filename) != file["sha256"]:
                raise ValueError("Staged model changed")
        thermal_guard(force=True)
        model = AutoExtractor.from_pretrained(str(model_path), map_location="cuda", use_flashdeberta=False)
        model.eval()
        captures, token_shapes = [], []

        def head_hook(module, args, output):
            captures.append(output.detach().float().cpu().numpy())

        def encoder_hook(module, args, kwargs):
            token_shapes.append(list(kwargs["input_ids"].shape))

        head_handle = model.classifier.register_forward_hook(head_hook)
        encoder_handle = model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True)
        for index, case in enumerate(CASES):
            captures.clear()
            token_shapes.clear()
            thermal_guard(force=True)
            with torch.inference_mode():
                answer = model.classify_text(case["text"], {"decision": {"labels": case["options"],
                                             "prompt": case["instructions"], "multi_label": False, "class_act": "softmax"}},
                                             include_confidence=True, max_len=None)
            if len(captures) != 1 or captures[0].shape != (len(case["options"]), 1) or len(token_shapes) != 1:
                raise ValueError("Unexpected native head/encoder calls or option count")
            logits = captures[0][:, 0]
            probabilities = np.exp(logits.astype(float) - logits.max())
            probabilities /= probabilities.sum()
            labels = list(case["options"])
            native = answer["decision"]
            if (not np.isfinite(probabilities).all() or native["label"] != labels[int(probabilities.argmax())]
                    or abs(native["confidence"] - float(probabilities.max())) > 1e-6):
                raise ValueError("Full captured probabilities differ from the native answer")
            if model._inference_collator.max_len is not None:
                raise ValueError("Native collator would truncate text")
            receipt["rows"].append({"case": index, "labels": labels, "logits": logits.tolist(),
                                    "probabilities": probabilities.tolist(), "native": native, "encoded_shape": token_shapes[0]})
            write_json(receipt_path, receipt)
        head_handle.remove()
        encoder_handle.remove()
        receipt.update(status="complete", architecture=model.architecture, dtype=str(next(model.parameters()).dtype),
                       gpu=torch.cuda.get_device_name(), cases_verified=len(CASES), passed=True)
        return 0
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                       thermal_governor=THERMAL)
        write_json(receipt_path, receipt)
        print(json.dumps({key: receipt.get(key) for key in ("model", "status", "cases_verified", "error")}), flush=True)


def driver():
    import psutil
    prep = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if prep["status"] != "staged":
        raise ValueError("Complete preparation before queuing this probe")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    shutil.copy2(__file__, DIRECTORY / "executed_source.py")
    write_json(DIRECTORY / "cases.json", CASES)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / prep["overlay"]) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", USE_TF="0")
    if ATTEMPT == 2:
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Fixed non-benchmark interface probe only", "model_returncodes": {}, "per_worker_timeout_seconds": 180}
    write_json(path, receipt)
    selected = MODELS[:1] if ATTEMPT == 2 else MODELS
    receipt["models"] = list(selected)
    receipt["attempt"] = ATTEMPT
    for name in selected:
        with (DIRECTORY / (name.split("/")[-1] + ".log")).open("x", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", name, "--attempt", str(ATTEMPT)], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill()
                proc.wait()
                code = 124
        receipt["model_returncodes"][name] = code
        write_json(path, receipt)
    okay = all(code == 0 for code in receipt["model_returncodes"].values())
    receipt.update(status="complete" if okay else "failed", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if okay else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    ATTEMPT = args.attempt
    if ATTEMPT == 2:
        DIRECTORY = ROOT / "results/pilot-gliner2-probe-attempt2-20260927"
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    raise SystemExit(worker(args.worker) if args.worker else driver())
