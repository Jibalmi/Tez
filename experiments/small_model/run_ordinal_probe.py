"""Run the registered ordinal probe in sequential, bounded GPU workers under run_heavy.py."""
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
import traceback

from ordinal_probe import DIRECTORY, ROOT, prediction_row
from structure_probe import digest, write_json


def worker(arm, directory, seconds):
    model_root = ("runs/improvement-artifacts-h200-20260927/runs/student-data" if arm == "a-base-12ep"
                  else "runs/student-artifacts-h200-20260927/runs/student-data")
    os.environ.update(TEZ_PILOT_RUNS=str(ROOT / model_root), TEZ_PILOT_RESULTS=str(directory),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    from common import THERMAL, thermal_guard
    from competitor_models import predictor, model_revision
    from latency import thermal
    from train_c import kernel_packages

    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    if arm not in registration["models"] or digest(directory / "inputs.json") != registration["input_sha256"]:
        raise ValueError("Inputs or arm differ from registration")
    receipt_path, rows_path = directory / f"execution_{arm}.json", directory / f"rows_{arm}.jsonl"
    if receipt_path.exists() or rows_path.exists():
        raise ValueError("Never overwrite an existing attempt")
    source_dir = directory / "executed_source" / arm
    source_dir.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ("ordinal_probe.py", "run_ordinal_probe.py", "structure_probe.py", "eval_arms.py", "competitor_models.py",
                 "common.py", "student.py", "train_a.py", "train_c.py", "latency.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        sources[name] = digest(source)
    receipt = {"status": "running", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "input_sha256": registration["input_sha256"], "registration_sha256": digest(directory / "registration.json"),
               "source_hashes": sources, "packages": kernel_packages(), "threads": torch.get_num_threads(),
               "gpu": torch.cuda.get_device_name(), "thermal_start": thermal(), "rows": 0,
               "scope": "Diagnostic inference only, no valid latency claim"}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        thermal_guard(force=True)
        pred = predictor(arm)
        if pred.weights != registration["weights_sha256"][arm]:
            raise ValueError("Frozen model weight hashes changed")
        receipt.update(weights=pred.weights, model_info=pred.info)
        if arm.startswith("c-"):
            from huggingface_hub import snapshot_download
            from train_c import BASE
            base = Path(snapshot_download(BASE, revision=model_revision(BASE), allow_patterns=["*.safetensors"],
                                          local_files_only=True))
            receipt["base"] = {"revision": model_revision(BASE), "weights": {p.name: digest(p) for p in base.glob("*.safetensors")}}
        groups = [[{key: c[key] for key in ("state", "qtype", "instructions", "options")}] for c in inputs]
        if arm.startswith("a-"):
            from student import Q, option_text
            lengths = [len(pred.enc.encode(c["state"], [Q(c["qtype"], c["instructions"], c["options"])]).ids) for c in inputs]
            clipped = sum(len(pred.enc.ids(f" ({c['qtype']}) {c['instructions']}")) > 160
                          or any(len(pred.enc.ids(" " + option_text(k, d))) > 48 for k, d in c["options"]) for c in inputs)
            limit = 8192
        else:
            from train_c import prompt_for
            lengths = [len(pred.tok(prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]),
                                    add_special_tokens=False)["input_ids"]) for c in inputs]
            clipped, limit = 0, 4096
        receipt["token_audit"] = {"max_tokens": max(lengths), "context_limit": limit, "clipped_parts": clipped}
        if max(lengths) >= limit or clipped:
            raise ValueError("Input truncation would confound the probe")
        chunk_size = registration["groups_per_chunk"]
        with rows_path.open("x", encoding="utf-8") as output:
            for start in range(0, len(inputs), chunk_size):
                if time.monotonic() - started > seconds:
                    raise TimeoutError("Worker time budget reached")
                thermal_guard(force=True)
                chunk = groups[start:start + chunk_size]
                result = pred.groups(chunk)
                if len(result) != len(chunk) or any(len(group) != 1 for group in result):
                    raise ValueError("Wrong output count")
                for case, group in zip(inputs[start:start + chunk_size], result):
                    z = np.asarray(group[0], dtype=float)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    output.write(json.dumps(prediction_row(case, p)) + "\n")
                    receipt["rows"] += 1
                output.flush()
                write_json(receipt_path, receipt)
                print(json.dumps({"arm": arm, "rows": receipt["rows"]}), flush=True)
        receipt.update(status="complete", rows_sha256=digest(rows_path))
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_end=thermal(), thermal_governor=THERMAL)
        write_json(receipt_path, receipt)


def driver(directory):
    receipt_path = directory / "execution.json"
    if receipt_path.exists():
        raise ValueError("An execution already exists")
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 12 * 60
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "pid": os.getpid(), "stages": []}
    write_json(receipt_path, receipt)
    for arm in registration["models"]:
        remaining = deadline - time.monotonic()
        if remaining < 30:
            break
        cmd = [sys.executable, str(Path(__file__).resolve()), "--directory", str(directory), "--arm", arm,
               "--seconds", str(max(1, min(330, remaining - 15)))]
        with (directory / f"{arm}.log").open("x", encoding="utf-8") as log:
            try:
                completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=min(remaining, 360), check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        stage = {"arm": arm, "returncode": code}
        receipt["stages"].append(stage)
        write_json(receipt_path, receipt)
        print(json.dumps(stage), flush=True)
        outcome = directory / f"execution_{arm}.json"
        if outcome.exists() and json.loads(outcome.read_text(encoding="utf-8")).get("error_type") == "ThermalCooldownTimeout":
            break
    receipt.update(status="complete" if len(receipt["stages"]) == len(registration["models"])
                   and all(s["returncode"] == 0 for s in receipt["stages"]) else "partial_failure",
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(receipt_path, receipt)
    return 0 if receipt["status"] == "complete" else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--arm")
    parser.add_argument("--seconds", type=float, default=330)
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if args.arm:
        worker(args.arm, args.directory.resolve(), args.seconds)
    else:
        raise SystemExit(driver(args.directory.resolve()))
