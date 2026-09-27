"""Bounded original-protocol laptop timing of frozen Qwen full and restricted output readouts."""
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
DIRECTORY = ROOT / "results/pilot-laptop-qwen-readout-20260927"
REGISTRATION = Path(__file__).with_name("qwen_readout_latency_registration.json")


def main():
    import psutil
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use the shared run_heavy compute lock")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists() or (DIRECTORY / "latency.json").exists():
        raise ValueError("Preserve previous timing attempts")
    for relative, expected in spec["sources"].items():
        source = ROOT / relative
        if digest(source) != expected:
            raise ValueError("Registered timing source changed")
        destination = DIRECTORY / "executed_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative, expected in spec["prerequisites"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Registered probability or weight prerequisite changed")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(DIRECTORY),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "stages": [],
               "registration_sha256": digest(REGISTRATION), "sources_verified": True}
    write_json(path, receipt)
    for readout in ("full", "restricted"):
        command = [sys.executable, str(ROOT / "experiments/small_model/latency.py"), "--arm", "c-qwen08-30k",
                   "--device", "cuda", "--threads", "4", "--warmup", "5", "--reps", "40", "--reps50", "15",
                   "--cooldown-seconds", "90", "--readout", readout]
        started = time.monotonic()
        with (DIRECTORY / f"{readout}.log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=420)
            except subprocess.TimeoutExpired:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill()
                proc.wait()
                code = 124
        stage = {"readout": readout, "returncode": code, "seconds": time.monotonic() - started}
        receipt["stages"].append(stage)
        write_json(path, receipt)
        print(json.dumps(stage), flush=True)
    receipt.update(status="complete" if all(stage["returncode"] == 0 for stage in receipt["stages"]) else "partial_failure",
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   validity="Execution success is not a gate pass; verify raw samples, weights, temperature, throttling and contamination separately.")
    write_json(path, receipt)
    return 0 if receipt["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
