"""Bound sequential probe workers and preserve every outcome, while a parent owns the compute lock."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from structure_probe import write_json


def main():
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Launch with run_heavy.py")
    root = Path(__file__).resolve().parents[2]
    directory = root / "results/pilot-structure-probe-20260927"
    original = root / "runs/student-artifacts-h200-20260927/runs/student-data"
    longer = root / "runs/improvement-artifacts-h200-20260927/runs/student-data"
    receipt_path = directory / "execution.json"
    if receipt_path.exists():
        raise ValueError("Do not overwrite a recorded attempt")
    deadline = time.monotonic() + 18 * 60
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "stages": []}
    write_json(receipt_path, receipt)
    for arm, model_root in (("a-base-30k", original), ("a-base-12ep", longer), ("c-qwen08-30k", original)):
        remaining = deadline - time.monotonic()
        if remaining < 60:
            break
        cmd = [sys.executable, str(Path(__file__).with_name("run_structure_probe.py")), "--arm", arm,
               "--directory", str(directory), "--model-root", str(model_root), "--minutes", "8"]
        started = time.monotonic()
        with (directory / f"{arm}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(remaining, 9 * 60))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        stage = {"arm": arm, "returncode": code, "seconds": time.monotonic() - started}
        receipt["stages"].append(stage)
        write_json(receipt_path, receipt)
        print(json.dumps(stage), flush=True)
    receipt.update(status="complete" if len(receipt["stages"]) == 3 and all(s["returncode"] == 0 for s in receipt["stages"])
                   else "partial_failure", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(receipt_path, receipt)
    return 0 if receipt["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
