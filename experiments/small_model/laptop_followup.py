"""Measure downloaded pilot weights under the machine-wide run_heavy.py compute lock."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[2]
    results = root / "results/pilot-laptop-20260927"
    results.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEZ_PILOT_RUNS=str(root / "runs/student-artifacts-h200-20260927/runs/student-data"),
               TEZ_PILOT_RESULTS=str(results), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": [],
              "weights_source": "runs/student-artifacts-h200-20260927/runs/student-data/models",
              "note": "Separate laptop measurements; do not substitute H200 latency into this record."}
    deadline = time.monotonic() + 18 * 60
    stages = [(arm, "cuda") for arm in ("a-base-30k", "a-small-30k", "c-qwen08-30k")]
    stages += [("a-small-30k", "cpu")]
    for arm, device in stages:
        left = deadline - time.monotonic()
        if left < 60:
            record["status"] = "time_limit"
            break
        cmd = [sys.executable, str(root / "experiments/small_model/latency.py"), "--arm", arm,
               "--device", device, "--threads", "4", "--warmup", "5", "--reps", "40", "--reps50", "15"]
        started = time.monotonic()
        with (results / f"{arm}_{device}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(left, 420))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        record["stages"].append({"arm": arm, "device": device, "returncode": code,
                                  "seconds": time.monotonic() - started})
        (results / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record["stages"][-1]), flush=True)
    else:
        record["status"] = "complete" if all(s["returncode"] == 0 for s in record["stages"]) else "partial_failure"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (results / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0 if record["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
