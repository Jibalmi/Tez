"""Run the registered larger-batch check after the existing GPU work, within its deadline."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, RESULTS, dump


def main():
    deadline = 1790470564.6581569
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting", "stages": []}
    target = RESULTS / "batch64_execution.json"
    dump(target, report)
    while time.time() < deadline - 150:
        p = RESULTS / "decider_execution.json"
        if p.exists() and json.loads(p.read_text())["status"] not in ("running", "waiting"):
            break
        time.sleep(5)
    else:
        report["status"] = "time_limit"
        dump(target, report)
        return
    report["status"] = "running"
    dump(target, report)
    stages = [("batch64_parity", "batch64_parity.py", [], 100),
              ("latency_c-qwen08-batch64", "latency.py", ["--arm", "c-qwen08-batch64", "--warmup", "5",
                                                           "--reps", "40", "--reps50", "15"], 120)]
    for name, script, args, cap in stages:
        left = deadline - time.time()
        if left < 30:
            report["status"] = "time_limit"
            break
        env = dict(os.environ)
        if name.startswith("latency_"):
            env["TEZ_PILOT_RESULTS"] = str(RESULTS / "cloud-latency")
        started = time.monotonic()
        with (RESULTS / (name + ".log")).open("a") as log:
            p = subprocess.Popen([sys.executable, str(Path(__file__).with_name(script)), *args],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=min(cap, left - 5))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                code = 124
        report["stages"].append({"name": name, "returncode": code, "seconds": time.monotonic() - started})
        dump(target, report)
        if code != 0:
            report["status"] = "parity_failed" if name == "batch64_parity" else "latency_failed"
            break
    else:
        report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(target, report)


if __name__ == "__main__":
    main()
