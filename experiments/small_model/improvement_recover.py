"""Retry the registered encoder after a missing-weight download, once the original run exits."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, RESULTS, dump, file_sha256


def main():
    # Pod stop minus two minutes reserved for export; never extends the paid trial.
    deadline = 1790470564.6581569
    spec = json.loads(Path(__file__).with_name("improvement_round1.json").read_text())
    enc = spec["encoder"]
    from huggingface_hub import snapshot_download
    backbone = snapshot_download(enc["backbone"], revision=enc["revision"],
                                 allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"])
    source = {"model": enc["backbone"], "revision": enc["revision"],
              "weights": {p.name: file_sha256(p) for p in Path(backbone).glob("*.bin")}}
    dump(RESULTS / "recovery_backbone_source.json", source)
    while time.time() < deadline:
        p = RESULTS / "execution.json"
        if p.exists() and json.loads(p.read_text())["status"] != "running":
            break
        time.sleep(5)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": [],
              "reason": "Original training failed before gradients: pinned model uses .bin, omitted from download filter",
              "selection": enc["selection"], "stop_deadline_unix": deadline}
    minutes = min(13, max(0, (deadline - time.time() - 150) / 60))
    stages = [("train_a-base-12ep_retry", "train_a.py", ["--arm", enc["arm"], "--backbone", backbone,
               "--epochs", str(enc["epochs"]), "--select-best-val", "--minutes", str(minutes)], minutes * 60 + 35),
              ("eval_a-base-12ep", "eval_arms.py", ["--arm", enc["arm"]], 150),
              ("latency_a-base-12ep", "latency.py", ["--arm", enc["arm"], "--warmup", "5", "--reps", "40", "--reps50", "15"], 90)]
    for name, script, args, cap in stages:
        remaining = deadline - time.time()
        if remaining < 45 or (name.startswith("train_") and minutes < 1):
            report["status"] = "time_limit"
            break
        env = dict(os.environ)
        if name.startswith("latency_"):
            env["TEZ_PILOT_RESULTS"] = str(RESULTS / "cloud-latency")
        started = time.monotonic()
        print("START", name, flush=True)
        with (RESULTS / (name + ".log")).open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name(script)), *args],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(cap, remaining - 5))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        row = {"name": name, "returncode": code, "seconds": time.monotonic() - started}
        report["stages"].append(row)
        dump(RESULTS / "recovery_execution.json", report)
        print("END", row, flush=True)
        if name.startswith("train_") and code != 0:
            report["status"] = "training_failed"
            break
    else:
        report["status"] = "complete" if all(s["returncode"] == 0 for s in report["stages"]) else "partial_failure"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(RESULTS / "recovery_execution.json", report)


if __name__ == "__main__":
    main()
