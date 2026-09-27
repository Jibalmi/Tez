"""Bounded follow-up run: registered comparators, validation-selected encoder, and merge parity."""
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
    spec_path = Path(__file__).with_suffix(".json")
    spec = json.loads(spec_path.read_text())
    deadline = time.monotonic() + spec["budget"]["compute_deadline_minutes_after_setup"] * 60
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "stages": [],
              "spec_sha256": file_sha256(spec_path)}
    dump(RESULTS / "registration.json", spec)
    from huggingface_hub import snapshot_download
    enc = spec["encoder"]
    backbone = snapshot_download(enc["backbone"], revision=enc["revision"],
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    stages = [("eval_qwen08-frozen", "eval_arms.py", ["--arm", "qwen08-frozen"], 300),
              ("eval_gliclass-multilang-mini", "eval_arms.py", ["--arm", "gliclass-multilang-mini"], 240),
              ("merge_parity", "merge_parity.py", [], 180),
              ("train_a-base-12ep", "train_a.py", ["--arm", enc["arm"], "--backbone", backbone,
               "--epochs", str(enc["epochs"]), "--select-best-val", "--minutes", "13"], 820),
              ("eval_a-base-12ep", "eval_arms.py", ["--arm", enc["arm"]], 180),
              ("eval_gliclass-instruct-base-v1.0", "eval_arms.py", ["--arm", "gliclass-instruct-base-v1.0"], 240),
              ("eval_gliclass-instruct-edge-v1.0", "eval_arms.py", ["--arm", "gliclass-instruct-edge-v1.0"], 180),
              ("eval_kev08-native", "eval_arms.py", ["--arm", "kev08-native"], 420)]
    for arm in ("c-qwen08-30k", "c-qwen08-merged", "a-base-12ep", "gliclass-multilang-mini", "gliclass-instruct-edge-v1.0"):
        stages.append(("latency_" + arm, "latency.py", ["--arm", arm, "--warmup", "5", "--reps", "40", "--reps50", "15"], 120))
    failed_training = False
    for name, script, args, cap in stages:
        if failed_training and name in ("eval_a-base-12ep", "latency_a-base-12ep"):
            continue
        remaining = deadline - time.monotonic()
        if remaining < 45:
            report["status"] = "time_limit"
            break
        env = dict(os.environ)
        if name.startswith("latency_"):
            env["TEZ_PILOT_RESULTS"] = str(RESULTS / "cloud-latency")
            (RESULTS / "cloud-latency").mkdir(exist_ok=True)
        started = time.monotonic()
        print("START", name, flush=True)
        with (RESULTS / (name + ".log")).open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name(script)), *args],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(cap, remaining - 15))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        failed_training |= name == "train_a-base-12ep" and code != 0
        row = {"name": name, "returncode": code, "seconds": time.monotonic() - started}
        report["stages"].append(row)
        dump(RESULTS / "execution.json", report)
        print("END", row, flush=True)
    else:
        report["status"] = "complete" if all(s["returncode"] == 0 for s in report["stages"]) else "partial_failure"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(RESULTS / "execution.json", report)
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
