"""Time-bounded native competitor measurements after the other registered GPU work."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

from common import ROOT, RESULTS, dump, file_sha256


def main():
    deadline = 1790470564.6581569
    spec_path = Path(__file__).with_name("decider_registration.json")
    spec = json.loads(spec_path.read_text())
    target = RESULTS / "decider_execution.json"
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting", "stages": [],
              "registration_sha256": file_sha256(spec_path), "stop_deadline_unix": deadline}
    dump(target, report)
    while time.time() < deadline - 60:
        p = RESULTS / "cache_execution.json"
        if p.exists() and json.loads(p.read_text())["status"] not in ("running", "waiting"):
            break
        time.sleep(5)
    else:
        report["status"] = "time_limit"
        dump(target, report)
        return
    report["status"] = "running"
    dump(target, report)
    request = urllib.request.Request("https://codeload.github.com/Mapika/decider/tar.gz/" + spec["code_commit"],
                                     headers={"User-Agent": "tez-research/0.1"})
    with urllib.request.urlopen(request, timeout=min(60, deadline - time.time())) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    source = ROOT / "runs/research/decider"
    source.mkdir(parents=True, exist_ok=True)
    for member in archive.getmembers():
        parts = Path(member.name).parts[1:]
        if parts:
            member.name = "/".join(parts)
            archive.extract(member, source, filter="data")
    (ROOT / "runs/research/decider-commit.txt").write_text(spec["code_commit"] + "\n")
    stages = [("eval_decider08-native", "eval_arms.py", ["--arm", "decider08-native"], 330)]
    for arm in ("kev08-native", "decider08-native"):
        stages.append(("latency_" + arm, "latency.py", ["--arm", arm, "--warmup", "5", "--reps", "40", "--reps50", "15"], 90))
    for name, script, args, cap in stages:
        left = deadline - time.time()
        if left < 45:
            report["status"] = "time_limit"
            break
        env = dict(os.environ)
        if name.startswith("latency_"):
            env["TEZ_PILOT_RESULTS"] = str(RESULTS / "cloud-latency")
        started = time.monotonic()
        with (RESULTS / (name + ".log")).open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name(script)), *args],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(cap, left - 5))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
        report["stages"].append({"name": name, "returncode": code, "seconds": time.monotonic() - started})
        dump(target, report)
    else:
        report["status"] = "complete" if all(s["returncode"] == 0 for s in report["stages"]) else "partial_failure"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(target, report)


if __name__ == "__main__":
    main()
