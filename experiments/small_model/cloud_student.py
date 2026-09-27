"""Snapshot public pilot inputs and run bounded, isolated student training on a Linux GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from common import ROOT, RESULTS, RUNS, dump, file_sha256


def prepare(output: Path):
    if output.exists():
        raise ValueError("Preparation snapshots must use a new directory")
    output.mkdir(parents=True)
    from eval_arms import suites
    suite_file = output / "eval-suites.json"
    cases = suites()
    dump(suite_file, cases)
    # Only code, public research inputs, frozen teacher labels, and reference measurements.
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "tez", "experiments", "data",
                                       "pyproject.toml", "README.md", "LICENSE"], cwd=ROOT).decode().split("\0")
    files = {p: ROOT / p for p in tracked if p and (ROOT / p).is_file()}
    for p in Path(__file__).parent.glob("*.py"):
        files[p.relative_to(ROOT).as_posix()] = p
    p = Path(__file__).with_name("cloud_student_bootstrap.sh")
    files[p.relative_to(ROOT).as_posix()] = p
    for p in RUNS.rglob("*"):
        if p.is_file() and (p.name in {"pool.jsonl", "assign.jsonl", "manifest.json"} or p.parent.name == "teacher"):
            files["runs/student-data/" + p.relative_to(RUNS).as_posix()] = p
    for directory in (ROOT / "results/h2h", ROOT / "results/vs_laya"):
        for p in directory.rglob("*.jsonl"):
            files[p.relative_to(ROOT).as_posix()] = p
    for name in ("authored144_gemma4-12b-q8_0.jsonl", "perturb108_gemma4-12b-q8_0.jsonl"):
        p = ROOT / "results" / name
        files[p.relative_to(ROOT).as_posix()] = p
    files["runs/student-input/eval-suites.json"] = suite_file
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source_dataset_manifest_sha256": file_sha256(RUNS / "manifest.json"),
                "evaluation_counts": {k: len(v) for k, v in cases.items()},
                "files": {n: file_sha256(p) for n, p in sorted(files.items())}}
    dump(output / "manifest.json", manifest)
    files["runs/student-input/manifest.json"] = output / "manifest.json"
    bundle = output / "student.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:
        for n, p in sorted(files.items()):
            archive.add(p, arcname=n, recursive=False)
    record = {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": file_sha256(bundle),
              "evaluation_counts": manifest["evaluation_counts"]}
    dump(output / "bundle.json", record)
    print(json.dumps(record), flush=True)


def run(minutes: float):
    manifest = json.loads((ROOT / "runs/student-input/manifest.json").read_text())
    for name, digest in manifest["files"].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or file_sha256(p) != digest:
            raise ValueError(f"Input checksum mismatch: {name}")
    deadline = time.monotonic() + minutes * 60
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "input_manifest_sha256": file_sha256(ROOT / "runs/student-input/manifest.json"),
              "stages": [], "deadline_minutes": minutes}
    dump(RESULTS / "execution.json", report)
    arms = [("a-base-30k", "jhu-clsp/mmBERT-base", "30k", "3e-5"),
            ("a-base-10k", "jhu-clsp/mmBERT-base", "10k", "3e-5"),
            ("a-small-30k", "jhu-clsp/mmBERT-small", "30k", "5e-5")]
    stages = []
    for arm, backbone, data, lr in arms:
        stages += [("train_" + arm, "train_a.py", ["--arm", arm, "--backbone", backbone, "--data", data, "--lr", lr]),
                   ("eval_" + arm, "eval_arms.py", ["--arm", arm])]
    stages += [("train_c-qwen08-30k", "train_c.py", ["--arm", "c-qwen08-30k", "--max-segments", "99"]),
               ("eval_c-qwen08-30k", "eval_arms.py", ["--arm", "c-qwen08-30k"]),
               ("eval_gliclass-x-base", "eval_arms.py", ["--arm", "gliclass-x-base"])]
    for name, script, arguments in stages:
        if (RESULTS / (name + ".json")).exists():
            continue
        left = deadline - time.monotonic()
        if left < 180:
            report["status"] = "checkpointed_time_limit"
            break
        command = [sys.executable, str(Path(__file__).with_name(script)), *arguments]
        if name.startswith("train_"):
            command += ["--minutes", str(max(1, (left - 150) / 60))]
        start = time.monotonic()
        print("START", name, flush=True)
        with (RESULTS / (name + ".log")).open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = process.wait(timeout=max(1, left - 30))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                rc = 124
        report["stages"].append({"name": name, "returncode": rc, "seconds": time.monotonic() - start})
        dump(RESULTS / "execution.json", report)
        print("END", name, rc, flush=True)
        if rc:
            report["status"] = "checkpointed_time_limit" if rc == 3 else "failed"
            break
    else:
        report["status"] = "training_and_evaluation_complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(RESULTS / "execution.json", report)
    return 0 if report["status"] == "training_and_evaluation_complete" else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--minutes", type=float, default=50)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        raise SystemExit(run(args.minutes))
