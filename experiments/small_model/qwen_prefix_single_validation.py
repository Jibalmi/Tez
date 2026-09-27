"""Run the unchanged native-prefix batch-one arm with a separately registered larger time budget."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "results/pilot-qwen-prefix-validation-20260927"
DIRECTORY = ROOT / "results/pilot-qwen-prefix-single-20260927"
REGISTRATION = Path(__file__).with_name("qwen_prefix_single_registration.json")
PARENT_REGISTRATION = Path(__file__).with_name("qwen_prefix_validation_registration.json")


def extended_spec(parent, verified, incomplete):
    if (verified.get("status") != "complete" or not verified.get("passed") or verified.get("batch_size") != 16
            or verified.get("real_predictions_per_arm") != 1390 or incomplete.get("status") != "failed"
            or incomplete.get("error_type") != "TimeoutError" or incomplete.get("batch_size") != 1
            or not 0 < len(incomplete.get("batches", [])) < len(parent["plans"]["1"])
            or incomplete["elapsed_seconds"] < parent["soft_deadline_seconds"]):
        raise ValueError("Require complete batch16 prefix parity and the retained incomplete original batch1 trial")
    result = copy.deepcopy(parent)
    result.update(batch_sizes=[1], plans={"1": result["plans"]["1"]}, soft_deadline_seconds=1080, worker_timeout_seconds=1200)
    return result


def register():
    if REGISTRATION.exists() or DIRECTORY.exists() or (PARENT / "batch-01").exists():
        raise ValueError("Preserve every registered or executed single-question arm")
    parent = json.loads(PARENT_REGISTRATION.read_text(encoding="utf-8"))
    failure = ROOT / "results/pilot-qwen-graph-validation-20260927/batch-01"
    verified = json.loads((PARENT / "batch-16/verification.json").read_text(encoding="utf-8"))
    incomplete = json.loads((failure / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((failure / "execution.json").read_text(encoding="utf-8"))
    if (verified["registration_sha256"] != digest(PARENT_REGISTRATION)
            or incomplete["registration_sha256"] != digest(failure / "registration.json")
            or execution["status"] != "failed" or execution["returncode"] != 1):
        raise ValueError("Require the matching terminal prerequisite protocols")
    spec = extended_spec(parent, verified, incomplete)
    for directory, filename in ((PARENT / "batch-16", "evidence.tar.gz"), (failure, "failed-artifacts.tar.gz")):
        artifact = json.loads((directory / "artifact_receipt.json").read_text(encoding="utf-8"))
        if digest(directory / filename) != artifact["sha256"]:
            raise ValueError("Prerequisite graph evidence changed")
    for name in ("qwen_prefix_single_validation.py", "test_qwen_prefix_single_validation.py"):
        path = Path(__file__).with_name(name)
        spec["sources"][path.relative_to(ROOT).as_posix()] = digest(path)
    for path in (PARENT_REGISTRATION, PARENT / "batch-16/verification.json", PARENT / "batch-16/artifact_receipt.json",
                 failure / "work.json", failure / "artifact_receipt.json"):
        spec["prerequisites"][path.relative_to(ROOT).as_posix()] = digest(path)
    for name, expected in {**spec["sources"], **spec["prerequisites"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prefix source or prerequisite changed")
    spec.update(status="registered_before_first_prefix_batch1_execution", registered_utc=datetime.now(timezone.utc).isoformat(),
                parent_registration_sha256=digest(PARENT_REGISTRATION),
                resource_amendment="The earlier left-padding batch1 trial ended at its720-second soft deadline with1348/1390 predictions; no prefix batch1 inference has run. Give the unchanged prefix batch1 arm1080seconds soft/1200seconds hard, within a22-minute shared lease. Inputs, all1390 decisions, native batch plan, frozen full-head batch16 reference, source, weights, math, warmups, temperatures, memory and parity thresholds remain unchanged. No partial rows are reused. Preserve the prior registration and failure. This replaces only the unexecuted prefix batch1 resource plan; it does not rerun the failed left-padding candidate.",
                compute="Shared run_heavy, four threads,22-minute lease,18-minute soft limit,20-minute hard child timeout,80/76C process pacing and84C wrapper. One graph resident. No paid resources.")
    DIRECTORY.mkdir()
    shutil.copy2(PARENT / "preflight.json", DIRECTORY / "preflight.json")
    shutil.copy2(PARENT / ".gitignore", DIRECTORY / ".gitignore")
    spec["prerequisites"][(DIRECTORY / "preflight.json").relative_to(ROOT).as_posix()] = digest(DIRECTORY / "preflight.json")
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "batch_sizes": spec["batch_sizes"],
                      "soft_deadline_seconds": spec["soft_deadline_seconds"], "worker_timeout_seconds": spec["worker_timeout_seconds"]}))


def worker(spec):
    import qwen_prefix_validation as frozen
    # Bind the receipt to this derivative protocol; the already pinned inference implementation is unchanged.
    frozen.REGISTRATION = REGISTRATION
    return frozen.worker(1, DIRECTORY / "batch-01", spec)


def driver(spec):
    import psutil
    for name, expected in {**spec["sources"], **spec["prerequisites"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered prefix input changed before launch")
    directory = DIRECTORY / "batch-01"
    directory.mkdir(exist_ok=False)
    shutil.copy2(REGISTRATION, directory / "registration.json")
    env = dict(os.environ, TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(directory),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_HUB_KERNELS="0",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "batch_size": 1,
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(directory / "execution.json", receipt)
    with (directory / "worker.log").open("x", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT, env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=spec["worker_timeout_seconds"])
        except subprocess.TimeoutExpired:
            for child in psutil.Process(process.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait()
            code = 124
    receipt.update(status="complete" if code in (0, 2) else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    else:
        if "RUN_HEAVY_THREADS" not in os.environ:
            raise RuntimeError("Use C:/temp/run_heavy.py")
        spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        raise SystemExit(worker(spec) if args.worker else driver(spec))
