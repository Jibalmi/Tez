"""Authenticated GPU quotes and explicitly bounded RunPod trials; credentials stay local.

No action defaults to allocation. `catalog` and `status` are read-only. `launch` requires an
explicit GPU ID, hourly price ceiling, time limit, and SSH public key. A local watcher stops
the exact created Pod at the deadline; storage remains billable until explicitly removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

IMAGE = "runpod/pytorch@sha256:0ad20c96682da30fd04c2f130c13edb68aa5ebe3d370f2d5096eb1fcbbbfddad"


def token(key_file: Path | None) -> str:
    value = key_file.read_text(encoding="utf-8").strip() if key_file else os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        raise ValueError("Configure RUNPOD_API_KEY locally or provide --key-file; do not paste credentials in chat")
    return value


def request(key: str, method: str, path: str, body: dict | None = None, graphql: bool = False):
    url = "https://api.runpod.io/graphql" if graphql else "https://rest.runpod.io/v1/" + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        # Do not print provider bodies: they can contain account information or echoed inputs.
        raise RuntimeError(f"RunPod {method} {path or 'graphql'} returned HTTP {exc.code}") from None


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_catalog(key: str) -> list[dict]:
    result = request(key, "POST", "", {"query": "query { gpuTypes { id displayName memoryInGb securePrice } }"}, graphql=True)
    if result.get("errors"):
        raise RuntimeError("RunPod rejected the GPU catalog query")
    return result["data"]["gpuTypes"]


def pod_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid Pod ID")
    return value


def watch(key: str, receipt_file: Path) -> None:
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    target = pod_id(receipt["pod_id"])
    while time.time() < receipt["stop_at_unix"]:
        time.sleep(min(15, max(0, receipt["stop_at_unix"] - time.time())))
    # Retry transient stop failures, using only the exact Pod created for this trial.
    for attempt in range(12):
        try:
            request(key, "POST", f"pods/{target}/stop")
            receipt.update(status="stop_requested", stopped_utc=datetime.now(timezone.utc).isoformat())
            save(receipt_file, receipt)
            return
        except (RuntimeError, OSError):
            if attempt == 11:
                receipt.update(status="stop_failed_requires_attention")
                save(receipt_file, receipt)
                raise
            time.sleep(10)


def launch(args, key: str) -> None:
    if args.receipt.exists():
        raise ValueError("Receipt already exists; refusing duplicate allocation")
    gpu = next((g for g in gpu_catalog(key) if g["id"] == args.gpu_id), None)
    if gpu is None or not isinstance(gpu.get("securePrice"), (int, float)):
        raise ValueError("GPU ID or current secure-cloud price is unavailable")
    if not 0 < gpu["securePrice"] <= args.max_hourly_usd:
        raise ValueError("Current GPU price exceeds the explicit hourly ceiling")
    if not 0 < args.minutes <= 60:
        raise ValueError("Trial duration must be within 1..60 minutes")
    public_key = args.public_key.read_text(encoding="utf-8").strip()
    if not public_key.startswith("ssh-ed25519 ") or "\n" in public_key:
        raise ValueError("Supply one dedicated ed25519 SSH public key")
    body = {"name": "tez-pilot-12b-speed-trial", "cloudType": "SECURE", "computeType": "GPU",
            "gpuCount": 1, "gpuTypeIds": [gpu["id"]], "imageName": IMAGE, "interruptible": False,
            "containerDiskInGb": 20, "volumeInGb": 60, "volumeMountPath": "/workspace",
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": public_key}, "minRAMPerGPU": 64,
            "minVCPUPerGPU": 8}
    receipt = {"status": "requesting", "created_utc": datetime.now(timezone.utc).isoformat(),
               "gpu_quote": gpu, "max_gpu_hourly_usd": args.max_hourly_usd, "minutes": args.minutes,
               "stop_at_unix": time.time() + args.minutes * 60,
               "image": IMAGE, "storage_gb": 60, "container_disk_gb": 20,
               "storage_note": "Storage is additional and persists after GPU stop"}
    save(args.receipt, receipt)
    try:
        pod = request(key, "POST", "pods", body)
    except Exception:
        receipt["status"] = "allocation_unconfirmed_check_account_before_retrying"
        save(args.receipt, receipt)
        raise
    receipt.update(pod_id=pod_id(pod["id"]), status="allocated", actual_cost_per_hr=pod.get("costPerHr"))
    save(args.receipt, receipt)
    # Allow only the separately quoted running-storage charge above the GPU ceiling.
    storage_allowance = (body["containerDiskInGb"] + body["volumeInGb"]) * 0.10 / (30 * 24)
    actual = receipt["actual_cost_per_hr"]
    if not isinstance(actual, (int, float)) or not 0 < actual <= args.max_hourly_usd + storage_allowance:
        request(key, "POST", f"pods/{receipt['pod_id']}/stop")
        receipt["status"] = "stopped_unverified_or_excess_price"
        save(args.receipt, receipt)
        raise ValueError("Allocated price could not be verified within the trial ceiling; Pod stopped")
    command = [sys.executable, str(Path(__file__).resolve())]
    if args.key_file:
        command += ["--key-file", str(args.key_file.resolve())]
    command += ["watch", "--receipt", str(args.receipt.resolve())]
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        with args.receipt.with_suffix(".watch.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                       creationflags=flags, start_new_session=os.name != "nt")
        receipt["watcher_pid"] = process.pid
        save(args.receipt, receipt)
    except Exception:
        request(key, "POST", f"pods/{receipt['pod_id']}/stop")
        raise
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("catalog")
    allocate = subs.add_parser("launch")
    allocate.add_argument("--gpu-id", required=True)
    allocate.add_argument("--max-hourly-usd", type=float, required=True)
    allocate.add_argument("--minutes", type=float, required=True)
    allocate.add_argument("--public-key", type=Path, required=True)
    for name in ("watch", "status", "stop"):
        command = subs.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
    allocate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    key = token(args.key_file)
    if args.command == "catalog":
        print(json.dumps(gpu_catalog(key), indent=2))
    elif args.command == "launch":
        launch(args, key)
    elif args.command == "watch":
        watch(key, args.receipt)
    else:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        target = pod_id(receipt["pod_id"])
        if args.command == "stop":
            request(key, "POST", f"pods/{target}/stop")
            print("Stop requested for", target)
        else:
            pod = request(key, "GET", f"pods/{target}")
            print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings", "costPerHr")}, indent=2))


if __name__ == "__main__":
    main()
