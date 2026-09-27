"""Record one read-only sensor check; this is not a model or latency benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

from windows_gpu_telemetry import NvidiaTelemetry, read_windows_counters, selected_memory, thermal_eligible


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output / "executed_source"
    sources.mkdir()
    for name in (Path(__file__).name, "windows_gpu_telemetry.py", "test_windows_gpu_telemetry.py"):
        shutil.copyfile(Path(__file__).with_name(name), sources / name)
    result = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(), "own_pid": os.getpid(),
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "argv": sys.argv,
              "scope": "One read-only sensor snapshot. No model, tensor, latency, GPU settings or workload change. "
                       "Other workloads may be running; these observations do not certify any timed inference run."}
    monitor = None
    error = None
    try:
        monitor = NvidiaTelemetry(args.device)
        result["identity"] = monitor.identity
        result["nvml_library"] = monitor.library._name
        result["before"] = monitor.sample()
        raw = read_windows_counters()
        save(args.output / "windows_counters.json", raw)
        result["after"] = monitor.sample()
        result["memory"] = selected_memory(raw, monitor.identity, os.getpid())
        result["snapshot_start_thermal_eligible"] = thermal_eligible(result["before"], maximum_c=65)
        result["snapshot_during_thermal_eligible"] = all(thermal_eligible(result[k]) for k in ("before", "after"))
        result["status"] = "complete"
    except Exception as exc:
        error = exc
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                error = exc
                result["status"] = "failed"
                result["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "probe.json", result)
        files = [{"path": p.relative_to(args.output).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(args.output.rglob("*")) if p.is_file()]
        for row in files:
            path = args.output / row["path"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Snapshot changed while making the evidence manifest")
        save(args.output / "manifest.json", {"status": "complete", "files_verified": len(files), "files": files})
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "snapshot_memory_contaminated": result.get("memory", {}).get("contaminated"),
                      "snapshot_start_thermal_eligible": result.get("snapshot_start_thermal_eligible"),
                      "error": result.get("error", {}).get("message")}))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
