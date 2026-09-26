"""Prepare a portable snapshot, verify cloud teacher parity, and time isolated GPU workers.

No provider credentials or paid resources are created by this script. The run command is for
an exclusive Linux GPU Pod. Every worker writes its own directory; laptop shards are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

from common import ROOT, RUNS, RESULTS, read_jsonl, write_jsonl
from dataset import softmax, temp_for

MODEL_SHA = "047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
MODEL_URL = f"https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:{MODEL_SHA}"


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(directory: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(directory.rglob("part-*.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in rows:
                raise ValueError(f"Duplicate teacher state: {row['id']}")
            rows[row["id"]] = row
    return rows


def parity(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    """Compare the actual temperature-adjusted training targets, preserving option alignment."""
    if set(reference) != set(candidate):
        raise ValueError("Canary state IDs differ or outputs are incomplete")
    max_delta = total_tv = 0.0
    count = agreed = 0
    for sid, expected in reference.items():
        actual = candidate[sid]
        if len(expected["questions"]) != len(actual["questions"]):
            raise ValueError("Question counts differ")
        for eq, aq in zip(expected["questions"], actual["questions"]):
            for field in ("qid", "type", "keys", "abstain"):
                if eq[field] != aq[field]:
                    raise ValueError(f"Question metadata differs: {sid}/{field}")
            targets = []
            for q in (eq, aq):
                z1, z2 = np.asarray(q["z_orig"], float), np.asarray(q["z_rev"], float)
                if z1.shape != (len(q["keys"]),) or z2.shape != z1.shape or not np.isfinite([z1, z2]).all():
                    raise ValueError("Invalid logits")
                temperature = temp_for(q["type"], len(q["keys"]))
                targets.append((softmax(z1, temperature) + softmax(z2, temperature)[::-1]) / 2)
            delta = np.abs(targets[0] - targets[1])
            max_delta = max(max_delta, float(delta.max()))
            total_tv += float(delta.sum() / 2)
            agreed += int(np.argmax(targets[0]) == np.argmax(targets[1]))
            count += 1
    if not count:
        raise ValueError("Empty parity sample")
    result = {"decisions": count, "max_probability_delta": max_delta,
              "mean_total_variation": total_tv / count, "argmax_agreement": agreed / count,
              "thresholds": {"max_probability_delta": 0.02, "mean_total_variation": 0.005,
                             "min_argmax_agreement": 0.99}}
    result["passed"] = max_delta <= 0.02 and total_tv / count <= 0.005 and agreed / count >= 0.99
    return result


def prepare(output: Path, canary_states: int) -> None:
    if output.exists():
        raise ValueError("Choose a new preparation directory; existing snapshots are immutable")
    output.mkdir(parents=True)
    reads = collect(RUNS / "teacher")
    assigned = read_jsonl(RUNS / "assign.jsonl")
    # Hash sampling avoids depending on which source appears first in assign.jsonl.
    ids = sorted(reads, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:canary_states]
    if len(ids) != canary_states:
        raise ValueError("Not enough saved states for the canary")
    write_jsonl(output / "canary-reference.jsonl", [reads[s] for s in ids])
    (output / "canary.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
    pending = [a["id"] for a in assigned if a["id"] not in reads]
    (output / "pending.ids").write_text("\n".join(pending) + "\n", encoding="utf-8")
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "tez", "experiments", "data",
                                       "pyproject.toml", "README.md", "LICENSE", "LICENSES", "THIRD_PARTY_NOTICES.md"],
                                      cwd=ROOT).decode().split("\0")
    files = {Path(name): ROOT / name for name in tracked if name and (ROOT / name).is_file()}
    # Include new preparation scripts before they have been committed.
    for path in Path(__file__).parent.glob("cloud_*"):
        if path.is_file():
            files[path.relative_to(ROOT)] = path
    for name in ("pool.jsonl", "assign.jsonl"):
        files[Path("runs/pilot") / name] = RUNS / name
    # Preserve all finished work for a later merge/training step, from the same frozen ID set.
    saved_rows = list(reads.values())
    snapshot = output / "saved-teacher"
    snapshot.mkdir()
    for index in range(0, len(saved_rows), 200):
        name = f"part-{index // 200:04d}.jsonl"
        path = snapshot / name
        write_jsonl(path, saved_rows[index:index + 200])
        files[Path("runs/pilot/teacher") / name] = path
    # Reference outputs are kept outside the production output tree.
    for name in ("canary-reference.jsonl", "canary.ids", "pending.ids"):
        files[Path("runs/cloud-input") / name] = output / name
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": MODEL_SHA,
                "model_download_url": MODEL_URL, "llama_cpp_build": "b11100", "saved_states": len(reads),
                "pending_states": len(pending), "canary_states": len(ids),
                "files": {str(name).replace("\\", "/"): sha(path) for name, path in sorted(files.items())}}
    save(output / "manifest.json", manifest)
    files[Path("runs/cloud-input/manifest.json")] = output / "manifest.json"
    bundle = output / "tez-pilot-cloud.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:
        for name, path in files.items():
            archive.add(path, arcname=name.as_posix(), recursive=False)
    save(output / "bundle.json", {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": sha(bundle)})
    print(json.dumps({"bundle": str(bundle), "bytes": bundle.stat().st_size,
                      "saved_states": len(reads), "pending_states": len(pending), "canary_states": len(ids)}))


def verify_inputs() -> dict:
    manifest = json.loads((ROOT / "runs/cloud-input/manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    return manifest


def run_group(base: Path, ids_file: Path, workers: int, minutes: float, deadline: float) -> dict:
    if base.exists():
        raise ValueError(f"Output already exists: {base}")
    processes, logs = [], []
    started = time.monotonic()
    try:
        for index in range(workers):
            directory = base / f"worker-{index}"
            directory.mkdir(parents=True)
            for name in ("pool.jsonl", "assign.jsonl"):
                (directory / name).symlink_to(ROOT / "runs/pilot" / name)
            env = dict(os.environ, TEZ_PILOT_RUNS=str(directory), TEZ_PILOT_RESULTS=str(directory / "results"))
            log = (directory / "teacher.log").open("w", encoding="utf-8")
            logs.append(log)
            command = [sys.executable, str(Path(__file__).with_name("teacher.py")), "--ids-file", str(ids_file),
                       "--num-shards", str(workers), "--shard-index", str(index), "--minutes", str(minutes)]
            processes.append(subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("Cloud compute deadline reached")
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A cloud teacher worker failed or checkpointed for cooling; see worker logs")
            time.sleep(1)
        codes = [p.returncode for p in processes]
        if any(codes):
            raise RuntimeError(f"Cloud worker return codes: {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    elapsed = time.monotonic() - started
    rows = collect(base)
    expected = set(ids_file.read_text(encoding="utf-8").splitlines())
    result = {"workers": workers, "states": len(rows), "expected_states": len(expected),
              "complete": set(rows) == expected, "wall_seconds": elapsed,
              "states_per_second": len(rows) / elapsed, "outputs": str(base.relative_to(ROOT)),
              "timing_scope": "Process startup, model loading, inference, cooling, checkpoints, and exit"}
    save(base / "metrics.json", result)
    return result


def run(args) -> None:
    if os.name == "nt":
        raise ValueError("Cloud runs require an exclusive Linux GPU Pod; do not run beside the laptop teacher")
    manifest = verify_inputs()
    output = ROOT / "runs/cloud-output" / args.name
    if output.exists():
        raise ValueError("Choose a new run name")
    output.mkdir(parents=True)
    deadline = time.monotonic() + args.minutes * 60
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True)
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
    if len(memory.strip().splitlines()) != 1:
        raise ValueError("This runner expects a Pod with exactly one visible GPU")
    max_workers = max(1, int(float(memory.strip()) / 1024 // 18))
    worker_counts = [n for n in args.workers if n <= max_workers]
    if not worker_counts:
        raise ValueError("Insufficient GPU memory for the requested worker counts")
    reference = {r["id"]: r for r in read_jsonl(ROOT / "runs/cloud-input/canary-reference.jsonl")}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "manifest": manifest,
              "trials": [], "status": "benchmarking", "deadline_minutes": args.minutes,
              "requested_workers": args.workers, "memory_safe_workers": worker_counts}
    save(output / "report.json", report)
    for workers in worker_counts:
        result = run_group(output / f"canary-{workers}", ROOT / "runs/cloud-input/canary.ids", workers, 10, deadline)
        result["parity"] = parity(reference, collect(output / f"canary-{workers}"))
        report["trials"].append(result)
        save(output / "report.json", report)
        print(json.dumps(result), flush=True)
        if not result["parity"]["passed"]:
            raise ValueError("Cloud canary targets changed beyond the fixed parity thresholds")
    best = max(report["trials"], key=lambda r: r["states_per_second"])
    report["selected_workers"] = best["workers"]
    report["remaining_teacher_eta_seconds_estimate"] = manifest["pending_states"] / best["states_per_second"]
    report["eta_note"] = "Estimate extrapolated from the canary including loading; excludes provisioning/download/setup and student training"
    report["status"] = "benchmark_complete"
    save(output / "report.json", report)
    if args.production:
        minutes_left = (deadline - time.monotonic()) / 60 - 1
        if minutes_left <= 0:
            raise TimeoutError("No time left in the cloud compute budget")
        report["production"] = run_group(output / "production", ROOT / "runs/cloud-input/pending.ids",
                                          best["workers"], minutes_left, deadline)
        report["status"] = "complete" if report["production"]["complete"] else "checkpointed"
        save(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("manifest", "trials")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--canary-states", type=int, default=100)
    runner = sub.add_parser("run")
    runner.add_argument("--name", default="h100-trial")
    runner.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    runner.add_argument("--minutes", type=float, default=45)
    runner.add_argument("--production", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve(), args.canary_states)
    else:
        if args.minutes <= 0 or not args.workers or any(n not in (1, 2, 4) for n in args.workers):
            parser.error("Use positive minutes and worker counts from 1,2,4")
        run(args)


if __name__ == "__main__":
    main()
