"""Bounded native GLiNER evaluation on the frozen public suites; no calibration fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from statistics import fmean
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-gliner2-eval-20260927"
REGISTRATION = Path(__file__).with_name("gliner2_registration.json")


def raw_metrics(rows):
    """Report T=1 measurements only; no temperatures or thresholds are fitted."""
    correct = [row["pred"] == row["gold"] for row in rows]
    confidence = [max(row["probabilities"]) for row in rows]
    ece = 0.0
    for index in range(15):
        members = [i for i, value in enumerate(confidence) if index / 15 < value <= (index + 1) / 15]
        if members:
            ece += len(members) / len(rows) * abs(fmean(confidence[i] for i in members) - fmean(correct[i] for i in members))
    result = {"n": len(rows), "accuracy": fmean(correct), "ece": ece,
              "nll": fmean(-math.log(max(row["probabilities"][row["gold"]], 1e-12)) for row in rows),
              "brier": fmean(sum((p - int(i == row["gold"])) ** 2 for i, p in enumerate(row["probabilities"])) for row in rows)}
    labels = sorted({row[key] for row in rows for key in ("gold", "pred")})
    result["macro_f1"] = fmean(2 * sum(row["gold"] == label == row["pred"] for row in rows)
                               / max(1, sum((row["gold"] == label) + (row["pred"] == label) for row in rows))
                               for label in labels)
    soft = [row for row in rows if sum(row.get("extra", {}).get("soft", [])) > 0]
    if soft:
        result["soft_accuracy"] = fmean(sum(p * target for p, target in zip(row["probabilities"], row["extra"]["soft"]))
                                        / sum(row["extra"]["soft"]) for row in soft)
    return result


def inputs_and_probe(arm, spec):
    preparation_path = ROOT / spec["preparation"]
    if digest(preparation_path) != spec["preparation_sha256"]:
        raise ValueError("Preparation changed")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation["status"] != "staged" or not preparation["baseline_unchanged"]:
        raise ValueError("Complete isolated preparation first")
    selected = spec["models"][arm]
    probe_path = ROOT / selected["probe"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if digest(probe_path) != selected["probe_sha256"]:
        raise ValueError("Frozen probe receipt changed")
    if (probe["status"] != "complete" or probe["cases_verified"] != 4 or not probe["passed"]
            or probe["source_sha256"] != selected["probe_source_sha256"]
            or probe["preparation_sha256"] != digest(preparation_path)
            or probe["revision"] != preparation["models"][selected["repository"]]["revision"]):
        raise ValueError("Complete the frozen native interface probe first")
    suite_path = ROOT / spec["suite"]
    if digest(suite_path) != spec["suite_sha256"]:
        raise ValueError("Public suite changed")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if len(suite) != 21 or sum(map(len, suite.values())) != 6151:
        raise ValueError("Unexpected public suite membership")
    return preparation, probe, suite


def worker(arm):
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    preparation, probe, suite = inputs_and_probe(arm, spec)
    os.environ.update(TEZ_PILOT_RESULTS=str(DIRECTORY), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from common import THERMAL, thermal_guard
    from gliner2_adapter import NativeGLiner
    from importlib.metadata import version

    folder = DIRECTORY / arm
    folder.mkdir(exist_ok=False)
    path = folder / "evaluation.json"
    receipt = {"status": "preparing", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}, "tasks": {},
               "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
               "packages": {name: version(name) for name in ("torch", "transformers", "gliner2", "tokenizers", "huggingface-hub")}}
    started = time.monotonic()
    write_json(path, receipt)
    try:
        sources = folder / "executed_source"
        sources.mkdir()
        for relative, expected in spec["inference_sources"].items():
            source = ROOT / relative
            if digest(source) != expected:
                raise ValueError("Registered inference source changed")
            target = sources / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            receipt["source_files"][relative] = expected
        shutil.copy2(REGISTRATION, folder / "registration.json")
        selected = spec["models"][arm]
        entry = preparation["models"][selected["repository"]]
        model_path = ROOT / entry["directory"]
        for name, file in entry["files"].items():
            if digest(model_path / name) != file["sha256"]:
                raise ValueError("Frozen model file changed")
        overlay = ROOT / preparation["overlay"]
        for name, expected in preparation["installed_file_hashes"].items():
            if digest(overlay / name) != expected:
                raise ValueError("Isolated dependency changed")
        receipt.update(model=selected["repository"], revision=entry["revision"], weights=entry["files"]["model.safetensors"]["sha256"],
                       preparation_sha256=digest(ROOT / spec["preparation"]), probe_sha256=digest(ROOT / selected["probe"]))
        thermal_guard(force=True)
        predictor = NativeGLiner(model_path, batch_size=spec["batch_size"])
        smoke = predictor.score_inputs(probe["cases"])
        delta = max(abs(p - q) for row, old in zip(smoke, probe["rows"])
                    for p, q in zip(row["probabilities"], old["probabilities"]))
        agreement = all(row["native"]["label"] == old["native"]["label"] for row, old in zip(smoke, probe["rows"]))
        receipt["batch_smoke"] = {"max_probability_delta": delta, "all_native_labels_match": agreement}
        if len(smoke) != 4 or not agreement or delta > 1e-5:
            raise ValueError("Batch interface changed the frozen smoke probabilities")
        receipt["status"] = "running"
        write_json(path, receipt)

        def run_cases(task, cases, suffix=""):
            destination = folder / f"rows_{task}{suffix}.jsonl"
            rows = []
            with destination.open("x", encoding="utf-8") as output:
                for offset in range(0, len(cases), 32):
                    if time.monotonic() - started > spec["checkpoint_deadline_seconds"]:
                        raise TimeoutError("Bounded evaluation deadline reached")
                    chunk = cases[offset:offset + 32]
                    safe = [{key: case[key] for key in ("state", "instructions", "options")} for case in chunk]
                    values = predictor.score_cases(safe)
                    if len(values) != len(chunk):
                        raise ValueError("Incomplete native predictions")
                    for case, value in zip(chunk, values):
                        p = value["probabilities"]
                        if len(p) != len(case["options"]) or abs(sum(p) - 1) > 1e-5:
                            raise ValueError("Invalid probability membership")
                        row = {key: case.get(key) for key in ("id", "lang", "qtype", "gold", "workflow", "group", "family")}
                        row.update(task=task, k=len(p), extra=case.get("extra", {}), **value,
                                   pred=max(range(len(p)), key=p.__getitem__), option_order="reversed" if suffix else "original")
                        rows.append(row)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    receipt["current"] = {"task": task + suffix, "completed_rows": len(rows)}
                    write_json(path, receipt)
            return rows, {"sha256": digest(destination), "rows": len(rows)}

        for task, cases in suite.items():
            rows, row_file = run_cases(task, cases)
            measured = {"t1": raw_metrics(rows), "row_file": row_file}
            if task.startswith("massive_"):
                reversed_cases = [{**case, "options": list(reversed(case["options"])), "gold": len(case["options"]) - 1 - case["gold"]}
                                  for case in cases]
                reverse_rows, reversed_file = run_cases(task, reversed_cases, "_reversed")
                measured["order_flip"] = fmean(a["pred"] != a["k"] - 1 - b["pred"] for a, b in zip(rows, reverse_rows))
                measured["reversed_row_file"] = reversed_file
            if task.startswith("semif_"):
                module_spec = importlib.util.spec_from_file_location("semif_frozen_evaluator", ROOT / "data/semif/evaluate.py")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                data = module.read_jsonl(ROOT / "data/semif" / (task.split("_", 1)[1] + ".jsonl"))
                predictions = [{"id": row["id"], "probabilities": row["probabilities"]} for row in rows]
                measured["semif"] = module.evaluate(data, predictions)
            receipt["tasks"][task] = measured
            write_json(path, receipt)
            print(json.dumps({"arm": arm, "task": task, "accuracy": measured["t1"]["accuracy"]}), flush=True)
        receipt.update(status="complete", primary_decisions=sum(len(cases) for cases in suite.values()),
                       reversed_decisions=1100, input_audit=predictor.audit)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_governor=THERMAL)
        write_json(path, receipt)


def driver(arm):
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    preparation, _, _ = inputs_and_probe(arm, spec)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / f"execution_{arm}.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / preparation["overlay"]) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", USE_TF="0",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(path, receipt)
    with (DIRECTORY / f"{arm}.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--worker"], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=spec["worker_timeout_seconds"])
        except subprocess.TimeoutExpired:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()
            proc.wait()
            code = 124
    receipt.update(status="complete" if code == 0 else "failed", returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("english", "multilingual"), required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if args.worker:
        worker(args.arm)
    else:
        raise SystemExit(driver(args.arm))
