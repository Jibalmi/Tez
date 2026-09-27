"""Bounded complete native Tev1 comparison on the frozen public suites, without fitting."""
from datetime import datetime, timezone
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import shutil
from statistics import fmean
import subprocess
import sys
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-tev1-eval-20260927"
REGISTRATION = Path(__file__).with_name("tev1_registration.json")


def inputs(spec):
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input or prerequisite changed")
    prep = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
    probe = json.loads((ROOT / spec["probe"]).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / spec["suite"]).read_text(encoding="utf-8"))
    if (prep["status"] != "staged" or not prep["baseline_unchanged"] or probe["status"] != "complete"
            or not probe["passed"] or probe["cases_verified"] != 4 or len(suite) != 21
            or sum(map(len, suite.values())) != 6151 or prep["revision"] != probe["revision"]):
        raise ValueError("Incomplete preparation, native probe or suite")
    return prep, probe, suite


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    prep, probe, suite = inputs(spec)
    import torch
    from common import THERMAL, thermal_guard
    from eval_gliner2 import raw_metrics
    from tev1_adapter import NativeTev
    path = DIRECTORY / "evaluation.json"
    if path.exists():
        raise ValueError("Preserve previous evaluation")
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}, "tasks": {},
               "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
               "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "huggingface-hub")}}
    started = time.monotonic()
    write_json(path, receipt)
    try:
        for relative, expected in spec["inference_sources"].items():
            source = ROOT / relative
            if digest(source) != expected:
                raise ValueError("Registered inference source changed")
            destination = DIRECTORY / "executed_source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            receipt["source_files"][relative] = expected
        model_path = ROOT / prep["directory"]
        for name, entry in prep["files"].items():
            if digest(model_path / name) != entry["sha256"]:
                raise ValueError("Frozen model file changed")
        receipt.update(model=prep["repository"], revision=prep["revision"],
                       weights={name: entry["sha256"] for name, entry in prep["files"].items() if name.endswith(".safetensors")},
                       preparation_sha256=digest(ROOT / spec["preparation"]), probe_sha256=digest(ROOT / spec["probe"]))
        thermal_guard(force=True)
        predictor = NativeTev(model_path, ROOT / spec["native_helper"], prep["repository"], spec["context_bound"])
        smoke = predictor.score_cases(probe["cases"])
        delta = max(abs(p - q) for row, old in zip(smoke, probe["rows"], strict=True)
                    for p, q in zip(row["probabilities"], old["probabilities"], strict=True))
        agreement = all(max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__) == old["pred"]
                        for row, old in zip(smoke, probe["rows"], strict=True))
        receipt["smoke"] = {"max_probability_delta": delta, "all_native_labels_match": agreement}
        if len(smoke) != 4 or not agreement or delta > 1e-5:
            raise ValueError("Adapter changed frozen single-case native probe probabilities")
        receipt["status"] = "running"
        write_json(path, receipt)

        def run_cases(task, cases, suffix=""):
            destination = DIRECTORY / f"rows_{task}{suffix}.jsonl"
            rows = []
            with destination.open("x", encoding="utf-8") as output:
                for case in cases:
                    if time.monotonic() - started > spec["checkpoint_deadline_seconds"]:
                        raise TimeoutError("Bounded evaluation deadline reached")
                    safe = {key: case[key] for key in ("state", "instructions", "options")}
                    value = predictor.score_cases([safe])[0]
                    p = value["probabilities"]
                    if len(p) != len(case["options"]):
                        raise ValueError("Probability membership changed")
                    row = {key: case.get(key) for key in ("id", "lang", "qtype", "gold", "workflow", "group", "family")}
                    row.update(task=task, k=len(p), extra=case.get("extra", {}), **value,
                               pred=max(range(len(p)), key=p.__getitem__), option_order="reversed" if suffix else "original")
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    if len(rows) % 32 == 0 or len(rows) == len(cases):
                        receipt["current"] = {"task": task + suffix, "completed_rows": len(rows)}
                        write_json(path, receipt)
            return rows, {"sha256": digest(destination), "rows": len(rows)}

        for task, cases in suite.items():
            rows, row_file = run_cases(task, cases)
            metrics = raw_metrics(rows)
            metrics["option_index_macro_f1_diagnostic"] = metrics.pop("macro_f1")
            measured = {"t1": metrics, "row_file": row_file}
            if task.startswith("massive_"):
                reverse = [{**c, "options": list(reversed(c["options"])), "gold": len(c["options"]) - 1 - c["gold"]} for c in cases]
                reversed_rows, reversed_file = run_cases(task, reverse, "_reversed")
                measured["order_flip"] = fmean(a["pred"] != a["k"] - 1 - b["pred"] for a, b in zip(rows, reversed_rows, strict=True))
                measured["reversed_row_file"] = reversed_file
            if task.startswith("semif_"):
                module_spec = importlib.util.spec_from_file_location("semif_frozen_evaluator", ROOT / "data/semif/evaluate.py")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                data = module.read_jsonl(ROOT / "data/semif" / (task.split("_", 1)[1] + ".jsonl"))
                measured["semif"] = module.evaluate(data, [{"id": r["id"], "probabilities": r["probabilities"]} for r in rows])
            receipt["tasks"][task] = measured
            write_json(path, receipt)
            print(json.dumps({"task": task, "accuracy": metrics["accuracy"]}), flush=True)
        receipt.update(status="complete", primary_decisions=6151, reversed_decisions=1100, input_audit=predictor.audit)
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_governor=THERMAL)
        write_json(path, receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    inputs(spec)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve previous execution")
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
               TEZ_PILOT_RESULTS=str(DIRECTORY), TOKENIZERS_PARALLELISM="false")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(path, receipt)
    with (DIRECTORY / "evaluation.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
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
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if "--worker" in sys.argv:
        worker()
    else:
        raise SystemExit(driver())
