"""Evaluate the frozen rule-continuation trial in bounded sequential laptop workers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from eval_reasoning_confirmation import prerequisites
from structure_probe import WEIGHTS, digest, write_json
from train_reasoning_round2 import ROOT, RESULTS, RUNS


def main():
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    config, trained = prerequisites()
    arms = [config["base_arm"], *config["arms"]]
    suite = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    cases = json.loads(suite.read_text(encoding="utf-8"))
    if len(cases) != 21 or sum(map(len, cases.values())) != 6151:
        raise ValueError("Unexpected public suite")
    path = RESULTS / "evaluation_execution.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    for arm in arms:
        for name in (f"confirmation_{arm}.json", f"confirmation_rows_{arm}.jsonl", f"eval_{arm}.json",
                     f"progress_eval_{arm}.json"):
            if (RESULTS / name).exists():
                raise ValueError(f"Preserve existing output: {name}")
        if list((RESULTS / "rows").glob(f"*_tez-s-{arm}.jsonl")):
            raise ValueError("Preserve existing public rows")
    sources = RESULTS / "executed_source" / "evaluation_driver"
    sources.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ("run_reasoning_evaluation.py", "eval_reasoning_confirmation.py", "train_reasoning_round2.py",
                 "reasoning_round2.json", "eval_arms.py", "competitor_models.py", "student.py", "common.py",
                 "train_a.py", "train_c.py", "structure_probe.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, sources / name)
        hashes[name] = digest(source)
    for name, source in (("bench_h2h.py", ROOT / "experiments/bench_h2h.py"),
                         ("semif_evaluate.py", ROOT / "data/semif/evaluate.py")):
        shutil.copy2(source, sources / name)
        hashes[name] = digest(source)
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "stages": [],
               "suite_sha256": digest(suite), "source_hashes": hashes,
               "scope": "Quality comparison; no valid latency claim. Frozen selection; no calibration fitting."}
    write_json(path, receipt)
    deadline = time.monotonic() + 22 * 60
    try:
        for phase in ("confirmation", "public"):
            for arm in arms:
                model_root = (ROOT / config["base_directory"]).parent.parent if arm == config["base_arm"] else RUNS
                expected = WEIGHTS[arm] if arm == config["base_arm"] else trained[arm]["weights"]
                model = model_root / "models" / arm
                actual = {"encoder": digest(model / "encoder/model.safetensors"), "head": digest(model / "head.pt")}
                if actual != expected:
                    raise ValueError("Frozen selected model changed")
                env = dict(os.environ, TEZ_PILOT_RUNS=str(model_root), TEZ_PILOT_RESULTS=str(RESULTS),
                           TEZ_PILOT_EVAL_SUITES=str(suite), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                script = "eval_reasoning_confirmation.py" if phase == "confirmation" else "eval_arms.py"
                seconds = min(390 if phase == "confirmation" else 360, deadline - time.monotonic())
                if seconds < 30:
                    raise TimeoutError("Overall evaluation time budget reached")
                stage = {"phase": phase, "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
                         "weights": actual, "timeout_seconds": seconds, "status": "running"}
                receipt["stages"].append(stage)
                write_json(path, receipt)
                cmd = [sys.executable, str(Path(__file__).with_name(script)), "--arm", arm]
                with (RESULTS / f"{phase}_{arm}.log").open("x", encoding="utf-8") as log:
                    try:
                        outcome = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 timeout=seconds, check=False)
                        stage["returncode"] = outcome.returncode
                    except subprocess.TimeoutExpired:
                        stage["returncode"] = 124
                stage.update(status="complete" if stage["returncode"] == 0 else "failed",
                             finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, receipt)
                print(json.dumps(stage), flush=True)
                if stage["returncode"]:
                    raise RuntimeError(f"Evaluation stage failed: {phase}/{arm}; inspect preserved log")
                if phase == "public":
                    measured = json.loads((RESULTS / f"eval_{arm}.json").read_text(encoding="utf-8"))
                    if measured["weights_sha256"] != expected or set(measured["tasks"]) != set(cases):
                        raise ValueError("Public evaluation model or task mismatch")
        receipt["status"] = "complete"
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, receipt)


if __name__ == "__main__":
    main()
