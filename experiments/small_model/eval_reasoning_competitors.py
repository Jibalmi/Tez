"""Run the registered native competitor comparison on the frozen authored confirmation corpus."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from eval_reasoning_confirmation import cases_for, prerequisites
from structure_probe import digest, write_json
from train_reasoning_round2 import ROOT

DIRECTORY = ROOT / "results/pilot-reasoning-competitors-20260927"
REGISTRATION = Path(__file__).with_name("reasoning_competitor_registration.json")


def worker(arm):
    prerequisites()
    reg = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    source = ROOT / reg["source"]
    if arm not in reg["arms"] or digest(source) != reg["source_sha256"]:
        raise ValueError("Unregistered model or changed corpus")
    os.environ.update(TEZ_PILOT_RUNS=str(ROOT / "runs/student-artifacts-h200-20260927/runs/student-data"),
                      TEZ_PILOT_RESULTS=str(DIRECTORY), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    from common import THERMAL, thermal_guard
    from competitor_models import predictor, kev_request, model_revision
    from train_c import BASE, kernel_packages, prompt_for

    rows_path, receipt_path = DIRECTORY / f"rows_{arm}.jsonl", DIRECTORY / f"execution_{arm}.json"
    if rows_path.exists() or receipt_path.exists():
        raise ValueError("Never overwrite an existing attempt")
    sources = DIRECTORY / "executed_source" / arm
    sources.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ("eval_reasoning_competitors.py", "reasoning_competitor_registration.json", "eval_reasoning_confirmation.py",
                 "competitor_models.py", "eval_arms.py", "train_c.py", "common.py", "structure_probe.py"):
        original = Path(__file__).with_name(name)
        shutil.copy2(original, sources / name)
        hashes[name] = digest(original)
    if arm == "decider08-native":
        code = ROOT / "runs/research/decider"
        commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(code), "status", "--porcelain"], text=True).strip()
        if commit != reg["decider_code_commit"] or dirty:
            raise ValueError("Decider source differs from registered clean checkout")
        (code.parent / "decider-commit.txt").write_text(commit + "\n", encoding="utf-8")
        for original in (code / "decider").rglob("*.py"):
            name = "native/" + original.relative_to(code).as_posix()
            target = sources / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            hashes[name] = digest(original)
        shutil.copy2(code / "LICENSE", sources / "native/LICENSE")
        hashes["native/LICENSE"] = digest(code / "LICENSE")
    receipt = {"status": "running", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": digest(source), "registration_sha256": digest(REGISTRATION), "source_files": hashes,
               "rows": 0, "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(), "packages": kernel_packages()}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        thermal_guard(force=True)
        pred = predictor(arm)
        if pred.weights != reg["weights"][arm]:
            raise ValueError("Competitor weights differ from registration")
        receipt.update(weights=pred.weights, model_info=pred.info)
        groups = cases_for(json.loads(source.read_text(encoding="utf-8")))
        safe = [[{k: c[k] for k in ("state", "instructions", "qtype", "options")} for c in group] for group in groups]
        if arm.startswith("c-"):
            from huggingface_hub import snapshot_download
            base = Path(snapshot_download(BASE, revision=model_revision(BASE), allow_patterns=["*.safetensors"], local_files_only=True))
            weights = {p.name: digest(p) for p in base.glob("*.safetensors")}
            if list(weights.values()) != [reg["qwen_base_sha256"]]:
                raise ValueError("Qwen base weights differ from registration")
            receipt["base_weights"] = weights
            lengths = [len(pred.tok(prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]),
                                    add_special_tokens=False)["input_ids"]) for group in safe for c in group]
            limit = 4096
        else:
            lengths = []
            for group in safe:
                request, _ = kev_request(group)
                questions = {key: {k: v for k, v in q.items() if k not in ("src", "label")} for key, q in request["questions"].items()}
                _, _, items = pred.pred._system_one_items(request["state"], questions, independent=True, max_state_tokens=100000)
                lengths.extend(len(item["ids"]) for item in items)
            limit = 8192
        receipt["token_audit"] = {"max_tokens": max(lengths), "limit": limit, "untruncated_prompts_checked": len(lengths)}
        if max(lengths) >= limit:
            raise ValueError("Competitor input would be truncated")
        with rows_path.open("x", encoding="utf-8") as output:
            for start in range(0, len(groups), 8):
                if time.monotonic() - started > 330:
                    raise TimeoutError("Worker time budget reached")
                thermal_guard(force=True)
                values = pred.groups(safe[start:start + 8])
                chunk = groups[start:start + 8]
                if len(values) != len(chunk) or any(len(a) != len(b) for a, b in zip(values, chunk)):
                    raise ValueError("Missing competitor predictions")
                for cases, predictions in zip(chunk, values):
                    for case, z in zip(cases, predictions):
                        z = np.asarray(z, dtype=float)
                        if z.shape != (len(case["options"]),) or not np.isfinite(z).all():
                            raise ValueError("Invalid competitor logits")
                        p = np.exp(z - z.max())
                        p /= p.sum()
                        row = {k: case[k] for k in ("id", "group", "operation", "domain", "gold")}
                        row.update(pred=int(p.argmax()), probabilities=p.tolist(), logits=z.tolist())
                        output.write(json.dumps(row) + "\n")
                        receipt["rows"] += 1
                output.flush()
                write_json(receipt_path, receipt)
        receipt.update(status="complete", rows_sha256=digest(rows_path))
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_governor=THERMAL)
        write_json(receipt_path, receipt)


def driver():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "execution.json"
    if path.exists():
        raise ValueError("Preserve existing execution")
    reg = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
    write_json(path, receipt)
    for arm in reg["arms"]:
        with (DIRECTORY / f"{arm}.log").open("x", encoding="utf-8") as log:
            try:
                outcome = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", arm], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=360, check=False)
                code = outcome.returncode
            except subprocess.TimeoutExpired:
                code = 124
        receipt["stages"].append({"arm": arm, "returncode": code})
        write_json(path, receipt)
        print(json.dumps(receipt["stages"][-1]), flush=True)
        if code:
            break
    receipt.update(status="complete" if len(receipt["stages"]) == len(reg["arms"])
                   and all(s["returncode"] == 0 for s in receipt["stages"]) else "failed",
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)
    return 0 if receipt["status"] == "complete" else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if args.arm:
        worker(args.arm)
    else:
        raise SystemExit(driver())
