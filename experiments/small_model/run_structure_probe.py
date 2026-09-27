"""Evaluate a registered structure probe with frozen weights; run under the shared compute lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from structure_probe import FORMATS, digest, validate_records, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=12)
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py for laptop inference")
    os.environ["TEZ_PILOT_RUNS"] = str(args.model_root.resolve())
    os.environ["TEZ_PILOT_RESULTS"] = str(args.directory.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from common import THERMAL, thermal_guard
    from competitor_models import predictor, model_revision
    from latency import thermal
    from train_c import kernel_packages

    directory = args.directory
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if args.arm not in registration["models"] or digest(directory / "cases.json") != registration["cases_sha256"]:
        raise ValueError("Arm or inputs differ from registration")
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    validate_records(cases)
    path = directory / f"rows_{args.arm}.jsonl"
    receipt_path = directory / f"execution_{args.arm}.json"
    if path.exists() or receipt_path.exists():
        raise ValueError("Existing experiment is immutable; preserve failed attempts separately")
    receipt = {"status": "running", "arm": args.arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "cases_sha256": digest(directory / "cases.json"), "registration_sha256": digest(directory / "registration.json"),
               "runner_sha256": digest(__file__), "threads": torch.get_num_threads(), "device": torch.cuda.get_device_name(),
               "thermal_start": thermal(), "purpose": "Quality diagnostic only; no usable latency measurement", "rows": 0,
               "packages": kernel_packages(), "source_hashes": {}}
    source_dir = directory / "executed_source" / args.arm
    source_dir.mkdir(parents=True, exist_ok=False)
    for name in ("run_structure_probe.py", "structure_probe.py", "eval_arms.py", "competitor_models.py",
                 "common.py", "student.py", "train_a.py", "train_c.py", "latency.py"):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        receipt["source_hashes"][name] = digest(source)
    write_json(receipt_path, receipt)
    start = time.monotonic()
    try:
        thermal_guard(force=True)
        pred = predictor(args.arm)
        if pred.weights != registration["weights_sha256"][args.arm]:
            raise ValueError("Frozen model weight hashes differ from registration")
        receipt.update(weights=pred.weights, model_info=pred.info)
        if args.arm.startswith("c-"):
            from huggingface_hub import snapshot_download
            from train_c import BASE
            base = Path(snapshot_download(BASE, revision=model_revision(BASE), local_files_only=True))
            receipt["base"] = {"revision": model_revision(BASE),
                               "weights": {p.name: digest(p) for p in base.glob("*.safetensors")}}
        groups, metadata = [], []
        for index, record in enumerate(cases):
            for form in FORMATS[::1 if index % 2 == 0 else -1]:
                group = [{"state": record["states"][form], **{k: q[k] for k in ("qtype", "instructions", "options")}}
                         for q in record["questions"]]
                groups.append(group)
                metadata.append((record, form))
        if args.arm.startswith("a-"):
            from student import Q
            lengths = [len(pred.enc.encode(g[0]["state"], [Q(c["qtype"], c["instructions"], c["options"]) for c in g]).ids)
                       for g in groups]
            clipped = sum(len(pred.enc.ids(f" ({c['qtype']}) {c['instructions']}")) > 160
                          or any(len(pred.enc.ids(" " + k + ": " + d)) > 48 for k, d in c["options"])
                          for g in groups for c in g)
            limit = 8192
        else:
            from train_c import prompt_for
            lengths = [len(pred.tok(prompt_for(c["qtype"], c["instructions"], c["options"], c["state"]),
                                    add_special_tokens=False)["input_ids"]) for g in groups for c in g]
            clipped, limit = 0, 4096
        receipt["token_audit"] = {"max_tokens": max(lengths), "context_limit": limit, "clipped_questions_or_options": clipped}
        if max(lengths) >= limit or clipped:
            raise ValueError("Input truncation would confound the probe")
        with path.open("x", encoding="utf-8") as output:
            for offset in range(0, len(groups), 8):
                if time.monotonic() - start > args.minutes * 60:
                    raise TimeoutError("Probe time bound reached; preserve partial rows")
                thermal_guard(force=True)
                chunk = groups[offset:offset + 8]
                logits = pred.groups(chunk)
                if len(logits) != len(chunk):
                    raise ValueError("Missing predicted group")
                for (record, form), group, values in zip(metadata[offset:offset + 8], chunk, logits):
                    if len(values) != len(group):
                        raise ValueError("Missing predicted decision")
                    for question, z in zip(record["questions"], values):
                        z = np.asarray(z, dtype=float)
                        if z.shape != (len(question["options"]),) or not np.isfinite(z).all():
                            raise ValueError("Invalid model distribution")
                        p = np.exp(z - z.max())
                        p /= p.sum()
                        row = {"id": record["id"], "format": form, "operation": question["operation"],
                               "gold": question["gold"], "pred": int(p.argmax()), "probabilities": p.tolist()}
                        output.write(json.dumps(row) + "\n")
                        receipt["rows"] += 1
                output.flush()
                receipt["groups_completed"] = offset + len(chunk)
                write_json(receipt_path, receipt)
                print(json.dumps({"arm": args.arm, "rows": receipt["rows"]}), flush=True)
        receipt["rows_sha256"] = digest(path)
        receipt["status"] = "complete"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                       thermal_end=thermal(), thermal_governor=THERMAL)
        write_json(receipt_path, receipt)


if __name__ == "__main__":
    main()
