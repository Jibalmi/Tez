"""Validate a complete, independent cloud teacher dataset and export training-ready files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from cloud_teacher import MODEL_SHA, collect, parity, save, sha
from common import RUNS, read_jsonl, write_jsonl
from dataset import TEMPS, softmax, temp_for
from teacher import add_none
from tez.schema import parse_question


def export(source: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Choose a new export directory; existing datasets are immutable")
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("rebuild_all"):
        raise ValueError("Only a complete independent rebuild can be exported")
    if report["manifest"]["model_sha256"] != MODEL_SHA or not report["production"].get("fresh_state_cache"):
        raise ValueError("Unexpected model or state-cache protocol")
    for name in ("pool.jsonl", "assign.jsonl"):
        if sha(RUNS / name) != report["manifest"]["files"]["runs/pilot/" + name]:
            raise ValueError(f"Local input differs from the cloud snapshot: {name}")
    assigned = read_jsonl(RUNS / "assign.jsonl")
    expected = {a["id"] for a in assigned}
    rows = collect(source / "production")
    if len(expected) != len(assigned) or set(rows) != expected:
        raise ValueError("Production states are duplicated, missing, or unexpected")
    reference = collect(source / "canary-1")
    consistency = parity(reference, {sid: rows[sid] for sid in reference})
    if not consistency["passed"]:
        raise ValueError("Production targets do not match the cloud reference")
    targets = []
    for assigned_state in assigned:
        sid = assigned_state["id"]
        actual = rows[sid]["questions"]
        if len(actual) != len(assigned_state["questions"]):
            raise ValueError(f"Question count differs: {sid}")
        for specification, result in zip(assigned_state["questions"], actual):
            question = parse_question(specification["qid"], {
                k: specification[k] for k in ("type", "instructions", "criteria") if k in specification})
            abstain = question.type == "choice" and add_none(sid, question.id) and len(question.criteria) < 26
            metadata = (question.id, question.type, question.keys(abstain), abstain)
            if (result["qid"], result["type"], result["keys"], result["abstain"]) != metadata:
                raise ValueError(f"Question metadata differs: {sid}/{question.id}")
            vectors = [np.asarray(result[field], dtype=float) for field in ("z_orig", "z_rev")]
            if any(v.shape != (len(result["keys"]),) or not np.isfinite(v).all() for v in vectors):
                raise ValueError(f"Invalid logits: {sid}/{question.id}")
            temperature = temp_for(question.type, len(result["keys"]))
            probabilities = (softmax(vectors[0], temperature) + softmax(vectors[1], temperature)[::-1]) / 2
            targets.append({"state_id": sid, "qid": question.id, "type": question.type,
                            "keys": result["keys"], "probabilities": probabilities.tolist()})
    output.mkdir(parents=True)
    for name in ("pool.jsonl", "assign.jsonl"):
        shutil.copy2(RUNS / name, output / name)
    teacher = output / "teacher"
    teacher.mkdir()
    ordered = [rows[a["id"]] for a in assigned]
    for offset in range(0, len(ordered), 200):
        write_jsonl(teacher / f"part-{offset // 200:04d}.jsonl", ordered[offset:offset + 200])
    write_jsonl(output / "probability-targets.jsonl", targets)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source.resolve()),
                "model_sha256": MODEL_SHA, "llama_cpp_build": "b11100", "states": len(rows),
                "decisions": len(targets), "temperatures": TEMPS, "cache_prompt_between_states": False,
                "production_reference_consistency": consistency,
                "original_laptop_comparison": report["trials"][0]["local_reference_comparison"],
                "target_note": "Temperature-adjusted option distributions, not probabilities of correctness",
                "files": {str(p.relative_to(output)).replace("\\", "/"): sha(p)
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    save(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(args.source.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
