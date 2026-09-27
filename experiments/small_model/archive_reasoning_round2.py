"""Archive verified evaluation evidence and executed sources, excluding training examples and weights."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from structure_probe import digest, write_json
from train_reasoning_round2 import ROOT, RESULTS


def main():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    execution = json.loads((RESULTS / "evaluation_execution.json").read_text(encoding="utf-8"))
    if execution["status"] != "complete" or len(summary["models"]) != 3:
        raise ValueError("Complete and validate all registered evaluation arms first")
    for arm, model in summary["models"].items():
        if digest(RESULTS / f"eval_{arm}.json") != model["eval_sha256"]:
            raise ValueError("Evaluation summary checksum mismatch")
        for task, item in model["row_files"].items():
            if digest(RESULTS / "rows" / f"rows_{task}_tez-s-{arm}.jsonl") != item["sha256"]:
                raise ValueError("Public prediction checksum mismatch")
        receipt = summary["confirmation"]["models"][arm]["receipt"]
        if digest(RESULTS / f"confirmation_rows_{arm}.jsonl") != receipt["rows_sha256"]:
            raise ValueError("Confirmation prediction checksum mismatch")
        for name, expected in receipt["source_files"].items():
            if digest(RESULTS / "executed_source/confirmation" / arm / name) != expected:
                raise ValueError("Confirmation source checksum mismatch")
    for path in RESULTS.glob("train_*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in receipt["sources"].items():
            if digest(RESULTS / "executed_source" / receipt["arm"] / name) != expected:
                raise ValueError("Training source checksum mismatch")
    for name, expected in execution["source_hashes"].items():
        if digest(RESULTS / "executed_source/evaluation_driver" / name) != expected:
            raise ValueError("Execution source checksum mismatch")
    archive = RESULTS / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve an existing archive")
    excluded = {archive.name, "artifact_manifest.json", "artifact_receipt.json"}
    files = {p.relative_to(RESULTS).as_posix(): p for p in RESULTS.rglob("*") if p.is_file() and p.name not in excluded}
    config = json.loads((ROOT / "experiments/small_model/reasoning_round2.json").read_text(encoding="utf-8"))
    # Evaluation inputs are exportable; authored training examples stay local.
    files["inputs/confirmation.json"] = ROOT / config["new_data"] / "confirmation.json"
    files["inputs/public-suites.json"] = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    files["inputs/data_manifest.json"] = ROOT / config["new_data"] / "manifest.json"
    files["reference/comparison.json"] = ROOT / "results/pilot-improvement-20260927/comparison.json"
    files["reference/filter_summary.json"] = ROOT / "results/pilot-reasoning-data-20260927/filter_summary.json"
    for name in ("report_reasoning_round2.py", "archive_reasoning_round2.py", "test_reasoning_report.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    forbidden = {".pt", ".safetensors", ".gguf", ".npy", ".npz"}
    if any(p.suffix in forbidden for p in files.values()):
        raise ValueError("Model or cache file in measurement export")
    expected = {name: {"sha256": digest(p), "bytes": p.stat().st_size} for name, p in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if expected != actual:
        raise ValueError("Archive differs from measurement files")
    write_json(RESULTS / "artifact_manifest.json", expected)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "sha256": digest(archive),
               "bytes": archive.stat().st_size, "files_verified": len(actual),
               "public_rows_validated": sum(m["rows_validated"] for m in summary["models"].values()),
               "confirmation_rows_validated": sum(m["rows_validated"] for m in summary["confirmation"]["models"].values()),
               "all_executed_sources_verified": True, "all_archive_members_verified": True,
               "excludes": ["Training examples", "Model weights", "Feature caches"]}
    write_json(RESULTS / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
