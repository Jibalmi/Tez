"""Export and independently checksum the completed diagnostic, including failed preparation and inference."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from structure_probe import digest, write_json


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / "results/pilot-structure-probe-20260927"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if set(summary["models"]) != set(summary["registration"]["models"]):
        raise ValueError("Complete all registered models before final export")
    for arm, result in summary["models"].items():
        receipt = result["receipt"]
        if receipt["status"] != "complete" or digest(directory / f"rows_{arm}.jsonl") != receipt["rows_sha256"]:
            raise ValueError("Prediction receipt mismatch")
        for name, expected in receipt["source_hashes"].items():
            if digest(directory / "executed_source" / arm / name) != expected:
                raise ValueError("Executed source differs from receipt")
    archive = directory / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Do not overwrite an existing archive")
    excluded = {"measurement-artifacts.tar.gz", "artifact_manifest.json", "artifact_receipt.json"}
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and p.name not in excluded}
    preparation = root / "runs/pilot/structure-probe-preflight-v1"
    files.update({"unexecuted_preparation_v1/" + p.name: p for p in preparation.iterdir() if p.is_file()})
    for name in ("report_structure_probe.py", "structure_probe_driver.py", "archive_structure_probe.py",
                 "audit_structure_behavior.py", "test_structure_probe.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    expected = {key: {"sha256": digest(path), "bytes": path.stat().st_size} for key, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for key, path in sorted(files.items()):
            bundle.add(path, arcname=key, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected or duplicate archive member")
            source = bundle.extractfile(member)
            actual[member.name] = {"sha256": hashlib.sha256(source.read()).hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("Archive content differs from exported files")
    write_json(directory / "artifact_manifest.json", expected)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_sha256": digest(archive),
               "bytes": archive.stat().st_size, "files_verified": len(actual),
               "prediction_rows_validated": sum(r["rows_validated"] for r in summary["models"].values()),
               "all_source_hashes_verified": True, "archive_member_hashes_verified": True,
               "note": "Includes registered inputs, raw predictions, executed source, receipts, logs, failed metadata audit and unexecuted preparation."}
    write_json(directory / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
