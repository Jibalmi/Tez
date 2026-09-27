"""Verify and export the completed ordinal diagnostic and the exact source corpus it transforms."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from ordinal_probe import DIRECTORY, ROOT
from structure_probe import digest, write_json


def main():
    summary = json.loads((DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    registration = summary["registration"]
    if set(summary["models"]) != set(registration["models"]):
        raise ValueError("Complete the registered models before final export")
    for arm, model in summary["models"].items():
        receipt = model["receipt"]
        if digest(DIRECTORY / f"rows_{arm}.jsonl") != receipt["rows_sha256"]:
            raise ValueError("Prediction checksum mismatch")
        for name, expected in receipt["source_hashes"].items():
            if digest(DIRECTORY / "executed_source" / arm / name) != expected:
                raise ValueError("Executed source checksum mismatch")
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve existing archive")
    excluded = {archive.name, "artifact_manifest.json", "artifact_receipt.json"}
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*") if p.is_file() and p.name not in excluded}
    source = ROOT / registration["source_cases"]
    if digest(source) != registration["source_cases_sha256"]:
        raise ValueError("Source corpus checksum mismatch")
    files["source_corpus/cases.json"] = source
    files["source_corpus/registration.json"] = source.with_name("registration.json")
    for name in ("archive_ordinal_probe.py", "ordinal_probe.py", "test_ordinal_probe.py", "ordinal_diagnostic_registration.json"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
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
        raise ValueError("Archive content differs from export")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_sha256": digest(archive),
               "bytes": archive.stat().st_size, "files_verified": len(actual),
               "prediction_rows_validated": sum(m["rows_validated"] for m in summary["models"].values()),
               "all_executed_sources_verified": True, "all_archive_members_verified": True}
    write_json(DIRECTORY / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
