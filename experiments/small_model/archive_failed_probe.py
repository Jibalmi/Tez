"""Retain a terminal failed local probe, its receipts, logs and verified executed sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def main(directory):
    directory = directory.resolve()
    if not directory.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Archive only an explicitly selected results directory")
    archive = directory / "failed-artifacts.tar.gz"
    if archive.exists() or (directory / "artifact_receipt.json").exists():
        raise ValueError("Preserve prior archives")
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
    spec_path = directory / "registration.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (execution["status"] != "failed" or execution["returncode"] == 0 or probe["status"] != "failed"
            or execution["registration_sha256"] != digest(spec_path) or probe["registration_sha256"] != digest(spec_path)):
        raise ValueError("Require a terminal failed probe with matching registration")
    for name, expected in probe["source_files"].items():
        if spec["sources"].get(name) != expected or digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed source differs from its registration")
    analysis = {"status": "failed_preserved", "archived_utc": datetime.now(timezone.utc).isoformat(),
                "source_files_verified": len(probe["source_files"]), "registration_sha256": digest(spec_path),
                "error_type": probe.get("error_type"), "error": probe.get("error"),
                "scope": "Failure evidence only. No successful compatibility, quality, speed or release claim."}
    write_json(directory / "failure_analysis.json", analysis)
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    if any(p.suffix in {".safetensors", ".gguf", ".bin", ".pt", ".npy", ".npz"} for p in files.values()):
        raise ValueError("Do not include weights or feature caches")
    files["archive_source/archive_failed_probe.py"] = Path(__file__)
    files["archive_source/structure_probe.py"] = Path(__file__).with_name("structure_probe.py")
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("Archive content differs")
    write_json(directory / "artifact_manifest.json", expected)
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(directory / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
