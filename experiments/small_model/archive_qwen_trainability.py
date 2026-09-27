"""Verify and preserve the successful feasibility probe's source and execution receipts."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/trainability"


def main():
    archive = DIRECTORY / "probe-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve completed evidence")
    probe = json.loads((DIRECTORY / "probe.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    spec_path = DIRECTORY / "registration.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (probe["status"] != "complete" or not probe["passed"] or probe["weights_saved"]
            or execution["status"] != "complete" or execution["returncode"] != 0
            or probe["registration_sha256"] != digest(spec_path) or execution["registration_sha256"] != digest(spec_path)
            or not math.isfinite(probe["loss"]) or not math.isfinite(probe["gradient_norm_before_clip"])
            or probe["gradient_norm_before_clip"] <= 0 or probe["adapter_tensors_changed"] <= 0):
        raise ValueError("Require successful finite-update feasibility evidence")
    for name, expected in spec["sources"].items():
        if probe["source_files"].get(name) != expected or digest(DIRECTORY / "executed_source" / name) != expected:
            raise ValueError("Executed source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Probe input changed")
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    files["analysis_source/archive_qwen_trainability.py"] = Path(__file__)
    files["analysis_source/structure_probe.py"] = Path(__file__).with_name("structure_probe.py")
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
        raise ValueError("Archive verification failed")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
               "scope": "Executed-source and receipt integrity. Gradients and probe weights were not retained, so this is not an independent numerical replay or a trained candidate."}
    write_json(DIRECTORY / "artifact_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
