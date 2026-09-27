"""Verify completed filter receipts and retained data integrity without rerunning embeddings."""
import hashlib
import json
from pathlib import Path
import tarfile

from structure_probe import digest, write_json
from structured_choice_data import validate

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-structured-choice-20260927"
DATA = ROOT / "runs/pilot-structured-choice-20260927"


def main():
    archive = DIRECTORY / "filter-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve completed filter evidence")
    spec_path = DIRECTORY / "filter_registration.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution_filter.json").read_text(encoding="utf-8"))
    summary = json.loads((DIRECTORY / "filter_summary.json").read_text(encoding="utf-8"))
    receipt_path = DATA / "filter_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (execution["status"] != "complete" or execution["returncode"] != 0 or receipt["status"] != "complete"
            or execution["registration_sha256"] != digest(spec_path) or receipt["registration_sha256"] != digest(spec_path)
            or summary != {**{k: v for k, v in receipt.items() if k != "drops"}, "full_local_receipt_sha256": digest(receipt_path)}):
        raise ValueError("Incomplete or inconsistent leakage-filter result")
    for name, expected in spec["sources"].items():
        if digest(ROOT / name) != expected or digest(DIRECTORY / "executed_source" / name) != expected:
            raise ValueError("Registered filtering source changed")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen leakage-guard input changed")
    if (receipt["thresholds"] != {"minhash": .8, "cosine": .95, "choice_option_overlap": .5}
            or receipt["embedding_anchor_min_cosine"] < .999):
        raise ValueError("Guard thresholds or embedding control differ")
    data = {name: json.loads((DATA / ("train.filtered.json" if name == "train" else name + ".json")).read_text(encoding="utf-8"))
            for name in ("train", "validation", "confirmation")}
    validate(data)
    if (len(data["train"]) != receipt["retained_states"]
            or sum(len(row["questions"]) for row in data["train"]) != receipt["retained_decisions"]
            or digest(DATA / "train.filtered.json") != receipt["filtered_train_sha256"]):
        raise ValueError("Retained data counts or content differ")
    # This specific run retained every candidate; verify that claim directly, without exporting rows.
    if (receipt["drops"] or receipt["drop_counts"] or digest(DATA / "train.filtered.json") != digest(DATA / "train.json")
            or receipt["retained_states"] != 1200 or receipt["retained_decisions"] != 2400):
        raise ValueError("Unexpected filtering outcome for this recorded run")
    audit = {"status": "complete", "registration_sha256": digest(spec_path), "retained_states": 1200,
             "retained_decisions": 2400, "filtered_train_sha256": receipt["filtered_train_sha256"],
             "source_files_verified": len(spec["sources"]), "prerequisites_verified": len(spec["prerequisites"]),
             "scope": "Source/receipt and retained-data integrity plus exact-label validation. Embeddings, MinHash searches and centroid fits were not rerun by this CPU verifier. Passing automated guards does not prove absence of all semantic overlap. No model has trained or been evaluated on this data."}
    write_json(DIRECTORY / "filter_verification.json", audit)
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    files["analysis_source/archive_structured_choice_filter.py"] = Path(__file__)
    # Training examples and feature caches intentionally remain only in ignored runs/.
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
        raise ValueError("Filter archive verification failed")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"verification": audit, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
