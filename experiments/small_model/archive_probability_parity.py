"""Independently validate and archive a completed Qwen numerical-parity experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

from parity_evidence import compare_rows, read_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def main(experiment):
    directory = ROOT / f"results/pilot-{'restricted' if experiment == 'restricted' else 'fla'}-qwen-20260927"
    archive = directory / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve the completed archive")
    receipt_path = directory / "parity.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    spec_path = directory / "registration.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if receipt["status"] != "complete" or execution["status"] != "complete" or digest(spec_path) != receipt["registration_sha256"]:
        raise ValueError("Complete the registered experiment before analysis")
    for name, expected in receipt["sources"].items():
        if digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed source changed")
    data = ROOT / spec["teacher_directory"]
    if digest(data / "manifest.json") != spec["teacher_manifest_sha256"]:
        raise ValueError("Teacher manifest changed")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(data / name) != expected:
            raise ValueError("Teacher input changed")
    # Reconstruct validation inputs with the exact data loader used by inference, without importing a model.
    for name in ("dataset.py", "common.py"):
        if digest(Path(__file__).with_name(name)) != receipt["sources"][name]:
            raise ValueError("Validation reconstruction code changed")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    import dataset
    dataset.RUNS = data
    _, _, examples = dataset.load_examples()
    groups = [[{"state": e.state, "qtype": d.type, "instructions": d.instructions, "options": d.options}
               for d in e.decisions] for e in examples]
    ids = [(e.id, d.qid, len(d.options)) for e in examples for d in e.decisions]
    if len(examples) != 278 or len(ids) != 1390:
        raise ValueError("Unexpected validation membership")
    inputs_path = directory / "validation_inputs.json"
    inputs = {"ids": ids, "groups": groups}
    if inputs_path.exists():
        if json.loads(inputs_path.read_text(encoding="utf-8")) != json.loads(json.dumps(inputs)):
            raise ValueError("Recorded validation inputs differ")
    else:
        write_json(inputs_path, inputs)
    baseline_path = directory / "reference_rows.jsonl" if experiment == "restricted" else ROOT / spec["reference_rows"]
    candidate_path = directory / ("restricted_rows.jsonl" if experiment == "restricted" else "candidate_rows.jsonl")
    baseline_hash = receipt["reference_rows_sha256"] if experiment == "restricted" else spec["reference_rows_sha256"]
    candidate_hash = receipt["restricted_rows_sha256"] if experiment == "restricted" else receipt["candidate_rows_sha256"]
    if digest(baseline_path) != baseline_hash or digest(candidate_path) != candidate_hash:
        raise ValueError("Probability rows changed")
    metrics = compare_rows(read_rows(baseline_path, ids), read_rows(candidate_path, ids), spec)
    for name in ("argmax_agreement", "max_probability_delta", "mean_total_variation"):
        if not math.isclose(metrics[name], receipt[name], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Independent analysis differs from worker")
    if metrics["passed"] != receipt["passed"] or execution["returncode"] != (0 if metrics["passed"] else 2):
        raise ValueError("Parity decision differs")
    if experiment == "fla":
        for name in ("reference_receipt", "preparation"):
            if digest(ROOT / spec[name]) != spec[name + "_sha256"]:
                raise ValueError("Frozen kernel/reference receipt changed")
        if receipt["dispatch"]["torch_chunk_gated_delta_rule"]["calls"] <= 0:
            raise ValueError("No optimized dispatch recorded")
    write_json(directory / "analysis.json", {**metrics, "validation_inputs_sha256": digest(inputs_path),
               "reference_rows_sha256": digest(baseline_path), "candidate_rows_sha256": digest(candidate_path),
               "executed_sources_verified": True, "analyzed_utc": datetime.now(timezone.utc).isoformat()})
    lines = ["# Qwen probability-parity result", "", f"Candidate: {spec['candidate']}. All 278 original teacher-validation states and 1,390 decisions.",
             "The archived inputs contain validation cases only; no model weights or training examples are exported.", "",
             "| Measure | Observed | Unchanged requirement |", "|---|---:|---:|",
             f"| Argmax agreement | {metrics['argmax_agreement']:.8f} | >= {spec['argmax_agreement_min']} |",
             f"| Maximum probability change | {metrics['max_probability_delta']:.8f} | <= {spec['max_probability_delta_max']} |",
             f"| Mean total variation | {metrics['mean_total_variation']:.8f} | <= {spec['mean_total_variation_max']} |", "",
             f"**Parity {'passed' if metrics['passed'] else 'failed'}.** No runtime promotion or measured speedup follows from this result.",
             "Separately valid target-laptop latency remains required. No thresholds were relaxed and no cloud compute was used.", "",
             "An independent reader checked every probability's shape, range, normalization, unique identifier and validation membership,",
             "then reproduced all three metrics. The archive contains raw predictions, validation inputs, receipts and executed sources.", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    excluded = {archive.name, "artifact_manifest.json", "artifact_receipt.json"}
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and p.name not in excluded and "__pycache__" not in p.parts}
    files["inputs/teacher_manifest.json"] = data / "manifest.json"
    for name in ("archive_probability_parity.py", "parity_evidence.py", "test_parity_evidence.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    if experiment == "fla":
        files["reference/reference_rows.jsonl"] = baseline_path
        files["reference/parity.json"] = ROOT / spec["reference_receipt"]
        files["kernel/preparation.json"] = ROOT / spec["preparation"]
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
        raise ValueError("Archive contents differ")
    write_json(directory / "artifact_manifest.json", expected)
    result = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
              "probability_rows_validated": 2 * len(ids), "passed": metrics["passed"]}
    write_json(directory / "artifact_receipt.json", result)
    print(json.dumps({**metrics, "archive": result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("restricted", "fla"), required=True)
    main(parser.parse_args().experiment)
