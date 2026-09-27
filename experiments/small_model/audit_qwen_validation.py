"""Diagnose frozen Qwen teacher-validation errors without inference or test-label fitting."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tarfile

from parity_evidence import read_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-validation-audit-20260927"
REFERENCE = ROOT / "results/pilot-restricted-qwen-20260927"
TEACHER = ROOT / "runs/pilot-cloud-h200-20260927"


def summarize(rows):
    gold = [row for row in rows if row["gold"] is not None]
    result = {"n": len(rows), "kl": fmean(row["kl"] for row in rows),
              "teacher_agreement": fmean(row["student_argmax"] == row["teacher_argmax"] for row in rows),
              "student_confidence": fmean(max(row["probabilities"]) for row in rows),
              "teacher_confidence": fmean(max(row["target"]) for row in rows),
              "student_entropy": fmean(-sum(p * math.log(max(p, 1e-12)) for p in row["probabilities"]) for row in rows),
              "teacher_entropy": fmean(-sum(p * math.log(max(p, 1e-12)) for p in row["target"]) for row in rows),
              "gold_rows": len(gold)}
    if gold:
        result.update(student_gold_accuracy=fmean(row["student_argmax"] == row["gold"] for row in gold),
                      teacher_gold_accuracy=fmean(row["teacher_argmax"] == row["gold"] for row in gold))
    return result


def main():
    if (DIRECTORY / "audit.json").exists():
        raise ValueError("Preserve existing audit")
    reference_path = REFERENCE / "reference_rows.jsonl"
    if digest(reference_path) != "e353d17b87329541e191a14004e710ba57b292cad0a01a0ea860e80c1c0042e3":
        raise ValueError("Frozen validation probabilities changed")
    manifest_path = TEACHER / "manifest.json"
    if digest(manifest_path) != "8e3f7cf23f7b14c80203eca42eab9bb7e7108f8c7aab67abd9d9b136f7cfb807":
        raise ValueError("Teacher manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(TEACHER / name) != expected:
            raise ValueError("Teacher file changed")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import dataset
    dataset.RUNS = TEACHER
    _, _, examples = dataset.load_examples()
    expected = [(example.id, decision.qid, len(decision.options)) for example in examples for decision in example.decisions]
    if len(examples) != 278 or len(expected) != 1390:
        raise ValueError("Unexpected original teacher-validation membership")
    values = read_rows(reference_path, expected)
    rows = []
    for example in examples:
        for decision in example.decisions:
            p = values[example.id, decision.qid, len(decision.options)]
            target = decision.target.tolist()
            kl = sum(t * (math.log(max(t, 1e-12)) - math.log(max(q, 1e-12))) for t, q in zip(target, p))
            rows.append({"state_id": example.id, "qid": decision.qid, "source": example.source, "lang": example.lang,
                         "qtype": decision.type, "k": len(p), "gold": decision.gold, "target": target, "probabilities": p,
                         "teacher_argmax": max(range(len(p)), key=target.__getitem__),
                         "student_argmax": max(range(len(p)), key=p.__getitem__), "kl": kl})
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    groups = {}
    for key in ("qtype", "source", "lang"):
        partitions = defaultdict(list)
        for row in rows:
            partitions[row[key]].append(row)
        groups[key] = {name: summarize(part) for name, part in sorted(partitions.items())}
    sources = {name: digest(Path(__file__).with_name(name)) for name in
               ("audit_qwen_validation.py", "dataset.py", "common.py", "structure_probe.py", "parity_evidence.py")}
    receipt = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
               "reference_rows_sha256": digest(reference_path), "teacher_manifest_sha256": digest(manifest_path),
               "sources": sources, "overall": summarize(rows), "groups": groups,
               "scope": "Descriptive analysis of the original teacher-validation split, permitted for training monitoring. No public test, authored confirmation, inference or fitting. Targets are the original fixed type-temperature, two-order teacher distributions; this is agreement with that teacher, not general accuracy."}
    output = DIRECTORY / "validation_rows.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    receipt["validation_rows_sha256"] = digest(output)
    write_json(DIRECTORY / "audit.json", receipt)
    lines = ["# Qwen teacher-validation audit", "", "This reuses all 278 original validation states and 1,390 frozen Qwen probability vectors.",
             "The 12B targets use the original fixed type temperatures and two-order average. No inference or fitting was performed.",
             "Teacher agreement is not gold accuracy; most questions here have no independent gold label.", "",
             "| Group | Decisions | Teacher agreement | Mean KL | Student confidence | Teacher confidence |",
             "|---|---:|---:|---:|---:|---:|"]
    for kind, parts in [("all", {"all": receipt["overall"]}), *groups.items()]:
        for name, value in parts.items():
            lines.append(f"| {kind}: {name} | {value['n']} | {100 * value['teacher_agreement']:.2f}% | {value['kl']:.4f} | "
                         f"{value['student_confidence']:.4f} | {value['teacher_confidence']:.4f} |")
    lines += ["", "Small language/source subgroups are descriptive, not stable rankings. Different option counts also affect KL and confidence.",
              "These results may guide a registered training continuation using the original training split and validation-only selection.",
              "They do not justify changing a public-test prompt, fitting to test labels, weakening a gate or promoting a checkpoint.",
              "[Full metrics and provenance](audit.json). Raw teacher targets and frozen student probabilities are in the measurement archive.", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    files = {p.name: p for p in DIRECTORY.iterdir() if p.is_file() and p != archive}
    for name in sources:
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    files["inputs/teacher_manifest.json"] = manifest_path
    files["inputs/reference_parity.json"] = REFERENCE / "parity.json"
    expected_files = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in files.items()}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as bundle:
        actual = {member.name: {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
                  for member in bundle.getmembers()}
    if actual != expected_files:
        raise ValueError("Archive verification failed")
    write_json(DIRECTORY / "artifact_manifest.json", expected_files)
    write_json(DIRECTORY / "artifact_receipt.json", {"sha256": digest(archive), "bytes": archive.stat().st_size,
               "files_verified": len(actual), "probability_rows_validated": len(rows)})
    print(json.dumps({"overall": receipt["overall"], "types": groups["qtype"]}, indent=2))


if __name__ == "__main__":
    main()
