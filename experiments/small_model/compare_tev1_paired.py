"""Supplement the complete Tev1 comparison with exploratory paired state intervals."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile

from compare_round1 import paired_typed_delta
from competitor_overlap import read_checked
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT = ROOT / "results/pilot-tev1-eval-20260927"
DIRECTORY = ROOT / "results/pilot-tev1-paired-20260927"


def main():
    if DIRECTORY.exists():
        raise ValueError("Preserve previous paired analysis")
    comparison = MEASUREMENT / "comparison.json"
    artifact = MEASUREMENT / "artifact_receipt.json"
    summary = json.loads(comparison.read_text(encoding="utf-8"))
    receipt = json.loads(artifact.read_text(encoding="utf-8"))
    if (summary["status"] != "complete" or receipt["primary_rows_validated"] != 18453
            or receipt["reversed_rows_validated"] != 1100
            or digest(MEASUREMENT / "measurement-artifacts.tar.gz") != receipt["sha256"]):
        raise ValueError("Require the completed independently verified native comparison")
    spec = json.loads((MEASUREMENT / "registration.json").read_text(encoding="utf-8"))
    suite_path = ROOT / spec["suite"]
    if digest(suite_path) != spec["prerequisites"][spec["suite"]]:
        raise ValueError("Frozen evaluation membership changed")
    cases = json.loads(suite_path.read_text(encoding="utf-8"))["typed_decisions"]
    if len(cases) != 2000 or Counter(Counter(c["group"] for c in cases).values()) != {5: 400}:
        raise ValueError("Expected 400 states with all five decisions each")
    filenames = {
        "tev1": "rows_typed_decisions.jsonl",
        "qwen": "reference/qwen/rows_typed_decisions_tez-s-c-qwen08-30k.jsonl",
        "decider": "reference/decider/rows_typed_decisions_tez-s-decider08-native.jsonl",
    }
    inputs, rows = {}, {}
    with tarfile.open(MEASUREMENT / "measurement-artifacts.tar.gz", "r:gz") as bundle:
        for name in ("comparison.json", "registration.json"):
            if bundle.extractfile(name).read() != (MEASUREMENT / name).read_bytes():
                raise ValueError("Current analysis metadata differs from the verified archive")
        for name, filename in filenames.items():
            raw = bundle.extractfile(filename).read()
            if hashlib.sha256(raw).hexdigest() != summary["prediction_hashes"][filename]:
                raise ValueError("Archived typed prediction hash differs")
            inputs[name] = raw
    DIRECTORY.mkdir(parents=True)
    (DIRECTORY / ".gitignore").write_text("rows_*.jsonl\n", encoding="utf-8")
    files = {"inputs/comparison.json": comparison, "inputs/artifact_receipt.json": artifact,
             "inputs/registration.json": MEASUREMENT / "registration.json", "inputs/eval-suites.json": suite_path}
    for name, raw in inputs.items():
        path = DIRECTORY / f"rows_{name}.jsonl"
        path.write_bytes(raw)
        rows[name] = read_checked(path, cases)
        actual = sum(r["pred"] == r["gold"] for r in rows[name].values()) / len(cases)
        if not math.isclose(actual, summary["scores"][name]["typed_decisions"]["accuracy"], abs_tol=1e-12):
            raise ValueError("Recomputed typed accuracy differs")
        files[path.name] = path
    results = {name: paired_typed_delta(cases, rows["tev1"], rows[name]) for name in ("qwen", "decider")}
    sources = ["compare_tev1_paired.py", "compare_round1.py", "test_compare_round1.py",
               "competitor_overlap.py", "structure_probe.py", "common.py"]
    for name in sources:
        files["source/" + name] = Path(__file__).with_name(name)
    result = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
              "comparison_sha256": digest(comparison), "measurement_archive_sha256": receipt["sha256"],
              "analysis_source_sha256": digest(__file__), "tev1_minus_reference": results,
              "scope": "Post-hoc exploratory state-cluster intervals using 2000 resamples and seed 13, as in the earlier matched comparison. No multiplicity correction, fitting, checkpoint selection, release-gate change or new benchmark. Differing training exposures remain unresolved."}
    write_json(DIRECTORY / "paired.json", result)
    lines = ["# Tev1 paired typed-decision comparison", "",
             "This supplements the completed native comparison. All five decisions for each of the 400 states stay together in resampling.",
             "Intervals use 2,000 paired state resamples and seed 13, matching the earlier analysis method.",
             "They are exploratory, without multiplicity correction; training exposures differ.", "",
             "| Tev1 minus reference | Difference, percentage points | 95% state-bootstrap interval |",
             "|---|---:|---:|"]
    for name, stats in results.items():
        lo, hi = stats["ci95_points"]
        lines.append(f"| {name} | {stats['accuracy_delta_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "No thresholds, prompts, checkpoint choices or model routing are fitted from these cases.",
              "[Complete native evaluation](../pilot-tev1-eval-20260927/RESULTS.md), [paired results](paired.json).", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    files.update({name: DIRECTORY / name for name in ("paired.json", "RESULTS.md")})
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    archive = DIRECTORY / "analysis-artifacts.tar.gz"
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
        raise ValueError("Paired analysis archive failed verification")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    write_json(DIRECTORY / "artifact_receipt.json", {"sha256": digest(archive), "bytes": archive.stat().st_size,
                                                   "files_verified": len(actual), "typed_rows_validated": 6000})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
