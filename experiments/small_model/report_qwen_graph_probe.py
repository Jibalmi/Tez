"""Recompute and archive the four-case graph compatibility result before broader validation."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

from parity_evidence import compare_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-cuda-graph-attempt2-20260927"


def main():
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve completed probe evidence")
    registration = DIRECTORY / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    probe = json.loads((DIRECTORY / "probe.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    if (probe["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or any(row["registration_sha256"] != digest(registration) for row in (probe, execution))
            or not probe["passed"] or probe["base_sha256"] != spec["base_sha256"]):
        raise ValueError("Require the complete registered compatibility probe")
    for name, expected in spec["sources"].items():
        if probe["source_files"].get(name) != expected or digest(DIRECTORY / "executed_source" / name) != expected:
            raise ValueError("Executed graph source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen graph prerequisite changed")
    if len(probe["rows"]) != 3:
        raise ValueError("Missing batch-order check")
    arms = ("full_probabilities", "uncaptured_probabilities", "graph_probabilities")
    values = {name: {} for name in arms}
    for index, (row, order) in enumerate(zip(probe["rows"], ([0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]), strict=True)):
        counts = [len(spec["cases"][i]["options"]) for i in order]
        if row["order"] != order or row["option_counts"] != counts:
            raise ValueError("Case order or option membership differs")
        for name in arms:
            if len(row[name]) != 4:
                raise ValueError("Missing probe row")
            for case, count, vector in zip(order, counts, row[name], strict=True):
                p = vector[:count]
                if (len(vector) != max(counts) or any(v != 0 for v in vector[count:])
                        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > 1e-5):
                    raise ValueError("Invalid masked probability vector")
                values[name][case, index, count] = p
    limits = {"argmax_agreement_min": .99, "max_probability_delta_max": .04, "mean_total_variation_max": .005}
    full = compare_rows(values[arms[0]], values[arms[2]], limits)
    uncaptured = compare_rows(values[arms[1]], values[arms[2]], limits)
    metrics = {key: full[key] for key in ("argmax_agreement", "max_probability_delta", "mean_total_variation")}
    metrics["max_graph_vs_uncaptured_delta"] = uncaptured["max_probability_delta"]
    if (not full["passed"] or metrics["max_graph_vs_uncaptured_delta"] > 1e-5
            or any(not math.isclose(value, probe["metrics"][key], rel_tol=0, abs_tol=1e-12) for key, value in metrics.items())):
        raise ValueError("Recomputed compatibility decision differs")
    result = {"status": "complete", "passed": True, "metrics": metrics, "probability_vectors_validated": 36,
              "registration_sha256": digest(registration), "analysis_source_sha256": digest(__file__),
              "scope": "Four non-benchmark cases in three batch orders only. Exact probability agreement here does not establish full-validation parity, variable-length serving, latency or a release gate. Broader validation must precede timing or promotion."}
    write_json(DIRECTORY / "analysis.json", result)
    (DIRECTORY / "RESULTS.md").write_text(
        "# CUDA graph compatibility result\n\n"
        "The four fixed non-benchmark cases passed in original, reversed and restored batch order. "
        "An independent reader checked all 36 saved probability vectors and their case membership.\n\n"
        f"Full-reference argmax agreement: {metrics['argmax_agreement']:.6f}; maximum probability change: {metrics['max_probability_delta']:.8f}; "
        f"mean total variation: {metrics['mean_total_variation']:.8f}. "
        f"Captured versus identical uncaptured maximum change: {metrics['max_graph_vs_uncaptured_delta']:.8f}.\n\n"
        "This is compatibility evidence for four cases at one tensor shape, not a speed measurement. "
        "Full validation and the actual serving shapes remain to be tested before latency or promotion. "
        "The first metadata-cache failure remains preserved separately.\n\n"
        "[Raw probabilities](probe.json), [independent analysis](analysis.json), [registration](registration.json).\n",
        encoding="utf-8")
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    for name in ("report_qwen_graph_probe.py", "parity_evidence.py", "test_parity_evidence.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
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
        raise ValueError("Graph probe archive differs")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"metrics": metrics, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
