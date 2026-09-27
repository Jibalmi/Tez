"""Independently recompute the bounded FLA precision probe and retain its negative result."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile

from parity_evidence import compare_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-fla-precision-probe-20260927"


def main():
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists() or (DIRECTORY / "analysis.json").exists():
        raise ValueError("Preserve completed analysis")
    registration = DIRECTORY / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    probe = json.loads((DIRECTORY / "probe.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    if (probe["status"] != "complete" or probe["registration_sha256"] != digest(registration)
            or execution["registration_sha256"] != digest(registration)
            or execution["returncode"] != (0 if probe["passed"] else 2)
            or execution["status"] != ("complete" if probe["passed"] else "failed")
            or probe["weights"] != {"adapter": spec["adapter_sha256"]} or probe["base_sha256"] != spec["base_sha256"]
            or not probe["original_cache_setting"] or probe["packages"] != spec["packages"]):
        raise ValueError("Incomplete or inconsistent precision probe")
    for name, expected in spec["sources"].items():
        if probe["sources"].get(name) != expected or digest(DIRECTORY / "executed_source" / name) != expected:
            raise ValueError("Executed source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite changed")
    prep_path = ROOT / spec["preparation"]
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    for name, expected in prep["installed_file_hashes"].items():
        if digest(ROOT / prep["overlay"] / name) != expected:
            raise ValueError("Pinned kernel source or dependency changed")
    arms = ("torch_fallback", "fla_native", "fla_fp32_inputs")
    if [row["arm"] for row in probe["rows"]] != list(arms):
        raise ValueError("Incomplete or reordered probe arms")
    predictions = {}
    for row in probe["rows"]:
        name = row["arm"]
        if len(row["batches"]) != 2:
            raise ValueError("Incomplete batch order checks")
        values = {}
        for order_index, (batch, order) in enumerate(zip(row["batches"], ([0, 1, 2, 3], [3, 2, 1, 0]), strict=True)):
            counts = [len(spec["cases"][i]["options"]) for i in order]
            if batch["order"] != order or batch["option_counts"] != counts or len(batch["probabilities"]) != 4:
                raise ValueError("Probe cases or option counts changed")
            for case, count, probability in zip(order, counts, batch["probabilities"], strict=True):
                p = probability[:count]
                if (len(p) != count or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                        or abs(sum(p) - 1) > 1e-5):
                    raise ValueError("Invalid saved probability vector")
                values[case, order_index, count] = p
        predictions[name] = values
        dispatch = probe["dispatch"][name]["torch_chunk_gated_delta_rule"]
        expected_source = spec["qwen_source_sha256"] if name == "torch_fallback" else prep["installed_file_hashes"]["fla/ops/gated_delta_rule/chunk.py"]
        if dispatch["calls"] != 36 or dispatch["unwrapped_source_sha256"] != expected_source:
            raise ValueError("Actual attention dispatch is not verified")
        if name == "fla_fp32_inputs" and dispatch["called_dtypes"] != ["torch.float32"] * 5:
            raise ValueError("The promoted arm did not receive FP32 inputs")
    metrics = {arm: compare_rows(predictions["torch_fallback"], predictions[arm], spec["acceptance"])
               for arm in arms[1:]}
    for arm, result in metrics.items():
        for key, expected in probe["metrics"][arm].items():
            if not math.isclose(result[key], expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Recomputed probability metrics differ")
    if metrics["fla_fp32_inputs"]["passed"] != probe["passed"]:
        raise ValueError("Acceptance decision differs")
    result = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": digest(registration), "probe_sha256": digest(DIRECTORY / "probe.json"),
              "analysis_source_sha256": digest(__file__), "metrics": metrics,
              "probability_vectors_validated": sum(map(len, predictions.values())),
              "scope": "Four non-benchmark cases in two batch orders only. FP32 inputs fail the unchanged mean-TV limit; do not advance this candidate. Native FLA passing this small probe does not supersede its failed full-validation result. No quality, throughput or latency claim."}
    write_json(DIRECTORY / "analysis.json", result)
    lines = ["# FLA precision probe result", "", "All 24 probability vectors were independently checked across three arms and two orders of four non-benchmark inputs.",
             "The reader recomputed the numerical checks and verified registered source hashes and 36 attention calls per arm.", "",
             "| Arm versus Torch fallback | Argmax agreement | Maximum probability change | Mean total variation | Passed |",
             "|---|---:|---:|---:|---|"]
    for arm, values in metrics.items():
        lines.append(f"| {arm} | {values['argmax_agreement']:.6f} | {values['max_probability_delta']:.8f} | {values['mean_total_variation']:.8f} | {values['passed']} |")
    lines += ["", "Unchanged limits: agreement at least 0.99, maximum change at most 0.04, mean TV at most 0.005.",
              "FP32 inputs did not resolve drift and this candidate does not advance. FP32 dtype is not IEEE-only kernel arithmetic.",
              "Native FLA passing these four cases does not replace its [failed full validation](../pilot-fla-qwen-attempt2-20260927/RESULTS.md).",
              "The cause of the broader drift is not isolated by this experiment. No latency or quality improvement was measured.",
              "[Full analysis](analysis.json), [raw vectors and dispatch receipt](probe.json), [registration](registration.json).", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    files["inputs/kernel_preparation.json"] = prep_path
    for name in ("report_fla_precision.py", "parity_evidence.py", "test_parity_evidence.py", "test_fla_precision_probe.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    for name in ("fla/ops/gated_delta_rule/chunk.py", "fla/ops/gated_delta_rule/fused_recurrent.py"):
        files["kernel_source/" + name] = ROOT / prep["overlay"] / name
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
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"metrics": metrics, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
