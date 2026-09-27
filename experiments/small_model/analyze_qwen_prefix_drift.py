"""Post-hoc descriptive drift analysis; no subgroup can override a full-validation failure."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"batch16": ROOT / "results/pilot-qwen-prefix-validation-20260927/batch-16",
           "batch1": ROOT / "results/pilot-qwen-prefix-single-20260927/batch-01"}
OUTPUT = ROOT / "results/pilot-qwen-prefix-drift-20260927"


def archived(directory):
    receipt = json.loads((directory / "artifact_receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    archive = directory / "evidence.tar.gz"
    if digest(archive) != receipt["sha256"] or archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Complete validation evidence archive changed")
    selected = {"inputs.json", "graph_rows.jsonl", "reference_rows.jsonl", "work.json", "verification.json"}
    saved, actual = {}, {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            data = bundle.extractfile(member).read()
            actual[member.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            if member.name in selected:
                text = data.decode("utf-8")
                saved[member.name] = [json.loads(line) for line in text.splitlines()] if member.name.endswith(".jsonl") else json.loads(text)
    if actual != manifest or len(actual) != receipt["files_verified"] or set(saved) != selected:
        raise ValueError("Archived file membership or content differs")
    verified, work = saved["verification.json"], saved["work.json"]
    if verified["status"] != "complete" or work["status"] != "complete" or verified["real_predictions_per_arm"] != 1390:
        raise ValueError("Require independently verified complete validation")
    return saved, receipt


def indexed(rows):
    output = {}
    for row in rows:
        key = (row["state_id"], row["qid"], row["k"])
        p = row["probabilities"]
        if (key in output or len(p) != row["k"] or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or abs(math.fsum(p) - 1) > 1e-5):
            raise ValueError("Invalid archived probabilities")
        output[key] = row
    if len(output) != 1390:
        raise ValueError("Incomplete probability membership")
    return output


def difference(left, right):
    deltas = [abs(p - q) for p, q in zip(left, right, strict=True)]
    return {"total_variation": math.fsum(deltas) / 2, "maximum_delta": max(deltas),
            "argmax_changed": left.index(max(left)) != right.index(max(right))}


def measurements(rows, comparison):
    values = [row["comparisons"][comparison] for row in rows]
    return {"decisions": len(rows), "mean_total_variation": fmean(v["total_variation"] for v in values),
            "max_probability_delta": max(v["maximum_delta"] for v in values),
            "argmax_changes": sum(v["argmax_changed"] for v in values)}


def main():
    if OUTPUT.exists():
        raise ValueError("Preserve prior drift analysis")
    saved, receipts = {}, {}
    for name, directory in SOURCES.items():
        saved[name], receipts[name] = archived(directory)
    if (saved["batch16"]["inputs.json"] != saved["batch1"]["inputs.json"]
            or saved["batch16"]["reference_rows.jsonl"] != saved["batch1"]["reference_rows.jsonl"]):
        raise ValueError("The graph runs did not use identical original inputs and reference")
    cases = saved["batch16"]["inputs.json"]
    reference = indexed(saved["batch16"]["reference_rows.jsonl"])
    candidates = {name: indexed(value["graph_rows.jsonl"]) for name, value in saved.items()}
    if any(values.keys() != reference.keys() for values in candidates.values()):
        raise ValueError("Different decision membership")
    rows, groups = [], defaultdict(list)
    for case in cases:
        key = (case["state_id"], case["qid"], case["k"])
        p, b16, b1 = reference[key]["probabilities"], candidates["batch16"][key], candidates["batch1"][key]
        padding = b16["native_length"] - case["encoded_tokens"]
        if padding < 0 or b1["native_length"] != case["encoded_tokens"]:
            raise ValueError("Unexpected native prefix positions")
        row = {key: case[key] for key in ("state_id", "qid", "qtype", "k", "encoded_tokens", "prompt_sha256")}
        row.update(native_batch16_left_padding=padding, batch16_bucket=b16["padded_tokens"],
                   batch1_bucket=b1["padded_tokens"], reference_max_probability=max(p),
                   comparisons={"reference_to_batch16": difference(p, b16["probabilities"]),
                                "reference_to_batch1": difference(p, b1["probabilities"]),
                                "batch16_to_batch1": difference(b16["probabilities"], b1["probabilities"])})
        rows.append(row)
        input_bin = next(bound for bound in (128, 256, 512, 1024, 2048) if case["encoded_tokens"] <= bound)
        pad_bin = "zero" if padding == 0 else "1-8" if padding <= 8 else "9-32" if padding <= 32 else "33-128" if padding <= 128 else "129+"
        option_bin = "2" if case["k"] == 2 else "3-7" if case["k"] <= 7 else "8-10" if case["k"] <= 10 else "11+"
        confidence = "<=0.5" if max(p) <= .5 else "0.5-0.8" if max(p) <= .8 else ">0.8"
        for group in ("all", "qtype/" + case["qtype"], f"tokens/upper-{input_bin}", "native_left_pad/" + pad_bin,
                      "options/" + option_bin, "reference_confidence/" + confidence):
            groups[group].append(row)
    comparisons = ("reference_to_batch16", "reference_to_batch1", "batch16_to_batch1")
    results = {group: {comparison: measurements(subset, comparison) for comparison in comparisons}
               for group, subset in sorted(groups.items())}
    for name in ("batch16", "batch1"):
        recomputed = results["all"]["reference_to_" + name]
        previous = saved[name]["verification.json"]["full_reference_parity"]
        if (recomputed["decisions"] != previous["decisions_validated"]
                or not math.isclose(1 - recomputed["argmax_changes"] / len(rows), previous["argmax_agreement"], abs_tol=1e-12, rel_tol=0)
                or any(not math.isclose(recomputed[key], previous[key], abs_tol=1e-12, rel_tol=0)
                       for key in ("mean_total_variation", "max_probability_delta"))):
            raise ValueError("Independent archived aggregate arithmetic differs")
    OUTPUT.mkdir()
    with (OUTPUT / "drift_rows.jsonl").open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")
    result = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(), "cpu_threads": 1,
              "gpu_model_loaded": False, "archives_verified": receipts, "groups": results,
              "rows_sha256": digest(OUTPUT / "drift_rows.jsonl"), "source_sha256": digest(__file__),
              "scope": "Post-hoc descriptive breakdown of complete already-observed validation predictions. No causal attribution, significance test, subgroup-based acceptance, threshold change, model inference, training or speed claim. The failed single-question candidate remains failed; a native full-head batch-one control is still needed."}
    write_json(OUTPUT / "diagnostics.json", result)
    from report_qwen_public import archive_files
    files = {name: OUTPUT / name for name in ("diagnostics.json", "drift_rows.jsonl")}
    files.update({"source/" + name: Path(__file__).with_name(name)
                  for name in (Path(__file__).name, "structure_probe.py", "report_qwen_public.py", "qwen_public_evaluation.py")})
    for name, directory in SOURCES.items():
        files.update({f"inputs/{name}/{filename}": directory / filename
                      for filename in ("verification.json", "artifact_receipt.json", "artifact_manifest.json")})
    artifact = archive_files(OUTPUT, files)
    print(json.dumps({"all": results["all"], "groups_written": len(results), "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
