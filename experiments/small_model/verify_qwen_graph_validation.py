"""Independently recompute full graph parity and archive passing or failing completed runs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"

from parity_evidence import compare_rows, read_rows  # noqa: E402
from structure_probe import digest, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def verify_rows(path, cases, plan):
    indexed = {(c["state_id"], c["qid"], c["k"]): c for c in cases}
    probabilities = read_rows(path, list(indexed))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected_order = [(batch_index, batch["length"], cases[i]) for batch_index, batch in enumerate(plan) for i in batch["indices"]]
    if len(rows) != len(expected_order):
        raise ValueError("Missing or extra real predictions")
    for row, (batch_index, length, case) in zip(rows, expected_order, strict=True):
        if (any(row[key] != case[key] for key in ("state_id", "qid", "k", "qtype", "encoded_tokens", "prompt_sha256"))
                or row["batch_index"] != batch_index or row["padded_tokens"] != length):
            raise ValueError("Prediction order or input provenance differs")
        logp = row["log_probabilities"]
        if (len(logp) != 26 or any(not math.isfinite(v) for v in logp)
                or any(v > -9000 for v in logp[case["k"]:])):
            raise ValueError("Invalid option masking")
        real = logp[:case["k"]]
        expected = [math.exp(v - max(real)) for v in real]
        mass = sum(expected)
        if any(not math.isclose(p, q / mass, abs_tol=1e-12, rel_tol=0)
               for p, q in zip(row["probabilities"], expected, strict=True)):
            raise ValueError("Saved probabilities do not match the readout")
    return probabilities


def main(directory):
    archive = directory / "evidence.tar.gz"
    if archive.exists():
        raise ValueError("Preserve completed graph evidence")
    registration = directory / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    work = json.loads((directory / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if (work["status"] != "complete" or execution["status"] != "complete"
            or any(row["registration_sha256"] != digest(registration) for row in (work, execution))
            or digest(__file__) != spec["sources"][Path(__file__).relative_to(ROOT).as_posix()]):
        raise ValueError("Require a completed registered graph run and frozen analysis")
    for name, expected in spec["sources"].items():
        if work["source_files"].get(name) != expected or digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed source differs from registration")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite differs")
    batch_size = work["batch_size"]
    if (batch_size not in spec["batch_sizes"] or execution["batch_size"] != batch_size
            or work["base_sha256"] != spec["base_sha256"] or work["weights"] != {"adapter": spec["adapter_sha256"]}
            or work["packages"] != spec["packages"] or work["kernel_packages"] != spec["kernel_packages"]
            or work["attention_implementation"] != "sdpa"
            or work["cache"] or work["lora_merged"] or work["projected_letters"] != 26):
        raise ValueError("Model or execution settings differ")
    if digest(directory / "inputs.json") != spec["input_sha256"] or work["inputs_sha256"] != spec["input_sha256"]:
        raise ValueError("Inputs differ from preparation")
    cases = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    if len(cases) != 1390 or len({c["state_id"] for c in cases}) != 278 or work["decisions"] != 1390:
        raise ValueError("Original validation size differs")
    # Rebuild grouping without calling the graph implementation or its plan helper.
    plan = []
    for length in spec["buckets"]:
        selected = []
        for i, case in enumerate(cases):
            if (case["encoded_tokens"] != len(case["token_ids"])
                    or case["prompt_sha256"] != hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()):
                raise ValueError("Input audit is inconsistent")
            fitting = [n for n in spec["buckets"] if n > case["encoded_tokens"]]
            if not fitting:
                raise ValueError("Complete input does not fit a registered shape")
            if min(fitting) == length:
                selected.append(i)
        selected.sort(key=lambda i: (cases[i]["encoded_tokens"], i))
        for start in range(0, len(selected), batch_size):
            indices = selected[start:start + batch_size]
            plan.append({"length": length, "indices": indices, "dummy_rows": batch_size - len(indices)})
    if plan != spec["plans"][str(batch_size)] or len(work["batches"]) != len(plan):
        raise ValueError("Batch plan differs")
    captures, replays, previous_length = [], 0, None
    for index, (expected, actual) in enumerate(zip(plan, work["batches"], strict=True)):
        if expected["length"] != previous_length:
            replays = 0
            captures.append((index, expected["length"]))
            previous_length = expected["length"]
        replays += 1
        if (any(actual[key] != value for key, value in expected.items()) or actual["batch_index"] != index
                or actual["graph_replay"] != replays or actual["peak_allocated_bytes"] > spec["maximum_allocated_cuda_bytes"]):
            raise ValueError("Actual replays or resource receipt differ")
    if len(work["captures"]) != len(captures):
        raise ValueError("Missing or extra graph capture")
    for actual, (index, length) in zip(work["captures"], captures, strict=True):
        if (actual["shape"] != [batch_size, length] or actual["first_batch"] != index or actual["warmup_forwards"] != 3
                or actual["free_cuda_bytes_before"] < spec["minimum_free_cuda_bytes"]):
            raise ValueError("Capture receipt differs")
    expected_ids = [(c["state_id"], c["qid"], c["k"]) for c in cases]
    reference = read_rows(ROOT / spec["reference_rows"], expected_ids)
    values = {}
    for name in ("graph", "uncaptured"):
        path = directory / f"{name}_rows.jsonl"
        if digest(path) != work[name + "_rows_sha256"]:
            raise ValueError("Saved row hash differs")
        values[name] = verify_rows(path, cases, plan)
    full = compare_rows(reference, values["graph"], spec)
    replay = compare_rows(values["uncaptured"], values["graph"], spec)
    passed = full["passed"] and replay["max_probability_delta"] <= spec["max_graph_eager_delta"]
    if (full != work["full_reference_parity"] or replay != work["graph_uncaptured_parity"]
            or passed != work["passed"] or execution["returncode"] != (0 if passed else 2)):
        raise ValueError("Independent metrics or outcome differ")
    result = {"status": "complete", "passed": passed, "batch_size": batch_size,
              "full_reference_parity": full, "graph_uncaptured_parity": replay,
              "real_predictions_per_arm": len(cases), "captures_verified": len(captures), "replays_verified": len(plan),
              "registration_sha256": digest(registration), "analysis_source_sha256": digest(__file__),
              "scope": "Saved-row arithmetic, input membership, capture/replay receipts and source provenance verified. Not a numerical GPU rerun, latency result or runtime promotion."}
    write_json(directory / "verification.json", result)
    (directory / "RESULTS.md").write_text(
        f"# Qwen length-bucket graph parity, batch {batch_size}\n\n"
        f"The complete 1,390-decision validation **{'passed' if passed else 'failed'}** the unchanged numerical limits. "
        f"The independent reader checked both sets of predictions and all {len(plan)} batch receipts.\n\n"
        f"Reference argmax agreement: {full['argmax_agreement']:.8f}; maximum probability change: "
        f"{full['max_probability_delta']:.8f}; mean total variation: {full['mean_total_variation']:.8f}. "
        f"Graph versus identical uncaptured maximum change: {replay['max_probability_delta']:.8f}.\n\n"
        "These are the original frozen weights, with unmerged LoRA, native attention, cache disabled and a 26-letter head. "
        "Every input is complete and left-padded to a strictly larger registered length bucket. "
        "Final incomplete batches duplicate their last row; dummy outputs are excluded. "
        "Each length has one capture reused across its batches.\n\n"
        "This run measures probability parity only. Capture, padding, native-mask construction and buffer copying "
        "must be accounted for in any future speed measurement. A pass does not establish concurrent graph caching, "
        "public accuracy, speed, or a release gate. A failure remains preserved without relaxing its limits.\n",
        encoding="utf-8")
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    files["reference_rows.jsonl"] = ROOT / spec["reference_rows"]
    files["preflight.json"] = directory.parent / "preflight.json"
    manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != manifest:
        raise ValueError("Archive does not reproduce source evidence")
    write_json(directory / "artifact_manifest.json", manifest)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(directory / "artifact_receipt.json", artifact)
    print(json.dumps({"verification": result, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory.resolve())
