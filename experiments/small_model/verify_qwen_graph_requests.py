"""Reconstruct the complete multi-shape request evidence without running a GPU model."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import fmean

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  TOKENIZERS_PARALLELISM="false")

from qwen_graph_request_probe import AUDIT, DIRECTORY, ROOT, audit_request, json_sha, schedule  # noqa: E402
from structure_probe import digest, write_json  # noqa: E402


def probabilities(values, k):
    if (len(values) != k or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not math.isclose(math.fsum(math.exp(v) for v in values), 1, rel_tol=0, abs_tol=1e-5)):
        raise ValueError("Invalid normalized option log probabilities")
    exp = [math.exp(v - max(values)) for v in values]
    return [v / math.fsum(exp) for v in exp]


def summarize(pairs, spec):
    deltas = [[abs(p - q) for p, q in zip(left, right, strict=True)] for left, right in pairs]
    result = {"decisions": len(pairs),
              "argmax_agreement": fmean(left.index(max(left)) == right.index(max(right)) for left, right in pairs),
              "max_probability_delta": max(map(max, deltas)),
              "mean_total_variation": fmean(math.fsum(row) / 2 for row in deltas)}
    result["passed"] = (result["argmax_agreement"] >= spec["argmax_agreement_min"]
                        and result["max_probability_delta"] <= spec["max_probability_delta_max"]
                        and result["mean_total_variation"] <= spec["mean_total_variation_max"])
    return result


def verify_saved_rows(rows, audits, spec):
    if not rows or len(rows) != len(audits):
        raise ValueError("Incomplete request membership")
    reference, uncaptured = {}, {}
    counts, captures = {}, []
    for request_index, (row, audit) in enumerate(zip(rows, audits, strict=True)):
        if (any(row[key] != audit[key] for key in ("request", "n_questions", "phase"))
                or row["audit_sha256"] != json_sha(audit) or len(row["batches"]) != len(audit["batches"])):
            raise ValueError("Request order, input audit or batch count changed")
        for actual, expected in zip(row["batches"], audit["batches"], strict=True):
            shape = (expected["batch_size"], expected["length"])
            if list(shape) not in spec["allowed_shapes"]:
                raise ValueError("Unregistered graph shape")
            if shape not in counts:
                counts[shape] = 0
                captures.append({"shape": list(shape), "request_index": request_index})
            counts[shape] += 1
            if actual != {**expected, "graph_replay": counts[shape]}:
                raise ValueError("Native membership or graph replay history changed")
        if row["resident_shapes"] != [list(shape) for shape in counts] or row["captures_total"] != len(counts):
            raise ValueError("Graph lifetime or capture count changed")
        arms = ("reference_log_probabilities", "graph_log_probabilities", "uncaptured_log_probabilities")
        if any(len(row[arm]) != row["n_questions"] for arm in arms) or len(audit["cases"]) != row["n_questions"]:
            raise ValueError("Incomplete decision membership")
        size = str(row["n_questions"])
        for index, case in enumerate(audit["cases"]):
            full, graph, eager = (probabilities(row[arm][index], case["k"]) for arm in arms)
            reference.setdefault(size, []).append((full, graph))
            uncaptured.setdefault(size, []).append((eager, graph))
    return ({n: summarize(pairs, spec) for n, pairs in reference.items()},
            {n: summarize(pairs, spec) for n, pairs in uncaptured.items()}, captures)


def main():
    directory = DIRECTORY / "validation"
    if (directory / "evidence.tar.gz").exists():
        raise ValueError("Preserve existing verification")
    registration = directory / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    work = json.loads((directory / "work.json").read_text(encoding="utf-8"))
    if (execution["status"] != "complete" or work["status"] != "complete"
            or any(row["registration_sha256"] != digest(registration) for row in (execution, work))
            or digest(__file__) != spec["sources"][Path(__file__).relative_to(ROOT).as_posix()]):
        raise ValueError("Require complete immutable execution and independent analysis")
    for name, expected in spec["sources"].items():
        if work["source_files"].get(name) != expected or digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite differs")
    if (work["threads"] != 4 or work["capability"] != [12, 0] or work["packages"] != spec["packages"]
            or work["kernel_packages"] != spec["kernel_packages"] or work["attention_implementation"] != "sdpa"
            or work["reference_cache"] is not True or work["graph_cache"] is not False or work["lora_merged"]
            or work["projected_letters"] != 26 or work["base_dtype"] != "torch.bfloat16"
            or work["adapter_dtypes"] != ["torch.float32"] or work["autocast_dtype"] != "torch.bfloat16"
            or work["weights"] != {"adapter": spec["adapter_sha256"]} or work["base_sha256"] != spec["base_sha256"]
            or work["peak_allocated_bytes"] > spec["maximum_allocated_cuda_bytes"]):
        raise ValueError("Actual hardware, model or resource settings differ")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True)
    audits = [audit_request(tokenizer, request, spec["buckets"]) for request in schedule()]
    if audits != json.loads(AUDIT.read_text(encoding="utf-8")) or work["input_audit_sha256"] != digest(AUDIT):
        raise ValueError("Complete prompt/token audit differs")
    path = directory / "requests.jsonl"
    if digest(path) != work["requests_sha256"]:
        raise ValueError("Output rows changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != 370 or work["requests_completed"] != 370 or sum(r["n_questions"] for r in rows) != 3760
            or work["output_lifetime_checks"] != 3760):
        raise ValueError("Incomplete registered workload")
    full, eager, captures = verify_saved_rows(rows, audits, spec)
    for independent, saved in ((full, work["reference_parity_by_request_size"]), (eager, work["graph_uncaptured_by_request_size"])):
        if independent.keys() != saved.keys():
            raise ValueError("Missing request-size metrics")
        for n, values in independent.items():
            if values.keys() != saved[n].keys() or any(not math.isclose(v, saved[n][k], rel_tol=0, abs_tol=1e-12)
                                                      for k, v in values.items()):
                raise ValueError("Independent arithmetic differs")
    if len(captures) != len(work["captures"]) or sorted(c["shape"] for c in captures) != spec["allowed_shapes"]:
        raise ValueError("Incomplete shape coverage")
    for expected, actual in zip(captures, work["captures"], strict=True):
        if (any(actual[key] != value for key, value in expected.items()) or actual["warmup_forwards"] != 3
                or actual["free_cuda_bytes_before"] < spec["minimum_free_cuda_bytes"]
                or not math.isfinite(actual["operational_seconds"]) or actual["operational_seconds"] < 0):
            raise ValueError("Capture resource or lifetime receipt differs")
    passed = all(v["passed"] for v in full.values()) and all(
        v["max_probability_delta"] <= spec["max_graph_eager_delta"] for v in eager.values())
    if passed != work["passed"] or execution["returncode"] != (0 if passed else 2):
        raise ValueError("Independent outcome differs")
    result = {"status": "complete", "passed": passed, "verified_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": digest(registration), "verifier_sha256": digest(__file__),
              "requests": 370, "decisions_per_arm": 3760, "captures": captures,
              "reference_parity_by_request_size": full, "graph_uncaptured_by_request_size": eager,
              "scope": "Independent CPU saved-row arithmetic, complete input audit and graph lifetime receipts. No latency or release claim."}
    write_json(directory / "verification.json", result)
    from report_qwen_public import archive_files
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    files.update({"input_audit.json": AUDIT, "preflight.json": DIRECTORY / "preflight.json"})
    artifact = archive_files(directory, files)
    print(json.dumps({"verification": result, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
