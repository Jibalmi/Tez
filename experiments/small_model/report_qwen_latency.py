"""Independently validate and archive the frozen laptop readout timing, including ineligible runs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-laptop-qwen-readout-20260927"


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_trace(rows, recorded):
    """Reconstruct every expected request and all timing aggregates from the raw trace."""
    cursor, vectors, sizes = 0, [], {}
    for n, reps in ((1, 40), (5, 40), (10, 40), (50, 15)):
        timings = []
        for phase, count in (("warmup", 5), ("timed", reps)):
            for _ in range(count):
                if cursor >= len(rows):
                    raise ValueError("Incomplete raw request sequence")
                row = rows[cursor]
                cursor += 1
                if (row["request"], row["n_questions"], row["phase"]) != (cursor, n, phase):
                    raise ValueError("Raw protocol or request membership changed")
                if not math.isfinite(row["ms"]) or row["ms"] <= 0:
                    raise ValueError("Invalid raw elapsed time")
                groups = row["log_probabilities"]
                if len(groups) != 1 or len(groups[0]) != n:
                    raise ValueError("Incomplete probability output")
                for i, lp in enumerate(groups[0]):
                    if len(lp) != (3 if i % 2 == 0 else 2) or any(not math.isfinite(v) or v > 1e-6 for v in lp):
                        raise ValueError("Invalid option membership or log probability")
                    p = [math.exp(v) for v in lp]
                    if abs(sum(p) - 1) > 1e-5:
                        raise ValueError("Probability normalization changed")
                    vectors.append({"request": cursor, "question": i, "phase": phase, "p": p})
                if phase == "timed":
                    timings.append(row["ms"])
        p50 = percentile(timings, .5)
        measured = {"n_questions": n, "reps": reps, "p50_ms": round(p50, 2),
                    "p95_ms": round(percentile(timings, .95), 2), "mean_ms": round(fmean(timings), 2),
                    "ms_per_question": round(p50 / n, 3)}
        if any(recorded["sizes"][str(n)][key] != value for key, value in measured.items()):
            raise ValueError("Reported aggregate does not reproduce from raw samples")
        sizes[str(n)] = measured
    if cursor != len(rows):
        raise ValueError("Extra raw requests")
    return vectors, sizes


def eligibility(record):
    samples = record["thermal"]
    expected = ["start", "after n=1", "before n=5", "after n=5", "before n=10", "after n=10", "before n=50", "after n=50"]
    if ([row.get("at") for row in samples] != expected
            or any(not {"temp_c", "sw_thermal", "hw_thermal"}.issubset(row) for row in samples)):
        raise ValueError("Missing registered thermal telemetry")
    hot = [row for row in samples if row["temp_c"] > 70]
    slowdown = [row for row in samples if row["sw_thermal"] or row["hw_thermal"]]
    starts = [row for row in samples if row["at"] == "start" or row["at"].startswith("before")]
    cooled = all(row["temp_c"] <= 65 and row["waited_s"] <= 90 for row in starts)
    # A foreign process present before model loading is unambiguously external to this run.
    # The legacy trace has no own PID, so a clean before snapshot alone could not clear the after snapshot.
    before, after = record["vram_before"], record["vram_after"]
    if any("processes" not in snapshot for snapshot in (before, after)):
        raise ValueError("Missing registered memory telemetry")
    foreign = [p for p in before["processes"] if p["kind"].lower().startswith("dedicated") and p["mb"] > 300]
    if not foreign:
        raise ValueError("This retrospective verifier needs an explicit own PID to certify a clean after snapshot")
    if record["throttled"] != bool(hot or slowdown) or record["contaminated"] is not True:
        raise ValueError("Recorded eligibility flags differ from independent audit")
    return {"eligible": False, "cooldown_starts_passed": cooled,
            "temperature_limit_exceeded": [{"at": r["at"], "temp_c": r["temp_c"]} for r in hot],
            "hardware_or_software_thermal_slowdown_observed": bool(slowdown),
            "foreign_processes_over_300_mb_before_load": foreign,
            "reason": "Foreign dedicated GPU memory exceeds the unchanged 300 MB limit; recorded temperatures also exceed 70 C."}


def main():
    output = DIRECTORY / "analysis.json"
    archive = DIRECTORY / "artifact.tar.gz"
    if output.exists() or archive.exists():
        raise ValueError("Preserve the prior analysis and artifact")
    spec_path = DIRECTORY / "registration.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    if (execution["status"] != "complete" or execution["registration_sha256"] != digest(spec_path)
            or [(r["readout"], r["returncode"]) for r in execution["stages"]] != [("full", 0), ("restricted", 0)]):
        raise ValueError("Incomplete or changed execution")
    for name, sha in spec["sources"].items():
        if digest(DIRECTORY / "executed_source" / name) != sha:
            raise ValueError("Executed source hash changed")
    for name, sha in spec["prerequisites"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen model prerequisite changed")
    base = json.loads((DIRECTORY / "base_weight_verification.json").read_text(encoding="utf-8"))
    if (not base["registered_hash_matched"] or base["revision"] != spec["base_revision"]
            or {entry["sha256"] for entry in base["files"].values()} != {spec["base_sha256"]}):
        raise ValueError("Base weight verification changed")
    results = json.loads((DIRECTORY / "latency.json").read_text(encoding="utf-8"))
    vectors, scores = {}, {}
    for readout in ("full", "restricted"):
        tag = "c-qwen08-30k_cuda" + ("_restricted" if readout == "restricted" else "")
        record = results[tag]
        if (record["readout"] != readout or record["warmup"] != 5 or record["reps"] != 40 or record["reps50"] != 15
                or record["weights_sha256"]["adapter"] not in spec["prerequisites"].values()):
            raise ValueError("Model or protocol mismatch")
        raw_path = DIRECTORY / record["raw_samples_file"]
        if digest(raw_path) != record["raw_samples_sha256"]:
            raise ValueError("Raw trace changed")
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        vectors[readout], sizes = validate_trace(rows, record)
        scores[readout] = {"sizes": sizes, "eligibility": eligibility(record), "raw_sha256": digest(raw_path),
                           "requests": len(rows), "timed_requests": sum(r["phase"] == "timed" for r in rows),
                           "timed_decisions": sum(r["phase"] == "timed" for r in vectors[readout]),
                           "gate_pass": False}
    differences, tv, agreement = [], [], []
    for a, b in zip(vectors["full"], vectors["restricted"], strict=True):
        if any(a[key] != b[key] for key in ("request", "question", "phase")):
            raise ValueError("Readout request alignment differs")
        diff = [abs(p - q) for p, q in zip(a["p"], b["p"], strict=True)]
        differences.extend(diff)
        tv.append(sum(diff) / 2)
        agreement.append(max(range(len(a["p"])), key=a["p"].__getitem__) == max(range(len(b["p"])), key=b["p"].__getitem__))
    parity = {"decisions_including_warmup": len(tv), "argmax_agreement": fmean(agreement),
              "max_probability_delta": max(differences), "mean_total_variation": fmean(tv),
              "scope": "Timing-ticket diagnostic only; the pre-existing frozen validation parity remains separate."}
    summary = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(spec_path), "analysis_source_sha256": digest(__file__),
               "scores": scores, "readout_probability_diagnostic": parity,
               "conclusion": "Both executions completed, but neither is eligible for the unchanged laptop latency gate. No clean speedup is established.",
               "telemetry_limit": "Temperatures were sampled at group boundaries, not continuously. The legacy throttled flag includes any reading over 70 C, even with both hardware flags false.",
               "environment_limit": "model_info.kernels describes original H200 training, not this laptop's inference. Logs record current PyTorch fallback dispatch. The worker did not separately persist inference package versions or actual Torch thread count."}
    write_json(output, summary)
    lines = ["# Qwen laptop timing: completed, ineligible", "", summary["conclusion"], "",
             "Frozen Qwen 0.8B LoRA, BF16 autocast, original eager kernels on RTX 5080 Laptop GPU.",
             "Same fresh-ticket protocol, 5 warmups per size; 40 timed requests at 1/5/10 questions and 15 at 50.",
             "Raw requests, response vectors, source hashes and timing aggregates were independently checked.", "",
             "| Diagnostic p50, ms per request | Full readout | Restricted readout |", "|---|---:|---:|"]
    for n in ("1", "5", "10", "50"):
        lines.append(f"| {n} questions | {scores['full']['sizes'][n]['p50_ms']:.2f} | {scores['restricted']['sizes'][n]['p50_ms']:.2f} |")
    lines += ["", "These are diagnostic measurements under ineligible conditions, not a clean speed comparison.",
              "Full readout reached 73 C and restricted readout 76 C, above the registered 70 C ceiling.",
              "Neither reported hardware/software thermal slowdown at the recorded samples; the legacy `throttled` flag also covers temperature-limit violations.",
              "Windows Desktop Window Manager held 667 MB of dedicated GPU memory before both runs, above the unchanged 300 MB foreign-process limit.",
              "No unrelated processes were changed. No threshold was relaxed.", "",
              "The latency gates remain 30 ms for one question and 2.8 ms/question for 50. Both diagnostic timings exceed them.",
              f"Across {len(tv):,} decisions including warmups, readout argmax agreement was {100 * fmean(agreement):.4f}%,",
              f"maximum probability difference {max(differences):.8f}, and mean total variation {fmean(tv):.8f}.",
              "This repeated-ticket diagnostic does not replace the broader validation parity test.", "",
              summary["environment_limit"], "", "[Analysis](analysis.json), [raw timing metadata](latency.json), [protocol](registration.json).", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
             if p.is_file() and p.name not in {".gitignore", "artifact_manifest.json", "artifact_receipt.json", archive.name}
             and "__pycache__" not in p.parts}
    files["analysis_source/report_qwen_latency.py"] = Path(__file__)
    for name in spec["prerequisites"]:
        if not name.endswith(".safetensors"):
            files["prerequisites/" + name] = ROOT / name
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive entry")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("Archive did not verify")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
               "raw_requests_verified": 310, "timed_decisions_verified": 2780}
    write_json(DIRECTORY / "artifact_receipt.json", receipt)
    print(json.dumps({"artifact": receipt, "parity": parity, "conclusion": summary["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
