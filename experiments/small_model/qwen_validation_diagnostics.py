"""Describe a verified continuation's validation changes without changing checkpoint selection."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import fmean

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

from export_qwen_continuation import archive_records, validation_metrics  # noqa: E402
from report_qwen_public import archive_files  # noqa: E402
from structure_probe import digest, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "results/pilot-qwen-continuation-20260927/training"


def measurements(rows):
    gold = [r for r in rows if r["gold"] is not None]
    return {"n": len(rows), "teacher_agreement": fmean(r["student_argmax"] == r["teacher_argmax"] for r in rows),
            "mean_kl": fmean(r["kl"] for r in rows), "mean_max_probability": fmean(max(r["probabilities"]) for r in rows),
            "mean_entropy": fmean(-sum(p * math.log(p) for p in r["probabilities"] if p > 0) for r in rows),
            "gold_count": len(gold), "gold_accuracy": fmean(r["student_argmax"] == r["gold"] for r in gold) if gold else None}


def main(candidate):
    registration = Path(__file__).with_name("qwen_continuation_registration.json")
    analysis = Path(__file__).with_name("qwen_continuation_analysis_registration.json")
    baseline = TRAINING / "attempt-02-baseline-000"
    saved, results, indexed, files = {}, {}, {}, {"registration.json": registration}
    for name, directory in (("baseline", baseline), ("candidate", candidate.resolve())):
        work, verified, rows, artifact = archive_records(directory, digest(registration), digest(analysis))
        metrics, keyed = validation_metrics(rows, indexed.get("baseline"))
        if metrics.keys() != verified["validation"].keys() or any(
                not math.isclose(value, verified["validation"][key], rel_tol=0, abs_tol=1e-12)
                for key, value in metrics.items()):
            raise ValueError("Validation arithmetic changed")
        indexed[name] = keyed
        results[name] = rows
        saved[name] = {"directory": directory.relative_to(ROOT).as_posix(), "step": work["completed_step"],
                       "archive": artifact, "rows_sha256": work["validation"]["rows_sha256"]}
        for filename in ("validation_rows.jsonl", "verification.json", "artifact_receipt.json"):
            files[f"inputs/{name}/{filename}"] = directory / filename
    step = saved["candidate"]["step"]
    if saved["baseline"]["step"] != 0 or step not in (50, 100, 150):
        raise ValueError("Compare the verified original checkpoint with a registered trained validation")
    directory = TRAINING.parent / "validation_diagnostics" / f"step-{step:03d}"
    if directory.exists():
        raise ValueError("Preserve prior validation diagnostics")
    groups = {"all": lambda r: True}
    for qtype in sorted({r["qtype"] for r in results["baseline"]}):
        groups[qtype] = lambda r, ty=qtype: r["qtype"] == ty
    groups.update(choice_le10=lambda r: r["qtype"] == "choice" and r["k"] <= 10,
                  choice_gt10=lambda r: r["qtype"] == "choice" and r["k"] > 10)
    statistics = {}
    for group, predicate in groups.items():
        subsets = {name: [r for r in rows if predicate(r)] for name, rows in results.items()}
        if not subsets["baseline"]:
            continue
        statistics[group] = {name: measurements(rows) for name, rows in subsets.items()}
        statistics[group]["candidate_minus_baseline"] = {
            key: statistics[group]["candidate"][key] - statistics[group]["baseline"][key]
            for key in ("teacher_agreement", "mean_kl", "mean_max_probability", "mean_entropy")}
    result = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(), "inputs": saved,
              "groups": statistics, "cpu_threads": 1, "gpu_model_loaded": False,
              "scope": "Post-hoc descriptive analysis of original validation only. No causal attribution, significance claim, public scores, data reweighting, training change or checkpoint choice. The frozen four-checkpoint selector and all release gates remain unchanged."}
    sources = ["qwen_validation_diagnostics.py", "export_qwen_continuation.py", "report_qwen_public.py",
               "qwen_public_evaluation.py", "structure_probe.py"]
    result["source_hashes"] = {name: digest(Path(__file__).with_name(name)) for name in sources}
    files.update({"source/" + name: Path(__file__).with_name(name) for name in sources})
    directory.mkdir(parents=True)
    write_json(directory / "diagnostics.json", result)
    files["diagnostics.json"] = directory / "diagnostics.json"
    artifact = archive_files(directory, files)
    print(json.dumps({"groups": statistics, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    main(parser.parse_args().candidate)
