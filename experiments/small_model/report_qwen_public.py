"""Verify every matched public probability and retain complete evidence before comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from statistics import fmean
import tarfile

from qwen_public_evaluation import (
    ARMS, AUDIT, DIRECTORY, PHASES, REGISTRATION, ROOT, SUITE, audit_case, bind_inputs, inputs, task_phase,
)
from structure_probe import digest, write_json


def check_rows(path, task, cases, expected_audit, reverse=False):
    """Check identities, labels, complete input audit, and both coordinate systems independently."""
    from competitor_overlap import read_checked
    rows = read_checked(path, cases)
    for case in cases:
        row = rows[case["id"]]
        for key in ("id", "lang", "qtype", "gold", "workflow", "group", "family"):
            if row.get(key) != case.get(key):
                raise ValueError("Public row metadata changed")
        if row["task"] != task or row["k"] != len(case["options"]) or row["extra"] != case.get("extra", {}):
            raise ValueError("Public row task, option count or soft target changed")
        for key, value in expected_audit[case["id"]].items():
            if row.get(key) != value:
                raise ValueError("Complete public input audit differs")
        z = row["presented_log_probabilities"]
        if len(z) != row["k"] or not all(map(math.isfinite, z)):
            raise ValueError("Invalid presented log probabilities")
        values = [math.exp(value - max(z)) for value in z]
        p = [value / sum(values) for value in values]
        order = list(reversed(range(row["k"]))) if reverse else list(range(row["k"]))
        if row["option_order"] != order or row["logits"] != [z[i] for i in order]:
            raise ValueError("Incorrect canonical option mapping")
        for actual, saved in ((p, row["presented_probabilities"]), ([p[i] for i in order], row["probabilities"])):
            if len(saved) != len(actual) or any(not math.isfinite(b) or abs(a - b) > 1e-12
                                               for a, b in zip(actual, saved, strict=True)):
                raise ValueError("Readout does not reproduce saved probabilities")
    return rows


def task_metrics(task, cases, rows):
    from report_gliner2 import scoped_metrics
    result = scoped_metrics(task, cases, rows)
    if task == "typed_decisions":
        result["by_workflow"] = {}
        result["by_qtype"] = {}
        for field in ("workflow", "qtype"):
            for name in sorted({c[field] for c in cases}):
                subset = [c for c in cases if c[field] == name]
                result["by_" + field][name] = scoped_metrics(task, subset, {c["id"]: rows[c["id"]] for c in subset})
    if task.startswith("semif_"):
        module_spec = importlib.util.spec_from_file_location("frozen_semif", ROOT / "data/semif/evaluate.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        gold = module.read_jsonl(ROOT / "data/semif" / (task.split("_", 1)[1] + ".jsonl"))
        result["semif"] = module.evaluate(gold, [{"id": c["id"], "probabilities": rows[c["id"]]["probabilities"]} for c in cases])
    return result


def archive_files(directory, files):
    archive = directory / "evidence.tar.gz"
    if archive.exists() or any(p.suffix in {".pt", ".safetensors", ".bin", ".npy", ".npz"} for p in files.values()):
        raise ValueError("Preserve existing archive and exclude private weights/caches")
    manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected public evidence member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != manifest:
        raise ValueError("Public evidence archive differs")
    write_json(directory / "artifact_manifest.json", manifest)
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(manifest)}
    write_json(directory / "artifact_receipt.json", receipt)
    return receipt


def report(arm, phase, spec):
    directory = DIRECTORY / arm / phase
    if (directory / "verification.json").exists() or (directory / "evidence.tar.gz").exists():
        raise ValueError("Preserve completed public verification")
    suite = inputs(spec)
    bindings = bind_inputs(spec)
    work = json.loads((directory / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if (work["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or digest(directory / "registration.json") != digest(REGISTRATION)
            or json.loads((directory / "bindings.json").read_text(encoding="utf-8")) != bindings):
        raise ValueError("Require a complete phase with unchanged registered model bindings")
    for receipt in (work, execution):
        if (receipt["arm"] != arm or receipt["phase"] != phase or receipt["registration_sha256"] != digest(REGISTRATION)
                or receipt["bindings_sha256"] != digest(directory / "bindings.json")):
            raise ValueError("Public phase provenance differs")
    for name, expected in spec["sources"].items():
        if work["source_files"].get(name) != expected or digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed public source differs")
    expected_settings = {"packages": spec["packages"], "kernel_packages": spec["kernel_packages"],
                         "weights": {"adapter": bindings["arms"][arm]["adapter_sha256"]},
                         "base_sha256": spec["base_sha256"], "selected_step": bindings["selected_step"],
                         "threads": 4, "batch_size": 16, "dtype": "bf16", "lora_merged": False, "cache": True,
                         "attention_implementation": "sdpa", "temperature": 1.0, "context_bound": 4096,
                         "prompt_layout": "question_first", "output_head": "full_vocabulary",
                         "base_parameter_dtypes": ["torch.bfloat16"], "adapter_parameter_dtypes": ["torch.float32"],
                         "model_training": False, "compute_capability": [12, 0], "cuda_version": "13.0"}
    if any(work[key] != value for key, value in expected_settings.items()) or "RTX 5080 Laptop" not in work["gpu"]:
        raise ValueError("Recorded local inference settings differ")
    selected_tasks = {task: cases for task, cases in suite.items() if task_phase(task) == phase}
    if work["tasks"].keys() != selected_tasks.keys():
        raise ValueError("Incomplete public phase task membership")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    scores = {}
    primary_count = reversed_count = 0
    for task, cases in selected_tasks.items():
        orders = ("primary", "reversed") if task.startswith("massive_") else ("primary",)
        if set(work["tasks"][task]) != set(orders):
            raise ValueError("Incomplete option-order coverage")
        rows_by_order = {}
        for order in orders:
            expected = {c["id"]: audit_case(tokenizer, c, order == "reversed") for c in cases}
            if expected != audit[task][order]:
                raise ValueError("Independent public token audit differs")
            path = directory / f"rows_{task}_{order}.jsonl"
            rows_by_order[order] = check_rows(path, task, cases, expected, order == "reversed")
            if work["tasks"][task][order] != {"sha256": digest(path), "rows": len(cases)}:
                raise ValueError("Public raw output receipt differs")
        scores[task] = task_metrics(task, cases, rows_by_order["primary"])
        primary_count += len(cases)
        if "reversed" in rows_by_order:
            reversed_count += len(cases)
            scores[task]["order_flip"] = fmean(rows_by_order["primary"][c["id"]]["pred"] !=
                                               rows_by_order["reversed"][c["id"]]["pred"] for c in cases)
    result = {"status": "complete", "arm": arm, "phase": phase, "scores": scores,
              "primary_decisions": primary_count, "reversed_decisions": reversed_count,
              "registration_sha256": digest(REGISTRATION), "bindings_sha256": digest(directory / "bindings.json"),
              "actual_settings": {**expected_settings, "gpu": work["gpu"]}, "scope": spec["scope"]}
    write_json(directory / "verification.json", result)
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    files.update({"inputs/eval-suites.json": SUITE, "inputs/public_input_audit.json": AUDIT})
    artifact = archive_files(directory, files)
    print(json.dumps({"verification": result, "artifact": artifact}, indent=2))


def compare(spec):
    from compare_round1 import paired_typed_delta
    from competitor_overlap import read_checked
    target = DIRECTORY / "comparison"
    if target.exists():
        raise ValueError("Preserve completed matched comparison")
    suite = inputs(spec)
    bindings = bind_inputs(spec)
    scores, predictions, files = {}, {}, {"registration.json": REGISTRATION, "eval-suites.json": SUITE}
    for arm in ARMS:
        scores[arm], predictions[arm] = {}, {}
        primary_count = reversed_count = 0
        for phase in PHASES:
            directory = DIRECTORY / arm / phase
            result = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
            artifact = json.loads((directory / "artifact_receipt.json").read_text(encoding="utf-8"))
            manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
            if (result["status"] != "complete" or result["registration_sha256"] != digest(REGISTRATION)
                    or digest(directory / "evidence.tar.gz") != artifact["sha256"]
                    or json.loads((directory / "bindings.json").read_text(encoding="utf-8")) != bindings):
                raise ValueError("Require all eight verified phases under the same selected-model binding")
            # Re-read the immutable archive, rather than trusting a loose manifest or current row files.
            with tarfile.open(directory / "evidence.tar.gz", "r:gz") as bundle:
                members = bundle.getmembers()
                if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
                    raise ValueError("Unexpected public phase archive member")
                actual = {m.name: {"sha256": hashlib.sha256(bundle.extractfile(m).read()).hexdigest(), "bytes": m.size}
                          for m in members}
                if actual != manifest or bundle.extractfile("verification.json").read() != (directory / "verification.json").read_bytes():
                    raise ValueError("Verified public phase archive differs")
            scores[arm].update(result["scores"])
            primary_count += result["primary_decisions"]
            reversed_count += result["reversed_decisions"]
            for task, cases in suite.items():
                if task_phase(task) != phase:
                    continue
                path = directory / f"rows_{task}_primary.jsonl"
                if digest(path) != manifest[path.name]["sha256"]:
                    raise ValueError("Current public predictions differ from archived predictions")
                predictions[arm][task] = read_checked(path, cases)
            files[f"phases/{arm}/{phase}/artifact_receipt.json"] = directory / "artifact_receipt.json"
            files[f"phases/{arm}/{phase}/verification.json"] = directory / "verification.json"
        if primary_count != 6151 or reversed_count != 1100 or scores[arm].keys() != suite.keys():
            raise ValueError("Incomplete matched model coverage")
    for name, paths in spec["reference_rows"].items():
        scores[name], predictions[name] = {}, {}
        for task, relative in paths.items():
            path = ROOT / relative
            rows = read_checked(path, suite[task])
            predictions[name][task] = rows
            scores[name][task] = task_metrics(task, suite[task], rows)
            files[f"reference/{name}/{task}.jsonl"] = path
    paired = {name: paired_typed_delta(suite["typed_decisions"], predictions["selected"]["typed_decisions"], rows["typed_decisions"])
              for name, rows in predictions.items() if name != "selected"}
    result = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(), "scores": scores,
              "selected_minus_reference_typed": paired, "selected_step": bindings["selected_step"],
              "retained_original_weights": bindings["selected_step"] == 0,
              "registration_sha256": digest(REGISTRATION), "selection_sha256": bindings["selection_sha256"],
              "reference_scope": spec["reference_scope"], "scope": spec["scope"]}
    target.mkdir()
    write_json(target / "comparison.json", result)
    lines = ["# Matched Qwen continuation comparison", "", spec["scope"], "", spec["reference_scope"], "",
             f"Selected update: {bindings['selected_step']}. Selection used only the original validation set.", "",
             "| Model | Typed accuracy | Routing accuracy | MASSIVE-11 mean accuracy |",
             "|---|---:|---:|---:|"]
    for name, tasks in scores.items():
        intent = fmean(v["accuracy"] for task, v in tasks.items() if task.startswith("massive_"))
        lines.append(f"| {name} | {100*tasks['typed_decisions']['accuracy']:.2f}% | "
                     f"{100*tasks['app_model_routing_domain']['accuracy']:.2f}% | {100*intent:.2f}% |")
    if bindings["selected_step"] == 0:
        lines.extend(["", "The frozen selector retained the original weights; this continuation has no selected training gain."])
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.update({"comparison.json": target / "comparison.json", "README.md": target / "README.md"})
    files.update({"source/" + name: ROOT / name for name in spec["sources"]})
    print(json.dumps({"selected_step": result["selected_step"], "paired_typed": paired,
                      "artifact": archive_files(target, files)}, indent=2))


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                      TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare == bool(args.arm or args.phase) or (not args.compare and not (args.arm and args.phase)):
        parser.error("Use --compare or both --arm and --phase")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    compare(spec) if args.compare else report(args.arm, args.phase, spec)
