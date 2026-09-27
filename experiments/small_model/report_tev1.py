"""Validate complete native Tev1 predictions and archive a matched Qwen/Decider comparison."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tarfile

from competitor_overlap import SOURCES, read_checked
from report_gliner2 import close_metrics, scoped_metrics
from structure_probe import digest, write_json
from tev1_adapter import load_native, render_case

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-tev1-eval-20260927"


def main():
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists() or (DIRECTORY / "comparison.json").exists():
        raise ValueError("Preserve completed analysis")
    registration = DIRECTORY / "registration.json"
    analysis_path = Path(__file__).with_name("tev1_analysis_registration.json")
    analysis_spec = json.loads(analysis_path.read_text(encoding="utf-8"))
    if digest(registration) != analysis_spec["inference_registration_sha256"]:
        raise ValueError("Analysis references a different inference protocol")
    for name, expected in {**analysis_spec["sources"], **analysis_spec["reference_rows"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen analysis source or comparison rows changed")
    spec = json.loads(registration.read_text(encoding="utf-8"))
    receipt = json.loads((DIRECTORY / "evaluation.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    if (receipt["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or receipt["registration_sha256"] != digest(registration) or execution["registration_sha256"] != digest(registration)
            or receipt["primary_decisions"] != 6151 or receipt["reversed_decisions"] != 1100
            or receipt["input_audit"]["records_checked"] != 7255 or not receipt["smoke"]["all_native_labels_match"]
            or receipt["smoke"]["max_probability_delta"] > 1e-5):
        raise ValueError("Complete the registered native evaluation first")
    for name, expected in spec["inference_sources"].items():
        if receipt["source_files"][name] != expected or digest(DIRECTORY / "executed_source" / name) != expected:
            raise ValueError("Executed inference source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite changed")
    prep = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
    for name, entry in prep["files"].items():
        if not name.endswith(".safetensors") and digest(ROOT / prep["directory"] / name) != entry["sha256"]:
            raise ValueError("Frozen native tokenizer or template changed")
    if (receipt["revision"] != prep["revision"] or receipt["model"] != prep["repository"]
            or receipt["weights"] != {name: entry["sha256"] for name, entry in prep["files"].items() if name.endswith(".safetensors")}):
        raise ValueError("Checkpoint evidence differs")
    suite = json.loads((ROOT / spec["suite"]).read_text(encoding="utf-8"))
    if set(receipt["tasks"]) != set(suite):
        raise ValueError("Incomplete task membership")
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / prep["directory"], local_files_only=True)
    native = load_native(DIRECTORY / "executed_source" / spec["native_helper"])
    files = {"analysis_registration.json": analysis_path}
    files.update({"analysis_source/" + name: ROOT / name for name in analysis_spec["sources"]})
    hashes, scores = {}, {name: {} for name in ("tev1", "qwen", "decider")}

    def verify_native(rows, cases):
        for case in cases:
            row = rows[case["id"]]
            record, messages = render_case(native, case, prep["repository"])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            payload_sha = hashlib.sha256(json.dumps(record, ensure_ascii=False).encode("utf-8")).hexdigest()
            tokens = [tokenizer.encode(o["label"], add_special_tokens=False) for o in record["options"]]
            if (any(len(t) != 1 for t in tokens) or row["letter_token_ids"] != [t[0] for t in tokens]
                    or row["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    or row["payload_sha256"] != payload_sha
                    or row["encoded_tokens"] != len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
                    or row["encoded_tokens"] > spec["context_bound"]):
                raise ValueError("Native input audit does not reproduce")
            logits = row["logits"]
            if len(logits) != len(row["probabilities"]) or not all(map(math.isfinite, logits)):
                raise ValueError("Invalid option logits")
            p = [math.exp(v - max(logits)) for v in logits]
            total = sum(p)
            if max(abs(v / total - saved) for v, saved in zip(p, row["probabilities"], strict=True)) > 1e-12:
                raise ValueError("Option logits do not reproduce probabilities")
            mass = row["unrestricted_option_mass"]
            if not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-12:
                raise ValueError("Invalid vocabulary option mass")

    for task, cases in suite.items():
        path = DIRECTORY / f"rows_{task}.jsonl"
        recorded = receipt["tasks"][task]
        rows = read_checked(path, cases)
        verify_native(rows, cases)
        measured = scoped_metrics(task, cases, rows)
        close_metrics({k: v for k, v in measured.items() if k != "semantic_macro_f1"}, recorded["t1"])
        if recorded["row_file"] != {"sha256": digest(path), "rows": len(rows)}:
            raise ValueError("Raw prediction receipt changed")
        hashes[path.name] = digest(path)
        measured["mean_unrestricted_option_mass"] = fmean(r["unrestricted_option_mass"] for r in rows.values())
        if task.startswith("massive_"):
            reverse_cases = [{**c, "options": list(reversed(c["options"])), "gold": len(c["options"]) - 1 - c["gold"]} for c in cases]
            reversed_path = DIRECTORY / f"rows_{task}_reversed.jsonl"
            reversed_rows = read_checked(reversed_path, reverse_cases)
            verify_native(reversed_rows, reverse_cases)
            if recorded["reversed_row_file"] != {"sha256": digest(reversed_path), "rows": len(reversed_rows)}:
                raise ValueError("Reversed row receipt changed")
            hashes[reversed_path.name] = digest(reversed_path)
            measured["order_flip"] = fmean(rows[c["id"]]["pred"] != len(c["options"]) - 1 - reversed_rows[c["id"]]["pred"] for c in cases)
            if not math.isclose(measured["order_flip"], recorded["order_flip"], abs_tol=1e-12):
                raise ValueError("Reversed-order metric differs")
        if task.startswith("semif_"):
            module_spec = importlib.util.spec_from_file_location("semif_frozen_evaluator", DIRECTORY / "executed_source/data/semif/evaluate.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            data = module.read_jsonl(DIRECTORY / "executed_source/data/semif" / (task.split("_", 1)[1] + ".jsonl"))
            official = module.evaluate(data, [{"id": r["id"], "probabilities": r["probabilities"]} for r in rows.values()])
            if official != recorded["semif"]:
                raise ValueError("Official SemIf score differs")
            measured["semif"] = official
        scores["tev1"][task] = measured
        for name, (folder, arm) in SOURCES.items():
            baseline = ROOT / folder / f"rows_{task}_tez-s-{arm}.jsonl"
            reference = read_checked(baseline, cases)
            scores[name][task] = scoped_metrics(task, cases, reference)
            target = f"reference/{name}/{baseline.name}"
            files[target] = baseline
            hashes[target] = digest(baseline)
    summary = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(registration), "analysis_source_sha256": digest(__file__),
               "analysis_registration_sha256": digest(analysis_path),
               "scores": scores, "prediction_hashes": hashes,
               "massive_11": {name: {"accuracy": fmean(v["accuracy"] for task, v in tasks.items() if task.startswith("massive_"))}
                              for name, tasks in scores.items()},
               "scope": "Matched public utility comparison across different training exposures. No fitting, runtime promotion, hosted-endpoint equivalence or latency claim. Full vocabulary logits were not retained; unrestricted option mass is an inference diagnostic, while option-softmax vectors are independently reproducible."}
    write_json(DIRECTORY / "comparison.json", summary)
    names = ("tev1", "qwen", "decider")
    lines = ["# Native Tev1 comparison", "", "All 6,151 original public decisions verified per model; Tev1 also has 1,100 reversed-option intent cases.",
             "Raw temperature-one probabilities. Tev1 uses the pinned full checkpoint in BF16, eager attention, native JSON/chat format and batch size one.",
             "These are already-observed public tasks with differing and incompletely reconstructed training exposures.", "",
             "| Accuracy, % | Tev1 0.8B | Qwen student | Decider 0.8B |", "|---|---:|---:|---:|"]
    for task in suite:
        lines.append("| " + task + " | " + " | ".join(f"{100 * scores[name][task]['accuracy']:.2f}" for name in names) + " |")
    lines += ["| Intent, 11-language mean | " + " | ".join(f"{100 * summary['massive_11'][name]['accuracy']:.2f}" for name in names) + " |",
              "", "| Raw typed probability metric | Tev1 | Qwen | Decider |", "|---|---:|---:|---:|"]
    for metric in ("ece", "nll", "brier", "soft_accuracy"):
        lines.append("| " + metric + " | " + " | ".join(f"{scores[name]['typed_decisions'][metric]:.4f}" for name in names) + " |")
    lines += ["", "Semantic app/intent macro-F1 and the official SemIf family metrics are in the full analysis.",
              "Typed rows combine different label meanings; option-index F1 remains diagnostic only.",
              "Execution time includes loading and thermal pauses and is not a latency result. No release gates changed.",
              "[Full analysis](comparison.json), [registered protocol](registration.json), [broader comparison](../pilot-improvement-20260927/COMPETITIVENESS.md).", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    files.update({p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
                  if p.is_file() and p.name not in {archive.name, "artifact_manifest.json", "artifact_receipt.json", ".gitignore"}
                  and "__pycache__" not in p.parts})
    for name in spec["prerequisites"]:
        files["inputs/" + name] = ROOT / name
    for name in ("report_tev1.py", "report_gliner2.py", "competitor_overlap.py", "test_tev1_adapter.py", "test_gliner2_report.py"):
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
        raise ValueError("Archive verification failed")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
                "primary_rows_validated": 3 * 6151, "reversed_rows_validated": 1100}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"artifact": artifact, "massive_11": summary["massive_11"]}, indent=2))


if __name__ == "__main__":
    main()
