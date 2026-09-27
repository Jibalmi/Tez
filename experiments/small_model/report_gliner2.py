"""Validate complete GLiNER runs and compare their frozen predictions with Qwen and Decider."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import fmean
import tarfile

from competitor_overlap import SOURCES, read_checked
from eval_gliner2 import DIRECTORY, REGISTRATION, ROOT, raw_metrics
from gliner2_adapter import native_probabilities
from structure_probe import digest, write_json


def close_metrics(actual, expected):
    if actual.keys() != expected.keys() or any(not math.isclose(actual[key], expected[key], abs_tol=1e-12, rel_tol=1e-12)
                                               for key in actual):
        raise ValueError("Saved metric differs from the raw rows")


def main():
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve the completed comparison")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    suite_path = ROOT / spec["suite"]
    if digest(suite_path) != spec["suite_sha256"]:
        raise ValueError("Public suite changed")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    hashes, scores, rows_by_model, files = {}, {}, {}, {}
    for arm in ("english", "multilingual"):
        arm_root, arm_spec = DIRECTORY, spec
        if arm in spec.get("reuse_arms", {}):
            reused = spec["reuse_arms"][arm]
            arm_root = ROOT / reused["directory"]
            arm_registration = arm_root / arm / "registration.json"
            if digest(arm_registration) != reused["registration_sha256"]:
                raise ValueError("Reused arm registration differs")
            arm_spec = json.loads(arm_registration.read_text(encoding="utf-8"))
            if arm_spec["suite_sha256"] != spec["suite_sha256"] or arm_spec["preparation_sha256"] != spec["preparation_sha256"]:
                raise ValueError("Reused arm has different inputs or preparation")
        folder = arm_root / arm
        registered_hash = digest(folder / "registration.json")
        receipt = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        execution = json.loads((arm_root / f"execution_{arm}.json").read_text(encoding="utf-8"))
        if (receipt["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
                or receipt["registration_sha256"] != registered_hash or execution["registration_sha256"] != registered_hash
                or (arm_root == DIRECTORY and registered_hash != digest(REGISTRATION))
                or set(receipt["tasks"]) != set(suite) or receipt["primary_decisions"] != 6151
                or receipt["reversed_decisions"] != 1100 or receipt["input_audit"]["records_checked"] != 7255):
            raise ValueError("Both registered arms must complete before comparison")
        for relative, expected in arm_spec["inference_sources"].items():
            if receipt["source_files"][relative] != expected or digest(folder / "executed_source" / relative) != expected:
                raise ValueError("Executed source differs from registration")
        selected = arm_spec["models"][arm]
        if receipt["probe_sha256"] != selected["probe_sha256"] or receipt["preparation_sha256"] != spec["preparation_sha256"]:
            raise ValueError("Frozen eligibility evidence differs")
        if not receipt["batch_smoke"]["all_native_labels_match"] or receipt["batch_smoke"]["max_probability_delta"] > 1e-5:
            raise ValueError("Batched smoke did not pass")
        scores[arm], rows_by_model[arm] = {}, {}
        if arm_root != DIRECTORY:
            files.update({arm + "/" + p.relative_to(folder).as_posix(): p for p in folder.rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts})
            for name in (f"execution_{arm}.json", f"{arm}.log"):
                files[name] = arm_root / name
        for task, cases in suite.items():
            path = folder / f"rows_{task}.jsonl"
            rows = read_checked(path, cases)
            measured = raw_metrics(list(rows.values()))
            recorded = receipt["tasks"][task]
            close_metrics(measured, recorded["t1"])
            if digest(path) != recorded["row_file"]["sha256"] or len(rows) != recorded["row_file"]["rows"]:
                raise ValueError("Primary row receipt differs")
            for case in cases:
                row = rows[case["id"]]
                p = native_probabilities(row["logits"], [key for key, _ in case["options"]], row["native"])
                if max(abs(a - b) for a, b in zip(p, row["probabilities"])) > 1e-12:
                    raise ValueError("Saved logits and probabilities differ")
            if task.startswith("massive_"):
                reverse_cases = [{**case, "options": list(reversed(case["options"])),
                                  "gold": len(case["options"]) - 1 - case["gold"]} for case in cases]
                reverse_path = folder / f"rows_{task}_reversed.jsonl"
                reverse = read_checked(reverse_path, reverse_cases)
                for case in reverse_cases:
                    row = reverse[case["id"]]
                    p = native_probabilities(row["logits"], [key for key, _ in case["options"]], row["native"])
                    if max(abs(a - b) for a, b in zip(p, row["probabilities"])) > 1e-12:
                        raise ValueError("Reversed logits and probabilities differ")
                flip = fmean(rows[case["id"]]["pred"] != len(case["options"]) - 1 - reverse[case["id"]]["pred"] for case in cases)
                if (digest(reverse_path) != recorded["reversed_row_file"]["sha256"]
                        or len(reverse) != recorded["reversed_row_file"]["rows"] or flip != recorded["order_flip"]):
                    raise ValueError("Reversed-option receipt differs")
                measured["order_flip"] = flip
                hashes[reverse_path.relative_to(ROOT).as_posix()] = digest(reverse_path)
            if task.startswith("semif_"):
                source = folder / "executed_source/data/semif/evaluate.py"
                module_spec = importlib.util.spec_from_file_location("semif_recheck", source)
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                gold = module.read_jsonl(source.parent / (task.split("_", 1)[1] + ".jsonl"))
                predictions = [{"id": row["id"], "probabilities": row["probabilities"]} for row in rows.values()]
                if module.evaluate(gold, predictions) != recorded["semif"]:
                    raise ValueError("Official SemIf recheck differs")
                measured["semif_mean_family_balanced_accuracy"] = recorded["semif"]["mean_family_balanced_accuracy"]
            scores[arm][task], rows_by_model[arm][task] = measured, rows
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    for name, (directory, candidate) in SOURCES.items():
        scores[name], rows_by_model[name] = {}, {}
        for task, cases in suite.items():
            path = ROOT / directory / f"rows_{task}_tez-s-{candidate}.jsonl"
            rows = read_checked(path, cases)
            scores[name][task], rows_by_model[name][task] = raw_metrics(list(rows.values())), rows
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
            files[f"reference/{name}/{path.name}"] = path
    summary = {"status": "complete", "analyzed_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "suite_sha256": digest(suite_path),
               "analysis_source_sha256": digest(__file__), "prediction_hashes": hashes, "scores": scores,
               "scope": "Exploratory comparison of native frozen checkpoints on already-observed public cases; training exposure differs. No fitting, routing selection, latency claim or runtime promotion."}
    for name, tasks in scores.items():
        languages = [values for task, values in tasks.items() if task.startswith("massive_")]
        summary.setdefault("massive_11", {})[name] = {"accuracy": fmean(values["accuracy"] for values in languages)}
        if name in ("english", "multilingual"):
            summary["massive_11"][name]["order_flip"] = fmean(values["order_flip"] for values in languages)
        typed = rows_by_model[name]["typed_decisions"]
        summary.setdefault("typed_by_question_type", {})[name] = {
            kind: raw_metrics([typed[c["id"]] for c in suite["typed_decisions"] if c["qtype"] == kind])
            for kind in sorted({c["qtype"] for c in suite["typed_decisions"]})}
    write_json(DIRECTORY / "comparison.json", summary)
    names = ("english", "multilingual", "qwen", "decider")
    lines = ["# Native GLiNER comparison", "", "Complete, matched inputs: 6,151 original public decisions per model.",
             "GLiNER additionally has 1,100 reversed-option intent decisions per model. Raw T=1 probabilities; no calibration fitting.",
             "The GLiNER arms use original FP32 native eager inference; Qwen and Decider use their previously frozen implementations.",
             ("English uses batch size one after a preserved batch-four resource cancellation; multilingual retains its registered batch four."
              if spec.get("reuse_arms") else "Both GLiNER arms use the registered batch size four."),
             "These are utility comparisons across different training exposures, not controlled architecture or new held-out confirmation tests.", "",
             "| Accuracy, % | GLiNER English | GLiNER multilingual | Qwen student | Decider 0.8B |",
             "|---|---:|---:|---:|---:|"]
    for task in suite:
        lines.append("| " + task + " | " + " | ".join(f"{100 * scores[name][task]['accuracy']:.2f}" for name in names) + " |")
    lines += ["| Intent, mean of 11 languages | " + " | ".join(f"{100 * summary['massive_11'][name]['accuracy']:.2f}" for name in names) + " |",
              "", "SemIf rows above show ordinary accuracy; the official mean family balanced metric is separate below.", "",
              "| Raw typed probability metric | GLiNER English | GLiNER multilingual | Qwen student | Decider 0.8B |",
              "|---|---:|---:|---:|---:|"]
    for metric in ("ece", "nll", "brier", "soft_accuracy"):
        lines.append("| " + metric + " | " + " | ".join(f"{scores[name]['typed_decisions'][metric]:.4f}" for name in names) + " |")
    lines += ["", "| GLiNER arm | Intent option-order flip | SemIf authored balanced | SemIf perturbations balanced |",
              "|---|---:|---:|---:|"]
    for name in names[:2]:
        lines.append(f"| {name} | {summary['massive_11'][name]['order_flip']:.4f} | "
                     f"{scores[name]['semif_authored144']['semif_mean_family_balanced_accuracy']:.4f} | "
                     f"{scores[name]['semif_perturbations108']['semif_mean_family_balanced_accuracy']:.4f} |")
    lines += ["", "Job elapsed time includes loading, input audits and thermal pauses; it is not a latency result.",
              "No gates were relaxed and no runtime was changed. Competitor training membership is unknown.",
              f"[Full measurements and hashes](comparison.json), [frozen protocol](../../experiments/small_model/{REGISTRATION.name}),",
              "[broader comparison and release requirements](../pilot-improvement-20260927/COMPETITIVENESS.md).", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    excluded = {archive.name, "artifact_manifest.json", "artifact_receipt.json"}
    files.update({p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*")
                  if p.is_file() and p.name not in excluded and "__pycache__" not in p.parts})
    files["inputs/eval-suites.json"] = suite_path
    files["inputs/preparation.json"] = ROOT / spec["preparation"]
    for name, selected in spec["models"].items():
        files[f"inputs/probe_{name}.json"] = ROOT / selected["probe"]
    for name in ("report_gliner2.py", "competitor_overlap.py", "test_gliner2_metrics.py"):
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
        raise ValueError("Archive membership or content changed")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
                "primary_rows_validated": 4 * 6151, "reversed_rows_validated": 2 * 1100}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"artifact": artifact, "massive_11": summary["massive_11"],
                      "typed": {name: scores[name]["typed_decisions"] for name in names}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if args.attempt == 2:
        DIRECTORY = ROOT / "results/pilot-gliner2-eval-attempt2-20260927"
        REGISTRATION = Path(__file__).with_name("gliner2_registration_attempt2.json")
    elif args.attempt == 3:
        DIRECTORY = ROOT / "results/pilot-gliner2-eval-serial-20260927"
        REGISTRATION = Path(__file__).with_name("gliner2_registration_serial.json")
    main()
