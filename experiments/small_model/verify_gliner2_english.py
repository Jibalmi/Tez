"""Preserve and independently verify the completed English arm while multilingual is pending."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import fmean
import tarfile

from competitor_overlap import read_checked
from eval_gliner2 import raw_metrics
from gliner2_adapter import native_probabilities
from report_gliner2 import close_metrics, scoped_metrics
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-gliner2-eval-serial-20260927"


def main():
    folder = DIRECTORY / "english"
    archive = DIRECTORY / "english-artifacts.tar.gz"
    if archive.exists() or (DIRECTORY / "validation_english.json").exists():
        raise ValueError("Preserve existing completed-arm verification")
    registration = folder / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    if digest(registration) != "fb3234d8e3fac9e08da7a70a15eef992bb824730d0c9202c9efd23c37d1bb141":
        raise ValueError("Serial registration changed")
    receipt = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution_english.json").read_text(encoding="utf-8"))
    if (receipt["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or receipt["registration_sha256"] != digest(registration) or execution["registration_sha256"] != digest(registration)
            or receipt["primary_decisions"] != 6151 or receipt["reversed_decisions"] != 1100
            or receipt["input_audit"]["records_checked"] != 7255 or not receipt["batch_smoke"]["all_native_labels_match"]
            or receipt["batch_smoke"]["max_probability_delta"] > 1e-5):
        raise ValueError("English arm has not completed its exact protocol")
    for name, expected in spec["inference_sources"].items():
        if receipt["source_files"][name] != expected or digest(folder / "executed_source" / name) != expected:
            raise ValueError("Executed inference source changed")
    selected = spec["models"]["english"]
    for name, expected in ((spec["suite"], spec["suite_sha256"]), (spec["preparation"], spec["preparation_sha256"]),
                           (selected["probe"], selected["probe_sha256"])):
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input or probe changed")
    prep = json.loads((ROOT / spec["preparation"]).read_text(encoding="utf-8"))
    if (receipt["weights"] != prep["models"][selected["repository"]]["files"]["model.safetensors"]["sha256"]
            or receipt["revision"] != selected["revision"] or receipt["probe_sha256"] != selected["probe_sha256"]
            or receipt["preparation_sha256"] != spec["preparation_sha256"]):
        raise ValueError("Model or prerequisite receipt changed")
    suite = json.loads((ROOT / spec["suite"]).read_text(encoding="utf-8"))
    scores, counts = {}, {"primary": 0, "reversed": 0}
    for task, cases in suite.items():
        original = None
        for suffix in (("", "_reversed") if task.startswith("massive_") else ("",)):
            aligned = [{**c, "options": list(reversed(c["options"])), "gold": len(c["options"]) - 1 - c["gold"]}
                       for c in cases] if suffix else cases
            path = folder / f"rows_{task}{suffix}.jsonl"
            rows = read_checked(path, aligned)
            expected_file = receipt["tasks"][task]["reversed_row_file" if suffix else "row_file"]
            if expected_file != {"sha256": digest(path), "rows": len(rows)}:
                raise ValueError("Prediction file differs from its receipt")
            for case in aligned:
                row = rows[case["id"]]
                p = native_probabilities(row["logits"], [key for key, _ in case["options"]], row["native"])
                if max(abs(a - b) for a, b in zip(p, row["probabilities"], strict=True)) > 1e-12:
                    raise ValueError("Native logits, winner and probabilities differ")
            counts["reversed" if suffix else "primary"] += len(rows)
            if not suffix:
                close_metrics(raw_metrics(list(rows.values())), receipt["tasks"][task]["t1"])
                scores[task] = scoped_metrics(task, cases, rows)
                original = rows
            else:
                flip = fmean(original[c["id"]]["pred"] != len(c["options"]) - 1 - rows[c["id"]]["pred"] for c in cases)
                if flip != receipt["tasks"][task]["order_flip"]:
                    raise ValueError("Option-order metric differs")
                scores[task]["order_flip"] = flip
        if task.startswith("semif_"):
            source = folder / "executed_source/data/semif/evaluate.py"
            module_spec = importlib.util.spec_from_file_location("semif_recheck", source)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            gold = module.read_jsonl(source.parent / (task.split("_", 1)[1] + ".jsonl"))
            predictions = [{"id": r["id"], "probabilities": r["probabilities"]} for r in original.values()]
            official = module.evaluate(gold, predictions)
            if official != receipt["tasks"][task]["semif"]:
                raise ValueError("Official semantic metric differs")
            scores[task]["semif_mean_family_balanced_accuracy"] = official["mean_family_balanced_accuracy"]
    if counts != {"primary": 6151, "reversed": 1100} or set(receipt["tasks"]) != set(suite):
        raise ValueError("Incomplete suite")
    summary = {"status": "complete", "verified_utc": datetime.now(timezone.utc).isoformat(), "rows_verified": counts,
               "scores": scores, "massive_11_accuracy": fmean(v["accuracy"] for k, v in scores.items() if k.startswith("massive_")),
               "source_sha256": digest(__file__), "registration_sha256": digest(registration),
               "scope": "Complete English serial arm only; multilingual and combined report remain separate. Already-observed public comparison, no fitting or latency claim."}
    write_json(DIRECTORY / "validation_english.json", summary)
    files = {"english/" + p.relative_to(folder).as_posix(): p for p in folder.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    for name in ("execution_english.json", "english.log", "validation_english.json"):
        files[name] = DIRECTORY / name
    for name in ("verify_gliner2_english.py", "report_gliner2.py", "competitor_overlap.py"):
        files["analysis_source/" + name] = Path(__file__).with_name(name)
    files["inputs/eval-suites.json"] = ROOT / spec["suite"]
    files["inputs/probe.json"] = ROOT / selected["probe"]
    files["inputs/preparation.json"] = ROOT / spec["preparation"]
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
    write_json(DIRECTORY / "artifact_manifest_english.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual), "rows_verified": counts}
    write_json(DIRECTORY / "artifact_receipt_english.json", artifact)
    print(json.dumps({"artifact": artifact, "typed": scores["typed_decisions"], "massive_11_accuracy": summary["massive_11_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
