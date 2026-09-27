"""Select from four verified validations and export only the chosen private LoRA adapter."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import fmean
import tarfile

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", CUDA_VISIBLE_DEVICES="")

from qwen_continuation_selection import choose_checkpoint  # noqa: E402
from structure_probe import digest, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "results/pilot-qwen-continuation-20260927/training"
CHECKPOINTS = ROOT / "runs/pilot-qwen-continuation-20260927/training"
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/selection"
MODEL_DIRECTORY = ROOT / "runs/pilot-qwen-continuation-20260927/export/models/c-qwen08-continuation"
REGISTRATION = Path(__file__).with_name("qwen_continuation_export_registration.json")
CONTINUATION = Path(__file__).with_name("qwen_continuation_registration.json")
ANALYSIS = Path(__file__).with_name("qwen_continuation_analysis_registration.json")


def canonical_adapter(tensors, original):
    """Remove precisely the LoRA adapter-name segment and retain the original tensor layout."""
    import torch
    converted = {}
    for name, tensor in tensors.items():
        suffix = next((s for s in (".lora_A.default.weight", ".lora_B.default.weight") if name.endswith(s)), None)
        if suffix is None:
            raise ValueError("Only this continuation's default LoRA A/B matrices may be exported")
        key = name[:-len(suffix)] + suffix.replace(".default.", ".")
        if key in converted or key not in original:
            raise ValueError("Duplicate or unexpected exported tensor")
        if (tensor.shape != original[key].shape or tensor.dtype != original[key].dtype
                or not torch.isfinite(tensor).all().item()):
            raise ValueError("Export would change tensor layout or contain non-finite values")
        converted[key] = tensor.detach().cpu().contiguous().clone()
    if not converted or set(converted) != set(original):
        raise ValueError("The complete original adapter layout is required")
    return converted


def save_adapter(tensors, original, config_path, destination):
    """Write immutable Safetensors and unchanged config, then verify every reloaded tensor."""
    import torch
    from safetensors.torch import load_file, save_file
    converted = canonical_adapter(tensors, original)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "adapter_model.safetensors"
    save_file(converted, path, metadata={"format": "pt"})
    shutil.copy2(config_path, destination / "adapter_config.json")
    reloaded = load_file(path, device="cpu")
    if (reloaded.keys() != converted.keys()
            or any(reloaded[key].dtype != value.dtype or not torch.equal(reloaded[key], value) for key, value in converted.items())
            or digest(config_path) != digest(destination / "adapter_config.json")):
        raise ValueError("Adapter serialization did not preserve every tensor and the original config")
    return {"adapter_sha256": digest(path), "adapter_bytes": path.stat().st_size,
            "config_sha256": digest(destination / "adapter_config.json"), "tensors_verified": len(converted),
            "parameters_verified": sum(value.numel() for value in converted.values()),
            "scope": "Exact CPU tensor and config serialization check; no Qwen inference or GPU parity claim."}


def archive_records(directory, continuation_sha, analysis_sha):
    """Read the immutable verification archive and reject any changed local evidence."""
    directory = directory.resolve()
    if not directory.is_relative_to(TRAINING.resolve()):
        raise ValueError("Validation must belong to this continuation")
    artifact = json.loads((directory / "artifact_receipt.json").read_text(encoding="utf-8"))
    archive = directory / "evidence.tar.gz"
    if digest(archive) != artifact["sha256"] or archive.stat().st_size != artifact["bytes"]:
        raise ValueError("Validation evidence archive changed")
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    actual, content = {}, {}
    required = {"registration.json", "work.json", "verification.json", "execution.json", "validation_rows.jsonl"}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected validation archive member")
            data = bundle.extractfile(member).read()
            actual[member.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": member.size}
            if member.name in required:
                if digest(directory / member.name) != actual[member.name]["sha256"]:
                    raise ValueError("Local validation evidence differs from its archive")
                content[member.name] = data
    if actual != manifest or len(actual) != artifact["files_verified"] or set(content) != required:
        raise ValueError("Validation archive is incomplete or differs from its manifest")
    work, verified, execution = [json.loads(content[name]) for name in ("work.json", "verification.json", "execution.json")]
    if (actual["registration.json"]["sha256"] != continuation_sha
            or any(value["registration_sha256"] != continuation_sha for value in (work, verified, execution))
            or verified["analysis_registration_sha256"] != analysis_sha or verified["status"] != "complete"
            or actual["analysis_registration.json"]["sha256"] != analysis_sha
            or verified["verifier_sha256"] != digest(Path(__file__).with_name("verify_qwen_continuation.py"))
            or work["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or work["operation"] not in ("baseline", "validate") or work["operation"] != verified["operation"]
            or work["completed_step"] != verified["completed_step"] or work["target_step"] != work["completed_step"]):
        raise ValueError("Require a complete independently verified validation operation")
    if work["operation"] == "baseline" and not verified["baseline_parity"]["passed"]:
        raise ValueError("The unchanged baseline must pass its original probability check")
    for name, expected in json.loads(content["registration.json"])["sources"].items():
        if (work["source_files"].get(name) != expected
                or actual["executed_source/" + name]["sha256"] != expected):
            raise ValueError("Executed validation source does not match its registration")
    rows = [json.loads(line) for line in content["validation_rows.jsonl"].decode("utf-8").splitlines()]
    if actual["validation_rows.jsonl"]["sha256"] != work["validation"]["rows_sha256"]:
        raise ValueError("Validation prediction hash differs")
    return work, verified, rows, artifact


def validation_metrics(rows, baseline_rows=None):
    """Recompute teacher KL/agreement, requiring identical cases and targets at every step."""
    if len(rows) != 1390:
        raise ValueError("Require all original held-out decisions")
    indexed = {(r["state_id"], r["qid"], r["k"]): r for r in rows}
    if len(indexed) != 1390 or len({r["state_id"] for r in rows}) != 278:
        raise ValueError("Duplicated or incomplete held-out membership")
    if baseline_rows is not None and indexed.keys() != baseline_rows.keys():
        raise ValueError("Checkpoint validation membership changed")
    losses, agreement = [], []
    for key, row in indexed.items():
        p, target, logits = row["probabilities"], row["target"], row["log_probabilities"]
        if (not 2 <= row["k"] <= 26 or len(p) != row["k"] or len(target) != len(p) or len(logits) != len(p)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p + target)
                or abs(sum(p) - 1) > 1e-5 or abs(sum(target) - 1) > 1e-5 or not all(map(math.isfinite, logits))):
            raise ValueError("Invalid held-out probability vector")
        if baseline_rows is not None and any(row[name] != baseline_rows[key][name]
                                            for name in ("target", "gold", "qtype", "encoded_tokens", "prompt_sha256")):
            raise ValueError("Checkpoint validation inputs or targets changed")
        exponentials = [math.exp(value - max(logits)) for value in logits]
        if any(abs(value / sum(exponentials) - probability) > 1e-12
               for value, probability in zip(exponentials, p, strict=True)):
            raise ValueError("Saved readout does not reproduce probabilities")
        loss = sum(t * (math.log(max(t, 1e-12)) - math.log(max(q, 1e-12))) for t, q in zip(target, p, strict=True))
        teacher, student = max(range(len(p)), key=target.__getitem__), max(range(len(p)), key=p.__getitem__)
        if (not math.isclose(row["kl"], loss, rel_tol=0, abs_tol=1e-12)
                or row["teacher_argmax"] != teacher or row["student_argmax"] != student):
            raise ValueError("Per-decision metric differs from saved probabilities")
        losses.append(loss)
        agreement.append(teacher == student)
    return {"n": len(rows), "mean_kl": fmean(losses), "teacher_agreement": fmean(agreement)}, indexed


def register():
    preflight_path = TRAINING.parent / "export_preflight.json"
    if REGISTRATION.exists() or preflight_path.exists():
        raise ValueError("Preserve the export registration")
    if any(json.loads(path.read_text(encoding="utf-8")).get("operation") == "validate"
           for path in TRAINING.glob("attempt-*/work.json")):
        raise ValueError("This selection/export registration must precede trained checkpoint validation")
    baseline_directory = TRAINING / "attempt-02-baseline-000"
    work, verified, rows, artifact = archive_records(baseline_directory, digest(CONTINUATION), digest(ANALYSIS))
    metrics, _ = validation_metrics(rows)
    if work["completed_step"] != 0 or any(not math.isclose(value, verified["validation"][key], rel_tol=0, abs_tol=1e-12)
                                          for key, value in metrics.items()):
        raise ValueError("Preflight did not reproduce the verified unchanged baseline")
    write_json(preflight_path, {"status": "complete", "cpu_threads": 1,
                               "baseline_directory": baseline_directory.relative_to(ROOT).as_posix(),
                               "baseline_artifact": artifact, "validation_metrics": metrics,
                               "selection_performed": False, "model_exported": False,
                               "scope": "Existing baseline archive and metric-reader integration only; no trained validation or selection."})
    sources = [Path(__file__).with_name(name) for name in (
        "export_qwen_continuation.py", "test_export_qwen_continuation.py", "qwen_continuation_selection.py",
        "test_qwen_continuation_selection.py", "verify_qwen_continuation.py", "structure_probe.py")]
    sources.append(ROOT / ".venv-pilot/Lib/site-packages/peft/utils/save_and_load.py")
    spec = {"status": "registered_before_any_trained_checkpoint_validation",
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "continuation_registration_sha256": digest(CONTINUATION), "analysis_registration_sha256": digest(ANALYSIS),
            "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
            "prerequisites": {preflight_path.relative_to(ROOT).as_posix(): digest(preflight_path)},
            "packages": {name: version(name) for name in ("torch", "peft", "safetensors")},
            "selection": "Use the already frozen choose_checkpoint function on all four complete verified original-validation results. Step zero stays eligible. No public or authored test scores enter the selector.",
            "export": "Verify all four evidence archives, recompute validation metrics, and require a single complete checkpoint validation history. Export only the selected checkpoint's complete unmerged float32 LoRA layout, dropping the default adapter-name segment. Preserve original adapter config byte-for-byte. Safetensors reload must preserve each tensor exactly. New student metadata declares T1 and provenance; it does not inherit historical performance or fitted temperatures.",
            "compute": "One CPU thread, no GPU model, no paid resource. Private adapter export only; no weights or training rows in public evidence.",
            "scope": "Serialization is not Qwen inference parity, public accuracy or release readiness. Reloaded Qwen must match the selected validation predictions before public evaluation. If step zero wins, report no continuation gain; do not choose another checkpoint using public results.",
            "reference": "https://huggingface.co/docs/peft/developer_guides/checkpoint"}
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "status": spec["status"]}))


def main(directories):
    if len(directories) != 4 or len({path.resolve() for path in directories}) != 4:
        raise ValueError("Provide exactly four distinct verified validation directories")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    if (digest(CONTINUATION) != spec["continuation_registration_sha256"]
            or digest(ANALYSIS) != spec["analysis_registration_sha256"]):
        raise ValueError("Continuation or independent verification registration changed")
    for name, expected in {**continuation["sources"], **spec["sources"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered source changed")
    for name, expected in {**continuation["prerequisites"], **spec["prerequisites"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite changed")
    if any(version(name) != expected for name, expected in spec["packages"].items()):
        raise ValueError("Export package versions changed")
    entries = {}
    for directory in directories:
        work, verified, rows, artifact = archive_records(directory, digest(CONTINUATION), digest(ANALYSIS))
        step = work["completed_step"]
        if step in entries or step not in (0, 50, 100, 150):
            raise ValueError("Require one validation at every planned step")
        entries[step] = {"work": work, "verified": verified, "rows": rows, "artifact": artifact,
                         "directory": directory.resolve().relative_to(ROOT).as_posix()}
    if set(entries) != {0, 50, 100, 150}:
        raise ValueError("Missing planned validation")
    baseline, metrics = None, {}
    for step, entry in sorted(entries.items()):
        metrics[step], indexed = validation_metrics(entry["rows"], baseline)
        baseline = indexed if step == 0 else baseline
        for key, value in metrics[step].items():
            if any(not math.isclose(value, recorded[key], rel_tol=0, abs_tol=1e-12)
                   for recorded in (entry["work"]["validation"], entry["verified"]["validation"])):
                raise ValueError("Independently recomputed checkpoint metrics differ")
        if entry["work"]["validation"]["directory"] != entry["directory"]:
            raise ValueError("Validation directory provenance differs")
    choice = choose_checkpoint(metrics)
    import torch
    from safetensors.torch import load_file
    from verify_qwen_continuation import checkpoint_payload
    torch.set_num_threads(1)
    original_directory = ROOT / continuation["adapter_directory"]
    original = load_file(original_directory / "adapter_model.safetensors", device="cpu")
    schedule = json.loads((ROOT / continuation["schedule"]).read_text(encoding="utf-8"))
    cumulative = [0]
    for batch in schedule["batches"]:
        cumulative.append(cumulative[-1] + sum(r["encoded_tokens"] for r in batch))
    selected, selected_checkpoint = None, None
    for step, entry in sorted(entries.items()):
        checkpoint = entry["work"]["checkpoints"][-1]
        if entry["verified"]["checkpoints"][-1]["sha256"] != checkpoint["sha256"]:
            raise ValueError("Final checkpoint differs from the independent verification")
        path = (ROOT / checkpoint["path"]).resolve()
        if not path.is_relative_to(CHECKPOINTS.resolve()) or digest(path) != checkpoint["sha256"]:
            raise ValueError("Validation checkpoint path or content changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        check = checkpoint_payload(saved, original, digest(CONTINUATION), cumulative)
        expected_history = {str(s): entries[s]["work"]["validation"] for s in entries if s <= step}
        if check["step"] != step or saved["state"]["validations"] != expected_history:
            raise ValueError("Checkpoint validation history does not form one complete registered sequence")
        if step == choice["selected_step"]:
            selected = {key: tensor.clone() for key, tensor in saved["adapter"].items()}
            selected_checkpoint = checkpoint
        del saved
    if DIRECTORY.exists() or MODEL_DIRECTORY.exists():
        raise ValueError("Preserve any previous selection or private export")
    DIRECTORY.mkdir(parents=True)
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    for name in spec["sources"]:
        target = DIRECTORY / "executed_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    exported = save_adapter(selected, original, original_directory / "adapter_config.json", MODEL_DIRECTORY / "adapter")
    info = {"arm": "c-qwen08-continuation", "backbone": "Qwen/Qwen3.5-0.8B",
            "base_revision": continuation["base_revision"], "base_sha256": continuation["base_sha256"],
            "lora": {"r": 16, "alpha": 32, "targets": "all-linear except lm_head"},
            "temperatures": {key: 1.0 for key in ("score", "noul", "choice", "choice_many")},
            "selected_continuation_step": choice["selected_step"], "parent_adapter_sha256": digest(original_directory / "adapter_model.safetensors"),
            "continuation_registration_sha256": digest(CONTINUATION), "selected_checkpoint": selected_checkpoint,
            "adapter_sha256": exported["adapter_sha256"], "inference_parity": "Pending reload verification"}
    write_json(MODEL_DIRECTORY / "student.json", info)
    result = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "export_registration_sha256": digest(REGISTRATION), "continuation_registration_sha256": digest(CONTINUATION),
              **choice, "validation_metrics": metrics, "retained_original_weights": choice["selected_step"] == 0,
              "selected_checkpoint": selected_checkpoint, "model_directory": MODEL_DIRECTORY.relative_to(ROOT).as_posix(),
              "student_sha256": digest(MODEL_DIRECTORY / "student.json"), "export": exported,
              "validation_evidence": {step: {"directory": entry["directory"], "artifact": entry["artifact"]}
                                      for step, entry in sorted(entries.items())},
              "scope": spec["scope"]}
    write_json(DIRECTORY / "selection.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--validation-directories", type=Path, nargs="*")
    args = parser.parse_args()
    if args.register:
        if args.validation_directories:
            raise ValueError("Registration does not select or export a checkpoint")
        register()
    else:
        main(args.validation_directories or [])
