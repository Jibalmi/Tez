"""Verify completed continuation attempts without loading a model or using the GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import tarfile

from parity_evidence import compare_rows, read_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/pilot-qwen-continuation-20260927/training"
RUNS = ROOT / "runs/pilot-qwen-continuation-20260927/training"
ANALYSIS = Path(__file__).with_name("qwen_continuation_analysis_registration.json")


def checkpoint_payload(saved, original, registration_sha256, cumulative_tokens):
    import torch
    if saved["format"] != 1 or saved["registration_sha256"] != registration_sha256:
        raise ValueError("Checkpoint belongs to another experiment")
    names = saved["parameter_names"]
    if len(names) != len(set(names)) or set(names) != set(saved["adapter"]):
        raise ValueError("Incomplete or duplicate checkpoint parameter layout")
    canonical = {name.replace(".default.", "."): tensor for name, tensor in saved["adapter"].items()}
    if len(canonical) != len(names) or set(canonical) != set(original):
        raise ValueError("Checkpoint must contain exactly the original adapter layout")
    state = saved["state"]
    step = state["step"]
    if (type(step) is not int or not 0 <= step < len(cumulative_tokens)
            or state["decisions"] != step * 16 or state["tokens"] != cumulative_tokens[step]):
        raise ValueError("Checkpoint progress differs from the frozen schedule")
    changed = 0
    for name, tensor in canonical.items():
        initial = original[name]
        if tensor.shape != initial.shape or tensor.dtype != initial.dtype or not torch.isfinite(tensor).all().item():
            raise ValueError("Invalid adapter tensor")
        changed += not torch.equal(tensor, initial)
    if step == 0 and changed:
        raise ValueError("Step zero must equal the original adapter exactly")
    optimizer = saved["optimizer"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Unexpected optimizer groups")
    group = optimizer["param_groups"][0]
    if (group["params"] != list(range(len(names))) or group["lr"] != 5e-5 or group["weight_decay"] != 0
            or tuple(group["betas"]) != (.9, .999) or group["eps"] != 1e-8):
        raise ValueError("Registered AdamW configuration changed")
    if set(optimizer["state"]) != (set(range(len(names))) if step else set()):
        raise ValueError("Incomplete or unexpected optimizer state")
    for index, values in optimizer["state"].items():
        tensor = saved["adapter"][names[index]]
        if values["step"].item() != step:
            raise ValueError("Optimizer update count differs")
        for key in ("exp_avg", "exp_avg_sq"):
            if values[key].shape != tensor.shape or not torch.isfinite(values[key]).all().item():
                raise ValueError("Invalid optimizer moment")
    if saved["torch_rng"].dtype != torch.uint8 or len(saved["cuda_rng"]) != 1 or saved["cuda_rng"][0].dtype != torch.uint8:
        raise ValueError("CPU and single-GPU random states must be retained")
    return {"step": step, "adapter_tensors": len(names), "changed_from_original": changed, "tokens": state["tokens"]}


def main(directory):
    directory = directory.resolve()
    if not directory.is_relative_to(RESULTS.resolve()):
        raise ValueError("Select an attempt from this continuation")
    archive = directory / "evidence.tar.gz"
    if archive.exists() or (directory / "verification.json").exists():
        raise ValueError("Preserve completed verification")
    registration = directory / "registration.json"
    spec = json.loads(registration.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if digest(registration) != analysis["continuation_registration_sha256"]:
        raise ValueError("The verifier belongs to another continuation")
    for name, expected in analysis["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered verification source changed")
    work = json.loads((directory / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if (execution["status"] != "complete" or execution["returncode"] != 0
            or work["status"] not in ("complete", "checkpointed")
            or any(row["registration_sha256"] != digest(registration) for row in (work, execution))
            or work["operation"] != request["operation"] or work["target_step"] != request["target_step"]):
        raise ValueError("Require a terminal successful operation with matching registration")
    for name, expected in spec["sources"].items():
        if (work["source_files"].get(name) != expected or digest(directory / "executed_source" / name) != expected
                or digest(ROOT / name) != expected):
            raise ValueError("Executed source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
        os.environ[name] = "1"
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    original = load_file(ROOT / spec["adapter_directory"] / "adapter_model.safetensors", device="cpu")
    schedule = json.loads((ROOT / spec["schedule"]).read_text(encoding="utf-8"))
    cumulative = [0]
    for batch in schedule["batches"]:
        cumulative.append(cumulative[-1] + sum(r["encoded_tokens"] for r in batch))
    history = {}
    if request["checkpoint"]:
        parent = (ROOT / request["checkpoint"]).resolve()
        if (not parent.is_relative_to(RUNS.resolve()) or digest(parent) != request["checkpoint_sha256"]
                or work["input_checkpoint"] != {"path": parent.relative_to(ROOT).as_posix(), "sha256": request["checkpoint_sha256"]}):
            raise ValueError("Input checkpoint provenance differs")
        saved = torch.load(parent, map_location="cpu", weights_only=True)
        checkpoint_payload(saved, original, digest(registration), cumulative)
        if saved["state"]["step"] != work["initial_step"]:
            raise ValueError("The resumed step differs from its checkpoint")
        history = dict(saved["state"]["validations"])
        del saved
    elif work["operation"] != "baseline" or work["initial_step"] != 0:
        raise ValueError("Only the unchanged baseline may lack a parent checkpoint")
    if work["operation"] in ("baseline", "validate"):
        history[str(work["completed_step"])] = work["validation"]
    if (work["completed_step"] > work["target_step"]
            or (work["status"] == "complete") != (work["completed_step"] == work["target_step"])):
        raise ValueError("Operation completion does not match its interval")
    checks = []
    last = None
    for entry in work["checkpoints"]:
        checkpoint = (ROOT / entry["path"]).resolve()
        if not checkpoint.is_relative_to(RUNS.resolve()) or digest(checkpoint) != entry["sha256"]:
            raise ValueError("Checkpoint content or path differs")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checks.append({**checkpoint_payload(saved, original, digest(registration), cumulative), "sha256": entry["sha256"]})
        if saved["state"]["validations"] != history:
            raise ValueError("The checkpoint lost or rewrote its validation history")
        for step, metrics in saved["state"]["validations"].items():
            location = (ROOT / metrics["directory"]).resolve()
            if not location.is_relative_to(RESULTS.resolve()):
                raise ValueError("Validation provenance points outside this run")
            recorded = json.loads((location / "work.json").read_text(encoding="utf-8"))
            if (recorded["status"] != "complete" or recorded["completed_step"] != int(step)
                    or recorded["registration_sha256"] != digest(registration) or recorded["validation"] != metrics
                    or digest(location / "validation_rows.jsonl") != metrics["rows_sha256"]):
                raise ValueError("Checkpoint validation history does not reproduce")
        last = saved["state"]
        del saved
    if not checks or last["step"] != work["completed_step"]:
        raise ValueError("Missing final checkpoint")
    verification = {"status": "complete", "verified_utc": datetime.now(timezone.utc).isoformat(),
                    "registration_sha256": digest(registration), "verifier_sha256": digest(__file__),
                    "analysis_registration_sha256": digest(ANALYSIS),
                    "operation": work["operation"], "completed_step": work["completed_step"], "checkpoints": checks}
    if work["operation"] == "train":
        expected_steps = list(range(work["initial_step"] + 1, work["completed_step"] + 1))
        if [row["step"] for row in work["updates"]] != expected_steps or len(expected_steps) > spec["max_updates_per_segment"]:
            raise ValueError("Training update sequence differs")
        if any(not math.isfinite(row[key]) or row[key] < 0 for row in work["updates"]
               for key in ("gradient_norm_before_clip", "elapsed_seconds")):
            raise ValueError("Invalid training update diagnostics")
        if any(not math.isfinite(row["mean_loss"]) for row in work["updates"]):
            raise ValueError("Non-finite saved training loss")
        verification["updates_verified"] = len(expected_steps)
    else:
        import dataset
        from train_c import prompt_for
        from transformers import AutoTokenizer
        data = ROOT / spec["teacher_directory"]
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest(data / name) != expected:
                raise ValueError("Teacher data file changed")
        dataset.RUNS = data
        train, _, validation = dataset.load_examples()
        if len(validation) != 278 or {e.id for e in train} & {e.id for e in validation}:
            raise ValueError("Invalid held-out membership")
        cases = {(e.id, d.qid, len(d.options)): (e, d) for e in validation for d in e.decisions}
        probabilities = read_rows(directory / "validation_rows.jsonl", list(cases))
        rows = [json.loads(line) for line in (directory / "validation_rows.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != 1390 or digest(directory / "validation_rows.jsonl") != work["validation"]["rows_sha256"]:
            raise ValueError("Validation row count or hash differs")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True)
        losses, agreement = [], []
        with (directory / "validation_inputs.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                e, d = cases[row["state_id"], row["qid"], row["k"]]
                p, target = row["probabilities"], d.target.tolist()
                prompt = prompt_for(d.type, d.instructions, d.options, e.state)
                encoded = len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
                if (row["target"] != target or row["gold"] != d.gold or row["qtype"] != d.type
                        or row["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        or row["encoded_tokens"] != encoded or encoded > spec["context_bound"]):
                    raise ValueError("Saved validation input or target does not reproduce")
                logits = row["log_probabilities"]
                if len(logits) != len(p) or not all(map(math.isfinite, logits)):
                    raise ValueError("Invalid saved log probabilities")
                softmax = [math.exp(v - max(logits)) for v in logits]
                if max(abs(v / sum(softmax) - q) for v, q in zip(softmax, p, strict=True)) > 1e-12:
                    raise ValueError("Saved probability transform differs")
                kl = sum(t * (math.log(max(t, 1e-12)) - math.log(max(q, 1e-12))) for t, q in zip(target, p, strict=True))
                teacher, student = max(range(len(p)), key=target.__getitem__), max(range(len(p)), key=p.__getitem__)
                if (row["teacher_argmax"] != teacher or row["student_argmax"] != student
                        or not math.isclose(row["kl"], kl, abs_tol=1e-12)):
                    raise ValueError("Saved per-decision validation metrics differ")
                losses.append(kl)
                agreement.append(teacher == student)
                stream.write(json.dumps({"state_id": e.id, "qid": d.qid, "state": e.state, "qtype": d.type,
                                         "instructions": d.instructions, "options": d.options, "target": target,
                                         "gold": d.gold}, ensure_ascii=False) + "\n")
        metrics = {"n": len(rows), "mean_kl": fmean(losses), "teacher_agreement": fmean(agreement)}
        if any(not math.isclose(value, work["validation"][key], rel_tol=0, abs_tol=1e-12) for key, value in metrics.items()):
            raise ValueError("Validation summary does not reproduce")
        verification["validation"] = metrics
        if work["operation"] == "baseline":
            parity = compare_rows(read_rows(ROOT / spec["reference_rows"], list(cases)), probabilities, spec)
            if not parity["passed"] or parity != work["baseline_parity"]:
                raise ValueError("Baseline probability check does not reproduce")
            verification["baseline_parity"] = parity
    verification["scope"] = "One-thread CPU integrity and arithmetic checks, including saved adapter/optimizer values. No GPU numerical replay, throughput or new quality result; original release gates remain unchanged."
    write_json(directory / "verification.json", verification)
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitignore"}
    if any(p.suffix in {".safetensors", ".pt", ".bin", ".gguf", ".npy", ".npz"} for p in files.values()):
        raise ValueError("Keep weights and caches outside result archives")
    for name in ("verify_qwen_continuation.py", "parity_evidence.py", "structure_probe.py"):
        files["verification_source/" + name] = Path(__file__).with_name(name)
    files["analysis_registration.json"] = ANALYSIS
    for name in analysis["sources"]:
        files["verification_source/" + name] = ROOT / name
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
        raise ValueError("Archive content differs")
    write_json(directory / "artifact_manifest.json", expected)
    write_json(directory / "artifact_receipt.json", {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)})
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
