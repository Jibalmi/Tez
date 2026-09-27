"""Require the selected private adapter to reproduce its saved validation predictions after reload."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from qwen_continuation_selection import choose_checkpoint
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/reload"
SELECTION = ROOT / "results/pilot-qwen-continuation-20260927/selection/selection.json"
MODEL = ROOT / "runs/pilot-qwen-continuation-20260927/export/models/c-qwen08-continuation"
REGISTRATION = Path(__file__).with_name("qwen_continuation_reload_registration.json")
CONTINUATION = Path(__file__).with_name("qwen_continuation_registration.json")
EXPORT = Path(__file__).with_name("qwen_continuation_export_registration.json")


def selected_step(selection, continuation_sha, export_sha):
    """Reject a different checkpoint, incomplete selection, or inconsistent unchanged-model flag."""
    if (selection.get("status") != "complete" or selection["continuation_registration_sha256"] != continuation_sha
            or selection["export_registration_sha256"] != export_sha):
        raise ValueError("Require the complete registered selection")
    metrics = {int(key): value for key, value in selection["validation_metrics"].items()}
    choice = choose_checkpoint(metrics)
    if (any(selection[key] != value for key, value in choice.items())
            or selection["retained_original_weights"] != (choice["selected_step"] == 0)
            or set(selection["validation_evidence"]) != {"0", "50", "100", "150"}):
        raise ValueError("Selected checkpoint differs from the frozen validation-only rule")
    return choice["selected_step"]


def register():
    if REGISTRATION.exists() or SELECTION.exists() or DIRECTORY.exists():
        raise ValueError("Preserve the reload protocol")
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    exported = json.loads(EXPORT.read_text(encoding="utf-8"))
    sources = {**continuation["sources"], **exported["sources"]}
    for name in ("qwen_continuation_reload.py", "test_qwen_continuation_reload.py"):
        source = Path(__file__).with_name(name)
        sources[source.relative_to(ROOT).as_posix()] = digest(source)
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("A prerequisite source changed before registration")
    spec = {"status": "registered_before_selection_or_reload", "registered_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources, "continuation_registration_sha256": digest(CONTINUATION),
            "export_registration_sha256": digest(EXPORT), "packages": continuation["packages"],
            "argmax_agreement_min": 1.0, "max_probability_delta_max": 1e-6, "mean_total_variation_max": 1e-7,
            "soft_deadline_seconds": 480, "worker_timeout_seconds": 600,
            "method": "After all four validations and frozen checkpoint selection, hash the selected private adapter, unchanged config, new student metadata, selected checkpoint, raw selected validation rows and selection record before model loading. Reload using PredictC, then reuse the frozen continuation validation routine: all1390 original decisions, batch16 sorted prompt character length, native full head/cache-on/BF16 autocast/T1, complete inputs. No graph path or optimization is introduced.",
            "acceptance": "All1390 probabilities must match the selected validation checkpoint: argmax exactly1, maximum delta<=0.000001 and mean TV<=0.0000001. No limits relaxed on failure. This tests serialization and reload, not new quality, speed or export-to-another-runtime parity.",
            "compute": "Shared run_heavy lock, four threads, twelve-minute lease, ten-minute hard worker limit, eight-minute soft validation limit,80/76C process pacing and84C wrapper. No additional cloud spend.",
            "scope": "No public inputs or labels. A pass permits separately registered selected-candidate evaluation; it is not a quality improvement or release gate. Failed reload must be repaired and reverified without selecting a different checkpoint using public results."}
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "status": spec["status"]}))


def bind_inputs(spec):
    if (digest(CONTINUATION) != spec["continuation_registration_sha256"]
            or digest(EXPORT) != spec["export_registration_sha256"]):
        raise ValueError("Upstream registration changed")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    step = selected_step(selection, digest(CONTINUATION), digest(EXPORT))
    if selection["model_directory"] != MODEL.relative_to(ROOT).as_posix():
        raise ValueError("Select only this continuation's private export")
    reference_dir = (ROOT / selection["validation_evidence"][str(step)]["directory"]).resolve()
    if not reference_dir.is_relative_to(DIRECTORY.parent / "training"):
        raise ValueError("Selected reference lies outside this continuation")
    checkpoint = (ROOT / selection["selected_checkpoint"]["path"]).resolve()
    if not checkpoint.is_relative_to(ROOT / "runs/pilot-qwen-continuation-20260927/training"):
        raise ValueError("Selected checkpoint lies outside this continuation")
    work = json.loads((reference_dir / "work.json").read_text(encoding="utf-8"))
    if (work["status"] != "complete" or work["completed_step"] != step
            or work["checkpoints"][-1] != selection["selected_checkpoint"]):
        raise ValueError("Selected checkpoint does not belong to its validation")
    expected = {MODEL / "adapter/adapter_model.safetensors": selection["export"]["adapter_sha256"],
                MODEL / "adapter/adapter_config.json": selection["export"]["config_sha256"],
                MODEL / "student.json": selection["student_sha256"], checkpoint: selection["selected_checkpoint"]["sha256"],
                reference_dir / "validation_rows.jsonl": work["validation"]["rows_sha256"],
                reference_dir / "evidence.tar.gz": selection["validation_evidence"][str(step)]["artifact"]["sha256"]}
    if any(digest(path) != expected_sha for path, expected_sha in expected.items()):
        raise ValueError("Selected export, checkpoint or reference evidence changed")
    info = json.loads((MODEL / "student.json").read_text(encoding="utf-8"))
    if (info["selected_continuation_step"] != step or info["selected_checkpoint"] != selection["selected_checkpoint"]
            or info["temperatures"] != {key: 1.0 for key in ("choice", "choice_many", "noul", "score")}):
        raise ValueError("Export metadata does not describe the selected T1 checkpoint")
    expected.update({SELECTION: digest(SELECTION), reference_dir / "work.json": digest(reference_dir / "work.json")})
    return {"selected_step": step, "selection_sha256": digest(SELECTION),
            "reference_directory": reference_dir.relative_to(ROOT).as_posix(),
            "reference_rows": (reference_dir / "validation_rows.jsonl").relative_to(ROOT).as_posix(),
            "adapter_sha256": selection["export"]["adapter_sha256"],
            "files": {path.relative_to(ROOT).as_posix(): value for path, value in expected.items()}}


def worker():
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    bindings = json.loads((DIRECTORY / "bindings.json").read_text(encoding="utf-8"))
    path = DIRECTORY / "work.json"
    receipt = {"status": "preparing", "registration_sha256": digest(REGISTRATION),
               "bindings_sha256": digest(DIRECTORY / "bindings.json"), "source_files": {},
               "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    write_json(path, receipt)
    try:
        if bind_inputs(spec) != bindings:
            raise ValueError("Bound selected inputs changed before inference")
        for name, expected in spec["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Registered reload source changed")
            target = DIRECTORY / "executed_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            receipt["source_files"][name] = expected
        continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
        for name, expected in continuation["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen continuation input changed")
        if any(version(name) != expected for name, expected in spec["packages"].items()):
            raise ValueError("Reload packages changed")
        import torch
        import common
        from functools import partial
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        from eval_arms import PredictC
        from huggingface_hub import snapshot_download
        from parity_evidence import compare_rows, read_rows
        from qwen_continuation import data_for, validate
        from train_c import kernel_packages
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=continuation["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [continuation["base_sha256"]]:
            raise ValueError("Frozen base checkpoint changed")
        thermal_guard(force=True)
        candidate = PredictC("c-qwen08-continuation")
        if candidate.weights != {"adapter": bindings["adapter_sha256"]}:
            raise ValueError("Loaded another adapter")
        _, heldout = data_for(candidate.tok, continuation)
        receipt.update(status="running", gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
                       packages={name: version(name) for name in spec["packages"]}, kernel_packages=kernel_packages(),
                       weights=candidate.weights, base_sha256=continuation["base_sha256"],
                       attention_implementation=candidate.model.config._attn_implementation,
                       selected_step=bindings["selected_step"])
        write_json(path, receipt)
        torch.cuda.reset_peak_memory_stats()
        probabilities, metrics = validate(candidate.model, candidate.tok, candidate.letter_ids, heldout, DIRECTORY,
                                          {**continuation, "soft_deadline_seconds": spec["soft_deadline_seconds"]},
                                          thermal_guard, started)
        ids = [(c["state_id"], c["qid"], c["k"]) for c in heldout]
        parity = compare_rows(read_rows(ROOT / bindings["reference_rows"], ids), probabilities, spec)
        receipt.update(status="complete", passed=parity["passed"], parity=parity, validation=metrics,
                       cache=candidate.model.config.use_cache, lora_merged=candidate.info["lora_merged"],
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return 0 if parity["passed"] else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    bindings = bind_inputs(spec)
    DIRECTORY.mkdir(exist_ok=False)
    shutil.copy2(REGISTRATION, DIRECTORY / "registration.json")
    write_json(DIRECTORY / "bindings.json", bindings)
    env = dict(os.environ)
    env.update(TEZ_PILOT_RUNS=str(MODEL.parent.parent), TEZ_PILOT_RESULTS=str(DIRECTORY), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "bindings_sha256": digest(DIRECTORY / "bindings.json"),
               "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(DIRECTORY / "execution.json", receipt)
    with (DIRECTORY / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=spec["worker_timeout_seconds"])
        except subprocess.TimeoutExpired:
            for child in psutil.Process(process.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait()
            code = 124
    receipt.update(status="complete" if code in (0, 2) else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(DIRECTORY / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


def report():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
        os.environ[name] = "1"
    from export_qwen_continuation import archive_records, validation_metrics
    from parity_evidence import compare_rows, read_rows
    archive = DIRECTORY / "evidence.tar.gz"
    if archive.exists():
        raise ValueError("Preserve completed reload evidence")
    spec = json.loads((DIRECTORY / "registration.json").read_text(encoding="utf-8"))
    if digest(REGISTRATION) != digest(DIRECTORY / "registration.json"):
        raise ValueError("Executed reload registration differs")
    bindings = json.loads((DIRECTORY / "bindings.json").read_text(encoding="utf-8"))
    work = json.loads((DIRECTORY / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    if (work["status"] != "complete" or execution["status"] != "complete" or bind_inputs(spec) != bindings
            or any(r["registration_sha256"] != digest(REGISTRATION) for r in (work, execution))
            or any(r["bindings_sha256"] != digest(DIRECTORY / "bindings.json") for r in (work, execution))):
        raise ValueError("Require a complete reload with unchanged bound inputs")
    for name, expected in spec["sources"].items():
        if (work["source_files"].get(name) != expected or digest(DIRECTORY / "executed_source" / name) != expected
                or digest(ROOT / name) != expected):
            raise ValueError("Reload source provenance differs")
    analysis = Path(__file__).with_name("qwen_continuation_analysis_registration.json")
    _, _, reference_rows, _ = archive_records(ROOT / bindings["reference_directory"], digest(CONTINUATION), digest(analysis))
    _, reference_cases = validation_metrics(reference_rows)
    rows = [json.loads(line) for line in (DIRECTORY / "validation_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics, _ = validation_metrics(rows, reference_cases)
    reference = read_rows(ROOT / bindings["reference_rows"], list(reference_cases))
    candidate = read_rows(DIRECTORY / "validation_rows.jsonl", list(reference_cases))
    parity = compare_rows(reference, candidate, spec)
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    if (parity != work["parity"] or parity["passed"] != work["passed"]
            or execution["returncode"] != (0 if parity["passed"] else 2)
            or work["weights"] != {"adapter": bindings["adapter_sha256"]}
            or work["selected_step"] != bindings["selected_step"] or work["base_sha256"] != continuation["base_sha256"]
            or work["packages"] != spec["packages"] or not work["cache"] or work["lora_merged"]
            or work["attention_implementation"] != "sdpa"
            or digest(DIRECTORY / "validation_rows.jsonl") != work["validation"]["rows_sha256"]
            or any(not math.isclose(value, work["validation"][key], rel_tol=0, abs_tol=1e-12) for key, value in metrics.items())):
        raise ValueError("Recomputed reload outcome or actual execution settings differ")
    result = {"status": "complete", "passed": parity["passed"], "parity": parity, "validation": metrics,
              "selected_step": bindings["selected_step"], "registration_sha256": digest(REGISTRATION),
              "selection_sha256": bindings["selection_sha256"], "scope": spec["scope"]}
    write_json(DIRECTORY / "verification.json", result)
    files = {path.relative_to(DIRECTORY).as_posix(): path for path in DIRECTORY.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.name != ".gitignore"}
    files.update({"selected_reference_rows.jsonl": ROOT / bindings["reference_rows"], "selection.json": SELECTION})
    if any(path.suffix in {".pt", ".safetensors", ".bin", ".npy", ".npz"} for path in files.values()):
        raise ValueError("Do not archive private weights or caches")
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected reload evidence member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("Reload evidence archive differs")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual)}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"verification": result, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
    elif args.report:
        report()
    else:
        if "RUN_HEAVY_THREADS" not in os.environ:
            raise RuntimeError("Use C:/temp/run_heavy.py")
        raise SystemExit(worker() if args.worker else driver())
