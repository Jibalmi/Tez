"""Matched local public evaluation after frozen continuation selection and reload parity."""
from collections import defaultdict
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

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-continuation-20260927/public"
REGISTRATION = Path(__file__).with_name("qwen_public_registration.json")
CONTINUATION = Path(__file__).with_name("qwen_continuation_registration.json")
RELOAD = Path(__file__).with_name("qwen_continuation_reload_registration.json")
SUITE = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
PREPARATION = DIRECTORY.parent / "public_preflight.json"
AUDIT = ROOT / "runs/pilot-qwen-continuation-20260927/public_input_audit.json"
ARMS = ("baseline", "selected")
PHASES = ("typed", "apps", "intent", "semif")


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def task_phase(task):
    if task == "typed_decisions":
        return "typed"
    for prefix, phase in (("app_", "apps"), ("massive_", "intent"), ("semif_", "semif")):
        if task.startswith(prefix):
            return phase
    raise ValueError("Unexpected public task")


def grouped_cases(task, cases):
    if task != "typed_decisions":
        return [[c] for c in cases]
    groups = defaultdict(list)
    for case in cases:
        groups[case["group"]].append(case)
    return list(groups.values())


def presented(case, reverse):
    # No labels, soft targets, benchmark identifiers or groups are passed to the predictor.
    safe = {key: case[key] for key in ("qtype", "instructions", "options", "state")}
    if reverse:
        safe["options"] = list(reversed(safe["options"]))
    return safe


def audit_case(tokenizer, case, reverse=False):
    from train_c import prompt_for
    safe = presented(case, reverse)
    prompt = prompt_for(safe["qtype"], safe["instructions"], list(safe["options"]), safe["state"])
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    if not ids or len(ids) > 4096:
        raise ValueError("Public input would be empty or truncated")
    return {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids_sha256": json_sha(ids), "encoded_tokens": len(ids), "prompt_characters": len(prompt),
            "case_sha256": json_sha(case), "option_order": list(reversed(range(len(case["options"]))))
            if reverse else list(range(len(case["options"])))}


def make_row(task, case, logp, audit, reverse=False):
    z = [float(value) for value in logp]
    k = len(case["options"])
    order = list(reversed(range(k))) if reverse else list(range(k))
    if len(z) != k or not all(map(math.isfinite, z)) or audit["option_order"] != order:
        raise ValueError("Invalid option readout or option-order audit")
    e = [math.exp(value - max(z)) for value in z]
    p = [value / sum(e) for value in e]
    canonical = list(reversed(p)) if reverse else p
    row = {key: case.get(key) for key in ("id", "lang", "qtype", "gold", "workflow", "group", "family")}
    row.update(task=task, k=k, extra=case.get("extra", {}), **audit, presented_log_probabilities=z,
               presented_probabilities=p, probabilities=canonical, logits=list(reversed(z)) if reverse else z,
               pred=max(range(k), key=canonical.__getitem__))
    return row


def inputs(spec):
    for name, expected in {**spec["sources"], **spec["prerequisites"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered public source or prerequisite changed")
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    if len(suite) != 21 or sum(map(len, suite.values())) != 6151:
        raise ValueError("Public suite membership changed")
    for task, cases in suite.items():
        if len({c["id"] for c in cases}) != len(cases) or task_phase(task) not in PHASES:
            raise ValueError("Repeated public case or unknown task")
        for case in cases:
            k = len(case["options"])
            if not 2 <= k <= 26 or not isinstance(case["gold"], int) or not 0 <= case["gold"] < k:
                raise ValueError("Invalid public option count or label")
    return suite


def bind_inputs(spec):
    """Both arms require the same completed validation-only selection and strict reload check."""
    from qwen_continuation_reload import DIRECTORY as reload_dir, MODEL, bind_inputs as reload_bind
    reload_spec = json.loads(RELOAD.read_text(encoding="utf-8"))
    selected = reload_bind(reload_spec)
    verified = json.loads((reload_dir / "verification.json").read_text(encoding="utf-8"))
    artifact = json.loads((reload_dir / "artifact_receipt.json").read_text(encoding="utf-8"))
    if (verified["status"] != "complete" or not verified["passed"]
            or verified["selected_step"] != selected["selected_step"]
            or verified["selection_sha256"] != selected["selection_sha256"]
            or verified["registration_sha256"] != digest(RELOAD)
            or digest(reload_dir / "evidence.tar.gz") != artifact["sha256"]):
        raise ValueError("Require the verified selected-model reload")
    with tarfile.open(reload_dir / "evidence.tar.gz", "r:gz") as bundle:
        for name in ("verification.json", "bindings.json"):
            if bundle.extractfile(name).read() != (reload_dir / name).read_bytes():
                raise ValueError("Reload metadata differs from its verified archive")
    if json.loads((reload_dir / "bindings.json").read_text(encoding="utf-8")) != selected:
        raise ValueError("Reload belongs to another selected export")
    baseline = ROOT / spec["baseline_model"]
    paths = {**selected["files"], **{(reload_dir / name).relative_to(ROOT).as_posix(): digest(reload_dir / name)
                                  for name in ("verification.json", "artifact_receipt.json", "evidence.tar.gz")}}
    return {"selected_step": selected["selected_step"], "selection_sha256": selected["selection_sha256"],
            "files": paths, "arms": {
                "baseline": {"model_directory": baseline.relative_to(ROOT).as_posix(), "arm": baseline.name,
                             "adapter_sha256": spec["baseline_adapter_sha256"]},
                "selected": {"model_directory": MODEL.relative_to(ROOT).as_posix(), "arm": MODEL.name,
                             "adapter_sha256": selected["adapter_sha256"]}}}


def worker(arm, phase, directory, spec):
    receipt = {"status": "preparing", "arm": arm, "phase": phase, "source_files": {}, "tasks": {},
               "registration_sha256": digest(REGISTRATION), "bindings_sha256": digest(directory / "bindings.json"),
               "started_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    path = directory / "work.json"
    write_json(path, receipt)
    try:
        suite = inputs(spec)
        bindings = json.loads((directory / "bindings.json").read_text(encoding="utf-8"))
        if bind_inputs(spec) != bindings or any(version(k) != v for k, v in spec["packages"].items()):
            raise ValueError("Bound model or environment changed")
        for name, expected in spec["sources"].items():
            target = directory / "executed_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            receipt["source_files"][name] = expected
        import torch
        import common
        previous_guard = common.thermal_guard

        def guarded(*args, **kwargs):
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Bounded public phase deadline reached")
            previous_guard(*args, **dict(kwargs, hot=80, cool=76, every=0.0))
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Bounded public phase deadline reached during thermal pause")

        common.thermal_guard = guarded
        from eval_arms import PredictC
        from huggingface_hub import snapshot_download
        from train_c import kernel_packages
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if ([digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]
                or kernel_packages() != spec["kernel_packages"]
                or any(digest(base / name) != expected for name, expected in spec["base_metadata"].items())):
            raise ValueError("Base weights or native kernel environment changed")
        guarded(force=True)
        candidate = PredictC(bindings["arms"][arm]["arm"])
        base_dtypes = sorted({str(p.dtype) for name, p in candidate.model.named_parameters() if "lora_" not in name})
        adapter_dtypes = sorted({str(p.dtype) for name, p in candidate.model.named_parameters() if "lora_" in name})
        if (candidate.weights != {"adapter": bindings["arms"][arm]["adapter_sha256"]}
                or not candidate.model.config.use_cache or candidate.info["lora_merged"]
                or candidate.model.config._attn_implementation != "sdpa" or candidate.batch_size != 16
                or candidate.model.training or base_dtypes != ["torch.bfloat16"] or adapter_dtypes != ["torch.float32"]
                or torch.get_num_threads() != 4 or "RTX 5080 Laptop" not in torch.cuda.get_device_name()):
            raise ValueError("Unexpected matched inference settings")
        receipt.update(status="running", gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
                       packages={k: version(k) for k in spec["packages"]}, kernel_packages=kernel_packages(),
                       weights=candidate.weights, base_sha256=spec["base_sha256"], selected_step=bindings["selected_step"],
                       batch_size=16, dtype="bf16", lora_merged=False, cache=True, attention_implementation="sdpa",
                       temperature=1.0, context_bound=4096, prompt_layout="question_first", output_head="full_vocabulary",
                       base_parameter_dtypes=base_dtypes, adapter_parameter_dtypes=adapter_dtypes, model_training=False,
                       compute_capability=list(torch.cuda.get_device_capability()), cuda_version=torch.version.cuda)
        write_json(path, receipt)
        torch.cuda.reset_peak_memory_stats()
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        for task, cases in suite.items():
            if task_phase(task) != phase:
                continue
            receipt["tasks"][task] = {}
            for reverse in ((False, True) if task.startswith("massive_") else (False,)):
                order_name = "reversed" if reverse else "primary"
                groups = grouped_cases(task, cases)
                flat = [c for group in groups for c in group]
                actual_audit = {c["id"]: audit_case(candidate.tok, c, reverse) for c in flat}
                if actual_audit != audit[task][order_name]:
                    raise ValueError("Public prompt, tokens or option ordering changed")
                safe = [[presented(c, reverse) for c in group] for group in groups]
                logp = [z for group in candidate.groups(safe) for z in group]
                rows = [make_row(task, c, z, actual_audit[c["id"]], reverse) for c, z in zip(flat, logp, strict=True)]
                destination = directory / f"rows_{task}_{order_name}.jsonl"
                with destination.open("x", encoding="utf-8") as output:
                    for row in rows:
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                receipt["tasks"][task][order_name] = {"rows": len(rows), "sha256": digest(destination)}
                write_json(path, receipt)
                print(json.dumps({"task": task, "order": order_name, "saved_rows": len(rows)}), flush=True)
        receipt.update(status="complete", peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return 0
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "common" in locals():
            receipt["thermal_governor"] = dict(common.THERMAL)
        receipt.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, receipt)


def driver(arm, phase, directory, spec):
    import psutil
    inputs(spec)
    bindings = bind_inputs(spec)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("rows_*.jsonl\n", encoding="utf-8")
    shutil.copy2(REGISTRATION, directory / "registration.json")
    write_json(directory / "bindings.json", bindings)
    model = ROOT / bindings["arms"][arm]["model_directory"]
    env = dict(os.environ, TEZ_PILOT_RUNS=str(model.parent.parent), TEZ_PILOT_RESULTS=str(directory),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "arm": arm, "phase": phase, "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "bindings_sha256": digest(directory / "bindings.json"),
               "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(directory / "execution.json", receipt)
    with (directory / "worker.log").open("x", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm,
                                    "--phase", phase], env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
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
    receipt.update(status="complete" if code == 0 else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    directory = DIRECTORY / args.arm / args.phase
    raise SystemExit(worker(args.arm, args.phase, directory, spec) if args.worker else driver(args.arm, args.phase, directory, spec))
