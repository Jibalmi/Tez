"""Diagnose native batch-size sensitivity without replacing the frozen graph reference."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from parity_evidence import compare_rows, read_rows
from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-native-single-20260927"
REGISTRATION = Path(__file__).with_name("qwen_native_single_registration.json")
PARENT = Path(__file__).with_name("qwen_prefix_validation_registration.json")
SINGLE = ROOT / "results/pilot-qwen-prefix-single-20260927/batch-01"


def verify_control_rows(path, cases):
    expected_ids = [(c["state_id"], c["qid"], c["k"]) for c in cases]
    values = read_rows(path, expected_ids)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ordered = sorted(cases, key=lambda c: len(c["prompt"]))
    for index, (row, case) in enumerate(zip(rows, ordered, strict=True)):
        if (row["native_batch"] != index or row["batch_size"] != 1
                or any(row[key] != case[key] for key in ("state_id", "qid", "k", "qtype", "prompt_sha256", "encoded_tokens"))):
            raise ValueError("Native control membership or input provenance changed")
        logp = row["log_probabilities"]
        if (len(logp) != case["k"] or any(type(v) not in (int, float) or not math.isfinite(v) for v in logp)
                or not math.isclose(math.fsum(math.exp(v) for v in logp), 1, rel_tol=0, abs_tol=1e-5)):
            raise ValueError("Invalid native control log probabilities")
        exp = [math.exp(v - max(logp)) for v in logp]
        if any(not math.isclose(p, q / math.fsum(exp), rel_tol=0, abs_tol=1e-12)
               for p, q in zip(row["probabilities"], exp, strict=True)):
            raise ValueError("Native control probabilities do not reproduce its readout")
    return values


def prepare():
    if REGISTRATION.exists() or DIRECTORY.exists():
        raise ValueError("Preserve registered control and evidence")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    result = json.loads((SINGLE / "verification.json").read_text(encoding="utf-8"))
    if (result["status"] != "complete" or result["real_predictions_per_arm"] != 1390
            or result["passed"] is not False or result["graph_uncaptured_parity"]["max_probability_delta"] != 0):
        raise ValueError("Require the complete observed graph failure with exact uncaptured agreement")
    receipt = json.loads((SINGLE / "artifact_receipt.json").read_text(encoding="utf-8"))
    if digest(SINGLE / "evidence.tar.gz") != receipt["sha256"]:
        raise ValueError("Failed graph evidence changed")
    if digest(SINGLE / "inputs.json") != parent["input_sha256"]:
        raise ValueError("Original validation inputs changed")
    sources = dict(parent["sources"])
    for name in (Path(__file__).name, "test_qwen_native_single_control.py", "report_qwen_public.py", "qwen_public_evaluation.py"):
        path = Path(__file__).with_name(name)
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    prerequisites = dict(parent["prerequisites"])
    prerequisites[PARENT.relative_to(ROOT).as_posix()] = digest(PARENT)
    for name in ("verification.json", "artifact_receipt.json", "registration.json", "graph_rows.jsonl", "inputs.json"):
        path = SINGLE / name
        prerequisites[path.relative_to(ROOT).as_posix()] = digest(path)
    for mapping in (sources, prerequisites):
        if any(digest(ROOT / name) != expected for name, expected in mapping.items()):
            raise ValueError("A frozen source or prerequisite changed")
    spec = {key: parent[key] for key in ("base_revision", "base_sha256", "teacher_directory", "packages", "kernel_packages",
                                        "model_directory", "adapter_sha256", "reference_rows", "input_sha256",
                                        "argmax_agreement_min", "max_probability_delta_max", "mean_total_variation_max")}
    spec.update(status="registered_before_native_batch_one_control", registered_utc=datetime.now(timezone.utc).isoformat(),
                sources=sources, prerequisites=prerequisites,
                graph_rows=(SINGLE / "graph_rows.jsonl").relative_to(ROOT).as_posix(),
                soft_deadline_seconds=1080, worker_timeout_seconds=1200,
                inference="Original frozen full-head PredictC model with native SDPA/PyTorch recurrence, unmerged FP32 LoRA over BF16 base, cache on, eval, no_grad and BF16 autocast. Original1390 validation prompts in stable native character-length order; batch size1 is the only intended native-inference change. No trailing graph padding, graph capture, restricted head, temperature fitting or input truncation.",
                analysis="Independently validate all1390 probabilities and exact prompt/token membership. Compare native full-head B16 reference versus fresh native B1 control, then native B1 control versus the saved failed prefix graph B1. Report both using the unchanged descriptive numerical limits. Do not replace the original B16 reference or reconsider the failed graph's acceptance. The second comparison includes padding, readout/cache changes and capture together; this control does not isolate each of them.",
                compute="Shared run_heavy, four threads,22-minute lease,18-minute soft deadline,20-minute hard child timeout,80/76C pacing,84C wrapper. No latency or paid resources.",
                scope="Diagnostic control only. No training, checkpoint selection, public accuracy, graph acceptance, threshold change, speed measurement or runtime promotion.")
    DIRECTORY.mkdir()
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "status": spec["status"]}))


def worker(spec, directory):
    started = time.monotonic()
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}, "rows_completed": 0}
    write_json(directory / "work.json", receipt)
    try:
        for group in ("sources", "prerequisites"):
            for name, expected in spec[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Frozen {group} changed: {name}")
                if group == "sources":
                    destination = directory / "executed_source" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, destination)
                    receipt["source_files"][name] = expected
        if {name: version(name) for name in spec["packages"]} != spec["packages"]:
            raise ValueError("Package versions changed")
        import torch
        from functools import partial
        from huggingface_hub import snapshot_download
        import common
        paced = partial(common.thermal_guard, hot=80, cool=76, every=0.0)

        def guard():
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Preserve incomplete native control")
            paced(force=True)
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Native control deadline reached while pacing")

        from eval_arms import PredictC
        from qwen_prefix_validation import validation_inputs
        from train_c import kernel_packages, letter_logp
        if kernel_packages() != spec["kernel_packages"]:
            raise ValueError("Native kernel availability changed")
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Base weights changed")
        guard()
        reference = PredictC("c-qwen08-30k")
        if (reference.weights != {"adapter": spec["adapter_sha256"]} or reference.model.training
                or not reference.model.config.use_cache or reference.model.config._attn_implementation != "sdpa"):
            raise ValueError("Native reference inference settings changed")
        if torch.get_num_threads() != 4 or list(torch.cuda.get_device_capability()) != [12, 0]:
            raise ValueError("Require the registered laptop and four threads")
        cases = validation_inputs(reference.tok, spec)
        write_json(directory / "inputs.json", cases)
        if digest(directory / "inputs.json") != spec["input_sha256"]:
            raise ValueError("Exact original validation input audit changed")
        receipt.update(status="running", own_pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                       capability=list(torch.cuda.get_device_capability()), cuda=torch.version.cuda,
                       threads=torch.get_num_threads(), packages=spec["packages"], kernel_packages=kernel_packages(),
                       weights=reference.weights, base_sha256=spec["base_sha256"], attention_implementation="sdpa",
                       batch_size=1, cache=True, lora_merged=False, readout="full_vocabulary", gradient_mode="no_grad",
                       autocast_dtype="torch.bfloat16", inputs_sha256=digest(directory / "inputs.json"))
        write_json(directory / "work.json", receipt)
        torch.cuda.reset_peak_memory_stats()
        path = directory / "control_rows.jsonl"
        with path.open("x", encoding="utf-8") as output, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, case in enumerate(sorted(cases, key=lambda c: len(c["prompt"]))):
                guard()
                logp, _ = letter_logp(reference.model, reference.tok, reference.letter_ids,
                                      [case["prompt"]], [case["k"]], "cuda", max_len=4096)
                values = logp[0].double().cpu().tolist()
                if len(values) != case["k"] or not all(map(math.isfinite, values)):
                    raise ValueError("Invalid full-head native readout")
                exp = [math.exp(v - max(values)) for v in values]
                row = {key: case[key] for key in ("state_id", "qid", "k", "qtype", "prompt_sha256", "encoded_tokens")}
                row.update(native_batch=index, batch_size=1, log_probabilities=values,
                           probabilities=[v / math.fsum(exp) for v in exp])
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                receipt["rows_completed"] += 1
                if receipt["rows_completed"] % 25 == 0:
                    write_json(directory / "work.json", receipt)
        values = read_rows(path, [(c["state_id"], c["qid"], c["k"]) for c in cases])
        receipt.update(status="complete", native_batch_size_parity=compare_rows(read_rows(ROOT / spec["reference_rows"]), values, spec),
                       graph_against_native_single=compare_rows(values, read_rows(ROOT / spec["graph_rows"]), spec),
                       rows_sha256=digest(path), peak_allocated_bytes=torch.cuda.max_memory_allocated(), scope=spec["scope"])
        return 0
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if "common" in locals():
            receipt["thermal_governor"] = common.THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(directory / "work.json", receipt)


def report():
    directory = DIRECTORY / "control"
    if (directory / "evidence.tar.gz").exists():
        raise ValueError("Preserve completed diagnostic")
    spec = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    work = json.loads((directory / "work.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if (work["status"] != "complete" or execution["status"] != "complete" or execution["returncode"] != 0
            or any(r["registration_sha256"] != digest(directory / "registration.json") for r in (work, execution))
            or digest(__file__) != spec["sources"][Path(__file__).relative_to(ROOT).as_posix()]):
        raise ValueError("Require complete registered native control")
    for name, expected in spec["sources"].items():
        if work["source_files"].get(name) != expected or digest(directory / "executed_source" / name) != expected:
            raise ValueError("Executed source differs")
    for name, expected in spec["prerequisites"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen prerequisite differs")
    expected_settings = {"batch_size": 1, "cache": True, "lora_merged": False, "readout": "full_vocabulary",
                         "gradient_mode": "no_grad", "autocast_dtype": "torch.bfloat16", "threads": 4,
                         "capability": [12, 0], "packages": spec["packages"], "kernel_packages": spec["kernel_packages"],
                         "weights": {"adapter": spec["adapter_sha256"]}, "base_sha256": spec["base_sha256"],
                         "attention_implementation": "sdpa", "rows_completed": 1390}
    if any(work.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Native control settings differ")
    if (digest(directory / "inputs.json") != spec["input_sha256"] or work["inputs_sha256"] != spec["input_sha256"]
            or digest(directory / "control_rows.jsonl") != work["rows_sha256"]):
        raise ValueError("Saved inputs or rows differ")
    cases = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    if len(cases) != 1390 or len({c["state_id"] for c in cases}) != 278:
        raise ValueError("Incomplete original validation membership")
    values = verify_control_rows(directory / "control_rows.jsonl", cases)
    native = compare_rows(read_rows(ROOT / spec["reference_rows"]), values, spec)
    graph = compare_rows(values, read_rows(ROOT / spec["graph_rows"]), spec)
    if native != work["native_batch_size_parity"] or graph != work["graph_against_native_single"]:
        raise ValueError("Independent saved-row arithmetic differs")
    result = {"status": "complete", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": digest(directory / "registration.json"), "verifier_sha256": digest(__file__),
              "decisions": 1390, "native_batch_size_parity": native, "graph_against_native_single": graph,
              "scope": spec["scope"]}
    write_json(directory / "verification.json", result)
    from report_qwen_public import archive_files
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    files.update({"native_batch16_reference.jsonl": ROOT / spec["reference_rows"], "failed_graph_rows.jsonl": ROOT / spec["graph_rows"]})
    artifact = archive_files(directory, files)
    print(json.dumps({"verification": result, "artifact": artifact}, indent=2))


def driver():
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise ValueError("Use the shared compute wrapper")
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    directory = DIRECTORY / "control"
    directory.mkdir(exist_ok=False)
    shutil.copyfile(REGISTRATION, directory / "registration.json")
    env = dict(os.environ)
    env.update(TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(directory),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(directory / "execution.json", receipt)
    with (directory / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
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
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.prepare or args.report:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
            os.environ[name] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        prepare() if args.prepare else report()
    else:
        raise SystemExit(worker(json.loads(REGISTRATION.read_text(encoding="utf-8")), DIRECTORY / "control")
                         if args.worker else driver())
