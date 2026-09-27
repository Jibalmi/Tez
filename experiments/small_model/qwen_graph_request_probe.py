"""Validate a multi-shape graph cache on the entire planned fresh-state latency workload."""
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
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-graph-requests-20260927"
REGISTRATION = Path(__file__).with_name("qwen_graph_request_registration.json")
AUDIT = ROOT / "runs/pilot-qwen-graph-requests-20260927/input_audit.json"
PARENT = Path(__file__).with_name("qwen_prefix_validation_registration.json")
PRIOR = (ROOT / "results/pilot-qwen-prefix-validation-20260927/batch-16",
         ROOT / "results/pilot-qwen-prefix-single-20260927/batch-01")


def schedule():
    counter = 0
    for n in (1, 5, 10, 50):
        for phase, repetitions in (("warmup", 10), ("timed", 30 if n == 50 else 100)):
            for _ in range(repetitions):
                counter += 1
                yield {"request": counter, "n_questions": n, "phase": phase}


def request_cases(request):
    sys.path.insert(0, str(ROOT / "experiments"))
    from vs_laya_speed import fresh_state, qs
    from tez.schema import parse_question
    state = fresh_state(request["request"])
    cases = []
    for qid, definition in qs(request["n_questions"]).items():
        question = parse_question(qid, definition)
        cases.append({"state": state, "qtype": question.type, "instructions": question.instructions_text,
                      "options": question.options()})
    return cases


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_request(tokenizer, request, buckets):
    from qwen_graph_requests import request_batches
    from train_c import prompt_for
    cases = request_cases(request)
    encoded = []
    for case in cases:
        prompt = prompt_for(case["qtype"], case["instructions"], list(case["options"]), case["state"])
        tokens = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        encoded.append({"prompt": prompt, "encoded_tokens": len(tokens), "k": len(case["options"]),
                        "qtype": case["qtype"], "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "token_ids_sha256": json_sha(tokens)})
    return {**request, "case_sha256": json_sha(cases), "batches": request_batches(encoded, buckets),
            "cases": [{k: v for k, v in c.items() if k != "prompt"} for c in encoded]}


def normalized(logp, k):
    if len(logp) != k or not all(type(v) in (int, float) and math.isfinite(v) for v in logp):
        raise ValueError("Missing or invalid option readout")
    exp = [math.exp(v - max(logp)) for v in logp]
    if not math.isclose(sum(math.exp(v) for v in logp), 1, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Readout is not normalized log probability")
    return [v / sum(exp) for v in exp]


def metrics(pairs, spec):
    if not pairs:
        raise ValueError("No complete predictions")
    argmax = sum(max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__) for p, q in pairs) / len(pairs)
    maximum = max(abs(a - b) for p, q in pairs for a, b in zip(p, q, strict=True))
    tv = sum(sum(abs(a - b) for a, b in zip(p, q, strict=True)) / 2 for p, q in pairs) / len(pairs)
    return {"decisions": len(pairs), "argmax_agreement": argmax, "max_probability_delta": maximum,
            "mean_total_variation": tv, "passed": argmax >= spec["argmax_agreement_min"]
            and maximum <= spec["max_probability_delta_max"] and tv <= spec["mean_total_variation_max"]}


def require_prior(spec):
    for directory in PRIOR:
        result = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
        if (result["status"] != "complete" or result["passed"] is not True
                or result["real_predictions_per_arm"] != 1390):
            raise ValueError("Both full native-prefix graph validations must pass")
        if digest(directory / "evidence.tar.gz") != spec["prerequisites"][
                (directory / "evidence.tar.gz").relative_to(ROOT).as_posix()]:
            raise ValueError("Full validation archive changed")


def worker(spec, directory):
    started = time.monotonic()
    receipt = {"status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "source_files": {}, "requests_completed": 0}
    write_json(directory / "work.json", receipt)
    cache = None
    try:
        require_prior(spec)
        for group in ("sources", "prerequisites"):
            for name, expected in spec[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Registered {group} changed: {name}")
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

        def guard(force=False):
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Preserve partial request evidence")
            paced(force=True)
            if time.monotonic() - started > spec["soft_deadline_seconds"]:
                raise TimeoutError("Deadline reached while pacing")

        common.thermal_guard = guard
        from eval_arms import PredictC
        from qwen_graph_requests import QwenGraphRequests
        from train_c import kernel_packages
        if kernel_packages() != spec["kernel_packages"]:
            raise ValueError("Native kernel availability changed")
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Base weights changed")
        guard()
        reference = PredictC("c-qwen08-30k")
        if reference.weights != {"adapter": spec["adapter_sha256"]} or not reference.model.config.use_cache:
            raise ValueError("Original reference weights or cache setting changed")
        if torch.get_num_threads() != 4 or list(torch.cuda.get_device_capability()) != [12, 0]:
            raise ValueError("Require the registered laptop and four CPU threads")
        audits = [audit_request(reference.tok, request, spec["buckets"]) for request in schedule()]
        if audits != json.loads(AUDIT.read_text(encoding="utf-8")):
            raise ValueError("Full request input audit changed")
        torch.cuda.reset_peak_memory_stats()
        cache = QwenGraphRequests(reference, spec["buckets"], spec["allowed_shapes"], guard,
                                  spec["minimum_free_cuda_bytes"], spec["maximum_allocated_cuda_bytes"])
        receipt.update(status="running", own_pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                       capability=list(torch.cuda.get_device_capability()), threads=torch.get_num_threads(),
                       packages=spec["packages"], kernel_packages=kernel_packages(), cuda=torch.version.cuda,
                       weights=reference.weights, base_sha256=spec["base_sha256"], attention_implementation="sdpa",
                       reference_cache=True, graph_cache=False, lora_merged=False, projected_letters=26,
                       base_dtype=str(reference.model.get_base_model().model.embed_tokens.weight.dtype),
                       adapter_dtypes=sorted({str(p.dtype) for name, p in reference.model.named_parameters() if "lora_" in name}),
                       autocast_dtype="torch.bfloat16", input_audit_sha256=digest(AUDIT))
        write_json(directory / "work.json", receipt)
        pairs, replay_pairs = {str(n): [] for n in (1, 5, 10, 50)}, {str(n): [] for n in (1, 5, 10, 50)}
        retained = []
        with (directory / "requests.jsonl").open("x", encoding="utf-8") as output:
            for request, audit in zip(schedule(), audits, strict=True):
                guard()
                cases = request_cases(request)
                full = [v.tolist() for v in reference.groups([cases])[0]]
                result = cache.request(cases, check_eager=True)
                actual = [v.tolist() for v in result["log_probabilities"]]
                eager = [v.tolist() for v in result["uncaptured_log_probabilities"]]
                retained.extend(zip(result["log_probabilities"], actual, strict=True))
                for case, p, q, e in zip(cases, full, actual, eager, strict=True):
                    k = len(case["options"])
                    pairs[str(request["n_questions"])].append((normalized(p, k), normalized(q, k)))
                    replay_pairs[str(request["n_questions"])].append((normalized(e, k), normalized(q, k)))
                row = {**request, "audit_sha256": json_sha(audit), "reference_log_probabilities": full,
                       "graph_log_probabilities": actual, "uncaptured_log_probabilities": eager,
                       "batches": result["batches"], "resident_shapes": result["resident_shapes"],
                       "captures_total": result["captures_total"]}
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                receipt["requests_completed"] += 1
                receipt["captures"] = cache.captures
                write_json(directory / "work.json", receipt)
        cache.close()
        if any(vector.tolist() != saved for vector, saved in retained):
            raise ValueError("A later replay or graph release overwrote a retained CPU result")
        measured = {n: metrics(values, spec) for n, values in pairs.items()}
        replays = {n: metrics(values, spec) for n, values in replay_pairs.items()}
        passed = all(v["passed"] for v in measured.values()) and all(
            v["max_probability_delta"] <= spec["max_graph_eager_delta"] for v in replays.values())
        receipt.update(status="complete", passed=passed, reference_parity_by_request_size=measured,
                       graph_uncaptured_by_request_size=replays, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       output_lifetime_checks=len(retained),
                       requests_sha256=digest(directory / "requests.jsonl"),
                       scope="Probability and sequential multi-graph lifetime check only. No latency, quality gain or runtime promotion.")
        return 0 if passed else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if cache is not None:
            cache.close()
        if "common" in locals():
            receipt["thermal_governor"] = common.THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(directory / "work.json", receipt)


def driver():
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    require_prior(spec)
    directory = DIRECTORY / "validation"
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
    receipt.update(status="complete" if code in (0, 2) else "failed", returncode=code,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "execution.json", receipt)
    print(json.dumps(receipt), flush=True)
    return code


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--worker"]):
        raise ValueError("Only the registered request probe is supported")
    raise SystemExit(worker(json.loads(REGISTRATION.read_text(encoding="utf-8")), DIRECTORY / "validation")
                     if "--worker" in sys.argv else driver())
