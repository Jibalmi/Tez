"""Full validation of CUDA graphs that retain original batch composition and prefix positions."""
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
import time

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-prefix-validation-20260927"
REGISTRATION = Path(__file__).with_name("qwen_prefix_validation_registration.json")


def validation_inputs(tokenizer, spec):
    import dataset
    from train_c import prompt_for
    data = ROOT / spec["teacher_directory"]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(data / name) != expected:
            raise ValueError("Original teacher data changed")
    dataset.RUNS = data
    train, _, examples = dataset.load_examples()
    if len(examples) != 278 or {e.id for e in train} & {e.id for e in examples}:
        raise ValueError("Validation split changed")
    cases = []
    for example in examples:
        for decision in example.decisions:
            prompt = prompt_for(decision.type, decision.instructions, decision.options, example.state)
            tokens = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
            cases.append({"state_id": example.id, "qid": decision.qid, "qtype": decision.type,
                          "k": len(decision.options), "prompt": prompt, "token_ids": tokens,
                          "encoded_tokens": len(tokens), "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    if len(cases) != 1390 or len({(c["state_id"], c["qid"]) for c in cases}) != 1390:
        raise ValueError("Incomplete or duplicate validation membership")
    return cases


def probability_row(case, values, batch_index, length):
    if (len(values) != 26 or any(not math.isfinite(v) for v in values)
            or any(v > -9000 for v in values[case["k"]:])):
        raise ValueError("Invalid graph readout or unmasked tail")
    logits = values[:case["k"]]
    probabilities = [math.exp(x - max(logits)) for x in logits]
    denominator = sum(probabilities)
    probabilities = [x / denominator for x in probabilities]
    return {**{key: case[key] for key in ("state_id", "qid", "qtype", "k", "prompt_sha256", "encoded_tokens")},
            "probabilities": probabilities, "log_probabilities": values, "batch_index": batch_index, "padded_tokens": length}


def worker(batch_size, directory, spec):
    path = directory / "work.json"
    receipt = {"status": "preparing", "registration_sha256": digest(REGISTRATION),
               "started_utc": datetime.now(timezone.utc).isoformat(), "batch_size": batch_size,
               "source_files": {}, "batches": [], "captures": []}
    started = time.monotonic()
    graph = None
    write_json(path, receipt)
    try:
        for name, expected in spec["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Registered graph source changed")
            target = directory / "executed_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            receipt["source_files"][name] = expected
        for name, expected in spec["prerequisites"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Frozen graph prerequisite changed")
        if any(version(name) != expected for name, expected in spec["packages"].items()):
            raise ValueError("Registered package versions changed")
        import torch
        from huggingface_hub import snapshot_download
        import common
        from functools import partial
        common.thermal_guard = partial(common.thermal_guard, hot=80, cool=76, every=0.0)
        from common import THERMAL, thermal_guard
        from eval_arms import PredictC
        from qwen_prefix_graph import NativePrefixGraph, native_batch_plan
        from parity_evidence import compare_rows, read_rows
        from restricted_qwen import RestrictedQwen
        from train_c import kernel_packages
        if kernel_packages() != spec["kernel_packages"]:
            raise ValueError("Native kernel package availability changed")
        base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True))
        if [digest(p) for p in base.glob("*.safetensors")] != [spec["base_sha256"]]:
            raise ValueError("Frozen base weights changed")
        thermal_guard(force=True)
        reference = PredictC("c-qwen08-30k")
        if reference.weights != {"adapter": spec["adapter_sha256"]}:
            raise ValueError("Frozen adapter changed")
        candidate = RestrictedQwen(reference)
        if candidate.lm.config._attn_implementation != "sdpa":
            raise ValueError("Keep the registered native SDPA implementation")
        cases = validation_inputs(reference.tok, spec)
        write_json(directory / "inputs.json", cases)
        if digest(directory / "inputs.json") != spec["input_sha256"]:
            raise ValueError("Inputs differ from the CPU preflight")
        plan = native_batch_plan(cases, batch_size, spec["buckets"])
        if plan != spec["plans"][str(batch_size)]:
            raise ValueError("Batch membership changed")
        expected_ids = [(c["state_id"], c["qid"], c["k"]) for c in cases]
        frozen = read_rows(ROOT / spec["reference_rows"], expected_ids)
        receipt.update(status="running", gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                       threads=torch.get_num_threads(), packages={n: version(n) for n in spec["packages"]},
                       kernel_packages=kernel_packages(), weights=reference.weights, base_sha256=spec["base_sha256"],
                       attention_implementation=candidate.lm.config._attn_implementation,
                       cache=False, lora_merged=False, projected_letters=len(candidate.letters),
                       inputs_sha256=digest(directory / "inputs.json"), decisions=len(cases),
                       padded_input_policy="Preserve original character-sorted native batches and their left-padded prefix. Add masked tokens only after the native batch; read its last real-token column. No dummy rows.")
        write_json(path, receipt)
        torch.cuda.reset_peak_memory_stats()
        graph_path, eager_path = directory / "graph_rows.jsonl", directory / "uncaptured_rows.jsonl"
        with graph_path.open("x", encoding="utf-8") as graph_rows, eager_path.open("x", encoding="utf-8") as eager_rows:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for index, batch in enumerate(plan):
                    if time.monotonic() - started > spec["soft_deadline_seconds"]:
                        raise TimeoutError("Preserve partial rows; incomplete validation is ineligible")
                    selected = [cases[i] for i in batch["indices"]]
                    tokens, counts = [c["token_ids"] for c in selected], [c["k"] for c in selected]
                    if graph is None or (graph.batch_size, graph.length) != (batch["batch_size"], batch["length"]):
                        if graph is not None:
                            graph.close()
                            del graph
                            graph = None
                            torch.cuda.empty_cache()
                        thermal_guard(force=True)
                        free, total = torch.cuda.mem_get_info()
                        if free < spec["minimum_free_cuda_bytes"]:
                            raise MemoryError("Insufficient free CUDA memory for another registered shape")
                        graph = NativePrefixGraph(candidate, batch["batch_size"], batch["length"], tokens, counts, thermal_guard)
                        receipt["captures"].append({"shape": [batch["batch_size"], batch["length"]], "first_batch": index,
                                                    "warmup_forwards": 3, "free_cuda_bytes_before": free,
                                                    "total_cuda_bytes": total})
                    captured, uncaptured = graph.run(tokens, counts, thermal_guard, check_eager=True)
                    for output, values in ((graph_rows, captured), (eager_rows, uncaptured)):
                        for case, vector in zip(selected, values.double().cpu().tolist(), strict=True):
                            row = probability_row(case, vector, index, batch["length"])
                            row.update(original_batch=batch["original_batch"], native_length=batch["native_length"])
                            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                    del captured, uncaptured, values
                    peak = torch.cuda.max_memory_allocated()
                    receipt["batches"].append({**batch, "batch_index": index, "graph_replay": graph.replays,
                                                "peak_allocated_bytes": peak})
                    write_json(path, receipt)
                    if peak > spec["maximum_allocated_cuda_bytes"]:
                        raise MemoryError("Registered CUDA allocation bound exceeded")
        graph.close()
        graph = None
        captured = read_rows(graph_path, expected_ids)
        uncaptured = read_rows(eager_path, expected_ids)
        full_metrics = compare_rows(frozen, captured, spec)
        eager_metrics = compare_rows(uncaptured, captured, spec)
        passed = full_metrics["passed"] and eager_metrics["max_probability_delta"] <= spec["max_graph_eager_delta"]
        receipt.update(status="complete", passed=passed, full_reference_parity=full_metrics,
                       graph_uncaptured_parity=eager_metrics, graph_rows_sha256=digest(graph_path),
                       uncaptured_rows_sha256=digest(eager_path), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       scope="All1390 original validation decisions with unchanged native batch membership and prefix token positions, followed by trailing graph padding. No speed, trained quality or release claim.")
        return 0 if passed else 2
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if graph is not None:
            graph.close()
        if "THERMAL" in locals():
            receipt["thermal_governor"] = THERMAL
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, receipt)


def driver(batch_size):
    import psutil
    spec = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if batch_size not in spec["batch_sizes"]:
        raise ValueError("Unregistered batch size")
    directory = DIRECTORY / f"batch-{batch_size:02d}"
    directory.mkdir(exist_ok=False)
    shutil.copy2(REGISTRATION, directory / "registration.json")
    env = dict(os.environ)
    env.update(TEZ_PILOT_RUNS=str(ROOT / spec["model_directory"]), TEZ_PILOT_RESULTS=str(directory),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_HUB_KERNELS="0",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "registration_sha256": digest(REGISTRATION), "batch_size": batch_size,
               "timeout_seconds": spec["worker_timeout_seconds"]}
    write_json(directory / "execution.json", receipt)
    with (directory / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--batch-size", str(batch_size), "--worker"],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    if args.worker:
        raise SystemExit(worker(args.batch_size, DIRECTORY / f"batch-{args.batch_size:02d}",
                                json.loads(REGISTRATION.read_text(encoding="utf-8"))))
    raise SystemExit(driver(args.batch_size))
