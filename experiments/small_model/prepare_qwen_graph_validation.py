"""One-thread CPU input audit and pre-execution registration for the graph parity trial."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", CUDA_VISIBLE_DEVICES="")

from structure_probe import digest, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    from transformers import AutoTokenizer
    from graph_qwen import batch_plan
    from parity_evidence import read_rows
    from qwen_graph_validation import DIRECTORY, REGISTRATION, validation_inputs
    from train_c import kernel_packages
    if REGISTRATION.exists() or (DIRECTORY / "preflight.json").exists():
        raise ValueError("Preserve the registered plan")
    prior = json.loads(Path(__file__).with_name("qwen_cuda_graph_attempt2_registration.json").read_text(encoding="utf-8"))
    continuation = json.loads(Path(__file__).with_name("qwen_continuation_registration.json").read_text(encoding="utf-8"))
    names = [name for name in prior["sources"] if not name.endswith(("qwen_cuda_graph_probe.py", "qwen_cuda_graph_attempt2.py"))]
    names += ["experiments/small_model/" + name for name in (
        "graph_qwen.py", "test_graph_qwen.py", "qwen_graph_validation.py", "verify_qwen_graph_validation.py",
        "prepare_qwen_graph_validation.py", "test_verify_qwen_graph_validation.py", "parity_evidence.py", "test_parity_evidence.py")]
    names += [".venv-pilot/Lib/site-packages/" + name for name in (
        "peft/peft_model.py", "peft/tuners/lora/layer.py", "torch/cuda/graphs.py", "transformers/integrations/hub_kernels.py")]
    prerequisite_names = list(prior["prerequisites"])
    prerequisite_names += ["runs/pilot-cloud-h200-20260927/manifest.json",
                          "results/pilot-restricted-qwen-20260927/reference_rows.jsonl",
                          "results/pilot-qwen-cuda-graph-attempt2-20260927/analysis.json",
                          "results/pilot-qwen-cuda-graph-attempt2-20260927/artifact_receipt.json"]
    spec = {key: continuation[key] for key in ("base_revision", "base_sha256", "teacher_directory", "packages",
                                               "reference_rows", "argmax_agreement_min", "max_probability_delta_max",
                                               "mean_total_variation_max")}
    spec.update(status="registered_before_gpu_execution", registered_utc=datetime.now(timezone.utc).isoformat(),
                kernel_packages=kernel_packages(),
                model_directory=prior["model_directory"], adapter_sha256=continuation["prerequisites"][
                    continuation["adapter_directory"] + "/adapter_model.safetensors"],
                sources={name: digest(ROOT / name) for name in names},
                prerequisites={name: digest(ROOT / name) for name in prerequisite_names},
                buckets=[128, 256, 512, 1024, 2048, 4096], batch_sizes=[1, 16],
                soft_deadline_seconds=720, worker_timeout_seconds=840, max_graph_eager_delta=1e-5,
                minimum_free_cuda_bytes=4 * 2**30, maximum_allocated_cuda_bytes=12 * 2**30,
                hypothesis="Reusing fixed-shape CUDA graphs may remove Python/kernel dispatch overhead. This experiment establishes numerical parity only.",
                changes="Original unmerged LoRA, BF16 base, SDPA and native recurrence; cache off, restricted 26-letter projection. Hub kernels explicitly disabled and optional kernel packages pinned to their recorded availability. Left-pad complete inputs to the first strictly larger bucket. Native masks built outside capture. Explicit default-equivalent positions. Three side-stream warmups. One graph active at a time, reused for all batches of its length; no shared memory pools. Duplicate the final real row only to fill a batch; exclude dummy outputs. Compare every real graph output with both the frozen reference and an identical uncaptured forward.",
                acceptance="Each batch-size arm independently needs all 1390 decisions, original argmax >= .99, max delta <= .04, mean TV <= .005, and graph-vs-identical-eager maximum delta <= .00001. Keep failures. Never relax limits. Neither arm's success substitutes for the other arm.",
                compute="Laptop run_heavy lock, four threads, fifteen-minute lease, fourteen-minute hard worker timeout, twelve-minute soft deadline. Process-local 80/76 C pacing, wrapper 84 C. No cloud. Each arm releases the lock separately.",
                scope="Validation inputs only, no weight, temperature or label fitting. No changed training. A pass does not establish latency, variable-shape graph-cache residency, benchmark accuracy or runtime readiness. Any later timing must include preprocessing, native masks, copies, output ownership, and capture for cache misses.",
                references=["https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs"])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True)
    cases = validation_inputs(tokenizer, spec)
    read_rows(ROOT / spec["reference_rows"], [(c["state_id"], c["qid"], c["k"]) for c in cases])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json(DIRECTORY / "inputs.json", cases)
    spec["input_sha256"] = digest(DIRECTORY / "inputs.json")
    lengths = [c["encoded_tokens"] for c in cases]
    spec["plans"] = {str(batch): batch_plan(lengths, batch, spec["buckets"]) for batch in spec["batch_sizes"]}
    preflight = {"status": "complete", "cpu_threads": 1, "gpu_model_loaded": False,
                 "states": len({c["state_id"] for c in cases}), "decisions": len(cases),
                 "max_input_tokens": max(lengths), "total_input_tokens": sum(lengths), "input_sha256": spec["input_sha256"],
                 "plans": {str(batch): {"batches": len(spec["plans"][str(batch)]),
                             "batches_by_length": dict(Counter(b["length"] for b in spec["plans"][str(batch)])),
                             "dummy_rows": sum(b["dummy_rows"] for b in spec["plans"][str(batch)]),
                             "real_decisions": sum(len(b["indices"]) for b in spec["plans"][str(batch)])}
                           for batch in spec["batch_sizes"]}}
    write_json(DIRECTORY / "preflight.json", preflight)
    spec["prerequisites"][(DIRECTORY / "preflight.json").relative_to(ROOT).as_posix()] = digest(DIRECTORY / "preflight.json")
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "preflight": preflight}, indent=2))


if __name__ == "__main__":
    main()
