"""One-thread full-input prefix preservation audit before any candidate GPU execution."""
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
    import torch
    from transformers import AutoTokenizer
    from parity_evidence import read_rows
    from qwen_prefix_graph import native_batch_plan, pack_native_prefix
    from qwen_prefix_validation import DIRECTORY, REGISTRATION, validation_inputs
    torch.set_num_threads(1)
    if REGISTRATION.exists() or (DIRECTORY / "preflight.json").exists():
        raise ValueError("Preserve the prefix protocol and preflight")
    previous = Path(__file__).with_name("qwen_graph_validation_registration.json")
    old = json.loads(previous.read_text(encoding="utf-8"))
    spec = {key: old[key] for key in ("base_revision", "base_sha256", "teacher_directory", "packages", "kernel_packages",
                                     "reference_rows", "model_directory", "adapter_sha256", "buckets", "batch_sizes",
                                     "argmax_agreement_min", "max_probability_delta_max", "mean_total_variation_max",
                                     "max_graph_eager_delta", "minimum_free_cuda_bytes", "maximum_allocated_cuda_bytes",
                                     "soft_deadline_seconds", "worker_timeout_seconds", "references")}
    sources = dict(old["sources"])
    for name in ("qwen_prefix_graph.py", "test_qwen_prefix_graph.py", "qwen_prefix_validation.py",
                 "verify_qwen_prefix_validation.py", "test_verify_qwen_prefix_validation.py", "prepare_qwen_prefix_validation.py"):
        path = Path(__file__).with_name(name)
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    prerequisites = dict(old["prerequisites"])
    for name in ("verification.json", "artifact_receipt.json"):
        path = ROOT / "results/pilot-qwen-graph-validation-20260927/batch-16" / name
        prerequisites[path.relative_to(ROOT).as_posix()] = digest(path)
    for name, expected in {**sources, **prerequisites}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("A frozen prerequisite changed")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"], local_files_only=True)
    cases = validation_inputs(tokenizer, spec)
    read_rows(ROOT / spec["reference_rows"], [(c["state_id"], c["qid"], c["k"]) for c in cases])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_json(DIRECTORY / "inputs.json", cases)
    if digest(DIRECTORY / "inputs.json") != old["input_sha256"]:
        raise ValueError("The original validation inputs changed")
    plans = {str(batch): native_batch_plan(cases, batch, spec["buckets"]) for batch in spec["batch_sizes"]}
    summary = {}
    for batch_size in spec["batch_sizes"]:
        plan = plans[str(batch_size)]
        visited = []
        for batch in plan:
            selected = [cases[i] for i in batch["indices"]]
            native = tokenizer([c["prompt"] for c in selected], add_special_tokens=False, truncation=False,
                               padding=True, padding_side="left", return_tensors="pt")
            ids, padding, _, end = pack_native_prefix([c["token_ids"] for c in selected], [c["k"] for c in selected],
                                                      batch["batch_size"], batch["length"], tokenizer.pad_token_id)
            length = batch["native_length"]
            if (native["input_ids"].shape != (batch["batch_size"], length)
                    or not torch.equal(ids[:, :length], native["input_ids"])
                    or not torch.equal(padding[:, :length], native["attention_mask"])
                    or padding[:, length:].any() or end.tolist() != [length - 1]):
                raise ValueError("The complete original left-padded batch or its final position changed")
            visited.extend(batch["indices"])
        if sorted(visited) != list(range(1390)):
            raise ValueError("Each real decision must appear exactly once")
        summary[str(batch_size)] = {"batches": len(plan), "real_decisions": len(visited), "dummy_rows": 0,
                                    "batches_by_shape": dict(Counter(f"{b['batch_size']}x{b['length']}" for b in plan)),
                                    "native_prefix_ids_and_masks_equal": True, "native_final_positions_equal": True}
    preflight = {"status": "complete", "cpu_threads": 1, "gpu_model_loaded": False, "states": 278, "decisions": 1390,
                 "max_input_tokens": max(c["encoded_tokens"] for c in cases),
                 "input_sha256": digest(DIRECTORY / "inputs.json"), "plans": summary,
                 "scope": "Real tokenizer and input-position equality only; no Qwen inference or probability parity established."}
    write_json(DIRECTORY / "preflight.json", preflight)
    prerequisites[(DIRECTORY / "preflight.json").relative_to(ROOT).as_posix()] = digest(DIRECTORY / "preflight.json")
    spec.update(status="registered_after_failed_left_padding_candidate_before_execution",
                registered_utc=datetime.now(timezone.utc).isoformat(), sources=sources, prerequisites=prerequisites,
                input_sha256=preflight["input_sha256"], plans=plans,
                hypothesis="Preserving native batch composition and prefix positions may avoid the drift observed after regrouping and adding leading padding. Added trailing padding may still change floating-point kernels; this is unproven until full validation.",
                changes="Sort prompts by character length and form native batches at the requested batch size. Preserve each batch including a short final batch with no dummy rows. Group whole batches by (actual batch size, strictly larger token bucket) for capture reuse. Keep the native left-padded batch as an unchanged prefix, append masked trailing tokens, use default-equivalent positions, and index the native last real-token column through a static CUDA index buffer. All input/mask/option/end buffers update before replay. Original unmerged adapter, base, SDPA/recurrence, restricted letter head and cache-off path remain frozen; Hub kernels disabled. One graph resident; three guarded side-stream warmups per shape. Compare every graph output with an identical uncaptured forward.",
                reference_scope="The same frozen full-head original validation at batch16 remains the reference for both arms. The batch1 comparison includes the effect of changing the original validation batch size. No reference or probability limit is changed to obtain a pass.",
                acceptance=old["acceptance"], compute=old["compute"],
                scope="A new inference candidate following a retained failed graph experiment. Complete1390-decision parity independently required per arm; no public labels, fitting, temperature change, training change or timing. Passing one arm does not substitute for the other. Capture and shape-cache misses must be included or explicitly separated in any later latency protocol; no runtime promotion here.")
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "preflight": preflight}, indent=2))


if __name__ == "__main__":
    main()
