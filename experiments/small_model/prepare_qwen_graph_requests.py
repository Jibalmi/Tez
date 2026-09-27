"""Freeze all fresh-state request shapes before testing a sequential multi-graph cache."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  TOKENIZERS_PARALLELISM="false")

from qwen_graph_request_probe import AUDIT, DIRECTORY, PARENT, PRIOR, REGISTRATION, ROOT, audit_request, schedule  # noqa: E402
from structure_probe import digest, write_json  # noqa: E402


def main(preflight_only=False):
    if REGISTRATION.exists():
        raise ValueError("Preserve existing registration")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    sources = dict(parent["sources"])
    for name in ("qwen_graph_requests.py", "test_qwen_graph_requests.py", "qwen_graph_request_probe.py",
                 "prepare_qwen_graph_requests.py", "verify_qwen_graph_requests.py", "test_verify_qwen_graph_requests.py",
                 "report_qwen_public.py", "qwen_public_evaluation.py"):
        path = Path(__file__).with_name(name)
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    sources["experiments/vs_laya_speed.py"] = digest(ROOT / "experiments/vs_laya_speed.py")
    if any(digest(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("Frozen source changed")
    prerequisites = {PARENT.relative_to(ROOT).as_posix(): digest(PARENT)}
    prerequisites.update({name: expected for name, expected in parent["prerequisites"].items()
                          if "/models/c-qwen08-30k/" in name})
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=parent["base_revision"], local_files_only=True)
    audits = [audit_request(tokenizer, request, parent["buckets"]) for request in schedule()]
    counts = Counter((r["n_questions"], r["phase"]) for r in audits)
    shapes = sorted({(b["batch_size"], b["length"]) for r in audits for b in r["batches"]})
    lengths = [c["encoded_tokens"] for r in audits for c in r["cases"]]
    if len(audits) != 370 or len(lengths) != 3760 or max(lengths) >= 4096:
        raise ValueError("Unexpected workload or truncation")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    if AUDIT.exists():
        if json.loads(AUDIT.read_text(encoding="utf-8")) != audits:
            raise ValueError("Saved request input audit differs")
    else:
        write_json(AUDIT, audits)
    preflight = {"status": "complete", "cpu_threads": 1, "gpu_model_loaded": False, "requests": len(audits),
                 "decisions": len(lengths), "counts": {f"{n}_{phase}": v for (n, phase), v in counts.items()},
                 "allowed_shapes": [list(shape) for shape in shapes], "max_input_tokens": max(lengths),
                 "truncated_decisions": 0, "input_audit_sha256": digest(AUDIT), "source_hashes": sources}
    if (DIRECTORY / "preflight.json").exists():
        if json.loads((DIRECTORY / "preflight.json").read_text(encoding="utf-8")) != preflight:
            raise ValueError("Saved request preflight differs")
    else:
        write_json(DIRECTORY / "preflight.json", preflight)
    if preflight_only:
        print(json.dumps({"status": "preflight_only", "registration_created": False, "preflight": preflight}, indent=2))
        return
    for directory in PRIOR:
        result = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
        if result["passed"] is not True or result["status"] != "complete" or result["real_predictions_per_arm"] != 1390:
            raise ValueError("Both original full-validation graph arms must pass before cache registration")
        for name in ("verification.json", "artifact_receipt.json", "evidence.tar.gz", "registration.json"):
            path = directory / name
            prerequisites[path.relative_to(ROOT).as_posix()] = digest(path)
        receipt = json.loads((directory / "artifact_receipt.json").read_text(encoding="utf-8"))
        if digest(directory / "evidence.tar.gz") != receipt["sha256"]:
            raise ValueError("Prior validation archive differs")
    for path in (AUDIT, DIRECTORY / "preflight.json"):
        prerequisites[path.relative_to(ROOT).as_posix()] = digest(path)
    spec = {key: parent[key] for key in ("base_revision", "base_sha256", "packages", "kernel_packages", "model_directory",
                                        "adapter_sha256", "buckets", "argmax_agreement_min", "max_probability_delta_max",
                                        "mean_total_variation_max", "max_graph_eager_delta", "minimum_free_cuda_bytes",
                                        "maximum_allocated_cuda_bytes")}
    spec.update(status="registered_before_multi_graph_request_execution", registered_utc=datetime.now(timezone.utc).isoformat(),
                sources=sources, prerequisites=prerequisites, allowed_shapes=preflight["allowed_shapes"],
                soft_deadline_seconds=1080, worker_timeout_seconds=1200,
                inference="Original unmerged BF16-base/FP32-LoRA weights, native SDPA/recurrence, 26-letter graph head, explicit cache off. Reference remains full-head/cache-on native B16. Fresh request prompts and tokenization every call. Preserve native character-length ordering and real batch sizes, including B2 at the end of 50-question calls. Each predeclared shape owns a separate static buffer and private graph pool; sequential reuse, no shared pools, no dummy rows, no evictions. Three side-stream warmups per capture. CPU-owned copies preserve output lifetime.",
                analysis="All370 requests and3760 decisions per arm, including the planned warmup calls. Independently reconstruct input hashes, native batch order, persistent shape/replay history and every probability. Require unchanged full-reference parity limits separately for each of1/5/10/50 questions; captured-versus-identical-eager max delta<=1e-5 for each size. Preserve partial/failing evidence, no post-result threshold or shape changes.",
                compute="Shared run_heavy, four CPU threads,22-minute lease,20-minute child timeout,18-minute soft deadline,80/76C pacing and84C wrapper. No cloud compute. Operational capture duration may include thermal pauses and is not latency.",
                scope="Correctness prerequisite for a future fresh-state timing protocol, not a timing measurement, release gate, public accuracy result or runtime promotion. The single-graph full-validation evidence cannot substitute for this new cache check.")
    for name, expected in prerequisites.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Prerequisite changed during preparation")
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "preflight": preflight}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Audit inputs only; do not create a GPU registration")
    main(parser.parse_args().preflight)
