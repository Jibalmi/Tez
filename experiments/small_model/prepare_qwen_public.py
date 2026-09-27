"""Freeze matched public inputs and analysis before any trained-checkpoint public evaluation."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  TOKENIZERS_PARALLELISM="false")

from qwen_public_evaluation import (  # noqa: E402
    AUDIT, CONTINUATION, DIRECTORY, PREPARATION, REGISTRATION, RELOAD, ROOT, SUITE, audit_case, task_phase,
)
from structure_probe import digest, write_json  # noqa: E402


def main():
    if REGISTRATION.exists() or PREPARATION.exists() or AUDIT.exists() or DIRECTORY.exists():
        raise ValueError("Preserve public protocol and preparation")
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    reload_spec = json.loads(RELOAD.read_text(encoding="utf-8"))
    sources = dict(reload_spec["sources"])
    for name in ("qwen_public_evaluation.py", "report_qwen_public.py", "prepare_qwen_public.py", "test_qwen_public.py",
                 "competitor_overlap.py", "compare_round1.py", "test_compare_round1.py", "report_gliner2.py",
                 "eval_gliner2.py", "gliner2_adapter.py"):
        path = Path(__file__).with_name(name)
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    for relative in ("data/semif/evaluate.py", "data/semif/authored144.jsonl", "data/semif/perturbations108.jsonl"):
        sources[relative] = digest(ROOT / relative)
    if any(digest(ROOT / name) != expected for name, expected in sources.items()):
        raise ValueError("A frozen source changed")
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    if digest(SUITE) != "fea63e7530b2166d51b2b6b8f4adfd8994c75db48fcbaf6cca13d339f959def5":
        raise ValueError("Frozen public membership changed")
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from train_c import kernel_packages
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=continuation["base_revision"], local_files_only=True)
    base = Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=continuation["base_revision"], local_files_only=True))
    audit, counts, lengths = {}, Counter(), []
    for task, cases in suite.items():
        if len({c["id"] for c in cases}) != len(cases):
            raise ValueError("Duplicate public case")
        audit[task] = {}
        for reverse in ((False, True) if task.startswith("massive_") else (False,)):
            order = "reversed" if reverse else "primary"
            audit[task][order] = {c["id"]: audit_case(tokenizer, c, reverse) for c in cases}
            counts[f"{task_phase(task)}_{order}"] += len(cases)
            lengths.extend(v["encoded_tokens"] for v in audit[task][order].values())
    if len(suite) != 21 or len(lengths) != 7251 or Counter(Counter(c["group"] for c in suite["typed_decisions"]).values()) != {5: 400}:
        raise ValueError("Unexpected public counts or typed-state clusters")
    for cases in suite.values():
        for c in cases:
            if not 2 <= len(c["options"]) <= 26 or not 0 <= c["gold"] < len(c["options"]):
                raise ValueError("Unexpected option count or public label")
    write_json(AUDIT, audit)
    preflight = {"status": "complete", "cpu_threads": 1, "gpu_model_loaded": False, "tasks": len(suite),
                 "primary_decisions": 6151, "reversed_decisions": 1100, "counts": dict(counts),
                 "max_input_tokens": max(lengths), "total_input_tokens_including_reversed": sum(lengths),
                 "context_bound": 4096, "truncated_decisions": 0, "audit_sha256": digest(AUDIT),
                 "suite_sha256": digest(SUITE), "scope": "Complete token and prompt audit only; no public inference or scores."}
    write_json(PREPARATION, preflight)
    prerequisites = {p.relative_to(ROOT).as_posix(): digest(p) for p in (SUITE, AUDIT, PREPARATION, CONTINUATION, RELOAD)}
    prerequisites.update({name: expected for name, expected in continuation["prerequisites"].items()
                          if "/models/c-qwen08-30k/" in name})
    references = {}
    locations = {"qwen_h200": ("runs/student-artifacts-h200-20260927/results/pilot-student-h200-20260927/rows", "_tez-s-c-qwen08-30k"),
                 "decider_h200": ("runs/improvement-artifacts-h200-20260927/results/pilot-improvement-20260927/rows", "_tez-s-decider08-native"),
                 "tev1_laptop": ("results/pilot-tev1-eval-20260927", "")}
    for name, (folder, suffix) in locations.items():
        references[name] = {task: f"{folder}/rows_{task}{suffix}.jsonl" for task in suite}
        prerequisites.update({relative: digest(ROOT / relative) for relative in references[name].values()})
    baseline = (ROOT / continuation["adapter_directory"]).parent
    spec = {"status": "registered_before_selected_checkpoint_public_evaluation", "registered_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources, "prerequisites": prerequisites, "packages": continuation["packages"],
            "kernel_packages": kernel_packages(), "base_revision": continuation["base_revision"],
            "base_sha256": continuation["base_sha256"],
            "base_metadata": {p.name: digest(p) for p in base.iterdir() if p.is_file() and p.suffix != ".safetensors"},
            "baseline_model": baseline.relative_to(ROOT).as_posix(),
            "baseline_adapter_sha256": digest(baseline / "adapter/adapter_model.safetensors"),
            "reference_rows": references, "soft_deadline_seconds": 1080, "worker_timeout_seconds": 1200,
            "selection": "Require all four original validations, the frozen validation-only selection, exported adapter and independently verified strict reload parity before either arm. Step0 remains eligible. Dynamic selected-weight bindings are frozen in each phase before loading the model and must agree across all eight phases.",
            "inference": "Unchanged PredictC on both arms, four separate typed/apps/intent/semif phases per model. Native SDPA/PyTorch recurrence, original full vocabulary head, unmerged float32 LoRA over BF16 base, cache on, eval mode, BF16 autocast, batch16, stable prompt-character-length sorting, question-first prompts, complete inputs<=4096. All probabilities T1; no calibration. Typed groups preserve existing input order but each decision has its own prompt, not packed multi-question attention. All1100 MASSIVE cases also use reversed options; retain presented log probabilities and canonical probabilities with original labels.",
            "analysis": "Independently rebuild full prompt/token audit, validate every primary/reversed row and recompute raw T1 metrics. Include semantic-label F1 for apps/intent, official SemIf family-balanced scores, ECE15, NLL, Brier, typed soft accuracy and per-workflow/type metrics. Compare typed accuracy with paired whole-state percentile bootstrap,2000 resamples,seed13; no multiplicity correction. No aggregate mixed-task win score or public-result-driven checkpoint selection.",
            "reference_scope": "baseline and selected are fresh matched RTX5080 Laptop/native Torch2.10 results. qwen_h200 and decider_h200 retain their earlier cloud runtime; tev1_laptop retains its separately measured native adapter. Historical references contextualize quality; only selected versus baseline isolates this continuation under the same hardware/backend. Training exposure differs and is unresolved for external models.",
            "compute": "Each of eight independent phases uses shared run_heavy, four threads,22-minute lease,20-minute hard child limit,18-minute soft limit checked before/after every80/76C thermal pause; wrapper84C. Completed tasks survive failures. No paid resources, no timing gate or production promotion.",
            "scope": "Exploratory public quality evaluation after fixed validation-only selection. These public benchmarks have already informed research direction. No public fitting, temperature selection, release-threshold change or valid latency claim. If step0 wins, report no selected continuation gain. All original quality and latency gates remain in force."}
    for name, expected in prerequisites.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Public prerequisite changed during preparation")
    write_json(REGISTRATION, spec)
    print(json.dumps({"registration_sha256": digest(REGISTRATION), "preflight": preflight}, indent=2))


if __name__ == "__main__":
    main()
