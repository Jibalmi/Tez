"""Prepare a deterministic training-only continuation schedule without loading model weights."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/pilot-qwen-continuation-20260927"
RUNS = ROOT / "runs/pilot-qwen-continuation-20260927"
TEACHER = ROOT / "runs/pilot-cloud-h200-20260927"


def main():
    if (RESULTS / "preparation.json").exists() or (RUNS / "schedule.json").exists():
        raise ValueError("Preserve existing preparation")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    import dataset
    from train_c import prompt_for
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    manifest_path = TEACHER / "manifest.json"
    if digest(manifest_path) != "8e3f7cf23f7b14c80203eca42eab9bb7e7108f8c7aab67abd9d9b136f7cfb807":
        raise ValueError("Frozen teacher manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(TEACHER / name) != expected:
            raise ValueError("Teacher file changed")
    dataset.RUNS = TEACHER
    train, _, validation = dataset.load_examples()
    if ({e.id for e in train} & {e.id for e in validation}
            or (len(train), sum(len(e.decisions) for e in train), len(validation), sum(len(e.decisions) for e in validation))
            != (5998, 29990, 278, 1390)):
        raise ValueError("Training/validation membership changed")
    rng = random.Random(113)
    candidates = []
    for example in train:
        for decision in example.decisions:
            order = list(range(len(decision.options)))
            rng.shuffle(order)
            prompt = prompt_for(decision.type, decision.instructions, [decision.options[i] for i in order], example.state)
            candidates.append({"state_id": example.id, "qid": decision.qid, "permutation": order,
                               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                               "prompt_chars": len(prompt), "qtype": decision.type, "source": example.source, "prompt": prompt})
    candidates.sort(key=lambda row: row["prompt_chars"])
    batches = [candidates[start:start + 16] for start in range(0, len(candidates), 16)]
    rng.shuffle(batches)
    selected = [batch for batch in batches if len(batch) == 16][:150]
    if len(selected) != 150:
        raise ValueError("Insufficient full training batches")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                                             local_files_only=True)
    lengths, types, sources = [], Counter(), Counter()
    for batch in selected:
        for row in batch:
            count = len(tokenizer.encode(row.pop("prompt"), add_special_tokens=False, truncation=False))
            row["encoded_tokens"] = count
            lengths.append(count)
            types[row["qtype"]] += 1
            sources[row["source"]] += 1
    validation_lengths = [len(tokenizer.encode(prompt_for(d.type, d.instructions, d.options, e.state),
                                               add_special_tokens=False, truncation=False)) for e in validation for d in e.decisions]
    if max(lengths + validation_lengths) > 2048:
        raise ValueError("The complete continuation inputs exceed the registered 2048-token safety bound")
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    schedule = RUNS / "schedule.json"
    write_json(schedule, {"seed": 113, "steps": 150, "effective_batch": 16, "batches": selected})
    paths = ["experiments/small_model/" + name for name in
             ("prepare_qwen_continuation.py", "train_c.py", "dataset.py", "common.py", "structure_probe.py", "train_a.py", "student.py")]
    paths += ["tez/prompt.py", "tez/schema.py", "tez/errors.py"]
    receipt = {"status": "complete", "prepared_utc": datetime.now(timezone.utc).isoformat(),
               "teacher_manifest_sha256": digest(manifest_path), "teacher_files_verified": len(manifest["files"]),
               "schedule_sha256": digest(schedule), "schedule": schedule.relative_to(ROOT).as_posix(),
               "training_decisions": len(lengths), "training_states": len({r["state_id"] for b in selected for r in b}),
               "training_types": dict(types), "training_sources": dict(sources),
               "training_tokens": sum(lengths), "max_training_tokens": max(lengths), "max_validation_tokens": max(validation_lengths),
               "context_bound": 2048, "training_over_original_1536": sum(n > 1536 for n in lengths),
               "validation_over_original_1536": sum(n > 1536 for n in validation_lengths),
               "validation_states": len(validation), "validation_decisions": len(validation_lengths), "torch_threads": torch.get_num_threads(),
               "sources": {name: digest(ROOT / name) for name in paths},
               "scope": "Training schedule only. All selected decisions belong to the original frozen training split; no validation or public-test labels select training examples. Seed and complete batches chosen before any continuation. No new teacher labels, training, inference or GPU work performed."}
    write_json(RESULTS / "preparation.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
