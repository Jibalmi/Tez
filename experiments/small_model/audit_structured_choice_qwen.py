"""Audit eligible choice inputs for future Qwen supervision without model inference or confirmation access."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "RUN_HEAVY_THREADS"):
    os.environ[name] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0",
                  TOKENIZERS_PARALLELISM="false")

from structure_probe import digest, write_json  # noqa: E402
from structured_choice_data import independent_gold, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs/pilot-structured-choice-20260927"
FILTER = ROOT / "results/pilot-structured-choice-20260927"
PRIVATE = ROOT / "runs/pilot-structured-choice-qwen-preflight-20260927"
OUTPUT = ROOT / "results/pilot-structured-choice-qwen-preflight-20260927"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    if PRIVATE.exists() or OUTPUT.exists():
        raise ValueError("Preserve prior tokenizer audit")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    filtered = json.loads((FILTER / "filter_verification.json").read_text(encoding="utf-8"))
    filter_summary = json.loads((FILTER / "filter_summary.json").read_text(encoding="utf-8"))
    if (filtered["status"] != "complete" or filter_summary["status"] != "complete"
            or filtered["filtered_train_sha256"] != filter_summary["filtered_train_sha256"]
            or filtered["retained_states"] != 1200 or filtered["retained_decisions"] != 2400
            or digest(Path(__file__).with_name("structured_choice_data.py")) != manifest["generator_sha256"]):
        raise ValueError("Require completed unchanged filtering and exact-label generator")
    paths = {"train": DATA / "train.filtered.json", "validation": DATA / "validation.json"}
    expected = {"train": filtered["filtered_train_sha256"], "validation": manifest["splits"]["validation"]["sha256"]}
    if any(digest(path) != expected[split] for split, path in paths.items()):
        raise ValueError("Training or validation input changed")
    data = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in paths.items()}
    validate(data)
    if {split: (len(rows), sum(len(r["questions"]) for r in rows)) for split, rows in data.items()} != {
            "train": (1200, 2400), "validation": (200, 400)}:
        raise ValueError("Incomplete eligible training or validation membership")
    import torch
    from transformers import AutoTokenizer
    from train_c import prompt_for
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=REVISION, local_files_only=True)
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCDEFG"]
    if any(len(ids) != 1 for ids in letter_ids) or len({ids[0] for ids in letter_ids}) != 7:
        raise ValueError("Required option letters are not distinct single tokens")
    audit, summary = [], {}
    for split, records in data.items():
        lengths, operations, positions = [], Counter(), Counter()
        for record in records:
            targets = independent_gold(record)
            for question in record["questions"]:
                for order in ("primary", "reversed"):
                    options = question["options"] if order == "primary" else list(reversed(question["options"]))
                    keys = [key for key, _ in options]
                    gold = keys.index(targets[question["operation"]])
                    if keys[gold] != question["gold_key"]:
                        raise ValueError("Option permutation changed the exact-rule label")
                    prompt = prompt_for("choice", question["instructions"], options, record["state"])
                    tokens = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
                    if not 0 < len(tokens) <= 2048:
                        raise ValueError("Complete future-training input exceeds the audited context")
                    audit.append({"id": record["id"], "operation": question["operation"], "split": split,
                                  "order": order, "domain": record["domain"], "options": options, "gold": gold,
                                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                                  "token_ids": tokens, "token_ids_sha256": json_sha(tokens), "encoded_tokens": len(tokens)})
                    lengths.append(len(tokens))
                    operations[f"{order}/{question['operation']}"] += 1
                    positions[f"{order}/k{len(options)}/gold{gold}"] += 1
        summary[split] = {"states": len(records), "decisions": sum(len(r["questions"]) for r in records),
                          "encoded_sequences": len(lengths), "maximum_tokens": max(lengths), "minimum_tokens": min(lengths),
                          "total_tokens": sum(lengths), "operations": dict(operations), "gold_positions": dict(positions),
                          "input_sha256": digest(paths[split])}
    if len(audit) != 5600 or torch.cuda.is_initialized():
        raise ValueError("Unexpected input coverage or CUDA initialization")
    PRIVATE.mkdir()
    OUTPUT.mkdir()
    write_json(PRIVATE / "input_audit.json", audit)
    sources = {name: Path(__file__).with_name(name) for name in (Path(__file__).name, "structured_choice_data.py", "structure_probe.py",
                                                               "train_c.py", "report_qwen_public.py", "qwen_public_evaluation.py")}
    sources.update({"tez/prompt.py": ROOT / "tez/prompt.py", "tez/schema.py": ROOT / "tez/schema.py"})
    result = {"status": "complete", "audited_utc": datetime.now(timezone.utc).isoformat(), "cpu_threads": torch.get_num_threads(),
              "cuda_initialized": torch.cuda.is_initialized(), "gpu_model_loaded": False,
              "tokenizer": "Qwen/Qwen3.5-0.8B", "revision": REVISION, "context_bound": 2048,
              "truncated_sequences": 0, "letter_token_ids": [ids[0] for ids in letter_ids], "splits": summary,
              "audit_sha256": digest(PRIVATE / "input_audit.json"), "audit_bytes": (PRIVATE / "input_audit.json").stat().st_size,
              "source_hashes": {name: digest(path) for name, path in sources.items()},
              "scope": "Training/validation tokenization and exact-label permutation audit only. Confirmation file was not opened. No neural model inference, supervision change, training registration, checkpoint selection or quality result. Current original-data continuation excludes this data. Raw training tokens and labels remain private under runs; logic families recur between splits."}
    write_json(OUTPUT / "preflight.json", result)
    from report_qwen_public import archive_files
    files = {"preflight.json": OUTPUT / "preflight.json", "generation.json": FILTER / "generation.json",
             "filter_verification.json": FILTER / "filter_verification.json", "filter_summary.json": FILTER / "filter_summary.json"}
    files.update({"source/" + name: path for name, path in sources.items()})
    artifact = archive_files(OUTPUT, files)
    print(json.dumps({"splits": {name: {key: value[key] for key in ("states", "decisions", "encoded_sequences", "maximum_tokens", "total_tokens")}
                                for name, value in summary.items()}, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
