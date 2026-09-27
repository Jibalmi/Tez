"""Reconstruct historical Qwen token limits without inference or exporting training text."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tarfile

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-qwen-context-audit-20260927"
TEACHER = ROOT / "runs/pilot-cloud-h200-20260927"


def main():
    if DIRECTORY.exists():
        raise ValueError("Preserve existing context audit")
    DIRECTORY.mkdir(parents=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    import dataset
    from train_c import prompt_for
    from tez.prompt import TEMPLATES
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    manifest_path = TEACHER / "manifest.json"
    if digest(manifest_path) != "8e3f7cf23f7b14c80203eca42eab9bb7e7108f8c7aab67abd9d9b136f7cfb807":
        raise ValueError("Frozen teacher manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(TEACHER / name) != expected:
            raise ValueError("Original teacher file changed")
    dataset.RUNS = TEACHER
    train, _, validation = dataset.load_examples()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                                             local_files_only=True)
    footer = tokenizer.encode(TEMPLATES["qwen3"][1], add_special_tokens=False)
    if tokenizer.truncation_side != "right":
        raise ValueError("Historical truncation-side assumption must be reviewed")
    rows_path = DIRECTORY / "token_metadata.jsonl"
    groups = {}
    with rows_path.open("x", encoding="utf-8") as output:
        for split, examples, seed in (("training_epoch0", train, 13000), ("training_validation", validation, 0),
                                     ("current_validation_original_order", validation, None)):
            rng = random.Random(seed) if seed is not None else None
            counts, affected_type, affected_source, lengths = Counter(), Counter(), Counter(), []
            for example in examples:
                for decision in example.decisions:
                    order = list(range(len(decision.options)))
                    if rng:
                        rng.shuffle(order)
                    prompt = prompt_for(decision.type, decision.instructions, [decision.options[i] for i in order], example.state)
                    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
                    clipped = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=1536)
                    if clipped != ids[:1536] or ids[-len(footer):] != footer:
                        raise ValueError("Unexpected actual truncation or prompt-footer behavior")
                    length = len(ids)
                    affected = length > 1536
                    footer_retained = clipped[-len(footer):] == footer
                    row = {"split": split, "state_id": example.id, "qid": decision.qid, "qtype": decision.type,
                           "source": example.source, "lang": example.lang, "tokens": length, "clipped_tokens": len(clipped),
                           "over_1536": affected, "over_4096": length > 4096, "complete_assistant_footer_retained_at_1536": footer_retained,
                           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    lengths.append(length)
                    counts["decisions"] += 1
                    counts["over_1536"] += affected
                    counts["over_4096"] += length > 4096
                    counts["assistant_footer_not_intact"] += not footer_retained
                    if affected:
                        affected_type[decision.type] += 1
                        affected_source[example.source] += 1
            groups[split] = {**counts, "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths),
                             "affected_fraction": counts["over_1536"] / counts["decisions"],
                             "affected_qtypes": dict(affected_type), "affected_sources": dict(affected_source)}
    if groups["training_epoch0"]["decisions"] != 29990 or groups["training_validation"]["decisions"] != 1390:
        raise ValueError("Unexpected original data membership")
    paths = ["experiments/small_model/" + name for name in
             ("audit_qwen_context.py", "train_c.py", "dataset.py", "common.py", "structure_probe.py", "train_a.py", "student.py")]
    paths += ["tez/prompt.py", "tez/schema.py", "tez/errors.py"]
    sources = {name: digest(ROOT / name) for name in paths}
    summary = {"status": "complete", "audited_utc": datetime.now(timezone.utc).isoformat(),
               "teacher_manifest_sha256": digest(manifest_path), "sources": sources, "groups": groups,
               "tokenizer_revision": "2fc06364715b967f1860aea9cf38778875588b17", "truncation_side": tokenizer.truncation_side,
               "raw_metadata_sha256": digest(rows_path), "torch_threads": torch.get_num_threads(),
               "scope": "Descriptive reconstruction from frozen training data and pinned prompt/tokenizer code. Historical train uses option-shuffle seed 13000, validation seed 0; current validation is unshuffled. No inference or fitting, and no causal accuracy effect is established. Metadata rows omit prompt text, targets and model weights."}
    write_json(DIRECTORY / "audit.json", summary)
    lines = ["# Qwen training-context audit", "", "Reconstructed from the frozen teacher data, original option-shuffle seeds and pinned Qwen tokenizer.",
             "The original training helper used right truncation at 1,536 tokens, including when fitting its validation temperatures.",
             "Later inference and the frozen local validation reference allow 4,096 tokens.", "",
             "| Input protocol | Decisions | Over 1,536 | Complete assistant footer lost | Maximum tokens |",
             "|---|---:|---:|---:|---:|"]
    for name, value in groups.items():
        lines.append(f"| {name} | {value['decisions']} | {value['over_1536']} | {value['assistant_footer_not_intact']} | {value['max_tokens']} |")
    lines += ["", "Right truncation was verified against the tokenizer's actual output for every row.",
              "Losing the complete assistant footer means affected examples train or validate a decision at a different prompt ending.",
              "This audit does not prove how much accuracy that cost. Existing trained checkpoints and benchmark results remain unchanged.",
              "A continuation should audit complete inputs before training and preserve the answer-position suffix.",
              "[Full audit](audit.json). The archive includes source code and per-decision length/hash metadata; no training text, targets or weights.", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    files = {p.name: p for p in DIRECTORY.iterdir() if p.is_file()}
    files.update({"source/" + name: ROOT / name for name in paths})
    files["inputs/teacher_manifest.json"] = manifest_path
    expected = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in files.items()}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if actual != expected:
        raise ValueError("Archive mismatch")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    artifact = {"sha256": digest(archive), "bytes": archive.stat().st_size, "files_verified": len(actual),
                "metadata_rows": sum(value["decisions"] for value in groups.values())}
    write_json(DIRECTORY / "artifact_receipt.json", artifact)
    print(json.dumps({"groups": groups, "artifact": artifact}, indent=2))


if __name__ == "__main__":
    main()
