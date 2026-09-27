"""CPU-only, single-thread input audit before the Tev1 public benchmark; no gold is scored."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from structure_probe import digest, write_json
from tev1_adapter import load_native, render_case

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-tev1-eval-20260927"


def main():
    os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    from transformers import AutoTokenizer
    prep_path = ROOT / "results/pilot-tev1-discovery-20260927/preparation.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    native_path = ROOT / "runs/research/tev1-discovery-20260927/source/examples/decide.py"
    suite_path = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    if digest(suite_path) != "fea63e7530b2166d51b2b6b8f4adfd8994c75db48fcbaf6cca13d339f959def5":
        raise ValueError("Frozen public suite changed")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / prep["directory"], local_files_only=True)
    native = load_native(native_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "input_audit.json"
    if path.exists():
        raise ValueError("Preserve existing input audit")
    tasks = {}
    for task, cases in suite.items():
        lengths = []
        for case in cases:
            _, messages = render_case(native, case, prep["repository"])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False)))
        tasks[task] = {"n": len(cases), "max_encoded_tokens": max(lengths), "total_tokens": sum(lengths)}
    receipt = {"status": "complete", "audited_utc": datetime.now(timezone.utc).isoformat(), "tasks": tasks,
               "suite_sha256": digest(suite_path), "native_helper_sha256": digest(native_path),
               "preparation_sha256": digest(prep_path), "source_sha256": digest(__file__),
               "adapter_sha256": digest(Path(__file__).with_name("tev1_adapter.py")),
               "scope": "Only original inputs and lengths; no model inference, gold labels, fitting or quality scores. Reversing option order preserves total prompt length for these native JSON option records."}
    write_json(path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
