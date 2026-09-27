"""Measure whether the fixed encoder text limits discard held-out question or option text."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from common import ROOT, file_sha256  # noqa: E402


def main():
    from tokenizers import Tokenizer
    from tez.prompt import render_state
    path = ROOT / "runs/student-artifacts-h200-20260927/runs/student-data/models/a-base-30k/encoder/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))

    @lru_cache(maxsize=20000)
    def length(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    source = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    suites = json.loads(source.read_text(encoding="utf-8"))
    tasks = {}
    for name, cases in suites.items():
        heads, options, schema_sizes = [], [], {}
        groups = defaultdict(list)
        for case in cases:
            groups[case.get("group", case["id"])].append(case)
            head = length(f" ({case['qtype']}) {case['instructions']}")
            heads.append(head)
            option_lengths = []
            for key, description in case["options"]:
                shown = "none" if key == "__none__" else str(key)
                text = f"{shown}: {description}" if isinstance(description, str) and description and description != shown else shown
                option_lengths.append(length(" " + text))
            options.extend(option_lengths)
            schema_sizes[case["id"]] = 1 + min(160, head) + sum(1 + min(48, size) for size in option_lengths)
        packed = []
        for group in groups.values():
            schema = sum(schema_sizes[c["id"]] for c in group)
            state = length(render_state(group[0]["state"]))
            packed.append({"state_tokens": state, "schema_tokens_after_limits": schema,
                           "state_clipped_at_eval_8192": state > max(32, 8192 - schema - 3),
                           "state_would_clip_at_training_2048": state > max(32, 2048 - schema - 3)})
        tasks[name] = {"decisions": len(cases), "states_or_groups": len(groups),
                       "instruction_limit_tokens": 160, "instructions_clipped": sum(h > 160 for h in heads),
                       "instruction_tokens_max": max(heads), "option_limit_tokens": 48,
                       "options": len(options), "options_clipped": sum(o > 48 for o in options),
                       "option_tokens_max": max(options),
                       "states_clipped_at_eval_8192": sum(r["state_clipped_at_eval_8192"] for r in packed),
                       "states_that_would_clip_at_training_2048": sum(r["state_would_clip_at_training_2048"] for r in packed)}
    report = {"eval_suites_sha256": file_sha256(source), "tokenizer_sha256": file_sha256(path),
              "scope": "Descriptive input audit only; no model inference or fitting and no gold labels used",
              "implementation": "Matches student.Encoder's 160-token question and 48-token option limits",
              "tasks": tasks}
    output = ROOT / "results/pilot-improvement-20260927/encoder_input_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
