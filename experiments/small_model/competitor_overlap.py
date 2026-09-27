"""Measure shared errors and a non-deployable correctness-oracle bound from frozen predictions."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-improvement-20260927"
SUITE = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
SOURCES = {
    "qwen": ("runs/student-artifacts-h200-20260927/results/pilot-student-h200-20260927/rows", "c-qwen08-30k"),
    "decider": ("runs/improvement-artifacts-h200-20260927/results/pilot-improvement-20260927/rows", "decider08-native"),
}


def read_checked(path, cases):
    rows = {}
    expected = {case["id"]: case for case in cases}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key, p = row["id"], row["probabilities"]
        if key not in expected or key in rows:
            raise ValueError("Unknown or repeated decision ID")
        case = expected[key]
        if (len(p) != len(case["options"]) or row["gold"] != case["gold"]
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > 1e-5
                or row["pred"] != max(range(len(p)), key=p.__getitem__)):
            raise ValueError("Invalid aligned probability vector")
        rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError("Incomplete prediction membership")
    return rows


def counts(cases, qwen, decider):
    matrix = Counter()
    agreement = 0
    for case in cases:
        q, d = qwen[case["id"]], decider[case["id"]]
        matrix[(q["pred"] == case["gold"], d["pred"] == case["gold"])] += 1
        agreement += q["pred"] == d["pred"]
    n = len(cases)
    return {"n": n, "both_correct": matrix[True, True], "qwen_only_correct": matrix[True, False],
            "decider_only_correct": matrix[False, True], "both_wrong": matrix[False, False],
            "agreement": agreement / n,
            "qwen_accuracy": (matrix[True, True] + matrix[True, False]) / n,
            "decider_accuracy": (matrix[True, True] + matrix[False, True]) / n,
            "correctness_oracle_accuracy": 1 - matrix[False, False] / n}


def main():
    if (DIRECTORY / "overlap.json").exists():
        raise ValueError("Preserve existing overlap analysis")
    if digest(SUITE) != "fea63e7530b2166d51b2b6b8f4adfd8994c75db48fcbaf6cca13d339f959def5":
        raise ValueError("Frozen public suite changed")
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    tasks, hashes, typed = {}, {}, {}
    for task, cases in suite.items():
        rows = {}
        for name, (directory, arm) in SOURCES.items():
            path = ROOT / directory / f"rows_{task}_tez-s-{arm}.jsonl"
            rows[name] = read_checked(path, cases)
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        tasks[task] = counts(cases, rows["qwen"], rows["decider"])
        if task == "typed_decisions":
            groups = defaultdict(list)
            for case in cases:
                groups[case["qtype"]].append(case)
            typed = {name: counts(cases, rows["qwen"], rows["decider"]) for name, cases in groups.items()}
    summary = {"suite_sha256": digest(SUITE), "analysis_source_sha256": digest(__file__), "prediction_hashes": hashes,
               "tasks": tasks, "typed_by_question_type": typed,
               "scope": "Post-hoc descriptive error overlap. The correctness oracle uses test gold to choose the correct model, so it is an upper bound, not a deployable model, measured ensemble, new release result or checkpoint-selection criterion."}
    write_json(DIRECTORY / "overlap.json", summary)
    lines = ["# Shared errors: Qwen and Decider", "", "These counts use the complete frozen public cases and validate all raw probability rows against them.",
             "The oracle chooses a correct answer whenever either model is correct. It uses test labels, so it cannot be deployed.",
             "It is an upper bound on choosing between these two answers, not a measured ensemble or a quality gate result.", "",
             "| Task | Qwen accuracy | Decider accuracy | Oracle ceiling | Qwen-only correct | Decider-only correct | Both wrong |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stats in tasks.items():
        lines.append(f"| {name} | {100 * stats['qwen_accuracy']:.2f}% | {100 * stats['decider_accuracy']:.2f}% | "
                     f"{100 * stats['correctness_oracle_accuracy']:.2f}% | {stats['qwen_only_correct']} | "
                     f"{stats['decider_only_correct']} | {stats['both_wrong']} |")
    lines += ["", "All columns here use ordinary accuracy. SemIf's official mean family balanced accuracy remains in the matched comparison;",
              "it is a different metric. The counts include every decision rather than filtering to disagreements.",
              "No confidence thresholds, task routing rules, model weights or temperatures were selected from this diagnostic.",
              "The earlier [fixed equal-probability ensemble](ENSEMBLE.md) remains a separate measured result.",
              "[Machine-readable counts and source hashes](overlap.json).", ""]
    (DIRECTORY / "OVERLAP.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"typed": tasks["typed_decisions"], "typed_by_question_type": typed}, indent=2))


if __name__ == "__main__":
    main()
