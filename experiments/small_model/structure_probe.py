"""A paired diagnostic of JSON nesting, using independently authored deterministic records.

This is an English reasoning diagnostic, not a replacement for the registered public quality gates.
The generator never opens a benchmark or training file. Both representations share every leaf,
question, option order and label. No probe example may enter training or checkpoint selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random


DOMAINS = [
    ("observatory", "instrument", "filter_mode", ("clear", "amber", "violet"), "exposure_seconds", "operator_ready", "mount"),
    ("greenhouse", "bed", "irrigation_mode", ("drip", "mist", "soak"), "soil_reading", "valve_ready", "section"),
    ("ceramics", "kiln", "firing_mode", ("bisque", "glaze", "anneal"), "temperature_reading", "lid_closed", "shelf"),
    ("theater", "stage", "lighting_mode", ("wash", "spot", "flood"), "light_reading", "rig_ready", "zone"),
]
FORMATS = ("flat", "nested")
OPERATIONS = ("lookup", "threshold", "conjunction", "band", "missing")
WEIGHTS = {
    "a-base-30k": {"encoder": "738d16f5b200ae110fbf4d705c2c34a16807c0c80dd3725192ffa3766210d948",
                   "head": "2544f844a2a86a2c748e89bf4fd37c2aa8076fd396b63ffc512d77121faef299"},
    "a-base-12ep": {"encoder": "3b3df2cfa30876e9766e61007b8b10f8f23117fbccb8a743817cce0898ca305b",
                    "head": "c8c4303cf7ecdcaf9a783455b9882e04a515807be90b366991ae6ed02a6023ad"},
    "c-qwen08-30k": {"adapter": "fee3b3fec20902f247bf7758ad697ae4ea75aa40d862c060f9aafecb613ed482"},
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def leaves_by_key(state):
    """Reject ambiguous leaf names; the questions refer to names, not a particular path."""
    result = {}
    for key, value in state.items():
        additions = leaves_by_key(value) if isinstance(value, dict) else {key: value}
        if result.keys() & additions.keys():
            raise ValueError("Duplicate leaf key")
        result.update(additions)
    return result


def oracle(state, spec):
    """Evaluate the documented rules independently of serialization and option positions."""
    fact = leaves_by_key(state)
    quantity = fact[spec["quantity"]]
    lower, upper = fact["lower_limit"], fact["upper_limit"]
    return {
        "lookup": fact[spec["mode"]],
        "threshold": "true" if quantity >= lower else "false",
        "conjunction": "true" if fact[spec["flag"]] and quantity >= lower else "false",
        "band": str(0 if quantity < lower else 1 if quantity <= upper else 2),
        "missing": fact.get(spec["optional"]) or "__none__",
    }


def generate(seed=2719):
    rng = random.Random(seed)
    records = []
    for domain, container, mode, modes, quantity, flag, optional in DOMAINS:
        spec = {"mode": mode, "quantity": quantity, "flag": flag, "optional": optional}
        for index in range(36):
            mode_index, bucket, flag_value = index // 12, (index // 2) % 6, bool(index % 2)
            lower = 10 + 3 * rng.randrange(10)
            upper = lower + rng.choice((5, 7, 11))
            value = (lower - 1, lower, upper, upper + 1, lower - 3, upper + 3)[bucket]
            facts = {mode: modes[mode_index], quantity: value, flag: flag_value,
                     "lower_limit": lower, "upper_limit": upper, "inspection_note": "Routine record; apply the listed rules."}
            optional_value = ("north", "central", "south", None)[index % 4]
            if optional_value is not None:
                facts[optional] = optional_value
            elif index % 8 == 7:
                facts[optional] = None
            nested = {domain: {container: {mode: facts[mode], quantity: facts[quantity], flag: facts[flag]},
                               "limits": {"lower_limit": lower, "upper_limit": upper}},
                      "annotation": {"inspection_note": facts["inspection_note"]}}
            if optional in facts:
                nested[domain][container][optional] = facts[optional]
            # Keep the traversal order of all leaf keys and values identical in both representations.
            facts = leaves_by_key(nested)
            gold = oracle(facts, spec)
            definitions = [
                ("lookup", "choice", f"Read the {mode} field. Which mode is explicitly recorded?",
                 [(m, f"The {mode} field is {m}.") for m in modes]),
                ("threshold", "noul", f"Is {quantity} greater than or equal to lower_limit?",
                 [("true", "Yes"), ("false", "No")]),
                ("conjunction", "noul", f"Are both conditions true: {flag} is true AND {quantity} is at least lower_limit?",
                 [("true", "Yes"), ("false", "No")]),
                ("band", "score", f"Classify {quantity} using lower_limit and upper_limit. Boundaries belong to the middle band.",
                 [("0", "Below lower_limit"), ("1", "At least lower_limit and at most upper_limit"), ("2", "Above upper_limit")]),
                ("missing", "choice", f"Read the {optional} field. If the field is missing or null, choose none; do not infer a value.",
                 [(v, f"The {optional} field is {v}.") for v in ("north", "central", "south")]
                 + [("__none__", "The field is missing or null.")]),
            ]
            questions = []
            for operation, qtype, instruction, options in definitions:
                if qtype != "score":
                    rng.shuffle(options)
                questions.append({"operation": operation, "qtype": qtype, "instructions": instruction,
                                  "options": options, "gold": [k for k, _ in options].index(gold[operation])})
            records.append({"id": f"structure-v1/{domain}/{index:02d}", "domain": domain, "spec": spec,
                            "states": {"flat": facts, "nested": nested}, "questions": questions})
    rng.shuffle(records)
    return records


def validate_records(records):
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(record["id"])
        flat, nested = (record["states"][form] for form in FORMATS)
        if list(leaves_by_key(flat).items()) != list(leaves_by_key(nested).items()):
            raise ValueError("Paired states differ in content")
        if [q["operation"] for q in record["questions"]] != list(OPERATIONS):
            raise ValueError("Question set differs")
        for form in FORMATS:
            expected = oracle(record["states"][form], record["spec"])
            for question in record["questions"]:
                keys = [k for k, _ in question["options"]]
                if len(set(keys)) != len(keys) or keys[question["gold"]] != expected[question["operation"]]:
                    raise ValueError("Invalid options or oracle label")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    records = generate()
    validate_records(records)
    write_json(output / "cases.json", records)
    distributions = {}
    for op in OPERATIONS:
        questions = [q for r in records for q in r["questions"] if q["operation"] == op]
        distributions[op] = {"gold_option_positions": dict(Counter(q["gold"] for q in questions)),
                             "gold_keys": dict(Counter(q["options"][q["gold"]][0] for q in questions))}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 2719,
                "states": len(records), "paired_decisions": sum(len(r["questions"]) for r in records),
                "representations": list(FORMATS), "domains": [d[0] for d in DOMAINS],
                "generator_sha256": digest(__file__), "cases_sha256": digest(output / "cases.json"),
                "distributions": distributions,
                "scope": "Fresh English synthetic diagnostic; fixed repeated rule families, not generalization certification.",
                "isolation": "Generated without reading training or public benchmark records. Evaluation only; never fit or select on it.",
                "primary": "Paired nested-minus-flat accuracy and label-flip rate; per-operation accuracy diagnoses content difficulty.",
                "uncertainty": "2000 bootstrap resamples of whole states, seed 13; unadjusted exploratory percentile intervals.",
                "interpretation": "A nesting effect supports serialization sensitivity. Low accuracy in both forms suggests content difficulty. Neither establishes a training intervention will help.",
                "models": ["a-base-30k", "a-base-12ep", "c-qwen08-30k"],
                "weights_sha256": WEIGHTS,
                "execution": "Original frozen weights, T=1, original predictor batch settings; 8 groups per chunk, interleaved formats. No tuning. Laptop through shared run_heavy lock, no RunPod spending.",
                "not_measured": "No new latency claim, training, probability calibration or release promotion."}
    write_json(output / "registration.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("states", "paired_decisions", "cases_sha256")}))


def score(records, rows):
    import numpy as np
    validate_records(records)
    expected = {(r["id"], form, q["operation"]): q for r in records for form in FORMATS for q in r["questions"]}
    actual = {}
    for row in rows:
        key = (row["id"], row["format"], row["operation"])
        if key in actual or key not in expected:
            raise ValueError("Duplicated or unexpected prediction")
        question = expected[key]
        p = np.asarray(row["probabilities"], dtype=float)
        if (p.shape != (len(question["options"]),) or not np.isfinite(p).all() or np.any(p < 0)
                or abs(p.sum() - 1) > 1e-5 or row["pred"] != int(p.argmax()) or row["gold"] != question["gold"]):
            raise ValueError("Invalid prediction, probability or label")
        actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError("Incomplete predictions")
    metrics = {}
    for form in FORMATS:
        metrics[form] = {}
        for op in ("all", *OPERATIONS):
            selected = [r for r in rows if r["format"] == form and (op == "all" or r["operation"] == op)]
            metrics[form][op] = {"n": len(selected), "accuracy": sum(r["pred"] == r["gold"] for r in selected) / len(selected)}
    differences, flips, tv = [], [], []
    by_domain = defaultdict(list)
    for record in records:
        state_diffs = []
        for q in record["questions"]:
            flat, nested = (actual[(record["id"], form, q["operation"])] for form in FORMATS)
            delta = int(nested["pred"] == q["gold"]) - int(flat["pred"] == q["gold"])
            state_diffs.append(delta)
            flips.append(flat["pred"] != nested["pred"])
            tv.append(float(np.abs(np.asarray(flat["probabilities"]) - nested["probabilities"]).sum() / 2))
        differences.append(np.mean(state_diffs))
        by_domain[record["domain"]].extend(state_diffs)
    rng = np.random.default_rng(13)
    sample = rng.choice(differences, size=(2000, len(differences))).mean(axis=1) * 100
    return {"rows_validated": len(rows), "accuracy": metrics,
            "nested_minus_flat_points": float(np.mean(differences) * 100),
            "paired_state_bootstrap_ci95_points": np.percentile(sample, [2.5, 97.5]).tolist(),
            "prediction_flip_fraction": float(np.mean(flips)), "mean_total_variation": float(np.mean(tv)),
            "per_domain_delta_points": {key: float(np.mean(value) * 100) for key, value in by_domain.items()}}


def summarize(directory):
    records = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if digest(directory / "cases.json") != registration["cases_sha256"]:
        raise ValueError("Registered inputs changed")
    models = {}
    for arm in registration["models"]:
        path = directory / f"rows_{arm}.jsonl"
        if not path.exists():
            continue
        receipt = json.loads((directory / f"execution_{arm}.json").read_text(encoding="utf-8"))
        if receipt["status"] != "complete":
            continue
        if digest(path) != receipt["rows_sha256"] or receipt["cases_sha256"] != registration["cases_sha256"]:
            raise ValueError("Execution receipt hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        models[arm] = {**score(records, rows), "receipt": receipt}
    write_json(directory / "summary.json", {"registration": registration, "models": models})
    print(json.dumps({arm: {k: m[k] for k in ("nested_minus_flat_points", "paired_state_bootstrap_ci95_points")}
                      for arm, m in models.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else summarize)(args.directory)


if __name__ == "__main__":
    main()
