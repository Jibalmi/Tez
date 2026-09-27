"""Prepare and verify the registered question-type/option-order ordinal diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from structure_probe import WEIGHTS, digest, validate_records, write_json

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "results/pilot-ordinal-probe-20260927"


def conditions(registration):
    return [(f"{kind}-{' '.join(order).replace(' ', '')}", kind, order)
            for kind in registration["conditions"]["question_type"]
            for order in registration["conditions"]["semantic_option_order"]]


def make_inputs(records, registration):
    validate_records(records)
    inputs = []
    for record in records:
        q = next(q for q in record["questions"] if q["operation"] == "band")
        gold = q["options"][q["gold"]][0]
        options = dict(q["options"])
        for name, kind, order in conditions(registration):
            if sorted(order) != ["0", "1", "2"] or set(options) != set(order):
                raise ValueError("Invalid semantic option permutation")
            inputs.append({"id": record["id"], "condition": name, "state": record["states"]["nested"],
                           "qtype": kind, "instructions": q["instructions"],
                           "options": [(key, options[key]) for key in order], "gold_key": gold})
    return inputs


def prepare(directory):
    original = ROOT / "experiments/small_model/ordinal_diagnostic_registration.json"
    registration = json.loads(original.read_text(encoding="utf-8"))
    source = ROOT / registration["source_cases"]
    if digest(source) != registration["source_cases_sha256"]:
        raise ValueError("Registered source corpus changed")
    records = json.loads(source.read_text(encoding="utf-8"))
    inputs = make_inputs(records, registration)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "inputs.json", inputs)
    registration.update(prepared_utc=datetime.now(timezone.utc).isoformat(), input_sha256=digest(directory / "inputs.json"),
                        original_registration_sha256=digest(original), preparation_source_sha256=digest(__file__),
                        weights_sha256={arm: WEIGHTS[arm] for arm in registration["models"]},
                        groups_per_chunk=48, decisions_per_model=len(inputs),
                        status="prepared_before_inference")
    write_json(directory / "registration.json", registration)
    print(json.dumps({"inputs_sha256": registration["input_sha256"], "decisions_per_model": len(inputs)}))


def prediction_row(case, probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid probability vector")
    if abs(float(probabilities.sum()) - 1) > 1e-5:
        raise ValueError("Probability vector does not sum to one")
    order = [key for key, _ in case["options"]]
    return {"id": case["id"], "condition": case["condition"], "gold_key": case["gold_key"],
            "pred_key": order[int(probabilities.argmax())], "pred_position": int(probabilities.argmax()),
            "presentation_probabilities": probabilities.tolist(),
            "canonical_probabilities": [float(probabilities[order.index(key)]) for key in ("0", "1", "2")]}


def score(inputs, rows, registration):
    import numpy as np
    expected = {(case["id"], case["condition"]): case for case in inputs}
    if len(expected) != len(inputs):
        raise ValueError("Duplicate input key")
    actual = {}
    for row in rows:
        key = row["id"], row["condition"]
        if key not in expected or key in actual:
            raise ValueError("Unexpected or duplicate prediction")
        correct = prediction_row(expected[key], row["presentation_probabilities"])
        if row != correct:
            raise ValueError("Semantic mapping, prediction or gold differs from input")
        actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError("Incomplete prediction set")
    ids = list(dict.fromkeys(case["id"] for case in inputs))
    baseline = np.array([actual[sid, "score-012"]["pred_key"] == actual[sid, "score-012"]["gold_key"] for sid in ids])
    metrics = {}
    rng = np.random.default_rng(13)
    sampled_indices = rng.integers(0, len(ids), size=(2000, len(ids)))
    for name, _, _ in conditions(registration):
        selected = [actual[sid, name] for sid in ids]
        outcomes = np.array([r["pred_key"] == r["gold_key"] for r in selected])
        delta = outcomes.astype(float) - baseline
        metrics[name] = {"n": len(ids), "accuracy": float(outcomes.mean()),
                         "predicted_semantic_labels": dict(Counter(r["pred_key"] for r in selected)),
                         "predicted_positions": dict(Counter(r["pred_position"] for r in selected)),
                         "delta_from_score012_points": float(delta.mean() * 100),
                         "paired_state_ci95_points": np.percentile(delta[sampled_indices].mean(axis=1) * 100,
                                                                    [2.5, 97.5]).tolist()}
    invariance = {}
    for kind in registration["conditions"]["question_type"]:
        names = [name for name, ty, _ in conditions(registration) if ty == kind]
        invariance[kind] = sum(len({actual[sid, name]["pred_key"] for name in names}) == 1 for sid in ids) / len(ids)
    return {"rows_validated": len(rows), "conditions": metrics, "semantic_order_invariance": invariance}


def report(directory):
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    if digest(directory / "inputs.json") != registration["input_sha256"]:
        raise ValueError("Frozen inputs changed")
    source = ROOT / registration["source_cases"]
    if digest(source) != registration["source_cases_sha256"]:
        raise ValueError("Source corpus changed")
    inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    if inputs != json.loads(json.dumps(make_inputs(json.loads(source.read_text(encoding="utf-8")), registration))):
        raise ValueError("Transformed inputs do not reproduce from registered source")
    models = {}
    for arm in registration["models"]:
        receipt_path = directory / f"execution_{arm}.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "complete":
            continue
        rows_path = directory / f"rows_{arm}.jsonl"
        if (receipt["input_sha256"] != registration["input_sha256"] or digest(rows_path) != receipt["rows_sha256"]
                or receipt["weights"] != registration["weights_sha256"][arm]):
            raise ValueError("Execution receipt mismatch")
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        models[arm] = {**score(inputs, rows, registration), "receipt": receipt}
    write_json(directory / "summary.json", {"registration": registration, "models": models})
    lines = ["# Ordinal question-type and answer-position diagnostic", "",
             "This secondary experiment keeps state facts, instructions and semantic answers fixed.",
             "It varies only the declared question type and cyclic presentation order. Every condition",
             "uses one question per call, including the recomputed unmodified score/original-order baseline.",
             "The 144 English records were already observed in the earlier diagnostic; this is not fresh confirmation.", "",
             "| Model | Type/order | Accuracy | Most frequent semantic answer | Share | Difference from score/012 | Paired 95% interval |",
             "|---|---|---:|---|---:|---:|---:|"]
    for arm, model in models.items():
        for name, metric in model["conditions"].items():
            answer, count = max(metric["predicted_semantic_labels"].items(), key=lambda item: item[1])
            lo, hi = metric["paired_state_ci95_points"]
            lines.append(f"| {arm} | {name} | {metric['accuracy']:.2%} | {answer} | {count / metric['n']:.2%} | "
                         f"{metric['delta_from_score012_points']:+.2f} points | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Semantic keys: 0 = below lower limit; 1 = between the inclusive limits; 2 = above upper limit.",
              "Each correct label appears equally often; constant guesses score 33.33%. Intervals resample paired",
              "states 2,000 times with seed 13 and have no multiplicity adjustment. Native tie order is preserved.", "",
              "| Model | Score: same semantic prediction across orders | Choice: same semantic prediction across orders |",
              "|---|---:|---:|"]
    for arm, model in models.items():
        values = model["semantic_order_invariance"]
        lines.append(f"| {arm} | {values['score']:.2%} | {values['choice']:.2%} |")
    lines += ["", "No model was trained, calibrated or promoted. No new RunPod resource was used.",
              "Inference runs under the shared laptop compute lock and unchanged thermal governor; elapsed times",
              "do not establish a latency gate. A follow-up change requires separate representative validation",
              "and fresh confirmation; these diagnostic examples must not select or train it.", "",
              "[Registration](registration.json) · [Verified results and receipts](summary.json)"]
    missing = set(registration["models"]) - models.keys()
    if missing:
        lines += ["", "Incomplete models: " + ", ".join(sorted(missing))]
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({arm: {name: value["accuracy"] for name, value in model["conditions"].items()}
                      for arm, model in models.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "report"))
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else report)(args.directory)
