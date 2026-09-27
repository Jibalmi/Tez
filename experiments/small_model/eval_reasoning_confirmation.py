"""Evaluate frozen selected continuations on sealed authored confirmation states, after both finish."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import time

from structure_probe import WEIGHTS, digest, write_json
from train_reasoning_round2 import ROOT, RESULTS, RUNS


def prerequisites():
    config_path = ROOT / "experiments/small_model/reasoning_round2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = {}
    for arm in config["arms"]:
        receipt = json.loads((RESULTS / f"train_{arm}.json").read_text(encoding="utf-8"))
        if (receipt["status"] != "complete" or receipt["step"] != config["steps"]
                or receipt["specification_sha256"] != digest(config_path)):
            raise ValueError("Finish both registered continuations and freeze selection before opening confirmation labels")
        completed[arm] = receipt
    data_manifest = ROOT / config["new_data"] / "manifest.json"
    if any(receipt["data_manifest_sha256"] != digest(data_manifest) for receipt in completed.values()):
        raise ValueError("Data manifest changed after training selection")
    return config, completed


def cases_for(records):
    groups = []
    for record in records:
        groups.append([{"id": record["id"] + "/" + q["operation"], "group": record["id"], "operation": q["operation"],
                        "domain": record["domain"], "state": record["state"], "qtype": q["qtype"],
                        "instructions": q["instructions"], "options": q["options"],
                        "gold": [k for k, _ in q["options"]].index(q["gold_key"])} for q in record["questions"]])
    return groups


def evaluate(arm):
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    config, trained = prerequisites()
    if arm == config["base_arm"]:
        model_root = (ROOT / config["base_directory"]).parent.parent
        expected_weights = WEIGHTS[arm]
    elif arm in trained:
        model_root, expected_weights = RUNS, trained[arm]["weights"]
    else:
        raise ValueError("Unregistered confirmation model")
    os.environ.update(TEZ_PILOT_RUNS=str(model_root), TEZ_PILOT_RESULTS=str(RESULTS), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import torch
    from common import THERMAL, thermal_guard
    from competitor_models import predictor
    from train_c import kernel_packages

    data_root = ROOT / config["new_data"]
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    source = data_root / "confirmation.json"
    if digest(source) != manifest["splits"]["confirmation"]["sha256"]:
        raise ValueError("Sealed confirmation corpus changed")
    groups = cases_for(json.loads(source.read_text(encoding="utf-8")))
    path = RESULTS / f"confirmation_rows_{arm}.jsonl"
    receipt_path = RESULTS / f"confirmation_{arm}.json"
    if path.exists() or receipt_path.exists():
        raise ValueError("Existing evaluation is immutable")
    receipt = {"status": "running", "arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
               "confirmation_sha256": digest(source), "source_sha256": digest(__file__), "rows": 0,
               "packages": kernel_packages(), "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads()}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        thermal_guard(force=True)
        pred = predictor(arm)
        if pred.weights != expected_weights:
            raise ValueError("Selected model differs from its completed training receipt")
        receipt.update(weights=pred.weights, model_info=pred.info)
        with path.open("x", encoding="utf-8") as output:
            for offset in range(0, len(groups), 8):
                if time.monotonic() - started > 6 * 60:
                    raise TimeoutError("Confirmation worker time limit reached")
                thermal_guard(force=True)
                chunk = groups[offset:offset + 8]
                safe = [[{key: c[key] for key in ("state", "qtype", "instructions", "options")} for c in group] for group in chunk]
                values = pred.groups(safe)
                if len(values) != len(chunk) or any(len(cs) != len(ps) for cs, ps in zip(chunk, values)):
                    raise ValueError("Prediction count differs from confirmation cases")
                for cases, predictions in zip(chunk, values):
                    for case, z in zip(cases, predictions):
                        z = np.asarray(z, dtype=float)
                        if z.shape != (len(case["options"]),) or not np.isfinite(z).all():
                            raise ValueError("Invalid confirmation logits")
                        p = np.exp(z - z.max())
                        p /= p.sum()
                        row = {key: case[key] for key in ("id", "group", "operation", "domain", "gold")}
                        row.update(pred=int(p.argmax()), probabilities=p.tolist())
                        output.write(json.dumps(row) + "\n")
                        receipt["rows"] += 1
                output.flush()
                write_json(receipt_path, receipt)
        receipt.update(status="complete", rows_sha256=digest(path))
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                       thermal_governor=THERMAL)
        write_json(receipt_path, receipt)


def report():
    import numpy as np
    config, _ = prerequisites()
    source = ROOT / config["new_data"] / "confirmation.json"
    manifest = json.loads(source.with_name("manifest.json").read_text(encoding="utf-8"))
    if digest(source) != manifest["splits"]["confirmation"]["sha256"]:
        raise ValueError("Sealed confirmation corpus changed")
    records = json.loads(source.read_text(encoding="utf-8"))
    cases = {c["id"]: c for group in cases_for(records) for c in group}
    models, predictions = {}, {}
    for arm in [config["base_arm"], *config["arms"]]:
        receipt = json.loads((RESULTS / f"confirmation_{arm}.json").read_text(encoding="utf-8"))
        path = RESULTS / f"confirmation_rows_{arm}.jsonl"
        if (receipt["status"] != "complete" or digest(path) != receipt["rows_sha256"]
                or receipt["confirmation_sha256"] != digest(source)):
            raise ValueError("Confirmation predictions are incomplete or changed")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        keyed = {row["id"]: row for row in rows}
        if len(keyed) != len(rows) or set(keyed) != set(cases):
            raise ValueError("Missing or duplicated confirmation case")
        metrics = defaultdict(list)
        for key, row in keyed.items():
            case = cases[key]
            p = np.asarray(row["probabilities"])
            if (any(row[field] != case[field] for field in ("group", "operation", "domain", "gold"))
                    or p.shape != (len(case["options"]),) or not np.isfinite(p).all() or np.any(p < 0)
                    or abs(float(p.sum()) - 1) > 1e-5 or row["pred"] != int(p.argmax())):
                raise ValueError("Invalid confirmation row")
            correct = row["pred"] == row["gold"]
            for name in ("all", case["operation"], case["domain"]):
                metrics[name].append(correct)
        models[arm] = {"rows_validated": len(rows), "accuracy": {key: float(np.mean(value)) for key, value in metrics.items()},
                       "receipt": receipt}
        predictions[arm] = keyed
    paired = {}
    treatment = "a-base-rules150"
    for reference in (config["base_arm"], "a-base-replay150"):
        by_state = defaultdict(list)
        for key, case in cases.items():
            a, b = predictions[treatment][key], predictions[reference][key]
            by_state[case["group"]].append(int(a["pred"] == a["gold"]) - int(b["pred"] == b["gold"]))
        differences = np.array([np.mean(values) for values in by_state.values()])
        rng = np.random.default_rng(13)
        boot = rng.choice(differences, size=(2000, len(differences))).mean(1) * 100
        paired[reference] = {"difference_points": float(differences.mean() * 100), "ci95_points": np.percentile(boot, [2.5, 97.5]).tolist()}
    write_json(RESULTS / "confirmation_summary.json", {"models": models, "rules_minus_reference": paired,
                "scope": "Fresh authored states and domains, recurrent logic families; not a substitute for original public and latency gates."})
    print(json.dumps({"accuracy": {arm: result["accuracy"] for arm, result in models.items()}, "paired": paired}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm")
    args = parser.parse_args()
    evaluate(args.arm) if args.arm else report()
