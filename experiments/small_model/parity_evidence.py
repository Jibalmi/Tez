"""Validate saved probability vectors and compare them without a model dependency."""
import json
import math
from pathlib import Path
from statistics import fmean


def read_rows(path, expected_ids=None):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["state_id"], row["qid"], row["k"])
        p = row["probabilities"]
        if (key in rows or type(row["k"]) is not int or not 2 <= row["k"] <= 26 or len(p) != row["k"]
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or abs(sum(p) - 1) > 1e-5):
            raise ValueError("Invalid or duplicate probability row")
        rows[key] = p
    if not rows or len({key[:2] for key in rows}) != len(rows):
        raise ValueError("Empty rows or duplicate decision identifiers")
    if expected_ids is not None and (set(rows) != set(map(tuple, expected_ids)) or len(rows) != len(expected_ids)):
        raise ValueError("Prediction membership differs from validation")
    return rows


def compare_rows(reference, candidate, spec):
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Parity requires identical nonempty decision sets")
    agreement, distances, variations = [], [], []
    for key, p in reference.items():
        q = candidate[key]
        agreement.append(max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__))
        differences = [abs(a - b) for a, b in zip(p, q)]
        distances.append(max(differences))
        variations.append(sum(differences) / 2)
    metrics = {"argmax_agreement": fmean(agreement), "max_probability_delta": max(distances),
               "mean_total_variation": fmean(variations), "decisions_validated": len(reference)}
    metrics["passed"] = (metrics["argmax_agreement"] >= spec["argmax_agreement_min"]
                         and metrics["max_probability_delta"] <= spec["max_probability_delta_max"]
                         and metrics["mean_total_variation"] <= spec["mean_total_variation_max"])
    return metrics
