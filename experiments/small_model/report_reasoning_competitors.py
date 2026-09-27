"""Validate and archive native competitor predictions beside the frozen encoder rule test."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from eval_reasoning_confirmation import cases_for
from eval_reasoning_competitors import DIRECTORY
from structure_probe import digest, write_json
from train_reasoning_round2 import ROOT, RESULTS


def main():
    execution = json.loads((DIRECTORY / "execution.json").read_text(encoding="utf-8"))
    reg_path = DIRECTORY / "registration.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    source = ROOT / reg["source"]
    if execution["status"] != "complete" or digest(source) != reg["source_sha256"]:
        raise ValueError("Complete both competitors on the unchanged corpus first")
    cases = {c["id"]: c for group in cases_for(json.loads(source.read_text(encoding="utf-8"))) for c in group}
    initial = json.loads((RESULTS / "confirmation_summary.json").read_text(encoding="utf-8"))
    models, keyed_models = {}, {}
    for arm in [*initial["models"], *reg["arms"]]:
        if arm in initial["models"]:
            receipt = initial["models"][arm]["receipt"]
            path = RESULTS / f"confirmation_rows_{arm}.jsonl"
        else:
            receipt = json.loads((DIRECTORY / f"execution_{arm}.json").read_text(encoding="utf-8"))
            path = DIRECTORY / f"rows_{arm}.jsonl"
            if (receipt["registration_sha256"] != digest(reg_path) or receipt["weights"] != reg["weights"][arm]
                    or receipt["source_sha256"] != digest(source)):
                raise ValueError("Competitor registration, corpus or weights mismatch")
        if receipt["status"] != "complete" or digest(path) != receipt["rows_sha256"]:
            raise ValueError("Incomplete or changed predictions")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        keyed = {r["id"]: r for r in rows}
        if len(keyed) != len(rows) or set(keyed) != set(cases):
            raise ValueError("Missing or duplicated decisions")
        scores = defaultdict(list)
        for key, row in keyed.items():
            case = cases[key]
            p = np.asarray(row["probabilities"], dtype=float)
            if (any(row[k] != case[k] for k in ("group", "domain", "operation", "gold"))
                    or p.shape != (len(case["options"]),) or not np.isfinite(p).all() or np.any(p < 0)
                    or abs(float(p.sum()) - 1) > 1e-5 or row["pred"] != int(p.argmax())):
                raise ValueError("Invalid semantic or probability mapping")
            if "logits" in row:
                z = np.asarray(row["logits"], dtype=float)
                if z.shape != p.shape or not np.isfinite(z).all():
                    raise ValueError("Invalid saved logits")
                expected = np.exp(z - z.max())
                expected /= expected.sum()
                if not np.allclose(p, expected, rtol=1e-6, atol=1e-7):
                    raise ValueError("Probabilities differ from saved logits")
            for subset in ("all", case["domain"], case["operation"]):
                scores[subset].append(row["pred"] == row["gold"])
        models[arm] = {"rows_validated": len(rows), "accuracy": {s: float(np.mean(v)) for s, v in scores.items()}, "receipt": receipt}
        keyed_models[arm] = keyed
    reference = "a-base-rules150"
    paired = {}
    for arm, rows in keyed_models.items():
        if arm == reference:
            continue
        differences = defaultdict(list)
        for key, case in cases.items():
            other = keyed_models[reference][key]
            differences[case["group"]].append(int(rows[key]["pred"] == case["gold"]) - int(other["pred"] == case["gold"]))
        values = np.array([np.mean(v) for v in differences.values()])
        draws = np.random.default_rng(13).choice(values, size=(2000, len(values))).mean(1) * 100
        paired[arm] = {"difference_points": float(values.mean() * 100), "ci95_points": np.percentile(draws, [2.5, 97.5]).tolist()}
    summary = {"models": models, "minus_rule_supervision": paired, "registration": reg,
               "scope": "Exploratory comparison of frozen checkpoints. Fresh domains, recurring logic families; not public benchmark accuracy or a release gate."}
    write_json(DIRECTORY / "summary.json", summary)
    lines = ["# Frozen competitors on the authored rule test", "", "360 states and 1,800 decisions per model. Same facts, instructions, options and gold labels.",
             "Native model rendering; no truncation, tuning or calibration fitting. All probability rows are validated.",
             "The competitor addition was registered after the encoder confirmation result was observed.", "",
             "| Model | Overall | Compare | Combined | Policy | Count | Largest |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, model in models.items():
        lines.append(f"| {arm} | " + " | ".join(f"{100 * model['accuracy'][s]:.2f}" for s in ("all", "compare", "combined", "policy", "count", "largest")) + " |")
    lines += ["", "Values are percentages. Original encoder confirmations are reused, with their checksums verified.",
              "The rule-supervision model trained on independently authored examples of compare, combined and count.",
              "These logic families recur; the domains and states are fresh. Choice-schema overlap filtering removed policy/largest from added training.",
              "Different training exposure means this is a utility comparison, not a controlled architecture experiment.", "",
              "| Model minus rule-supervision encoder | Difference, points | 95% state-bootstrap interval |", "|---|---:|---:|"]
    for arm, stats in paired.items():
        lo, hi = stats["ci95_points"]
        lines.append(f"| {arm} | {stats['difference_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Exploratory intervals use 2,000 state resamples, seed 13, without multiplicity correction.",
              "Decider retains native independent questions, isolated score levels and four-decimal API rounding; Qwen retains its existing Tez letter prompt.",
              "These are quality measurements, not valid laptop speed measurements. No extra cloud compute was used.",
              "[Original public regression results](../pilot-reasoning-round2-20260927/RESULTS.md) remain separate and continue to govern the gates.", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    archive = DIRECTORY / "measurement-artifacts.tar.gz"
    if archive.exists():
        raise ValueError("Preserve existing archive")
    for arm in reg["arms"]:
        for name, expected in models[arm]["receipt"]["source_files"].items():
            if digest(DIRECTORY / "executed_source" / arm / name) != expected:
                raise ValueError("Executed source checksum mismatch")
    excluded = {archive.name, "artifact_manifest.json", "artifact_receipt.json"}
    files = {p.relative_to(DIRECTORY).as_posix(): p for p in DIRECTORY.rglob("*") if p.is_file() and p.name not in excluded}
    files["inputs/confirmation.json"] = source
    files["analysis_source/report_reasoning_competitors.py"] = Path(__file__)
    for arm in initial["models"]:
        files[f"reference/rows_{arm}.jsonl"] = RESULTS / f"confirmation_rows_{arm}.jsonl"
        files[f"reference/receipt_{arm}.json"] = RESULTS / f"confirmation_{arm}.json"
    expected = {name: {"sha256": digest(p), "bytes": p.stat().st_size} for name, p in sorted(files.items())}
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
    actual = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected archive member")
            actual[member.name] = {"sha256": hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), "bytes": member.size}
    if expected != actual:
        raise ValueError("Archive checksum mismatch")
    write_json(DIRECTORY / "artifact_manifest.json", expected)
    write_json(DIRECTORY / "artifact_receipt.json", {"sha256": digest(archive), "bytes": archive.stat().st_size,
                "files_verified": len(actual), "prediction_rows_validated": sum(m["rows_validated"] for m in models.values()),
                "created_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({arm: m["accuracy"] for arm, m in models.items()}, indent=2))


if __name__ == "__main__":
    main()
