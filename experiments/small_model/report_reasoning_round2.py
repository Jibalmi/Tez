"""Validate the frozen continuation predictions and compare them with recorded competitors."""
from __future__ import annotations

import json
import statistics

import numpy as np

from compare_round1 import paired_typed_delta, read_rows
from eval_reasoning_confirmation import prerequisites, report as confirmation_report
from structure_probe import digest, write_json
from train_reasoning_round2 import ROOT, RESULTS


def validate_rows(cases, rows, task):
    expected = {case["id"]: case for case in cases}
    if len(expected) != len(cases) or set(rows) != set(expected):
        raise ValueError("Missing, extra or duplicate decision IDs")
    for key, row in rows.items():
        case = expected[key]
        p = np.asarray(row["probabilities"], dtype=float)
        z = np.asarray(row["logits"], dtype=float)
        if (row["task"] != task or row["gold"] != case["gold"] or row["qtype"] != case["qtype"]
                or row["k"] != len(case["options"]) or row.get("extra", {}) != case.get("extra", {})
                or p.shape != (row["k"],) or z.shape != p.shape or not np.isfinite(p).all()
                or not np.isfinite(z).all() or np.any(p < 0) or abs(float(p.sum()) - 1) > 1e-5
                or row["pred"] != int(p.argmax())):
            raise ValueError(f"Invalid aligned prediction: {task}/{key}")
        recomputed = np.exp(z - z.max())
        recomputed /= recomputed.sum()
        if not np.allclose(p, recomputed, rtol=1e-6, atol=1e-7):
            raise ValueError("Saved probabilities do not match T=1 logits")
    return float(np.mean([row["pred"] == row["gold"] for row in rows.values()]))


def main():
    config, trained = prerequisites()
    execution = json.loads((RESULTS / "evaluation_execution.json").read_text(encoding="utf-8"))
    if execution["status"] != "complete":
        raise ValueError("Complete the bounded evaluation before reporting")
    suite = ROOT / "runs/pilot/student-prep-20260927-v2/eval-suites.json"
    if digest(suite) != execution["suite_sha256"]:
        raise ValueError("Frozen suite changed")
    cases = json.loads(suite.read_text(encoding="utf-8"))
    confirmation_report()
    confirm = json.loads((RESULTS / "confirmation_summary.json").read_text(encoding="utf-8"))
    models, predictions = {}, {}
    for arm in [config["base_arm"], *config["arms"]]:
        measured = json.loads((RESULTS / f"eval_{arm}.json").read_text(encoding="utf-8"))
        if set(measured["tasks"]) != set(cases):
            raise ValueError("Incomplete public suite")
        manifests, rows_by_task = {}, {}
        for task, inputs in cases.items():
            path = RESULTS / "rows" / f"rows_{task}_tez-s-{arm}.jsonl"
            rows = read_rows(path)
            accuracy = validate_rows(inputs, rows, task)
            metrics = measured["tasks"][task]
            if metrics["n"] != len(rows) or abs(accuracy - metrics["t1"]["accuracy"]) > .000051:
                raise ValueError("Saved task metric differs from raw predictions")
            manifests[task] = {"sha256": digest(path), "rows": len(rows), "accuracy": accuracy}
            rows_by_task[task] = rows
        td = measured["tasks"]["typed_decisions"]
        tasks = measured["tasks"]
        models[arm] = {"rows_validated": sum(v["rows"] for v in manifests.values()), "row_files": manifests,
                       "eval_sha256": digest(RESULTS / f"eval_{arm}.json"), "weights": measured["weights_sha256"],
                       "typed_accuracy": manifests["typed_decisions"]["accuracy"], "typed_soft_accuracy": td["t1"].get("soft_accuracy"),
                       "typed_ece_raw": td["t1"]["ece"], "packing_delta_points": td.get("packing_delta_points"),
                       "massive11_macro": statistics.mean(v["accuracy"] for t, v in manifests.items() if t.startswith("massive_")),
                       "massive_km": manifests["massive_km"]["accuracy"],
                       "apps": {t[4:]: v["accuracy"] for t, v in manifests.items() if t.startswith("app_")},
                       "semif": {t: v["semif"]["mean_family_balanced_accuracy"] for t, v in tasks.items() if t.startswith("semif_")}}
        predictions[arm] = rows_by_task
    rules = "a-base-rules150"
    paired = {arm: paired_typed_delta(cases["typed_decisions"], predictions[rules]["typed_decisions"],
                                    rows["typed_decisions"]) for arm, rows in predictions.items() if arm != rules}
    competitor_path = ROOT / "results/pilot-improvement-20260927/comparison.json"
    competitors = json.loads(competitor_path.read_text(encoding="utf-8"))["systems"]
    control_parity = {}
    for task, base_rows in predictions[config["base_arm"]].items():
        control = predictions["a-base-replay150"][task]
        control_parity[task] = {
            "argmax_agreement": float(np.mean([r["pred"] == control[k]["pred"] for k, r in base_rows.items()])),
            "max_probability_delta": max(float(np.max(np.abs(np.asarray(r["probabilities"]) - control[k]["probabilities"])))
                                         for k, r in base_rows.items())}
    result = {"models": models, "rules_minus_reference_typed": paired, "control_parity": control_parity,
              "confirmation": confirm, "competitor_source_sha256": digest(competitor_path),
              "competitors": {k: competitors[k] for k in ("tez-12b", "laya-td", "decider08-native", "c-qwen08-30k", "kev08-native")},
              "scope": "Original public tests are exploratory secondary regressions; authored confirmation uses fresh domains with recurring logic families."}
    write_json(RESULTS / "summary.json", result)
    lines = ["# Rule-supervision continuation results", "", "All three frozen models completed the authored confirmation and original public suites.",
             "Raw probabilities, aligned labels and all 23,853 primary predictions were independently validated.",
             "", "## Fresh authored confirmation", "", "360 states, 1,800 decisions per model; two new domains, recurring logic families.",
             "Selection used separate validation data and completed before these labels were opened.", "",
             "| Model | Selected step | Accuracy |", "|---|---:|---:|"]
    for arm, model in confirm["models"].items():
        step = trained[arm]["selected_step"] if arm in trained else "starting model"
        lines.append(f"| {arm} | {step} | {model['accuracy']['all'] * 100:.2f}% |")
    for arm, stats in confirm["rules_minus_reference"].items():
        lo, hi = stats["ci95_points"]
        lines += ["", f"Rules minus {arm}: {stats['difference_points']:+.2f} points, state-bootstrap 95% interval [{lo:+.2f}, {hi:+.2f}]."]
    operations = [k for k in confirm["models"][rules]["accuracy"] if k != "all"]
    lines += ["", "| Confirmation subset | Starting model | Replay | Rule supervision |", "|---|---:|---:|---:|"]
    for key in operations:
        values = [confirm["models"][arm]["accuracy"][key] * 100 for arm in models]
        lines.append(f"| {key} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += ["", "## Matched public comparisons", "", "Same frozen decisions; raw T=1. Continuations were evaluated on the same laptop.",
              "Competitor quality comes from the earlier pinned H200 runs. Hardware timings are not pooled.", "",
              "| Model | Typed | MASSIVE 11 | Routing | Jailbreak | Toxicity | Typed raw ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
    shown = {**models, **result["competitors"]}
    for arm, model in shown.items():
        values = [model["typed_accuracy"], model["massive11_macro"],
                  *[model["apps"][task] for task in ("model_routing_domain", "guardrails_jailbreak", "moderation_toxicity")], model["typed_ece_raw"]]
        lines.append(f"| {arm} | " + " | ".join(f"{100 * v:.2f}" for v in values) + " |")
    lines += ["", "Laya-td trained on this benchmark's training split; these Tez models excluded the whole dataset.",
              "Other released models' training overlap is not established. These distinctions prevent a blanket ranking claim.", ""]
    for arm, stats in paired.items():
        lo, hi = stats["ci95_points"]
        lines.append(f"Rules minus {arm}, typed: {stats['accuracy_delta_points']:+.2f} points, exploratory state-bootstrap interval [{lo:+.2f}, {hi:+.2f}].")
    lines += ["", "## Public regressions and gate checks", "", "| Model | Typed soft accuracy | Packing delta, points | Khmer | SemIf authored | SemIf perturbations |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, model in models.items():
        lines.append(f"| {arm} | {100 * model['typed_soft_accuracy']:.2f} | {model['packing_delta_points']:+.2f} | {100 * model['massive_km']:.2f} | "
                     f"{100 * model['semif']['semif_authored144']:.2f} | {100 * model['semif']['semif_perturbations108']:.2f} |")
    lines += ["", "Every task, language, workflow, option-order check and probability file is retained in the evaluation reports and artifact bundle.",
              "Packing and reversed-order figures are evaluator summaries; primary rows, not those extra diagnostic predictions, were independently audited.",
              "The G1 typed target remains 56.3%, soft accuracy 46%, and MASSIVE 70%. Passing these alone would not pass G1.",
              "No valid target-laptop latency is claimed. The unchanged thermal governor was used for quality work.",
              "No model is promoted to the runtime by this report. Fresh confirmation cannot replace the original gates.",
              "", "One seed, 150 optimizer steps per arm; no convergence claim. The intervention changes content and exact-label supervision together.",
              "The replay model selected step zero. Its saved head archive has a different file hash; prediction parity is measured in summary.json.",
              "All cloud Pods remain terminated; this round used the laptop and added no RunPod charge.", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"public_rows_validated": sum(m["rows_validated"] for m in models.values()),
                      "typed": {a: m["typed_accuracy"] for a, m in models.items()}, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
