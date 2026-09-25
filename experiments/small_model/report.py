"""Pilot day 5: gather every arm's metrics into results/pilot/pilot.json and apply G1 (plan §8).

    python experiments/small_model/report.py

G1 (arm A-base, 30k), every line must pass:
  typed-decisions: accuracy >= 0.563 and soft accuracy >= 0.46 (80 % of the 12B's 0.704 / 0.575)
  Laya's held-out workflows: above the best Laya checkpoint on at least 2 of 3 (0.659 routing, 0.805 jailbreak,
    0.535 toxicity)
  MASSIVE, 11 languages: macro >= 0.70 and km (unseen) above 3x random (0.15)
  retention at 30,000 decisions at least 3 points above retention at 10,000
  §1b's protocol, as fast as laya-multilingual: <= 30 ms for one question, <= 2.8 ms per question at 50 (PyTorch)
  ECE-15 <= 0.15 as read
Decision: A passes -> M2 with A; C beats A by >= 5 points of retention and meets the latency line -> C; neither keeps
70 % on typed-decisions (0.493) -> stop the zero-shot student, D3 (per-schema students).
"""
from __future__ import annotations

import json

import numpy as np

from common import MASSIVE11, RESULTS, dump

G1 = {"td_acc": 0.563, "td_soft": 0.46, "laya_best": {"app_model_routing_domain": 0.659,
      "app_guardrails_jailbreak": 0.805, "app_moderation_toxicity": 0.535}, "massive_macro": 0.70, "km_min": 0.15,
      "retention_gain_pts": 3.0, "ms_1q": 30.0, "ms_per_q_50": 2.8, "ece": 0.15, "stop_td": 0.493}
TEACHER = {"td_acc": 0.704, "td_soft": 0.575}
G1_TASKS = ["typed_decisions", "app_model_routing_domain", "app_guardrails_jailbreak", "app_moderation_toxicity"] + \
    [f"massive_{l}" for l in MASSIVE11]


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arm_view(ev: dict, key: str = "t1") -> dict:
    t = ev["tasks"]
    v = {"td_acc": t["typed_decisions"][key]["accuracy"], "td_soft": t["typed_decisions"][key].get("soft_accuracy"),
         "td_retention": t["typed_decisions"].get("retention_vs_12b"),
         "td_one_q_acc": t["typed_decisions"].get("one_question_per_row", {}).get("t1", {}).get("accuracy"),
         "packing_delta_points": t["typed_decisions"].get("packing_delta_points"),
         "apps": {k: t[k][key]["accuracy"] for k in t if k.startswith("app_")},
         "massive": {l: t[f"massive_{l}"][key]["accuracy"] for l in MASSIVE11 if f"massive_{l}" in t},
         "semif": {k: t[k].get("semif", {}).get("mean_family_balanced_accuracy") for k in t if k.startswith("semif_")},
         "flip20_massive_mean": float(np.mean([t[f"massive_{l}"]["order_flip_20"] for l in MASSIVE11
                                               if "order_flip_20" in t.get(f"massive_{l}", {})] or [np.nan]))}
    v["massive_macro"] = float(np.mean(list(v["massive"].values())))
    rets = [t[k]["retention_vs_12b"] for k in G1_TASKS if k in t and t[k].get("retention_vs_12b") is not None]
    v["mean_retention_g1_suites"] = float(np.mean(rets))
    # "ECE-15 as read" (BENCHMARKS §4c's usage): the raw output, T = 1, mean over the G1 suites; the figure with the
    # student's validation-fitted per-type temperatures ("cal") is reported next to it
    v["ece_cal_mean"] = float(np.mean([t[k]["cal"]["ece"] for k in G1_TASKS if k in t]))
    v["ece_t1_mean"] = float(np.mean([t[k]["t1"]["ece"] for k in G1_TASKS if k in t]))
    v["ece_td_cal"] = t["typed_decisions"]["cal"]["ece"]
    v["ece_td_t1"] = t["typed_decisions"]["t1"]["ece"]
    return v


def main() -> None:
    arms = {}
    for name in ("a-base-30k", "a-base-10k", "a-small-30k", "c-qwen08-30k", "gliclass-x-base"):
        ev = load(f"eval_{name}.json")
        if ev:
            arms[name] = {"eval": arm_view(ev), "train": load(f"train_{name}.json")}
    lat = load("latency.json") or {}
    teacher_ref = load("eval_a-base-30k.json")
    rows = []
    a = arms.get("a-base-30k", {}).get("eval")
    a10 = arms.get("a-base-10k", {}).get("eval")
    c = arms.get("c-qwen08-30k", {}).get("eval")
    if a:
        rows.append({"criterion": "typed-decisions accuracy >= 0.563", "value": round(a["td_acc"], 4),
                     "pass": a["td_acc"] >= G1["td_acc"]})
        rows.append({"criterion": "typed-decisions soft accuracy >= 0.46", "value": round(a["td_soft"], 4),
                     "pass": a["td_soft"] >= G1["td_soft"]})
        wins = {k: (a["apps"][k], a["apps"][k] > thr) for k, thr in G1["laya_best"].items()}
        rows.append({"criterion": "above the best Laya checkpoint on >= 2 of 3 held-out workflows",
                     "value": {k: v[0] for k, v in wins.items()}, "pass": sum(v[1] for v in wins.values()) >= 2,
                     "wins": sum(v[1] for v in wins.values())})
        rows.append({"criterion": "MASSIVE 11 languages macro >= 0.70", "value": round(a["massive_macro"], 4),
                     "pass": a["massive_macro"] >= G1["massive_macro"]})
        rows.append({"criterion": "MASSIVE km (unseen) > 0.15 (3x random)", "value": a["massive"].get("km"),
                     "pass": (a["massive"].get("km") or 0) > G1["km_min"]})
        if a10:
            g = 100 * (a["mean_retention_g1_suites"] - a10["mean_retention_g1_suites"])
            gtd = 100 * (a["td_retention"] - a10["td_retention"])
            rows.append({"criterion": "retention at 30k >= retention at 10k + 3 points (mean over G1 suites)",
                         "value": {"30k": round(a["mean_retention_g1_suites"], 4), "10k": round(a10["mean_retention_g1_suites"], 4),
                                   "gain_points": round(g, 2), "typed_decisions_gain_points": round(gtd, 2)},
                         "pass": g >= G1["retention_gain_pts"]})
        lg = lat.get("a-base-30k_cuda")
        if lg:
            ms1 = lg["sizes"]["1"]["p50_ms"]
            m50 = lg["sizes"]["50"]["ms_per_question"]
            rows.append({"criterion": "<= 30 ms for one question (GPU, PyTorch)", "value": ms1, "pass": ms1 <= G1["ms_1q"]})
            rows.append({"criterion": "<= 2.8 ms per question at 50 (GPU, PyTorch)", "value": m50, "pass": m50 <= G1["ms_per_q_50"]})
        rows.append({"criterion": "ECE-15 <= 0.15 as read (T = 1, mean over the G1 suites)",
                     "value": {"as_read_t1": round(a["ece_t1_mean"], 4), "val_fitted_temps": round(a["ece_cal_mean"], 4),
                               "typed_decisions_t1": a["ece_td_t1"]},
                     "pass": a["ece_t1_mean"] <= G1["ece"]})
    all_pass = bool(rows) and all(r["pass"] for r in rows)
    decision, why = None, []
    if a:
        c_better = bool(c) and (100 * (c["mean_retention_g1_suites"] - a["mean_retention_g1_suites"]) >= 5)
        lc = lat.get("c-qwen08-30k_cuda")
        c_fast = bool(lc) and lc["sizes"]["50"]["ms_per_question"] <= 10 and lc["sizes"]["1"]["p50_ms"] <= G1["ms_1q"]
        best_td = max(a["td_acc"], c["td_acc"] if c else 0)
        if all_pass:
            decision = "continue with A (go to M2)"
        elif c_better and c_fast:
            decision = "switch to C"
        elif best_td < G1["stop_td"]:
            decision = "stop the zero-shot student; build per-schema students (D3)"
        else:
            decision = "A fails G1 without meeting a stop rule: see the recommendation"
        why = {"all_g1_pass": all_pass, "c_retention_minus_a_points": round(100 * (c["mean_retention_g1_suites"] - a["mean_retention_g1_suites"]), 2) if c else None,
               "c_meets_latency": c_fast, "best_typed_decisions_acc": round(best_td, 4)}
    dump(RESULTS / "pilot.json", {"title": "Small-model pilot (week 1): every arm, same rows as the 12B and Laya",
                                   "g1_thresholds": G1, "teacher_reference": TEACHER, "g1_arm": "a-base-30k",
                                   "g1": rows, "g1_all_pass": all_pass, "decision": decision, "decision_inputs": why,
                                   "arms": arms, "latency": {k: {kk: v[kk] for kk in ("device", "threads", "sizes", "load_s", "peak_vram_gb", "contaminated")}
                                                             for k, v in lat.items()}})
    for r in rows:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['criterion']}: {r['value']}")
    print("decision:", decision)


if __name__ == "__main__":
    main()
