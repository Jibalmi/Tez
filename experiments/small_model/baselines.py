"""The baselines on the pilot's rows, from rows already in the repo (no model is run).

    python experiments/small_model/baselines.py        # -> results/pilot/baselines.json

Systems: the 12B zero-shot (Tez letters at T = 1: results/h2h/rows_*_tez.jsonl, results/vs_laya/apps_rows/*_tez.jsonl,
results/authored144|perturb108_gemma4-12b-q8_0.jsonl) and Laya's three checkpoints (laya, laya-multilingual,
laya-typed-decisions) where their rows exist. Same metric code as the students (bench_h2h.summarise; SemIf's own
evaluator).
"""
from __future__ import annotations

import numpy as np

from common import MASSIVE11, RESULTS, dump
from eval_arms import load_ref_rows, semif_score, suites, summarise


def main() -> None:
    S = suites()
    out = {}
    for task, cases in S.items():
        refs = load_ref_rows(task)
        ids = [c["id"] for c in cases]
        out[task] = {}
        for sysname, rows in refs.items():
            recs = []
            for c in cases:
                r = rows[c["id"]]
                p = np.asarray(r["probabilities"], float)
                recs.append({"id": c["id"], "gold": c["gold"], "probabilities": (p / p.sum()).tolist(),
                             "logits": r.get("logits") or r.get("option_logprobs") or np.log(np.clip(p, 1e-12, None)).tolist(),
                             "ms": r.get("ms", 0.0) or 0.0, "extra": c.get("extra", {}), "qtype": c["qtype"]})
            s = summarise(recs)
            if task.startswith("semif_"):
                s["semif"] = semif_score(task, recs)
            out[task][sysname] = s
        print(task, {k: v.get("accuracy") for k, v in out[task].items()}, flush=True)
    macro = {}
    for sysname in ("tez-12b", "laya-en", "laya-ml", "laya-td"):
        accs = [out[f"massive_{l}"][sysname]["accuracy"] for l in MASSIVE11 if sysname in out[f"massive_{l}"]]
        if len(accs) == len(MASSIVE11):
            macro[sysname] = round(float(np.mean(accs)), 4)
    dump(RESULTS / "baselines.json", {"title": "Baselines on the pilot's rows (existing rows, same metric code)",
                                      "massive11_macro": macro, "tasks": out})
    print("MASSIVE-11 macro:", macro)


if __name__ == "__main__":
    main()
