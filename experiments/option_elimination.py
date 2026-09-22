"""Option elimination (process-of-elimination, PoE 2310.15575): score all options once, drop the ones the
model already rules out, re-score the survivors in a second cached pass, answer from the second pass.

Measured on the head-to-head rows (bench_h2h.build_task, same seed): single pass vs elimination with
three keep-rules (top-m by probability, above-mean log-prob, cumulative mass 0.95). Tasks with <= 26
options only (the tournament already eliminates for Banking77). Reports accuracy, ECE-15, mean
confidence, flip rate vs single pass, calls per decision, ms per decision.

  py experiments/option_elimination.py --tasks massive:en,emotion,ag_news,typed_decisions --out results/option_elimination_gemma4-12b-q8_0.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("bench_h2h", ROOT / "experiments" / "bench_h2h.py")
h2h = importlib.util.module_from_spec(_spec); sys.modules["bench_h2h"] = h2h; _spec.loader.exec_module(h2h)  # type: ignore[union-attr]
SERVER = os.environ.get("TEZ_SERVER", "http://127.0.0.1:8091")

RULES = {
    "top4": lambda p, z: [int(i) for i in np.argsort(-p)[:4]],
    "top8": lambda p, z: [int(i) for i in np.argsort(-p)[:8]],
    "above_mean_logp": lambda p, z: [int(i) for i in np.where(z >= z[np.isfinite(z)].mean())[0]],
    "mass95": lambda p, z: [int(i) for i in np.argsort(-p)[: int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.95) + 1)]],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="massive:en,emotion,ag_news,typed_decisions")
    ap.add_argument("--n", type=int, default=0, help="rows per task (0 = the head-to-head sizes)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for task in args.tasks.split(","):
        n = args.n or (100 if ":" in task else 400)
        cases = h2h.build_task(task, n)
        if task == "typed_decisions":
            cases = [c for c in cases if c["qtype"] == "choice" and len(c["options"]) >= 3]
        cases = [c for c in cases if len(c["options"]) <= 26]
        rows = []
        t0 = time.perf_counter()
        for i, c in enumerate(cases):
            k = len(c["options"])
            t1 = time.perf_counter()
            p1, z1, _ = h2h.tez_score_letters(SERVER, h2h.tez_prompt(c, c["options"]), k)
            ms1 = (time.perf_counter() - t1) * 1000
            row = dict(id=c["id"], gold=int(c["gold"]), k=k, single=dict(pred=int(np.argmax(p1)), conf=float(p1.max()), ms=ms1, calls=1))
            for name, rule in RULES.items():
                keep = sorted(rule(p1, z1))
                if c["gold"] not in keep:
                    pass  # counted as a miss below; elimination can throw the gold away
                if len(keep) >= k or len(keep) < 2:
                    row[name] = dict(pred=int(np.argmax(p1)), conf=float(p1.max()), ms=ms1, calls=1, kept=len(keep), gold_kept=bool(c["gold"] in keep))
                    continue
                t2 = time.perf_counter()
                opts = [c["options"][j] for j in keep]
                p2, z2, _ = h2h.tez_score_letters(SERVER, h2h.tez_prompt(c, opts), len(opts))
                ms2 = (time.perf_counter() - t2) * 1000
                row[name] = dict(pred=keep[int(np.argmax(p2))], conf=float(p2.max()), ms=ms1 + ms2, calls=2, kept=len(keep), gold_kept=bool(c["gold"] in keep))
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {task} {i + 1}/{len(cases)} single acc {np.mean([r['single']['pred'] == r['gold'] for r in rows]):.3f}", flush=True)
        summ = {}
        for name in ["single"] + list(RULES):
            g = np.array([r["gold"] for r in rows]); pr = np.array([r[name]["pred"] for r in rows]); cf = np.array([r[name]["conf"] for r in rows])
            summ[name] = dict(acc=float((pr == g).mean()), ece15=float(h2h.ece15(cf, (pr == g).astype(float))), mean_conf=float(cf.mean()),
                              flip_vs_single=float(np.mean([r[name]["pred"] != r["single"]["pred"] for r in rows])),
                              calls=float(np.mean([r[name]["calls"] for r in rows])), ms_p50=float(np.median([r[name]["ms"] for r in rows])),
                              gold_kept=float(np.mean([r[name].get("gold_kept", True) for r in rows])), kept_mean=float(np.mean([r[name].get("kept", r["k"]) for r in rows])))
        res[task] = dict(n=len(rows), k_mean=float(np.mean([r["k"] for r in rows])), seconds=time.perf_counter() - t0, summary=summ, rows=rows)
        print(task, {k: round(v["acc"], 3) for k, v in summ.items()}, "calls", {k: round(v["calls"], 2) for k, v in summ.items()}, flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
