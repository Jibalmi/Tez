"""Conformal prediction sets on the typed-decisions head-to-head rows (results/h2h/rows_typed_decisions_<model>.jsonl).

Split-conformal with the score s = 1 - p(gold) (Angelopoulos & Bates 2021), Mondrian by question
type (choice / score / noul) so each primitive gets its own quantile. Calibrate on half the CASES
(not decisions -- all questions of a case stay together), evaluate on the other half, both folds.

Reports, for alpha in {0.10, 0.05}: empirical coverage, mean set size, the act / escalate / fallback
split (set size 1 / >1 / 0), accuracy of the "act" decisions, and the same after a per-type
temperature (fitted on the calibration fold), per model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def softmax_t(z, t):
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max()); return e / e.sum()


def fit_temperature(rows):
    best, bt = 1e9, 1.0
    for t in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
        nll = -np.mean([math.log(max(softmax_t(r["logits"], t)[r["gold"]], 1e-12)) for r in rows])
        if nll < best:
            best, bt = nll, float(t)
    return bt


def conformal(cal, ev, alpha, temps):
    """Mondrian split conformal by qtype. Returns metrics on ev."""
    q = {}
    for qt in {r["qtype"] for r in cal}:
        rs = [r for r in cal if r["qtype"] == qt]
        s = np.array([1 - softmax_t(r["logits"], temps[qt])[r["gold"]] for r in rs])
        n = len(s)
        k = min(n - 1, int(math.ceil((n + 1) * (1 - alpha))) - 1)
        q[qt] = float(np.sort(s)[k])
    cov, sizes, act_ok, n_act, n_esc, n_fb = 0, [], 0, 0, 0, 0
    for r in ev:
        p = softmax_t(r["logits"], temps[r["qtype"]])
        S = [i for i in range(len(p)) if 1 - p[i] <= q[r["qtype"]]]
        cov += r["gold"] in S; sizes.append(len(S))
        if len(S) == 1:
            n_act += 1; act_ok += S[0] == r["gold"]
        elif len(S) == 0:
            n_fb += 1
        else:
            n_esc += 1
    n = len(ev)
    return dict(coverage=cov / n, mean_set_size=float(np.mean(sizes)), act_rate=n_act / n, escalate_rate=n_esc / n, fallback_rate=n_fb / n,
                acc_when_acting=act_ok / n_act if n_act else None, quantiles=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/h2h")
    ap.add_argument("--models", default="tez,laya-en,laya-td")
    ap.add_argument("--out", default="results/conformal_typed_decisions.json")
    args = ap.parse_args()
    out = {}
    for m in args.models.split(","):
        p = Path(args.rows) / f"rows_typed_decisions_{m}.jsonl"
        if not p.exists():
            continue
        R = load(p)
        cases = sorted({r["id"].rsplit("-", 1)[0] for r in R})
        rng = np.random.default_rng(13); rng.shuffle(cases)
        half = set(cases[: len(cases) // 2])
        A = [r for r in R if r["id"].rsplit("-", 1)[0] in half]; B = [r for r in R if r["id"].rsplit("-", 1)[0] not in half]
        res = {}
        for alpha in (0.10, 0.05):
            for temp in (False, True):
                accs = []
                for cal, ev in ((A, B), (B, A)):
                    temps = {qt: (fit_temperature([r for r in cal if r["qtype"] == qt]) if temp else 1.0) for qt in {r["qtype"] for r in cal}}
                    accs.append(conformal(cal, ev, alpha, temps))
                agg = {k: float(np.mean([a[k] for a in accs if a[k] is not None])) for k in ("coverage", "mean_set_size", "act_rate", "escalate_rate", "fallback_rate", "acc_when_acting")}
                res[f"alpha={alpha} temp={temp}"] = agg
        acc = float(np.mean([r["pred"] == r["gold"] for r in R]))
        out[m] = dict(n=len(R), accuracy=acc, **res)
        print(f"== {m}  n={len(R)} acc={acc:.3f}")
        for k, v in res.items():
            print(f"  {k:22s} cov={v['coverage']:.3f} size={v['mean_set_size']:.2f} act={v['act_rate']:.2f} (acc {v['acc_when_acting']:.3f}) escalate={v['escalate_rate']:.2f} fallback={v['fallback_rate']:.2f}")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
