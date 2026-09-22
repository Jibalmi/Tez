"""Post-hoc calibration and debiasing analysis on direct-readout predictions.

Inputs are run_direct.py outputs for the same rows:
  --pred     original option order (required)
  --cf       content-free run (state replaced by "N/A")  -> Zhao et al. contextual calibration
  --rev      reversed-option run                          -> permutation averaging (PriDe-lite)

For each method we report proper scores (NLL, Brier), binned ECE (secondary), accuracy,
a risk-coverage table (accuracy when the least-confident x% abstain), AURC, and the AUROC of
max-probability as an error detector. Temperature is fitted out-of-fold with folds split by
group_id so meaning-preserving variants never leak into their own calibration fold (as SemIf does).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def logsoftmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())


def softmax(z):
    return np.exp(logsoftmax(z))


# ----------------------------------------------------------------------------- scoring
def scores(P, Y):
    """P: list of prob vectors, Y: gold indices."""
    nll, brier, conf, corr = [], [], [], []
    for p, y in zip(P, Y):
        p = np.asarray(p)
        nll.append(-math.log(max(p[y], 1e-12)))
        oh = np.zeros_like(p)
        oh[y] = 1
        brier.append(float(((p - oh) ** 2).sum()))
        conf.append(float(p.max()))
        corr.append(1.0 if int(p.argmax()) == y else 0.0)
    conf, corr = np.asarray(conf), np.asarray(corr)
    return dict(
        n=len(Y),
        accuracy=float(corr.mean()),
        nll=float(np.mean(nll)),
        brier=float(np.mean(brier)),
        mean_confidence=float(conf.mean()),
        ece15=ece(conf, corr, 15),
        auroc_error_detection=auroc(conf, corr),
        risk_coverage=risk_coverage(conf, corr),
        aurc=aurc(conf, corr),
    )


def ece(conf, corr, bins):
    e = 0.0
    n = len(conf)
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        m = (conf >= lo) & (conf < hi) if k < bins - 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def auroc(conf, corr):
    """P(conf_correct > conf_wrong): 0.5 = confidence carries no error signal."""
    pos, neg = conf[corr == 1], conf[corr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def risk_coverage(conf, corr):
    order = np.argsort(-conf)  # most confident first
    out = {}
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        k = max(1, int(round(cov * len(conf))))
        sel = order[:k]
        out[f"cov{int(cov*100)}"] = dict(accuracy=float(corr[sel].mean()), threshold=float(conf[sel].min()))
    return out


def aurc(conf, corr):
    order = np.argsort(-conf)
    risks = []
    wrong = 0
    for i, idx in enumerate(order, 1):
        wrong += 1 - corr[idx]
        risks.append(wrong / i)
    return float(np.mean(risks))


# ----------------------------------------------------------------------------- methods
def fit_temperature(logits_list, Y, grid=np.linspace(0.2, 10.0, 491)):
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        nll = 0.0
        for z, y in zip(logits_list, Y):
            lp = logsoftmax(np.asarray(z) / T)
            nll -= lp[y]
        nll /= len(Y)
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def oof_temperature(rows, k=5, seed=291607):
    """Out-of-fold temperature scaling with folds split by group_id."""
    groups = sorted({r["group_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    P = [None] * len(rows)
    Ts = []
    for f in range(k):
        train = [i for i, r in enumerate(rows) if fold_of[r["group_id"]] != f]
        test = [i for i, r in enumerate(rows) if fold_of[r["group_id"]] == f]
        T = fit_temperature([rows[i]["option_logprobs"] for i in train], [rows[i]["gold"] for i in train])
        Ts.append(T)
        for i in test:
            P[i] = softmax(np.asarray(rows[i]["option_logprobs"]) / T).tolist()
    return P, Ts


def contextual_calibration(rows, cf_rows):
    """Zhao et al. 2021: divide by the content-free prior, renormalise. One extra pass per row here
    (per template in production, since the prior depends only on question+options)."""
    cf_by_id = {r["id"]: r for r in cf_rows}
    P = []
    for r in rows:
        p = np.asarray(r["probabilities"])
        q = np.asarray(cf_by_id[r["id"]]["probabilities"]) + 1e-9
        w = p / q
        P.append((w / w.sum()).tolist())
    return P


def permutation_average(rows, rev_rows):
    """Average probabilities BY OPTION ID across original and reversed orderings (PriDe-lite)."""
    rev_by_id = {r["id"]: r for r in rev_rows}
    P = []
    for r in rows:
        rr = rev_by_id[r["id"]]
        pm = dict(zip(rr["option_ids"], rr["probabilities"]))
        avg = [(p + pm[oid]) / 2.0 for oid, p in zip(r["option_ids"], r["probabilities"])]
        P.append(avg)
    return P


def to_logits(P):
    return [np.log(np.asarray(p) + 1e-12).tolist() for p in P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cf")
    ap.add_argument("--rev")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = read(args.pred)
    Y = [r["gold"] for r in rows]
    results = {}

    results["raw"] = scores([r["probabilities"] for r in rows], Y)

    P_T, Ts = oof_temperature(rows)
    results["temperature_oof"] = scores(P_T, Y) | {"fold_temperatures": [round(t, 2) for t in Ts]}

    if args.cf:
        cf = read(args.cf)
        P_cc = contextual_calibration(rows, cf)
        results["contextual_calibration"] = scores(P_cc, Y)
        # contextual calibration then temperature (temperature fitted OOF on the debiased logits)
        rows_cc = [dict(r, option_logprobs=z) for r, z in zip(rows, to_logits(P_cc))]
        P_ccT, Ts2 = oof_temperature(rows_cc)
        results["contextual_calibration+temperature_oof"] = scores(P_ccT, Y) | {"fold_temperatures": [round(t, 2) for t in Ts2]}

    if args.rev:
        rev = read(args.rev)
        P_pa = permutation_average(rows, rev)
        results["permutation_average"] = scores(P_pa, Y)
        rows_pa = [dict(r, option_logprobs=z) for r, z in zip(rows, to_logits(P_pa))]
        P_paT, Ts3 = oof_temperature(rows_pa)
        results["permutation_average+temperature_oof"] = scores(P_paT, Y) | {"fold_temperatures": [round(t, 2) for t in Ts3]}
        # order-sensitivity diagnostics on the full set
        rev_by_id = {r["id"]: r for r in rev}
        flips = 0
        tv = []
        for r in rows:
            rr = rev_by_id[r["id"]]
            a = r["option_ids"][int(np.argmax(r["probabilities"]))]
            b = rr["option_ids"][int(np.argmax(rr["probabilities"]))]
            flips += a != b
            pm = dict(zip(rr["option_ids"], rr["probabilities"]))
            tv.append(sum(abs(p - pm[o]) for o, p in zip(r["option_ids"], r["probabilities"])) / 2)
        pos_pred = defaultdict(int)
        for r in rows:
            pos_pred[int(np.argmax(r["probabilities"]))] += 1
        pos_gold = defaultdict(int)
        for y in Y:
            pos_gold[y] += 1
        results["order_diagnostics"] = dict(
            full_set_flip_rate=flips / len(rows),
            mean_tv_distance=float(np.mean(tv)),
            predicted_position_counts=dict(sorted(pos_pred.items())),
            gold_position_counts=dict(sorted(pos_gold.items())),
        )

    if args.cf and args.rev:
        P_both = contextual_calibration([dict(r, probabilities=p) for r, p in zip(rows, P_pa)], cf)
        results["permutation_average+contextual_calibration"] = scores(P_both, Y)
        rows_b = [dict(r, option_logprobs=z) for r, z in zip(rows, to_logits(P_both))]
        P_bT, Ts4 = oof_temperature(rows_b)
        results["all_three"] = scores(P_bT, Y) | {"fold_temperatures": [round(t, 2) for t in Ts4]}

    # compact table
    print(f"{'method':46s} {'acc':>6s} {'NLL':>7s} {'Brier':>7s} {'ECE15':>6s} {'AUROC':>6s} {'AURC':>6s}  acc@cov90 acc@cov80")
    for name, s in results.items():
        if "accuracy" not in s:
            continue
        rc = s["risk_coverage"]
        au = s["auroc_error_detection"]
        print(f"{name:46s} {s['accuracy']:6.3f} {s['nll']:7.3f} {s['brier']:7.3f} {s['ece15']:6.3f} {au if au is None else round(au,3)!s:>6s} {s['aurc']:6.3f}  {rc['cov90']['accuracy']:8.3f}  {rc['cov80']['accuracy']:8.3f}")
    if "order_diagnostics" in results:
        print("order:", json.dumps(results["order_diagnostics"]))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
