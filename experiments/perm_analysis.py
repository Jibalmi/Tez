"""Full-permutation and content-free analyses on run_direct.py outputs.

Requires, for one model tag, the authored144 runs for all 6 orderings of 3 options
(orig = 0,1,2; rev = 2,1,0; and perms 021, 102, 120, 201) plus content-free runs
with "N/A", "[MASK]" and "" placeholders.

Reports, all keyed by OPTION ID so orderings are comparable:
  * 2-permutation averaging (orig + rev)      -- what calibrate.py calls permutation_average
  * 6-permutation averaging (prob mean)        -- full permutation voting
  * 6-permutation geometric mean
  * PriDe with a full-permutation prior estimate: prior(letter) = mean over rows and orderings
    of P(letter); debiased p(content) ∝ p_orig(letter holding content) / prior(letter)
  * any-order agreement: fraction of rows where all 6 orderings share the argmax
  * Zhao et al. contextual calibration with the 3-placeholder AVERAGE prior, and damped
    variants p / q^alpha for alpha in {0.25, 0.5}
Each is scored on accuracy / NLL / Brier / ECE-15 (via calibrate.scores), and temperature is
fitted out-of-fold by group for the two headline estimators.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("calib", HERE / "calibrate.py")
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)  # type: ignore[union-attr]


def read(p):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}


def by_option(row):
    return dict(zip(row["option_ids"], row["probabilities"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", required=True, help="model tag, e.g. gemma4-12b-q8_0")
    ap.add_argument("--out")
    args = ap.parse_args()
    R = Path(args.results)
    T = args.tag

    orig = read(R / f"authored144_{T}.jsonl")
    perms = {
        "012": orig,
        "210": read(R / f"authored144_rev_{T}.jsonl"),
        "021": read(R / f"authored144_perm021_{T}.jsonl"),
        "102": read(R / f"authored144_perm102_{T}.jsonl"),
        "120": read(R / f"authored144_perm120_{T}.jsonl"),
        "201": read(R / f"authored144_perm201_{T}.jsonl"),
    }
    cfs = {
        "N/A": read(R / f"authored144_cf_{T}.jsonl"),
        "[MASK]": read(R / f"authored144_cfmask_{T}.jsonl"),
        "": read(R / f"authored144_cfempty_{T}.jsonl"),
    }
    ids = list(orig)
    Y = [orig[i]["gold"] for i in ids]
    opt_order = {i: orig[i]["option_ids"] for i in ids}  # report everything in the ORIGINAL option order

    def vec(i, pmap):
        return [pmap[o] for o in opt_order[i]]

    results = {}
    results["orig"] = calib.scores([orig[i]["probabilities"] for i in ids], Y)

    # 2-perm and 6-perm averaging by option id
    P2, P6, G6 = [], [], []
    agree = 0
    for i in ids:
        maps = [by_option(perms[k][i]) for k in perms]
        m2 = {o: (maps[0][o] + maps[1][o]) / 2 for o in opt_order[i]}
        m6 = {o: float(np.mean([m[o] for m in maps])) for o in opt_order[i]}
        g6 = {o: float(np.exp(np.mean([np.log(m[o] + 1e-12) for m in maps]))) for o in opt_order[i]}
        z = sum(g6.values())
        g6 = {o: v / z for o, v in g6.items()}
        P2.append(vec(i, m2)); P6.append(vec(i, m6)); G6.append(vec(i, g6))
        argmaxes = {max(m, key=m.get) for m in maps}
        agree += len(argmaxes) == 1
    results["perm2_mean"] = calib.scores(P2, Y)
    results["perm6_mean"] = calib.scores(P6, Y)
    results["perm6_geomean"] = calib.scores(G6, Y)
    results["any_order_agreement"] = agree / len(ids)

    # PriDe: letter prior from all orderings (letter = position index in each ordering)
    letter_mass = np.zeros(3)
    n = 0
    for k, runs in perms.items():
        for i in ids:
            letter_mass += np.asarray(runs[i]["probabilities"])
            n += 1
    prior = letter_mass / n
    results["letter_prior_A_B_C"] = prior.tolist()
    Ppride = []
    for i in ids:
        p = np.asarray(orig[i]["probabilities"]) / prior
        Ppride.append((p / p.sum()).tolist())
    results["pride_full_prior"] = calib.scores(Ppride, Y)

    # Zhao: 3-placeholder average prior, plus damped variants
    def cc(alpha, which):
        P = []
        for i in ids:
            q = np.mean([np.asarray(cfs[w][i]["probabilities"]) for w in which], axis=0) + 1e-9
            p = np.asarray(orig[i]["probabilities"]) / (q ** alpha)
            P.append((p / p.sum()).tolist())
        return P
    results["zhao_NA_only"] = calib.scores(cc(1.0, ["N/A"]), Y)
    results["zhao_3placeholder_avg"] = calib.scores(cc(1.0, ["N/A", "[MASK]", ""]), Y)
    results["zhao_3avg_damped_0.5"] = calib.scores(cc(0.5, ["N/A", "[MASK]", ""]), Y)
    results["zhao_3avg_damped_0.25"] = calib.scores(cc(0.25, ["N/A", "[MASK]", ""]), Y)
    cf_argmax = {w: sum(1 for i in ids if cfs[w][i]["option_ids"][int(np.argmax(cfs[w][i]["probabilities"]))] == "insufficient") for w in cfs}
    results["cf_picks_insufficient"] = cf_argmax

    # OOF temperature on top of the two headline estimators
    rows2 = [dict(orig[i], option_logprobs=np.log(np.asarray(p) + 1e-12).tolist()) for i, p in zip(ids, P6)]
    PT, Ts = calib.oof_temperature(rows2)
    results["perm6_mean+temperature_oof"] = calib.scores(PT, Y) | {"fold_temperatures": [round(t, 2) for t in Ts]}
    rowsP = [dict(orig[i], option_logprobs=np.log(np.asarray(p) + 1e-12).tolist()) for i, p in zip(ids, Ppride)]
    PT2, Ts2 = calib.oof_temperature(rowsP)
    results["pride+temperature_oof"] = calib.scores(PT2, Y) | {"fold_temperatures": [round(t, 2) for t in Ts2]}

    print(f"{'method':32s} {'acc':>6s} {'NLL':>7s} {'Brier':>7s} {'ECE15':>6s} {'AUROC':>6s}  acc@cov90")
    for k, s in results.items():
        if isinstance(s, dict) and "accuracy" in s:
            au = s["auroc_error_detection"]
            print(f"{k:32s} {s['accuracy']:6.3f} {s['nll']:7.3f} {s['brier']:7.3f} {s['ece15']:6.3f} {au if au is None else round(au,3)!s:>6s}  {s['risk_coverage']['cov90']['accuracy']:.3f}")
    print("any-order agreement:", round(results["any_order_agreement"], 4), "| letter prior A,B,C:", [round(x, 3) for x in prior], "| cf picks insufficient:", cf_argmax)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
