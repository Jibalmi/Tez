"""Small-to-large cascade, offline, from existing per-row outputs on SemIf's authored144.

A FrugalGPT-style cascade: answer with the cheap model (Gemma 3 4B, ~40 ms) when its evidence is
strong, otherwise escalate to the 12B (~110 ms). Two gates are compared:
  * max-probability of the 4B (raw)             -- the naive gate
  * agreement of the 4B across two orderings      -- permutation disagreement as the signal
    (and the mean of the two as the 4B's answer)
Sweeps the gate threshold and reports accuracy vs mean latency (ms) and escalation rate, next to
the two single-model points. Latencies are the measured p50s of each server run (manifest).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}


def by_option(r):
    return dict(zip(r["option_ids"], r["probabilities"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/authored144_gemma3-4b-q4km.jsonl")
    ap.add_argument("--small-rev", default="results/authored144_rev_gemma3-4b-q4km.jsonl")
    ap.add_argument("--large", default="results/authored144_gemma4-12b-q8_0.jsonl")
    ap.add_argument("--ms-small", type=float, default=40.4)
    ap.add_argument("--ms-large", type=float, default=110.1)
    ap.add_argument("--out", default="results/cascade_authored144.json")
    args = ap.parse_args()
    S, SR, L = load(args.small), load(args.small_rev), load(args.large)
    ids = [i for i in L if i in S and i in SR]
    gold = {i: L[i]["option_ids"][L[i]["gold"]] for i in ids}

    def argmax_id(pm):
        return max(pm, key=pm.get)

    small_pred = {i: argmax_id(by_option(S[i])) for i in ids}
    small_pmax = {i: max(S[i]["probabilities"]) for i in ids}
    small_rev_pred = {i: argmax_id(by_option(SR[i])) for i in ids}
    small_avg = {i: {o: (by_option(S[i])[o] + by_option(SR[i])[o]) / 2 for o in S[i]["option_ids"]} for i in ids}
    small_avg_pred = {i: argmax_id(small_avg[i]) for i in ids}
    small_avg_pmax = {i: max(small_avg[i].values()) for i in ids}
    large_pred = {i: argmax_id(by_option(L[i])) for i in ids}
    n = len(ids)
    res = dict(n=n, small_only=dict(acc=np.mean([small_pred[i] == gold[i] for i in ids]), ms=args.ms_small),
               small_perm2_only=dict(acc=np.mean([small_avg_pred[i] == gold[i] for i in ids]), ms=2 * args.ms_small),
               large_only=dict(acc=np.mean([large_pred[i] == gold[i] for i in ids]), ms=args.ms_large), sweeps={})
    # gate A: raw pmax threshold
    sw = []
    for tau in np.linspace(0.5, 1.0, 26):
        esc = [i for i in ids if small_pmax[i] < tau]
        pred = {i: (large_pred[i] if i in esc else small_pred[i]) for i in ids}
        sw.append(dict(tau=float(tau), acc=float(np.mean([pred[i] == gold[i] for i in ids])), escalate=len(esc) / n,
                       ms=args.ms_small + args.ms_large * len(esc) / n))
    res["sweeps"]["pmax_gate"] = sw
    # gate B: permutation agreement (+ pmax of the averaged distribution)
    sw = []
    for tau in np.linspace(0.5, 1.0, 26):
        esc = [i for i in ids if (small_pred[i] != small_rev_pred[i]) or small_avg_pmax[i] < tau]
        pred = {i: (large_pred[i] if i in esc else small_avg_pred[i]) for i in ids}
        sw.append(dict(tau=float(tau), acc=float(np.mean([pred[i] == gold[i] for i in ids])), escalate=len(esc) / n,
                       ms=2 * args.ms_small + args.ms_large * len(esc) / n))
    res["sweeps"]["perm_agreement_gate"] = sw
    # gate C: agreement only (no threshold)
    esc = [i for i in ids if small_pred[i] != small_rev_pred[i]]
    pred = {i: (large_pred[i] if i in esc else small_avg_pred[i]) for i in ids}
    res["agreement_only"] = dict(acc=float(np.mean([pred[i] == gold[i] for i in ids])), escalate=len(esc) / n, ms=2 * args.ms_small + args.ms_large * len(esc) / n)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "sweeps"}, indent=2, default=float))
    for name, sw in res["sweeps"].items():
        print(name)
        for s in sw[::5]:
            print(f"  tau={s['tau']:.2f} acc={s['acc']:.3f} escalate={s['escalate']:.2f} ms={s['ms']:.0f}")


if __name__ == "__main__":
    main()
