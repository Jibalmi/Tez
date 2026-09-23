"""Collapsing decision bound for streaming voice commands (drift-diffusion idea, on existing logs).

Today's commit rule (stream_policy.py) is a fixed confidence threshold tau plus a per-class
stability count. A drift-diffusion observer instead uses a bound that shrinks with elapsed time:
demand near-certainty on the first word, accept less as the utterance goes on. Here the bound is
    thr(k) = floor + (ceil - floor) * exp(-(k - 1) / T)            (k = words heard so far)
applied to the prefix posterior pmax, with the same class-aware structure (OPEN fires early, slot
plays open their app, DEFERRED actions wait for the end, 'none' never commits).

Everything runs on the logged per-prefix posteriors (no new model calls). Protocol: two folds split
by utterance; on each fold pick, for both families, the policy with the earliest first action among
those whose harmful + out-of-scope false actions do not exceed the current default policy's on that
fold; report the pick on the OTHER fold.

  py experiments/voice_ddm.py --stream results/voicefast_partial_last_gemma4-12b-q8_0_stream.jsonl --full results/voicefast_partial_last_gemma4-12b-q8_0.jsonl --out results/voice_ddm_partial_last.json
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("sp", ROOT / "experiments" / "stream_policy.py")
sp = importlib.util.module_from_spec(_s); _s.loader.exec_module(sp)  # type: ignore[union-attr]


def executed(traj, thr_fn, need_fn):
    """stream_policy.executed_sequence with a pluggable threshold thr_fn(k) and stability need_fn(action)."""
    done, run = [], []
    for t in traj:
        a, p = t["pred"], t["pmax"]
        run = run + [a] if (run and run[-1] == a) else [a]
        if a == "none" or p < thr_fn(t["k"]) or a in sp.DEFERRED:
            continue
        if a in sp.SLOT_PLAY:
            anc = sp.REFINES[a]
            if len(run) >= need_fn(a) and anc not in [x for _, x in done]:
                done.append((t["k"], anc))
            continue
        if len(run) < need_fn(a) or (done and done[-1][1] == a):
            continue
        done.append((t["k"], a))
    final = traj[-1]["pred"]
    if final != "none" and (not done or done[-1][1] != final):
        done.append((traj[-1]["k"], final))
    return done


def score(rows, F, thr_fn, need_fn):
    harm = oos = 0; first = []; before = 0; nact = 0; noos = 0
    for s in rows:
        seq = executed(s["trajectory"], thr_fn, need_fn)
        if s["gold"] == "none":
            noos += 1; oos += bool(seq); continue
        nact += 1; f = F[s["id"]]
        acc = {s["gold"]} | ({f["then"]} if f.get("then") else set()) | ({f["alt"]} if f.get("alt") else set())
        harm += any(not sp.consistent(a, acc) for _, a in seq)
        if seq:
            first.append(seq[0][0]); before += s["human_commit_word"] > 0 and seq[0][0] <= s["human_commit_word"]
    return dict(harmful=harm, oos_false=oos, errors=harm + oos, actionable=nact, oos_rows=noos,
                mean_first_word=statistics.mean(first) if first else None, at_or_before_human=before / max(1, nact))


def fixed_family():
    for tau, ms, ps, ss in itertools.product((0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999), (1, 2, 3), (1, 2, 3), (1, 2, 3)):
        yield dict(kind="fixed", tau=tau, media_stable=ms, play_stable=ps, slot_stable=ss), \
            (lambda k, tau=tau: tau), (lambda a, ms=ms, ps=ps, ss=ss: 1 if a in sp.OPEN else ms if a in sp.MEDIA else ss if a in sp.SLOT_PLAY else ps)


def collapsing_family():
    for ceil, floor, T, stab in itertools.product((0.99, 0.999, 0.9999, 0.99999), (0.5, 0.7, 0.8, 0.9, 0.95), (0.5, 1, 2, 3, 5), (1, 2)):
        if floor >= ceil:
            continue
        yield dict(kind="collapsing", ceil=ceil, floor=floor, T=T, stability=stab), \
            (lambda k, c=ceil, f=floor, T=T: f + (c - f) * np.exp(-(k - 1) / T)), \
            (lambda a, stab=stab: 1 if a in sp.OPEN else stab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = sp.load(args.stream); F = {r["id"]: r for r in sp.load(args.full)}
    rng = np.random.default_rng(0); ids = sorted({s["id"] for s in S}); rng.shuffle(ids)
    fold_of = {i: n % 2 for n, i in enumerate(ids)}
    folds = [[s for s in S if fold_of[s["id"]] == f] for f in (0, 1)]
    default = (lambda k: 0.9), (lambda a: 1 if a in sp.OPEN else 2)
    res = {"default_policy": {"all": score(S, F, *default)}, "folds": []}
    for f in (0, 1):
        sel, ev = folds[f], folds[1 - f]
        base = score(sel, F, *default)
        out = {"default_on_eval": score(ev, F, *default)}
        for fam, gen in (("fixed", fixed_family), ("collapsing", collapsing_family)):
            best = None
            for cfg, thr, need in gen():
                r = score(sel, F, thr, need)
                if r["errors"] <= base["errors"] and r["mean_first_word"] is not None:
                    key = (r["mean_first_word"], r["errors"])
                    if best is None or key < best[0]:
                        best = (key, cfg, thr, need)
            out[fam] = dict(config=best[1], selection_fold=score(sel, F, best[2], best[3]), eval_fold=score(ev, F, best[2], best[3]))
        res["folds"].append(out)
        print(f"fold {f}: default eval {out['default_on_eval']}", flush=True)
        for fam in ("fixed", "collapsing"):
            print(f"   {fam:10s} {out[fam]['config']}\n              eval {out[fam]['eval_fold']}", flush=True)
    agg = {}
    for key in ("default_on_eval", "fixed", "collapsing"):
        rows = [fo[key] if key == "default_on_eval" else fo[key]["eval_fold"] for fo in res["folds"]]
        agg[key] = dict(harmful=sum(r["harmful"] for r in rows), oos_false=sum(r["oos_false"] for r in rows),
                        actionable=sum(r["actionable"] for r in rows), oos_rows=sum(r["oos_rows"] for r in rows),
                        mean_first_word=float(np.mean([r["mean_first_word"] for r in rows])),
                        at_or_before_human=float(np.mean([r["at_or_before_human"] for r in rows])))
    res["held_out_summary"] = agg
    print(json.dumps(agg, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
