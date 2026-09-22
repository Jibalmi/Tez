"""Batch Calibration (Zhou et al., arXiv:2309.17249) applied offline to saved one-pass outputs.

Where Zhao et al.'s contextual calibration estimates the label prior from a content-free input
(and failed here because the model correctly answers "insufficient" to empty evidence), Batch
Calibration estimates it from the mean log-probability of each option over a batch of REAL,
unlabeled inputs, then subtracts it: z' = z - mean_batch(z). Zero-shot, inference-only, no extra
forward passes. We evaluate it transductively (whole set) and out-of-fold (estimate on one half,
apply to the other) so the gain is not optimistic.

Groups: options must be comparable within a group, so the prior is estimated per option-set
signature (the tuple of option ids / letters). Reports accuracy and NLL before/after.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def evaluate(rows, key_fn, gold_fn, logit_fn):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[key_fn(r)].append(i)
    raw_acc, raw_nll, bc_acc, bc_nll, oof_acc, oof_nll = [], [], [], [], [], []
    for key, idx in groups.items():
        Z = np.array([logit_fn(rows[i]) for i in idx]); G = [gold_fn(rows[i]) for i in idx]
        # log-softmax per row first (BC works on log-probabilities)
        LP = Z - np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1, keepdims=True)) - Z.max(1, keepdims=True)
        mu = LP.mean(0)
        for j, g in enumerate(G):
            p = softmax(LP[j]); q = softmax(LP[j] - mu)
            raw_acc.append(int(np.argmax(p)) == g); raw_nll.append(-math.log(max(p[g], 1e-12)))
            bc_acc.append(int(np.argmax(q)) == g); bc_nll.append(-math.log(max(q[g], 1e-12)))
        if len(idx) >= 8:
            for fold in (0, 1):
                fit = [j for j in range(len(idx)) if j % 2 == fold]; ev = [j for j in range(len(idx)) if j % 2 != fold]
                mu_f = LP[fit].mean(0)
                for j in ev:
                    q = softmax(LP[j] - mu_f); oof_acc.append(int(np.argmax(q)) == G[j]); oof_nll.append(-math.log(max(q[G[j]], 1e-12)))
        else:
            for j, g in enumerate(G):
                p = softmax(LP[j]); oof_acc.append(int(np.argmax(p)) == g); oof_nll.append(-math.log(max(p[g], 1e-12)))
    return dict(n=len(rows), groups=len(groups), raw_acc=float(np.mean(raw_acc)), raw_nll=float(np.mean(raw_nll)),
                bc_acc=float(np.mean(bc_acc)), bc_nll=float(np.mean(bc_nll)), bc_oof_acc=float(np.mean(oof_acc)), bc_oof_nll=float(np.mean(oof_nll)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/batch_calibration.json")
    args = ap.parse_args()
    out = {}
    # SemIf authored144 (option ids differ per row; group by option-id tuple => mostly singleton groups, so also group by family)
    for tag in ["gemma4-12b-q8_0", "gemma3-4b-q4km", "qwen35-4b-bf16-hf"]:
        p = Path(f"results/authored144_{tag}.jsonl")
        if p.exists():
            R = load(p)
            out[f"authored144/{tag}/by_family"] = evaluate(R, lambda r: r["family"], lambda r: r["gold"], lambda r: r["option_logprobs"])
            out[f"authored144/{tag}/by_optionset"] = evaluate(R, lambda r: tuple(r["option_ids"]), lambda r: r["gold"], lambda r: r["option_logprobs"])
    # head-to-head rows: group by task (same option set within a task) except typed-decisions (group by question id)
    for f in sorted(Path("results/h2h").glob("rows_*_tez.jsonl")):
        R = load(f)
        if not R:
            continue
        task = R[0]["task"]
        if task == "typed_decisions":
            key = lambda r: (r["workflow"], r["id"].rsplit("-", 1)[1], r["k"])  # noqa: E731
        elif task.startswith("massive"):
            key = lambda r: r["k"]  # noqa: E731  (options are sampled per row; BC by position only -- a position-bias estimate)
        else:
            key = lambda r: r["k"]  # noqa: E731
        out[f"h2h/{task}"] = evaluate(R, key, lambda r: r["gold"], lambda r: r["logits"])
    for k, v in out.items():
        print(f"{k:45s} n={v['n']:5d} g={v['groups']:4d} acc {v['raw_acc']:.3f} -> BC {v['bc_acc']:.3f} (oof {v['bc_oof_acc']:.3f}) | nll {v['raw_nll']:.3f} -> {v['bc_nll']:.3f} (oof {v['bc_oof_nll']:.3f})")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
