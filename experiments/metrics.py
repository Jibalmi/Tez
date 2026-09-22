"""Metrics for direct-readout prediction files produced by run_direct.py.

Reports SemIf's headline metric (mean-family balanced accuracy) so numbers are comparable
to their published table, plus proper scoring rules (NLL, Brier) as the primary calibration
evidence, binned ECE only as a secondary figure, formatting-leakage counts, latency, and
an optional paired stability comparison for the perturbation set (joined on provenance.base_id).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read(p: str) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def basic(rows: list[dict]) -> dict:
    """Classes are OPTION IDS, not gold positions: options are shuffled per row, so keying
    recall on position index (an earlier bug caught in review) inflates balanced accuracy.
    This matches SemIf's evaluate.py, which keys on gold_id."""
    if not rows:
        return dict(n=0, accuracy=None, balanced_accuracy=None, macro_f1=None)
    gold_ids = [r["option_ids"][r["gold"]] for r in rows]
    pred_ids = [r["option_ids"][argmax(r["probabilities"])] for r in rows]
    classes = sorted(set(gold_ids))
    tp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    correct = 0
    for g, p in zip(gold_ids, pred_ids):
        if p == g:
            correct += 1
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] = fp.get(p, 0) + 1
    recalls = [tp[c] / (tp[c] + fn[c]) for c in classes if (tp[c] + fn[c])]
    f1 = []
    for c in classes:
        p = tp[c] / (tp[c] + fp.get(c, 0)) if (tp[c] + fp.get(c, 0)) else 0.0
        rr = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1.append(2 * p * rr / (p + rr) if (p + rr) else 0.0)
    return dict(
        n=len(rows),
        accuracy=correct / len(rows),
        balanced_accuracy=sum(recalls) / len(recalls),
        macro_f1=sum(f1) / len(f1),
    )


def by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r.get(key)].append(r)
    return d


def mean_family_balanced_accuracy(rows: list[dict]) -> float:
    fams = by(rows, "family")
    return sum(basic(part)["balanced_accuracy"] for part in fams.values()) / len(fams)


def proper_scores(rows: list[dict]) -> dict:
    nll, brier, conf, corr = [], [], [], []
    for r in rows:
        p = r["probabilities"]
        g = r["gold"]
        pg = max(p[g], 1e-12)
        nll.append(-math.log(pg))
        onehot = [1.0 if i == g else 0.0 for i in range(len(p))]
        brier.append(sum((pi - oi) ** 2 for pi, oi in zip(p, onehot)))
        conf.append(max(p))
        corr.append(1.0 if argmax(p) == g else 0.0)
    return dict(nll=sum(nll) / len(nll), brier=sum(brier) / len(brier), mean_confidence=sum(conf) / len(conf), accuracy=sum(corr) / len(corr))


def ece(rows: list[dict], bins: int = 15) -> dict:
    """Binned ECE on max-probability (SECONDARY figure: biased and binning-sensitive)."""
    buckets = [[] for _ in range(bins)]
    for r in rows:
        p = r["probabilities"]
        c = max(p)
        k = min(bins - 1, int(c * bins))
        buckets[k].append((c, 1.0 if argmax(p) == r["gold"] else 0.0))
    n = len(rows)
    e = 0.0
    diag = []
    for k, b in enumerate(buckets):
        if not b:
            continue
        mc = sum(x for x, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        e += len(b) / n * abs(acc - mc)
        diag.append(dict(bin=k, n=len(b), confidence=round(mc, 3), accuracy=round(acc, 3)))
    return dict(ece=e, bins=bins, reliability=diag)


def group_bootstrap_ci(rows: list[dict], samples: int = 1000, seed: int = 217) -> tuple[float, float]:
    """Resample source groups within each family (SemIf's 'source-group bootstrap')."""
    rng = random.Random(seed)
    fams = by(rows, "family")
    fam_groups = {f: by(part, "group_id") for f, part in fams.items()}
    draws = []
    for _ in range(samples):
        sample = []
        for f, groups in fam_groups.items():
            keys = list(groups)
            for _ in keys:
                sample.extend(groups[rng.choice(keys)])
        try:
            draws.append(mean_family_balanced_accuracy(sample))
        except ZeroDivisionError:
            continue
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def latency(rows: list[dict]) -> dict:
    pm = sorted(r["prompt_ms"] for r in rows if r.get("prompt_ms") is not None)
    hm = sorted(r["http_ms"] for r in rows if r.get("http_ms") is not None)
    q = lambda xs, f: xs[int(f * (len(xs) - 1))] if xs else None
    return dict(prompt_ms_p50=q(pm, 0.5), prompt_ms_p95=q(pm, 0.95), http_ms_p50=q(hm, 0.5), http_ms_p95=q(hm, 0.95),
                mean_prompt_tokens=sum(r["prompt_n"] for r in rows if r.get("prompt_n")) / max(1, len(rows)))


def stability(pert: list[dict], base: list[dict]) -> dict:
    """Pair each perturbation row with its original via provenance.base_id.
    Compare by OPTION ID (not index) so option reversal is handled correctly."""
    base_by_id = {r["id"]: r for r in base}
    out = defaultdict(lambda: dict(n=0, flips=0, abs_dp=[], acc_pert=0, acc_base=0))
    for r in pert:
        prov = r.get("provenance") or {}
        b = base_by_id.get(prov.get("base_id"))
        if not b:
            continue
        v = prov.get("variant", "unknown")
        s = out[v]
        s["n"] += 1
        pmap_b = dict(zip(b["option_ids"], b["probabilities"]))
        pmap_p = dict(zip(r["option_ids"], r["probabilities"]))
        arg_b = b["option_ids"][argmax(b["probabilities"])]
        arg_p = r["option_ids"][argmax(r["probabilities"])]
        if arg_b != arg_p:
            s["flips"] += 1
        s["abs_dp"].append(sum(abs(pmap_b[k] - pmap_p.get(k, 0.0)) for k in pmap_b) / 2.0)  # total variation
        s["acc_base"] += 1 if argmax(b["probabilities"]) == b["gold"] else 0
        s["acc_pert"] += 1 if argmax(r["probabilities"]) == r["gold"] else 0
    return {
        v: dict(n=s["n"], flips=s["flips"], flip_rate=s["flips"] / s["n"], flip_rate_ci95=wilson(s["flips"], s["n"]),
                mean_tv_distance=sum(s["abs_dp"]) / s["n"],
                acc_base=s["acc_base"] / s["n"], acc_pert=s["acc_pert"] / s["n"])
        for v, s in out.items() if s["n"]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--base", help="original predictions, for paired stability vs a perturbation file")
    ap.add_argument("--out", help="write JSON summary here")
    args = ap.parse_args()

    rows = read(args.pred)
    fams = by(rows, "family")
    ci = group_bootstrap_ci(rows)
    summary = dict(
        file=args.pred,
        n=len(rows),
        overall=basic(rows),
        mean_family_balanced_accuracy=mean_family_balanced_accuracy(rows),
        mfba_ci95=ci,
        families={f: basic(part) for f, part in sorted(fams.items(), key=lambda kv: str(kv[0]))},
        proper=proper_scores(rows),
        ece=ece(rows),
        leakage=dict(
            missing_letter_rows=sum(1 for r in rows if r["missing_letters"]),
            top1_not_slot_rows=sum(1 for r in rows if not r["top1_is_slot"]),
            top1_non_slot_tokens=sorted({r["top1_token"] for r in rows if not r["top1_is_slot"]})[:12],
        ),
        latency=latency(rows),
    )
    if args.base:
        summary["stability"] = stability(rows, read(args.base))

    printable = {k: v for k, v in summary.items() if k != "ece"}
    printable["ece"] = round(summary["ece"]["ece"], 4)
    print(json.dumps(printable, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else o))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
