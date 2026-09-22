"""JevBench public tiers (github.com/fstandhartinger/jevbench, datasets/public/{original,easy,hard}.jsonl)
scored with Tez's one-pass readout, so the result is directly comparable with the leaderboard's
public-tier numbers (the leaderboard's headline uses a held-out tier we cannot see).

Per item: state, question {type?, instructions, criteria}, labels (exact label set), expected.
Intelligence per tier = (accuracy - chance) / (1 - chance), chance = mean 1/|labels|, clipped at 0.
Calibration = ECE-15 on max-probability. Speed score = 100 - 20*log10(seconds / 0.1).

  py experiments/bench_jevbench.py --out results/jevbench_tez.json          (TEZ_BACKEND=ollama when llama-server is down)
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bh", ROOT / "experiments" / "bench_h2h.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)  # type: ignore[union-attr]


def parse(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:  # noqa: BLE001
        return json.loads(v)


def load_tier(name):
    rows = [json.loads(l) for l in (ROOT / "data" / "jevbench" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = []
    for r in rows:
        labels = parse(r["labels"]); q = parse(r["question"])
        crit = q.get("criteria") or {}
        qtype = q.get("type") or ("noul" if set(labels) <= {"no", "yes", "false", "true"} else "choice")
        if qtype == "noul" or set(labels) == {"no", "yes"}:
            # keep the item's own label order; map criteria false/true onto no/yes
            opts = []
            for lab in labels:
                key = "true" if lab in ("yes", "true") else "false"
                opts.append((lab, crit.get(key) or crit.get(lab) or ("yes, the statement holds" if key == "true" else "no, the statement does not hold")))
        else:
            opts = [(lab, crit.get(lab) if isinstance(crit, dict) else "") for lab in labels]
            opts = [(k, d if isinstance(d, str) else json.dumps(d)) for k, d in opts]
        cases.append(dict(id=r["id"], task=f"jevbench:{name}", lang="en", state=r["state"], qtype="noul" if qtype == "noul" else qtype,
                          instructions=q.get("instructions", ""), options=opts, gold=labels.index(r["expected"]), family=r.get("family"), group=r.get("group")))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="original,easy,hard")
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    all_rows = []
    for tier in args.tiers.split(","):
        cases = load_tier(tier)
        recs = []
        for c in cases:
            t = time.perf_counter()
            try:
                p, z, pn, calls = bh.tez_decide(args.server, c)
            except Exception as exc:  # noqa: BLE001
                print("ERR", c["id"], str(exc)[:120]); continue
            recs.append(dict(id=c["id"], tier=tier, family=c["family"], group=c["group"], qtype=c["qtype"], gold=c["gold"], k=len(c["options"]),
                             probabilities=[float(x) for x in p], logits=[float(x) for x in z], pred=int(np.argmax(p)), s=time.perf_counter() - t, prompt_n=pn))
        all_rows += recs
        acc = float(np.mean([r["pred"] == r["gold"] for r in recs])); chance = float(np.mean([1 / r["k"] for r in recs]))
        conf = np.array([max(r["probabilities"]) for r in recs]); corr = np.array([float(r["pred"] == r["gold"]) for r in recs])
        sec = float(np.median([r["s"] for r in recs]))
        out[tier] = dict(n=len(recs), accuracy=acc, chance=chance, intelligence=max(0.0, (acc - chance) / (1 - chance)) * 100, ece15=bh.ece15(conf, corr),
                         nll=float(np.mean([-math.log(max(r["probabilities"][r["gold"]], 1e-12)) for r in recs])), median_s=sec,
                         speed_score=100 - 20 * math.log10(max(sec, 1e-3) / 0.1),
                         by_family={f: float(np.mean([r["pred"] == r["gold"] for r in recs if r["family"] == f])) for f in sorted({r["family"] for r in recs})})
        # paraphrase-pair consistency on the original tier (36 pairs share a group)
        if tier == "original":
            groups = {}
            for r in recs:
                groups.setdefault(r["group"], []).append(r["pred"] == r["gold"])
            pairs = [v for v in groups.values() if len(v) == 2]
            out[tier]["pair_consistency"] = float(np.mean([a == b for a, b in pairs])) if pairs else None
        print(tier, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[tier].items() if k != "by_family"})
        print("   by family", {k: round(v, 3) for k, v in out[tier]["by_family"].items()})
    Path(args.out).write_text(json.dumps(dict(summary=out, rows=all_rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
