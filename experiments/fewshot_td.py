"""Few-shot in the cached prefix: the decoder's answer to Laya's fine-tuning, with zero training.

Laya's 0.766 on typed-decisions comes from a checkpoint fine-tuned on the benchmark's own train
split. A cached-decoder can use the same data access without training: k worked examples from the
train split (state, question, answer letter) go into the prompt BEFORE the instruction/options, so
they sit in the cached prefix and cost nothing per query after the first. This script scores the
2,000 test decisions with k examples of the same workflow and question id (when available).

  py experiments/fewshot_td.py --k 4 --out results/h2h/rows_typed_decisions_tez-fewshot4.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bh", ROOT / "experiments" / "bench_h2h.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)  # type: ignore[union-attr]


def train_examples():
    from datasets import load_dataset
    d = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    ex = {}
    for r in d:
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        try:
            st = json.loads(r["state"])
        except Exception:  # noqa: BLE001
            st = r["state"]
        for qid, qd in qs.items():
            g = gold[qid]
            if qd["type"] == "choice":
                keys = list(qd["criteria"].keys()); gi = keys.index(str(g["label"]))
            elif qd["type"] == "noul":
                gi = 1 if str(g["label"]).lower() == "true" else 0
            else:
                gi = int(g["label"])
            ex.setdefault((r["workflow"], qid), []).append((st, qd, gi))
    return ex


def shots_block(examples, k, rng):
    picks = rng.sample(examples, min(k, len(examples)))
    parts = []
    for st, qd, gi in picks:
        if qd["type"] == "choice":
            opts = [(kk, qd["criteria"][kk] if isinstance(qd["criteria"][kk], str) else json.dumps(qd["criteria"][kk])) for kk in qd["criteria"]]
        elif qd["type"] == "noul":
            opts = [("false", "no, the statement does not hold"), ("true", "yes, the statement holds")]
        else:
            opts = [(f"level {i}", c if isinstance(c, str) else json.dumps(c)) for i, c in enumerate(qd["criteria"])]
        o = "\n".join(f"{bh.LETTERS[i]}. {kk}: {dd}" for i, (kk, dd) in enumerate(opts))
        s = json.dumps(st, ensure_ascii=False) if not isinstance(st, str) else st
        parts.append(f"Example input:\n{s}\nQuestion ({qd['type']}): {qd['instructions']}\nOptions:\n{o}\nAnswer: {bh.LETTERS[gi]}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ex = train_examples()
    cases = bh.build_task("typed_decisions", 0)
    if args.limit:
        cases = cases[: args.limit]
    rng = random.Random(13)
    # one fixed shot block per (workflow, qid) so the prefix is cached across all test cases of that question
    blocks = {key: shots_block(v, args.k, rng) for key, v in ex.items()}
    recs, t0 = [], time.perf_counter()
    for c in cases:
        key = (c["workflow"], c["id"].rsplit("-", 1)[1])
        block = blocks.get(key) or blocks.get(next((kk for kk in blocks if kk[0] == c["workflow"]), None), "")
        head = ("You are a decision engine. Read the question and the options, then look at the input and answer "
                "with the single letter of the best option. Answer with the letter only. Worked examples of this exact question follow.")
        q = f"Question ({c['qtype']}): {c['instructions']}"
        opts = "Options:\n" + "\n".join(f"{bh.LETTERS[i]}. {k}: {d}" for i, (k, d) in enumerate(c["options"]))
        st = json.dumps(c["state"], ensure_ascii=False) if not isinstance(c["state"], str) else c["state"]
        prompt = f"<|turn>user\n{head}\n\n{block}\n\n{q}\n\n{opts}\n\nInput:\n{st}{bh.TAIL}"
        t = time.perf_counter()
        p, z, pn = bh.tez_score_letters(args.server, prompt, len(c["options"]))
        recs.append(dict(id=c["id"], task="typed_decisions", lang="en", qtype=c["qtype"], gold=c["gold"], k=len(c["options"]),
                         probabilities=[float(x) for x in p], logits=[float(x) for x in z], pred=int(np.argmax(p)),
                         ms=(time.perf_counter() - t) * 1000, prompt_n=pn, calls=1, extra=c.get("extra", {}), workflow=c["workflow"]))
    s = bh.summarise(recs)
    s["by_workflow"] = {w: bh.summarise([r for r in recs if r["workflow"] == w]) for w in sorted({r["workflow"] for r in recs})}
    s["by_qtype"] = {q: bh.summarise([r for r in recs if r["qtype"] == q]) for q in sorted({r["qtype"] for r in recs})}
    s["k_shots"] = args.k; s["wall_s"] = time.perf_counter() - t0
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("by_workflow", "by_qtype")}, indent=1))
    print({w: round(v["accuracy"], 3) for w, v in s["by_workflow"].items()}, {q: round(v["accuracy"], 3) for q, v in s["by_qtype"].items()})


if __name__ == "__main__":
    main()
