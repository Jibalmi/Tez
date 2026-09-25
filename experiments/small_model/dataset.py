"""Training examples from the teacher's reads (plan §4, "Teacher targets").

Target per question: the average of the two option orders' letter distributions, each at the §5c default temperature
of its question type (Gemma 4 12B Q8_0, gemma4 template: 6.01 yes/no, 4.71 choice with up to 10 options shown, 2.66
choice with 11-26, 5.18 score). A gold label rides along where a licensed gold set has one.

Splits: 5 % of the states (by hash) are the teacher-labelled validation split, used only for the per-type temperature
fit and for monitoring; never for training. The 10,000-decision subset for the learning curve is the first 2,000
training states in hash order.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from common import RUNS, read_jsonl

TEMPS = {"noul": 6.01, "choice": 4.71, "choice_many": 2.66, "score": 5.18}


def temp_for(qtype: str, k: int) -> float:
    if qtype == "choice":
        return TEMPS["choice_many" if k > 10 else "choice"]
    return TEMPS[qtype]


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def hkey(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@dataclass
class Decision:
    qid: str
    type: str
    instructions: str
    options: list              # [(key, description)] in the teacher's original order (__none__ last when present)
    target: np.ndarray         # teacher distribution over options, same order
    t1_orig: np.ndarray        # teacher at T = 1, original order (for diagnostics)
    gold: int | None


@dataclass
class Example:
    id: str
    source: str
    lang: str
    state: object
    decisions: list


def load_examples() -> tuple[list[Example], list[Example], list[Example]]:
    """(train_all, train_10k, val)."""
    from tez.schema import parse_question
    pool = {r["id"]: r for r in read_jsonl(RUNS / "pool.jsonl")}
    assign = {r["id"]: r for r in read_jsonl(RUNS / "assign.jsonl")}
    reads = {}
    for p in sorted((RUNS / "teacher").glob("part-*.jsonl")):
        for r in read_jsonl(p):
            reads[r["id"]] = r
    out = []
    for sid, rd in reads.items():
        a, st = assign[sid], pool[sid]
        by_q = {q["qid"]: q for q in a["questions"]}
        decs = []
        for tq in rd["questions"]:
            qd = by_q[tq["qid"]]
            q = parse_question(tq["qid"], {k: qd[k] for k in ("type", "instructions", "criteria") if k in qd})
            opts = q.options(tq["abstain"])
            k = len(opts)
            T = temp_for(q.type, k)
            zo, zr = np.asarray(tq["z_orig"]), np.asarray(tq["z_rev"])
            tgt = 0.5 * (softmax(zo, T) + softmax(zr, T)[::-1])
            gold = None
            if "gold" in qd:
                gold = list(q.criteria).index(qd["gold"])
            decs.append(Decision(tq["qid"], q.type, q.instructions_text, opts, tgt, softmax(zo), gold))
        out.append(Example(sid, st["source"], st["lang"], st["state"], decs))
    out.sort(key=lambda e: hkey(e.id))
    val = [e for e in out if hkey("val|" + e.id) < 0.05]
    train = [e for e in out if hkey("val|" + e.id) >= 0.05]
    return train, train[:2000], val


def summary(exs: list[Example]) -> dict:
    import collections
    return {"states": len(exs), "decisions": sum(len(e.decisions) for e in exs),
            "sources": dict(collections.Counter(e.source for e in exs)),
            "types": dict(collections.Counter(d.type for e in exs for d in e.decisions)),
            "gold_decisions": sum(d.gold is not None for e in exs for d in e.decisions),
            "teacher_order_agreement": float(np.mean([int(np.argmax(d.t1_orig) == np.argmax(d.target))
                                                      for e in exs for d in e.decisions])) if exs else None}


if __name__ == "__main__":
    tr, tr10, va = load_examples()
    print(json.dumps({"train": summary(tr), "train_10k": summary(tr10), "val": summary(va)}, indent=1))
