"""'System 1.5': a small hidden thinking budget before the one-pass readout.

Gemma 4 has a thought channel. Instead of closing it immediately (the System-One readout), let the
model write up to N thought tokens, close the channel, and read the answer letter at the next
position exactly as before. Cost = N decode steps (~25 ms each on the 12B Q8 here). This measures
the accuracy / latency frontier between pure System One (N=0) and short reasoning on the
typed-decisions test set (subset), against the zero-shot 0.704.

  py experiments/think_td.py --budget 32 --limit 500 --out results/h2h/rows_typed_decisions_tez-think32.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bh", ROOT / "experiments" / "bench_h2h.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)  # type: ignore[union-attr]


def think_then_read(server, prompt_wo_tail, budget, k):
    """prompt_wo_tail ends with the user turn; we open the thought channel, generate up to `budget`
    tokens (stop at the channel close), then close it and read the letter."""
    import importlib.util as _iu
    _s = _iu.spec_from_file_location("backend", ROOT / "experiments" / "backend.py"); be = _iu.module_from_spec(_s); _s.loader.exec_module(be)  # type: ignore[union-attr]
    # The System-One instruction makes the model close the thought channel immediately (1 token);
    # invite a short think and seed the channel (s1-style budget forcing), then stop at the close.
    prompt_wo_tail = prompt_wo_tail.replace("Answer with the letter only.", "Think briefly in your thought channel first, then answer with the letter only.")
    seed = "Let me check the input against the options. "
    open_ch = f"{prompt_wo_tail}<turn|>\n<|turn>model\n<|channel>thought\n{seed}"
    t0 = time.perf_counter()
    thought, n_gen = be.generate(open_ch, budget, ["<channel|>"])
    final = open_ch + thought + "<channel|>"
    p, z, pn = be.score_letters(final, k)
    return p, z, thought, n_gen, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = bh.build_task("typed_decisions", 0)
    # deterministic subset spread across workflows
    cases = [c for i, c in enumerate(cases) if i % max(1, len(cases) // args.limit) == 0][: args.limit]
    recs = []
    for c in cases:
        prompt = bh.tez_prompt(c, c["options"])
        assert prompt.endswith(bh.TAIL)
        wo = prompt[: -len(bh.TAIL)]
        p, z, thought, n_gen, ms = think_then_read(args.server, wo, args.budget, len(c["options"]))
        recs.append(dict(id=c["id"], task="typed_decisions", lang="en", qtype=c["qtype"], gold=c["gold"], k=len(c["options"]),
                         probabilities=[float(x) for x in p], logits=[float(x) for x in z], pred=int(np.argmax(p)), ms=ms,
                         thought_tokens=n_gen, thought=thought[:400], extra=c.get("extra", {}), workflow=c["workflow"]))
    s = bh.summarise(recs); s["budget"] = args.budget; s["mean_thought_tokens"] = float(np.mean([r["thought_tokens"] for r in recs]))
    s["by_qtype"] = {q: bh.summarise([r for r in recs if r["qtype"] == q]) for q in sorted({r["qtype"] for r in recs})}
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "by_qtype"}, indent=1))
    print({q: round(v["accuracy"], 3) for q, v in s["by_qtype"].items()})


if __name__ == "__main__":
    main()
