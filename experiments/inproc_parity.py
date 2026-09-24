"""Letter parity between the in-process backend and a llama-server serving the same GGUF: the docs request's three
questions about three states, in both prompt layouts (18 prompts), read in process (one by one and in a batch) and over
HTTP (/completion, top-200 log-probabilities). Results: results/speed/inproc_parity_<tag>.json.

  llama-server -m <gguf> -ngl 99 -c 4096 -b 512 -np 1 --no-webui --embeddings --pooling last --port 8095
  python experiments/inproc_parity.py --model <gguf> --lib C:/temp/llamacpp --template qwen3 --url http://127.0.0.1:8095 --tag qwen35_4b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed_common as sc  # noqa: E402
from tez import InprocBackend, LlamaCppBackend  # noqa: E402
from tez.prompt import build_prompt  # noqa: E402
from tez.readout import softmax  # noqa: E402
from tez.schema import parse_question  # noqa: E402

STATES = ["Help! My payouts have been failing for 3 days.",
          "The app crashes every time I open the invoices page, since this morning's update. We cannot bill anyone.",
          {"ticket": {"subject": "Upgrade", "messages": [{"from": "customer", "text": "What would 40 more seats cost?"}]}}]
QUESTIONS = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "topic": {"type": "choice", "instructions": "What is the message about?",
              "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": None}},
    "anger": {"type": "score", "instructions": "How upset is the writer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    b = InprocBackend(a.model, a.template, lib=a.lib).load()
    http = LlamaCppBackend(a.url, template=a.template)
    qs = [parse_question(k, v) for k, v in QUESTIONS.items()]
    rows = []
    for layout in ("question_first", "state_first"):
        for s in STATES:
            ps = [build_prompt(q, s, a.template, layout=layout) for q in qs]
            ks = [len(q.options()) for q in qs]
            batch = [r.logits for r, _ in b.read_many(ps, ks)]
            for q, p, k, zb in zip(qs, ps, ks, batch):
                zi, zh = b.letters(p, k).logits, http.letters(p, k).logits
                pi, pb, ph = softmax(zi), softmax(zb), softmax(zh)
                rows.append({"layout": layout, "question": q.id, "state": s if isinstance(s, str) else "ticket (JSON)",
                             "inproc": zi.tolist(), "inproc_batch": zb.tolist(), "http": zh.tolist(),
                             "same_answer": int(np.argmax(zi)) == int(np.argmax(zh)) == int(np.argmax(zb)),
                             "prob_gap_http": float(np.max(np.abs(pi - ph))), "prob_gap_batch": float(np.max(np.abs(pi - pb))),
                             "logprob_gap_top_http": float(abs(zi.max() - zh.max()))})
    info = {"inproc": b.info(), "http": http.info()}
    out = {"meta": {"script": "experiments/inproc_parity.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(a),
                    **info},
           "n": len(rows), "same_answer": sum(r["same_answer"] for r in rows),
           "prob_gap_http_max": max(r["prob_gap_http"] for r in rows),
           "prob_gap_http_median": float(np.median([r["prob_gap_http"] for r in rows])),
           "prob_gap_batch_max": max(r["prob_gap_batch"] for r in rows),
           "logprob_gap_top_letter_http_max": max(r["logprob_gap_top_http"] for r in rows), "rows": rows}
    print({k: v for k, v in out.items() if k not in ("rows", "meta")})
    print("wrote", sc.dump(f"inproc_parity_{a.tag}.json", out))


if __name__ == "__main__":
    main()
