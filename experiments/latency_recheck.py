"""SUPERSEDED by latency_breakdown.py. Caveat found after running it: each /embedding call here follows a /completion on
the SAME prompt, so llama.cpp reuses that prompt's KV and the embedding timings (25-31 ms) are cache hits, not real
per-decision costs. Use results/latency_breakdown.json (one endpoint at a time) for every latency number.

Clean latency comparison of the depth-pruned GGUFs: one server at a time, same 200 typed-decisions prompts,
letters (/completion, n_predict 1) and embeddings (/embedding), 3 warm-up requests, prompt caching off.

  py experiments/latency_recheck.py --out results/pruned_latency_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEZ_CACHE_PROMPT", "0")
_s = importlib.util.spec_from_file_location("psb", ROOT / "experiments" / "pruned_server_bench.py")
psb = importlib.util.module_from_spec(_s); sys.modules["psb"] = psb; _s.loader.exec_module(psb)  # type: ignore[union-attr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default="L20:tools/models/Qwen3.5-4B-Q8_0-L20.gguf,L24:tools/models/Qwen3.5-4B-Q8_0-L24.gguf,L29:tools/models/Qwen3.5-4B-Q8_0-L29.gguf,L32:tools/models/Qwen3.5-4B-Q8_0.gguf")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = psb.hp.load_split("test")[: args.n]
    server = f"http://127.0.0.1:{args.port}"; res = {}
    for item in args.cuts.split(","):
        name, path = item.split(":", 1)
        proc = psb.start(path, args.port, Path(os.environ.get("TEMP", ".")) / f"latency_{name}.log")
        try:
            for c in cases[:3]:
                psb.h2h.tez_score_letters(server, psb.wrap(c), len(c["options"])); psb.embed(server, psb.wrap(c))
            ml, me = [], []
            for c in cases:
                t0 = time.perf_counter(); psb.h2h.tez_score_letters(server, psb.wrap(c), len(c["options"])); ml.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter(); psb.embed(server, psb.wrap(c)); me.append((time.perf_counter() - t0) * 1000)
            res[name] = dict(ms_letters_p50=float(np.median(ml)), ms_letters_p90=float(np.percentile(ml, 90)),
                             ms_embed_p50=float(np.median(me)), ms_embed_p90=float(np.percentile(me, 90)))
            print(name, {k: round(v, 1) for k, v in res[name].items()}, flush=True)
        finally:
            proc.kill(); time.sleep(3)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
