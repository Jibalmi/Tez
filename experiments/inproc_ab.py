"""A/B in one process on one loaded model: the speed study's batched arm (experiments/speed_multiq_inproc.py,
Engine.run_batch: prefix on sequence 0, copies, every suffix in one decode, the 26 letter logits read raw) against the
runtime's InprocBackend.read_many and the whole engine (Tez.handle, no HTTP), interleaved call by call so the laptop's
thermal and power state hits every arm alike. Laya's protocol, distinct questions, the study's pacing.
Results: results/speed/inproc_runtime_ab.json.

  python experiments/inproc_ab.py --model <gguf> --lib C:/temp/llamacpp
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed_common as sc  # noqa: E402
from tez import InprocBackend, Tez  # noqa: E402
from tez.prompt import build_prompt  # noqa: E402
from tez.schema import parse_questions  # noqa: E402


def lcp(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def study_batch(b: InprocBackend, prompts: list[str], ks: list[int]) -> list[np.ndarray]:
    """speed_multiq_inproc.Engine.run_batch on the runtime's loaded model: the same decodes, letters read raw."""
    m = b._m
    m.clear()
    b._cached = []
    toks = [m.tokenize(p) for p in prompts]
    n = len(toks)
    lp = min(len(t) for t in toks)
    for t in toks[1:]:
        lp = min(lp, lcp(toks[0], t))
    lp = max(0, min(lp, min(len(t) for t in toks) - 1))
    if lp:
        b._eval(toks[0][:lp], 0, 0, output_last=False)
        for j in range(1, n):
            m.seq_cp(0, j)
    bt, bp, bs, bo = [], [], [], []
    for j, t in enumerate(toks):
        rest = t[lp:]
        bt += rest
        bp += range(lp, len(t))
        bs += [j] * len(rest)
        bo += [False] * (len(rest) - 1) + [True]
    outs = m.decode(bt, bp, bs, bo)
    ids = b._letter_ids
    z = [m.logits_row(p)[ids[:k]].astype(np.float64) for p, k in zip(outs, ks)]
    m.clear()
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--threads", default="", help="comma-separated CPU thread counts to try for the runtime arm")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--gap", type=float, default=0.3)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--cool", type=float, default=85.0)
    ap.add_argument("--tag", default="inproc_runtime_ab")
    a = ap.parse_args()
    b = InprocBackend(a.model, "gemma4", lib=a.lib).load()
    tez = Tez(backend=b)
    lib = C.CDLL(str(Path(a.lib) / "llama.dll"))
    lib.llama_set_n_threads.argtypes = [C.c_void_p, C.c_int32, C.c_int32]
    lib.llama_set_n_threads.restype = None
    default_threads = max(1, (os.cpu_count() or 2) // 2)          # InprocBackend's default
    threads = [int(x) for x in a.threads.split(",") if x] or [default_threads]
    qs_wire = sc.laya_questions(a.n, unique=True)
    qs = parse_questions(qs_wire)
    ks = [len(q.options()) for q in qs.values()]
    arms = ["study_batch"] + [f"read_many_t{t}" for t in threads] + [f"engine_t{t}" for t in threads]
    rec = {x: [] for x in arms}
    agree = []
    counter = 800_000
    before = sc.wait_cool(a.cool, 240) if a.cool else sc.gpu_now()
    with sc.GpuMonitor() as mon:
        for i in range(a.warmup + a.reps):
            for arm in arms:
                counter += 1
                st = sc.fresh_state(counter)
                prompts = [build_prompt(q, st, "gemma4", layout="state_first") for q in qs.values()]
                if "_t" in arm:
                    t = int(arm.rsplit("_t", 1)[1])
                    lib.llama_set_n_threads(b._m.ctx, t, t)
                t0 = time.perf_counter()
                if arm == "study_batch":
                    study_batch(b, prompts, ks)
                elif arm.startswith("read_many"):
                    got = b.read_many(prompts, ks)
                else:
                    tez.handle({"state": st, "questions": qs_wire})
                wall = (time.perf_counter() - t0) * 1000.0
                if arm.startswith("read_many") and i == a.warmup:
                    ref = study_batch(b, prompts, ks)
                    agree.append(max(float(np.max(np.abs((x - x.max()) - (g.logits - g.logits.max()))))
                                     for x, (g, _) in zip(ref, got)))
                time.sleep(max(a.gap, a.duty * wall / 1000.0))
                if i >= a.warmup:
                    rec[arm].append(wall)
        g = mon.summary()
    out = {"meta": {"script": "experiments/inproc_ab.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(a),
                    "model": a.model, "details": {k: v for k, v in b.details().items() if k != "chat_template"}},
           "arms": {arm: sc.per_call_stats(w, a.n) | {"samples_ms": [round(x, 1) for x in w]} for arm, w in rec.items()},
           "letter_logit_gap_read_many_vs_study": agree, "gpu": {"before": before, "during": g},
           "telemetry": [list(r) for r in mon.rows]}
    for arm, s in out["arms"].items():
        print(f"{arm:18s} p50 {s['p50']:8.1f}  p95 {s['p95']:8.1f}  per q {s['ms_per_question_p50']:6.2f}", flush=True)
    print("gpu", g, flush=True)
    print("wrote", sc.dump(f"{a.tag}.json", out))


if __name__ == "__main__":
    main()
