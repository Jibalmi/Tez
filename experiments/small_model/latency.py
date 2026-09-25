"""Pilot day 4: latency on §1b's protocol (Laya's): 1, 5, 10 and 50 questions per call, a new state every call.

    python experiments/small_model/latency.py --arm a-base-30k --device cuda
    python experiments/small_model/latency.py --arm a-base-30k --device cpu --threads 24
    python experiments/small_model/latency.py --arm a-base-30k --device cpu --threads 4
    python experiments/small_model/latency.py --arm c-qwen08-30k --device cuda

The state is vs_laya_speed.fresh_state(i) (Laya's Stripe ticket with a new id each call) and the questions are
vs_laya_speed.qs(n) (Laya's choice and yes/no questions, alternating). Timed end to end: tokenising the row, the
forward pass and the per-question softmax (torch.cuda.synchronize on the GPU). Option A reads all n questions of a
call packed after the state in ONE row; option C reads n question-first prompts (each with the state) in one batched
forward. PyTorch, bf16 autocast on the GPU, fp32 on the CPU (as Laya's CPU table). Also cold load time and peak memory.
GPU runs record a VRAM snapshot before and after (C:/temp/gpu_lock.py snapshot): another process holding more than
300 MB of GPU memory marks the run as contaminated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT / "experiments"))
NS = [1, 5, 10, 50]


def snapshot() -> dict:
    try:
        out = subprocess.run([sys.executable, "C:/temp/gpu_lock.py", "snapshot"], capture_output=True, text=True,
                             timeout=60).stdout
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def stats(ts, n):
    ts = np.asarray(ts, float)
    p50 = float(np.percentile(ts, 50))
    return {"n_questions": n, "reps": len(ts), "p50_ms": round(p50, 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(ts.mean()), 2), "ms_per_question": round(p50 / n, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--reps50", type=int, default=30)
    args = ap.parse_args()
    import vs_laya_speed as V
    from tez.schema import parse_question
    dev = args.device
    if dev == "cpu":
        torch.set_num_threads(args.threads)
    before = snapshot() if dev == "cuda" else None
    cpu0 = psutil.cpu_percent(interval=1.0)
    t_load = time.perf_counter()
    if args.arm.startswith("c-"):
        from eval_arms import PredictC
        pred = PredictC(args.arm, device=dev)
    else:
        from eval_arms import PredictA
        pred = PredictA(args.arm, device=dev)
        if dev == "cpu":
            pred.model.float()
    load_s = time.perf_counter() - t_load

    def cases(i, n):
        st = V.fresh_state(i)
        out = []
        for qid, qd in V.qs(n).items():
            q = parse_question(qid, qd)
            out.append({"state": st, "qtype": q.type, "instructions": q.instructions_text, "options": q.options()})
        return out

    def call(i, n):
        cs = cases(i, n)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        if dev == "cpu" and not args.arm.startswith("c-"):
            res = cpu_groups(pred, [cs])
        else:
            res = pred.groups([cs])
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000, res

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sizes, counter = {}, 0
    for n in NS:
        reps = args.reps50 if n == 50 else args.reps
        for _ in range(args.warmup):
            counter += 1
            call(counter, n)
        ts = []
        for _ in range(reps):
            counter += 1
            ms, _ = call(counter, n)
            ts.append(ms)
        sizes[str(n)] = stats(ts, n)
        print(f"{args.arm} {dev} n={n}: p50 {sizes[str(n)]['p50_ms']} ms, {sizes[str(n)]['ms_per_question']} ms/q", flush=True)
    after = snapshot() if dev == "cuda" else None
    contaminated = None
    if dev == "cuda" and isinstance(after, dict) and "processes" in after:
        import os
        others = [p for p in after["processes"] if p["pid"] != os.getpid() and p["mb"] > 300 and p["kind"].startswith("Dedicated")]
        contaminated = bool(others)
    res = {"arm": args.arm, "device": dev, "threads": args.threads if dev == "cpu" else None,
           "dtype": "bf16 autocast" if dev == "cuda" else "fp32", "load_s": round(load_s, 2),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if dev == "cuda" else None,
           "rss_gb": round(psutil.Process().memory_info().rss / 2 ** 30, 2), "cpu_load_before_pct": cpu0,
           "protocol": "vs_laya_speed.fresh_state(i) + qs(n); warmup/reps as args", "warmup": args.warmup,
           "reps": args.reps, "reps50": args.reps50, "sizes": sizes, "vram_before": before, "vram_after": after,
           "contaminated": contaminated, "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    tag = f"{args.arm}_{dev}" + (f"{args.threads}t" if dev == "cpu" else "")
    path = RESULTS / "latency.json"
    allr = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allr[tag] = res
    path.write_text(json.dumps(allr, indent=1), encoding="utf-8")


@torch.no_grad()
def cpu_groups(pred, groups):
    """PredictA.groups without CUDA autocast (fp32 on the CPU)."""
    from student import Q, collate
    rows = [pred.enc.encode(g[0]["state"], [Q(c["qtype"], c["instructions"], list(c["options"])) for c in g]) for g in groups]
    batch = collate(rows, "cpu")
    lp = pred.model(batch).float().numpy()
    return lp


if __name__ == "__main__":
    main()
