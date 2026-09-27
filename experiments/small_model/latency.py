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
GPU runs record a VRAM snapshot before and after (C:/temp/gpu_lock.py on Windows, nvidia-smi on Linux): another
process holding more than 300 MB of GPU memory marks the run as contaminated.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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
        if sys.platform != "win32":
            output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,process_name",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    timeout=30, check=True).stdout
            processes = [{"pid": int(pid.strip()), "mb": float(memory.strip()), "name": name.strip(),
                          "kind": "Dedicated compute"} for pid, memory, name in csv.reader(output.splitlines())]
            return {"source": "nvidia-smi compute applications", "processes": processes}
        out = subprocess.run([sys.executable, "C:/temp/gpu_lock.py", "snapshot"], capture_output=True, text=True,
                             timeout=60).stdout
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def thermal() -> dict:
    """GPU temperature and throttle state (the machine rule: timings count only below about 70 C with no
    sw_thermal_slowdown)."""
    q = ("temperature.gpu,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,"
         "clocks.sm,utilization.gpu,memory.used")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10).stdout.strip().splitlines()[0]
        t, sw, hw, sm, util, mem = (x.strip() for x in out.split(","))
        return {"temp_c": int(t), "sw_thermal": sw == "Active", "hw_thermal": hw == "Active", "sm_mhz": int(sm),
                "util_pct": int(util), "mem_mb": int(mem)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120]}


def cool_down(limit_c: int = 65, max_wait_s: int = 900) -> dict:
    t0 = time.time()
    th = thermal()
    while th.get("temp_c", 0) > limit_c and time.time() - t0 < max_wait_s:
        time.sleep(10)
        th = thermal()
    th["waited_s"] = round(time.time() - t0)
    if "error" in th or th.get("temp_c", limit_c + 1) > limit_c:
        raise RuntimeError(f"GPU did not meet the {limit_c} C cooldown requirement: {th}")
    return th


def stats(ts, n):
    ts = np.asarray(ts, float)
    p50 = float(np.percentile(ts, 50))
    return {"n_questions": n, "reps": len(ts), "p50_ms": round(p50, 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(ts.mean()), 2), "ms_per_question": round(p50 / n, 3)}


def contamination(reading, own_pid):
    if not isinstance(reading, dict) or "processes" not in reading:
        return None
    return any(p["pid"] != own_pid and p["mb"] > 300 and p["kind"].lower().startswith("dedicated")
               for p in reading["processes"])


def throttling(samples):
    if any(t.get("temp_c", 0) > 70 or t.get("sw_thermal") or t.get("hw_thermal") for t in samples):
        return True
    if not samples or any(not {"temp_c", "sw_thermal", "hw_thermal"}.issubset(t) for t in samples):
        return None
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=int(__import__("os").environ.get("RUN_HEAVY_THREADS", "6")),
                    help="CPU threads (the machine's cap is 6; the plan's 24-core run is not allowed)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--reps50", type=int, default=30)
    ap.add_argument("--cooldown-seconds", type=int, default=900)
    ap.add_argument("--readout", choices=("full", "restricted"), default="full")
    args = ap.parse_args()
    import vs_laya_speed as V
    from tez.schema import parse_question
    dev = args.device
    if dev == "cpu":
        torch.set_num_threads(args.threads)
    before = snapshot() if dev == "cuda" else None
    cpu0 = psutil.cpu_percent(interval=1.0)
    t_load = time.perf_counter()
    from competitor_models import predictor
    from eval_arms import PredictA
    pred = predictor(args.arm, device=dev)
    if args.readout == "restricted":
        if args.arm != "c-qwen08-30k" or dev != "cuda":
            raise ValueError("Restricted timing is registered only for the frozen CUDA Qwen control")
        from restricted_qwen import RestrictedQwen
        pred = RestrictedQwen(pred)
    if dev == "cpu" and isinstance(pred, PredictA):
        pred.model.float()
    load_s = time.perf_counter() - t_load
    trace_path = RESULTS / f"latency_samples_{args.arm}_{dev}_{args.readout}.jsonl"
    trace = trace_path.open("x", encoding="utf-8")

    def cases(i, n):
        st = V.fresh_state(i)
        out = []
        for qid, qd in V.qs(n).items():
            q = parse_question(qid, qd)
            out.append({"state": st, "qtype": q.type, "instructions": q.instructions_text, "options": q.options()})
        return out

    def call(i, n, phase="timed"):
        cs = cases(i, n)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        if dev == "cpu" and isinstance(pred, PredictA):
            res = cpu_groups(pred, [cs])
        else:
            res = pred.groups([cs])
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        trace.write(json.dumps({"request": i, "n_questions": n, "phase": phase, "ms": ms,
                                "log_probabilities": res}, default=lambda value: value.tolist(), allow_nan=False) + "\n")
        trace.flush()
        return ms, res

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sizes, counter, thermals = {}, 0, []
    if dev == "cuda":
        thermals.append({"at": "start", **cool_down(max_wait_s=args.cooldown_seconds)})
    for n in NS:
        if dev == "cuda" and n != NS[0]:
            thermals.append({"at": f"before n={n}", **cool_down(max_wait_s=args.cooldown_seconds)})
        reps = args.reps50 if n == 50 else args.reps
        for _ in range(args.warmup):
            counter += 1
            call(counter, n, phase="warmup")
        ts = []
        for _ in range(reps):
            counter += 1
            ms, _ = call(counter, n)
            ts.append(ms)
        sizes[str(n)] = stats(ts, n)
        if dev == "cuda":
            th = thermal()
            sizes[str(n)]["thermal_after"] = th
            thermals.append({"at": f"after n={n}", **th})
        print(f"{args.arm} {dev} n={n}: p50 {sizes[str(n)]['p50_ms']} ms, {sizes[str(n)]['ms_per_question']} ms/q", flush=True)
    after = snapshot() if dev == "cuda" else None
    trace.close()
    contaminated = None
    if dev == "cuda":
        import os
        readings = [contamination(reading, os.getpid()) for reading in (before, after)]
        contaminated = True if True in readings else (None if None in readings else False)
    throttled = None
    if dev == "cuda":
        throttled = throttling(thermals)
    res = {"arm": args.arm, "device": dev, "readout": args.readout, "weights_sha256": pred.weights, "model_info": pred.info,
           "raw_samples_file": trace_path.name, "raw_samples_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
           "gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "thermal": thermals, "throttled": throttled, "threads": args.threads if dev == "cpu" else None,
           "dtype": "bf16 autocast" if dev == "cuda" else "fp32", "load_s": round(load_s, 2),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if dev == "cuda" else None,
           "rss_gb": round(psutil.Process().memory_info().rss / 2 ** 30, 2), "cpu_load_before_pct": cpu0,
           "protocol": "vs_laya_speed.fresh_state(i) + qs(n); warmup/reps as args", "warmup": args.warmup,
           "reps": args.reps, "reps50": args.reps50, "sizes": sizes, "vram_before": before, "vram_after": after,
           "contaminated": contaminated, "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    tag = f"{args.arm}_{dev}" + (f"{args.threads}t" if dev == "cpu" else "")
    if args.readout == "restricted":
        tag += "_restricted"
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
