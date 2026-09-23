"""Speed per call: 1, 5, 10 and 50 questions on one state (results/vs_laya/speed_per_call.json).

Laya's protocol, verbatim (research/scripts/build_benchmark_nb.py section 7, the numbers behind "Speed on one T4",
and research/scripts/bench_latency.py): the Stripe payout ticket as the state; questions alternate Q_CHOICE (even
index) and Q_NOUL (odd index); 3 warm-up calls, then 20 timed calls per size (--reps); p50 / p95 per call and
p50 / n per question.

  tez                  POST http://127.0.0.1:8787/v1/systemone (the `tez serve` runtime over llama-server :8091),
                       default readout (letters, no schema); wall clock measured by the client around the HTTP call
  laya                 Agent.system_one in-process: all questions of a call in one batched forward pass
  laya-multilingual    (Laya's own batched predict), torch.cuda.synchronize() around each timed call on CUDA

Usage (repo root; the published file used --warmup 10 --reps 100 --reps50 30 for the main entries, and the defaults
3 / 20, Laya's own numbers, for the "(3 warm-ups, 20 calls)" extras; nothing else should load the CPU or the GPU)
  python experiments/vs_laya_speed.py --system tez --warmup 10 --reps 100 --reps50 30
  python experiments/vs_laya_speed.py --system tez --fresh-state --warmup 10 --reps 100 --reps50 30
  python experiments/vs_laya_speed.py --system laya --device cuda --warmup 10 --reps 100 --reps50 30
  python experiments/vs_laya_speed.py --system laya-multilingual --device cuda --warmup 10 --reps 100 --reps50 30
  python experiments/vs_laya_speed.py --aggregate
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "vs_laya"
RAW = OUT / "speed_rows"
NS = [1, 5, 10, 50]

LAT_STATE = {"ticket": {"subject": "Payout failing",
             "messages": [{"from": "customer",
                           "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}
LAYA = {"laya": None, "laya-multilingual": "multilingual"}


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def stats(ts, n):
    ts = np.asarray(ts, float)
    p50 = float(np.percentile(ts, 50))
    return {"n_questions": n, "reps": len(ts), "p50_ms": round(p50, 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
            "mean_ms": round(float(ts.mean()), 2), "min_ms": round(float(ts.min()), 2),
            "ms_per_question": round(p50 / n, 2), "samples_ms": [round(float(x), 2) for x in ts]}


def fresh_state(i):
    """Same ticket, a different state on every call (new id at the start of the subject and the message), so no
    part of the state can come from llama.cpp's prompt cache: the per-call cost of a new ticket."""
    return {"ticket": {"subject": f"Ticket {i}: payout failing",
                       "messages": [{"from": "customer",
                                     "text": f"Ticket {i}. " + "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}


def run_tez(url, warmup, reps, fresh=False, reps50=None):
    import requests
    s = requests.Session()
    health = s.get(url.rsplit("/v1/", 1)[0] + "/healthz", timeout=30).json()
    out = {"system": "tez", "endpoint": url, "health": health, "fresh_state": fresh, "sizes": {}}
    counter = 0
    for n in NS:
        body = {"model": "tez-latest", "state": LAT_STATE, "questions": qs(n)}
        for _ in range(warmup):
            if fresh:
                counter += 1; body["state"] = fresh_state(counter)
            r = s.post(url, json=body, timeout=600); r.raise_for_status()
        ts, server_ms, answers = [], [], None
        for _ in range(reps if n < 50 or not reps50 else reps50):
            if fresh:
                counter += 1; body["state"] = fresh_state(counter)
            t = time.perf_counter()
            r = s.post(url, json=body, timeout=600)
            ts.append((time.perf_counter() - t) * 1000)
            r.raise_for_status()
            j = r.json()
            server_ms.append(j.get("tez", {}).get("latency_ms"))
            answers = j["answers"]
        st = stats(ts, n)
        st["server_reported_p50_ms"] = float(np.median([x for x in server_ms if x is not None])) if any(server_ms) else None
        st["example_answers"] = {k: answers[k] for k in list(answers)[:2]}
        out["sizes"][str(n)] = st
        print(f"tez  {n:2d}q  p50 {st['p50_ms']:8.1f} ms  p95 {st['p95_ms']:8.1f}  {st['ms_per_question']:6.2f} ms/q  "
              f"(server {st['server_reported_p50_ms']})", flush=True)
    return out


def run_laya(system, device, warmup, reps, threads, reps50=None):
    import torch
    import laya
    if threads:
        torch.set_num_threads(threads)
    info = {}
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        info["cuda_free_mib_before_load"] = round(free / 2**20)
        info["cuda_total_mib"] = round(total / 2**20)
    t = time.perf_counter()
    agent = laya.load("convaiinnovations/laya", subfolder=LAYA[system], device=device)
    info["load_s"] = round(time.perf_counter() - t, 2)
    cuda = agent.device.type == "cuda"
    if cuda:
        free, _ = torch.cuda.mem_get_info()
        info["cuda_free_mib_after_load"] = round(free / 2**20)
        info["gpu"] = torch.cuda.get_device_name(0)
    out = {"system": system, "device": str(agent.device), "dtype": str(agent.dtype), "autocast": cuda,
           "threads": torch.get_num_threads(), "laya_version": laya.__version__, "info": info, "sizes": {}}
    print(f"{system}: device {agent.device} dtype {agent.dtype} {info}", flush=True)
    for n in NS:
        q = qs(n)
        for _ in range(warmup):
            agent.system_one(LAT_STATE, q)
        ts = []
        for _ in range(reps if n < 50 or not reps50 else reps50):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            res = agent.system_one(LAT_STATE, q)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        st = stats(ts, n)
        st["input_tokens"] = res["usage"]["input_tokens"]
        st["example_answers"] = {k: res["answers"][k] for k in list(res["answers"])[:2]}
        out["sizes"][str(n)] = st
        print(f"{system} {n:2d}q  p50 {st['p50_ms']:8.1f} ms  p95 {st['p95_ms']:8.1f}  {st['ms_per_question']:6.2f} ms/q", flush=True)
    if cuda:
        out["info"]["cuda_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    return out


def aggregate():
    raws = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RAW.glob("*.json"))}
    systems, devices, extras = {}, {}, {}
    # Tez's main entry is a new ticket state in every call: Laya's protocol repeats the identical call, which an encoder
    # pays in full every time but llama.cpp serves partly from its prompt cache (see tez_note); the repeat is in extras.
    pick = {"tez": ["tez_fresh_state", "tez"], "laya": ["laya_cuda", "laya_cpu"],
            "laya-multilingual": ["laya-multilingual_cuda", "laya-multilingual_cpu"]}
    used = set()
    for s, cands in pick.items():
        for c in cands:
            if c in raws:
                r = raws[c]
                systems[s] = {"p50_ms_per_call": [r["sizes"][str(n)]["p50_ms"] for n in NS],
                              "p50_ms_per_question": [r["sizes"][str(n)]["ms_per_question"] for n in NS],
                              "p95_ms_per_call": [r["sizes"][str(n)]["p95_ms"] for n in NS],
                              "reps_per_size": [r["sizes"][str(n)]["reps"] for n in NS]}
                if s == "tez":
                    devices[s] = ("RTX 5080 Laptop GPU: Gemma 4 12B Q8_0 in llama-server :8091 (-ngl 99, all layers on the GPU), "
                                  "called through the `tez serve` runtime on :8787; " +
                                  ("a new ticket state in every call" if c == "tez_fresh_state" else "the identical call repeated"))
                else:
                    devices[s] = f"{r['device']} ({r['dtype']}, autocast={r['autocast']}), {r['info']}"
                used.add(c)
                break
    names = {"tez_fresh_state": "tez (new state every call)",
             "tez": "tez (identical call repeated, Laya's protocol verbatim; prompt-cache hits)",
             "tez_protocol_w3_r20": "tez (identical call repeated, 3 warm-ups, 20 calls)",
             "laya_cuda_protocol_w3_r20": "laya (3 warm-ups, 20 calls)",
             "laya-multilingual_cuda_protocol_w3_r20": "laya-multilingual (3 warm-ups, 20 calls)"}
    for c, r in raws.items():
        if c not in used:
            extras[names.get(c, c)] = {"device": r.get("device") or "cuda (llama-server)",
                                       "p50_ms_per_call": [r["sizes"][str(n)]["p50_ms"] for n in NS],
                                       "p50_ms_per_question": [r["sizes"][str(n)]["ms_per_question"] for n in NS],
                                       "p95_ms_per_call": [r["sizes"][str(n)]["p95_ms"] for n in NS],
                                       "reps_per_size": [r["sizes"][str(n)]["reps"] for n in NS]}
    t4 = OUT / "laya_published" / "t4_colab_latency.json"
    pub = json.loads(t4.read_text(encoding="utf-8")) if t4.exists() else None
    doc = {
        "meta": {
            "title": "p50 latency per call at 1, 5, 10, 50 questions on one state",
            "date": datetime.date.today().isoformat(),
            "script": "experiments/vs_laya_speed.py",
            "protocol": ("Laya's latency protocol (build_benchmark_nb.py section 7 / bench_latency.py): Stripe payout ticket "
                         "state, questions alternate a 3-option choice and a noul, warm-up calls then timed calls per size; "
                         "p50 per call, p50/n per question. Main entries: 10 warm-up calls and 100 timed calls per size (30 at "
                         "50 questions), because single-question GPU calls on this laptop jitter (Laya 1 question: 25-97 ms "
                         "over 20 calls). Laya's exact 3 warm-ups / 20 calls were run first and are in extras."),
            "hardware": {"gpu": "NVIDIA GeForce RTX 5080 Laptop GPU 16 GB (Windows WDDM)", "cpu": "Intel Core Ultra 9 275HX (24 cores)",
                         "os": platform.platform()},
            "device_per_system": devices,
            "tez_note": ("tez serve reads each question separately (one llama-server /completion per question; above 26 "
                         "options a tournament), so a call costs about n x one decision, and the state is re-read for every "
                         "question because it sits after the question in the prompt. systems['tez'] puts a new ticket id at "
                         "the start of the subject and the message in every call: the state is never in llama.cpp's prompt "
                         "cache (the instruction+options prefix is, as in production), i.e. the cost of deciding a new ticket. "
                         "Laya's protocol repeats the identical call; an encoder pays that in full every time, but llama.cpp "
                         "serves it from its prompt cache (at 1 question only the last token is recomputed: ~41 ms vs ~139 ms), "
                         "so that measurement is kept in extras and is not a fair comparison."),
            "laya_note": "Agent.system_one scores all questions of a call in one batched encoder forward pass.",
            "gpu_sharing": ("llama-server keeps ~14 GB of the 16 GB resident; Laya was timed on the same GPU while llama-server "
                            "was idle (Windows WDDM pages the idle process's memory out; torch reported 13-14 GB free after the "
                            "Laya load), see device_per_system. No other benchmark job ran during any timing run (the CPU "
                            "scoring jobs were suspended)."),
            "noise": ("Laptop GPU under Windows: the same Laya measurement moved by up to ~50% at 1 question between two "
                      "sessions (laya 54 vs 36 ms, laya-multilingual 21 vs 30 ms); both sessions are kept (main = 100 calls, "
                      "extras = Laya's 20 calls)."),
            "raw": "results/vs_laya/speed_rows/*.json (every timed sample)",
        },
        "questions_per_call": NS,
        "systems": systems,
        "extras": extras,
        "laya_published_t4": ({s: {"p50_ms_per_call": [pub["latency"][s]["%d_questions" % n]["p50_ms"] for n in NS],
                                   "p50_ms_per_question": [pub["latency"][s]["%d_questions" % n]["ms_per_question"] for n in NS]}
                               for s in ("laya", "laya-multilingual")} if pub else None),
    }
    (OUT / "speed_per_call.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'speed_per_call.json'}")
    for s, v in systems.items():
        print(f"  {s:18s} per call {v['p50_ms_per_call']}  per question {v['p50_ms_per_question']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["tez", "laya", "laya-multilingual"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--url", default="http://127.0.0.1:8787/v1/systemone")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20, help="timed calls per size (Laya's protocol: 20)")
    ap.add_argument("--reps50", type=int, default=None, help="timed calls at 50 questions (default: --reps)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--fresh-state", action="store_true", help="tez: a new ticket state on every call (no prompt-cache hit on the state)")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.system == "tez":
        out = run_tez(a.url, a.warmup, a.reps, a.fresh_state, a.reps50)
        out["date"] = datetime.datetime.now().isoformat(timespec="seconds")
        (RAW / ("tez_fresh_state.json" if a.fresh_state else "tez.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    elif a.system:
        out = run_laya(a.system, a.device, a.warmup, a.reps, a.threads, a.reps50)
        out["date"] = datetime.datetime.now().isoformat(timespec="seconds")
        dev = "cuda" if out["device"].startswith("cuda") else "cpu"
        (RAW / f"{a.system}_{dev}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
