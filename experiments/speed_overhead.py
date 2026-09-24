"""Exp 4 -- where a single question's latency goes: tez serve (FastAPI + Python) versus llama-server.

One question (Laya's 3-option choice question) on a NEW payout ticket every call (vs_laya_speed.fresh_state), so the
instruction+options prefix comes from llama.cpp's prompt cache and the state is evaluated, as in production. The modes
are interleaved call by call so drift hits them equally:

  tez_serve      POST :8787/v1/systemone (the runtime as deployed); client round trip + the server's own latency_ms
  engine         the same Tez engine in this process (no FastAPI/uvicorn), backend calls timed by wrapping _request
  direct_np200   POST :8091/completion with the runtime's exact prompt, n_probs 200 (what the runtime sends)
  direct_np20    ... n_probs 20       direct_np1 ... n_probs 1     direct_np0 ... n_probs 0 (no probabilities: timing only)
  health         GET /health on both servers (the bare HTTP floor)

For every mode: client round-trip ms, llama-server prompt_ms / predicted_ms / prompt_n from `timings`, response bytes.

Reproduce (production servers running: llama-server :8091 with the brief's command, `tez serve` :8787):
  python experiments/speed_overhead.py --reps 100 --out results/speed/overhead.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import speed_common as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gap", type=float, default=0.3, help="seconds between requests (keeps the laptop GPU out of thermal throttling)")
    ap.add_argument("--llama", default=sc.PROD_URL)
    ap.add_argument("--tez", default=sc.TEZ_URL)
    ap.add_argument("--out", default=str(sc.OUT / "overhead.json"))
    args = ap.parse_args()

    from tez.engine import Tez
    from tez.prompt import build_prompt
    from tez.schema import parse_question

    health = sc.S.get(f"{args.tez}/healthz", timeout=30).json()
    pr = sc.props(args.llama)
    eng = Tez(backend=args.llama, template="gemma4")
    backend = eng.backend
    timing = {"backend_ms": 0.0, "calls": 0}
    orig = backend._request

    def timed_request(method, path, body=None, timeout=None):
        t = time.perf_counter()
        try:
            return orig(method, path, body, timeout)
        finally:
            if path == "/completion":
                timing["backend_ms"] += (time.perf_counter() - t) * 1000
                timing["calls"] += 1
    backend._request = timed_request

    q = sc.Q_CHOICE
    Q = parse_question("q0", q)
    rec = {m: [] for m in ("tez_serve", "engine", "direct_np200", "direct_np20", "direct_np1", "direct_np0",
                           "health_tez", "health_llama")}
    counter = 10_000

    def next_state():
        nonlocal counter
        counter += 1
        return sc.fresh_state(counter)

    s_tez, s_ll = sc.session(), sc.session()
    guard = sc.Guard(port=sc.port_of(args.llama)).__enter__()
    mon = sc.GpuMonitor().__enter__()
    for it in range(args.warmup + args.reps):
        keep = it >= args.warmup
        # 1. the runtime as deployed
        body = {"model": "tez-latest", "state": next_state(), "questions": {"q0": q}}
        t = time.perf_counter()
        r = s_tez.post(f"{args.tez}/v1/systemone", json=body, timeout=600)
        ms = (time.perf_counter() - t) * 1000
        j = r.json()
        if keep:
            rec["tez_serve"].append(dict(ms=ms, server_ms=j["tez"]["latency_ms"], bytes=len(r.content)))
        time.sleep(args.gap)
        # 2. the engine in-process (same code path minus FastAPI/uvicorn/the client hop)
        timing.update(backend_ms=0.0, calls=0)
        st = next_state()
        t = time.perf_counter()
        eng.decide(st, questions={"q0": q})
        ms = (time.perf_counter() - t) * 1000
        if keep:
            rec["engine"].append(dict(ms=ms, backend_ms=timing["backend_ms"], calls=timing["calls"]))
        time.sleep(args.gap)
        # 3. direct llama-server calls with the runtime's exact prompt
        for n_probs in (200, 20, 1, 0):
            prompt = build_prompt(Q, next_state(), "gemma4")
            b = sc.completion_body(prompt, n_probs, True)
            t = time.perf_counter()
            r = s_ll.post(f"{args.llama}/completion", json=b, timeout=600)
            t_http = (time.perf_counter() - t) * 1000
            t2 = time.perf_counter()
            d = r.json()
            parse_ms = (time.perf_counter() - t2) * 1000
            tm = d.get("timings", {})
            if keep:
                rec[f"direct_np{n_probs}"].append(dict(ms=t_http + parse_ms, http_ms=t_http, parse_ms=parse_ms, bytes=len(r.content),
                                                       prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"),
                                                       predicted_ms=tm.get("predicted_ms"), cache_n=tm.get("cache_n")))
            time.sleep(args.gap)
        # 4. bare HTTP floors
        for name, url in (("health_tez", f"{args.tez}/healthz"), ("health_llama", f"{args.llama}/health")):
            t = time.perf_counter()
            (s_tez if name == "health_tez" else s_ll).get(url, timeout=30)
            if keep:
                rec[name].append(dict(ms=(time.perf_counter() - t) * 1000))

    mon.__exit__(None, None, None)
    guard.__exit__(None, None, None)

    def col(mode, key):
        return [r[key] for r in rec[mode] if r.get(key) is not None]

    summary = {}
    for m, rows in rec.items():
        s = {"round_trip_ms": sc.stats(col(m, "ms"))}
        for key in ("server_ms", "backend_ms", "http_ms", "parse_ms", "prompt_ms", "predicted_ms", "prompt_n", "cache_n", "bytes"):
            v = col(m, key)
            if v:
                s[key] = sc.stats(v)
        summary[m] = s
    # decomposition at p50 (medians of each component, same calls)
    p = lambda m, k: summary[m][k]["p50"] if k in summary[m] else None  # noqa: E731
    comp = {
        "llama_compute_ms (prompt_ms + predicted_ms, direct n_probs 200)": round(np.median([r["prompt_ms"] + r["predicted_ms"] for r in rec["direct_np200"]]), 2),
        "direct_http_round_trip_np200_ms": p("direct_np200", "round_trip_ms"),
        "direct_http_round_trip_np0_ms": p("direct_np0", "round_trip_ms"),
        "n_probs_200_cost_ms (np200 - np0 round trip)": round(p("direct_np200", "round_trip_ms") - p("direct_np0", "round_trip_ms"), 2),
        "engine_total_ms": p("engine", "round_trip_ms"),
        "engine_backend_http_ms": p("engine", "backend_ms"),
        "engine_python_ms (engine - backend http)": round(np.median([r["ms"] - r["backend_ms"] for r in rec["engine"]]), 2),
        "tez_serve_client_round_trip_ms": p("tez_serve", "round_trip_ms"),
        "tez_serve_reported_latency_ms": p("tez_serve", "server_ms"),
        "fastapi_uvicorn_client_ms (round trip - reported)": round(np.median([r["ms"] - r["server_ms"] for r in rec["tez_serve"]]), 2),
        "health_round_trip_tez_ms": p("health_tez", "round_trip_ms"),
        "health_round_trip_llama_ms": p("health_llama", "round_trip_ms"),
    }
    doc = {"meta": {"script": "experiments/speed_overhead.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "reps": args.reps,
                    "question": q, "state": "vs_laya_speed.fresh_state(i), new ticket id per call",
                    "llama_model_path": pr.get("model_path"), "llama_build": pr.get("build_info"), "tez_health": health},
           "decomposition_p50": comp, "modes": summary, "gpu": mon.summary(), "gpu_guard": guard.result()}
    sc.dump(args.out.split("/")[-1].split("\\")[-1], doc)
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()
