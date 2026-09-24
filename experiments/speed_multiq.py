"""Exp 1 -- many questions about one state, over HTTP: speed on Laya's protocol and accuracy on typed-decisions.

Layouts (speed_common):
  today        instructions + question + options first, state LAST: every question re-reads the state
  statefirst   instructions + state FIRST, then the question: with prompt caching the state is evaluated once per
               call and every further question only evaluates its own suffix (question + options + template tail)
Transports:
  tez          POST <tez>/v1/systemone (the runtime as deployed; layout today only)
  http         one llama-server /completion per question, sequential (slot 0 keeps the state in its cache)
  http-par     all questions of a call sent concurrently, one thread each (needs llama-server -np N); with
               --prime the state prefix is evaluated once first, then the questions are sent
  http-multi   one /completion whose "prompt" is the list of all the call's prompts (the server spreads them over slots)

Modes:
  laya   Laya's latency protocol (vs_laya_speed.py): payout ticket, questions alternate a 3-option choice and a noul,
         1 / 5 / 10 / 50 questions per call, a NEW ticket id every call (the state is never a cache hit), warm-up calls
         then timed calls; p50 / p95 per call and per question.
  td     typed-decisions test split (400 rows x 5 questions = 2,000 decisions), zero-shot letters, accuracy + per-row
         and per-question latency (all questions of a row form one call).

Reproduce (the production llama-server on :8091 and tez serve on :8787, -np 1):
  python experiments/speed_multiq.py --mode laya --layout today      --transport tez  --tag prod_today_tez
  python experiments/speed_multiq.py --mode laya --layout today      --transport http --tag prod_today_http
  python experiments/speed_multiq.py --mode laya --layout statefirst --transport http --tag prod_statefirst_http
  python experiments/speed_multiq.py --mode td   --layout today      --transport http --tag prod_today_http
  python experiments/speed_multiq.py --mode td   --layout statefirst --transport http --tag prod_statefirst_http
Parallel slots (a side server, e.g. llama-server ... -np 8 -kvu --port 8095):
  python experiments/speed_multiq.py --mode laya --layout statefirst --transport http-par --url http://127.0.0.1:8095 --tag np8_statefirst_par
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import speed_common as sc

_tls = threading.local()


def tsession():
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = sc.session()
    return s


def build_prompts(cases, layout, template):
    if layout == "today":
        return [sc.prompt_today(c, template) for c in cases]
    return [sc.prompt_statefirst(c, template) for c in cases]


def run_call(cases, layout, transport, url, template, n_probs, pool=None, prime=False, tez_url=None, wire=None):
    """Decide all `cases` (one state). Returns (wall_ms, per-question records)."""
    if transport == "tez":
        body = {"model": "tez-latest", "state": wire["state"], "questions": wire["questions"]}
        t = time.perf_counter()
        r = tsession().post(f"{tez_url}/v1/systemone", json=body, timeout=900)
        wall = (time.perf_counter() - t) * 1000
        j = r.json()
        recs = []
        for c in cases:
            a = j["answers"][c["id"]]
            if a["type"] == "noul":
                p = np.array([1 - a["noul"], a["noul"]])
            else:
                keys = list(a["probabilities"])
                p = np.array([a["probabilities"][k] for k in keys])
            recs.append(dict(pred=int(np.argmax(p)), p=p.tolist()))
        return wall, recs, {"server_ms": j["tez"]["latency_ms"]}
    prompts = build_prompts(cases, layout, template)
    ks = [len(c["options"]) for c in cases]
    t = time.perf_counter()
    extra = {}
    if transport == "http":
        outs = [sc.score_http(url, p, k, n_probs, True, tsession()) for p, k in zip(prompts, ks)]
    elif transport == "http-par":
        if prime:
            pre = sc.statefirst_prefix(cases[0]["state"], template)
            tp = time.perf_counter()
            tsession().post(f"{url}/completion", json={"prompt": pre, "n_predict": 0, "cache_prompt": True}, timeout=600)
            extra["prime_ms"] = (time.perf_counter() - tp) * 1000
        outs = list(pool.map(lambda pk: sc.score_http(url, pk[0], pk[1], n_probs, True, tsession()), zip(prompts, ks)))
    elif transport == "http-multi":
        r = tsession().post(f"{url}/completion", json=sc.completion_body(prompts, n_probs, True), timeout=900)
        data = r.json()
        data = data if isinstance(data, list) else [data]
        data = sorted(data, key=lambda d: d.get("index", 0))
        outs = []
        for d, k in zip(data, ks):
            tm = d.get("timings", {}) or {}
            outs.append(dict(z=sc.letters_from_tops(d["completion_probabilities"][0]["top_logprobs"], k), http_ms=None,
                             prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"), cache_n=tm.get("cache_n")))
    else:
        raise ValueError(transport)
    wall = (time.perf_counter() - t) * 1000
    recs = [dict(pred=int(np.argmax(o["z"])), z=o["z"].tolist(), http_ms=o.get("http_ms"), prompt_n=o.get("prompt_n"),
                 prompt_ms=o.get("prompt_ms"), cache_n=o.get("cache_n")) for o in outs]
    return wall, recs, extra


def mode_laya(args, pool):
    """Every arm (transport:layout) answers the same sequence of calls, interleaved call by call, so thermal drift
    of the laptop GPU hits all arms equally. Each call gets a new ticket id (never a prompt-cache hit on the state)."""
    arms = [tuple(a.split(":")) for a in args.arms.split(",")] if args.arms else [(args.transport, args.layout)]
    out = {"arms": {f"{t}:{l}": {} for t, l in arms}, "gpu": {}}
    counter = args.seed_offset
    for n in sc.NS:
        qs = sc.laya_questions(n, args.unique)
        rec = {f"{t}:{l}": dict(walls=[], tok=[], pms=[], extras=[]) for t, l in arms}
        reps = args.reps if n < 50 else args.reps50
        out["gpu"][str(n)] = {"before": sc.wait_cool(args.cool, 240) if args.cool else sc.gpu_now()}
        with sc.GpuMonitor() as mon:
            for i in range(args.warmup + reps):
                for t, l in arms:
                    counter += 1
                    st = sc.fresh_state(counter)
                    cases = [sc.wire_to_case(qid, q, st) for qid, q in qs.items()]
                    wall, recs, extra = run_call(cases, l, t, args.url, args.template, args.n_probs, pool,
                                                 args.prime, args.tez, {"state": st, "questions": qs})
                    if args.gap or args.duty:
                        time.sleep(max(args.gap, args.duty * wall / 1000.0))   # idle time proportional to the call: bounded GPU duty cycle
                    if i >= args.warmup:
                        r = rec[f"{t}:{l}"]
                        r["walls"].append(wall)
                        r["tok"].append(sum(x.get("prompt_n") or 0 for x in recs))
                        r["pms"].append(sum(x.get("prompt_ms") or 0 for x in recs))
                        r["extras"].append(extra)
        out["gpu"][str(n)]["during"] = mon.summary()
        for key, r in rec.items():
            st_ = sc.per_call_stats(r["walls"], n)
            st_["prompt_tokens_evaluated_per_call_p50"] = float(np.median(r["tok"]))
            st_["sum_prompt_ms_per_call_p50"] = round(float(np.median(r["pms"])), 2)
            ex = r["extras"]
            if ex and ex[0].get("server_ms") is not None:
                st_["tez_reported_ms_p50"] = float(np.median([e["server_ms"] for e in ex]))
            if ex and ex[0].get("prime_ms") is not None:
                st_["prime_ms_p50"] = round(float(np.median([e["prime_ms"] for e in ex])), 2)
            st_["samples_ms"] = [round(w, 2) for w in r["walls"]]
            out["arms"][key][str(n)] = st_
            print(f"{args.tag:24s} {key:22s} {n:2d}q  p50 {st_['p50']:8.1f}  p95 {st_['p95']:8.1f}  per q {st_['ms_per_question_p50']:6.2f}"
                  f"  tokens/call {st_['prompt_tokens_evaluated_per_call_p50']:.0f}  sm {out['gpu'][str(n)]['during'].get('sm_mhz_busy_p50')}"
                  f"  {out['gpu'][str(n)]['during'].get('temp_c_p50')}C", flush=True)
    return out


def mode_td(args, pool):
    cases = sc.typed_decisions("test")
    rows = sc.rows_of(cases)
    if args.limit_rows:
        rows = dict(list(rows.items())[: args.limit_rows])
    recs_all, walls, per_q_ms = [], [], []
    t0 = time.perf_counter()
    mon = sc.GpuMonitor().__enter__()
    for r_i, (rid, row) in enumerate(rows.items()):
        wall, recs, _ = run_call(row, args.layout, args.transport, args.url, args.template, args.n_probs, pool, args.prime)
        walls.append(wall)
        for c, r in zip(row, recs):
            r.update(id=c["id"], gold=c["gold"], qtype=c["qtype"], k=len(c["options"]))
            recs_all.append(r)
            if r.get("http_ms") is not None:
                per_q_ms.append(r["http_ms"])
        if (r_i + 1) % 100 == 0:
            acc = np.mean([r["pred"] == r["gold"] for r in recs_all])
            print(f"  {r_i + 1}/{len(rows)} rows  acc {acc:.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
    mon.__exit__(None, None, None)
    preds = [r["pred"] for r in recs_all]
    golds = [r["gold"] for r in recs_all]
    kept = [c for c in cases if c["id"] in {r["id"] for r in recs_all}]
    summ = {"n_decisions": len(recs_all), "n_rows": len(rows), "accuracy": round(sc.accuracy(preds, golds), 4),
            "by_type": sc.by_type(kept, preds), "per_row_call_ms": sc.stats(walls), "per_question_http_ms": sc.stats(per_q_ms),
            "prompt_tokens_evaluated_total": int(sum(r.get("prompt_n") or 0 for r in recs_all)),
            "cache_tokens_total": int(sum(r.get("cache_n") or 0 for r in recs_all)),
            "gpu": mon.summary()}
    Path(sc.OUT / f"td_rows_{args.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs_all), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items()}, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["laya", "td"], required=True)
    ap.add_argument("--layout", choices=["today", "statefirst"], default="today")
    ap.add_argument("--transport", choices=["tez", "http", "http-par", "http-multi"], default="http")
    ap.add_argument("--arms", default="", help="laya: comma-separated transport:layout arms, interleaved call by call")
    ap.add_argument("--cool", type=float, default=0.0, help="wait until the GPU is at or below this many C before each size")
    ap.add_argument("--gap", type=float, default=0.0, help="seconds of idle GPU after every call (keeps the laptop GPU out of thermal throttling)")
    ap.add_argument("--unique", action="store_true", help="laya: every question of a call distinct (see speed_common.laya_questions)")
    ap.add_argument("--duty", type=float, default=0.0, help="after each call also idle duty x its duration (1.0 = at most 50%% GPU duty cycle)")
    ap.add_argument("--url", default=sc.PROD_URL)
    ap.add_argument("--tez", default=sc.TEZ_URL)
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--prime", action="store_true", help="http-par: evaluate the state prefix once before the questions")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--reps50", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=100_000)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.transport == "tez" and args.layout != "today":
        ap.error("the runtime only has the 'today' layout")
    pool = ThreadPoolExecutor(max_workers=args.workers) if ("http-par" in args.transport or "http-par" in args.arms) else None
    pr = sc.props(args.url)
    meta = {"script": "experiments/speed_multiq.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
            "llama_model_path": pr.get("model_path"), "llama_build": pr.get("build_info"), "total_slots": pr.get("total_slots")}
    with sc.Guard(port=sc.port_of(args.url)) as guard:
        res = mode_laya(args, pool) if args.mode == "laya" else mode_td(args, pool)
    g = guard.result()
    print("gpu_guard contaminated:", g["contaminated"], g["reasons_before"] + g["reasons_after"], flush=True)
    sc.dump(f"{args.mode}_{args.tag}.json", {"meta": meta, **res, "gpu_guard": g})


if __name__ == "__main__":
    main()
