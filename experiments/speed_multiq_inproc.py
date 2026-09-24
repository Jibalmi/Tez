"""Exp 1 (in-process) -- many questions about one state through llama.dll, no HTTP: speed on Laya's protocol and
typed-decisions accuracy, with the batch API that llama-server does not expose.

Arms (letters readout = logits of the bare option-letter tokens at the answer position; exact, nothing floored):
  seq_today        today's layout, one question after the other on sequence 0, reusing the longest common token
                   prefix with the previous prompt by rolling the KV back (what llama-server's cache_prompt does)
  seq_statefirst   the state-first layout the same way: the state is evaluated once per call
  batch_today      today's layout, every question on its own sequence, all in one batch (no prefix sharing)
  batch_statefirst the state-first layout: the call's common token prefix (instructions + state) is evaluated ONCE on
                   sequence 0 and copied to sequences 1..n-1 (llama_memory_seq_cp, unified KV: no copy of the
                   data), then every question's suffix goes in ONE llama_decode with an output at each suffix end
  onepass_markers  exploratory: state first, then all questions in one sequence, each followed by "Answer n:";
                   the letter is read at every marker (" A" or "A", as probe_multiq.py) -- no per-question template
                   tail, one forward pass, but each question sees the ones before it

Modes
  laya  Laya's protocol (new ticket id per call; 1 / 5 / 10 / 50 questions), arms interleaved call by call, --gap s
        of idle GPU after each call (the laptop GPU throttles under continuous load)
  td    typed-decisions test split, 400 rows x 5 questions; one call per row; accuracy per arm (+ agreement with the
        HTTP run of the same layout when --compare gives its rows file)

Reproduce (llama-server on :8091 stopped: the 12B does not fit twice on 16 GB):
  python experiments/speed_multiq_inproc.py --mode laya --arms seq_today,seq_statefirst,batch_statefirst --tag 12b_inproc
  python experiments/speed_multiq_inproc.py --mode td --arms batch_statefirst,onepass_markers --tag 12b_inproc
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import speed_common as sc
import speed_llama as L

HEAD_ONEPASS = ("You are a decision engine. Read the input, then answer every question below with the single letter of "
                "its best option.")


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Engine:
    def __init__(self, model, template, n_ctx, n_batch, n_seq_max, n_ubatch=512):
        self.llm = L.Llama(model, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, n_seq_max=n_seq_max, n_outputs_max=64)
        self.template = template
        self.ids = self.llm.letter_ids(sc.LETTERS)
        self.sp_ids = [self._sp(Lt) for Lt in sc.LETTERS]
        self.cached: list[int] = []            # tokens currently on sequence 0 (sequential arms)

    def _sp(self, letter):
        t = self.llm.tokenize(" " + letter, False, False)
        return t[-1]

    def reset(self):
        self.llm.clear()
        self.cached = []

    # ---------------------------------------------------------------- sequential with rollback (= cache_prompt)
    def seq_one(self, prompt, k):
        llm = self.llm
        T = llm.tokenize(prompt, add_special=True)
        n = lcp(self.cached, T)
        if n == len(T):
            n -= 1                                # always evaluate at least the last token
        if n < len(self.cached) or not self.cached:
            llm.seq_rm(0, n, -1)                  # (an empty record means another arm used sequence 0: drop it)
        outs = llm.eval_seq(T[n:], 0, n)
        self.cached = T
        return llm.logits_at(outs[-1], self.ids[:k]), len(T) - n

    def run_seq(self, prompts, ks):
        z, ev = [], 0
        for p, k in zip(prompts, ks):
            zz, e = self.seq_one(p, k)
            z.append(zz)
            ev += e
        return z, ev

    # ---------------------------------------------------------------- one batch, shared prefix via seq_cp
    def run_batch(self, prompts, ks, share=True):
        llm = self.llm
        llm.clear()
        self.cached = []
        toks = [llm.tokenize(p, add_special=True) for p in prompts]
        n = len(toks)
        Lp = 0
        if share and n > 1:
            Lp = min(len(t) for t in toks)
            for t in toks[1:]:
                Lp = min(Lp, lcp(toks[0], t))
            Lp = max(0, min(Lp, min(len(t) for t in toks) - 1))
        ev = 0
        if Lp:
            llm.eval_seq(toks[0][:Lp], 0, 0, want_last=False)
            ev += Lp
            for j in range(1, n):
                llm.seq_cp(0, j)
        z = [None] * n
        # pack suffixes into decodes of at most n_batch tokens (and n_seq_max sequences)
        pending, bt, bp, bs, bo = [], [], [], [], []
        for j, t in enumerate(toks):
            rest = t[Lp:]
            if bt and len(bt) + len(rest) > llm.cap:
                self._flush(bt, bp, bs, bo, pending, z, ks)
                pending, bt, bp, bs, bo = [], [], [], [], []
            bt += rest
            bp += list(range(Lp, len(t)))
            bs += [j] * len(rest)
            bo += [False] * (len(rest) - 1) + [True]
            pending.append(j)
            ev += len(rest)
        if bt:
            self._flush(bt, bp, bs, bo, pending, z, ks)
        llm.clear()
        return z, ev

    def _flush(self, bt, bp, bs, bo, pending, z, ks):
        if len(bt) > self.llm.cap:
            raise RuntimeError("one suffix longer than n_batch")
        opos = self.llm.decode(bt, bp, bs, bo)
        for p, j in zip(opos, pending):
            z[j] = self.llm.logits_at(p, self.ids[: ks[j]])
        for j in pending:                        # free the finished sequences (the shared prefix stays for the rest)
            self.llm.seq_rm(j)

    # ---------------------------------------------------------------- one sequence, read at markers
    def run_onepass(self, cases):
        llm = self.llm
        llm.clear()
        self.cached = []
        pre, post = sc.TEMPLATES[self.template]
        parts = [pre + HEAD_ONEPASS + "\n\nInput:\n" + sc.state_text(cases[0]["state"]) + "\n"]
        for n_, c in enumerate(cases, 1):
            parts.append(f"\nQuestion {n_} ({c['qtype']}): {c['instructions']}\nOptions:\n{sc.option_lines(c['options'])}\nAnswer {n_}:")
        toks, marks = [], []
        for i, p in enumerate(parts):
            t = llm.tokenize(p, add_special=(i == 0))
            toks += t
            if i > 0:
                marks.append(len(toks) - 1)
        outs_flags = [False] * len(toks)
        for m in marks:
            outs_flags[m] = True
        # chunk by n_batch; outputs are indexed per decode call
        z = []
        pos = 0
        while pos < len(toks):
            chunk = toks[pos: pos + llm.cap]
            fl = outs_flags[pos: pos + llm.cap]
            for p in llm.decode(chunk, list(range(pos, pos + len(chunk))), [0] * len(chunk), fl):
                z.append(llm.logits_at(p))
            pos += len(chunk)
        out = []
        for c, full in zip(cases, z):
            k = len(c["options"])
            a = full[self.ids[:k]]
            b = full[self.sp_ids[:k]]
            out.append(np.logaddexp(a, b))
        llm.clear()
        return out, len(toks)

    def run(self, arm, cases):
        ks = [len(c["options"]) for c in cases]
        if arm == "seq_today":
            return self.run_seq([sc.prompt_today(c, self.template) for c in cases], ks)
        if arm == "seq_statefirst":
            return self.run_seq([sc.prompt_statefirst(c, self.template) for c in cases], ks)
        if arm == "batch_today":
            return self.run_batch([sc.prompt_today(c, self.template) for c in cases], ks, share=True)
        if arm == "batch_statefirst":
            return self.run_batch([sc.prompt_statefirst(c, self.template) for c in cases], ks, share=True)
        if arm == "onepass_markers":
            return self.run_onepass(cases)
        raise ValueError(arm)


def mode_laya(a, eng):
    arms = a.arms.split(",")
    out = {"arms": {x: {} for x in arms}, "gpu": {}}
    counter = a.seed_offset
    for n in sc.NS:
        qs = sc.laya_questions(n, a.unique)
        rec = {x: dict(walls=[], ev=[]) for x in arms}
        reps = a.reps if n < 50 else a.reps50
        out["gpu"][str(n)] = {"before": sc.wait_cool(a.cool, 240) if a.cool else sc.gpu_now()}
        with sc.GpuMonitor() as mon:
            for i in range(a.warmup + reps):
                for arm in arms:
                    counter += 1
                    st = sc.fresh_state(counter)
                    cases = [sc.wire_to_case(qid, q, st) for qid, q in qs.items()]
                    t = time.perf_counter()
                    z, ev = eng.run(arm, cases)
                    _ = [int(np.argmax(x)) for x in z]
                    wall = (time.perf_counter() - t) * 1000
                    if a.gap or a.duty:
                        time.sleep(max(a.gap, a.duty * wall / 1000.0))   # idle time proportional to the call: bounded GPU duty cycle
                    if i >= a.warmup:
                        rec[arm]["walls"].append(wall)
                        rec[arm]["ev"].append(ev)
        out["gpu"][str(n)]["during"] = mon.summary()
        for arm, r in rec.items():
            st_ = sc.per_call_stats(r["walls"], n)
            st_["tokens_evaluated_per_call_p50"] = float(np.median(r["ev"]))
            st_["samples_ms"] = [round(w, 2) for w in r["walls"]]
            out["arms"][arm][str(n)] = st_
            g = out["gpu"][str(n)]["during"]
            print(f"{a.tag:14s} {arm:17s} {n:2d}q  p50 {st_['p50']:8.1f}  p95 {st_['p95']:8.1f}  per q {st_['ms_per_question_p50']:6.2f}"
                  f"  tokens {st_['tokens_evaluated_per_call_p50']:.0f}  sm {g.get('sm_mhz_busy_p50')} {g.get('temp_c_p50')}C thr {g.get('thermal_throttle_frac')}", flush=True)
    return out


def mode_td(a, eng):
    cases = sc.typed_decisions("test")
    rows = sc.rows_of(cases)
    if a.limit_rows:
        rows = dict(list(rows.items())[: a.limit_rows])
    out = {}
    for arm in a.arms.split(","):
        recs, walls = [], []
        t0 = time.perf_counter()
        with sc.GpuMonitor() as mon:
            for rid, row in rows.items():
                t = time.perf_counter()
                z, ev = eng.run(arm, row)
                walls.append((time.perf_counter() - t) * 1000)
                for c, zz in zip(row, z):
                    recs.append(dict(id=c["id"], gold=c["gold"], qtype=c["qtype"], pred=int(np.argmax(zz)), z=[float(x) for x in zz], ev=ev))
        preds = [r["pred"] for r in recs]
        golds = [r["gold"] for r in recs]
        kept = [c for c in cases if c["id"] in {r["id"] for r in recs}]
        s = {"n_decisions": len(recs), "accuracy": round(sc.accuracy(preds, golds), 4), "by_type": sc.by_type(kept, preds),
             "per_row_call_ms": sc.stats(walls), "wall_s": round(time.perf_counter() - t0, 1), "gpu": mon.summary()}
        if a.compare:
            for comp in a.compare.split(","):
                other = {json.loads(l)["id"]: json.loads(l) for l in open(sc.OUT / comp, encoding="utf-8")}
                pairs = [(r, other[r["id"]]) for r in recs if r["id"] in other]
                s[f"vs_{comp}"] = {"n": len(pairs), "agreement": round(float(np.mean([r["pred"] == o["pred"] for r, o in pairs])), 4),
                                   "mcnemar": sc.mcnemar([r["pred"] == r["gold"] for r, _ in pairs], [o["pred"] == o["gold"] for _, o in pairs]),
                                   "other_accuracy": round(float(np.mean([o["pred"] == o["gold"] for _, o in pairs])), 4)}
        out[arm] = s
        (sc.OUT / f"td_rows_{a.tag}_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        print(arm, json.dumps({k: v for k, v in s.items()}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["laya", "td"], required=True)
    ap.add_argument("--arms", default="seq_today,seq_statefirst,batch_statefirst")
    ap.add_argument("--model", default=sc.GEMMA)
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-batch", type=int, default=2048)
    ap.add_argument("--n-seq-max", type=int, default=64)
    ap.add_argument("--n-ubatch", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps50", type=int, default=8)
    ap.add_argument("--gap", type=float, default=0.3)
    ap.add_argument("--cool", type=float, default=0.0)
    ap.add_argument("--seed-offset", type=int, default=300_000)
    ap.add_argument("--unique", action="store_true", help="laya: every question of a call distinct (see speed_common.laya_questions)")
    ap.add_argument("--duty", type=float, default=0.0, help="after each call also idle duty x its duration (1.0 = at most 50%% GPU duty cycle)")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--compare", default="", help="td: comma-separated td_rows_*.jsonl files to compare predictions with")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    model = a.model if a.model.startswith("C:") else str(sc.ROOT / a.model)
    eng = Engine(model, a.template, a.n_ctx, a.n_batch, a.n_seq_max, a.n_ubatch)
    meta = {"script": "experiments/speed_multiq_inproc.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(a),
            "params": eng.llm.params, "load_s": round(eng.llm.load_s, 1), "n_layer": eng.llm.n_layer}
    with sc.Guard(inproc=True) as guard:
        res = mode_laya(a, eng) if a.mode == "laya" else mode_td(a, eng)
    g = guard.result()
    print("gpu_guard contaminated:", g["contaminated"], g["reasons_before"] + g["reasons_after"], flush=True)
    sc.dump(f"{a.mode}_{a.tag}.json", {"meta": meta, **res, "gpu_guard": g})


if __name__ == "__main__":
    main()
