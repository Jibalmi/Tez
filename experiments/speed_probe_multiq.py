"""Exp 2 -- many questions with probes on the depth-pruned Qwen3.5-4B (24 of 32 blocks).

Feature layouts (last-token state, output-normed layer 24, as llama-server `--embeddings --pooling last` returns it):
  stateonly  '<|im_start|>user\\nInput:\\n<state><|im_end|>...' -- ONE vector per state, read by every question's probe
             (probe_multiq.py layout B, there on the bf16 transformers model: 0.748 at layer 24)
  qprompt    today's layout (question + options first, state last): one vector per question (0.793 served, BENCHMARKS)
  sfq        state first, then the question: one vector per question, but the state is a shared prefix, so a call
             evaluates the state once and each question's suffix (needs sequence copies: in-process only)

Subcommands
  extract   served features over HTTP for a layout (train + test splits) -> results/speed/probe_feats_<tag>.npz
  extract-inproc  the same through llama.dll (sfq needs it: prefix sharing is not safe over llama-server for Qwen3.5)
  accuracy  per-question logistic probes (C by 3-fold CV on train from {0.05, 0.5, 5}, as pruned_server_bench.py) on
            the typed-decisions test split (2,000 decisions); also re-scores results/probe_cache_served_L24.npz (qprompt)
  latency   Laya's protocol (payout ticket, new ticket id per call, alternating 3-option choice / noul questions),
            N = 1, 2, 5, 10, 20, 50 questions per call; per call: the feature pass(es) + N probe evaluations. The
            probe weights are random with the right shapes (a probe's cost does not depend on its weights; accuracy
            comes from `accuracy`).
            --engine http:   stateonly = one /embedding;  qprompt = N /embedding calls sent concurrently (-np N server)
            --engine inproc: stateonly = one decode;  sfq = state once + N suffixes in one decode (seq copies);
                             qprompt = N full prompts in one batch (no sharing)

Reproduce (GPU free; the 24-block GGUF served on :8095 with --embeddings --pooling last, -np 1, caching off):
  python experiments/speed_probe_multiq.py extract --layout stateonly --url http://127.0.0.1:8095 --tag L24_stateonly
  python experiments/speed_probe_multiq.py accuracy --feats L24_stateonly
  python experiments/speed_probe_multiq.py latency --engine http --layout stateonly --url http://127.0.0.1:8095 --tag L24_http
  python experiments/speed_probe_multiq.py extract-inproc --layout sfq --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --tag L24_sfq
  python experiments/speed_probe_multiq.py latency --engine inproc --layout stateonly,sfq,qprompt --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --tag L24_inproc
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import speed_common as sc

NS_PROBE = [1, 2, 5, 10, 20, 50]


def stateonly_text(state) -> str:
    pre, post = sc.TEMPLATES["qwen3"]
    return f"{pre}Input:\n{sc.state_text(state)}{post}"


def qprompt_text(case) -> str:
    return sc.prompt_today(case, "qwen3")


def sfq_parts(case) -> tuple[str, str]:
    return sc.statefirst_prefix(case["state"], "qwen3"), sc.statefirst_suffix(case["qtype"], case["instructions"], case["options"], "qwen3")


def embed_http(url, content, sess=None):
    s = sess or sc.S
    d = s.post(f"{url}/embedding", json={"content": content, "embd_normalize": -1}, timeout=600).json()
    if isinstance(content, list):
        d = sorted(d, key=lambda x: x.get("index", 0))
        return [np.asarray(x["embedding"][-1] if isinstance(x["embedding"][0], list) else x["embedding"], np.float32) for x in d]
    d = d[0] if isinstance(d, list) else d
    e = d["embedding"]
    return np.asarray(e[-1] if isinstance(e[0], list) else e, np.float32)


# ---------------------------------------------------------------------------------------------- extraction
def cmd_extract(a):
    feats = {}
    for split in ("train", "test"):
        cases = sc.typed_decisions(split)
        t0 = time.perf_counter()
        if a.layout == "stateonly":
            rows = sc.rows_of(cases)
            row_vec = {rid: embed_http(a.url, stateonly_text(row[0]["state"])) for rid, row in rows.items()}
            X = []
            for c in cases:
                qid = str(c["qkey"][1])
                X.append(row_vec[c["id"][: -(len(qid) + 1)]])
        elif a.layout == "qprompt":
            X = [embed_http(a.url, qprompt_text(c)) for c in cases]
        else:
            raise SystemExit("sfq needs extract-inproc")
        feats[split] = np.stack(X)
        print(split, feats[split].shape, f"{time.perf_counter() - t0:.0f}s", flush=True)
    np.savez(sc.OUT / f"probe_feats_{a.tag}.npz", **feats)


def cmd_extract_inproc(a):
    import speed_llama as L
    llm = L.Llama(str(sc.ROOT / a.model), n_ctx=8192, n_batch=4096, n_ubatch=512, n_seq_max=8, embeddings=False,
                  pooling="none", n_outputs_max=16)
    feats = {}
    for split in ("train", "test"):
        cases = sc.typed_decisions(split)
        rows = sc.rows_of(cases)
        vec = {}
        t0 = time.perf_counter()
        for rid, row in rows.items():
            if a.layout == "sfq":
                outs = sfq_call(llm, [sfq_parts(c) for c in row])
            elif a.layout == "stateonly":
                outs = [stateonly_call(llm, row[0]["state"])] * len(row)
            else:
                outs = qprompt_call(llm, [qprompt_text(c) for c in row])
            for c, v in zip(row, outs):
                vec[c["id"]] = v
        feats[split] = np.stack([vec[c["id"]] for c in cases])
        print(split, feats[split].shape, f"{time.perf_counter() - t0:.0f}s", flush=True)
    np.savez(sc.OUT / f"probe_feats_{a.tag}.npz", **feats)


# ---------------------------------------------------------------------------------------------- in-process calls
def stateonly_call(llm, state):
    llm.clear()
    toks = llm.tokenize(stateonly_text(state), add_special=True)
    return llm.last_state(toks, 0, 0)


def sfq_call(llm, parts):
    """parts: [(prefix, suffix)] sharing one prefix. The common token prefix is evaluated once on seq 0 and copied
    to seqs 1..n-1; all suffixes then go in one decode (outputs at each suffix end)."""
    llm.clear()
    toks = [llm.tokenize(p + s, add_special=True) for p, s in parts]
    base = toks[0]
    L = min(len(t) for t in toks)
    for t in toks[1:]:
        i = 0
        while i < L and t[i] == base[i]:
            i += 1
        L = i
    L = max(1, L - 1)
    llm.set_embeddings(False)
    llm.eval_seq(base[:L], 0, 0, want_last=False)
    for j in range(1, len(toks)):
        llm.seq_cp(0, j)
    # every suffix but its last token in one decode (no outputs), then the last tokens in a second decode with
    # embeddings on: libllama outputs every token of a decode while embeddings are on
    bt, bp, bs = [], [], []
    for j, t in enumerate(toks):
        body = t[L:-1]
        bt += body
        bp += list(range(L, len(t) - 1))
        bs += [j] * len(body)
    if bt:
        llm.decode(bt, bp, bs, [False] * len(bt))
    llm.set_embeddings(True)
    opos = llm.decode([t[-1] for t in toks], [len(t) - 1 for t in toks], list(range(len(toks))), [True] * len(toks))
    out_vecs = [llm.embd_at(p) for p in opos]
    llm.set_embeddings(False)
    return out_vecs


def qprompt_call(llm, prompts):
    """Every prompt on its own sequence, bodies packed into decodes of <= n_batch tokens (no outputs), then all last
    tokens in one decode with embeddings on (no prefix sharing: the state is at the end of each prompt)."""
    llm.clear()
    llm.set_embeddings(False)
    toks = [llm.tokenize(p, add_special=True) for p in prompts]
    if len(toks) > llm.n_seq_max:
        raise ValueError("more prompts than sequences")
    bt, bp, bs = [], [], []
    for j, t in enumerate(toks):
        body = t[:-1]
        for s in range(0, len(body), llm.cap):
            piece = body[s: s + llm.cap]
            if len(bt) + len(piece) > llm.cap:
                llm.decode(bt, bp, bs, [False] * len(bt))
                bt, bp, bs = [], [], []
            bt += piece
            bp += list(range(s, s + len(piece)))
            bs += [j] * len(piece)
    if bt:
        llm.decode(bt, bp, bs, [False] * len(bt))
    llm.set_embeddings(True)
    opos = llm.decode([t[-1] for t in toks], [len(t) - 1 for t in toks], list(range(len(toks))), [True] * len(toks))
    vecs = [llm.embd_at(p) for p in opos]
    llm.set_embeddings(False)
    return vecs


# ---------------------------------------------------------------------------------------------- accuracy
def per_question_probe(Xtr, Xte, train, test, Cs=(0.05, 0.5, 5.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    gtr = np.array([c["gold"] for c in train])
    gte = np.array([c["gold"] for c in test])
    qkeys = sorted(set(c["qkey"] for c in train))
    itr = {q: np.array([i for i, c in enumerate(train) if c["qkey"] == q]) for q in qkeys}
    ite = {q: np.array([i for i, c in enumerate(test) if c["qkey"] == q]) for q in qkeys}
    best = None
    for C in Cs:
        s = np.mean([cross_val_score(LogisticRegression(max_iter=3000, C=C), Xtr[itr[q]], gtr[itr[q]], cv=3).mean()
                     for q in qkeys if len(np.unique(gtr[itr[q]])) > 1])
        if best is None or s > best[0]:
            best = (s, C)
    pred = np.zeros(len(test), int)
    for q in qkeys:
        y = gtr[itr[q]]
        if len(np.unique(y)) < 2:
            pred[ite[q]] = y[0]
            continue
        pred[ite[q]] = LogisticRegression(max_iter=3000, C=best[1]).fit(Xtr[itr[q]], y).predict(Xte[ite[q]])
    return dict(accuracy=round(float((pred == gte).mean()), 4), C=best[1], train_cv=round(float(best[0]), 4),
                by_type=sc.by_type(test, pred.tolist())), pred


def cmd_accuracy(a):
    train, test = sc.typed_decisions("train"), sc.typed_decisions("test")
    res = {}
    for tag in a.feats.split(","):
        F = np.load(sc.OUT / f"probe_feats_{tag}.npz")
        r, pred = per_question_probe(F["train"].astype(np.float32), F["test"].astype(np.float32), train, test)
        res[tag] = r
        np.save(sc.OUT / f"probe_pred_{tag}.npy", pred)
        print(tag, r, flush=True)
    if a.with_reference:
        F = np.load(sc.ROOT / "results" / "probe_cache_served_L24.npz")
        r, pred = per_question_probe(F["train"].astype(np.float32), F["test"].astype(np.float32), train, test)
        res["qprompt_served_L24 (results/probe_cache_served_L24.npz)"] = r
        np.save(sc.OUT / "probe_pred_qprompt_served_L24.npy", pred)
        print("reference qprompt", r, flush=True)
    prev = sc.OUT / "probe_accuracy.json"
    old = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    old.update(res)
    sc.dump("probe_accuracy.json", old)


# ---------------------------------------------------------------------------------------------- latency
def laya_cases(n, st):
    qs = sc.laya_questions(n)
    return [sc.wire_to_case(qid, q, st) for qid, q in qs.items()]


def random_probes(cases, d, rng):
    return [(rng.standard_normal((len(c["options"]), d)).astype(np.float32) * 0.01, np.zeros(len(c["options"]), np.float32)) for c in cases]


def apply_probes(vecs, probes):
    out = []
    for v, (W, b) in zip(vecs, probes):
        z = W @ v + b
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def cmd_latency(a):
    rng = np.random.default_rng(0)
    layouts = a.layout.split(",")
    res = {"meta": {"script": "experiments/speed_probe_multiq.py latency", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "engine": a.engine, "url": a.url, "model": a.model, "warmup": a.warmup, "reps": a.reps}, "layouts": {}}
    llm = None
    pool = None
    if a.engine == "inproc":
        import speed_llama as L
        llm = L.Llama(str(sc.ROOT / a.model), n_ctx=16384, n_batch=4096, n_ubatch=512, n_seq_max=64, embeddings=False,
                      pooling="none", n_outputs_max=64)
        res["meta"]["params"] = llm.params
    else:
        pr = sc.props(a.url)
        res["meta"].update(model_path=pr.get("model_path"), total_slots=pr.get("total_slots"))
        pool = ThreadPoolExecutor(max_workers=64)
    counter = 500_000
    d = None
    guard = sc.Guard(port=sc.port_of(a.url) if a.engine == "http" else None, inproc=a.engine == "inproc").__enter__()
    for layout in layouts:
        sizes = {}
        for n in NS_PROBE:
            walls, feat_ms = [], []
            probes = None
            if a.cool:
                sc.wait_cool(a.cool, 240)
            for i in range(a.warmup + a.reps):
                counter += 1
                st = sc.fresh_state(counter)
                cases = laya_cases(n, st)
                t0 = time.perf_counter()
                if a.engine == "http":
                    if layout == "stateonly":
                        v = embed_http(a.url, stateonly_text(st))
                        vecs = [v] * n
                    elif layout == "qprompt":
                        if a.multi:
                            vecs = embed_http(a.url, [qprompt_text(c) for c in cases])
                        else:
                            vecs = list(pool.map(lambda c: embed_http(a.url, qprompt_text(c), _sess()), cases))
                    else:
                        raise SystemExit("sfq is in-process only")
                else:
                    if layout == "stateonly":
                        vecs = [stateonly_call(llm, st)] * n
                    elif layout == "sfq":
                        vecs = sfq_call(llm, [sfq_parts(c) for c in cases]) if n > 1 else [sfq_single(llm, cases[0])]
                    else:
                        vecs = qprompt_call(llm, [qprompt_text(c) for c in cases])
                t1 = time.perf_counter()
                if d is None:
                    d = len(vecs[0])
                if probes is None:
                    probes = random_probes(cases, d, rng)
                t2 = time.perf_counter()
                apply_probes(vecs, probes)
                t3 = time.perf_counter()
                if i >= a.warmup:
                    walls.append((t1 - t0 + t3 - t2) * 1000)
                    feat_ms.append((t1 - t0) * 1000)
                if a.gap:
                    time.sleep(a.gap)
            st_ = sc.per_call_stats(walls, n)
            st_["feature_ms_p50"] = round(float(np.median(feat_ms)), 2)
            st_["probe_ms_p50"] = round(float(np.median(walls) - np.median(feat_ms)), 3)
            sizes[str(n)] = st_
            print(f"{a.tag:14s} {layout:9s} {n:2d}q  p50 {st_['p50']:8.2f}  p95 {st_['p95']:8.2f}  per q {st_['ms_per_question_p50']:7.2f}", flush=True)
        res["layouts"][layout] = sizes
    guard.__exit__(None, None, None)
    res["gpu_guard"] = guard.result()
    print("gpu_guard contaminated:", res["gpu_guard"]["contaminated"], res["gpu_guard"]["reasons_before"] + res["gpu_guard"]["reasons_after"], flush=True)
    sc.dump(f"probe_latency_{a.tag}.json", res)


def sfq_single(llm, case):
    llm.clear()
    p, s = sfq_parts(case)
    toks = llm.tokenize(p + s, add_special=True)
    return llm.last_state(toks, 0, 0)


_tls = None


def _sess():
    import threading
    global _tls
    if _tls is None:
        _tls = threading.local()
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = sc.session()
    return s


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--layout", choices=["stateonly", "qprompt"], required=True)
    e.add_argument("--url", default="http://127.0.0.1:8095")
    e.add_argument("--tag", required=True)
    ei = sub.add_parser("extract-inproc")
    ei.add_argument("--layout", choices=["stateonly", "qprompt", "sfq"], required=True)
    ei.add_argument("--model", default="tools/models/Qwen3.5-4B-Q8_0-L24.gguf")
    ei.add_argument("--tag", required=True)
    ac = sub.add_parser("accuracy")
    ac.add_argument("--feats", required=True, help="comma-separated probe_feats_<tag>.npz tags")
    ac.add_argument("--with-reference", action="store_true")
    la = sub.add_parser("latency")
    la.add_argument("--engine", choices=["http", "inproc"], required=True)
    la.add_argument("--layout", default="stateonly")
    la.add_argument("--url", default="http://127.0.0.1:8095")
    la.add_argument("--model", default="tools/models/Qwen3.5-4B-Q8_0-L24.gguf")
    la.add_argument("--multi", action="store_true", help="http qprompt: one /embedding with the list of prompts")
    la.add_argument("--warmup", type=int, default=5)
    la.add_argument("--reps", type=int, default=50)
    la.add_argument("--gap", type=float, default=0.3)
    la.add_argument("--cool", type=float, default=0.0)
    la.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"extract": cmd_extract, "extract-inproc": cmd_extract_inproc, "accuracy": cmd_accuracy, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
