"""Exp 3 -- a streamed spoken word to a recognised action: latency per word and intent accuracy.

Data: data/voice (16 actions incl. `none`, 220 commands, 1,158 words). The prompt is run_voice_fast's "final" header with
the transcript LAST, so the instruction + action list is a constant prefix; every streamed word re-reads only what
follows it. Each utterance is scored word by word (prefix k = first k words), as a live loop would see it.

Engines
  http     llama-server /completion per word (letters; n_probs 200; --cache on/off) or /embedding (probe)
  inproc   the same model in this process through llama.dll (experiments/speed_llama.py): no HTTP/JSON, exact letter
           logits. Prefix reuse is done with sequences, never with rollback, so it is safe for hybrid Qwen3.5:
             seq 2  the constant prefix (instruction + actions + 'Transcript: "'), evaluated once per model
             seq 0  prefix + the words every later prompt shares (extended, never rolled back)
             seq 1  a copy of seq 0 that receives the new word + the closing quote + the template tail (letters)
           One llama_decode per word evaluates both extensions; seq 1 is then dropped.
Readouts
  letters  softmax over the 16 action letters at the answer position
  probe    (Qwen3.5-4B cut to 24 blocks) last-token state of 'prefix + Transcript: "<words so far>' (no tail),
           per-class logistic probe, 2-fold cross-validation over the 220 commands (stratified by intent, seed 13).
           Two training sets: `full` (the complete utterances of the other fold) and `prefix` (every prefix of the
           other fold, labelled `none` before the human commit word and the intent from it on; out-of-scope
           commands are `none` throughout). The decision on a prefix of a held-out command uses only the other fold.

Metrics: full-utterance intent accuracy (strict / accepting `alt` / accepting `then`+`alt`), none recall/precision,
per-word latency p50/p95 (compute = llama_decode + readout, or llama-server prompt_ms; round trip = everything from the
word's text to the probabilities), and the class-aware early-commit policy of stream_policy.py (tau 0.9, media/play
stability 2): harmful actions / 198 actionable, out-of-scope false actions / 22, first action word.

Reproduce
  # 12B letters over HTTP (production server :8091, caching on) = the published baseline
  python experiments/speed_voice.py --engine http --url http://127.0.0.1:8091 --template gemma4 --tag 12b_http
  # in-process (stop the server first; one model on the GPU at a time)
  python experiments/speed_voice.py --engine inproc --model <gemma gguf> --template gemma4 --tag 12b_inproc
  python experiments/speed_voice.py --engine inproc --model tools/models/Qwen3.5-4B-Q8_0.gguf --template qwen3 --tag q4b_inproc
  python experiments/speed_voice.py --engine inproc --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --template qwen3 --tag q4bL24_inproc
  python experiments/speed_voice.py --engine inproc --readout probe --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --template qwen3 --tag q4bL24_probe
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import speed_common as sc

INTENTS = json.loads((sc.ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
IDS = [i["id"] for i in INTENTS]
VL = "ABCDEFGHIJKLMNOP"
HEADER = ("You are the command router of a voice-controlled computer assistant. Read the user's "
          "transcript and decide which ONE action they are asking for. Answer with the letter only. "
          "Choose 'none' if the request is chit-chat, a question, out of scope, or too unclear to act on.")
OPTS = "Actions:\n" + "\n".join(f"{VL[i]}. {it['id']}: {it['description']}" for i, it in enumerate(INTENTS))
sp = sc._load_module("stream_policy", "stream_policy.py")


def commands():
    return [json.loads(l) for l in (sc.ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def prompt_letters(text: str, template: str) -> str:
    """run_voice_fast.build_prompt(text, intents, 'final', 'last') for gemma4; the same user text in the qwen3 template."""
    pre, post = sc.TEMPLATES[template]
    return f'{pre}{HEADER}\n\n{OPTS}\n\nTranscript: "{text}"{post}'


def prompt_stem(text: str, template: str) -> str:
    pre, _ = sc.TEMPLATES[template]
    return f'{pre}{HEADER}\n\n{OPTS}\n\nTranscript: "{text}'


END_OF_TURN = {"qwen3": "<|im_end|>", "gemma4": "<turn|>"}


def prompt_end(text: str, template: str) -> str:
    """The stem closed by the quote and the end-of-turn token: a short, fixed read position (probe_end)."""
    return f'{prompt_stem(text, template)}"{END_OF_TURN[template]}'


def lcp(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------------------------- engines
class InprocStream:
    """Word-by-word scoring with sequence copies (see the module docstring)."""

    def __init__(self, llm, template: str, readout: str, defer: bool = False):
        self.llm, self.template, self.readout, self.defer = llm, template, readout, defer
        self.ids = llm.letter_ids(VL) if readout == "letters" else None
        # letters and probe_tail read at the answer position, probe_end after '"' + end-of-turn, probe at the last word
        build = {"probe": prompt_stem, "probe_end": prompt_end}.get(readout, prompt_letters)
        self.build = lambda text: build(text, template)
        a = llm.tokenize(self.build("open notes"), add_special=True)
        b = llm.tokenize(self.build("what time is it"), add_special=True)
        self.P0 = a[: max(1, lcp(a, b) - 1)]         # the constant token prefix, one token of margin
        llm.clear()
        llm.set_embeddings(False)                    # (with embeddings on, libllama outputs every token of a decode)
        llm.eval_seq(self.P0, 2, 0, want_last=False)
        llm.sync()

    def start(self):
        self.llm.seq_rm(0)
        self.llm.seq_rm(1)
        self.llm.seq_cp(2, 0)
        self.C = list(self.P0)
        self.T_prev = list(self.P0)

    def step(self, text: str) -> dict:
        if self.readout == "probe":
            return self.step_extend(text)
        if self.defer:
            return self.step_deferred(text)
        llm = self.llm
        t0 = time.perf_counter()
        T = llm.tokenize(self.build(text), add_special=True)
        m = lcp(self.T_prev, T)
        restored = False
        if len(self.C) > m:                          # a token inside the committed part changed: restart from seq 2
            llm.seq_rm(0)
            llm.seq_cp(2, 0)
            self.C = list(self.P0)
            restored = True
            if m < len(self.C):
                raise RuntimeError("prompt diverged inside the constant prefix")
        c = len(self.C)
        ext, rest = T[c:m], T[c:]
        llm.seq_cp(0, 1)
        toks = ext + rest
        pos = list(range(c, m)) + list(range(c, len(T)))
        seqs = [0] * len(ext) + [1] * len(rest)
        outs = [False] * (len(toks) - 1) + [True]
        t1 = time.perf_counter()
        llm.set_embeddings(self.readout != "letters")
        opos = llm.decode(toks, pos, seqs, outs)[-1]
        if self.readout == "letters":
            vec = llm.logits_at(opos, self.ids)
        else:
            vec = llm.embd_at(opos)
        t2 = time.perf_counter()
        llm.seq_rm(1)
        self.C = T[:m]
        self.T_prev = T
        return dict(vec=vec, compute_ms=(t2 - t1) * 1000, total_ms=(time.perf_counter() - t0) * 1000, n_eval=len(toks),
                    restored=restored)

    def step_deferred(self, text: str) -> dict:
        """Deferred commit: the decision decodes ONE sequence (a copy of the committed transcript + the new word +
        the closing part), so hybrid models do not split it into several ubatches; the new word is committed to
        sequence 0 afterwards, in the gap before the next word (commit_ms, not part of the decision latency)."""
        llm = self.llm
        t0 = time.perf_counter()
        T = llm.tokenize(self.build(text), add_special=True)
        restored = False
        if lcp(self.C, T) < len(self.C):
            llm.seq_rm(0)
            llm.seq_cp(2, 0)
            self.C = list(self.P0)
            restored = True
        c = len(self.C)
        rest = T[c:]
        llm.seq_cp(0, 1)
        t1 = time.perf_counter()
        llm.set_embeddings(self.readout != "letters")
        opos = llm.decode(rest, list(range(c, len(T))), [1] * len(rest), [False] * (len(rest) - 1) + [True])[-1]
        vec = llm.logits_at(opos, self.ids) if self.readout == "letters" else llm.embd_at(opos)
        t2 = time.perf_counter()
        # ---- after the decision: drop the copy, commit this word's transcript tokens to sequence 0
        llm.seq_rm(1)
        S = llm.tokenize(prompt_stem(text, self.template), add_special=True)
        s = max(c, lcp(S, T))
        commit = T[c:s]
        if commit:   # (embeddings stay as the readout set them: toggling per word would re-reserve the graph)
            llm.decode(commit, list(range(c, s)), [0] * len(commit), [False] * len(commit))
            llm.sync()
        self.C = T[:s]
        t3 = time.perf_counter()
        return dict(vec=vec, compute_ms=(t2 - t1) * 1000, total_ms=(t2 - t0) * 1000, commit_ms=(t3 - t2) * 1000,
                    n_eval=len(rest), restored=restored)

    def step_extend(self, text: str) -> dict:
        """Probe readout: the stem has no tail, so every word is an exact extension of sequence 0 -- only the new
        word's tokens are evaluated. If an earlier token ever changes, sequence 0 is rebuilt from the checkpoint."""
        llm = self.llm
        t0 = time.perf_counter()
        T = llm.tokenize(self.build(text), add_special=True)
        m = lcp(self.C, T)
        restored = False
        if m < len(self.C) or m == len(T):
            llm.seq_rm(0)
            llm.seq_cp(2, 0)
            self.C = list(self.P0)
            restored = True
        new = T[len(self.C):]
        t1 = time.perf_counter()
        llm.set_embeddings(True)
        opos = llm.decode(new, list(range(len(self.C), len(T))), [0] * len(new), [False] * (len(new) - 1) + [True])[-1]
        vec = llm.embd_at(opos)
        t2 = time.perf_counter()
        self.C = T
        return dict(vec=vec, compute_ms=(t2 - t1) * 1000, total_ms=(time.perf_counter() - t0) * 1000, n_eval=len(new),
                    restored=restored)


class HttpStream:
    def __init__(self, url: str, template: str, readout: str, cache: bool, n_probs: int = 200):
        self.url, self.template, self.readout, self.cache, self.n_probs = url, template, readout, cache, n_probs
        self.s = sc.session()

    def start(self):
        pass

    def step(self, text: str) -> dict:
        if self.readout == "letters":
            r = sc.score_http(self.url, prompt_letters(text, self.template), len(VL), self.n_probs, self.cache, self.s)
            return dict(vec=r["z"], compute_ms=r["prompt_ms"], total_ms=r["http_ms"], n_eval=r["prompt_n"], cache_n=r["cache_n"])
        t = time.perf_counter()
        d = self.s.post(f"{self.url}/embedding", json={"content": prompt_stem(text, self.template), "embd_normalize": -1,
                                                          "cache_prompt": self.cache}, timeout=600).json()
        ms = (time.perf_counter() - t) * 1000
        d = d[0] if isinstance(d, list) else d
        e = d["embedding"]
        return dict(vec=np.asarray(e[-1] if isinstance(e[0], list) else e, np.float32), compute_ms=None, total_ms=ms, n_eval=None)


# ---------------------------------------------------------------------------------------------- scoring
def run_stream(eng, cmds, warm=3, gap=0.0):
    for c in cmds[:warm]:                     # warm-up (not recorded)
        eng.start()
        for k in range(1, len(c["text"].split()) + 1):
            eng.step(" ".join(c["text"].split()[:k]))
    out = []
    t0 = time.perf_counter()
    for n, c in enumerate(cmds, 1):
        words = c["text"].split()
        eng.start()
        traj = []
        for k in range(1, len(words) + 1):
            r = eng.step(" ".join(words[:k]))
            r["k"] = k
            traj.append(r)
            if gap:
                time.sleep(gap)            # words arrive at speech rate; also keeps the GPU out of thermal throttling
        out.append(traj)
        if n % 55 == 0:
            print(f"  {n}/{len(cmds)} {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def trajectories_from_probs(cmds, P):
    """P[i][k-1] = probability vector over IDS for command i, prefix k -> stream records in run_voice_fast's format."""
    stream, full = [], []
    for c, probs in zip(cmds, P):
        traj = []
        for k, p in enumerate(probs, 1):
            j = int(np.argmax(p))
            traj.append(dict(k=k, pred=IDS[j], pmax=float(p[j])))
        stream.append(dict(id=c["id"], gold=c["intent"], style=c["style"], n_words=len(c["text"].split()),
                           human_commit_word=c.get("commit_word", -1), trajectory=traj))
        full.append(dict(id=c["id"], gold_intent=c["intent"], pred_intent=traj[-1]["pred"], pmax=traj[-1]["pmax"],
                         style=c["style"], then=c.get("then"), alt=c.get("alt")))
    return stream, full


def accuracy_block(full):
    ok = lambda r: r["pred_intent"] == r["gold_intent"]  # noqa: E731
    by_style = defaultdict(list)
    for r in full:
        by_style[r["style"]].append(ok(r))
    none_rows = [r for r in full if r["gold_intent"] == "none"]
    none_pred = [r for r in full if r["pred_intent"] == "none"]
    tp = sum(1 for r in none_rows if r["pred_intent"] == "none")
    return dict(
        intent_accuracy=round(statistics.mean(ok(r) for r in full), 4),
        intent_accuracy_alt_ok=round(statistics.mean(ok(r) or (bool(r.get("alt")) and r["pred_intent"] == r["alt"]) for r in full), 4),
        intent_accuracy_alt_or_then_ok=round(statistics.mean(ok(r) or (bool(r.get("alt")) and r["pred_intent"] == r["alt"])
                                                             or (bool(r.get("then")) and r["pred_intent"] == r["then"]) for r in full), 4),
        accuracy_by_style={s: round(statistics.mean(v), 3) for s, v in by_style.items()},
        none_recall=round(tp / len(none_rows), 4) if none_rows else None,
        none_precision=round(tp / len(none_pred), 4) if none_pred else None)


def policy_block(stream, full, tau=0.9, media_stable=2, play_stable=2, slot_stable=2):
    """stream_policy.main's scoring with its defaults (class-aware commit, refinement graph)."""
    F = {r["id"]: r for r in full}
    act = [s for s in stream if s["gold"] != "none"]
    oos = [s for s in stream if s["gold"] == "none"]
    harmful, first_k, before, correct_final, harm_rows = 0, [], 0, 0, []
    for s in act:
        f = F[s["id"]]
        accepted = {s["gold"]} | ({f["then"]} if f.get("then") else set()) | ({f["alt"]} if f.get("alt") else set())
        seq = sp.executed_sequence(s["trajectory"], tau, media_stable, play_stable, slot_stable)
        bad = [a for _, a in seq if not sp.consistent(a, accepted)]
        if bad:
            harmful += 1
            harm_rows.append(dict(id=s["id"], seq=seq, accepted=sorted(accepted)))
        if seq:
            first_k.append(seq[0][0])
            before += s["human_commit_word"] > 0 and seq[0][0] <= s["human_commit_word"]
        correct_final += bool(seq) and sp.consistent(seq[-1][1], accepted)
    oos_bad = [s["id"] for s in oos if sp.executed_sequence(s["trajectory"], tau, media_stable, play_stable, slot_stable)]
    return dict(policy=dict(tau=tau, media_stable=media_stable, play_stable=play_stable, slot_stable=slot_stable),
                actionable=len(act), harmful=harmful, harmful_rate=round(harmful / len(act), 4),
                final_action_consistent_rate=round(correct_final / len(act), 4),
                mean_first_action_word=round(statistics.mean(first_k), 3) if first_k else None,
                first_action_at_or_before_human=round(before / len(act), 4),
                oos_rows=len(oos), oos_false_actions=len(oos_bad), harmful_detail=harm_rows, oos_false_detail=oos_bad)


def latency_block(trajs):
    """Per-word latency; incremental words (k >= 2) are what a live loop pays per streamed word."""
    def col(key, kmin):
        return [t[key] for tr in trajs for t in tr if t["k"] >= kmin and t.get(key) is not None]
    return dict(all_words=dict(compute_ms=sc.stats(col("compute_ms", 1)), round_trip_ms=sc.stats(col("total_ms", 1)),
                               tokens_evaluated=sc.stats(col("n_eval", 1))),
                incremental_words=dict(compute_ms=sc.stats(col("compute_ms", 2)), round_trip_ms=sc.stats(col("total_ms", 2)),
                                       tokens_evaluated=sc.stats(col("n_eval", 2))),
                first_word=dict(compute_ms=sc.stats([tr[0]["compute_ms"] for tr in trajs if tr[0].get("compute_ms") is not None]),
                                round_trip_ms=sc.stats([tr[0]["total_ms"] for tr in trajs])),
                deferred_commit_ms=sc.stats(col("commit_ms", 1)))


def folds_of(cmds, seed=13):
    rng = np.random.default_rng(seed)
    fold = {}
    by = defaultdict(list)
    for c in cmds:
        by[c["intent"]].append(c["id"])
    for intent in sorted(by):
        ids = list(by[intent])
        rng.shuffle(ids)
        for j, i in enumerate(ids):
            fold[i] = j % 2
    return fold


def probe_cv(cmds, feats, variant):
    """2-fold CV. feats[i] = array (n_words, d). Returns P[i] = (n_words, 16) probabilities over IDS."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    fold = folds_of(cmds)
    P = [None] * len(cmds)
    chosen = {}
    for f in (0, 1):
        X, y = [], []
        for i, c in enumerate(cmds):
            if fold[c["id"]] == f:
                continue
            F = feats[i]
            if variant == "full":
                X.append(F[-1]); y.append(c["intent"])
            else:
                cw = c.get("commit_word", -1)
                for k in range(1, len(F) + 1):
                    X.append(F[k - 1]); y.append(c["intent"] if (c["intent"] != "none" and cw > 0 and k >= cw) else "none")
        X, y = np.stack(X), np.array(y)
        best = None
        for Cr in (0.01, 0.05, 0.5, 5.0):
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
            try:
                s = cross_val_score(LogisticRegression(max_iter=4000, C=Cr), X, y, cv=cv).mean()
            except ValueError:
                s = -1
            if best is None or s > best[0]:
                best = (s, Cr)
        clf = LogisticRegression(max_iter=4000, C=best[1]).fit(X, y)
        chosen[f] = dict(C=best[1], inner_cv=round(float(best[0]), 4), n_train=int(len(y)))
        cls = list(clf.classes_)
        for i, c in enumerate(cmds):
            if fold[c["id"]] != f:
                continue
            pr = clf.predict_proba(feats[i])
            full_p = np.zeros((len(feats[i]), len(IDS)))
            for j, name in enumerate(cls):
                full_p[:, IDS.index(name)] = pr[:, j]
            P[i] = full_p
    return P, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["http", "inproc"], required=True)
    ap.add_argument("--readout", choices=["letters", "probe", "probe_tail", "probe_end"], default="letters",
                    help="probe = state at the last transcript token (exact extension); probe_tail = state at the answer "
                         "position after the closing quote and template tail (the letters' position, sequence-copy path); "
                         "probe_end = state after the closing quote and the end-of-turn token (2 extra tokens)")
    ap.add_argument("--defer", action="store_true", help="in-process: decide on one sequence, commit the word afterwards")
    ap.add_argument("--url", default=sc.PROD_URL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--cache", choices=["on", "off"], default="on")
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.25, help="seconds between streamed words (speech is ~3 words/s)")
    ap.add_argument("--cool", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cmds = commands()
    if args.limit:
        cmds = cmds[: args.limit]
    meta = {"script": "experiments/speed_voice.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
            "data_sha256": hashlib.sha256((sc.ROOT / "data" / "voice" / "commands.jsonl").read_bytes()).hexdigest(),
            "prompt_example": {"probe": prompt_stem, "probe_end": prompt_end}.get(args.readout, prompt_letters)("open notes", args.template)}
    if args.engine == "http":
        pr = sc.props(args.url)
        meta.update(model_path=pr.get("model_path"), build=pr.get("build_info"), total_slots=pr.get("total_slots"))
        eng = HttpStream(args.url, args.template, args.readout, args.cache == "on", args.n_probs)
    else:
        import speed_llama as L
        llm = L.Llama(args.model, n_ctx=2048, n_batch=512, n_ubatch=512, n_seq_max=3, embeddings=False,
                      pooling="none", n_outputs_max=32)
        meta.update(model_path=args.model, load_s=round(llm.load_s, 1), n_layer=llm.n_layer, n_embd=llm.n_embd, params=llm.params)
        eng = InprocStream(llm, args.template, args.readout, defer=args.defer)
        meta["constant_prefix_tokens"] = len(eng.P0)
    res_gpu = {"before": sc.wait_cool(args.cool, 300) if args.cool else sc.gpu_now()}
    with sc.Guard(port=sc.port_of(args.url) if args.engine == "http" else None, inproc=args.engine == "inproc") as guard:
        with sc.GpuMonitor() as mon:
            trajs = run_stream(eng, cmds, gap=args.gap)
    res_gpu["during"] = mon.summary()
    g = guard.result()
    print("gpu_guard contaminated:", g["contaminated"], g["reasons_before"] + g["reasons_after"], flush=True)
    lat = latency_block(trajs)
    res = {"meta": meta, "latency_per_word": lat, "gpu": res_gpu, "gpu_guard": g,
           "restored_from_checkpoint": int(sum(1 for tr in trajs for t in tr if t.get("restored")))}
    if args.readout == "letters":
        P = [[sc.softmax(t["vec"]) for t in tr] for tr in trajs]
        stream, full = trajectories_from_probs(cmds, P)
        res["accuracy"] = accuracy_block(full)
        res["streaming_policy"] = policy_block(stream, full)
        tag_rows = {"stream": stream, "full": full}
    else:
        feats = [np.stack([t["vec"] for t in tr]) for tr in trajs]
        np.savez_compressed(sc.OUT / f"voice_feats_{args.tag}.npz", **{c["id"]: f for c, f in zip(cmds, feats)})
        tag_rows = {}
        for variant in ("full", "prefix"):
            P, chosen = probe_cv(cmds, feats, variant)
            stream, full = trajectories_from_probs(cmds, P)
            res[f"probe_{variant}"] = dict(cv=chosen, accuracy=accuracy_block(full), streaming_policy=policy_block(stream, full))
            tag_rows[variant] = {"stream": stream, "full": full}
    sc.dump(f"voice_{args.tag}.json", res)
    (sc.OUT / f"voice_rows_{args.tag}.json").write_text(json.dumps(tag_rows), encoding="utf-8")
    brief = {"latency_incremental": lat["incremental_words"], "first_word": lat["first_word"]}
    for k in ("accuracy", "streaming_policy", "probe_full", "probe_prefix"):
        if k in res:
            v = res[k]
            brief[k] = {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else []) if not kk.endswith("detail")}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
