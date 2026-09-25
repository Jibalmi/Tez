"""Word-aligned replay of voice command sets: streaming ASR -> tez.stream session -> a no-op native executor.

Sets (WAVs + labels + word timings):
  tez220   data/voice: Tez's 220 commands (16 actions), Windows SAPI speech; word times from faster-whisper
           (experiments/voice_align.py -> data/voice/align.jsonl)
  theioma  data/voice/theioma: 269 commands over 23 Theioma voice tools + none, Kokoro-82M speech in six voices,
           labelled synthetic; word times from Kokoro's phoneme durations (experiments/voice_tts.py)
Each set's schema (schema.yaml next to its commands) is the Tez question the session answers and its actions: block
is the early-commit policy.

Replay. Audio is pushed in real time as 10 ms frames: frame i is handed over at t0 + (i + 1) * 10 ms on the
perf_counter clock (QueryPerformanceCounter), the moment its last sample would have been captured. The recogniser
steps whenever it has a chunk (80 ms for Nemotron at right context 0); every step that changes the transcript becomes
a session.update(text) (the last word may still grow), the policy fires actions into a no-op executor that makes one
native call (user32.GetForegroundWindow through ctypes) and time-stamps it. The ASR runs in this process (sherpa,
whisper) or in a worker process that paces the audio itself (nemocpp: its ggml DLLs collide with llama.cpp's), then
the session runs as the emissions arrive (a pipelined two-process loop; the pipe hop is logged). --asr oracle feeds
each reference word at its forced end time (a perfect, lag-free recogniser): the decision and dispatch alone.

Metrics (per set, and per action class where it matters)
  word end -> decision    t(decision on the first update that holds the decisive word, as the recogniser emitted it)
                          - t(the decisive word's last sample captured), for the human-annotated decisive word
                          (commit_word) of every actionable command; also for every word ("all words")
  word end -> action      t(the first action's native call) - t(decisive word end); negative when it fired earlier
  projected word end -> action  word end -> decision + the measured dispatch
  emission (ASR)          audio position of the step that first emitted the decisive word - its end (delay), and
                          minus the end of the chunk that holds the word's end (lag beyond chunk quantisation)
  intent accuracy         the final decision, strict and accepting then/alt; harmful actions (an executed action that
                          is neither in {gold, then, alt} nor an ancestor of one); out-of-scope false actions;
                          first action word (words fully heard when it fired) and how often it was at or before the
                          decisive word; WER of the final transcripts
Every run writes results/voice/replay_<tag>.json (manifest: data hashes, model and recogniser, policy, CPU load and
the GPU lock's VRAM snapshot before and after; summary) and replay_<tag>_rows.jsonl (per utterance: every emission and
update with its timings and full distribution, the actions). --rescore ROWS re-scores a finished run's logged
decisions with another threshold (Policy.replay), no model needed.

  python experiments/voice_replay.py --set theioma --asr oracle --backend inproc:<gemma gguf> --tag th_oracle_12bq8
  python experiments/voice_replay.py --set theioma --asr nemocpp:<gguf> --rc 0 --backend none --tag th_nemo80_asr
  python experiments/voice_replay.py --rescore results/voice/replay_x_rows.jsonl --tau 0.5
"""
from __future__ import annotations

import argparse
import ctypes as C
import difflib
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_asr import SR, Emission, make_asr, normalise  # noqa: E402

OUT = ROOT / "results" / "voice"
SETS = {"tez220": ROOT / "data" / "voice", "theioma": ROOT / "data" / "voice" / "theioma"}
FRAME = 160                                    # 10 ms at 16 kHz
GPU_LOCK = "C:/temp/gpu_lock.py"


# ---------------------------------------------------------------------------------------------- data
def load_set(name: str, limit: int = 0, ids: set | None = None) -> tuple[list[dict], Path, dict]:
    d = SETS[name]
    cmds = [json.loads(x) for x in (d / "commands.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    align = {r["id"]: r for r in (json.loads(x) for x in (d / "align.jsonl").read_text(encoding="utf-8").splitlines()
                                  if x.strip())}
    rows = []
    for c in cmds:
        a = align.get(c["id"])
        if a is None or (ids is not None and c["id"] not in ids):
            continue
        rows.append({**c, "wav": str(d / a["wav"]), "words": a["words"], "voice": a.get("voice") or a.get("tts")})
    if limit:
        rows = rows[:limit]
    hashes = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("commands.jsonl", "align.jsonl", "schema.yaml")}
    return rows, d / "schema.yaml", hashes


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        if f.getframerate() != SR or f.getsampwidth() != 2:
            raise ValueError(f"{path}: 16 kHz 16-bit PCM expected")
        x = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32) / 32768.0
        if f.getnchannels() > 1:
            x = x.reshape(-1, f.getnchannels()).mean(axis=1)
    return x


# ---------------------------------------------------------------------------------------------- machine state
class Load:
    """CPU load sampled every second during a run (psutil), and the GPU lock's VRAM snapshots."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()

    def __enter__(self):
        try:
            import psutil
            psutil.cpu_percent(None)
            self._t = threading.Thread(target=self._run, args=(psutil,), daemon=True)
            self._t.start()
        except ImportError:
            self._t = None
        return self

    def _run(self, psutil):
        while not self._stop.wait(1.0):
            self.samples.append(psutil.cpu_percent(None))

    def __exit__(self, *exc):
        self._stop.set()

    def summary(self) -> dict:
        s = self.samples
        return {"cpu_pct_mean": round(statistics.mean(s), 1) if s else None, "cpu_pct_max": max(s) if s else None,
                "cpu_pct_p50": round(statistics.median(s), 1) if s else None, "samples": len(s)}


def gpu_snapshot() -> dict:
    try:
        out = subprocess.run([sys.executable, GPU_LOCK, "snapshot"], capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"error": str(exc)}


# ---------------------------------------------------------------------------------------------- the executor
class NoopExecutor:
    """What a native executor does on the hot path, minus the effect: one Win32 call through ctypes (the foreground
    window, which a real executor checks before typing), time-stamped. Records (action, t_called, t_issued)."""

    def __init__(self):
        self.user32 = C.windll.user32 if sys.platform == "win32" else None
        self.log: list[dict] = []

    def __call__(self, a) -> None:
        t0 = time.perf_counter()
        if self.user32 is not None:
            self.user32.GetForegroundWindow()
        t1 = time.perf_counter()
        self.log.append({"name": a.name, "kind": a.kind, "word": a.word, "p": round(a.p, 4), "gated": a.gated,
                         "t_called": t0, "t_issued": t1, "dispatch_ms": (t1 - t0) * 1000.0})


def undo_log(log: list[dict]):
    def on_undo(a) -> None:
        log.append({"name": a.name, "kind": "undo", "word": a.word, "t_called": time.perf_counter()})
    return on_undo


# ---------------------------------------------------------------------------------------------- one utterance
def oracle_emissions(row: dict) -> list[tuple[float, str]]:
    """(audio time, text so far) for a perfect recogniser: each reference word at its end, rounded up to its frame."""
    out, words = [], []
    for w in row["words"]:
        words.append(w["w"])
        out.append((math.ceil(w["end"] * 100 - 1e-9) / 100.0, " ".join(words)))
    return out


def realtime_process() -> dict:
    """What a voice loop's process does on Windows: a 1 ms timer period (timeBeginPeriod) so short sleeps wake on
    time, and the high priority class so a busy machine delays its frames less. Returns what was set."""
    out = {}
    if sys.platform != "win32":
        return out
    try:
        C.windll.winmm.timeBeginPeriod(1)
        out["timer_period_ms"] = 1
    except Exception as exc:  # noqa: BLE001
        out["timer_error"] = str(exc)
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        out["priority"] = "high"
    except Exception as exc:  # noqa: BLE001
        out["priority_error"] = str(exc)
    return out


def sleep_until(t: float) -> None:
    while True:
        now = time.perf_counter()
        if now >= t:
            return
        if t - now > 0.002:
            time.sleep(t - now - 0.0015)


def record_update(u, t_in: float, e: Emission | None, due: float | None = None) -> dict:
    return {"t_in": t_in, "t_decided": t_in + u.timings.get("decide_ms", 0.0) / 1000.0, "text": u.transcript, "k": u.k,
            "late_ms": round((t_in - due) * 1000.0, 3) if due is not None else None,
            "partial": u.partial, "decision": u.decision, "p": round(u.confidence, 5), "stable": u.stable,
            "committed": u.committed, "actions": [a.name for a in u.actions if a.kind != "undo"][-2:],
            "decide_ms": round(u.timings.get("decide_ms", 0.0), 3), "commit_ms": round(u.timings.get("commit_ms", 0.0), 3),
            "dispatch_ms": round(u.timings.get("dispatch_ms", 0.0), 4), "tokens": u.timings.get("tokens"),
            "restored": u.timings.get("restored"), "temperature": u.meta.get("temperature"),
            "probs": {k: round(v, 5) for k, v in u.probabilities.items()},
            "asr_audio_s": e.audio_s if e is not None else None, "asr_ms": e.compute_ms if e is not None else None,
            "t_emit": e.t if e is not None else None}


def trim(row: dict, x: np.ndarray, lead: float = 0.3, tail: float = 0.5) -> tuple[np.ndarray, list[dict]]:
    """The audio from `lead` s before the first word to `tail` s after the last (the synthetic WAVs carry more
    silence than that), with the word times shifted to the cut."""
    w = row["words"]
    a = max(0.0, w[0]["start"] - lead)
    b = min(len(x) / SR, w[-1]["end"] + tail)
    i, j = int(a * SR) // FRAME * FRAME, int(math.ceil(b * SR))
    shift = i / SR
    return x[i:j], [{**v, "start": v["start"] - shift, "end": v["end"] - shift} for v in w]


def run_utterance(row: dict, args, session, asr, worker, executor: NoopExecutor, undos: list) -> dict:
    x = read_wav(row["wav"])
    if args.trim:
        x, words = trim(row, x)
        row = {**row, "words": words}
    executor.log.clear()
    undos.clear()
    if session is not None:
        session.reset()
    updates, emits = [], []
    steps, busy = [], None
    t0 = time.perf_counter() + 0.03                  # sample 0 is captured at t0

    def on_emit(e: Emission, t_recv: float) -> None:
        emits.append({"text": e.text, "audio_s": e.audio_s, "asr_ms": round(e.compute_ms, 3), "t_emit": e.t,
                      "t_recv": t_recv, "final": e.final, "late_ms": round((t_recv - (t0 + e.audio_s)) * 1000.0, 3)})
        if session is None or not e.text:
            return
        t_in = time.perf_counter()
        u = session.update(e.text, final=e.final)
        if u is not None:
            updates.append(record_update(u, t_in, e, t0 + e.audio_s))

    if args.asr == "oracle":
        for audio_s, text in oracle_emissions(row):
            due = t0 + audio_s
            if args.realtime:
                sleep_until(due)
            on_emit(Emission(text, audio_s, 0.0, time.perf_counter(), final=True),
                    time.perf_counter() if args.realtime else due)
        t_last = time.perf_counter()
    elif worker is not None:
        conn = worker
        conn.send(("utt", x.tobytes(), t0 if args.realtime else None))
        while True:
            msg = conn.recv()
            t_recv = time.perf_counter()
            if msg[0] == "done":
                t_last, steps, busy = msg[1], msg[2], msg[3]
                break
            _, text, audio_s, ms, t_emit, final = msg
            on_emit(Emission(text, audio_s, ms, t_emit, final), t_recv)
    else:
        asr.reset()
        for i in range(0, len(x), FRAME):
            if args.realtime:
                sleep_until(t0 + (i + FRAME) / SR)
            for e in asr.push(x[i: i + FRAME]):
                on_emit(e, time.perf_counter())
        for e in asr.finish():
            e.final = True
            on_emit(e, time.perf_counter())
        t_last = time.perf_counter()
        steps, busy = list(asr.steps), asr.busy_ms
    end = None
    if session is not None:
        if session.words and not session._final:
            u = session.update(session.transcript, final=True)
            if u is not None:
                updates.append(record_update(u, time.perf_counter(), None))
        u = session.end()
        end = {"decision": u.decision, "p": round(u.confidence, 5), "handoff": u.handoff,
               "action": u.action.name if u.action is not None and u.action.kind != "undo" else None}
    return {"id": row["id"], "style": row.get("style"), "gold": row["intent"], "then": row.get("then"), "alt": row.get("alt"),
            "commit_word": row.get("commit_word", -1), "text": row["text"], "voice": row.get("voice"),
            "words": row["words"], "t0": t0, "t_last": t_last, "duration_s": len(x) / SR, "emissions": emits,
            "asr_steps_ms": [round(s, 3) for s in steps], "asr_busy_ms": busy,
            "updates": updates, "end": end, "executed": [dict(a) for a in executor.log], "undos": list(undos)}


# ---------------------------------------------------------------------------------------------- scoring
def pct(v: list[float], q: float) -> float | None:
    return round(float(np.percentile(v, q)), 2) if v else None


def stats(v: list[float]) -> dict:
    return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "mean": round(float(np.mean(v)), 2) if v else None,
            "min": round(float(min(v)), 2) if v else None, "max": round(float(max(v)), 2) if v else None}


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(hyp)], len(ref)


def first_heard(ref: list[str], texts: list[str], idx: int) -> int | None:
    """The first transcript (index into texts) in which reference word idx appears, aligned to the reference."""
    r = [normalise(w) for w in ref]
    for n, text in enumerate(texts):
        h = text.split()
        sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
        for blk in sm.get_matching_blocks():
            if blk.a <= idx < blk.a + blk.size:
                return n
    return None


def score(rows: list[dict], policy, chunk_s: float | None) -> dict:
    """The metrics of the module docstring from the logged rows."""
    none_keys = {"none", "__none__"}
    act = [r for r in rows if r["gold"] not in none_keys]
    oos = [r for r in rows if r["gold"] in none_keys]
    lat_dec, lat_all, lat_act, lat_proj, emit_delay, emit_lag, disp = [], [], [], [], [], [], []
    by_class: dict[str, list[float]] = {}
    harmful, harm_rows, final_ok, final_ok_alt, first_words, before = 0, [], 0, 0, [], 0
    wer_e, wer_n, missed_decisive = 0, 0, 0
    decide_ms = [u["decide_ms"] for r in rows for u in r["updates"]]
    commit_ms = [u["commit_ms"] for r in rows for u in r["updates"]]
    asr_ms = [s for r in rows for s in r.get("asr_steps_ms") or []]
    busy_per_chunk = [r["asr_busy_ms"] / max(1, r["duration_s"] / chunk_s) for r in rows
                      if r.get("asr_busy_ms") is not None and chunk_s]
    pipe_ms = [(e["t_recv"] - e["t_emit"]) * 1000.0 for r in rows for e in r["emissions"] if e.get("t_emit") and e.get("t_recv")]
    late_ms = [e["late_ms"] for r in rows for e in r["emissions"] if e.get("late_ms") is not None]
    for r in rows:
        ref = [w["w"] for w in r["words"]]
        texts = [e["text"] for e in r["emissions"]]
        if texts:
            e_, n_ = wer([normalise(w) for w in ref], texts[-1].split())
            wer_e, wer_n = wer_e + e_, wer_n + n_
        disp += [a["dispatch_ms"] for a in r["executed"]]
        upd_by_text = {}
        for u in r["updates"]:
            upd_by_text.setdefault(u["text"], u)
        # every word: from its end to the decision on the first update that holds it
        for i, w in enumerate(r["words"]):
            n = first_heard(ref, texts, i)
            if n is None:
                continue
            u = upd_by_text.get(texts[n])
            if u is not None:
                lat_all.append((u["t_decided"] - (r["t0"] + w["end"])) * 1000.0)
        if r["gold"] in none_keys:
            continue
        cw = r["commit_word"]
        if cw and cw > 0:
            w = r["words"][cw - 1]
            t_end = r["t0"] + w["end"]
            n = first_heard(ref, texts, cw - 1)
            if n is None:
                missed_decisive += 1
                n = next((j for j, t in enumerate(texts) if len(t.split()) >= cw), None)
            if n is not None:
                e = r["emissions"][n]
                u = upd_by_text.get(texts[n])
                if e.get("audio_s") is not None:
                    emit_delay.append((e["audio_s"] - w["end"]) * 1000.0)
                    if chunk_s:
                        emit_lag.append((e["audio_s"] - math.ceil(w["end"] / chunk_s - 1e-9) * chunk_s) * 1000.0)
                if u is not None:
                    d = (u["t_decided"] - t_end) * 1000.0
                    lat_dec.append(d)
                    lat_proj.append(d + (statistics.mean(disp) if disp else 0.0))
                    by_class.setdefault(policy.spec(r["gold"]).cls, []).append(d)
            ex = [a for a in r["executed"] if not a.get("gated")]
            if ex:
                lat_act.append((ex[0]["t_issued"] - t_end) * 1000.0)
        accepted = {r["gold"]} | ({r["then"]} if r.get("then") else set()) | ({r["alt"]} if r.get("alt") else set())
        ex = [a for a in r["executed"] if not a.get("gated")]
        bad = [a["name"] for a in ex if not any(policy.consistent(a["name"], g) for g in accepted)]
        if bad:
            harmful += 1
            harm_rows.append({"id": r["id"], "text": r["text"], "executed": [a["name"] for a in ex],
                              "accepted": sorted(accepted)})
        final = r["end"]["decision"] if r.get("end") else None
        final_ok += final == r["gold"]
        final_ok_alt += final in accepted
        if ex:
            k = sum(1 for w in r["words"] if r["t0"] + w["end"] <= ex[0]["t_issued"])
            first_words.append(k)
            before += r["commit_word"] > 0 and k <= r["commit_word"]
    oos_false = [{"id": r["id"], "text": r["text"], "executed": [a["name"] for a in r["executed"] if not a.get("gated")]}
                 for r in oos if any(not a.get("gated") for a in r["executed"])]
    oos_gated = sum(1 for r in oos if any(a.get("gated") for a in r["executed"]))
    none_ok = sum(1 for r in oos if r.get("end") and r["end"]["decision"] in none_keys)
    has_decisions = any(r["updates"] for r in rows)
    out = {"rows": len(rows), "actionable": len(act), "oos": len(oos),
           "asr": {"wer": round(wer_e / wer_n, 4) if wer_n else None, "step_compute_ms": stats(asr_ms),
                   "busy_ms_per_chunk": stats(busy_per_chunk),
                   "emission_delay_ms": stats(emit_delay), "emission_lag_beyond_chunk_ms": stats(emit_lag),
                   "pipe_ms": stats(pipe_ms), "decisive_word_missed": missed_decisive},
           "emission_arrival_after_audio_ms": stats(late_ms)}
    if has_decisions:
        out.update({
            "word_end_to_decision_ms": stats(lat_dec), "word_end_to_decision_by_class_ms": {k: stats(v) for k, v in
                                                                                             sorted(by_class.items())},
            "all_words_end_to_decision_ms": stats(lat_all), "word_end_to_first_action_ms": stats(lat_act),
            "projected_word_end_to_action_ms": stats(lat_proj), "dispatch_ms": stats(disp),
            "decide_ms_per_update": stats(decide_ms), "commit_ms_per_update": stats(commit_ms),
            "intent_accuracy": round(final_ok / len(act), 4) if act else None,
            "intent_accuracy_then_alt": round(final_ok_alt / len(act), 4) if act else None,
            "none_recall": round(none_ok / len(oos), 4) if oos else None,
            "harmful": harmful, "harmful_rate": round(harmful / len(act), 4) if act else None,
            "oos_false_actions": len(oos_false), "oos_approval_requests": oos_gated,
            "first_action_word_mean": round(statistics.mean(first_words), 3) if first_words else None,
            "first_action_at_or_before_decisive": round(before / len(act), 4) if act else None,
            "harmful_detail": harm_rows, "oos_false_detail": oos_false})
    return out


def rescore(rows_path: Path, tau: float | None, schema_path: Path | None) -> dict:
    """Re-run the policy (Policy.replay) on a finished run's logged decisions with another threshold, and score it."""
    from tez.schema import load_schema
    from tez.stream import Policy
    rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    meta = json.loads((rows_path.parent / rows_path.name.replace("_rows.jsonl", ".json")).read_text(encoding="utf-8"))
    schema = load_schema(schema_path or meta["manifest"]["schema"])
    pol = Policy.from_schema(schema, None, tau=tau if tau is not None else meta["manifest"]["policy"]["tau"])
    for r in rows:
        ups = r["updates"]
        ex = []
        for i, _, kind, name in pol.replay([(u["decision"], u["p"], u["k"]) for u in ups]):
            t = r["t_last"] if kind in ("end", "approval") else ups[i]["t_decided"]
            ex.append({"name": name, "kind": kind, "gated": kind == "approval", "t_issued": t, "dispatch_ms": 0.0})
        r["executed"] = ex
    return score(rows, pol, meta["manifest"].get("asr_chunk_s"))


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="theioma")
    ap.add_argument("--asr", default="oracle", help="oracle | nemocpp:GGUF | sherpa:DIR | whisper:NAME")
    ap.add_argument("--rc", type=int, default=0, help="nemocpp right context (0 = 80 ms, 1 = 160 ms)")
    ap.add_argument("--asr-gpu", type=int, default=0)
    ap.add_argument("--asr-threads", type=int, default=4)
    ap.add_argument("--asr-proc", action="store_true", help="run the recogniser in a worker process (default for nemocpp)")
    ap.add_argument("--backend", default="none", help="none (ASR only) | inproc:GGUF | an llama-server URL")
    ap.add_argument("--template", default="gemma4")
    ap.add_argument("--lib", default="C:/temp/llamacpp")
    ap.add_argument("--schema", default=None, help="a schema file instead of the set's (a fitted copy)")
    ap.add_argument("--readout", default="auto")
    ap.add_argument("--default-temperature", default="auto")
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-seq-max", type=int, default=8)
    ap.add_argument("--n-outputs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default=None, help="a file of ids (one per line) to replay")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.add_argument("--no-trim", dest="trim", action="store_false",
                    help="keep the WAVs' own silence (default: 0.3 s before the first word, 0.5 s after the last)")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds between utterances")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rescore", default=None)
    args = ap.parse_args()
    if args.rescore:
        res = rescore(Path(args.rescore), args.tau, Path(args.schema) if args.schema else None)
        print(json.dumps({k: v for k, v in res.items() if not k.endswith("detail")}, indent=1))
        return
    ids = set(Path(args.ids).read_text(encoding="utf-8").split()) if args.ids else None
    rows, schema_path, hashes = load_set(args.set, args.limit, ids)
    schema_path = Path(args.schema) if args.schema else schema_path
    tag = args.tag or f"{args.set}_{args.asr.split(':')[0]}_{Path(args.backend).stem if args.backend != 'none' else 'asr'}"
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"script": "experiments/voice_replay.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args),
                "process": realtime_process() if args.realtime else {},
                "set": args.set, "data_sha256": hashes, "schema": str(schema_path), "rows": len(rows),
                "gpu_before": gpu_snapshot()}
    # ---- the recogniser
    asr = worker = proc = None
    kw = {}
    if args.asr.startswith("nemocpp:"):
        kw = {"right_context": args.rc, "gpu": args.asr_gpu}
    elif args.asr.startswith("sherpa:"):
        kw = {"threads": args.asr_threads}
    if args.asr != "oracle":
        if args.asr_proc or args.asr.startswith("nemocpp:"):
            from voice_asr import asr_worker
            ctx = mp.get_context("spawn")
            worker, child = ctx.Pipe()
            proc = ctx.Process(target=asr_worker, args=(child, args.asr, kw), daemon=True)
            proc.start()
            msg = worker.recv()
            manifest["asr"] = {**msg[1], "process": "worker"}
        else:
            asr = make_asr(args.asr, **kw)
            manifest["asr"] = {**asr.info(), "process": "same"}
        manifest["asr_chunk_s"] = manifest["asr"]["chunk_s"]
    else:
        manifest["asr"] = {"name": "oracle", "note": "reference words at their forced end times, rounded up to 10 ms"}
        manifest["asr_chunk_s"] = None
    # ---- the decision
    session = None
    from tez.schema import load_schema
    from tez.stream import Policy
    schema = load_schema(schema_path)
    policy = Policy.from_schema(schema, None, tau=args.tau)
    executor, undos = NoopExecutor(), []
    if args.backend != "none":
        from tez import Tez
        inproc = {"lib": args.lib, "n_ctx": args.n_ctx, "n_seq_max": args.n_seq_max, "n_outputs": args.n_outputs}
        tez = Tez(backend=args.backend, template=args.template, schemas=[schema], inproc=inproc,
                  default_temperature=args.default_temperature)
        t = time.perf_counter()
        session = tez.stream(schema=schema.name, policy=policy, readout=args.readout, on_action=executor,
                             on_undo=undo_log(undos)).load()
        manifest["decision"] = {"backend": args.backend, "template": args.template, "load_s": round(time.perf_counter() - t, 1),
                                "model": tez.backend.model_name(), "readout": session.setup.use_probe and "probe" or "letters",
                                "parts": [p[0] for p in session._parts], "incremental": session.incremental,
                                "default_temperature": args.default_temperature}
        if hasattr(tez.backend, "details"):
            det = tez.backend.details()
            manifest["decision"]["details"] = {k: det.get(k) for k in ("build", "n_ctx", "n_seq_max", "n_layer", "n_params",
                                                                       "size_mb", "hybrid", "desc")}
            manifest["decision"]["head_ms"] = round(session._reader.head_ms, 1) if session._reader else None
    manifest["policy"] = {"tau": policy.tau, "classes": {k: v.cls for k, v in policy.actions.items()}}
    # ---- warm-up (not recorded), then the run
    for r in rows[: args.warmup]:
        run_utterance(r, argparse.Namespace(**{**vars(args), "realtime": False}), session, asr, worker, executor, undos)
    out_rows = []
    t_run = time.perf_counter()
    with Load() as load:
        for n, r in enumerate(rows, 1):
            out_rows.append(run_utterance(r, args, session, asr, worker, executor, undos))
            if args.gap:
                time.sleep(args.gap)
            if n % 50 == 0:
                print(f"  {n}/{len(rows)} {time.perf_counter() - t_run:.0f}s", flush=True)
    manifest["run_s"] = round(time.perf_counter() - t_run, 1)
    manifest["load"] = load.summary()
    manifest["gpu_after"] = gpu_snapshot()
    if worker is not None:
        worker.send(("quit",))
        proc.join(timeout=30)
    res = score(out_rows, policy, manifest["asr_chunk_s"])
    manifest["timing"] = "realtime" if args.realtime else "offline: the latencies are not meaningful, only per-step compute"
    if not args.realtime:
        for k in [k for k in res if k.startswith(("word_end", "all_words", "projected"))]:
            res[k] = None
    (OUT / f"replay_{tag}.json").write_text(json.dumps({"manifest": manifest, "summary": res}, indent=1), encoding="utf-8")
    (OUT / f"replay_{tag}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows), encoding="utf-8")
    brief = {k: v for k, v in res.items() if not k.endswith("detail")}
    print(json.dumps({"load": manifest["load"], **brief}, indent=1))


if __name__ == "__main__":
    main()
