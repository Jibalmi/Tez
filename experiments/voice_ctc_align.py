"""Word end times for the replay sets: the forced alignment the harness measures latency from.

The replay harness measures latency from the end of each word (experiments/voice_replay.py), so word ends must be right
to a few tens of milliseconds. What each set uses, and how it was checked:

  theioma (Kokoro)  Kokoro's own timings (experiments/voice_tts.py) are the phoneme durations the audio was generated
                    from, so they are exact, except that the last word's end includes the utterance's trailing pause
                    (245 ms past the end of the speech energy, median of 80 utterances). `trim` keeps Kokoro's end but
                    cuts it to the end of the speech energy inside the word's own window (10 ms frames above the
                    utterance's loudest frame minus 35 dB): unchanged in connected speech, the true sound end before a
                    pause. The untrimmed end stays as `end_kokoro`.
  tez220 (SAPI)     SAPI reports no timings to us (its word events never fired through comtypes), and faster-whisper
                    large-v3 word timestamps (experiments/voice_align.py) are cross-attention guesses: 105 ms early at the
                    median against the Kokoro timings, 528 ms off at the 90th percentile. `ctc` forces the transcript
                    through wav2vec2-base-960h's CTC posteriors (20 ms frames, a Viterbi pass like torchaudio's
                    forced_align, written out in numpy): a word ends at its last character's last frame. The same
                    aligner run on the Kokoro set, against its trimmed timings, measures its error (--check); the
                    whisper end stays as `end_whisper`.

  python experiments/voice_ctc_align.py trim --set theioma            # rewrites data/voice/theioma/align.jsonl
  python experiments/voice_ctc_align.py ctc --set theioma --check     # the aligner's error against Kokoro's timings
  python experiments/voice_ctc_align.py ctc --set tez220 --install    # CTC ends into data/voice/align.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = {"tez220": ROOT / "data" / "voice", "theioma": ROOT / "data" / "voice" / "theioma"}
MODEL = "facebook/wav2vec2-base-960h"
FRAME_S = 0.02
ENERGY_DB = 35.0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()), "<i2").astype(np.float32) / 32768.0


def active_frames(x: np.ndarray) -> np.ndarray:
    fr = x[: len(x) // 160 * 160].reshape(-1, 160).astype(np.float64)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    return db > db.max() - ENERGY_DB


def load(d: Path, name: str = "align.jsonl") -> list[dict]:
    return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(d: Path, rows: list[dict], name: str = "align.jsonl") -> None:
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def pct(v: list[float]) -> dict:
    a = np.asarray(v, float)
    return {"n": len(v), "median": round(float(np.median(a)), 1), "mean": round(float(a.mean()), 1),
            "mae": round(float(np.abs(a).mean()), 1), "p10": round(float(np.percentile(a, 10)), 1),
            "p90": round(float(np.percentile(a, 90)), 1)}


# ---------------------------------------------------------------------------------------------- Kokoro: trim
def trim(set_name: str) -> None:
    d = SETS[set_name]
    rows = load(d)
    moved, last_moved = [], []
    for r in rows:
        x = read_wav(d / r["wav"])
        act = active_frames(x)
        for i, w in enumerate(r["words"]):
            end_k = w.get("end_kokoro", w["end"])
            i0, i1 = int(w["start"] * 100), min(len(act), int(np.ceil(end_k * 100)))
            idx = np.flatnonzero(act[i0:i1])
            end = min(end_k, (i0 + idx[-1] + 1) / 100.0) if len(idx) else end_k
            w["end_kokoro"], w["end"] = end_k, round(end, 4)
            moved.append((end_k - end) * 1000.0)
            if i == len(r["words"]) - 1:
                last_moved.append((end_k - end) * 1000.0)
        r["align"] = ("kokoro phoneme durations (12.5 ms units), each end cut to the end of the speech energy inside the "
                      f"word (-{ENERGY_DB:.0f} dB); end_kokoro = untrimmed")
    dump(d, rows)
    res = {"set": set_name, "rows": len(rows), "trimmed_ms_all_words": pct(moved), "trimmed_ms_last_word": pct(last_moved)}
    (ROOT / "results" / "voice" / f"align_trim_{set_name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- CTC forced alignment
def viterbi(logp: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int, int]]:
    """CTC forced alignment: the most probable path of `tokens` (with blanks) through the frame log-posteriors (T, V).
    Returns (token index, first frame, last frame) for every token the path visits."""
    T = logp.shape[0]
    ext = [blank]
    for t in tokens:
        ext += [t, blank]
    S = len(ext)
    neg = -1e30
    allow_skip = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
    dp = np.full((T, S), neg)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        step = np.concatenate([[neg], prev[:-1]])
        skip = np.where(allow_skip, np.concatenate([[neg, neg], prev[:-2]]), neg)
        cand = np.stack([prev, step, skip])
        bp[t] = np.argmax(cand, axis=0)
        dp[t] = cand[bp[t], np.arange(S)] + logp[t, ext]
    s = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    spans: dict[int, list[int]] = {}
    for t, s in enumerate(path):
        if ext[s] != blank:
            spans.setdefault((s - 1) // 2, []).append(t)
    return [(i, v[0], v[-1]) for i, v in sorted(spans.items())]


def ctc(set_name: str, threads: int, limit: int, check: bool, install: bool) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    vocab = proc.tokenizer.get_vocab()
    blank, sep = proc.tokenizer.pad_token_id, vocab["|"]
    d = SETS[set_name]
    rows = load(d)
    rows = rows[:limit] if limit else rows
    out, d_end, d_last, d_start = [], [], [], []
    for n, r in enumerate(rows):
        x = read_wav(d / r["wav"])
        with torch.inference_mode():
            iv = proc(x, sampling_rate=16000, return_tensors="pt").input_values
            logp = torch.log_softmax(model(iv).logits[0], dim=-1).numpy().astype(np.float64)
        words = [w["w"] for w in r["words"]]
        toks, owner = [], []
        for wi, w in enumerate(words):
            if wi:
                toks.append(sep)
                owner.append(-1)
            for ch in re.sub(r"[^A-Z']", "", w.upper()):
                toks.append(vocab[ch])
                owner.append(wi)
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for ti, f0, f1 in viterbi(logp, toks, blank):
            wi = owner[ti]
            if wi >= 0:
                first[wi] = min(first.get(wi, f0), f0)
                last[wi] = max(last.get(wi, f1), f1)
        aligned = []
        for wi, w in enumerate(words):
            s = first.get(wi, 0) * FRAME_S
            e = (last.get(wi, first.get(wi, 0)) + 1) * FRAME_S
            aligned.append({"w": w, "start": round(s, 3), "end": round(e, 3)})
            ref = r["words"][wi]
            d_end.append((e - ref["end"]) * 1000.0)
            d_start.append((s - ref["start"]) * 1000.0)
            if wi == len(words) - 1:
                d_last.append((e - ref["end"]) * 1000.0)
        out.append({"id": r["id"], "words": aligned})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    dump(d, out, "align_ctc.jsonl")
    res = {"set": set_name, "rows": len(out), "model": MODEL, "against": rows[0].get("align"),
           "ctc_end_minus_reference_ms": pct(d_end), "ctc_last_word_end_minus_reference_ms": pct(d_last),
           "ctc_start_minus_reference_ms": pct(d_start), "seconds": round(time.perf_counter() - t0, 1)}
    (ROOT / "results" / "voice" / f"align_ctc_{set_name}{'_check' if check else ''}.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    if install:
        by_id = {o["id"]: o for o in out}
        full = load(d)
        for r in full:
            o = by_id.get(r["id"])
            if o is None:
                continue
            for w, a in zip(r["words"], o["words"]):
                w.setdefault("end_whisper", w["end"])
                w.setdefault("start_whisper", w["start"])
                w["start"], w["end"] = a["start"], a["end"]
            r["align"] = (f"CTC forced alignment, {MODEL} (20 ms frames), experiments/voice_ctc_align.py; "
                          "end_whisper = faster-whisper large-v3 word timestamps")
        dump(d, full)
        print(f"installed CTC timings into {d / 'align.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trim", "ctc"])
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="ctc: only measure against the set's timings")
    ap.add_argument("--install", action="store_true", help="ctc: write the CTC timings into the set's align.jsonl")
    a = ap.parse_args()
    if a.mode == "trim":
        trim(a.set)
    else:
        ctc(a.set, min(a.threads, 4), a.limit, a.check, a.install)


if __name__ == "__main__":
    main()
