"""Word end times for Tez's 220 voice WAVs (data/voice/wav, SAPI speech), from faster-whisper word timestamps.

faster-whisper transcribes each WAV with word timestamps (cross-attention alignment); its words are matched to the
reference transcript (difflib on normalised words), matched reference words take the recognised word's start and end,
and the rest are interpolated between their matched neighbours. The result is the forced alignment the replay harness
measures latency against (experiments/voice_replay.py): data/voice/align.jsonl.

--check DIR scores the same procedure on a set whose true word times are known (the Kokoro-synthesised Theioma set,
data/voice/theioma, whose align.jsonl comes from Kokoro's phoneme durations): the error of faster-whisper's word ends.

  python experiments/voice_align.py [--model large-v3] [--device cpu]
  python experiments/voice_align.py --check data/voice/theioma --limit 60
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import statistics
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr, n, ch, width = f.getframerate(), f.getnframes(), f.getnchannels(), f.getsampwidth()
        raw = f.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: 16-bit PCM expected")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def align_words(ref: list[str], hyp: list[tuple[str, float, float]]) -> tuple[list[dict], int]:
    """Reference words with times: matched words take the recognised times, the others are interpolated."""
    a = [norm(w) for w in ref]
    b = [norm(w) for w, _, _ in hyp]
    times: list[tuple[float, float] | None] = [None] * len(ref)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for blk in sm.get_matching_blocks():
        for i in range(blk.size):
            _, s, e = hyp[blk.b + i]
            times[blk.a + i] = (s, e)
    matched = sum(t is not None for t in times)
    # interpolate the gaps (a merged or split word keeps a plausible end between its neighbours)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        s0, e0 = (hyp[0][1], hyp[-1][2]) if hyp else (0.0, 0.0)
        step = (e0 - s0) / max(1, len(ref))
        times = [(s0 + i * step, s0 + (i + 1) * step) for i in range(len(ref))]
    else:
        for i in range(len(ref)):
            if times[i] is not None:
                continue
            lo = max((j for j in known if j < i), default=None)
            hi = min((j for j in known if j > i), default=None)
            if lo is not None and hi is not None:
                span = times[hi][0] - times[lo][1]
                frac0, frac1 = (i - lo - 1) / (hi - lo - 1), (i - lo) / (hi - lo - 1)
                times[i] = (times[lo][1] + span * frac0, times[lo][1] + span * frac1)
            elif lo is not None:
                end = hyp[-1][2] if hyp else times[lo][1]
                n_after = len(ref) - lo - 1
                w = max(0.0, end - times[lo][1]) / max(1, n_after)
                times[i] = (times[lo][1] + (i - lo - 1) * w, times[lo][1] + (i - lo) * w)
            else:
                times[i] = (max(0.0, times[hi][0] - 0.3 * (hi - i)), max(0.0, times[hi][0] - 0.3 * (hi - i - 1)))
    return [{"w": w, "start": round(float(s), 4), "end": round(float(e), 4)} for w, (s, e) in zip(ref, times)], matched


def transcribe(model, x: np.ndarray) -> list[tuple[str, float, float]]:
    segs, _ = model.transcribe(x, language="en", beam_size=5, word_timestamps=True, vad_filter=False,
                               condition_on_previous_text=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--check", default=None, help="a data directory with a Kokoro align.jsonl to score against")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: CTranslate2's default)")
    args = ap.parse_args()
    if args.device == "cuda":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from voice_asr import add_nvidia_dlls
        add_nvidia_dlls()
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=args.threads)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.0f} s", flush=True)
    if args.check:
        d = ROOT / args.check
        rows = [json.loads(line) for line in (d / "align.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = rows[: args.limit] if args.limit else rows
        err_end, err_start, matched_all = [], [], 0
        for r in rows:
            x, _ = read_wav(d / r["wav"])
            got, matched = align_words([w["w"] for w in r["words"]], transcribe(model, x))
            matched_all += matched
            for g, w in zip(got, r["words"]):
                err_end.append((g["end"] - w["end"]) * 1000.0)
                err_start.append((g["start"] - w["start"]) * 1000.0)
        ae = np.abs(err_end)
        res = {"model": args.model, "rows": len(rows), "words": len(err_end), "matched": matched_all,
               "end_error_ms": {"mean": round(statistics.mean(err_end), 1), "median": round(statistics.median(err_end), 1),
                                "mae": round(float(ae.mean()), 1), "p90_abs": round(float(np.percentile(ae, 90)), 1)},
               "start_error_ms": {"mean": round(statistics.mean(err_start), 1),
                                  "mae": round(float(np.abs(err_start).mean()), 1)}}
        out = ROOT / "results" / "voice" / f"align_check_{args.model}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return
    cmds = [json.loads(line) for line in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cmds = cmds[: args.limit] if args.limit else cmds
    out = []
    for n, c in enumerate(cmds):
        path = ROOT / "data" / "voice" / "wav" / f"{c['id']}.wav"
        x, sr = read_wav(path)
        hyp = transcribe(model, x)
        words, matched = align_words(c["text"].split(), hyp)
        out.append({"id": c["id"], "wav": f"wav/{c['id']}.wav", "sr": sr, "duration_s": round(len(x) / sr, 4),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "synthetic": True, "tts": "Windows SAPI",
                    "align": f"faster-whisper {args.model} word timestamps ({args.device} {args.compute_type}), "
                             "matched to the reference, the rest interpolated",
                    "matched_words": matched, "asr": " ".join(w for w, _, _ in hyp), "words": words})
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(cmds)} {time.perf_counter() - t0:.0f}s", flush=True)
    (ROOT / "data" / "voice" / "align.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")
    m = sum(o["matched_words"] for o in out)
    n = sum(len(o["words"]) for o in out)
    print(f"wrote data/voice/align.jsonl: {len(out)} rows, {m}/{n} words matched, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
