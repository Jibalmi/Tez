"""Synthesise the Theioma voice command set (data/voice/theioma/commands.jsonl) with Kokoro-82M, with word timings.

Every utterance is spoken by one of six Kokoro voices (US and UK, female and male, assigned in turn by row) and is
labelled synthetic. Kokoro predicts a duration for every phoneme, so the start and end of every word in the audio are
known exactly (to Kokoro's 12.5 ms duration unit): that is the forced alignment the replay harness measures word ends
against (experiments/voice_replay.py). Each WAV gets 300 ms of silence before the speech and 700 ms after it, is
resampled to 16 kHz mono PCM16, and its word timings are shifted by the lead-in.

Run it from the TTS harness's environment (Kokoro's PyTorch package, experiments/tts_requirements.txt):
  ..\\Tez\\.venv-tts\\Scripts\\python experiments\\voice_tts.py [--device cpu] [--limit N]
Output: data/voice/theioma/wav/<id>.wav and data/voice/theioma/align.jsonl (one row per utterance: voice, duration,
SHA-256 of the WAV, words with start/end seconds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "voice" / "theioma"
VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george", "af_bella", "am_adam"]
SR_TTS, SR_OUT = 24000, 16000
LEAD_S, TRAIL_S = 0.3, 0.7


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (scipy when present, else an FFT resampler)."""
    try:
        from scipy.signal import resample_poly
        g = np.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * sr_out / sr_in))
        return np.fft.irfft(np.fft.rfft(x)[: n // 2 + 1], n).astype(np.float32) * (n / len(x))


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def word_times(tokens, ref_words: list[str]) -> list[dict]:
    """Kokoro's tokens (text, start_ts, end_ts) mapped onto the reference words in order. A reference word covered by
    several tokens takes the first start and the last end; punctuation tokens are skipped."""
    toks = [(norm(t.text), t.start_ts, t.end_ts) for t in tokens if norm(t.text)]
    out, i = [], 0
    for w in ref_words:
        target, got, start, end = norm(w), "", None, None
        while i < len(toks) and len(got) < len(target):
            text, s, e = toks[i]
            got += text
            start = s if start is None else start
            end = e if e is not None else end
            i += 1
        if got != target or start is None or end is None:
            raise ValueError(f"cannot align {ref_words} to Kokoro's tokens {toks}")
        out.append({"w": w, "start": float(start), "end": float(end)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import kokoro
    import torch
    from kokoro import KPipeline
    torch.set_num_threads(min(6, torch.get_num_threads()))
    rows = [json.loads(line) for line in (DATA / "commands.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    (DATA / "wav").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=args.device) for lang in "ab"}
    print(f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    out = []
    for n, r in enumerate(rows):
        voice = VOICES[n % len(VOICES)]
        words = r["text"].split()
        audio, tokens, offset = [], [], 0.0
        for res in pipes[voice[0]](r["text"], voice=voice):
            a = res.audio.detach().cpu().numpy() if hasattr(res.audio, "detach") else np.asarray(res.audio)
            for t in res.tokens:
                if t.start_ts is not None:
                    t.start_ts += offset
                if t.end_ts is not None:
                    t.end_ts += offset
            tokens += list(res.tokens)
            audio.append(a)
            offset += len(a) / SR_TTS
        x = np.concatenate(audio).astype(np.float32)
        timing = word_times(tokens, words)
        y = resample(x, SR_TTS, SR_OUT)
        y = np.concatenate([np.zeros(int(LEAD_S * SR_OUT), np.float32), y, np.zeros(int(TRAIL_S * SR_OUT), np.float32)])
        pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
        path = DATA / "wav" / f"{r['id']}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SR_OUT)
            f.writeframes(pcm.tobytes())
        for w in timing:
            w["start"] = round(w["start"] + LEAD_S, 4)
            w["end"] = round(w["end"] + LEAD_S, 4)
        out.append({"id": r["id"], "wav": f"wav/{r['id']}.wav", "voice": voice, "sr": SR_OUT,
                    "duration_s": round(len(y) / SR_OUT, 4), "speech_end_s": round(LEAD_S + len(x) / SR_TTS, 4),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "synthetic": True,
                    "tts": f"kokoro {getattr(kokoro, '__version__', '?')} hexgrad/Kokoro-82M v1.0 ({args.device})",
                    "align": "kokoro phoneme durations (12.5 ms units)", "words": timing})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    (DATA / "align.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")
    print(f"wrote {len(out)} WAVs and align.jsonl in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
